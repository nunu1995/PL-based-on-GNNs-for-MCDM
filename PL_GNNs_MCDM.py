import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from sklearn.neighbors import NearestNeighbors
import time



def c_index(true_preferences, predicted_scores):
    n = len(true_preferences)
    num_concordant = 0
    num_discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (true_preferences[i] < true_preferences[j] and predicted_scores[i] < predicted_scores[j]) or \
               (true_preferences[i] > true_preferences[j] and predicted_scores[i] > predicted_scores[j]):
                num_concordant += 1
            elif (true_preferences[i] < true_preferences[j] and predicted_scores[i] > predicted_scores[j]) or \
                 (true_preferences[i] > true_preferences[j] and predicted_scores[i] < predicted_scores[j]):
                num_discordant += 1
    cindex = num_concordant / (num_concordant + num_discordant)
    return cindex


def extract_features(split):
    features = []
    for i in range(0, 8):
        features.append(float(split[i]))
    return features


def train_get_format_data(path):
    with open(path, 'r') as file:
        for line in file:
            split = line.split()
            y_train.append(int(split[9]))
            x_train.append(extract_features(split[1:]))
    return x_train, y_train


def test_get_format_data(path):
    with open(path, 'r') as file:
        for line in file:
            split = line.split()
            y_test.append(int(split[9]))
            x_test.append(extract_features(split[1:]))
    return x_test, y_test


def build_graph(path):
    data_x, data_y = train_get_format_data(path)
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data_x)
    distances, indices = nbrs.kneighbors(data_x)
    num_nodes = len(data_x)
    edges = [(i, indices[i, j + 1]) for i in range(num_nodes) for j in range(k)]
    src, dst = zip(*edges)
    G = dgl.graph((src, dst), num_nodes=num_nodes)
    G = dgl.add_self_loop(G)
    return data_x, data_y, G


class CombinedModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, category_output_dim, feature_output_dim):
        super(CombinedModel, self).__init__()
        self.conv1 = dgl.nn.GraphConv(input_dim, hidden_dim)
        self.conv2 = dgl.nn.GraphConv(hidden_dim, hidden_dim)
        self.fc_category = nn.Linear(hidden_dim, category_output_dim)
        self.fc_features = nn.Linear(hidden_dim, feature_output_dim)
        self.attention_weights = nn.Parameter(torch.ones(2))

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = torch.relu(x)
        x = self.conv2(g, x)
        x = torch.relu(x)
        category_output = self.fc_category(x)
        features_output = self.fc_features(x)
        return category_output, features_output


class WeightNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WeightNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.fc4 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=1)
        return x

    def l2_regularization(self):
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg


def GraphTrain(data, labels, graph):

    learning_rate = 0.005
    num_epochs = 100
    data = torch.tensor(data)
    labels = torch.tensor(labels, dtype=torch.long)
    labels = labels - 1
    graph.ndata['x'] = torch.cat([data, torch.zeros(data.shape[0], feature_y)], dim=1)
    graph.ndata['x'][labels == 0, -feature_y:] = torch.tensor([1.0, 0.0])
    graph.ndata['x'][labels == 1, -feature_y:] = torch.tensor([0.0, 1.0])
    graph.ndata['mask'] = torch.as_tensor(labels.bernoulli(0.7), dtype=torch.bool)
    train_mask = graph.ndata['mask']

    combined_model.train()
    criterion_category = nn.CrossEntropyLoss()
    criterion_features = nn.MSELoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        category_output, features_output = combined_model(graph, graph.ndata['x'])
        attention_weights = torch.softmax(combined_model.attention_weights, dim=0)
        loss_category = criterion_category(category_output[train_mask].squeeze(), labels[train_mask])
        loss_features = criterion_features(features_output[train_mask].squeeze(), data[train_mask])
        total_loss = attention_weights[0] * loss_category + attention_weights[1] * loss_features
        total_loss.backward()
        optimizer.step()
    torch.save(combined_model.state_dict(), 'xxx.ckpt')


def GraphTest(data, labels, graph):

    data = torch.tensor(data)
    labels = torch.tensor(labels, dtype=torch.long)
    labels = labels - 1
    graph.ndata['x'] = torch.cat([data, torch.zeros(data.shape[0], feature_y)], dim=1)
    graph.ndata['x'][labels == 0, -feature_y:] = torch.tensor([1.0, 0.0])
    graph.ndata['x'][labels == 1, -feature_y:] = torch.tensor([0.0, 1.0])
    graph.ndata['mask'] = torch.as_tensor(labels.bernoulli(0.3), dtype=torch.bool)
    test_mask = graph.ndata['mask']

    with torch.no_grad():
        combined_model.eval()
        combined_model.load_state_dict(torch.load('xxx.ckpt'))
        category_output, features_output = combined_model(graph, graph.ndata['x'])
        predicted = torch.argmax(category_output[test_mask].squeeze(), dim=1)
        features = features_output[test_mask].detach().numpy()
        max_values = np.max(features, axis=0)
        min_values = np.min(features, axis=0)
        cin = c_index(predicted, labels[test_mask].view(-1))
        score = Score(data[test_mask].numpy().squeeze(), max_values, min_values, w)
        scin = c_index(score, labels[test_mask].view(-1))
        return max_values, min_values, cin, scin


def WeightTrain(data, labels):

    input = feature_x
    hidden = input * 2
    output = feature_y
    learning_rate = 0.005
    num_epochs = 100
    l2_lambda = 0.01
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    labels = labels - 1
    weight_model = WeightNet(input, hidden, output)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(weight_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = weight_model(data)
        loss = criterion(outputs, labels) + l2_lambda * weight_model.l2_regularization()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    weights_layer1 = weight_model.fc1.weight.data
    input_weights_sum = torch.sum(torch.abs(weights_layer1), dim=0)
    normalized_weights = input_weights_sum / torch.sum(input_weights_sum)
    normalized_weights = normalized_weights / torch.sum(normalized_weights)
    return normalized_weights.tolist()


def Score(data, max_value, min_value, weight):
    tmpmaxdist = np.power(np.sum(np.transpose(weight) * np.power((max_value-data), 2), axis=1), 0.5)
    tmpmindist = np.power(np.sum(np.transpose(weight) * np.power((min_value-data), 2), axis=1), 0.5)
    score_data = tmpmindist / (tmpmindist + tmpmaxdist)
    score_data = score_data / np.sum(score_data)
    return score_data



if __name__ == '__main__':

    num_runs = 100
    num_hidden = 68
    max_list = [0 for x in range(0, num_runs)]
    min_list = [0 for x in range(0, num_runs)]
    cin_list = [0 for x in range(0, num_runs)]
    scin_list = [0 for x in range(0, num_runs)]

    start_time = time.time()

    for i in range(num_runs):
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        feature_x = 8
        feature_y = 2
        input_dim = feature_x + feature_y
        hidden_dim = num_hidden
        category_output_dim = feature_y
        feature_output_dim = feature_x
        combined_model = CombinedModel(input_dim, hidden_dim, category_output_dim, feature_output_dim)

        data_path = 'xxx'
        x, y, g = build_graph(data_path)
        GraphTrain(x, y, g)
        w = WeightTrain(x, y)
        max_list[i], min_list[i], cin_list[i], scin_list[i] = GraphTest(x, y, g)

    av_cin = np.mean(cin_list)
    av_scin = np.mean(scin_list)
    std_cin = np.std(cin_list)
    std_scin = np.std(scin_list)

    print("Res:", " C-index {:.4f} | SC-index {:.4f}}".format(av_cin, av_scin))
    print("Dev:", " C-index {:.4f} | SC-index {:.4f}".format(std_cin, std_scin))

    end_time = time.time()
    print(f"GraphTest execution time: {end_time - start_time} seconds")