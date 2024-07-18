# Preference learning based on adaptive graph neural networks for multi-criteria decision support

 This is the PyTorch implementation by <a href='https://github.com/nunu1995'>@Zhenhu Meng</a> for preference learning model:

 >**Preference learning based on adaptive graph neural networks for multi-criteria decision support**  
 >Zhenhua Meng, Rongheng Lin*, Budan Wu


\* denotes corresponding author
<p align="center">
<img src="PL_GNNs_MCDM.png" alt="PLGNNsMCDM" width=70%>
</p>

Given the gap between **multi-criteria decision making (MCDM)** and **preference learning (PL)** and the growing complexity of decision scenarios, this paper introduces the definition of graphs to model decision issues and proposes a preference learning method based on **graph neural networks (GNNs)** for multi-criteria decision support. The proposed method, grounded in the TOPSIS framework, incorporates an adaptive GNN model and a weight determination model. The core idea of the GNN model is to update embeddings from the alternative (node)'s criterion features and category features, and use the attention mechanism to adaptively learn their respective importance. The weight determination model contains a weight neural network module for determining objective criteria weights and a game theory-based combination weighting module for computing criteria combination weights.

### Dependancy
- torch 1.12.1
- numpy 1.23.4
- networks 2.7.1
- dgl 0.8.0.post1

### Dataset
The original data of our dataset can be found from following links (thanks to their work):
- DBS, CPU, BCC, MPG, MMG, CEV: http://archive.ics.uci.edu/ml/.
- ESL, ERA, LEV: http://www.cs.waikato.ac.nz/ml/weka/datasets.html.

### Contact
**Thanks for your interest in our work! For mor information, contact via: zhmeng@bupt.edu.cn**
