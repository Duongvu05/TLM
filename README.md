# Ethereum Fraud Detection--TLmGNN
This is an implementation of the paper - "Ethereum Fraud Detection via Joint Transaction Language Model and Graph Representation Learning"
## Overview
We propose a hybrid approach combining a transaction language model with graph-based methods to improve fraudulent Ethereum account detection.  We first propose a transaction language model that converts historical transaction sequences into sentences, where transaction attributes are represented as words, to learn explicit transaction semantics. We then propose a transaction attribute similarity graph to model global semantic relationships and apply graph convolution to learn attribute embeddings that encode similarity information. A deep multi-head attention network is then employed to fuse transaction semantic and similarity embeddings. Additionally, we construct an account interaction graph to model transaction behaviors among accounts. Finally, we propose a joint training method for the multi-head attention network and the account interaction graph to leverage the benefits of both.
## Model Designs
![image](https://github.com/lincozz/TLmGNN/blob/main/framework.png)


## Requirements

```
Python 3.8
Pytorch 2.0.0
```

## Dataset

We evaluated the performance of the model using two publicly available and newly released datasets. The composition of the dataset is as follows, you can click on the **"Source"** to download them.

| *Dataset*        | *Nodes*      | *Edges*       | *Avg Degree*   |*Phisher* | *Source*  |
| ---------------- | ------------- | -------------- | -------------- |------- |---------- |
| MulDigraph       |  2,973,489    |  13,551,303    |  4.5574        | 1,165  |  XBlock     |
| B4E              |  597,258      |  11,678,901    |  19.5542       | 3,220  |    Github,Chrome   |
| Our dataset SPN  |  496,740      |  1831,082      |  1.6730        | 5,619  |    Github       |


## Main Results

Here only the F1-Score(Percentage) results are displayed; for other metrics, please refer to the paper.

| *Model*              | *MulDiGraph* | *B4E*     | *SPN*     |
| -------------------- | ------------ | --------- | --------- |
| *Role2Vec*           | 56.08        | 66.73     | 55.12     | 
| *Trans2Vec*          | 70.29        | 38.42     | 51.34     | 
| *GCN*                | 42.47        | 63.59     | 50.09     | 
| *GAT*                | 40.14        | 60.38     | 61.30     |
| *SAGE*               | 34.30        | 51.34     | 51.10     | 
| *BERT4ETH*           | 55.57        | 67.11     | 71.14     | 
| ***Our***            | **90.41**    | **81.23** | **81.46** | 
