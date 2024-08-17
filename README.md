# Ethereum Fraud Detection--TLmGNN
This is an implementation of the paper - "Ethereum Fraud Detection via Joint Transaction Language Model and Graph Representation Learning"
## Overview
we propose a hybrid approach combining a transaction language model with graph-based methods to improve fraudulent Ethereum account detection.  We first propose a transaction language model that converts historical transaction sequences into sentences, where transaction attributes are represented as words, to learn explicit transaction semantics. We then propose a transaction attribute similarity graph to model global semantic relationships and apply graph convolution to learn attribute embeddings that encode similarity information. A deep multi-head attention network is then employed to fuse transaction semantic and similarity embeddings. Additionally, we construct an account interaction graph to model transaction behaviors among accounts. Finally, we propose a joint training method for the multi-head attention network and the account interaction graph to leverage the benefits of both.
## Model Designs
![image](https://github.com/lincozz/TLmGNN/blob/main/framework.png)

## Dataset

We evaluated the performance of our model using four publicly available datasets. The composition of the datasets is as follows, and you can click on the dataset names to download them. Please note that you need to modify the code in the `CPG_generator` function in `run.py` to adapt to different dataset formats.

| *Dataset*        | *Nodes*      | *Edges*       | *Avg Degree*   |*Phisher* | *Source*  |
| ---------------- | ------------- | -------------- | -------------- |------- |---------- |
| MulDigraph       |  2,973,489    |  13,551,303    |  4.5574        | 1,165  |  XBlock     |
| B4E              |  597,258      |  11,678,901    |  19.5542       | 3,220  |    Github,Chrome       |
| Our dataset SPN  |  496,740      |  1831,082      |  1.6730        | 5,619  |    Github       |
