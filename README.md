# Ethereum Fraud Detection--TLmGNN
This is an implementation of the paper - "Ethereum Fraud Detection via Joint Transaction Language Model and Graph Representation Learning"
## Overview
we propose a hybrid approach combining a transaction language model with graph-based methods to improve fraudulent Ethereum account detection.  We first propose a transaction language model that converts historical transaction sequences into sentences, where transaction attributes are represented as words, to learn explicit transaction semantics. We then propose a transaction attribute similarity graph to model global semantic relationships and apply graph convolution to learn attribute embeddings that encode similarity information. A deep multi-head attention network is then employed to fuse transaction semantic and similarity embeddings. Additionally, we construct an account interaction graph to model transaction behaviors among accounts. Finally, we propose a joint training method for the multi-head attention network and the account interaction graph to leverage the benefits of both.
## Model Designs
![image](https://github.com/lincozz/TLmGNN/blob/main/framework.png)
