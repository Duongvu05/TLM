import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN
from .torch_gat import GAT
from .torch_sage import SAGE

class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model,abnormal_prompt , nb_class=2, m=0.7, gcn_layers=2, n_hidden=64, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.abnormal_prompt = abnormal_prompt
        self.nb_class = nb_class
        self.bert_model = pretrained_model
        self.feat_dim = 64
        # self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers - 1,
            activation=F.elu,
            dropout=dropout
        )  # 初始化GCN模型

        self.fc = th.nn.Linear(768,64)

    def forward(self, g, idx, gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask):
        if self.training:
            cls_logit, cls_feats = self.bert_model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_logit, cls_feats = self.bert_model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)

            cls_feats = g.ndata['cls_feats'][idx]
        # cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        cls_pred = cls_logit
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_logit = th.matmul(gcn_logit,self.abnormal_prompt)
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        gcn_pred = gcn_logit
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        # pred = th.log(pred)
        return pred


class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model, nb_class=2, m=0.5, gcn_layers=2, heads=8, n_hidden=32,
                 dropout=0.2):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.bert_model = pretrained_model
        self.feat_dim = 64
        # self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
            num_layers=gcn_layers - 1,
            in_dim=self.feat_dim,
            num_hidden=n_hidden,
            num_classes=nb_class,
            heads=[heads] * (gcn_layers - 1) + [1],
            activation=F.elu,
            feat_drop=dropout,
            attn_drop=dropout,
        )

    def forward(self, g, idx, gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask):
        if self.training:
            cls_logit, cls_feats = self.bert_model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_logit, cls_feats = self.bert_model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)
            cls_feats = g.ndata['cls_feats'][idx]
        # cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        cls_pred = cls_logit
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        # gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        gcn_pred = gcn_logit
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        # pred = th.log(pred)
        return pred

class BertSAGE(th.nn.Module):
    def __init__(self, pretrained_model, nb_class=2, m=0.7, n_hidden=32, dropout=0.2):
        super(BertSAGE, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.bert_model = pretrained_model
        self.feat_dim = 64
        # self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = SAGE(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=2,
            agg_type="mean",
            activation=F.elu,
            dropout=0.2)

    def forward(self, g, idx, gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask):
        if self.training:
            cls_logit, cls_feats = self.bert_model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_logit, cls_feats = self.bert_model(
                gcn_adj_list, gcn_swop_eye, input_ids, segment_ids, input_mask)
        # cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        cls_pred = cls_logit
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        # gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        gcn_pred = gcn_logit
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        # pred = th.log(pred)
        return pred


