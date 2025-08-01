from dgl.nn.pytorch import GraphConv
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
import torch as th

class GraphConvEdgeWeight(GraphConv):
    def forward(self, graph, feat, weight=None, edge_weights=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        'There are 0-in-degree nodes in the graph. '
                        'Add self-loop or set allow_zero_in_degree=True to suppress this.'
                    )

            feat_src, feat_dst = expand_as_pair(feat, graph)

            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5).view(-1, *[1] * (feat_src.dim() - 1))
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight provided but module already has weight. '
                                   'Set weight=False when constructing the module.')
            else:
                weight = self.weight

            graph.srcdata['h'] = feat_src

            if edge_weights is None:
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            else:
                graph.edata['a'] = edge_weights
                graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h'))

            rst = graph.dstdata['h']

            if self._in_feats <= self._out_feats and weight is not None:
                rst = th.matmul(rst, weight)
            elif self._in_feats > self._out_feats and weight is not None:
                feat_src = th.matmul(feat_src, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5) if self._norm == 'both' else 1.0 / degs
                norm = norm.view(-1, *[1] * (feat_dst.dim() - 1))
                rst = rst * norm


            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class Residual_Embedding(GraphConv):
    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        'There are 0-in-degree nodes in the graph. '
                        'Add self-loop or set allow_zero_in_degree=True to suppress this.'
                    )

            graph.srcdata['h'] = feat
            x = feat
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
            rst = graph.dstdata['h']
            residual_embedding = x - rst

            return residual_embedding
