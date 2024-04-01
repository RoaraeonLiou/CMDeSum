from torch import nn
from torch.nn.parameter import Parameter
import torch
import math
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, batch_size, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.weight = Parameter(torch.FloatTensor(batch_size, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        outputs = torch.zeros_like(input)
        for i in range(input.size(0)):
            support = torch.mm(input[i], self.weight[i])
            output = torch.mm(adj[i], support)
            if self.bias is not None:
                output + self.bias
            outputs[i] = output
        # print(outputs)
        return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNBlock(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, batch_size):
        super(GCNBlock, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, batch_size)
        self.gc2 = GraphConvolution(nhid, nout, batch_size)
        self.dropout = dropout


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x, self.dropout, training=self.training)
        outputs = F.relu(self.gc2(x1, adj)) + x
        return outputs


class ASTEmbedding(nn.Module):
    def __init__(self, nfeat, nhid, nout, d_model, batch_size, dropout):
        super(ASTEmbedding, self).__init__()
        self.gcn1 = GCNBlock(nfeat, nhid, nout, dropout, batch_size)
        self.gcn2 = GCNBlock(nfeat, nhid, nout, dropout, batch_size)
        self.gcn3 = GCNBlock(nfeat, nhid, nout, dropout, batch_size)

        self.ffn = nn.Linear(nout, d_model)

    def forward(self, x, adj, A2, A3, A4, A5):
        output1 = self.gcn1(x, adj)
        output2 = self.gcn2(output1, A2)
        output3 = self.gcn3(output2, A3)

        gcn_output = output3
        gcn_output = self.ffn(gcn_output)
        return gcn_output
