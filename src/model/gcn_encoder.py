from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class GraphConvolution(Module):
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
        return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_Two(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, batch_size):
        super(GCN_Two, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, batch_size)
        self.gc2 = GraphConvolution(nhid, nout, batch_size)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x, self.dropout, training=self.training)
        outputs = F.relu(self.gc2(x1, adj)) + x
        return outputs


class AST_Model(nn.Module):
    def __init__(self, nfeat, nhid, nout, d_model, batch_size, dropout):
        super(AST_Model, self).__init__()
        self.gcn1 = GCN_Two(nfeat, nhid, nout, dropout, batch_size)
        self.gcn2 = GCN_Two(nfeat, nhid, nout, dropout, batch_size)
        self.gcn3 = GCN_Two(nfeat, nhid, nout, dropout, batch_size)

        self.ffn = nn.Linear(nout, d_model)

    def forward(self, x, adj, A2, A3, A4, A5):
        output1 = self.gcn1(x, adj)
        output2 = self.gcn2(output1, A2)
        output3 = self.gcn3(output2, A3)

        gcn_output = output3
        gcn_output = self.ffn(gcn_output)

        return gcn_output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, device):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_ff = d_ff
        self.d_model = d_model
        self.device = device

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, device)

    def forward(self, ast_outputs):
        enc_outputs, attn = self.enc_self_attn(ast_outputs, ast_outputs, ast_outputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class GCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid, nout, d_model, batch_size, dropout, d_k, d_v, d_ff, n_heads, n_layers, device):
        super(GCNEncoder, self).__init__()
        self.ast_output = AST_Model(nfeat, nhid, nout, d_model, batch_size, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, d_ff, n_heads, device) for _ in range(n_layers)])

    def forward(self, x, a, a2, a3, a4, a5):
        ast_embed = self.ast_output(x, a, a2, a3, a4, a5)
        ast_self_attns = []
        for layer in self.layers:
            ast_outputs, enc_self_attn = layer(ast_embed)
            ast_self_attns.append(enc_self_attn)
        return ast_outputs, ast_embed
