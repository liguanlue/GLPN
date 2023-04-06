import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from torch_geometric.utils import to_dense_adj

DEVICE = 'cuda:0'

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    A = torch.randn(input_dim, output_dim)
    initial = A.uniform_(-init_range, init_range)
    initial.to(DEVICE)
    return initial

def weight_variable_glorot_M(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    A = torch.randn(input_dim, output_dim)

    initial = A.uniform_(-init_range, init_range)
    initial = torch.tensor(initial, dtype=torch.torch.float64)
    initial.to(DEVICE)
    return initial

class GraphConvolution_M(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(GraphConvolution_M, self).__init__()
        self.weights1 = nn.Parameter(weight_variable_glorot_M(input_dim, input_dim))
        self.weights2 = nn.Parameter(weight_variable_glorot_M(input_dim, output_dim))
        self.dropout = dropout
        self.act = act

    def forward(self, inputs, adj):
        self.adj = adj
        x = inputs
        #x = tf.nn.dropout(x, 1-self.dropout)
        x = torch.einsum("bmn,bnf->bmf",[self.adj, x])
        x = torch.einsum("bmn,nf->bmf",[x, self.weights1])
        x = F.relu(x)
        x = torch.einsum("bmn,nf->bmf", [x, self.weights2])
        outputs = self.act(x)
        return outputs

class GraphConvolution_(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(GraphConvolution_, self).__init__()
        self.weights1 = nn.Parameter(weight_variable_glorot(input_dim, input_dim))
        self.weights2 = nn.Parameter(weight_variable_glorot(input_dim, output_dim))
        self.dropout = dropout
        self.act = act

    def forward(self, inputs, adj):
        self.adj = adj
        x = inputs
        x = torch.matmul(self.adj, x)
        x = torch.matmul(x, self.weights1)
        x = F.relu(x)
        x = torch.matmul(x, self.weights2)
        outputs = self.act(x)
        return outputs

class Residual_GDN(nn.Module):

    def __init__(self, input_dim, output_dim, n, dropout=0.): # adj: [N,N]
        super(Residual_GDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n = n
        self.gcn_inverse = GraphConvolution_(input_dim, output_dim)
        self.gcn_denoise = GraphConvolution_(output_dim, output_dim)

    def forward(self, inputs, edge_index):
        self.n = inputs.shape[1]
        if edge_index.shape[0] == 2:
            adj_ori = to_dense_adj(edge_index)
        else:
            adj_ori = edge_index


        adj_inverse = self.preprocess_inverse_adj(adj_ori, self.n)
        adj_inverseT = self.preprocess_inverse_adj(adj_ori.permute(1,0), self.n)
        adj = self.preprocess_adj(adj_ori, self.n)
        adjT = self.preprocess_adj(adj_ori.permute(1,0), self.n)
        x = self.gcn_inverse(inputs, adj_inverseT)
        x = self.gcn_denoise(x, adj)
        x = self.gcn_denoise(x, adjT)
        return x
    
    def normalize_adj(self, adj):
        epis = 1e-15
        rowsum = torch.sum(adj, 1) + epis
        d_inv_sqrt = torch.pow(rowsum, -1)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return torch.matmul(adj, d_mat_inv_sqrt)

    def preprocess_adj(self, adj, n):
        adj_normalized = self.normalize_adj(adj + torch.eye(n).to(DEVICE))
        sym_l = torch.eye(n).to(DEVICE) - adj_normalized
        sym_l_2 = torch.matmul(sym_l, sym_l)
        sym_l_3 = torch.matmul(sym_l, sym_l_2)
        sym_l_4 = torch.matmul(sym_l, sym_l_3)
        true_a = 0.1 * torch.eye(n).to(DEVICE) + adj_normalized + 0.5 * sym_l_2 - 1 / 6 * sym_l_3 + 1 / 24 * sym_l_4
        return true_a

    def preprocess_inverse_adj(self, adj, n):
        adj_normalized = self.normalize_adj(adj + torch.eye(n).to(DEVICE))
        sym_l = torch.eye(n).to(DEVICE) - adj_normalized
        sym_l_2 = torch.matmul(sym_l, sym_l)
        sym_l_3 = torch.matmul(sym_l, sym_l_2)
        sym_l_4 = torch.matmul(sym_l, sym_l_3)
        true_a = 1.1 * torch.eye(n).to(DEVICE) + adj_normalized + 0.5 * sym_l_2 + 1 / 6 * sym_l_3 + 1 / 24 * sym_l_4
        return true_a

class GCN(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, n, dropout=0.):  # adj: [N,N]
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n = n
        self.gcn = DenseGCNConv(input_dim, output_dim)

    def forward(self, inputs, edge_index):
        x = self.gcn(inputs, edge_index)
        return x


class Residual_GDN_Multi(nn.Module):

    def __init__(self, input_dim, output_dim, n, dropout=0.):  # adj: [N,N]
        super(Residual_GDN_Multi, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n = n
        self.gcn_inverse = GraphConvolution_M(input_dim, output_dim)
        self.gcn_denoise = GraphConvolution_M(output_dim, output_dim)

    def forward(self, inputs, edge_index):
        self.n = inputs.shape[1]
        adj_ori = edge_index

        adj_inverse = self.preprocess_inverse_adj(adj_ori, self.n)
        adj_inverseT = self.preprocess_inverse_adj(adj_ori.permute(0, 2, 1), self.n)
        adj = self.preprocess_adj(adj_ori, self.n)
        adjT = self.preprocess_adj(adj_ori.permute(0, 2, 1), self.n)
        x = self.gcn_inverse(inputs, adj_inverseT)
        x = self.gcn_denoise(x, adj)
        x = self.gcn_denoise(x, adjT)
        return x

    def normalize_adj(self, adj):
        epis = 1e-15
        rowsum = torch.sum(adj, 1) + epis
        d_inv_sqrt = torch.pow(rowsum, -1)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        return torch.einsum("bmn,bnf->bmf", [adj, d_mat_inv_sqrt])

    def preprocess_adj(self, adj, n):
        one = torch.eye(n)
        one = one.reshape((1, n, n))
        one = one.repeat(adj.shape[0], 1, 1)

        adj_normalized = self.normalize_adj(adj + one.to(DEVICE))
        sym_l = one.to(DEVICE) - adj_normalized
        sym_l_2 = torch.einsum("bmn,bnf->bmf", [sym_l, sym_l])
        sym_l_3 = torch.einsum("bmn,bnf->bmf", [sym_l, sym_l_2])
        sym_l_4 = torch.einsum("bmn,bnf->bmf", [sym_l, sym_l_3])
        true_a = 0.1 * one.to(DEVICE) + adj_normalized + 0.5 * sym_l_2 - 1 / 6 * sym_l_3 + 1 / 24 * sym_l_4
        return true_a

    def preprocess_inverse_adj(self, adj, n):
        one = torch.eye(n)
        one = one.reshape((1, n, n))
        one = one.repeat(adj.shape[0], 1, 1)

        adj_normalized = self.normalize_adj(adj + one.to(DEVICE))
        sym_l = one.to(DEVICE) - adj_normalized
        sym_l_2 = torch.einsum("bmn,bnf->bmf", [sym_l, sym_l])
        sym_l_3 = torch.einsum("bmn,bnf->bmf", [sym_l, sym_l_2])
        sym_l_4 = torch.einsum("bmn,bnf->bmf", [sym_l, sym_l_3])

        true_a = 1.1 * one.to(DEVICE) + adj_normalized + 0.5 * sym_l_2 + 1 / 6 * sym_l_3 + 1 / 24 * sym_l_4
        return true_a