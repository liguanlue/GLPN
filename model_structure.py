from __future__ import division
import math
import torch
from torch import nn
import torch.nn.functional as F
from layers import Residual_GDN, GCN, Residual_GDN_Multi


class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """       
    def __init__(self, in_channels, out_channels, orders, activation = 'relu'): 

        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                             out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)
        
    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)
        
    def forward(self, X, A_q, A_h):

        batch_size = X.shape[0]
        num_node = X.shape[1]
        input_size = X.size(2)
        supports = []
        supports.append(A_q)
        supports.append(A_h)
        
        x0 = X.permute(1, 2, 0) #(num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1
                
        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])         
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)     
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)   
            
        return x

class D_GCN_MultiGraph(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """
    def __init__(self, in_channels, out_channels, orders, activation = 'relu'):

        super(D_GCN_MultiGraph, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.DoubleTensor(in_channels * self.num_matrices,out_channels))
        self.bias = nn.Parameter(torch.DoubleTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):

        batch_size = X.shape[0]
        num_node = X.shape[1]
        input_size = X.size(2)
        supports = []
        supports.append(A_q)
        supports.append(A_h)

        x0 = X
        x = torch.unsqueeze(x0,0)
        for support in supports:
            x1 = torch.einsum("bmn,bnf->bmf",[support, x0])
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.einsum("bmn,bnf->bmf",[support, x1]) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = x.permute(1, 2, 3, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        return x

class GLPN(nn.Module):

    def __init__(self, h, z, k, r, set='None'):
        super(GLPN, self).__init__()
        self.time_dimension = h
        self.hidden_dimnesion = z
        self.order = k
        self.d_a = 32
        self.r = r[0]
        if set == 'MultiGraph':
            self.GNN1 = D_GCN_MultiGraph(self.time_dimension, self.hidden_dimnesion, self.order)
            self.GNN2 = Residual_GDN_Multi(self.hidden_dimnesion, self.hidden_dimnesion, 99)
            self.GNN3 = D_GCN_MultiGraph(self.hidden_dimnesion * 2, self.time_dimension, self.order,
                                         activation='linear')
        else:
            self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimnesion, self.order)
            self.GNN2 = Residual_GDN(self.hidden_dimnesion, self.hidden_dimnesion, 99)
            self.GNN3 = D_GCN(self.hidden_dimnesion * 2, self.time_dimension, self.order, activation='linear')

        self.assign = nn.Parameter(torch.zeros(size=(self.hidden_dimnesion, 30)))
        nn.init.xavier_uniform_(self.assign.data, gain=1.414)

        if set == 'MultiGraph':
            self.W_s1f = nn.Parameter(torch.DoubleTensor(self.hidden_dimnesion, self.d_a))
            self.W_s2f = nn.Parameter(torch.DoubleTensor(self.d_a, self.r))
        else:
            self.W_s1f = nn.Parameter(torch.Tensor(self.hidden_dimnesion, self.d_a))
            self.W_s2f = nn.Parameter(torch.Tensor(self.d_a, self.r))
        nn.init.xavier_uniform_(self.W_s1f)
        nn.init.xavier_uniform_(self.W_s2f)

    def forward(self, X, A_q, A_h, Adj):

        X_S = X.permute(0, 2, 1)  # to correct the input dims

        X_s1 = self.GNN1(X_S, A_q, A_h) # [4, 150, 100]
        A = F.softmax(torch.matmul(torch.tanh(torch.matmul(X_s1, self.W_s1f)), self.W_s2f), dim=-1) # [4, 150, 60]
        x_pooled = torch.einsum("bmn,bnf->bmf",[torch.transpose(A, 2, 1), X_s1])  # KN . Nv -> Kv # [4, 24, 150]
        H_pup = torch.einsum("bnm,bmf->bnf", [A, x_pooled])  # NK . KV -> NV
        H_pup = H_pup / X_S.shape[1]
        X_s2 = self.GNN2(X_s1, Adj) + X_s1  # num_nodes, rank, treat as residual
        X_s3 = self.GNN3(torch.cat([X_s2, H_pup], -1), A_q, A_h)

        X_res = X_s3.permute(0, 2, 1)

        return X_res

class GCN_b(nn.Module):

    def __init__(self, h, z, k):
        super(GCN_b, self).__init__()
        self.time_dimension = h
        self.hidden_dimnesion = z
        self.order = k

        self.GNN1 = GCN(self.time_dimension, self.hidden_dimnesion, self.order)
        self.GNN2 = GCN(self.hidden_dimnesion, self.time_dimension, 99)


    def forward(self, X, A_q, A_h, Adj):

        X_S = X.permute(0, 2, 1)  # to correct the input dims

        X_s1 = self.GNN1(X_S, Adj) # [4, 150, 100]
        X_s1 = F.relu(X_s1)
        X_s2 = self.GNN2(X_s1, Adj)

        X_res = X_s2.permute(0, 2, 1)

        return X_res

