import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.STGCN.utils import Align, SpecialSpmm


class SpatioConvLayerGCN(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, node_emb_len, device):
        super(SpatioConvLayerGCN, self).__init__()
        self.device = device
        self.num_nodes = num_nodes

        self.adaptive_node_emb = node_emb_len > 0
        self.supports_len = 1
        if self.adaptive_node_emb:
            self.E1 = nn.Parameter(torch.randn(size=(num_nodes, node_emb_len)))
            self.E2 = nn.Parameter(torch.randn(size=(node_emb_len, num_nodes)))
            self.supports_len += 1

        self.thetas, self.bs = [], []
        for i in range(self.supports_len):
            theta = nn.Parameter(torch.FloatTensor(c_in, c_out).to(device))  # kernel: C_in*C_out*ks
            b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1).to(device))
            self.thetas.append(theta)
            self.bs.append(b)
            if i == 0:
                self.register_parameter(f"theta", theta)
                self.register_parameter(f"b", b)
            else:
                self.register_parameter(f"theta{i}", theta)
                self.register_parameter(f"b{i}", b)

        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.supports_len):
            init.kaiming_uniform_(self.thetas[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.thetas[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bs[i], -bound, bound)

    def gconv(self, x, adj, i):
        # adj: N N
        adj = adj + torch.eye(self.num_nodes).to(self.device)
        out_deg = torch.sum(adj, dim=1)
        sqr_out_deg = torch.diag(torch.reciprocal(torch.sqrt(out_deg)))
        norm_adj = torch.mm(torch.mm(sqr_out_deg, adj), sqr_out_deg)

        # N C_in V L
        x_c = torch.einsum("uv,ncvl->ncul", norm_adj, x)  # delete num_nodes(v)
        # N C_out V L
        x_gc = torch.einsum("io,nivl->novl", self.thetas[i], x_c) + self.bs[i]  # c_in(i) -> c_out(o)
        return x_gc

    def forward(self, x, adj):
        supports = [adj]
        if self.adaptive_node_emb:
            supports.append(F.softmax(F.relu(torch.mm(self.E1, self.E2)), dim=1))

        # x: N C_in V L
        # N C_out V L
        result = self.align(x)  # residual

        for i in range(self.supports_len):
            result += self.gconv(x, supports[i], i)  # residual connection

        return torch.relu(result)


class SpatioConvLayer(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, num_heads, adj, device):
        super(SpatioConvLayer, self).__init__()
        assert c_in == c_out

        self.alpha = 0.2
        self.dropout = 0.2
        self.num_heads = num_heads
        self.device = device

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.num_nodes = num_nodes
        self.c_in = c_in

        self.W = nn.Parameter(torch.zeros(size=(num_heads, c_in, c_out)))
        self.a = nn.Parameter(torch.zeros(size=(num_heads, 2 * c_in, 1)))

        self.adj = adj
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, adj):
        # x: N c_in V L(timestamp)
        L = x.shape[3]
        N = self.num_nodes

        # for x: N * n_dim
        # e_ij = (x_i * W) | (x_j * W)
        # h: x * W
        h = torch.einsum("ncvl, hcd -> hndvl", x, self.W)

        # e: H N 2C V*V L
        e = torch.cat([
            h.repeat(1, 1, 1, N, 1),
            h.permute(0, 1, 4, 3, 2).repeat(1, 1, 1, 1, N)
            .view(self.num_heads, -1, L, N * N, self.c_in).permute(0, 1, 4, 3, 2)
        ], dim=2)

        # e: H N V*V L 2C
        e = e.permute(0, 1, 3, 4, 2)

        # flatten_e: H N*(V*V)*L 2C
        # self.a:    H 2C    1
        flatten_e = e.reshape(self.num_heads, -1, 2 * self.c_in)
        # flatten_e_out: H N*(V*V)*L 1
        flatten_e_out = self.leakyrelu(torch.bmm(flatten_e, self.a))
        # e_out: H N V*V L
        e_out = flatten_e_out.view(self.num_heads, -1, N * N, L)
        # e_out: H N L V*V
        e_out = e_out.permute(0, 1, 3, 2)

        # ee, attention: H N L V V
        ee = e_out.view(e_out.shape[0], e_out.shape[1], e_out.shape[2], N, N)
        # inverse_softmax_adj: -inf when 0
        # mul_res: -inf or +inf when 0
        # attention: -9e15 when 0
        # method 1: log(adj)*att
        adj_mask = self.adj.repeat(ee.shape[0], ee.shape[1], ee.shape[2], 1, 1)
        inverse_softmax_adj = torch.log(adj + 1e-8).repeat(ee.shape[0], ee.shape[1], ee.shape[2], 1, 1)
        mul_res = ee * inverse_softmax_adj
        zeros = torch.ones_like(adj_mask).to(self.device) * -9e15
        attention = torch.where(adj_mask == 0, zeros, mul_res)

        # method 2: relu(adj-eps) as mask
        # sub_adj_mask = torch.relu(adj - 0.1).repeat(ee.shape[0], ee.shape[1], ee.shape[2], 1, 1)
        #
        # inverse_softmax_adj = adj.repeat(ee.shape[0], ee.shape[1], ee.shape[2], 1, 1)
        # mul_res = ee * inverse_softmax_adj
        # zeros = torch.ones_like(sub_adj_mask) * -9e15
        #
        # attention = torch.where(sub_adj_mask == 0, zeros, mul_res)

        # method 3: just mul
        # attention = ee * adj.repeat(ee.shape[0], ee.shape[1], ee.shape[2], 1, 1)

        attention = F.softmax(attention, dim=4)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # mean_attention: N L V V
        mean_attention = torch.mean(attention, dim=0)

        # N L V V  * N C V L -> N C V L

        # NL V V, NL V C
        att_ = mean_attention.reshape(-1, N, N)
        x_ = x.permute(0, 3, 2, 1).reshape(-1, N, self.c_in)

        # N C V L
        h_prime = torch.bmm(att_, x_).reshape(-1, L, N, self.c_in).permute(0, 3, 2, 1)
        return F.elu(h_prime)


class SparseSpatioConvLayer(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, num_heads, adj, batch_size, input_len, device):
        super(SparseSpatioConvLayer, self).__init__()
        assert c_in == c_out

        self.alpha = 0.2
        self.dropout = nn.Dropout(0.2)
        self.num_heads = num_heads

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.num_nodes = num_nodes
        self.c_in = c_in
        self.c_out = c_out

        self.W = nn.Parameter(torch.zeros(size=(num_heads, c_in, c_out)))
        self.a = nn.Parameter(torch.zeros(size=(num_heads, 2 * c_in, 1)))

        self.spmm = SpecialSpmm()
        self.device = device
        self.adj = adj

        self.edge = self.adj.nonzero().t()  # 2 E, indice
        self.edge_indices_num = num_heads * batch_size * input_len
        self.B = batch_size
        self.edge_indices = []
        for i in range(self.edge_indices_num):
            self.edge_indices.append(torch.cat([torch.ones_like(self.edge[0]).unsqueeze(0) * i, self.edge]))
        self.edge_indices = torch.cat(self.edge_indices, dim=1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, adj):
        # x: N c_in V L(timestamp)
        L = x.shape[3]
        B = x.shape[0]
        assert B == self.B, B
        V = self.num_nodes
        edge_val = adj[self.edge[0, :], self.edge[1, :]]  # E
        E = self.edge.shape[1]

        # for x: N * n_dim
        # e_ij = (x_i * W) | (x_j * W)
        # h: x * W, H N C V L
        h = torch.einsum("ncvl, hcd -> hndvl", x, self.W)

        # H N 2C E L
        edge_h = torch.cat((h[:, :, :, self.edge[0, :], :], h[:, :, :, self.edge[1, :], :]), dim=2)
        # H NEL 2C
        flatten_edge_h = edge_h.permute(0, 1, 3, 4, 2).reshape(self.num_heads, -1, 2 * self.c_in)
        # H NEL 1
        edge_e = torch.exp(-self.leakyrelu(torch.bmm(flatten_edge_h, self.a)))
        # HNL E
        edge_e = edge_e.reshape(self.num_heads, -1, E, L).permute(0, 1, 3, 2).reshape(-1, E)

        inverse_softmax_adj = edge_val.repeat(edge_e.shape[0], 1)
        inverse_softmax_adj = torch.log(inverse_softmax_adj)
        zero_vec = -9e15 * torch.ones_like(inverse_softmax_adj).to(self.device)
        inverse_softmax_adj = torch.where(inverse_softmax_adj < -9e15, zero_vec, inverse_softmax_adj)

        edge_e *= inverse_softmax_adj

        # HNL 1
        # 3 HNLE, HNLE
        e_rowsum = self.spmm(self.edge_indices, edge_e.view(-1), torch.Size([self.edge_indices_num, V, V]),
                             torch.ones(size=(self.edge_indices_num, V, 1), device=self.device), bmm=True)

        edge_e = self.dropout(edge_e)
        # edge_e: HNL E;
        # h: H N C V L -> flatten: HNL V C
        flatten_h = h.permute(0, 1, 4, 3, 2).reshape(-1, V, self.c_out)
        # h_prime: HNL C
        h_prime = self.spmm(self.edge_indices, edge_e.view(-1), torch.Size([self.edge_indices_num, V, V]),
                            flatten_h, bmm=True)

        # softmax
        h_prime = h_prime.div(e_rowsum)
        # H N L V C
        h_prime = h_prime.view(self.num_heads, -1, L, V, self.c_out)
        h_avg = torch.mean(h_prime, dim=0).permute(0, 3, 2, 1)
        return F.elu(h_avg)
