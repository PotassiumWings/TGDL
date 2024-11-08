import logging

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering as SC
from sklearn.manifold import TSNE

from configs.arguments import TrainingArguments
from models import loss
from models.md_block import MDBlock
from utils.normalize import Scaler


class Norm(nn.Module):
    def __init__(self, supports_01):
        super(Norm, self).__init__()
        self.supports_01 = supports_01  # (N, N) 0/1

    def forward(self, x, index):
        # norm and mask
        # return torch.sigmoid(x)  # * self.adj
        return (torch.tanh(x) + 1) / 2 * self.supports_01[index]


class NoneNorm(nn.Module):
    def __init__(self):
        super(NoneNorm, self).__init__()

    def forward(self, x, index):
        return x


def get_traditional_partition_single(graph, ns, spectral_clustering):
    deg = torch.sum(graph, dim=0)
    N = len(graph)
    subgraphs = [torch.zeros_like(graph) for i in range(ns)]
    if spectral_clustering:
        labels = SC(n_clusters=ns).fit(graph.cpu().numpy()).labels_
        for i in range(N):
            for j in range(N):
                if graph[i][j] == 0:
                    continue
                if labels[i] == labels[j]:
                    subgraphs[labels[i]][i][j] = 1
    else:
        for i in range(N):
            for j in range(i):
                if graph[i][j] == 0:
                    continue
                u, v = i, j
                if deg[i] < deg[j]:
                    v, u = j, i
                # deg[u] > deg[v]
                idx = u % ns
                subgraphs[idx][u][v] = subgraphs[idx][v][u] = 1
    subgraphs = [sg.to(graph.device) for sg in subgraphs]
    return subgraphs


class NormGraphs:
    def __init__(self, norm: nn.Module, graphs, supports, traditional=False, spectral_clustering=False):
        self.norm = norm
        self.graphs = graphs
        self.supports = supports
        if traditional:
            self.graphs = self.get_traditional_partition(spectral_clustering)

    def get_traditional_partition(self, spectral_clustering):
        # raise NotImplementedError("E")
        # gs: num_supports * num_subgraphs
        gs = self.graphs
        result = []
        for i in range(len(gs)):
            result.append(get_traditional_partition_single(gs[i][0], len(gs[i]), spectral_clustering))
        return result

    def get_graphs(self, softmax, gumbel=False, temperature=1e-6):
        # graphs = [self.norm(g) for g in self.graphs]
        # list(num_supports, num_subgraphs) Tensor([V, V]), p \in (0, 1)
        graphs = [[self.norm(g, i) for g in self.graphs[i]] for i in range(len(self.graphs))]
        result_graphs = []
        if softmax:
            raise NotImplementedError("Softmax norm is not implemented.")
            # for i in range(len(graphs)):
            #     graph = graphs[i]
            #     graph_tensor = torch.stack(graph)  # K V V
            #     softmax_graphs = torch.softmax(graph_tensor, dim=0)
            #     result_graphs = softmax_graphs * self.adj
            #     ret.append(result_graphs)
        elif gumbel:
            def bernoulli_sampling(input_graph):
                eps = 1e-20

                # get categorical distribution of graph
                inverse_graph = 1 - input_graph
                graph = torch.stack([inverse_graph, input_graph])

                # [0, 1) uniform distribution
                uniform_sample = torch.rand_like(graph)
                # gumbel distribution y = -log(-log(x))
                gumbel_sample = -torch.log(-torch.log(uniform_sample + eps) + eps)

                # gumbel + log(graph) ~ categorical(graph)
                gumbel_graph = gumbel_sample + torch.log(graph + eps)
                # argmax
                argmax_graph = torch.argmax(gumbel_graph, dim=0)

                return (argmax_graph - input_graph).detach() + input_graph

            def gumbel_softmax(input_graph):
                inverse_graph = 1 - input_graph
                graph = torch.stack([torch.log(inverse_graph + 1e-20), torch.log(input_graph + 1e-20)])
                return torch.nn.functional.gumbel_softmax(graph, tau=temperature, hard=True, dim=0)[1]

            ret = [[gumbel_softmax(graph) for graph in graphs] for graphs in self.graphs]
        else:
            ret = graphs

        for i in range(len(ret[0])):
            supports = []
            for j in range(len(ret)):
                supports.append(ret[j][i])
            result_graphs.append(supports)
        return result_graphs


class TGDL(nn.Module):
    def __init__(self, config: TrainingArguments, supports, scaler: Scaler, device):
        super(TGDL, self).__init__()
        self.config = config
        self.device = device
        self.scaler = scaler

        def to01(x):
            y = x.clone()
            y[torch.where(y != 0)] = 1
            return y

        self.supports = [torch.Tensor(support).to(device) for support in supports]  # supports (N, N)
        self.supports_01 = [to01(x) for x in self.supports]

        self.num_nodes = config.num_nodes
        assert self.num_nodes == self.supports[0].shape[0] == self.supports[0].shape[1]
        self.num_blocks = config.num_subgraphs

        if config.loss == "MAE":
            self.loss_func = loss.masked_mae_loss(config.mae_mask)
        elif config.loss == "Huber":
            self.loss_func = loss.masked_huber_loss(config.mae_mask, config.loss_delta)
        elif config.loss == "InvHuber":
            self.loss_func = loss.masked_inv_huber_loss(config.mae_mask, config.loss_delta)
        else:
            raise AttributeError(f"Loss {config.loss} is not implemented."
                                 f"Available losses are: MAE, Huber, InvHuber.")

        self.gs, self.mds = [], []  # self.gs: shape (num_supports, num_subgraphs) self.gs[i]: ith support subgraphs

        for j in range(self.config.num_supports):
            subgraph_list = []
            for i in range(self.config.num_subgraphs):
                if config.compare:  # 对比实验，K 个原图不训练
                    g = nn.Parameter(torch.Tensor(supports[j]).to(device), requires_grad=False)
                else:
                    g = nn.Parameter(torch.zeros(size=(self.num_nodes, self.num_nodes)).to(device),
                                     requires_grad=config.subgraph_grad and not config.traditional)
                self.register_parameter(f"g{j}-{i}", g)
                subgraph_list.append(g)
            self.gs.append(subgraph_list)

        for i in range(self.config.num_subgraphs):
            md = MDBlock(config, self.supports)
            self.add_module(f"md{i}", md)
            self.mds.append(md)

        if config.traditional:
            self.norm_subgraph = NormGraphs(NoneNorm(), self.gs, self.supports,
                                            traditional=True, spectral_clustering=config.spectral_clustering)
        elif not config.compare:
            self.norm_subgraph = NormGraphs(Norm(self.supports_01), self.gs, self.supports)
        else:
            self.norm_subgraph = NormGraphs(NoneNorm(), self.gs, self.supports)

        self.weights = []
        for i in range(self.config.num_subgraphs):
            weight = nn.Parameter(torch.zeros(size=(1, self.config.c_out, self.num_nodes, self.config.output_len)),
                                  requires_grad=True)
            self.register_parameter(f"w{i}", weight)
            self.weights.append(weight)

        self.reset_parameters()

    def reset_parameters(self):
        if self.config.compare:
            # 对比实验，K 个原图不训练
            return

        for i in range(self.num_blocks):
            nn.init.uniform_(self.weights[i].data, a=-0.5, b=0.5)

        if self.config.start_from_pre:
            for j in range(self.config.num_supports):
                graphs = get_traditional_partition_single(self.supports[j], self.num_blocks, False)
                for i in range(self.num_blocks):
                    self.gs[j][i].data = graphs[i] * 2e6 - 1e6
        else:
            for i in range(self.num_blocks):
                for j in range(self.config.num_supports):
                    nn.init.uniform_(self.gs[j][i].data, a=-3, b=3)

    def forward(self, x, temperature, return_part=False, tsne=False, time="Default"):
        # x: NCVL
        all_pred = torch.zeros(size=(x.shape[0], self.config.c_out, self.config.num_nodes, self.config.output_len)) \
            .to(self.device)
        part_preds = []
        part_residuals = []
        graphs = self.get_graphs(temperature)

        cur_x = x
        forward_loss = 0

        # get sum weight (for softmax)
        sum_exp_weight = 0
        for i in range(self.num_blocks):
            w = torch.exp(self.weights[i])
            sum_exp_weight += w

        for i in range(self.num_blocks):
            if self.config.use_origin_x:
                pred, residual = self.mds[i](x, [graphs[i][j] * self.supports[j] for j in range(len(graphs[i]))])
            else:
                pred, residual = self.mds[i](cur_x, [graphs[i][j] * self.supports[j] for j in range(len(graphs[i]))])

            forward_loss += self.mds[i].get_forward_loss()

            # pred: N C V L
            pred = self.scaler.inverse_transform(pred.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            # w: 1 C V L
            w = torch.exp(self.weights[i]) / sum_exp_weight

            weighted_pred = pred * w

            if return_part:
                part_preds.append(weighted_pred)
                part_residuals.append(self.scaler.inverse_transform(cur_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) -
                                      self.scaler.inverse_transform(residual.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

            if self.config.remove_backward or self.config.straight:
                cur_x = residual
            else:
                cur_x = cur_x - residual

            if self.config.straight:
                all_pred = pred
            else:
                all_pred += weighted_pred

        if tsne:
            data = []
            for i in range(self.num_blocks):
                embedding = self.mds[i].get_embedding()[0]  # N C V L -> C V L
                embedding = embedding.permute(1, 0, 2).reshape(self.num_nodes, -1)  # V C*L
                data.append(embedding)
            self.visualize_tsne(torch.cat(data, dim=0).cpu().numpy(), filename=time)

        if return_part:
            # B N C V L
            part_preds = torch.stack(part_preds)
            part_residuals = torch.stack(part_residuals)
        return self.scaler.inverse_transform(cur_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), all_pred, \
            part_preds, part_residuals, forward_loss

    def visualize(self, dir_name):
        # pass
        graphs = self.get_graphs()
        for i in range(len(graphs)):
            for j in range(len(graphs[i])):
                graph = graphs[i][j].data.cpu().numpy()
                np.save(f"{dir_name}/{i - j}", graph)

    def calculate_loss(self, x_residual, y_pred, y, loss_weights, lamb, temperature, ignore_loss34, loss5):
        loss1 = torch.mean(torch.abs(x_residual))

        if type(loss5) == int:
            loss5_val = loss5
        else:
            loss5_val = loss5.item()

        if self.config.c_out == 1:  # BJTraj c=1
            loss2 = self.loss_func(y_pred, y)
        elif self.config.c_out == 2:

            loss21 = self.loss_func(y_pred[:, 0, :, :], y[:, 0, :, :])
            loss22 = self.loss_func(y_pred[:, 1, :, :], y[:, 1, :, :])

            loss2 = lamb * loss21 + (1 - lamb) * loss22

            if ignore_loss34 or self.config.compare:
                logging.info(f"L1: {round(loss1.item(), 2)}, L2: {round(loss21.item(), 2)} {round(loss22.item(), 2)}, "
                             f"L5: {round(loss5_val, 2)}")
                return loss1 * loss_weights[0] + loss2 * loss_weights[1] + loss5 * loss_weights[4], \
                    np.array([loss1.item(), loss2.item(), 0, 0, loss5_val])
        else:
            losses = [self.loss_func(y_pred[:, i, :, :], y[:, i, :, :]) for i in range(self.config.c_out)]
            loss2 = sum(losses) / len(losses)

        graphs = self.get_graphs(temperature)
        loss3 = loss.get_supports_similarities(graphs, self.config.num_nodes, self.device,
                                               self.config.nnd, self.config.nndT, self.config.orth_reg)
        graphs = self.get_graphs(temperature)
        loss4 = loss.get_supports_subgraph_loss(graphs, self.supports_01, self.config.require_sum,
                                                self.config.require_le)

        info_str = f"L1: {round(loss1.item(), 2)}, "
        if self.config.c_out == 1:
            info_str += f"L2: {round(loss2.item(), 2)}"
        elif self.config.c_out == 2:
            info_str += f" L2: {round(loss21.item(), 2)} {round(loss22.item(), 2)}, "
        else:
            s = ""
            for i in range(self.config.c_out):
                s += f"{round(losses[i].item(), 2)} "
            info_str += f" L2: {s}, "
        logging.info(
            f"{info_str} L3: {round(loss3.item(), 2)}, L4: {round(loss4.item(), 2)}, L5: {round(loss5_val, 2)}")

        return loss1 * loss_weights[0] + loss2 * loss_weights[1] + \
               loss3 * loss_weights[2] + loss4 * loss_weights[3] + loss5 * loss_weights[4], \
            np.array([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5_val])

    def print(self):
        for g in self.get_graphs():
            logging.info(g)

    def get_graphs(self, temperature=0.):
        return self.norm_subgraph.get_graphs(self.config.softmax_graphs, self.config.gumbel_softmax, temperature)

    def visualize_tsne(self, data, filename):
        # KV * ?
        k = self.num_blocks
        v = self.num_nodes

        labels = []
        for i in range(k):
            for j in range(v):
                labels.append(i)
        labels = np.array(labels)

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2)  # Reduce data to 2 dimensions
        data_tsne = tsne.fit_transform(data[:, :])
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='tab10')

        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"figs/{filename}.pdf", format="pdf", dpi=300, bbox_inches="tight")
        exit(0)
