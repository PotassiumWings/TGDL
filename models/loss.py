import math

import numpy as np
import torch
import torch.nn as nn


def masked_mae_loss(mask_value):
    def loss(preds, labels):
        mae = mae_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss


def masked_huber_loss(mask_value, delta):
    def loss(preds, labels):
        return huber_torch(preds, labels, mask_value, delta)

    return loss


def masked_inv_huber_loss(mask_value, delta):
    def loss(preds, labels):
        return inv_huber_torch(preds, labels, mask_value, delta)

    return loss


def huber_torch(pred, true, mask_value, delta):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    a = torch.abs(pred - true)
    a_great = torch.masked_select(a, torch.gt(a, delta))
    a_small = torch.masked_select(a, torch.lt(a, delta))
    size = a_great.size(0) + a_small.size(0)
    return (torch.sum(a_small * a_small) * 0.5 + delta * torch.sum(a_great - 0.5 * delta)) / size


def inv_huber_torch(pred, true, mask_value, delta):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    a = torch.abs(pred - true)
    a_great = torch.masked_select(a, torch.gt(a, delta))
    a_small = torch.masked_select(a, torch.lt(a, delta))
    size = a_great.size(0) + a_small.size(0)
    return (torch.sum(a_great * a_great + 0.5 * delta * delta) * 0.5 + delta * torch.sum(a_small)) / size


def mae_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def mape_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def rmse_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean(torch.square(pred - true)))


def masked_mse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


# def masked_mae_torch(preds, labels, null_val=np.nan):
#     labels[torch.abs(labels) < 1e-4] = 0
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = labels.ne(null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(torch.sub(preds, labels))
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)


# def masked_mape_torch(preds, labels, null_val=np.nan):
#     labels[torch.abs(labels) < 1e-4] = 0
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = labels.ne(null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs((preds - labels) / labels)
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)


def get_graphs_from_supports(gs):
    # gs: [num_subgraphs, num_supports]
    graphs = [[] for i in range(len(gs[0]))]
    for supports in gs:
        for i in range(len(supports)):
            support = supports[i]
            graphs[i].append(support)
    return graphs


def get_supports_similarities(gs, node_count, device, nnd, T, orth):
    graphs = get_graphs_from_supports(gs)
    result = 0
    for i in range(len(graphs)):
        result += get_graph_similarities(graphs[i], node_count, device, nnd, T, orth)
    return result


def get_graph_similarities(gs, node_count, device, nnd, T, orth):
    result = torch.Tensor([0]).to(device)
    num_graphs = len(gs)
    for i in range(num_graphs):
        for j in range(num_graphs):
            if i == j:
                continue
            result += get_graph_similarity(gs[i], gs[j], device, nnd, T, orth)
    if torch.isnan(result):
        import pdb
        pdb.set_trace()

    if num_graphs > 1:
        return result / num_graphs / (num_graphs - 1)
    return result
    # return result / (num_graphs * (num_graphs - 1) / 2)


def get_graph_similarity(g1, g2, device, nnd=False, T=5, orth=False):
    if orth:
        return torch.mean(torch.mm(g1.t(), g2))

    if not nnd:
        return -torch.sum(torch.abs(g1 - g2))
    N = len(g1)
    p1 = get_squares(g1 + torch.eye(N).to(device), T) / (N - 1)  # pij: fraction of nodes that have distance i to node j
    p2 = get_squares(g2 + torch.eye(N).to(device), T) / (N - 1)
    mu1 = torch.mean(p1, dim=1)  # T
    mu2 = torch.mean(p2, dim=1)  # T
    # nnd1 = entropia(mu1) - entropia(p1) / N
    # nnd2 = entropia(mu2) - entropia(p2) / N
    w1 = 0.5
    w2 = 0.5
    # return -w1 * torch.sqrt(jensen_shannon_divergence(mu1, mu2) / math.log(2)) +\
    #        -w2 * torch.abs(torch.sqrt(max(nnd1, 0)) - torch.sqrt(max(nnd2, 0)))
    return -w1 * torch.sqrt(jensen_shannon_divergence(mu1, mu2) / math.log(2)) + \
        -w2 * jensen_shannon_divergence(p1, p2)


def jensen_shannon_divergence(p, q):
    kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
    # p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    p, q = p.view(-1, p.size(-1)) + 1e-8, q.view(-1, q.size(-1)) + 1e-8
    m = (0.5 * (p + q)).log()
    return 0.5 * (kl(m, p.log()) + kl(m, q.log()))


def entropia(x):  # x >= 0
    return -torch.sum(x * torch.log(x + 1e-8))


def get_squares(g, T):  # 这个必须返回的是 >0 的值！
    res = [g]
    for i in range(T - 1):
        res.append(torch.mm(res[-1], g))
    for i in range(T - 1, -1, -1):
        temp = f(res[i])
        if i > 0:
            temp -= f(res[i - 1])
        res[i] = torch.sum(temp, dim=1)
    res = torch.stack(res)
    zeros = torch.zeros_like(res)
    return torch.where(res < 0, zeros, res)


def get_supports_subgraph_loss(gs, g, require_sum=False, require_le=False):
    graphs = get_graphs_from_supports(gs)
    result = 0
    for i in range(len(graphs)):
        result += get_subgraph_loss(graphs[i], g[i], require_sum)
    return result


def get_subgraph_loss(gs, g, require_sum=False, require_le=False):
    sum_g = torch.zeros_like(gs[0])
    for i in range(0, len(gs)):
        sum_g += gs[i]
    if require_le:
        return torch.sum(torch.max(sum_g - g, 0))
    if not require_sum:
        sum_g = f(sum_g)
        g = f(g)
    return torch.sum(torch.abs(sum_g - g))


def f(x, T=4):
    return (torch.tanh(T * (x - 0.5)) + 1) / 2
