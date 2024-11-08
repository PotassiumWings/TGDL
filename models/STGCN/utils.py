import torch
import torch.nn as nn
import torch.nn.functional as F


class Align(nn.Module):
    """
    Align (batch_size, c_in, num_nodes, time_steps) to
          (batch_size, c_out, num_nodes, time_steps).
    if c_in < c_out, padding input to the same size of c_out.
    else, bottleneck down-sampling(conv input to c_out).
    """

    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x):  # x: (batch_size, c_in, num_nodes, time_steps)
        if self.c_in > self.c_out:
            return self.conv(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  # return: (batch_size, c_out, num_nodes, time_steps)


"""
Copied from https://github.com/Diego999/pyGAT/blob/master/layers.py
"""


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b, bmm):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        ctx.bmm = bmm
        if bmm:
            return torch.bmm(a, b)
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            if ctx.bmm:
                grad_a_dense = torch.bmm(grad_output, b.permute(0, 2, 1))
            else:
                grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            if ctx.bmm:
                grad_b = torch.bmm(torch.transpose(a, 1, 2), grad_output)
            else:
                grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b, None


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b, bmm):
        return SpecialSpmmFunction.apply(indices, values, shape, b, bmm)
