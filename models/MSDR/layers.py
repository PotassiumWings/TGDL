import torch
import torch.nn.functional as F
from torch import nn, Tensor

from configs.MSDR_configs import MSDRConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_random_walk_matrix(adj_mx):
    adj_mx = adj_mx + torch.eye(len(adj_mx)).to(adj_mx.device)
    d = torch.sum(adj_mx, dim=1)  # out dim
    d_inv = torch.pow(d, -1)
    d_inv_mask = torch.isinf(d_inv)
    zeros = torch.zeros_like(d_inv)
    d_inv = torch.where(d_inv_mask, zeros, d_inv)

    res = torch.mm(torch.diag(d_inv), adj_mx)
    return res


class Seq2SeqAttrs:
    def __init__(self, config: MSDRConfig):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_nodes = config.num_nodes
        self.max_diffusion_step = config.max_diffusion_step
        self.filter_type = config.filter_type
        self.num_rnn_layers = config.num_rnn_layers
        self.rnn_units = config.rnn_units
        self.hidden_state_size = config.num_nodes * config.rnn_units
        self.pre_k = config.pre_k
        self.pre_v = config.pre_v


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class GMSDRCell(torch.nn.Module):
    def __init__(self, num_units, input_dim, max_diffusion_step, num_nodes, pre_k, pre_v, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """

        :param num_units:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        # intermediate variables
        self.adp = None
        self._supports = None

        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._filter_type = filter_type
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        self.pre_k = pre_k
        self.pre_v = pre_v
        self.input_dim = input_dim
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')
        self.W = nn.Parameter(torch.zeros(self._num_units, self._num_units), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_nodes, self._num_units), requires_grad=True)
        self.R = nn.Parameter(torch.zeros(pre_k, num_nodes, self._num_units), requires_grad=True)
        self.attlinear = nn.Linear(num_nodes * self._num_units, 1)

    def forward(self, inputs, hx_k, graph):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx_k: (B, pre_k, num_nodes, rnn_units)
        :param graph: (num_nodes, num_nodes)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """

        if self._filter_type == "dual_random_walk":
            self._supports = [calculate_random_walk_matrix(graph).t(), calculate_random_walk_matrix(graph.t()).t()]
        else:
            raise NotImplementedError(f"Filter type {self._filter_type} is not implemented.")

        bs, k, n, d = hx_k.shape
        preH = hx_k[:, -1:]
        for i in range(1, self.pre_v):
            preH = torch.cat([preH, hx_k[:, -(i + 1):-i]], -1)
        preH = preH.reshape(bs, n, d * self.pre_v)
        self.adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        convInput = F.leaky_relu_(self._gconv(inputs, preH, d, bias_start=1.0))
        new_states = hx_k + self.R.unsqueeze(0)
        output = torch.matmul(convInput, self.W) + self.b.unsqueeze(0) + self.attention(new_states)
        output = output.unsqueeze(1)
        x = hx_k[:, 1:k]
        hx_k = torch.cat([x, output], dim=1)
        output = output.reshape(bs, n * d)
        return output, hx_k

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        # input / state: (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1
        x1 = self.adp.mm(x0)
        x = self._concat(x, x1)
        for k in range(2, self._max_diffusion_step + 1):
            x2 = self.adp.mm(x1) - x0
            x = self._concat(x, x2)
            x1, x0 = x2, x1
        num_matrices = (len(self._supports) + 1) * self._max_diffusion_step + 1
        # num_matrices = (len(self._supports)) * self._max_diffusion_step + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes, output_size])

    def attention(self, inputs: Tensor):
        bs, k, n, d = inputs.size()
        x = inputs.reshape(bs, k, -1)
        out = self.attlinear(x)
        weight = F.softmax(out, dim=1)
        outputs = (x * weight).sum(dim=1).reshape(bs, n, d)
        return outputs
