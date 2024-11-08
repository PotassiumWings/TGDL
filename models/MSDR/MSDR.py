import torch
import torch.nn as nn

from configs.MSDR_configs import MSDRConfig
from models.MSDR.layers import GMSDRCell, Seq2SeqAttrs
from models.abstract_st_encoder import AbstractSTEncoder


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config: MSDRConfig):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config)
        self.input_dim = config.c_in
        self.seq_len = config.input_len
        self.mlp = nn.Linear(self.input_dim, self.rnn_units)
        self.gmsdr_layers = nn.ModuleList(
            [GMSDRCell(self.rnn_units, self.input_dim, self.max_diffusion_step, self.num_nodes, self.pre_k, self.pre_v,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k, graph):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hx_k: (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
               optional, zeros if not provided
        :param graph: (num_nodes, num_nodes)
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hx_k # shape (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        batch = inputs.shape[0]
        x = inputs.reshape(batch, self.num_nodes, self.input_dim)
        output = self.mlp(x).view(batch, -1)
        for layer_num, dcgru_layer in enumerate(self.gmsdr_layers):
            next_hidden_state, new_hx_k = dcgru_layer(output, hx_k[layer_num], graph)
            hx_ks.append(new_hx_k)
            output = next_hidden_state
        return output, torch.stack(hx_ks)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config: MSDRConfig, c_out, output_len):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config)
        self.output_dim = c_out
        self.horizon = output_len
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.gmsdr_layers = nn.ModuleList(
            [GMSDRCell(self.rnn_units, self.rnn_units, self.max_diffusion_step, self.num_nodes, self.pre_k, self.pre_v,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k, graph):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hx_k: (num_layers, batch_size, pre_k, num_nodes, rnn_units)
               optional, zeros if not provided
        :param graph: (num_nodes, num_nodes)
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.gmsdr_layers):
            next_hidden_state, new_hx_k = dcgru_layer(output, hx_k[layer_num], graph)
            hx_ks.append(new_hx_k)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hx_ks)


class MSDR(AbstractSTEncoder, Seq2SeqAttrs):
    def __init__(self, config: MSDRConfig, gp_supports):
        super(MSDR, self).__init__(config, gp_supports)
        Seq2SeqAttrs.__init__(self, config)
        self.encoder_model = EncoderModel(config)
        self.decoder_pred = DecoderModel(config, c_out=config.c_out, output_len=config.output_len)
        self.decoder_residual = DecoderModel(config, c_out=config.c_in, output_len=config.input_len)

    def encoder(self, inputs, graph):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        :param graph: (num_nodes, num_nodes)
        """
        hx_k = torch.zeros(self.num_rnn_layers, inputs.shape[1], self.pre_k, self.num_nodes, self.rnn_units,
                           device=self.device)
        outputs = []
        for t in range(self.encoder_model.seq_len):
            output, hx_k = self.encoder_model(inputs[t], hx_k, graph)
            outputs.append(output)
        return torch.stack(outputs), hx_k

    def decoder(self, decode_model: DecoderModel, inputs, hx_k, graph):
        """
        Decoder forward pass
        :param decode_model: DecoderModel
        :param inputs: (seq_len, batch_size, num_sensor * rnn_units)
        :param hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        :param graph: (num_nodes, num_nodes)
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        decoder_hx_k = hx_k
        decoder_input = inputs

        outputs = []
        for t in range(decode_model.horizon):
            decoder_output, decoder_hx_k = decode_model(decoder_input[t], decoder_hx_k, graph)
            outputs.append(decoder_output)
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, x, supports):
        """
        seq2seq forward pass
        :param supports: supports
        :param x: shape (batch_size, input_dim, num_sensor, seq_len)
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size, c_in, num_nodes, input_len = x.size()
        # inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        inputs = x.permute(3, 0, 2, 1).view(input_len, batch_size, c_in * num_nodes)
        graph = supports[0]

        encoder_outputs, hx_k = self.encoder(inputs, graph)
        # (horizon, batch_size, self.num_nodes * self.output_dim)
        outputs_pred = self.decoder(self.decoder_pred, encoder_outputs, hx_k, graph)
        outputs_residual = self.decoder(self.decoder_residual, encoder_outputs, hx_k, graph)

        outputs_pred = outputs_pred.reshape(self.config.output_len, batch_size, num_nodes, self.config.c_out) \
            .permute(1, 3, 2, 0)
        outputs_residual = outputs_residual.reshape(self.config.input_len, batch_size, num_nodes, self.config.c_in) \
            .permute(1, 3, 2, 0)
        return outputs_pred, outputs_residual
