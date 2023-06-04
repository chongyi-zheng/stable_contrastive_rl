import numpy as np
import torch
from torch import nn as nn
import torchvision

from rlkit.pythonplusplus import identity
from rlkit.torch.core import PyTorchModule


class Residual(nn.Module):
    def __init__(self,
                 num_channels,
                 residual_hidden_dim,
                 hidden_init=nn.init.xavier_uniform_,
                 hidden_activation=nn.ReLU(),
                 output_activation=identity):
        super(Residual, self).__init__()
        # self._block = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=num_channels,
        #         out_channels=residual_hidden_dim,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False),
        #     # nn.ReLU(True),
        #     hidden_activation,
        #     nn.Conv2d(
        #         in_channels=residual_hidden_dim,
        #         out_channels=num_channels,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False),
        # )
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=residual_hidden_dim,
            kernel_size=3, stride=1, padding=1)
        hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)

        self.conv2 = nn.Conv2d(
            in_channels=residual_hidden_dim,
            out_channels=num_channels,
            kernel_size=3, stride=1, padding=1)
        hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def forward(self, x):
        h = self.conv1(x)
        h = self.hidden_activation(h)
        h = self.conv2(h)
        out = self.output_activation(h + x)

        return out


class CNN(PyTorchModule):
    # TODO: remove the FC parts of this code
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            conv_normalization_type='none',
            fc_normalization_type='none',
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_conv_channels=False,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
            dropout_prob=0,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert conv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.conv_normalization_type = conv_normalization_type
        self.fc_normalization_type = fc_normalization_type
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.output_conv_channels = output_conv_channels
        self.pool_type = pool_type
        self.dropout_prob = dropout_prob

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.conv_dropout_layers = nn.ModuleList()

        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()
        self.fc_dropout_layers = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == 'max2d':
                self.pool_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_sizes[i],
                        stride=pool_strides[i],
                        padding=pool_paddings[i],
                    )
                )

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.conv_normalization_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.conv_normalization_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none':
                test_mat = self.pool_layers[i](test_mat)
            if self.dropout_prob > 0:
                self.conv_dropout_layers.append(nn.Dropout(self.dropout_prob))

        self.conv_output_flat_size = int(np.prod(test_mat.shape))
        if self.output_conv_channels:
            self.last_fc = None
        else:
            fc_input_size = self.conv_output_flat_size
            # used only for injecting input directly into fc layers
            fc_input_size += added_fc_input_size
            for idx, hidden_size in enumerate(hidden_sizes):
                fc_layer = nn.Linear(fc_input_size, hidden_size)
                fc_input_size = hidden_size

                # fc_layer.weight.data.uniform_(-init_w, init_w)
                # fc_layer.bias.data.uniform_(-init_w, init_w)
                hidden_init(fc_layer.weight)
                fc_layer.bias.data.fill_(0)

                self.fc_layers.append(fc_layer)

                if self.fc_normalization_type == 'batch':
                    self.fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
                if self.fc_normalization_type == 'layer':
                    self.fc_norm_layers.append(nn.LayerNorm(hidden_size))
                if self.dropout_prob > 0:
                    self.fc_dropout_layers.append(nn.Dropout(self.dropout_prob))

            self.last_fc = nn.Linear(fc_input_size, output_size)
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            # self.last_fc.bias.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_last_activations=False):
        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        h = self.apply_forward_conv(h)

        if self.output_conv_channels:
            return h

        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((h, extra_fc_input), dim=1)
        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h
        return self.output_activation(self.last_fc(h))

    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.conv_normalization_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none':
                h = self.pool_layers[i](h)
            if self.dropout_prob > 0:
                h = self.conv_dropout_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def apply_forward_fc(self, h):
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            if self.dropout_prob > 0:
                h = self.fc_dropout_layers[i](h)
            h = self.hidden_activation(h)
        return h


class ResCNN(PyTorchModule):
    # TODO: remove the FC parts of this code
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            conv_normalization_type='none',
            fc_normalization_type='none',
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_conv_channels=False,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
            dropout_prob=0,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert conv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        # self.output_size = output_size
        # self.output_activation = output_activation
        # self.hidden_activation = hidden_activation
        # self.conv_normalization_type = conv_normalization_type
        # self.fc_normalization_type = fc_normalization_type
        # self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.output_conv_channels = output_conv_channels
        # self.pool_type = pool_type
        # self.dropout_prob = dropout_prob
        #
        # self.conv_layers = nn.ModuleList()
        # self.res_conv_layers = nn.ModuleList()
        # self.conv_norm_layers = nn.ModuleList()
        # self.pool_layers = nn.ModuleList()
        # self.conv_dropout_layers = nn.ModuleList()
        #
        # self.fc_layers = nn.ModuleList()
        # self.fc_norm_layers = nn.ModuleList()
        # self.fc_dropout_layers = nn.ModuleList()
        #
        # for i, (out_channels, kernel_size, stride, padding) in enumerate(
        #         zip(n_channels, kernel_sizes, strides, paddings)
        # ):
        #     conv = nn.Conv2d(input_channels,
        #                      out_channels,
        #                      kernel_size,
        #                      stride=stride,
        #                      padding=padding)
        #     hidden_init(conv.weight)
        #     conv.bias.data.fill_(0)
        #
        #     res_conv = Residual(
        #         out_channels, out_channels,
        #         hidden_init=hidden_init,
        #         hidden_activation=hidden_activation,
        #         output_activation=identity)
        #
        #     # conv_layer = conv
        #     self.conv_layers.append(conv)
        #     self.res_conv_layers.append(res_conv)
        #     input_channels = out_channels
        #
        #     if pool_type == 'max2d':
        #         self.pool_layers.append(
        #             nn.MaxPool2d(
        #                 kernel_size=pool_sizes[i],
        #                 stride=pool_strides[i],
        #                 padding=pool_paddings[i],
        #             )
        #         )
        #
        # # use torch rather than ptu because initially the model is on CPU
        # test_mat = torch.zeros(
        #     1,
        #     self.input_channels,
        #     self.input_width,
        #     self.input_height,
        # )
        # # find output dim of conv_layers by trial and add norm conv layers
        # for i, (conv_layer, res_conv_layer) in enumerate(
        #         zip(self.conv_layers, self.res_conv_layers)):
        #     test_mat = conv_layer(test_mat)
        #     test_mat = res_conv_layer(test_mat)
        #     if self.conv_normalization_type == 'batch':
        #         self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
        #     if self.conv_normalization_type == 'layer':
        #         self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
        #     if self.pool_type != 'none':
        #         test_mat = self.pool_layers[i](test_mat)
        #     if self.dropout_prob > 0:
        #         self.conv_dropout_layers.append(nn.Dropout(self.dropout_prob))
        #
        # self.conv_output_flat_size = int(np.prod(test_mat.shape))

        full_resnet = torchvision.models.resnet18(
            pretrained=False)
        self.resnet_conv = nn.Sequential(*list(full_resnet.children())[:-2])

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        test_mat = self.resnet_conv(test_mat)
        self.conv_output_flat_size = int(np.prod(test_mat.shape))

        if self.output_conv_channels:
            self.last_fc = None
        else:
            fc_input_size = self.conv_output_flat_size
            # used only for injecting input directly into fc layers
            fc_input_size += added_fc_input_size
            for idx, hidden_size in enumerate(hidden_sizes):
                fc_layer = nn.Linear(fc_input_size, hidden_size)
                fc_input_size = hidden_size

                # fc_layer.weight.data.uniform_(-init_w, init_w)
                # fc_layer.bias.data.uniform_(-init_w, init_w)
                hidden_init(fc_layer.weight)
                fc_layer.bias.data.fill_(0)

                self.fc_layers.append(fc_layer)

                if self.fc_normalization_type == 'batch':
                    self.fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
                if self.fc_normalization_type == 'layer':
                    self.fc_norm_layers.append(nn.LayerNorm(hidden_size))
                if self.dropout_prob > 0:
                    self.fc_dropout_layers.append(nn.Dropout(self.dropout_prob))

            self.last_fc = nn.Linear(fc_input_size, output_size)
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            # self.last_fc.bias.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_last_activations=False):
        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        h = self.apply_forward_conv(h)

        if self.output_conv_channels:
            return h

        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((h, extra_fc_input), dim=1)
        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h
        return self.output_activation(self.last_fc(h))

    def apply_forward_conv(self, h):
        # for i, (conv_layer, res_conv_layer) in enumerate(
        #         zip(self.conv_layers, self.res_conv_layers)):
        #     h = conv_layer(h)
        #     h = res_conv_layer(h)
        #     if self.conv_normalization_type != 'none':
        #         h = self.conv_norm_layers[i](h)
        #     if self.pool_type != 'none':
        #         h = self.pool_layers[i](h)
        #     if self.dropout_prob > 0:
        #         h = self.conv_dropout_layers[i](h)
        #     h = self.hidden_activation(h)
        h = self.resnet_conv(h)

        return h

    def apply_forward_fc(self, h):
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            if self.dropout_prob > 0:
                h = self.fc_dropout_layers[i](h)
            h = self.hidden_activation(h)
        return h


class TwoChannelCNN(PyTorchModule):
    """
    Two-headed CNN, both with same input dimensions.
    """
    # TODO: remove the FC parts of this code
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            conv_normalization_type='none',
            fc_normalization_type='none',
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_conv_channels=False,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
            dropout_prob=0,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert conv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.conv_normalization_type = conv_normalization_type
        self.fc_normalization_type = fc_normalization_type
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.output_conv_channels = output_conv_channels
        self.pool_type = pool_type
        self.dropout_prob = dropout_prob

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.conv_dropout_layers = nn.ModuleList()

        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()
        self.fc_dropout_layers = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)
            conv_layer = conv
            self.conv_layers.append(conv_layer)

            input_channels = out_channels

            if pool_type == 'max2d':
                self.pool_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_sizes[i],
                        stride=pool_strides[i],
                        padding=pool_paddings[i],
                    )
                )

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        test_mat1 = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            test_mat1 = conv_layer(test_mat1)
            if self.conv_normalization_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.conv_normalization_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none':
                test_mat = self.pool_layers[i](test_mat)
                test_mat1 = self.pool_layers[i](test_mat1)
            if self.dropout_prob > 0:
                self.conv_dropout_layers.append(nn.Dropout(self.dropout_prob))

        self.conv_output_flat_size = int(np.prod(test_mat.shape))
        self.conv_output_flat_size1 = int(np.prod(test_mat1.shape))
        if self.output_conv_channels:
            self.last_fc = None
        else:
            fc_input_size = self.conv_output_flat_size + self.conv_output_flat_size1
            # used only for injecting input directly into fc layers
            fc_input_size += added_fc_input_size
            for idx, hidden_size in enumerate(hidden_sizes):
                fc_layer = nn.Linear(fc_input_size, hidden_size)
                fc_input_size = hidden_size

                # fc_layer.weight.data.uniform_(-init_w, init_w)
                # fc_layer.bias.data.uniform_(-init_w, init_w)
                hidden_init(fc_layer.weight)
                fc_layer.bias.data.fill_(0)

                self.fc_layers.append(fc_layer)

                if self.fc_normalization_type == 'batch':
                    self.fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
                if self.fc_normalization_type == 'layer':
                    self.fc_norm_layers.append(nn.LayerNorm(hidden_size))
                if self.dropout_prob > 0:
                    self.fc_dropout_layers.append(nn.Dropout(self.dropout_prob))

            self.last_fc = nn.Linear(fc_input_size, output_size)
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            # self.last_fc.bias.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_last_activations=False):
        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        h = self.apply_forward_conv(h)

        conv_input1 = input.narrow(start=self.conv_input_length,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h1 = conv_input1.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        h1 = self.apply_forward_conv(h1)

        if self.output_conv_channels:
            return h

        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        h1 = h1.view(h1.size(0), -1)
        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=2*self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h1 = torch.cat((h1, extra_fc_input), dim=1)

        h = torch.cat((h, h1), dim=1)
        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h
        return self.output_activation(self.last_fc(h))

    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.conv_normalization_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none':
                h = self.pool_layers[i](h)
            if self.dropout_prob > 0:
                h = self.conv_dropout_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def apply_forward_fc(self, h):
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            if self.dropout_prob > 0:
                h = self.fc_dropout_layers[i](h)
            h = self.hidden_activation(h)
        return h
