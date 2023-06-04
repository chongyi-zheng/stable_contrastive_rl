from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F
from rlkit.torch import pytorch_util as ptu


class Residual(nn.Module):

    def __init__(self,
                 in_channels,
                 residual_hidden_dim,
                 out_activation='relu'):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=residual_hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=residual_hidden_dim,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
        )
        self._out_activation = out_activation

    def forward(self, x):
        if self._out_activation == 'relu':
            return F.relu(x + self._block(x))
        else:
            return x + self._block(x)


class ResidualStack(nn.Module):

    def __init__(self,
                 in_channels,
                 num_residual_layers,
                 residual_hidden_dim,
                 out_activation='relu'):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers

        layers = []
        for i in range(self._num_residual_layers):
            if i == self._num_residual_layers - 1:
                out_activation_i = out_activation
            else:
                out_activation_i = 'relu'

            layer = Residual(in_channels=in_channels,
                             residual_hidden_dim=residual_hidden_dim,
                             out_activation=out_activation_i)
            layers.append(layer)

        self._layers = nn.ModuleList(layers)

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 input_nc=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc,
                             inner_nc,
                             kernel_size=4,
                             stride=2,
                             padding=1,
                             bias=False)
        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)

        if innermost:
            upconv = nn.ConvTranspose2d(inner_nc,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv]

        elif outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)
            down = [downconv]
            up = [uprelu, upconv, nn.ReLU(True)]

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, cond):
        state = x
        state = self.down(state)
        state = self.submodule(state, cond)
        state = self.up(state)

        if self.outermost:
            return state
        else:
            # Add skip connections
            return torch.cat([x, state], 1)


class FlattenFCBlock(nn.Module):

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 cond_nc,
                 input_w=3,
                 input_h=3,
                 input_nc=None,
                 use_fc=False,
                 use_dropout=False):
        super(FlattenFCBlock, self).__init__()
        if input_nc is None:
            input_nc = outer_nc

        self.outer_nc = outer_nc
        self.inner_nc = inner_nc
        self.input_nc = input_nc
        self.input_w = input_w
        self.input_h = input_h

        self.flatten = Flatten()

        self.down = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(
                input_w * input_h * input_nc + cond_nc,
                inner_nc),
        )
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(
                inner_nc,
                input_w * input_h * outer_nc),
        )

    def forward(self, x, cond):
        state = self.flatten(x)
        state = torch.cat([state, cond], 1)
        state = self.down(state)
        state = self.up(state)
        state = state.view(-1, self.outer_nc, self.input_w, self.input_h)
        return torch.cat([x, state], 1)
