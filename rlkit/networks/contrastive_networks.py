import torch
from torch import nn

from rlkit.torch.networks import Mlp, CNN
from rlkit.torch import pytorch_util as ptu


class ContrastiveQf(nn.Module):
    def __init__(self,
                 hidden_sizes,
                 representation_dim,
                 action_dim,
                 obs_dim=None,
                 use_image_obs=False,
                 imsize=None,
                 img_encoder_arch='cnn',
                 repr_norm=False,
                 repr_norm_temp=True,
                 repr_log_scale=None,
                 twin_q=False,
                 **kwargs,
                 ):
        super().__init__()

        self._use_image_obs = use_image_obs
        self._imsize = imsize
        self._img_encoder_arch = img_encoder_arch
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._representation_dim = representation_dim
        self._repr_norm = repr_norm
        self._repr_norm_temp = repr_norm_temp
        self._twin_q = twin_q

        self._img_encoder = None
        if self._use_image_obs:
            assert isinstance(imsize, int)
            cnn_kwargs = kwargs.copy()
            layer_norm = cnn_kwargs.pop('layer_norm')

            if self._img_encoder_arch == 'cnn':
                cnn_kwargs['input_width'] = imsize
                cnn_kwargs['input_height'] = imsize
                cnn_kwargs['input_channels'] = 3
                cnn_kwargs['output_size'] = None
                cnn_kwargs['kernel_sizes'] = [8, 4, 3]
                cnn_kwargs['n_channels'] = [32, 64, 64]
                cnn_kwargs['strides'] = [4, 2, 1]
                cnn_kwargs['paddings'] = [2, 1, 1]
                cnn_kwargs['conv_normalization_type'] = 'layer' if layer_norm else 'none'
                cnn_kwargs['fc_normalization_type'] = 'layer' if layer_norm else 'none'
                cnn_kwargs['output_conv_channels'] = True
                self._img_encoder = CNN(**cnn_kwargs)
            else:
                raise RuntimeError("Unknown image encoder architecture: {}".format(
                    self._img_encoder_arch))

        state_dim = self._img_encoder.conv_output_flat_size if self._use_image_obs else self._obs_dim

        self._sa_encoder = Mlp(
            hidden_sizes, representation_dim, state_dim + self._action_dim,
            **kwargs,
        )
        self._g_encoder = Mlp(
            hidden_sizes, representation_dim, state_dim,
            **kwargs,
        )
        self._sa_encoder2 = Mlp(
            hidden_sizes, representation_dim, state_dim + self._action_dim,
            **kwargs,
        )
        self._g_encoder2 = Mlp(
            hidden_sizes, representation_dim, state_dim,
            **kwargs,
        )

        if self._repr_norm_temp:
            if repr_log_scale is None:
                self._repr_log_scale = nn.Parameter(
                    ptu.zeros(1, requires_grad=True))
            else:
                assert isinstance(repr_log_scale, float)
                self._repr_log_scale = repr_log_scale

    @property
    def repr_norm(self):
        return self._repr_norm

    @property
    def repr_log_scale(self):
        return self._repr_log_scale

    def _compute_representation(self, obs, action, hidden=None):
        # The optional input hidden is the image representations. We include this
        # as an input for the second Q value when twin_q = True, so that the two Q
        # values use the same underlying image representation.
        if hidden is None:
            if self._use_image_obs:
                imlen = self._imsize * self._imsize * 3
                state = self._img_encoder(obs[:, :imlen]).flatten(1)
                goal = self._img_encoder(obs[:, imlen:]).flatten(1)
            else:
                state = obs[:, :self._obs_dim]
                goal = obs[:, self._obs_dim:]
        else:
            state, goal = hidden

        if hidden is None:
            sa_repr = self._sa_encoder(torch.cat([state, action], dim=-1))
            g_repr = self._g_encoder(goal)
        else:
            sa_repr = self._sa_encoder2(torch.cat([state, action], dim=-1))
            g_repr = self._g_encoder2(goal)

        if self._repr_norm:
            sa_repr = sa_repr / torch.norm(sa_repr, dim=1, keepdim=True)
            g_repr = g_repr / torch.norm(g_repr, dim=1, keepdim=True)

            if self._repr_norm_temp:
                sa_repr = sa_repr / torch.exp(self._repr_log_scale)

        return sa_repr, g_repr, (state, goal)

    def forward(self, obs, action, repr=False):
        sa_repr, g_repr, hidden = self._compute_representation(
            obs, action)
        outer = torch.bmm(sa_repr.unsqueeze(0), g_repr.permute(1, 0).unsqueeze(0))[0]

        if self._twin_q:
            sa_repr2, g_repr2, _ = self._compute_representation(
                obs, action, hidden)
            outer2 = torch.bmm(sa_repr2.unsqueeze(0), g_repr2.permute(1, 0).unsqueeze(0))[0]

            outer = torch.stack([outer, outer2], dim=-1)

        if repr:
            sa_repr_norm = torch.norm(sa_repr, dim=-1)
            g_repr_norm = torch.norm(g_repr, dim=-1)

            sa_repr_norm2 = torch.norm(sa_repr2, dim=-1)
            g_repr_norm2 = torch.norm(g_repr2, dim=-1)

            sa_repr = torch.stack([sa_repr, sa_repr2], dim=-1)
            g_repr = torch.stack([g_repr, g_repr2], dim=-1)
            sa_repr_norm = torch.stack([sa_repr_norm, sa_repr_norm2], dim=-1)
            g_repr_norm = torch.stack([g_repr_norm, g_repr_norm2], dim=-1)

            return outer, sa_repr, g_repr, sa_repr_norm, g_repr_norm
        else:
            return outer
