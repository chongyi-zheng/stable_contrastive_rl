import gin
import torch
from torch import nn

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.util.io import load_local_or_remote_file

from rlkit.experimental.kuanfang.vae import network_utils


@gin.configurable
class CcVae(PyTorchModule):

    def __init__(
            self,
            data_channels,
            decoder_output_layer_channels=None,
            data_root_len=12,
            z_dim=8,

            hidden_dim=64,

            use_normalization=False,
            output_scaling_factor=None,
    ):
        super(CcVae, self).__init__()

        if data_root_len == 12:
            embedding_root_len = 3
        else:
            raise ValueError

        unet_inner_dim = hidden_dim * 8
        z_fc_dim = hidden_dim * 8

        self.representation_size = z_dim
        self.data_root_len = data_root_len

        if decoder_output_layer_channels:
            self.decoder_output_layer_channels = decoder_output_layer_channels
        else:
            self.decoder_output_layer_channels = data_channels
        self.data_channels = data_channels

        self._encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.data_channels * 2,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Conv2d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),

            network_utils.Flatten(),
            nn.Linear(
                hidden_dim * embedding_root_len * embedding_root_len,
                z_fc_dim,
                bias=False,
            ),
            nn.ReLU(True),
        )

        self._mu_layer = nn.Sequential(
            nn.Linear(z_fc_dim, z_dim),
        )

        self._logvar_layer = nn.Sequential(
            nn.Linear(z_fc_dim, z_dim),
        )

        self._z_encoder = nn.Sequential(
            nn.Linear(z_dim, z_fc_dim, bias=False),
            nn.ReLU(True),
        )

        # Decoder
        unet_block = network_utils.FlattenFCBlock(
            outer_nc=2 * hidden_dim,
            inner_nc=unet_inner_dim,
            input_nc=None,
            cond_nc=z_fc_dim,
        )
        unet_block = network_utils.UnetSkipConnectionBlock(
            outer_nc=hidden_dim,
            inner_nc=2 * hidden_dim,
            input_nc=None,
            submodule=unet_block)
        unet_block = network_utils.UnetSkipConnectionBlock(
            outer_nc=hidden_dim,
            inner_nc=hidden_dim,
            input_nc=self.data_channels,
            submodule=unet_block,
            outermost=True)

        self._decoder = unet_block

        self._output_layer = [
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=self.decoder_output_layer_channels,
                kernel_size=1,
                stride=1)
        ]

        if use_normalization:
            self._output_layer += [nn.Tanh()]

        self._output_layer = nn.Sequential(*self._output_layer)
        self._output_scaling_factor = output_scaling_factor

    def encode(self, data, cond):
        assert data.shape[-1] == self.data_root_len
        assert data.shape[-2] == self.data_root_len
        assert cond.shape[-1] == self.data_root_len
        assert cond.shape[-2] == self.data_root_len

        img_diff = cond - data
        encoder_input = torch.cat((img_diff, cond), dim=1)

        feat = self._encoder(encoder_input)
        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        return mu, logvar

    def decode(self, z, cond):
        cond = cond.view(
            -1,
            self.data_channels,
            self.data_root_len,
            self.data_root_len,
        )

        z_feat = self._z_encoder(z)

        state = self._decoder.forward(cond, cond=z_feat)
        recon = self._output_layer(state)

        if self._output_scaling_factor is not None:
            recon = recon * self._output_scaling_factor

        return recon

    def forward(self, data, cond):
        mu, logvar = self.encode(data, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return (mu, logvar), z, recon

    def sample_prior(self, batch_size):
        z_s = ptu.randn(batch_size, self.representation_size)
        return ptu.get_numpy(z_s)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


@gin.configurable
class SimpleCcVae(PyTorchModule):

    def __init__(
            self,
            data_dim,
            z_dim=8,

            hidden_dim=128,

            output_scaling_factor=None,
    ):
        super(SimpleCcVae, self).__init__()

        self.representation_size = z_dim

        # Encoder
        self._encoder = nn.Sequential(
            nn.Linear(
                data_dim + data_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),
        )

        self._mu_layer = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
        )

        self._logvar_layer = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
        )

        # Decoder
        self._decoder = nn.Sequential(
            nn.Linear(
                z_dim + data_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                data_dim,
                bias=False)
        )

    def encode(self, data, cond):
        encoder_input = torch.cat((data, cond), dim=1)
        feat = self._encoder(encoder_input)
        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        return mu, logvar

    def decode(self, z, cond):
        decoder_input = torch.cat((z, cond), dim=1)
        recon = self._decoder(decoder_input)
        return recon

    def forward(self, data, cond):
        mu, logvar = self.encode(data, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return (mu, logvar), z, recon

    def sample_prior(self, batch_size):
        z_s = ptu.randn(batch_size, self.representation_size)
        return ptu.get_numpy(z_s)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


@gin.configurable
class DeltaSimpleCcVae(PyTorchModule):

    def __init__(
            self,
            data_dim,
            z_dim=8,

            hidden_dim=128,

            output_scaling_factor=None,
    ):
        super(DeltaSimpleCcVae, self).__init__()

        self.representation_size = z_dim

        # Encoder
        self._encoder = nn.Sequential(
            nn.Linear(
                data_dim + data_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),
        )

        self._mu_layer = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
        )

        self._logvar_layer = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
        )

        # Decoder
        self._decoder = nn.Sequential(
            nn.Linear(
                z_dim + data_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                data_dim,
                bias=False)
        )

    def encode(self, data, cond):
        encoder_input = torch.cat((data, cond), dim=1)
        feat = self._encoder(encoder_input)
        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        return mu, logvar

    def decode(self, z, cond):
        decoder_input = torch.cat((z, cond), dim=1)
        recon = cond + self._decoder(decoder_input)
        return recon

    def forward(self, data, cond):
        mu, logvar = self.encode(data, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return (mu, logvar), z, recon

    def sample_prior(self, batch_size):
        z_s = ptu.randn(batch_size, self.representation_size)
        return ptu.get_numpy(z_s)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


class HierarchicalCcVae(PyTorchModule):

    def __init__(
            self,
            multiplier=2,
            num_levels=4,
            min_dt=10,
            path_list=[],
            load_from_trainer=False,
    ):
        super(HierarchicalCcVae, self).__init__()

        if not load_from_trainer:
            self._networks = []
            for level, path in enumerate(path_list):
                print('Level %d: Loading from %s ...' % (level, path))
                network = torch.load(path).to(ptu.device)
                self._networks.append(network)
        else:
            self._networks = []
            for level, path in enumerate(path_list):
                print('Level %d: Loading from %s ...' % (level, path))
                rl_model_dict = load_local_or_remote_file(path)
                network = rl_model_dict['trainer/affordance'].to(ptu.device)
                self._networks.append(network)

        try:
            self.data_root_len = self._networks[0].data_root_len
            for network in self._networks:
                assert self.data_root_len == network.data_root_len
        except Exception:
            pass

    @property
    def networks(self):
        return self._networks

    def encode(self, data, cond):
        return self._networks[-1].encode(data, cond)

    def decode(self, z, cond):
        return self._networks[-1].decode(z, cond)

    def forward(self, data, cond):
        raise NotImplementedError

    def sample_prior(self, batch_size):
        return self._networks[-1].sample_prior(batch_size)

    def reparameterize(self, mu, logvar):
        return self._networks[-1].reparameterize(mu, logvar)


class Classifier(PyTorchModule):

    def __init__(
            self,
            data_channels,
            hidden_dim=32,
            fc_dim=64,
            imsize=12,
            decay=0.0,
    ):
        super(Classifier, self).__init__()

        self.imsize = imsize
        self.hidden_dim = hidden_dim

        self._encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=data_channels,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.ReLU(),

            network_utils.Flatten(),

            nn.Linear(hidden_dim * 9, fc_dim),
            nn.ReLU(),

        )

        self._joint_layers = nn.Sequential(
            nn.Linear(2 * fc_dim, fc_dim),
            nn.ReLU(),

            nn.Linear(fc_dim, 1),
        )

    def __call__(self, h0, h1):
        assert h0.shape[-1] == self.imsize
        assert h0.shape[-2] == self.imsize
        assert h1.shape[-1] == self.imsize
        assert h1.shape[-2] == self.imsize

        encoded_h0 = self._encoder(h0)
        encoded_h1 = self._encoder(h1)
        net = torch.cat((encoded_h0, encoded_h1), dim=-1)
        logits = self._joint_layers(net)
        return logits


class Discriminator(PyTorchModule):

    def __init__(
            self,
            data_channels,
            hidden_dim=64,
            imsize=12,
            decay=0.0,
    ):
        super(Discriminator, self).__init__()

        self.imsize = imsize

        self.hidden_dim = hidden_dim

        self._layer = nn.Sequential(
            nn.Conv2d(
                in_channels=data_channels * 2,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            network_utils.Flatten(),

            nn.Linear(hidden_dim * 9, 1),
        )

    def __call__(self, h0, h1):
        assert h0.shape[-1] == self.imsize
        assert h0.shape[-2] == self.imsize
        assert h1.shape[-1] == self.imsize
        assert h1.shape[-2] == self.imsize

        inputs = torch.cat((h1 - h0, h1), dim=1)
        logits = self._layer(inputs)
        return logits
