import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule

from rlkit.experimental.kuanfang.vae import network_utils  # NOQA
from rlkit.experimental.kuanfang.vae import affordance_networks  # NOQA


def batch_gather(batched_data, batched_indices):
    batch_size = batched_data.shape[0]
    ret = [batched_data[i, batched_indices[i].item()]
           for i in range(batch_size)]
    ret = torch.stack(ret, 0)
    return ret


class MultitaskCcVae(PyTorchModule):

    def __init__(
            self,
            data_channels,
            data_root_len=12,
            z_dim=4,

            # hidden_dim=32,
            # unet_inner_dim=128,
            # z_fc_dim=128,

            hidden_dim=16,
            unet_inner_dim=64,
            z_fc_dim=64,

            num_tasks=6,
    ):
        super(MultitaskCcVae, self).__init__()

        self._num_tasks = num_tasks

        self._networks = []
        for k in range(self._num_tasks):
            network = affordance_networks.CcVae(
                data_channels=data_channels,
                data_root_len=data_root_len,
                z_dim=z_dim,
                hidden_dim=hidden_dim,
                unet_inner_dim=unet_inner_dim,
                z_fc_dim=z_fc_dim,
            ).to(ptu.device)
            self._networks.append(network)

        self.representation_size = self._networks[0].representation_size
        self.data_root_len = self._networks[0].data_root_len

    @property
    def networks(self):
        return self._networks

    @property
    def num_tasks(self):
        return self._num_tasks

    def encode(self, data, cond, skill_id):
        # skill_id = skill_id.item()
        mu, logvar = self._networks[skill_id].encode(data, cond)
        return mu, logvar

    def decode(self, z, cond, skill_id):
        recon = self._networks[skill_id].decode(z, cond)
        return recon

    def forward(self, data, cond, skill_id):
        mu, logvar = self.encode(data, cond, skill_id)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond, skill_id)
        return (mu, logvar), z, recon

    def sample_prior(self, batch_size):
        zs = ptu.randn(batch_size, self.representation_size)
        return ptu.get_numpy(zs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
