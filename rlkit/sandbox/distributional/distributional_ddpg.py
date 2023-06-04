from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.ddpg import DDPG


class FeedForwardZFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            observation_hidden_size,
            embedded_hidden_size,
            num_bins,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            batchnorm_obs=False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size
        self.hidden_init = hidden_init

        self.obs_fc = nn.Linear(obs_dim, observation_hidden_size)
        self.embedded_fc = nn.Linear(observation_hidden_size + action_dim,
                                     embedded_hidden_size)
        self.last_fc = nn.Linear(embedded_hidden_size, num_bins)

        self.init_weights(init_w)
        self.batchnorm_obs = batchnorm_obs
        if self.batchnorm_obs:
            self.bn_obs = nn.BatchNorm1d(obs_dim)

    def init_weights(self, init_w):
        self.hidden_init(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        self.hidden_init(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        if self.batchnorm_obs:
            obs = self.bn_obs(obs)
        h = obs
        h = F.relu(self.obs_fc(h))
        h = torch.cat((h, action), dim=1)
        h = F.relu(self.embedded_fc(h))
        return F.softmax(self.last_fc(h))


class DistributionalDDPG(DDPG):
    def __init__(
            self,
            env,
            zf,
            policy,
            exploration_strategy,
            num_bins,
            returns_min,
            returns_max,
            **kwargs
    ):
        super().__init__(
            env,
            zf,
            policy,
            exploration_strategy,
            **kwargs
        )
        self.num_bins = num_bins
        self.returns_min = returns_min
        self.returns_max = returns_max
        assert num_bins > 1, "Need at least two bins"
        self.bin_width = (returns_max - returns_min) / (num_bins - 1)

    def create_atom_values(self, batch_size):
        atom_values_batch = np.expand_dims(
            np.linspace(self.returns_min, self.returns_max, self.num_bins),
            0,
        ).repeat(batch_size, 0)
        return ptu.np_to_var(atom_values_batch)

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        batch_size = obs.size()[0]

        """
        Policy operations.
        """
        policy_actions = self.policy(obs)
        z_output = self.qf(obs, policy_actions)  # BATCH_SIZE x NUM_BINS
        q_output = (z_output * self.create_atom_values(batch_size)).sum(1)
        policy_loss = - q_output.mean()

        """
        Critic operations.
        """
        next_actions = self.target_policy(next_obs)
        target_qf_histogram = self.target_qf(
            next_obs,
            next_actions,
        )
        # z_target = ptu.Variable(torch.zeros(self.batch_size, self.num_bins))

        rewards_batch = rewards.repeat(1, self.num_bins)
        terminals_batch = terminals.repeat(1, self.num_bins)
        projected_returns = (
            self.reward_scale * rewards_batch
            + (1. - terminals_batch) * self.discount *
            self.create_atom_values(batch_size)
        )
        projected_returns = torch.clamp(
            projected_returns, self.returns_min, self.returns_max
        )
        bin_values = (projected_returns - self.returns_min) / self.bin_width
        lower_bin_indices = torch.floor(bin_values)
        upper_bin_indices = torch.ceil(bin_values)
        lower_bin_deltas = target_qf_histogram * (
            upper_bin_indices - bin_values
        )
        upper_bin_deltas = target_qf_histogram * (
            bin_values - lower_bin_indices
        )

        z_target_np = np.zeros((batch_size, self.num_bins))
        lower_deltas_np = lower_bin_deltas.data.numpy()
        upper_deltas_np = upper_bin_deltas.data.numpy()
        lower_idxs_np = lower_bin_indices.data.numpy().astype(int)
        upper_idxs_np = upper_bin_indices.data.numpy().astype(int)
        for batch_i in range(self.batch_size):
            for bin_i in range(self.num_bins):
                z_target_np[batch_i, bin_i] += (
                    lower_deltas_np[batch_i, lower_idxs_np[batch_i, bin_i]]
                )
                z_target_np[batch_i, bin_i] += (
                    upper_deltas_np[batch_i, upper_idxs_np[batch_i, bin_i]]
                )
        z_target = ptu.Variable(ptu.from_numpy(z_target_np).float())

        # for j in range(self.num_bins):
        #     import ipdb; ipdb.set_trace()
        #     atom_value = self.atom_values_batch[:, j:j+1]
        #     projected_returns = self.reward_scale * rewards + (1. - terminals) * self.discount * (
        #         atom_value
        #     )
        #     bin_values = (projected_returns - self.returns_min) / self.bin_width
        #     lower_bin_indices = torch.floor(bin_values)
        #     upper_bin_indices = torch.ceil(bin_values)
        #     lower_bin_deltas = target_qf_histogram[:, j:j+1] * (
        #         upper_bin_indices - bin_values
        #     )
        #     upper_bin_deltas = target_qf_histogram[:, j:j+1] * (
        #         bin_values - lower_bin_indices
        #     )
        #     new_lower_bin_values = torch.gather(
        #         z_target, 1, lower_bin_indices.long().data
        #     ) + lower_bin_deltas
        #     new_upper_bin_values = torch.gather(
        #         z_target, 1, upper_bin_indices.long().data
        #     ) + upper_bin_deltas

        # noinspection PyUnresolvedReferences
        z_pred = self.qf(obs, actions)
        qf_loss = - (z_target * torch.log(z_pred)).sum(1).mean(0)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
            ('Z targets', z_target),
            ('Z predictions', z_pred),
            ('QF Loss', qf_loss),
        ])

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = OrderedDict()

        train_dict = self.get_train_dict(batch)
        for name in [
            'QF Loss',
            'Policy Loss',
        ]:
            tensor = train_dict[name]
            statistics_name = "{} {} Mean".format(stat_prefix, name)
            statistics[statistics_name] = np.mean(ptu.get_numpy(tensor))

        # for name in [
        #     'Bellman Errors',
        # ]:
        #     tensor = train_dict[name]
        #     statistics.update(create_stats_ordered_dict(
        #         '{} {}'.format(stat_prefix, name),
        #         ptu.get_numpy(tensor)
        #     ))

        return statistics
