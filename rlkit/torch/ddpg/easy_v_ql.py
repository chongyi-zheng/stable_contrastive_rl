from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import init

from rlkit.torch.core import PyTorchModule
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.ddpg import DDPG
import rlkit.torch.pytorch_util as ptu
import rlkit.torch.networks.experimental as M


class EasyVQLearning(DDPG):
    """
    Continous action Q learning where the V function is easy:

    Q(s, a) = A(s, a) + V(s)

    The main thing is that the following needs to be enforced:

        max_a A(s, a) = 0

    """
    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs)
        v, a = self.qf(obs, policy_actions)
        policy_loss = - a.mean()

        """
        Critic operations.
        """
        next_actions = self.policy(next_obs)
        # TODO: try to get this to work
        # next_actions = None
        v_target, a_target = self.target_qf(
            next_obs,
            next_actions,
        )
        y_target = self.reward_scale * rewards + (1. - terminals) * self.discount * v_target
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        v_pred, a_pred = self.qf(obs, actions)
        y_pred = v_pred + a_pred
        bellman_errors = (y_pred - y_target)**2
        qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('Policy Action Value', v),
            ('Policy Action Advantage', a),
            ('Target Value', v_target),
            ('Target Advantage', a_target),
            ('Predicted Value', v_pred),
            ('Predicted Advantage', a_pred),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
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

        for name in [
            'Bellman Errors',
            'Target Value',
            'Target Advantage',
            'Predicted Value',
            'Predicted Advantage',
            'Policy Action Value',
            'Policy Action Advantage',
        ]:
            tensor = train_dict[name]
            statistics.update(create_stats_ordered_dict(
                '{} {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
            ))

        return statistics


class EasyVQFunction(PyTorchModule):
    """
    Parameterize Q function as the follows:

        Q(s, a) = A(s, a) + V(s)

    To ensure that max_a A(s, a) = 0, use the following form:

        A(s, a) = - diff(s, a)^T diag(exp(d(s))) diff(s, a)  *  f(s, a)^2

    where

        diff(s, a) = a - z(s)

    so that a = z(s) is at least one zero.

    d(s) and f(s, a) are arbitrary functions
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            diag_fc1_size,
            diag_fc2_size,
            af_fc1_size,
            af_fc2_size,
            zero_fc1_size,
            zero_fc2_size,
            vf_fc1_size,
            vf_fc2_size,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs_batchnorm = nn.BatchNorm1d(obs_dim)

        self.batch_square = M.BatchSquareDiagonal(action_dim)

        self.diag = nn.Sequential(
            nn.Linear(obs_dim, diag_fc1_size),
            nn.ReLU(),
            nn.Linear(diag_fc1_size, diag_fc2_size),
            nn.ReLU(),
            nn.Linear(diag_fc2_size, action_dim),
        )

        self.zero = nn.Sequential(
            nn.Linear(obs_dim, zero_fc1_size),
            nn.ReLU(),
            nn.Linear(zero_fc1_size, zero_fc2_size),
            nn.ReLU(),
            nn.Linear(zero_fc2_size, action_dim),
        )

        self.f = nn.Sequential(
            M.Concat(),
            nn.Linear(obs_dim + action_dim, af_fc1_size),
            nn.ReLU(),
            nn.Linear(af_fc1_size, af_fc2_size),
            nn.ReLU(),
            nn.Linear(af_fc2_size, 1),
        )

        self.vf = nn.Sequential(
            nn.Linear(obs_dim, vf_fc1_size),
            nn.ReLU(),
            nn.Linear(vf_fc1_size, vf_fc2_size),
            nn.ReLU(),
            nn.Linear(vf_fc2_size, 1),
        )

        self.apply(init_layer)

    def forward(self, obs, action):
        obs = self.obs_batchnorm(obs)
        V = self.vf(obs)
        if action is None:
            A = None
        else:
            diag_values = torch.exp(self.diag(obs))
            diff = action - self.zero(obs)
            quadratic = self.batch_square(diff, diag_values)
            f = self.f((obs, action))
            A = - quadratic * (f**2)

        return V, A


def init_layer(layer):
    if isinstance(layer, nn.Linear):
        init.kaiming_normal(layer.weight)
        layer.bias.data.fill_(0)
    elif isinstance(layer, nn.BatchNorm1d):
        layer.reset_parameters()
