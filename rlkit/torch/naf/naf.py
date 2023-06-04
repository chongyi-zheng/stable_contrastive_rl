from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from rlkit.core.eval_util import (
    create_stats_ordered_dict,
)
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.core import PyTorchModule


# noinspection PyCallingNonCallable
class NAF(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            target_policy,
            exploration_policy=None,
            policy_learning_rate=1e-3,
            target_hard_update_period=1000,
            tau=0.001,
            use_soft_update=False,
            **kwargs
    ):
        if exploration_policy is None:
            exploration_policy = policy
        super().__init__(
            env,
            exploration_policy,
            **kwargs
        )
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.target_policy = target_policy
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update

        self.policy_criterion = nn.MSELoss()
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )

    def _do_training(self):
        batch = self.get_batch()

        """
        Optimize Critic/Actor.
        """
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        _, _, v_pred = self.target_policy(next_obs, None)
        y_target = self.reward_scale * rewards + (1. - terminals) * self.discount * v_pred
        y_target = y_target.detach()
        mu, y_pred, v = self.policy(obs, actions)
        policy_loss = self.policy_criterion(y_pred, y_target)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Update Target Networks
        """
        if self.use_soft_update:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
        else:
            if self._n_train_steps_total% self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.policy, self.target_policy)

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy v',
                ptu.get_numpy(v),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(mu),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y targets',
                ptu.get_numpy(y_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y predictions',
                ptu.get_numpy(y_pred),
            ))

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            policy=self.eval_policy,
            trained_policy=self.policy,
            target_policy=self.target_policy,
            exploration_policy=self.exploration_policy,
        )
        return snapshot

    @property
    def networks(self):
        return [
            self.policy,
            self.target_policy,
        ]


class NafPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_size,
            use_batchnorm=False,
            b_init_value=0.01,
            hidden_init=ptu.fanin_init,
            use_exp_for_diagonal_not_square=True,
    ):
        super(NafPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_batchnorm = use_batchnorm
        self.use_exp_for_diagonal_not_square = use_exp_for_diagonal_not_square

        if use_batchnorm:
            self.bn_state = nn.BatchNorm1d(obs_dim)
            self.bn_state.weight.data.fill_(1)
            self.bn_state.bias.data.fill_(0)

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.L = nn.Linear(hidden_size, action_dim ** 2)

        self.tril_mask = ptu.Variable(
            torch.tril(
                torch.ones(action_dim, action_dim),
                -1
            ).unsqueeze(0)
        )
        self.diag_mask = ptu.Variable(torch.diag(
            torch.diag(
                torch.ones(action_dim, action_dim)
            )
        ).unsqueeze(0))

        hidden_init(self.linear1.weight)
        self.linear1.bias.data.fill_(b_init_value)
        hidden_init(self.linear2.weight)
        self.linear2.bias.data.fill_(b_init_value)
        hidden_init(self.V.weight)
        self.V.bias.data.fill_(b_init_value)
        hidden_init(self.L.weight)
        self.L.bias.data.fill_(b_init_value)
        hidden_init(self.mu.weight)
        self.mu.bias.data.fill_(b_init_value)

    def forward(self, state, action, return_P=False):
        if self.use_batchnorm:
            state = self.bn_state(state)
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        V = self.V(x)
        mu = torch.tanh(self.mu(x))

        Q = None
        P = None
        if action is not None or return_P:
            num_outputs = mu.size(1)
            raw_L = self.L(x).view(-1, num_outputs, num_outputs)
            L = raw_L * self.tril_mask.expand_as(raw_L)
            if self.use_exp_for_diagonal_not_square:
                L += torch.exp(raw_L) * self.diag_mask.expand_as(raw_L)
            else:
                L += torch.pow(raw_L, 2) * self.diag_mask.expand_as(raw_L)
            P = torch.bmm(L, L.transpose(2, 1))

            if action is not None:
                u_mu = (action - mu).unsqueeze(2)
                A = - 0.5 * torch.bmm(
                    torch.bmm(u_mu.transpose(2, 1), P), u_mu
                ).squeeze(2)

                Q = A + V

        if return_P:
            return mu, Q, V, P
        return mu, Q, V

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action, _, _ = self.__call__(obs, None)
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}

    def get_action_and_P_matrix(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action, _, _, P = self.__call__(obs, None, return_P=True)
        action = action.squeeze(0)
        P = P.squeeze(0)
        return ptu.get_numpy(action), ptu.get_numpy(P)
