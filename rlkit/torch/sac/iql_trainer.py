"""Torch implementation of Implicit Q-Learning (IQL)
https://github.com/ikostrikov/implicit_q_learning
"""

from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
from rlkit.torch.networks import LinearTransform


class IQLTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            vf,
            quantile=0.5,
            target_qf1=None,
            target_qf2=None,
            buffer_policy=None,
            z=None,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            policy_weight_decay=0,
            q_weight_decay=0,
            optimizer_class=optim.Adam,
            bc=False,

            policy_update_period=1,
            q_update_period=1,

            reward_transform_class=None,
            reward_transform_kwargs=None,
            terminal_transform_class=None,
            terminal_transform_kwargs=None,

            clip_score=None,
            soft_target_tau=1e-2,
            target_update_period=1,
            beta=1.0,
            min_value=None,
            max_value=None,
            *args,
            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.vf = vf
        self.z = z
        self.buffer_policy = buffer_policy
        self.bc = bc

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.optimizers = {}

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            weight_decay=policy_weight_decay,
            lr=policy_lr,
        )
        self.optimizers[self.policy] = self.policy_optimizer
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )

        if self.z:
            self.z_optimizer = optimizer_class(
                self.z.parameters(),
                weight_decay=q_weight_decay,
                lr=qf_lr,
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period

        self.reward_transform_class = reward_transform_class or LinearTransform
        self.reward_transform_kwargs = reward_transform_kwargs or dict(
            m=1, b=0)
        self.terminal_transform_class = (
            terminal_transform_class or LinearTransform)
        self.terminal_transform_kwargs = terminal_transform_kwargs or dict(
            m=1, b=0)
        self.reward_transform = self.reward_transform_class(
            **self.reward_transform_kwargs)
        self.terminal_transform = self.terminal_transform_class(
            **self.terminal_transform_kwargs)

        self.clip_score = clip_score
        self.beta = beta
        self.quantile = quantile

        self.min_value = min_value
        self.max_value = max_value

    def train_from_torch(self, batch, train=True, pretrain=False,):
        reward = batch['rewards']
        terminal = batch['terminals']
        obs = batch['observations']
        action = batch['actions']
        next_obs = batch['next_observations']

        if self.reward_transform:
            reward = self.reward_transform(reward)
        if self.terminal_transform:
            terminal = self.terminal_transform(terminal)

        dist = self.policy(obs)

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, action)
        q2_pred = self.qf2(obs, action)
        target_vf_pred = self.vf(next_obs).detach()
        if self.min_value is not None or self.max_value is not None:
            target_vf_pred = torch.clamp(
                target_vf_pred,
                min=self.min_value,
                max=self.max_value,
            )

        q_target = self.reward_scale * reward + \
            (1. - terminal) * self.discount * target_vf_pred
        q_target = q_target.detach()
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """
        VF Loss
        """
        q_pred = torch.min(
            self.target_qf1(obs, action),
            self.target_qf2(obs, action),
        ).detach()
        if self.min_value is not None or self.max_value is not None:
            q_pred = torch.clamp(
                q_pred,
                min=self.min_value,
                max=self.max_value,
            )

        vf_pred = self.vf(obs)
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + \
            vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()

        """
        Policy Loss
        """
        policy_logpp = dist.log_prob(action)

        adv = q_pred - vf_pred
        exp_adv = torch.exp(adv / self.beta)
        if self.clip_score is not None:
            exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        weights = ptu.from_numpy(np.ones(exp_adv[:, 0].shape)).detach(
        ) if self.bc else exp_adv[:, 0].detach()
        policy_loss = (-policy_logpp * weights).mean()

        """
        MSE Loss
        """
        with torch.no_grad():
            mse_loss = (dist.mean - action) ** 2

        """
        Update networks
        """
        if self._n_train_steps_total % self.q_update_period == 0:
            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

        if self._n_train_steps_total % self.policy_update_period == 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['MSE Loss'] = np.mean(ptu.get_numpy(
                mse_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Mean',
                ptu.get_numpy(dist.mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy STD',
                ptu.get_numpy(dist.stddev),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'reward',
                ptu.get_numpy(reward),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'terminal',
                ptu.get_numpy(terminal),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), 'policy/')
            self.eval_statistics.update(policy_statistics)
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Weights',
                ptu.get_numpy(weights),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Score',
                ptu.get_numpy(adv),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'V1 Predictions',
                ptu.get_numpy(vf_pred),
            ))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))

            self.eval_statistics['beta'] = self.beta
            self.eval_statistics['quantile'] = self.quantile

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.vf,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            vf=self.vf,
        )
