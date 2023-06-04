from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
from rlkit.core.timer import timer

PGLosses = namedtuple(
    'PGLosses',
    'policy_loss vf_loss',
)

class VPGTrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            env,
            policy,
            vf,
            replay_buffer,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            vf_lr=1e-3,
            optimizer_class=optim.Adam,

            vf_iters_per_step=80,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.vf = vf
        self.replay_buffer = replay_buffer

        self.vf_criterion = nn.MSELoss()
        self.vf_iters_per_step = vf_iters_per_step

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        timer.start_timer('pg training', unique=False)
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        obs = batch['observations']
        returns = batch['returns'][:, 0]

        for _ in range(self.vf_iters_per_step):
            v = self.vf(obs)[:, 0]
            v_error = (v - returns) ** 2
            vf_loss = v_error.mean()

            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

        self._n_train_steps_total += 1
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False

            stats["VF Loss"] = np.mean(ptu.get_numpy(
                vf_loss
            ))
            stats.update(create_stats_ordered_dict(
                "VF",
                ptu.get_numpy(v),
            ))

        timer.stop_timer('pg training')

        self.replay_buffer.empty_buffer()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[PGLosses, LossStatistics]:
        obs = batch['observations']
        actions = batch['actions']
        returns = batch['returns'][:, 0]
        """
        Policy operations.
        """

        dist = self.policy.forward(obs,)
        log_probs = dist.log_prob(actions)
        advantage = returns - self.vf(obs)[:, 0]
        J = log_probs * advantage
        policy_loss = -J.mean()

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))

        loss = PGLosses(
            policy_loss=policy_loss,
            vf_loss=None,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.vf,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.vf_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            vf=self.vf,
        )
