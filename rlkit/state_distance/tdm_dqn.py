from collections import OrderedDict

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.state_distance.tdm import TemporalDifferenceModel
from rlkit.torch.dqn.dqn import DQN


class TdmDqn(TemporalDifferenceModel, DQN):
    def __init__(
            self,
            env,
            qf,
            dqn_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            replay_buffer=None,
    ):
        DQN.__init__(self, env, qf, replay_buffer=replay_buffer,
                     policy=policy,
                     **dqn_kwargs,
                     **base_kwargs)
        super().__init__(**tdm_kwargs)

    def _do_training(self):
        if not self.vectorized:
            return DQN._do_training(self)
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['goals']
        num_steps_left = batch['num_steps_left']

        """
        Compute loss
        """

        target_q_values = self.target_qf(
            next_obs,
            goals,
            num_steps_left-1,
        ).detach().max(
            1, keepdim=False
        )[0]
        y_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(
            self.qf(obs, goals, num_steps_left) * actions.unsqueeze(2),
            dim=1,
            keepdim=False
        )
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        self._update_target_network()

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_pred),
            ))

    def evaluate(self, epoch):
        DQN.evaluate(self, epoch)
