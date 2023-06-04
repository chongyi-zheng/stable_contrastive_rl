from __future__ import print_function

from collections import OrderedDict
from typing import Iterable

import numpy as np
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import nn


class GenerativeGoalDynamicsModelTrainer(TorchTrainer):
    def __init__(
            self,
            model,
            optimizer,
            state_to_goal,
            observation_key='observations',
            action_key='actions',
            next_observation_key='next_observations',
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.state_to_goal = state_to_goal
        self._eval_statistics = OrderedDict()
        self._obs_key = observation_key
        self._action_key = action_key
        self._next_obs_key = next_observation_key
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        obs = batch[self._obs_key]
        action = batch[self._action_key]
        next_obs = batch[self._next_obs_key]
        next_goal = self.state_to_goal(next_obs)

        # from rlkit.torch.pgr.dynamics_model import EnsembleToGaussian
        # if isinstance(self.model, EnsembleToGaussian):
        #     print("TODO: probably want to train differently")
        dist = self.model(obs, action)
        log_likelihood = dist.log_prob(next_goal)
        loss = - log_likelihood.mean()

        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self._eval_statistics['loss'] = np.mean(ptu.get_numpy(loss))
            self._eval_statistics.update(create_stats_ordered_dict(
                'log_likelihood',
                ptu.get_numpy(log_likelihood),
            ))
            self._eval_statistics.update(create_stats_ordered_dict(
                'mean',
                ptu.get_numpy(dist.mean),
            ))
            self._eval_statistics.update(create_stats_ordered_dict(
                'variance',
                ptu.get_numpy(dist.variance),
            ))

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self._eval_statistics)
        return stats

    @property
    def networks(self) -> Iterable[nn.Module]:
        return [self.model]
