from __future__ import print_function

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from typing import Iterable

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DiscountModelTrainer(TorchTrainer):
    def __init__(
            self,
            model,
            optimizer,
            equality_threshold=0,
            observation_key='observations',
            action_key='actions',
            next_observation_key='next_observations',
            goal_key='desired_goal_key',
            state_to_goal_fn=None,
    ):
        #TODO: add KL regualrization
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.equality_threshold = equality_threshold
        self._criterion = nn.BCEWithLogitsLoss()
        self._eval_statistics = OrderedDict()
        self._obs_key = observation_key
        self._action_key = action_key
        self._next_obs_key = next_observation_key
        self._goal_key = goal_key
        self._need_to_update_eval_statistics = True
        if state_to_goal_fn is None:
            state_to_goal_fn = lambda x: x
        self._state_to_goal_fn = state_to_goal_fn

    def train_from_torch(self, batch):
        obs = batch[self._obs_key]
        action = batch[self._action_key]
        next_obs = batch[self._next_obs_key]
        goal = batch[self._goal_key]
        achieved = self._state_to_goal_fn(next_obs)
        not_reached = (
                torch.norm((achieved - goal), dim=1)
                > self.equality_threshold
        )

        not_reached_logit = self.model(obs, action, return_logits=True)
        not_reached_logit = not_reached_logit[:, 0]
        loss = self._criterion(
            not_reached_logit,
            not_reached.to(torch.float)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self._eval_statistics['loss'] = np.mean(ptu.get_numpy(loss))
            not_reached_predicted = torch.sigmoid(not_reached_logit)
            self._eval_statistics.update(create_stats_ordered_dict(
                'discount_predicted',
                ptu.get_numpy(not_reached_predicted),
            ))
            self._eval_statistics.update(create_stats_ordered_dict(
                'not_reached/mean',
                np.mean(ptu.get_numpy(not_reached))
            ))

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self._eval_statistics)
        return stats

    @property
    def networks(self) -> Iterable[nn.Module]:
        return [self.model]
