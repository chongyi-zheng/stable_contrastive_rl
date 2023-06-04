from collections import OrderedDict

import numpy as np
from cached_property import cached_property

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.util import split_paths
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rlkit.core import logger
from sandbox.rocky.tf.spaces import Box


class HiddenCartpoleEnv(CartpoleEnv):
    def __init__(self, num_steps=100, position_only=True):
        assert position_only, "I only added position_only due to some weird " \
                              "serialization bug"
        CartpoleEnv.__init__(self, position_only=position_only)
        self.num_steps = num_steps

    @cached_property
    def action_space(self):
        return Box(super().action_space.low,
                   super().action_space.high)

    @cached_property
    def observation_space(self):
        return Box(super().observation_space.low,
                   super().observation_space.high)

    @property
    def horizon(self):
        return self.num_steps

    @staticmethod
    def get_extra_info_dict_from_batch(batch):
        return dict()

    @staticmethod
    def get_flattened_extra_info_dict_from_subsequence_batch(batch):
        return dict()

    @staticmethod
    def get_last_extra_info_dict_from_subsequence_batch(batch):
        return dict()

    def log_diagnostics(self, paths, **kwargs):
        list_of_rewards, terminals, obs, actions, next_obs = split_paths(paths)

        returns = []
        for rewards in list_of_rewards:
            returns.append(np.sum(rewards))
        last_statistics = OrderedDict()
        last_statistics.update(create_stats_ordered_dict(
            'UndiscountedReturns',
            returns,
        ))
        last_statistics.update(create_stats_ordered_dict(
            'Actions',
            actions,
        ))

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)
        return returns

    def is_current_done(self):
        return False
