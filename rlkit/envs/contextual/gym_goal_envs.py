from collections import OrderedDict, defaultdict
import logging
from typing import List

import numpy as np
from gym import GoalEnv

from rlkit.core.distribution import DictDistribution
from rlkit.envs.contextual.contextual_env import (
    ContextualDiagnosticsFn,
    Path,
    Context,
    Diagnostics,
)
from rlkit.core.eval_util import create_stats_ordered_dict


class GoalDictDistributionFromGymGoalEnv(DictDistribution):
    def __init__(
            self,
            env: GoalEnv,
            desired_goal_key='desired_goal'
    ):
        self._env = env
        self._desired_goal_key = desired_goal_key
        env_spaces = self._env.observation_space.spaces
        self._spaces = {
            desired_goal_key: env_spaces[desired_goal_key]
        }

    def sample(self, batch_size: int):
        def sample_goal():
            o = self._env.reset()
            return o[self._desired_goal_key]
        if batch_size > 1:
            logging.warning("""
            Sampling many goals in GoalDictDistributionFromGymGoalEnv is slow.
            Hopefully you're presampling goals.
            """)

        goals = [sample_goal() for _ in range(batch_size)]
        goals = np.array(goals)
        return {
            self._desired_goal_key: goals,
        }

    @property
    def spaces(self):
        return self._spaces


class GenericGoalConditionedContextualDiagnostics(ContextualDiagnosticsFn):
    def __init__(
            self,
            desired_goal_key: str,
            achieved_goal_key: str,
            success_threshold,
    ):
        self._desired_goal_key = desired_goal_key
        self._achieved_goal_key = achieved_goal_key
        self._success_threshold = success_threshold

    def __call__(self, paths: List[Path],
                 contexts: List[Context]) -> Diagnostics:
        goals = [c[self._desired_goal_key] for c in contexts]
        achieved_goals = [
            np.array([o[self._achieved_goal_key] for o in path['observations']])
            for path in paths
        ]

        statistics = OrderedDict()
        stat_to_lists = defaultdict(list)
        for achieved, goal in zip(achieved_goals, goals):
            difference = achieved - goal
            distance = np.linalg.norm(difference, axis=-1)
            stat_to_lists['distance'].append(distance)
            stat_to_lists['success'].append(distance <= self._success_threshold)
        for stat_name, stat_list in stat_to_lists.items():
            statistics.update(create_stats_ordered_dict(
                stat_name,
                stat_list,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                '{}/final'.format(stat_name),
                [s[-1:] for s in stat_list],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        statistics.update(create_stats_ordered_dict(
            '{}/initial'.format('distance'),
            [s[:1] for s in stat_to_lists['distance']],
            always_show_all_stats=True,
            exclude_max_min=True,
        ))
        statistics.update(create_stats_ordered_dict(
            '{}/any'.format('success'),
            [any(s) for s in stat_to_lists['success']],
            always_show_all_stats=True,
            exclude_max_min=True,
        ))
        return statistics
