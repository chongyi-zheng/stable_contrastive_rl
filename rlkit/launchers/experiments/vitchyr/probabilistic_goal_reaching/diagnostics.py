from collections import OrderedDict, defaultdict
from typing import List, Union

import numpy as np

from rlkit.envs.contextual.contextual_env import (
    ContextualDiagnosticsFn,
    Path,
    Context,
    Diagnostics,
)
from rlkit.launchers.experiments.vitchyr.probabilistic_goal_reaching.env import (
    NormalizeAntFullPositionGoalEnv
)
from rlkit.core.eval_util import create_stats_ordered_dict


class AntFullPositionGoalEnvDiagnostics(ContextualDiagnosticsFn):
    def __init__(
            self,
            desired_goal_key: str,
            achieved_goal_key: str,
            success_threshold,
            normalize_env: Union[None, NormalizeAntFullPositionGoalEnv] = None,
    ):
        self._desired_goal_key = desired_goal_key
        self._achieved_goal_key = achieved_goal_key
        self.success_threshold = success_threshold
        self.normalize_env = normalize_env
        if normalize_env:
            self.qpos_weights = normalize_env.qpos_weights
        else:
            self.qpos_weights = None

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
            xy_difference = difference[..., :2]
            orientation_difference = difference[..., 3:7]
            joint_difference = difference[..., 7:]
            if self.qpos_weights is not None:
                stat_to_lists['normalized/total/distance'].append(
                    np.linalg.norm(difference, axis=-1)
                )
                stat_to_lists['normalized/xy/distance'].append(
                    np.linalg.norm(xy_difference, axis=-1)
                )
                stat_to_lists['normalized/orientation/distance'].append(
                    np.linalg.norm(orientation_difference, axis=-1)
                )
                stat_to_lists['normalized/joint/distance'].append(
                    np.linalg.norm(joint_difference, axis=-1)
                )
                stat_to_lists['normalized/xy/success'].append(
                    np.linalg.norm(xy_difference, axis=-1)
                    <= self.success_threshold
                )
                stat_to_lists['normalized/orientation/success'].append(
                    np.linalg.norm(orientation_difference, axis=-1)
                    <= self.success_threshold
                )
                stat_to_lists['normalized/joint/success'].append(
                    np.linalg.norm(joint_difference, axis=-1)
                    <= self.success_threshold
                )
                difference = (achieved - goal) / self.qpos_weights
                xy_difference = difference[..., :2]
                orientation_difference = difference[..., 3:7]
                joint_difference = difference[..., 7:]
            stat_to_lists['total/distance'].append(
                np.linalg.norm(difference, axis=-1)
            )
            stat_to_lists['xy/distance'].append(
                np.linalg.norm(xy_difference, axis=-1)
            )
            stat_to_lists['orientation/distance'].append(
                np.linalg.norm(orientation_difference, axis=-1)
            )
            stat_to_lists['joint/distance'].append(
                np.linalg.norm(joint_difference, axis=-1)
            )
            stat_to_lists['xy/success'].append(
                np.linalg.norm(xy_difference, axis=-1)
                <= self.success_threshold
            )
            stat_to_lists['orientation/success'].append(
                np.linalg.norm(orientation_difference, axis=-1)
                <= self.success_threshold
            )
            stat_to_lists['joint/success'].append(
                np.linalg.norm(joint_difference, axis=-1)
                <= self.success_threshold
            )
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
        return statistics


class HopperFullPositionGoalEnvDiagnostics(ContextualDiagnosticsFn):
    def __init__(
            self,
            desired_goal_key: str,
            achieved_goal_key: str,
            success_threshold,
    ):
        self._desired_goal_key = desired_goal_key
        self._achieved_goal_key = achieved_goal_key
        self.success_threshold = success_threshold

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
            x_difference = difference[..., :1]
            y_difference = difference[..., 1:2]
            z_difference = difference[..., 2:3]
            joint_difference = difference[..., 3:6]
            stat_to_lists['x/distance'].append(
                np.linalg.norm(x_difference, axis=-1)
            )
            stat_to_lists['y/distance'].append(
                np.linalg.norm(y_difference, axis=-1)
            )
            stat_to_lists['z/distance'].append(
                np.linalg.norm(z_difference, axis=-1)
            )
            stat_to_lists['joint/distance'].append(
                np.linalg.norm(joint_difference, axis=-1)
            )
            stat_to_lists['x/success'].append(
                np.linalg.norm(x_difference, axis=-1)
                <= self.success_threshold
            )
            stat_to_lists['y/success'].append(
                np.linalg.norm(y_difference, axis=-1)
                <= self.success_threshold
            )
            stat_to_lists['z/success'].append(
                np.linalg.norm(z_difference, axis=-1)
                <= self.success_threshold
            )
            stat_to_lists['joint/success'].append(
                np.linalg.norm(joint_difference, axis=-1)
                <= self.success_threshold
            )
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
        return statistics


class SawyerPickAndPlaceEnvAchievedFromObs(object):
    def __init__(self, key):
        self._key = key

    def __call__(self, observations):
        return observations[self._key][..., 1:]
