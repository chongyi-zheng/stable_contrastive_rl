from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import PusherEnv as GymPusherEnv

from rlkit.core.eval_util import create_stats_ordered_dict, get_stat_in_paths
from rlkit.core import logger


class PusherEnv(GymPusherEnv):
    def __init__(self):
        self.goal_cylinder_relative_x = 0
        self.goal_cylinder_relative_y = 0
        super().__init__()

    def reset_model(self):
        qpos = self.init_qpos
        goal_xy = np.array([
            self.goal_cylinder_relative_x,
            self.goal_cylinder_relative_y,
        ])

        while True:
            self.cylinder_pos = np.concatenate([
                self.np_random.uniform(low=-0.3, high=0, size=1),
                self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - goal_xy) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        # y-axis comes first in the xml
        qpos[-2] = self.goal_cylinder_relative_y
        qpos[-1] = self.goal_cylinder_relative_x
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005,
                                                       size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _step(self, a):
        arm_to_object = (
            self.get_body_com("tips_arm") - self.get_body_com("object")
        )
        object_to_goal = (
            self.get_body_com("object") - self.get_body_com("goal")
        )
        arm_to_goal = (
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        obs, reward, done, info_dict = super()._step(a)
        info_dict['arm to object distance'] = np.linalg.norm(arm_to_object)
        info_dict['object to goal distance'] = np.linalg.norm(object_to_goal)
        info_dict['arm to goal distance'] = np.linalg.norm(arm_to_goal)
        return obs, reward, done, info_dict

    def log_diagnostics(self, paths):
        statistics = OrderedDict()

        for stat_name in [
            'arm to object distance',
            'object to goal distance',
            'arm to goal distance',
        ]:
            stat = get_stat_in_paths(
                paths, 'env_infos', stat_name
            )
            statistics.update(create_stats_ordered_dict(
                stat_name, stat
            ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def _set_goal_xy(self, xy):
        # Based on XML
        self.goal_cylinder_relative_x = xy[0] - 0.45
        self.goal_cylinder_relative_y = xy[1] + 0.05
