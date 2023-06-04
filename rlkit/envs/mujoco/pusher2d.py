import abc
from collections import OrderedDict

import numpy as np

from gym.envs.mujoco import MujocoEnv

from rlkit.envs.env_utils import get_asset_full_path
from rlkit.core.eval_util import create_stats_ordered_dict, get_stat_in_paths
from rlkit.core import logger as default_logger


class Pusher2DEnv(MujocoEnv, metaclass=abc.ABCMeta):
    FILE = '3link_gripper_push_2d.xml'

    def __init__(self, goal=(-1, 0), randomize_goals=False,
                 use_hand_to_obj_reward=True,
                 use_sparse_rewards=False,
                 use_big_red_puck=False, **kwargs):
        if not isinstance(goal, np.ndarray):
            goal = np.array(goal)
        self._target_cylinder_position = goal
        self._target_hand_position = goal
        self.randomize_goals = randomize_goals
        self.use_hand_to_obj_reward = use_hand_to_obj_reward
        self.use_sparse_rewards = use_sparse_rewards
        self.use_big_red_puck = use_big_red_puck

        filename = "3link_gripper_push_2d_bigredpuck.xml" if use_big_red_puck else "3link_gripper_push_2d.xml"
        super().__init__(
            get_asset_full_path(filename),
            frame_skip=5,
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        hand_to_object_distance = np.linalg.norm(
            self.get_body_com("distal_4")[:2] - self.get_body_com("object")[:2]
        )
        object_to_goal_distance = np.linalg.norm(
            self.get_body_com("goal")[:2] - self.get_body_com("object")[:2]
        )
        hand_to_hand_goal_distance = np.linalg.norm(
            self.get_body_com("distal_4")[:2]
            - self.get_body_com("hand_goal")[:2]
        )
        success = float(object_to_goal_distance < 0.1)
        if self.use_sparse_rewards:
            reward = success
        else:
            reward = - object_to_goal_distance
            if self.use_hand_to_obj_reward:
                reward = reward - hand_to_object_distance
        done = False
        return ob, reward, done, dict(
            hand_to_hand_goal_distance=hand_to_hand_goal_distance,
            hand_to_object_distance=hand_to_object_distance,
            object_to_goal_distance=object_to_goal_distance,
            success=success,
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0
        rotation_angle = 90
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset_model(self):
        qpos = (
            np.random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos.squeeze()
        )
        qpos[-3:] = self.init_qpos.squeeze()[-3:]
        # Object position
        obj_pos = np.random.uniform(
            #         x      y
            np.array([0.3, -0.8]),
            np.array([0.8, -0.3]),
        )
        qpos[-6:-4] = obj_pos
        if self.randomize_goals:
            self._target_cylinder_position = np.random.uniform(
                np.array([-1, -1]),
                np.array([0, 0]),
                2
            )
        self._target_hand_position = self._target_cylinder_position
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        qvel = self.init_qvel.copy().squeeze()
        qvel[:] = 0

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:3],
            self.sim.data.qvel.flat[:3],
            self.get_body_com("distal_4")[:2],
            self.get_body_com("object")[:2],
        ])

    def log_diagnostics(self, paths, logger=default_logger):
        statistics = OrderedDict()
        for stat_name_in_paths, stat_name_to_print in [
            ('hand_to_object_distance', 'Distance hand to object'),
            ('object_to_goal_distance', 'Distance object to goal'),
            ('hand_to_hand_goal_distance', 'Distance hand to hand goal'),
            ('success', 'Success (within 0.1)'),
        ]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_to_print,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_to_print,
                final_stats,
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)


class RandomGoalPusher2DEnv(Pusher2DEnv):
    def __init__(self, goal=(-1, 0)):
        self.init_serialization(locals())
        super().__init__(goal, randomize_goals=True)
