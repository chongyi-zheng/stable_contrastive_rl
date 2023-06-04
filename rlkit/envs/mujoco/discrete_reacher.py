from collections import OrderedDict

import numpy as np
from rlkit.core import logger as default_logger
from gym import spaces
from gym.envs.mujoco import mujoco_env
import itertools

from rlkit.core.eval_util import get_stat_in_paths, create_stats_ordered_dict


class DiscreteReacherEnv(mujoco_env.MujocoEnv):
    def __init__(self, num_bins=7, frame_skip=2):
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', frame_skip=frame_skip)
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        joint_ranges = [np.linspace(low[i], high[i], num_bins) for i in
                        range(len(low))]
        self.idx_to_continuous_action = [
            np.array(x) for x in itertools.product(*joint_ranges)
        ]

        self.action_space = spaces.Discrete(len(self.idx_to_continuous_action))

    def _step(self, a):
        if not self.action_space or not self.action_space.contains(a):
            continuous_action = a
        else:
            continuous_action = self.idx_to_continuous_action[a]
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(continuous_action).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(continuous_action, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            distance_to_target=np.linalg.norm(vec),
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def log_diagnostics(self, paths, logger=default_logger):
        statistics = OrderedDict()
        for name_in_env_infos, name_to_log in [
            ('distance_to_target', 'Distance to Target'),
            ('reward_ctrl', 'Action Reward'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                name_to_log,
                stat,
            ))
        distances = get_stat_in_paths(paths, 'env_infos', 'distance_to_target')
        statistics.update(create_stats_ordered_dict(
            "Final Distance to Target",
            [ds[-1] for ds in distances],
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)
