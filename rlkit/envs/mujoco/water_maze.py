from collections import deque, OrderedDict

import numpy as np

from rlkit.envs.mujoco.mujoco_env import MujocoEnv
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.util import split_paths
from rlkit.core import logger
from sandbox.rocky.tf.spaces.box import Box


class WaterMaze(MujocoEnv):
    def __init__(
            self,
            horizon=200,
            l2_action_penalty_weight=1e-2,
            num_steps=None,
            include_velocity=False,
            use_small_maze=False,
            num_steps_until_reset=5,
    ):
        self.init_serialization(locals())
        if use_small_maze:
            self.TARGET_RADIUS = 0.04
            self.BOUNDARY_RADIUS = 0.02
            self.BOUNDARY_DIST = 0.12
            self.BALL_RADIUS = 0.01
            super().__init__('small_water_maze.xml')
        else:
            self.TARGET_RADIUS = 0.1
            self.BOUNDARY_RADIUS = 0.02
            self.BOUNDARY_DIST = 0.3
            self.BALL_RADIUS = 0.02
            super().__init__('water_maze.xml')
        self.BALL_START_DIST = (
            self.BOUNDARY_DIST - self.BOUNDARY_RADIUS - 2 * self.BALL_RADIUS
        )
        self.MAX_GOAL_DIST = self.BOUNDARY_DIST - self.BOUNDARY_RADIUS
        self.l2_action_penalty_weight = l2_action_penalty_weight
        if num_steps is not None:  # support backwards compatibility
            horizon = num_steps

        self._horizon = horizon
        self._t = 0
        self._on_platform_history = deque(maxlen=5)
        self.num_steps_until_reset = num_steps_until_reset
        self.teleport_after_a_while = self.num_steps_until_reset > 0
        if self.teleport_after_a_while:
            for _ in range(self.num_steps_until_reset):
                self._on_platform_history.append(False)
        self.include_velocity = include_velocity

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = self._create_observation_space()
        self.reset_model()

    @property
    def horizon(self):
        return self._horizon

    def _create_observation_space(self):
        num_obs = 4 if self.include_velocity else 2
        return Box(
            np.hstack((-np.inf + np.zeros(num_obs), [0])),
            np.hstack((np.inf + np.zeros(num_obs), [1])),
        )

    def _step(self, force_actions):
        self._t += 1
        mujoco_action = np.hstack([force_actions, [0, 0]])
        self.do_simulation(mujoco_action, self.frame_skip)
        observation, on_platform = self._get_observation_and_on_platform()

        self._on_platform_history.append(on_platform)

        if self.teleport_after_a_while and all(self._on_platform_history):
            self.reset_ball_position()

        reward = (
            on_platform
            - self.l2_action_penalty_weight * np.linalg.norm(force_actions)
        )
        done = self._t >= self.horizon
        info = {
            'radius': self.TARGET_RADIUS,
            'target_position': self._get_target_position(),
        }
        return observation, reward, done, info

    def reset_ball_position(self):
        new_ball_position = self.np_random.uniform(
            size=2, low=-self.BALL_START_DIST, high=self.BALL_START_DIST
        )
        target_position = self._get_target_position()
        qvel = np.zeros(self.model.nv)
        new_pos = np.hstack((new_ball_position, target_position))
        self.set_state(new_pos, qvel)

    def reset_model(self):
        qpos = self.np_random.uniform(size=self.model.nq,
                                      low=-self.MAX_GOAL_DIST,
                                      high=self.MAX_GOAL_DIST)
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        self.reset_ball_position()
        self._t = 0
        return self._get_observation_and_on_platform()[0]

    def _get_observation_and_on_platform(self):
        """
        :return: Tuple
        - Observation vector
        - Flag: on platform or not.
        """
        position = np.concatenate([self.model.data.qpos]).ravel()[:2]
        target_position = self._get_target_position()
        dist = np.linalg.norm(position - target_position)
        on_platform = dist <= self.TARGET_RADIUS
        if self.include_velocity:
            velocity = np.concatenate([self.model.data.qvel]).ravel()[:2]
            return np.hstack((position, velocity, [on_platform])), on_platform
        else:
            return np.hstack((position, [on_platform])), on_platform

    def _get_target_position(self):
        return np.concatenate([self.model.data.qpos]).ravel()[2:]

    def viewer_setup(self):
        pass


    def get_tf_loss(self, observations, actions, target_labels, **kwargs):
        """
        Return the supervised-learning loss.
        :param observation: Tensor
        :param action: Tensor
        :return: loss Tensor
        """
        return -(actions + observations - target_labels) ** 2

    def get_param_values(self):
        return None

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
            'Rewards',
            list_of_rewards,
        ))
        last_statistics.update(create_stats_ordered_dict(
            'Actions',
            actions,
        ))

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)
        return returns

    def terminate(self):
        self.close()

    @staticmethod
    def get_extra_info_dict_from_batch(batch):
        return {}

    @staticmethod
    def get_flattened_extra_info_dict_from_subsequence_batch(batch):
        return {}

    @staticmethod
    def get_last_extra_info_dict_from_subsequence_batch(batch):
        return {}


class WaterMazeEasy(WaterMaze):
    """
    Always see the target position.
    """

    def _create_observation_space(self):
        obs_space = super()._create_observation_space()
        return Box(
            np.hstack((
                obs_space.low,
                [-self.BOUNDARY_DIST, -self.BOUNDARY_DIST]
            )),
            np.hstack((
                obs_space.high,
                [self.BOUNDARY_DIST, self.BOUNDARY_DIST]
            )),
        )

    def _get_observation_and_on_platform(self):
        obs, on_platform = super()._get_observation_and_on_platform()
        target_position = self._get_target_position()
        return np.hstack((obs, target_position)), on_platform


ZERO_TARGET_POSITION = np.zeros(2)


class WaterMazeMemory(WaterMaze):
    """
    See the target position at the very first time step.
    """
    def _create_observation_space(self):
        obs_space = super()._create_observation_space()
        return Box(
            np.hstack((
                obs_space.low,
                [-self.BOUNDARY_DIST, -self.BOUNDARY_DIST]
            )),
            np.hstack((
                obs_space.high,
                [self.BOUNDARY_DIST, self.BOUNDARY_DIST]
            )),
        )

    def _get_observation_and_on_platform(self):
        obs, on_platform = super()._get_observation_and_on_platform()
        if self._t == 0:
            target_position = self._get_target_position()
        else:
            target_position = ZERO_TARGET_POSITION
        return np.hstack((obs, target_position)), on_platform
