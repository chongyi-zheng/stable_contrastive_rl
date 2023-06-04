from collections import OrderedDict, deque

import numpy as np
from gym import Env
from gym.spaces import Box
from pygame import Color

from rlkit.envs.pygame.pygame_viewer import PygameViewer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.util import split_paths
from rlkit.core import logger


class WaterMaze(Env):
    TARGET_RADIUS = 2
    BOUNDARY_DIST = 5
    BALL_RADIUS = 0.25

    def __init__(
            self,
            horizon=200,
            render_dt_msec=30,
            give_time=True,
            action_l2norm_penalty=0,
    ):
        self.MAX_TARGET_DISTANCE = self.BOUNDARY_DIST - self.TARGET_RADIUS

        self._horizon = horizon
        self._t = 0
        self._target_position = None
        self._position = None
        self.give_time = give_time
        self.action_l2norm_penalty = action_l2norm_penalty

        self._action_space = self._create_action_space()
        self._observation_space = self._create_observation_space()

        self.drawer = None
        self.render_dt_msec = render_dt_msec

    def _create_action_space(self):
        return Box(np.array([-1, -1]), np.array([1, 1]))

    def _create_observation_space(self):
        low = np.array([-self.BOUNDARY_DIST, -self.BOUNDARY_DIST, 0])
        high = np.array([self.BOUNDARY_DIST, self.BOUNDARY_DIST, 1])
        if self.give_time:
            low = np.hstack((low, [0]))
            high = np.hstack((high, [self.horizon]))
        return Box(low, high)

    def _step(self, velocities):
        self._t += 1
        velocities = np.clip(velocities, a_min=-1, a_max=1)
        distance_to_target = np.linalg.norm(
            self._target_position - self._position
        )
        self._position += velocities
        self._position = np.clip(
            self._position,
            a_min=-self.BOUNDARY_DIST,
            a_max=self.BOUNDARY_DIST,
        )
        observation, on_platform = self._get_observation_and_on_platform()

        reward = self.get_reward(velocities, on_platform)
        done = self._t >= self.horizon
        info = {
            'radius': self.TARGET_RADIUS,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
        }
        return observation, reward, done, info

    def get_reward(self, velocities, on_platform):
        reward = float(on_platform)
        reward -= np.linalg.norm(velocities) * self.action_l2norm_penalty
        return reward

    def _reset(self):
        self._target_position = np.random.uniform(
            size=2, low=-self.MAX_TARGET_DISTANCE, high=self.MAX_TARGET_DISTANCE
        )
        self._position = np.random.uniform(
            size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
        )
        self._t = 0
        return self._get_observation_and_on_platform()[0]

    def _get_observation_and_on_platform(self):
        """
        :return: Tuple
        - Observation vector
        - Flag: on platform or not.
        """
        dist = np.linalg.norm(self._position - self._target_position)
        on_platform = dist <= self.TARGET_RADIUS
        obs = np.hstack((self._position, [on_platform]))
        if self.give_time:
            obs = np.hstack((obs, [self._t]))
        return obs, on_platform

    def log_diagnostics(self, paths, **kwargs):
        list_of_rewards, terminals, obs, actions, next_obs = split_paths(paths)

        returns = []
        for rewards in list_of_rewards:
            returns.append(np.sum(rewards))
        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            'Undiscounted Returns',
            returns,
        ))
        statistics.update(create_stats_ordered_dict(
            'Rewards',
            list_of_rewards,
        ))
        statistics.update(create_stats_ordered_dict(
            'Actions',
            actions,
        ))

        fraction_of_time_on_platform = [
            o[2] for o in obs
        ]
        statistics['Fraction of time on platform'] = np.mean(fraction_of_time_on_platform)

        for key, value in statistics.items():
            logger.record_tabular(key, value)
        return returns

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def horizon(self):
        return self._horizon

    def render(self, mode='human', close=False):
        if close:
            self.drawer = None
            return

        if self.drawer is None or self.drawer.terminated:
            self.drawer = PygameViewer(
                500,
                500,
                x_bounds=(-self.BOUNDARY_DIST, self.BOUNDARY_DIST),
                y_bounds=(-self.BOUNDARY_DIST, self.BOUNDARY_DIST),
            )

        self.drawer.fill(Color('white'))
        self.drawer.draw_solid_circle(
            self._target_position,
            self.TARGET_RADIUS,
            Color('green'),
        )
        self.drawer.draw_solid_circle(
            self._position,
            self.BALL_RADIUS,
            Color('blue'),
        )

        self.drawer.render()
        self.drawer.tick(self.render_dt_msec)


class WaterMazeHard(WaterMaze):
    """
    WaterMaze + teleportation after some time steps of being on the platform.
    """
    def __init__(self, *args, num_steps_until_teleportation=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps_until_teleportation = num_steps_until_teleportation
        self.on_platform = deque(maxlen=self.num_steps_until_teleportation)
        self.on_platform.append(False)

    def _step(self, velocities):
        self._t += 1
        velocities = np.clip(velocities, a_min=-1, a_max=1)
        self._position += velocities
        self._position = np.clip(
            self._position,
            a_min=-self.BOUNDARY_DIST,
            a_max=self.BOUNDARY_DIST,
        )
        observation, on_platform = self._get_observation_and_on_platform()
        self.on_platform.append(on_platform)
        if all(self.on_platform):
            self._position = np.random.uniform(
                size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
            )
            observation, _ = self._get_observation_and_on_platform()
        reward = self.get_reward(velocities, on_platform)

        done = self._t >= self.horizon
        info = {
            'radius': self.TARGET_RADIUS,
            'target_position': self._target_position,
        }
        return observation, reward, done, info

    def _reset(self):
        self.on_platform.append(False)
        return super()._reset()


class WaterMazeEasy(WaterMaze):
    """
    See the target position at all time steps.
    """
    def _create_observation_space(self):
        box = super()._create_observation_space()
        low = np.hstack((box.low, [-self.BOUNDARY_DIST, -self.BOUNDARY_DIST]))
        high = np.hstack((box.high, [self.BOUNDARY_DIST, self.BOUNDARY_DIST]))
        return Box(low, high)

    def _get_hint(self):
        return self._target_position

    def _get_observation_and_on_platform(self):
        """
        :return: Tuple
        - Observation vector
        - Flag: on platform or not.
        """
        old_obs, on_platform = super()._get_observation_and_on_platform()
        hint = self._get_hint()
        obs = np.hstack((old_obs, hint))
        return obs, on_platform


class WaterMazeMemory(WaterMazeEasy):
    """
    See the target position at the very first time step.
    """
    def _get_hint(self):
        if self._t == 0:
            return self._target_position
        else:
            return np.zeros_like(self._target_position)


class WaterMaze1D(WaterMaze):
    def _create_action_space(self):
        return Box(np.array([-1]), np.array([1]))

    def _create_observation_space(self):
        low = np.array([-self.BOUNDARY_DIST, 0])
        high = np.array([self.BOUNDARY_DIST, 1])
        if self.give_time:
            low = np.hstack((low, [0]))
            high = np.hstack((high, [self.horizon]))
        return Box(low, high)

    def _step(self, velocity):
        velocities = np.hstack((velocity, 0))
        return super()._step(velocities)

    def _get_observation_and_on_platform(self):
        dist = np.linalg.norm(self._position - self._target_position)
        on_platform = dist <= self.TARGET_RADIUS
        obs = np.hstack((self._position[0], [on_platform]))
        if self.give_time:
            obs = np.hstack((obs, [self._t]))
        return obs, on_platform

    def _reset(self):
        self._target_position = np.random.uniform(
            size=2, low=-self.MAX_TARGET_DISTANCE, high=self.MAX_TARGET_DISTANCE
        )
        self._position = np.random.uniform(
            size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
        )
        self._target_position[1] = 0
        self._position[1] = 0
        self._t = 0
        return self._get_observation_and_on_platform()[0]

    def log_diagnostics(self, paths, **kwargs):
        list_of_rewards, terminals, obs, actions, next_obs = split_paths(paths)

        returns = []
        for rewards in list_of_rewards:
            returns.append(np.sum(rewards))
        statistics = OrderedDict()
        statistics.update(create_stats_ordered_dict(
            'Undiscounted Returns',
            returns,
        ))
        statistics.update(create_stats_ordered_dict(
            'Rewards',
            list_of_rewards,
        ))
        statistics.update(create_stats_ordered_dict(
            'Actions',
            actions,
        ))

        fraction_of_time_on_platform = [
            o[1] for o in obs
        ]
        statistics['Fraction of time on platform'] = np.mean(fraction_of_time_on_platform)

        for key, value in statistics.items():
            logger.record_tabular(key, value)
        return returns


class WaterMazeEasy1D(WaterMaze1D):
    """
    See the target position at all time steps.
    """
    def _create_observation_space(self):
        box = super()._create_observation_space()
        low = np.hstack((box.low, [-self.BOUNDARY_DIST]))
        high = np.hstack((box.high, [self.BOUNDARY_DIST]))
        return Box(low, high)

    def _get_hint(self):
        return self._target_position[0]

    def _get_observation_and_on_platform(self):
        """
        :return: Tuple
        - Observation vector
        - Flag: on platform or not.
        """
        old_obs, on_platform = super()._get_observation_and_on_platform()
        hint = self._get_hint()
        obs = np.hstack((old_obs, hint))
        return obs, on_platform


class WaterMazeMemory1D(WaterMazeEasy1D):
    """
    See the target position at the very first time step.
    """
    def _get_hint(self):
        if self._t == 0:
            return self._target_position[0]
        else:
            return 0
