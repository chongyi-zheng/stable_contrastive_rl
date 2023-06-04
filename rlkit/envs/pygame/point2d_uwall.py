from collections import OrderedDict

import numpy as np
from gym import Env
from gym.spaces import Box
from pygame import Color

from rlkit.core import logger as default_logger
from rlkit.envs.pygame.pygame_viewer import PygameViewer
from rlkit.envs.pygame.walls import VerticalWall, HorizontalWall
from rlkit.core.eval_util import create_stats_ordered_dict, get_path_lengths, \
    get_stat_in_paths


class Point2dUWall( Env):
    """
    A little 2D point whose life goal is to reach a target...but there's this
    darn U-shaped wall that punishes short-sighted behavior.
     _________
    |         |
    |         |
    |  | o |  |   o = start
    |  |   |  |   x = goal
    |  |___|  |
    |         |
    |    x    |
    |_________|
    """
    TARGET_RADIUS = 0.5
    OUTER_WALL_MAX_DIST = 4
    INNER_WALL_MAX_DIST = 1
    BALL_RADIUS = 0.25
    INIT_MAX_DISTANCE = INNER_WALL_MAX_DIST - BALL_RADIUS
    WALLS = [
        # Right wall
        VerticalWall(
            BALL_RADIUS,
            INNER_WALL_MAX_DIST,
            -INNER_WALL_MAX_DIST,
            INNER_WALL_MAX_DIST,
        ),
        # Left wall
        VerticalWall(
            BALL_RADIUS,
            -INNER_WALL_MAX_DIST,
            -INNER_WALL_MAX_DIST,
            INNER_WALL_MAX_DIST,
        ),
        # Bottom wall
        HorizontalWall(
            BALL_RADIUS,
            INNER_WALL_MAX_DIST,
            -INNER_WALL_MAX_DIST,
            INNER_WALL_MAX_DIST,
        )
    ]

    def __init__(
            self,
            render_dt_msec=30,
            action_l2norm_penalty=0,
    ):
        self.action_l2norm_penalty = action_l2norm_penalty
        self._target_position = np.array([
            0,
            # (self.OUTER_WALL_MAX_DIST + self.INNER_WALL_MAX_DIST) / 2
            self.OUTER_WALL_MAX_DIST,
        ])
        self._position = None

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = Box(
            -self.OUTER_WALL_MAX_DIST * np.ones(2),
            self.OUTER_WALL_MAX_DIST * np.ones(2),
        )

        self.drawer = None
        self.render_dt_msec = render_dt_msec

    def _step(self, velocities):
        velocities = np.clip(velocities, a_min=-1, a_max=1)
        new_position = self._position + velocities
        for wall in self.WALLS:
            new_position = wall.handle_collision(
                self._position, new_position
            )
        self._position = new_position
        self._position = np.clip(
            self._position,
            a_min=-self.OUTER_WALL_MAX_DIST,
            a_max=self.OUTER_WALL_MAX_DIST,
        )
        observation = self._get_observation()
        on_platform = self.is_on_platform()

        distance_to_target = np.linalg.norm(
            self._target_position - self._position
        )
        reward = float(on_platform)
        distance_reward = -distance_to_target
        action_reward = -np.linalg.norm(velocities) * self.action_l2norm_penalty
        reward += distance_reward + action_reward
        done = on_platform
        info = {
            'radius': self.TARGET_RADIUS,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
            'distance_reward': distance_reward,
            'action_reward': action_reward,
        }
        return observation, reward, done, info

    def is_on_platform(self):
        dist = np.linalg.norm(self._position - self._target_position)
        return dist <= self.TARGET_RADIUS + self.BALL_RADIUS

    def reset(self):
        self._position = np.random.uniform(
            size=2, low=-self.INIT_MAX_DISTANCE, high=self.INIT_MAX_DISTANCE
        )
        return self._get_observation()

    def _get_observation(self):
        return np.hstack(self._position)

    def log_diagnostics(self, paths, logger=default_logger):
        statistics = OrderedDict()
        for name_in_env_infos, name_to_log in [
            ('distance_to_target', 'Distance to Target'),
            ('speed', 'Speed'),
            ('distance_reward', 'Distance Reward'),
            ('action_reward', 'Action Reward'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                name_to_log,
                stat,
            ))
        statistics.update(create_stats_ordered_dict(
            "Path Lengths",
            get_path_lengths(paths),
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def render(self, mode='human', close=False):
        if close:
            self.drawer = None
            return

        if self.drawer is None or self.drawer.terminated:
            self.drawer = PygameViewer(
                500,
                500,
                x_bounds=(-self.OUTER_WALL_MAX_DIST, self.OUTER_WALL_MAX_DIST),
                y_bounds=(-self.OUTER_WALL_MAX_DIST, self.OUTER_WALL_MAX_DIST),
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

        # draw the walls
        for wall in self.WALLS:
            self.drawer.draw_segment(
                wall.endpoint1,
                wall.endpoint2,
                Color('black'),
            )

        self.drawer.render()
        self.drawer.tick(self.render_dt_msec)

    @staticmethod
    def true_model(state, action):
        velocities = np.clip(action, a_min=-1, a_max=1)
        position = state
        new_position = position + velocities
        for wall in Point2dUWall.WALLS:
            new_position = wall.handle_collision(
                position, new_position
            )
        return np.clip(
            new_position,
            a_min=-Point2dUWall.OUTER_WALL_MAX_DIST,
            a_max=Point2dUWall.OUTER_WALL_MAX_DIST,
        )


    @staticmethod
    def true_states(state, actions):
        real_states = [state]
        for action in actions:
            next_state = Point2dUWall.true_model(state, action)
            real_states.append(next_state)
            state = next_state
        return real_states


    @staticmethod
    def plot_trajectory(ax, states, actions, goal=None):
        assert len(states) == len(actions) + 1
        x = states[:, 0]
        y = -states[:, 1]
        num_states = len(states)
        plasma_cm = plt.get_cmap('plasma')
        for i, state in enumerate(states):
            color = plasma_cm(float(i) / num_states)
            ax.plot(state[0], -state[1],
                    marker='o', color=color, markersize=10,
                    )

        actions_x = actions[:, 0]
        actions_y = -actions[:, 1]

        ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
                  scale_units='xy', angles='xy', scale=1, width=0.005)
        ax.quiver(x[:-1], y[:-1], actions_x, actions_y, scale_units='xy',
                  angles='xy', scale=1, color='r',
                  width=0.0035, )

        ax.plot(
            [
                -Point2dUWall.INNER_WALL_MAX_DIST,
                -Point2dUWall.INNER_WALL_MAX_DIST
            ],
            [
                Point2dUWall.INNER_WALL_MAX_DIST,
                -Point2dUWall.INNER_WALL_MAX_DIST
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                -Point2dUWall.INNER_WALL_MAX_DIST,
                 Point2dUWall.INNER_WALL_MAX_DIST
            ],
            [
                -Point2dUWall.INNER_WALL_MAX_DIST,
                -Point2dUWall.INNER_WALL_MAX_DIST
            ],
            color='k', linestyle='-'
        )
        ax.plot(
            [
                Point2dUWall.INNER_WALL_MAX_DIST,
                Point2dUWall.INNER_WALL_MAX_DIST
            ],
            [
                Point2dUWall.INNER_WALL_MAX_DIST,
                -Point2dUWall.INNER_WALL_MAX_DIST
            ], color='k', linestyle='-'
        )
        # Outer Walls
        ax.plot(
            [
                -Point2dUWall.OUTER_WALL_MAX_DIST,
                -Point2dUWall.OUTER_WALL_MAX_DIST,
            ],
            [
                Point2dUWall.OUTER_WALL_MAX_DIST,
                -Point2dUWall.OUTER_WALL_MAX_DIST,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2dUWall.OUTER_WALL_MAX_DIST,
                -Point2dUWall.OUTER_WALL_MAX_DIST,
            ],
            [
                Point2dUWall.OUTER_WALL_MAX_DIST,
                Point2dUWall.OUTER_WALL_MAX_DIST,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2dUWall.OUTER_WALL_MAX_DIST,
                Point2dUWall.OUTER_WALL_MAX_DIST,
            ],
            [
                Point2dUWall.OUTER_WALL_MAX_DIST,
                -Point2dUWall.OUTER_WALL_MAX_DIST,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2dUWall.OUTER_WALL_MAX_DIST,
                -Point2dUWall.OUTER_WALL_MAX_DIST,
            ],
            [
                -Point2dUWall.OUTER_WALL_MAX_DIST,
                -Point2dUWall.OUTER_WALL_MAX_DIST,
            ],
            color='k', linestyle='-',
        )

        if goal is not None:
            ax.plot(goal[0], -goal[1], marker='*', color='g', markersize=15)
        ax.set_ylim(
            -Point2dUWall.OUTER_WALL_MAX_DIST-1,
            Point2dUWall.OUTER_WALL_MAX_DIST+1
        )
        ax.set_xlim(
            -Point2dUWall.OUTER_WALL_MAX_DIST-1,
            Point2dUWall.OUTER_WALL_MAX_DIST+1
        )
