from collections import OrderedDict

import numpy as np
from gym import Env
from gym.spaces import Box
from pygame import Color

from rlkit.core import logger as default_logger
from rlkit.envs.pygame.pygame_viewer import PygameViewer
from rlkit.core.eval_util import create_stats_ordered_dict, get_path_lengths, \
    get_stat_in_paths
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm


class Point2DEnv(Env):
    """
    A little 2D point whose life goal is to reach a target.
    """
    TARGET_RADIUS = 0.5
    BOUNDARY_DIST = 5
    BALL_RADIUS = 0.25

    def __init__(
            self,
            render_dt_msec=0,
            action_l2norm_penalty=0,
            render_onscreen=True,
            render_size=500,
            ball_radius=0.25,
    ):
        self.MAX_TARGET_DISTANCE = self.BOUNDARY_DIST - self.TARGET_RADIUS

        self.action_l2norm_penalty = action_l2norm_penalty
        self._target_position = None
        self._position = None

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = Box(
            -self.BOUNDARY_DIST * np.ones(4),
            self.BOUNDARY_DIST * np.ones(4)
            #dtype=np.float32
        )

        self.drawer = None
        self.render_dt_msec = render_dt_msec
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.BALL_RADIUS = ball_radius

    def _step(self, velocities):
        velocities = np.clip(velocities, a_min=-1, a_max=1)
        # Avoid += to avoid aliasing bugs
        self._position = self._position + velocities
        self._position = np.clip(
            self._position,
            a_min=-self.BOUNDARY_DIST,
            a_max=self.BOUNDARY_DIST,
        )
        distance_to_target = np.linalg.norm(
            self._target_position - self._position
        )
        observation = self._get_observation()
        on_platform = self.is_on_platform()

        distance_reward = -distance_to_target
        action_reward = -np.linalg.norm(velocities) * self.action_l2norm_penalty
        reward = self._reward()
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

    def _reward(self):
        distance_to_target = np.linalg.norm(
            self._target_position - self._position
        )
        distance_reward = -distance_to_target
        reward = distance_reward
        return reward

    def is_on_platform(self):
        dist = np.linalg.norm(self._position - self._target_position)
        return dist <= self.TARGET_RADIUS

    def reset(self):
        self._target_position = np.random.uniform(
            size=2, low=-self.MAX_TARGET_DISTANCE, high=self.MAX_TARGET_DISTANCE
        )
        self._position = np.random.uniform(
            size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
        )
        while self.is_on_platform():
            self._target_position = np.random.uniform(
                size=2, low=-self.MAX_TARGET_DISTANCE,
                high=self.MAX_TARGET_DISTANCE
            )
            self._position = np.random.uniform(
                size=2, low=-self.BOUNDARY_DIST, high=self.BOUNDARY_DIST
            )
        return self._get_observation()

    def _get_observation(self):
        return np.hstack((self._position, self._target_position))

    def log_diagnostics(self, paths, logger=default_logger):
        statistics = OrderedDict()
        for name_in_env_infos, name_to_log in [
            ('distance_to_target', 'Distance to Target'),
            ('speed', 'Speed'),
            ('distance_reward', 'Distance Reward'),
            ('action_reward', 'Action Reward'),
        ]:
            stats = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                name_to_log,
                stats,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + name_to_log,
                final_stats,
                always_show_all_stats=True,
            ))
        statistics.update(create_stats_ordered_dict(
            "Path Lengths",
            get_path_lengths(paths),
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def set_position(self, pos):
        self._position[0] = pos[0]
        self._position[1] = pos[1]

    def render(self, close=False, debug_info=None):
        if close:
            self.drawer = None
            return

        if self.drawer is None or self.drawer.terminated:
            self.drawer = PygameViewer(
                self.render_size,
                self.render_size,
                x_bounds=(-self.BOUNDARY_DIST, self.BOUNDARY_DIST),
                y_bounds=(-self.BOUNDARY_DIST, self.BOUNDARY_DIST),
                render_onscreen=self.render_onscreen,
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

        if debug_info is not None:
            debug_subgoals = debug_info.get('subgoal_seq', None)
            if debug_subgoals is not None:
                plasma_cm = plt.get_cmap('plasma')
                num_goals = len(debug_subgoals)
                for i, subgoal in enumerate(debug_subgoals):
                    color = plasma_cm(float(i) / num_goals)
                    # RGBA, but RGB need to be ints
                    color = Color(
                        int(color[0] * 255),
                        int(color[1] * 255),
                        int(color[2] * 255),
                        int(color[3] * 255),
                    )
                    self.drawer.draw_solid_circle(
                        subgoal,
                        self.BALL_RADIUS/2,
                        color,
                        )
            best_action = debug_info.get('oracle_qmax_action', None)
            if best_action is not None:
                self.drawer.draw_segment(self._position, self._position +
                                         best_action, Color('red'))
            policy_action = debug_info.get('learned_action', None)
            if policy_action is not None:
                self.drawer.draw_segment(self._position, self._position +
                                         policy_action, Color('green'))

        self.drawer.render()
        self.drawer.tick(self.render_dt_msec)

    def get_image(self):
        self.render()
        img = self.drawer.get_image()
        img = img / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        # img = (-r - g + b).flatten() # GREEN ignored for visualization
        img = (-r + b).flatten().copy()
        return img

    @staticmethod
    def true_model(state, action):
        velocities = np.clip(action, a_min=-1, a_max=1)
        position = state
        new_position = position + velocities
        return np.clip(
            new_position,
            a_min=-Point2DEnv.BOUNDARY_DIST,
            a_max=Point2DEnv.BOUNDARY_DIST,
        )


    @staticmethod
    def true_states(state, actions):
        real_states = [state]
        for action in actions:
            next_state = Point2DEnv.true_model(state, action)
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
                -Point2DEnv.BOUNDARY_DIST,
                -Point2DEnv.BOUNDARY_DIST,
            ],
            [
                Point2DEnv.BOUNDARY_DIST,
                -Point2DEnv.BOUNDARY_DIST,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.BOUNDARY_DIST,
                -Point2DEnv.BOUNDARY_DIST,
            ],
            [
                Point2DEnv.BOUNDARY_DIST,
                Point2DEnv.BOUNDARY_DIST,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.BOUNDARY_DIST,
                Point2DEnv.BOUNDARY_DIST,
            ],
            [
                Point2DEnv.BOUNDARY_DIST,
                -Point2DEnv.BOUNDARY_DIST,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.BOUNDARY_DIST,
                -Point2DEnv.BOUNDARY_DIST,
            ],
            [
                -Point2DEnv.BOUNDARY_DIST,
                -Point2DEnv.BOUNDARY_DIST,
            ],
            color='k', linestyle='-',
        )

        if goal is not None:
            ax.plot(goal[0], -goal[1], marker='*', color='g', markersize=15)
        ax.set_ylim(
            -Point2DEnv.BOUNDARY_DIST-1,
            Point2DEnv.BOUNDARY_DIST+1,
        )
        ax.set_xlim(
            -Point2DEnv.BOUNDARY_DIST-1,
            Point2DEnv.BOUNDARY_DIST+1,
        )


def plot_observations_and_actions(observations, actions):
    import matplotlib.pyplot as plt
    import rlkit.util.visualization_util as vu

    x_pos = observations[:, 0]
    y_pos = observations[:, 1]
    H, xedges, yedges = np.histogram2d(x_pos, y_pos)
    heatmap = vu.HeatMap(H, xedges, yedges, {})
    plt.subplot(2, 1, 1)
    plt.title("Observation Distribution")
    plt.xlabel("0-Dimenion")
    plt.ylabel("1-Dimenion")
    vu.plot_heatmap(heatmap)

    x_actions = actions[:, 0]
    y_actions = actions[:, 1]
    H, xedges, yedges = np.histogram2d(x_actions, y_actions)
    heatmap = vu.HeatMap(H, xedges, yedges, {})
    plt.subplot(2, 1, 2)
    plt.title("Action Distribution")
    plt.xlabel("0-Dimenion")
    plt.ylabel("1-Dimenion")
    vu.plot_heatmap(heatmap)

    plt.show()


if __name__ == "__main__":
    import argparse
    import joblib
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=30, help='Horizon for eval')
    args = parser.parse_args()

    data = joblib.load(args.file)
    replay_buffer = data['replay_buffer']
    max_i = replay_buffer._top - 1
    observations = replay_buffer._observations[:max_i, :]
    actions = replay_buffer._actions[:max_i, :]
    plot_observations_and_actions(observations, actions)
