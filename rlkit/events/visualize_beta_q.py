import argparse
import numpy as np
import matplotlib.pyplot as plt

import joblib

from rlkit.util.visualization_util import make_heat_map, plot_heatmap
from rlkit.policies.simple import RandomPolicy
from rlkit.state_distance.rollout_util import multitask_rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--mt', type=int, help='max time to goal', default=0)
    args = parser.parse_args()
    if args.pause:
        import ipdb; ipdb.set_trace()

    file = args.file
    data = joblib.load(file)
    beta_q = data['beta_q']
    env = data['env']

    random_policy = RandomPolicy(env.action_space)
    path = multitask_rollout(
        env,
        random_policy,
        init_tau=0,
        max_path_length=2,
        # animated=True,
    )
    path_obs = path['observations']
    path_actions = path['actions']
    path_next_obs = path['next_observations']
    num_steps_left = np.array([[args.mt]])

    obs = path_obs[0:1]
    next_obs = path_next_obs[0:1]
    actions = path_actions[0:1]
    beta_values = []

    def create_beta_eval(obs, goal):
        def beta_eval(a1, a2):
            actions = np.array([[a1, a2]])
            return beta_q.eval_np(
                observations=np.array([[
                    *obs
                ]]),
                actions=actions,
                goals=np.array([[
                    *goal
                ]]),
                num_steps_left=num_steps_left
            )[0, 0]
        return beta_eval

    def create_goal_eval(action, pos):
        def goal_eval(x1, x2):
            actions = np.array([[*action]])
            return beta_q.eval_np(
                observations=np.array([[
                    *pos
                ]]),
                actions=actions,
                goals=np.array([[
                    x1, x2
                ]]),
                num_steps_left=num_steps_left
            )[0, 0]
        return goal_eval

    print("obs:", obs)
    print("true action:", actions)
    print("next obs:", next_obs)

    obs = (4, 4)
    goal = (3, 4)
    plt.title("pos {}. goal {}".format(obs, goal))
    heatmap = make_heat_map(create_beta_eval(obs, goal),
                            [-1, 1], [-1, 1], resolution=50)
    plot_heatmap(heatmap)

    obs = (0, 0)
    goal = (1, 0)
    plt.figure()
    plt.title("pos {}. goal {}".format(obs, goal))
    heatmap = make_heat_map(create_beta_eval(obs, goal),
                            [-1, 1], [-1, 1], resolution=50)
    plot_heatmap(heatmap)

    obs = (1, 1)
    goal = (0, 4)
    plt.figure()
    plt.title("pos {}. goal {}".format(obs, goal))
    heatmap = make_heat_map(create_beta_eval(obs, goal),
                            [-1, 1], [-1, 1], resolution=50)
    plot_heatmap(heatmap)

    obs = (0, 2)
    goal = (0, 4)
    plt.figure()
    plt.title("pos {}. goal {}".format(obs, goal))
    heatmap = make_heat_map(create_beta_eval(obs, goal),
                            [-1, 1], [-1, 1], resolution=50)
    plot_heatmap(heatmap)

    obs = (0, 0.75)
    goal = (0, 1.75)
    plt.figure()
    plt.title("pos {}. goal {}".format(obs, goal))
    heatmap = make_heat_map(create_beta_eval(obs, goal),
                            [-1, 1], [-1, 1], resolution=50)
    plot_heatmap(heatmap)

    if True:
        pos = (0, 0.75)

        action = (0, 1)
        plt.figure()
        plt.title("pos = {}. action = {}".format(pos, action))
        action_eval = create_goal_eval(action, pos)
        heatmap = make_heat_map(action_eval, [-4, 4], [-4, 4], resolution=50)
        plot_heatmap(heatmap)

        pos = (0, 1.75)
        action = (0, 1)
        plt.figure()
        plt.title("pos = {}. action = {}".format(pos, action))
        action_eval = create_goal_eval(action, pos)
        heatmap = make_heat_map(action_eval, [-4, 4], [-4, 4], resolution=50)
        plot_heatmap(heatmap)

    if False:
        pos = (4, 4)

        action = (1, 0)
        plt.figure()
        plt.title("pos = {}. action = {}".format(pos, action))
        action_eval = create_goal_eval(action, pos)
        heatmap = make_heat_map(action_eval, [-4, 4], [-4, 4], resolution=50)
        plot_heatmap(heatmap)

        action = (-1, 0)
        plt.figure()
        plt.title("pos = {}. action = {}".format(pos, action))
        action_eval = create_goal_eval(action, pos)
        heatmap = make_heat_map(action_eval, [-4, 4], [-4, 4], resolution=50)
        plot_heatmap(heatmap)

        action = (0, 1)
        plt.figure()
        plt.title("pos = {}. action = {}".format(pos, action))
        action_eval = create_goal_eval(action, pos)
        heatmap = make_heat_map(action_eval, [-4, 4], [-4, 4], resolution=50)
        plot_heatmap(heatmap)

        action = (0, -1)
        plt.figure()
        plt.title("pos = {}. action = {}".format(pos, action))
        action_eval = create_goal_eval(action, pos)
        heatmap = make_heat_map(action_eval, [-4, 4], [-4, 4], resolution=50)
        plot_heatmap(heatmap)

    plt.show()
