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
    beta_v = data['beta_v']
    env = data['env']

    num_steps_left = np.array([[args.mt]])

    def create_beta_goal(obs):
        def beta_eval(g1, g2):
            return beta_v.eval_np(
                observations=np.array([[
                    *obs
                ]]),
                goals=np.array([[
                    g1, g2
                ]]),
                num_steps_left=num_steps_left
            )[0, 0]
        return beta_eval

    def create_beta_pos(goal):
        def beta_eval(x, y):
            return beta_v.eval_np(
                observations=np.array([[
                    x, y
                ]]),
                goals=np.array([[
                    *goal
                ]]),
                num_steps_left=num_steps_left
            )[0, 0]
        return beta_eval

    rng = [-4, 4]
    resolution = 30

    obs = (0, 0)
    plt.title("pos {}".format(obs))
    heatmap = make_heat_map(create_beta_goal(obs), rng, rng, resolution=resolution)
    plot_heatmap(heatmap)

    obs = (0, 0.75)
    plt.figure()
    plt.title("pos {}".format(obs))
    heatmap = make_heat_map(create_beta_goal(obs), rng, rng, resolution=resolution)
    plot_heatmap(heatmap)

    obs = (0, 1.75)
    plt.figure()
    plt.title("pos {}".format(obs))
    heatmap = make_heat_map(create_beta_goal(obs), rng, rng, resolution=resolution)
    plot_heatmap(heatmap)

    goal = (0, 1.25)
    plt.figure()
    plt.title("goal {}".format(goal))
    heatmap = make_heat_map(create_beta_pos(goal), rng, rng,
                            resolution=resolution)
    plot_heatmap(heatmap)

    goal = (0, 4)
    plt.figure()
    plt.title("goal {}".format(goal))
    heatmap = make_heat_map(create_beta_pos(goal), rng, rng,
                            resolution=resolution)
    plot_heatmap(heatmap)

    goal = (-4, -4)
    plt.figure()
    plt.title("goal {}".format(goal))
    heatmap = make_heat_map(create_beta_pos(goal), rng, rng,
                            resolution=resolution)
    plot_heatmap(heatmap)

    goal = (4, -4)
    plt.figure()
    plt.title("goal {}".format(goal))
    heatmap = make_heat_map(create_beta_pos(goal), rng, rng,
                            resolution=resolution)
    plot_heatmap(heatmap)

    goal = (4, 4)
    plt.figure()
    plt.title("goal {}".format(goal))
    heatmap = make_heat_map(create_beta_pos(goal), rng, rng,
                            resolution=resolution)
    plot_heatmap(heatmap)
    goal = (-4, 4)
    plt.figure()
    plt.title("goal {}".format(goal))
    heatmap = make_heat_map(create_beta_pos(goal), rng, rng,
                            resolution=resolution)
    plot_heatmap(heatmap)

    plt.show()
