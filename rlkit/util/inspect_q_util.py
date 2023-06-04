import matplotlib.pyplot as plt
import rlkit.visualization.visualization_util as vu
import numpy as np
from rlkit.torch.core import np_ify, torch_ify

fig_v_mean = None
fig_v_std = None
axes_v_mean = None
axes_v_std = None

def debug_q(ensemble_qs, policy, show_mean=True, show_std=True):
    global fig_v_mean, fig_v_std, axes_v_mean, axes_v_std

    if fig_v_mean is None:
        if show_mean:
            fig_v_mean, axes_v_mean = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(9, 9))
            fig_v_mean.canvas.set_window_title('Q Mean')
            # plt.suptitle("V Mean")
        if show_std:
            fig_v_std, axes_v_std = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(9, 9))
            fig_v_std.canvas.set_window_title('Q Std')
            # plt.suptitle("V Std")

    # obss = [(0, 0), (0, 0.75), (0, 1.75), (0, 1.25), (0, 4), (-4, 4), (4, -4), (4, 4), (-4, 4)]
    obss = []
    for x in [-3, 0, 3]:
        for y in [3, 0, -3]:
            obss.append((x, y))

    def create_eval_function(q, obs, ):
        def beta_eval(goals):
            # goals = np.array([[
            #     *goal
            # ]])
            N = len(goals)
            observations = np.tile(obs, (N, 1))
            new_obs = np.hstack((observations, goals))
            actions = torch_ify(policy.get_action(new_obs)[0])
            return np_ify(q(torch_ify(new_obs), actions)).flatten()
        return beta_eval

    for o in range(9):
        i = o % 3
        j = o // 3
        H = []
        for b in range(5):
            q = ensemble_qs[b]

            rng = [-4, 4]
            resolution = 30

            obs = obss[o]

            heatmap = vu.make_heat_map(create_eval_function(q, obs, ), rng, rng, resolution=resolution, batch=True)
            H.append(heatmap.values)
        p, x, y, _ = heatmap
        if show_mean:
            h1 = vu.HeatMap(np.mean(H, axis=0), x, y, _)
            vu.plot_heatmap(h1, ax=axes_v_mean[i, j])
            axes_v_mean[i, j].set_title("pos " + str(obs))

        if show_std:
            h2 = vu.HeatMap(np.std(H, axis=0), x, y, _)
            vu.plot_heatmap(h2, ax=axes_v_std[i, j])
            axes_v_std[i, j].set_title("pos " + str(obs))

        # axes_v_mean[i, j].plot(range(100), range(100))
        # axes_v_std[i, j].plot(range(100), range(100))

    plt.draw()
    plt.pause(0.01)
