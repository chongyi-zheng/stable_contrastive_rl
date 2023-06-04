from collections import OrderedDict
from os import path as osp

import numpy as np
from multiworld.core.image_env import ImageEnv
from rlkit.core import logger
from rlkit.core.logging import append_log
from rlkit.data_management.images import normalize_image
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.torch import pytorch_util as ptu
from rlkit.visualization.video import dump_video


def get_extra_imgs(path, index_in_path, env, img_keys):
    del env
    full_observation_dict = path[index_in_path]
    return [
        full_observation_dict[img_key]
        for img_key in img_keys
        if img_key in full_observation_dict.keys()
    ]


def get_save_video_function(
        rollout_function,
        env,
        policy,
        save_video_period=10,
        imsize=48,
        tag="",
        video_image_env_kwargs=None,
        **dump_video_kwargs
):
    logdir = logger.get_snapshot_dir()

    if not isinstance(env, ImageEnv) and not isinstance(env, VAEWrappedEnv):
        if video_image_env_kwargs is None:
            video_image_env_kwargs = {}
        image_env = ImageEnv(env, imsize, transpose=True, normalize=True,
                             **video_image_env_kwargs)
    else:
        image_env = env
        assert image_env.imsize == imsize, "Imsize must match env imsize"

    def save_video(algo, epoch):
        if epoch % save_video_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(
                logdir,
                'video_{}_{epoch}_env.mp4'.format(tag, epoch=epoch),
            )
            dump_video(image_env, policy, filename, rollout_function,
                       imsize=imsize, **dump_video_kwargs)
    return save_video


def plot_buffer_function(save_period, buffer_key):
    import matplotlib.pyplot as plt
    from rlkit.core import logger
    logdir = logger.get_snapshot_dir()

    def plot_buffer(algo, epoch):
        replay_buffer = algo.replay_buffer
        if epoch % save_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(logdir,
                                '{}_buffer_{epoch}_env.png'.format(
                                    buffer_key, epoch=epoch))
            goals = replay_buffer._next_obs[buffer_key][:replay_buffer._size]

            plt.clf()
            plt.scatter(goals[:, 0], goals[:, 1], alpha=0.2)
            plt.savefig(filename)
    return plot_buffer


def plot_encoder_function(variant, encoder, tag=""):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from rlkit.core import logger
    logdir = logger.get_snapshot_dir()

    def plot_encoder(algo, epoch, is_x=False):
        save_period = variant.get('save_video_period', 50)
        if epoch % save_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(logdir,
                                'encoder_{}_{}_{epoch}_env.gif'.format(
                                    tag,
                                    "x" if is_x else "y",
                                    epoch=epoch))

            vary = np.arange(-4, 4, .1)
            static = np.zeros(len(vary))

            points_x = np.c_[vary.reshape(-1, 1), static.reshape(-1, 1)]
            points_y = np.c_[static.reshape(-1, 1), vary.reshape(-1, 1)]

            encoded_points_x = ptu.get_numpy(encoder.forward(ptu.from_numpy(points_x)))
            encoded_points_y = ptu.get_numpy(encoder.forward(ptu.from_numpy(points_y)))

            plt.clf()
            fig = plt.figure()
            plt.xlim(min(min(encoded_points_x[:, 0]),
                         min(encoded_points_y[:, 0])),
                     max(max(encoded_points_x[:, 0]),
                         max(encoded_points_y[:, 0])))
            plt.ylim(min(min(encoded_points_x[:, 1]),
                         min(encoded_points_y[:, 1])),
                     max(max(encoded_points_x[:, 1]),
                         max(encoded_points_y[:, 1])))
            colors = ["red", "blue"]
            lines = [plt.plot([], [], 'o', color=colors[i], alpha=0.4)[0] for i in range(2)]

            def animate(i):
                lines[0].set_data(encoded_points_x[:i+1, 0], encoded_points_x[:i+1, 1])
                lines[1].set_data(encoded_points_y[:i+1, 0], encoded_points_y[:i+1, 1])
                return lines

            ani = FuncAnimation(fig, animate, frames=len(vary), interval=40)
            ani.save(filename, writer='imagemagick', fps=60)
    # def plot_encoder_x_and_y(algo, epoch):
        # plot_encoder(algo, epoch, is_x=True)
        # plot_encoder(algo, epoch, is_x=False)

    return plot_encoder


def add_heatmap_imgs_to_o_dict(env, agent, observation_key, full_o, v_function,
                               vectorized=False):
    o = full_o[observation_key]
    goal_grid = env.get_mesh_grid(observation_key)
    o_grid = np.c_[np.tile(o, (len(goal_grid), 1)), goal_grid]

    v_vals, indiv_v_vals = v_function(o_grid)
    v_vals = ptu.get_numpy(v_vals)
    indiv_v_vals = [
        ptu.get_numpy(indiv_v_val)
        for indiv_v_val in indiv_v_vals
    ]

    vmin = np.array(indiv_v_vals).min()
    vmax = np.array(indiv_v_vals).max()
    # Assuming square observation space, how many points on x axis
    vary_len = int(len(goal_grid) ** (1/2))
    if not vectorized:
        vmin = min(np.array(indiv_v_vals).min(), v_vals.min())
        vmax = max(np.array(indiv_v_vals).min(), v_vals.max())
        full_o['v_vals'] = normalize_image(
            env.get_image_plt(
                v_vals.reshape((vary_len, vary_len)),
                imsize=env.imsize, vmin=vmin, vmax=vmax
            )
        )

    for goal_dim in range(len(indiv_v_vals)):
        full_o['v_vals_dim_{}'.format(goal_dim)] = normalize_image(
            env.get_image_plt(
                indiv_v_vals[goal_dim].reshape((vary_len, vary_len)),
                imsize=env.imsize, vmin=vmin, vmax=vmax
            )
        )


def add_heatmap_img_to_o_dict(env, agent, observation_key, full_o, v_function):
    o = full_o[observation_key]
    goal_grid = env.get_mesh_grid(observation_key)
    o_grid = np.c_[np.tile(o, (len(goal_grid), 1)), goal_grid]

    v_vals = v_function(o_grid)
    v_vals = ptu.get_numpy(v_vals)

    # Assuming square observation space, how many points on x axis
    vary_len = int(len(goal_grid) ** (1/2))

    vmin = v_vals.min()
    vmax = v_vals.max()
    full_o['v_vals'] = normalize_image(
        env.get_image_plt(
            v_vals.reshape((vary_len, vary_len)),
            imsize=env.imsize, vmin=vmin, vmax=vmax
        )
    )


def train_ae(ae_trainer, training_distrib, num_epochs=100,
             num_batches_per_epoch=500, batch_size=512,
             goal_key='image_desired_goal', rl_csv_fname='progress.csv'):
    from rlkit.core import logger

    logger.remove_tabular_output(rl_csv_fname,
                                 relative_to_snapshot_dir=True)
    logger.add_tabular_output('ae_progress.csv',
                              relative_to_snapshot_dir=True)

    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            goals = ptu.from_numpy(
                training_distrib.sample(batch_size)[goal_key]
            )
            batch =  dict(
                raw_next_observations=goals,
            )
            ae_trainer.train_from_torch(batch)
        log = OrderedDict()
        log['epoch'] = epoch
        append_log(log, ae_trainer.eval_statistics,
                   prefix='ae/')
        logger.record_dict(log)
        logger.dump_tabular(with_prefix=True, with_timestamp=False)
        ae_trainer.end_epoch(epoch)

    logger.add_tabular_output(rl_csv_fname, relative_to_snapshot_dir=True)
    logger.remove_tabular_output('ae_progress.csv',
                                 relative_to_snapshot_dir=True)
