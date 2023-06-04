import os.path as osp
import time

import cv2
import numpy as np


def train_vae(variant, return_data=False):
    from rlkit.util.ml_util import PiecewiseLinearSchedule
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.core import logger
    beta = variant["beta"]
    use_linear_dynamics = variant.get('use_linear_dynamics', False)
    generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn',
                                            generate_vae_dataset)
    variant['generate_vae_dataset_kwargs']['use_linear_dynamics'] = use_linear_dynamics
    train_dataset, test_dataset, info = generate_vae_dataset_fctn(
        variant['generate_vae_dataset_kwargs'])
    if use_linear_dynamics:
        action_dim = train_dataset.data['actions'].shape[2]
    else:
        action_dim = 0
    model = get_vae(variant, action_dim)

    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(
            **variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None

    vae_trainer_class = variant.get('vae_trainer_class', ConvVAETrainer)
    trainer = vae_trainer_class(model, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']

    dump_skew_debug_plots = variant.get('dump_skew_debug_plots', False)
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        trainer.train_epoch(epoch, train_dataset)
        trainer.test_epoch(epoch, test_dataset)

        if should_save_imgs:
            trainer.dump_reconstructions(epoch)
            trainer.dump_samples(epoch)
            if dump_skew_debug_plots:
                trainer.dump_best_reconstruction(epoch)
                trainer.dump_worst_reconstruction(epoch)
                trainer.dump_sampling_histogram(epoch)

        stats = trainer.get_diagnostics()
        for k, v in stats.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        trainer.end_epoch(epoch)

        if epoch % 50 == 0:
            logger.save_itr_params(epoch, model)
    logger.save_extra_data(model, 'vae.pkl', mode='pickle')
    if return_data:
        return model, train_dataset, test_dataset
    return model

def get_vae(variant, action_dim):
    from rlkit.torch.vae.conv_vae import (
        ConvVAE,
        SpatialAutoEncoder,
        AutoEncoder,
    )
    import rlkit.torch.vae.conv_vae as conv_vae
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch
    representation_size = variant["representation_size"]
    use_linear_dynamics = variant.get('use_linear_dynamics', False)
    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
    architecture = variant['vae_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and variant.get('imsize') == 48:
        architecture = conv_vae.imsize48_default_architecture
    variant['vae_kwargs']['architecture'] = architecture
    variant['vae_kwargs']['imsize'] = variant.get('imsize')

    if variant['algo_kwargs'].get('is_auto_encoder', False):
        model = AutoEncoder(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
    elif variant.get('use_spatial_auto_encoder', False):
        model = SpatialAutoEncoder(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
    else:
        vae_class = variant.get('vae_class', ConvVAE)
        if use_linear_dynamics:
            model = vae_class(representation_size, decoder_output_activation=decoder_activation, action_dim=action_dim,**variant['vae_kwargs'])
        else:
            model = vae_class(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
    model.to(ptu.device)
    return model

def generate_vae_dataset(variant):
    print(variant)
    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs',None)
    env_id = variant.get('env_id', None)
    N = variant.get('N', 10000)
    test_p = variant.get('test_p', 0.9)
    use_cached = variant.get('use_cached', True)
    imsize = variant.get('imsize', 84)
    num_channels = variant.get('num_channels', 3)
    show = variant.get('show', False)
    init_camera = variant.get('init_camera', None)
    dataset_path = variant.get('dataset_path', None)
    oracle_dataset_using_set_to_goal = variant.get('oracle_dataset_using_set_to_goal', False)
    random_rollout_data = variant.get('random_rollout_data', False)
    random_rollout_data_set_to_goal = variant.get('random_rollout_data_set_to_goal', True)
    random_and_oracle_policy_data=variant.get('random_and_oracle_policy_data', False)
    random_and_oracle_policy_data_split=variant.get('random_and_oracle_policy_data_split', 0)
    policy_file = variant.get('policy_file', None)
    n_random_steps = variant.get('n_random_steps', 100)
    vae_dataset_specific_env_kwargs = variant.get('vae_dataset_specific_env_kwargs', None)
    save_file_prefix = variant.get('save_file_prefix', None)
    non_presampled_goal_img_is_garbage = variant.get('non_presampled_goal_img_is_garbage', None)

    conditional_vae_dataset = variant.get('conditional_vae_dataset', False)
    use_env_labels = variant.get('use_env_labels', False)
    use_linear_dynamics = variant.get('use_linear_dynamics', False)
    enviorment_dataset = variant.get('enviorment_dataset', False)
    save_trajectories = variant.get('save_trajectories', False)
    save_trajectories = save_trajectories or use_linear_dynamics or conditional_vae_dataset

    tag = variant.get('tag', '')

    from multiworld.core.image_env import ImageEnv, unormalize_image
    import rlkit.torch.pytorch_util as ptu
    from rlkit.util.io import load_local_or_remote_file
    from rlkit.data_management.dataset  import \
        TrajectoryDataset, ImageObservationDataset, InitialObservationDataset, EnvironmentDataset, ConditionalDynamicsDataset

    info = {}
    if dataset_path is not None:
        dataset = load_local_or_remote_file(dataset_path)
        dataset = dataset.item()
        N = dataset['observations'].shape[0] * dataset['observations'].shape[1]
        n_random_steps = dataset['observations'].shape[1]
    else:
        if env_kwargs is None:
            env_kwargs = {}
        if save_file_prefix is None:
            save_file_prefix = env_id
        if save_file_prefix is None:
            save_file_prefix = env_class.__name__
        filename = "/tmp/{}_N{}_{}_imsize{}_random_oracle_split_{}{}.npy".format(
            save_file_prefix,
            str(N),
            init_camera.__name__ if init_camera and hasattr(init_camera, '__name__') else '',
            imsize,
            random_and_oracle_policy_data_split,
            tag,
        )
        if use_cached and osp.isfile(filename):
            dataset = np.load(filename)
            if conditional_vae_dataset:
                dataset = dataset.item()
            print("loaded data from saved file", filename)
        else:
            now = time.time()

            if env_id is not None:
                import gym
                import multiworld
                multiworld.register_all_envs()
                env = gym.make(env_id)
            else:
                if vae_dataset_specific_env_kwargs is None:
                    vae_dataset_specific_env_kwargs = {}
                for key, val in env_kwargs.items():
                    if key not in vae_dataset_specific_env_kwargs:
                        vae_dataset_specific_env_kwargs[key] = val
                env = env_class(**vae_dataset_specific_env_kwargs)
            if not isinstance(env, ImageEnv):
                env = ImageEnv(
                    env,
                    imsize,
                    init_camera=init_camera,
                    transpose=True,
                    normalize=True,
                    non_presampled_goal_img_is_garbage=non_presampled_goal_img_is_garbage,
                )
            else:
                imsize = env.imsize
                env.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
            env.reset()
            info['env'] = env
            if random_and_oracle_policy_data:
                policy_file = load_local_or_remote_file(policy_file)
                policy = policy_file['policy']
                policy.to(ptu.device)
            if random_rollout_data:
                from rlkit.exploration_strategies.ou_strategy import OUStrategy
                policy = OUStrategy(env.action_space)

            if save_trajectories:
                dataset = {
                    'observations': np.zeros((N // n_random_steps, n_random_steps, imsize * imsize * num_channels), dtype=np.uint8),
                    'actions': np.zeros((N // n_random_steps, n_random_steps, env.action_space.shape[0]), dtype=np.float),
                    'env': np.zeros((N // n_random_steps, imsize * imsize * num_channels), dtype=np.uint8),
                    }
            else:
                dataset = np.zeros((N, imsize * imsize * num_channels), dtype=np.uint8)
            labels = []
            for i in range(N):
                if random_and_oracle_policy_data:
                    num_random_steps = int(N*random_and_oracle_policy_data_split)
                    if i < num_random_steps:
                        env.reset()
                        for _ in range(n_random_steps):
                            obs = env.step(env.action_space.sample())[0]
                    else:
                        obs = env.reset()
                        policy.reset()
                        for _ in range(n_random_steps):
                            policy_obs = np.hstack((
                                obs['state_observation'],
                                obs['state_desired_goal'],
                            ))
                            action, _ = policy.get_action(policy_obs)
                            obs, _, _, _ = env.step(action)
                elif random_rollout_data: #ADD DATA WHERE JUST PUCK MOVES
                    if i % n_random_steps == 0:
                        env.reset()
                        policy.reset()
                        env_img = env._get_obs()['image_observation']
                        if random_rollout_data_set_to_goal:
                            env.set_to_goal(env.get_goal())
                    obs = env._get_obs()
                    u = policy.get_action_from_raw_action(env.action_space.sample())
                    env.step(u)
                elif oracle_dataset_using_set_to_goal:
                    print(i)
                    goal = env.sample_goal()
                    env.set_to_goal(goal)
                    obs = env._get_obs()
                else:
                    env.reset()
                    for _ in range(n_random_steps):
                        obs = env.step(env.action_space.sample())[0]

                img = obs['image_observation']
                if use_env_labels:
                    labels.append(obs['label'])
                if save_trajectories:
                    dataset['observations'][i // n_random_steps, i % n_random_steps, :] = unormalize_image(img)
                    dataset['actions'][i // n_random_steps, i % n_random_steps, :] = u
                    dataset['env'][i // n_random_steps, :] = unormalize_image(env_img)
                else:
                    dataset[i, :] = unormalize_image(img)

                if show:
                    img = img.reshape(3, imsize, imsize).transpose()
                    img = img[::-1, :, ::-1]
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                    # radius = input('waiting...')
            print("done making training data", filename, time.time() - now)
            np.save(filename, dataset)
            np.save(filename[:-4] + 'labels.npy', np.array(labels))

    info['train_labels'] = []
    info['test_labels'] = []

    if use_linear_dynamics and conditional_vae_dataset:
        num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)
        train_dataset = ConditionalDynamicsDataset({
            'observations': dataset['observations'][:n, :, :],
            'actions': dataset['actions'][:n, :, :],
            'env': dataset['env'][:n, :]
        })
        test_dataset = ConditionalDynamicsDataset({
            'observations': dataset['observations'][n:, :, :],
            'actions': dataset['actions'][n:, :, :],
            'env': dataset['env'][n:, :]
        })

        num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)
        indices = np.arange(num_trajectories)
        np.random.shuffle(indices)
        train_i, test_i = indices[:n], indices[n:]

        try:
            train_dataset = ConditionalDynamicsDataset({
                'observations': dataset['observations'][train_i, :, :],
                'actions': dataset['actions'][train_i, :, :],
                'env': dataset['env'][train_i, :]
            })
            test_dataset = ConditionalDynamicsDataset({
                'observations': dataset['observations'][test_i, :, :],
                'actions': dataset['actions'][test_i, :, :],
                'env': dataset['env'][test_i, :]
            })
        except:
            train_dataset = ConditionalDynamicsDataset({
                'observations': dataset['observations'][train_i, :, :],
                'actions': dataset['actions'][train_i, :, :],

            })
            test_dataset = ConditionalDynamicsDataset({
                'observations': dataset['observations'][test_i, :, :],
                'actions': dataset['actions'][test_i, :, :],
            })
    elif use_linear_dynamics:
        num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)
        train_dataset = TrajectoryDataset({
            'observations': dataset['observations'][:n, :, :],
            'actions': dataset['actions'][:n, :, :]
        })
        test_dataset = TrajectoryDataset({
            'observations': dataset['observations'][n:, :, :],
            'actions': dataset['actions'][n:, :, :]
        })
    elif enviorment_dataset:
        n = int(n_random_steps * test_p)
        train_dataset = EnvironmentDataset({
            'observations': dataset['observations'][:, :n, :],
        })
        test_dataset = EnvironmentDataset({
            'observations': dataset['observations'][:, n:, :],
        })
    elif conditional_vae_dataset:
        num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)
        indices = np.arange(num_trajectories)
        np.random.shuffle(indices)
        train_i, test_i = indices[:n], indices[n:]

        try:
            train_dataset = InitialObservationDataset({
                'observations': dataset['observations'][train_i, :, :],
                'env': dataset['env'][train_i, :]
            })
            test_dataset = InitialObservationDataset({
                'observations': dataset['observations'][test_i, :, :],
                'env': dataset['env'][test_i, :]
            })
        except:
            train_dataset = InitialObservationDataset({
                'observations': dataset['observations'][train_i, :, :],
            })
            test_dataset = InitialObservationDataset({
                'observations': dataset['observations'][test_i, :, :],
            })
    else:
        n = int(N * test_p)
        train_dataset = ImageObservationDataset(dataset[:n, :])
        test_dataset = ImageObservationDataset(dataset[n:, :])
    return train_dataset, test_dataset, info

def train_vae_and_update_variant(variant):
    from rlkit.core import logger
    rl_variant = variant['rl_variant']
    vae_variant = variant['vae_variant']
    if rl_variant.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae, vae_train_data, vae_test_data = train_vae(vae_variant,
                                                       return_data=True)
        if rl_variant.get('save_vae_data', False):
            rl_variant['vae_train_data'] = vae_train_data
            rl_variant['vae_test_data'] = vae_test_data
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output(
            'vae_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        rl_variant['vae_path'] = vae  # just pass the VAE directly
    else:
        if rl_variant.get('save_vae_data', False):
            vae_train_data, vae_test_data, info = generate_vae_dataset(
                vae_variant['generate_vae_dataset_kwargs']
            )
            rl_variant['vae_train_data'] = vae_train_data
            rl_variant['vae_test_data'] = vae_test_data