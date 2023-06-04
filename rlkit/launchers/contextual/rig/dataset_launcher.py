import time
import numpy as np
from torch.utils import data

def generate_vae_dataset(
        env_kwargs,
        env_id,
        env_class,
        imsize,
        init_camera,
        N=10000,
        batch_size=128,
        test_p=0.9,
        use_cached=True,
        num_channels=3,
        show=False,
        dataset_path=None,
        save_dataset_path=None,
        oracle_dataset_using_set_to_goal=False,
        random_rollout_data=False,
        random_rollout_data_set_to_goal=True,
        random_and_oracle_policy_data=False,
        random_and_oracle_policy_data_split=0,
        policy_file=None,
        n_random_steps=100,
        vae_dataset_specific_env_kwargs=None,
        save_file_prefix=None,
        non_presampled_goal_img_is_garbage=None,
        conditional_vae_dataset=False,
        use_env_labels=False,
        use_linear_dynamics=False,
        enviorment_dataset=False,
        save_trajectories=False,
        tag="",
        train_batch_loader_kwargs=None,
        test_batch_loader_kwargs=None,
        vae_dataset_specific_kwargs=None,
    ):
    save_trajectories = save_trajectories or use_linear_dynamics or conditional_vae_dataset

    assert N % n_random_steps == 0, "Fix N/horizon or dataset generation will fail"

    from multiworld.core.image_env import ImageEnv, unormalize_image
    import rlkit.torch.pytorch_util as ptu
    from rlkit.util.io import load_local_or_remote_file
    from rlkit.data_management.dataset  import (
        TrajectoryDataset, ImageObservationDataset, EnvironmentDataset, ConditionalDynamicsDataset, InitialObservationNumpyDataset,
        InfiniteBatchLoader,
    )

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
            dataset = load_local_or_remote_file(filename)
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
            if save_dataset_path is not None:
                np.save(save_dataset_path, dataset)
            #np.save(filename[:-4] + 'labels.npy', np.array(labels))

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

        if 'env' in dataset:
            train_dataset = InitialObservationNumpyDataset({
                'observations': dataset['observations'][train_i, :, :],
                'env': dataset['env'][train_i, :]
            })
            test_dataset = InitialObservationNumpyDataset({
                'observations': dataset['observations'][test_i, :, :],
                'env': dataset['env'][test_i, :]
            })
        else:
            train_dataset = InitialObservationNumpyDataset({
                'observations': dataset['observations'][train_i, :, :],
            })
            test_dataset = InitialObservationNumpyDataset({
                'observations': dataset['observations'][test_i, :, :],
            })

        if train_batch_loader_kwargs is None:
            train_batch_loader_kwargs = dict(batch_size=batch_size, num_workers=0)
        if test_batch_loader_kwargs is None:
            test_batch_loader_kwargs = dict(batch_size=batch_size, num_workers=0)

        train_data_loader = data.DataLoader(train_dataset,
            shuffle=True, drop_last=True, **train_batch_loader_kwargs)
        test_data_loader = data.DataLoader(test_dataset,
            shuffle=True, drop_last=True, **test_batch_loader_kwargs)

        train_dataset = InfiniteBatchLoader(train_data_loader)
        test_dataset = InfiniteBatchLoader(test_data_loader)
    else:
        n = int(N * test_p)
        train_dataset = ImageObservationDataset(dataset[:n, :])
        test_dataset = ImageObservationDataset(dataset[n:, :])
    return train_dataset, test_dataset, info
