import os.path as osp
import time

import numpy as np

from rlkit.samplers.data_collector import VAEWrappedEnvPathCollector
from rlkit.visualization.video import VideoSaveFunction
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm


def grill_tdm_td3_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    if not variant['grill_variant'].get('do_state_exp', False):
        train_vae_and_update_variant(variant)
    grill_tdm_td3_experiment(variant['grill_variant'])


def grill_tdm_twin_sac_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_tdm_twin_sac_experiment(variant['grill_variant'])

def grill_her_twin_sac_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    if not variant['grill_variant'].get('do_state_exp', False):
        train_vae_and_update_variant(variant)
    grill_her_twin_sac_experiment(variant['grill_variant'])


def grill_her_td3_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    if not variant['grill_variant'].get('do_state_exp', False):
        train_vae_and_update_variant(variant)
    grill_her_td3_experiment(variant['grill_variant'])


def grill_her_td3_online_vae_full_experiment(variant):
    variant['grill_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    # variant['grill_variant']['vae_trainer_kwargs'] = \
    #     variant['train_vae_variant']['algo_kwargs']
    # if variant['double_algo']:
    #     grill_her_td3_experiment_online_vae_exploring(variant['grill_variant'])
    # else:
    grill_her_td3_experiment_online_vae(variant['grill_variant'])


def grill_her_td3_offpolicy_online_vae_full_experiment(variant):
    variant['grill_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    # variant['grill_variant']['vae_trainer_kwargs'] = \
    #     variant['train_vae_variant']['algo_kwargs']
    # if variant['double_algo']:
    #     grill_her_td3_experiment_online_vae_exploring(variant['grill_variant'])
    # else:
    grill_her_td3_experiment_offpolicy_online_vae(variant['grill_variant'])


def grill_her_twin_sac_online_vae_full_experiment(variant):
    variant['grill_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_her_twin_sac_experiment_online_vae(variant['grill_variant'])


def arl_full_experiment(variant):
    variant['grill_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    active_representation_learning_experiment(variant['grill_variant'])


def grill_tdm_td3_online_vae_full_experiment(variant):
    variant['grill_variant']['save_vae_data'] = True
    variant['grill_variant']['vae_trainer_kwargs'] = \
        variant['train_vae_variant']['algo_kwargs']

    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_tdm_td3_experiment_online_vae(variant['grill_variant'])

def HER_baseline_her_td3_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    HER_baseline_her_td3_experiment(variant['grill_variant'])

def HER_baseline_twin_sac_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    HER_baseline_twin_sac_experiment(variant['grill_variant'])

def full_experiment_variant_preprocess(variant):
    train_vae_variant = variant['train_vae_variant']
    grill_variant = variant['grill_variant']
    if 'env_id' in variant:
        assert 'env_class' not in variant
        env_id = variant['env_id']
        grill_variant['env_id'] = env_id
        train_vae_variant['generate_vae_dataset_kwargs']['env_id'] = env_id
    else:
        env_class = variant['env_class']
        env_kwargs = variant['env_kwargs']
        train_vae_variant['generate_vae_dataset_kwargs']['env_class'] = (
            env_class
        )
        train_vae_variant['generate_vae_dataset_kwargs']['env_kwargs'] = (
            env_kwargs
        )
        grill_variant['env_class'] = env_class
        grill_variant['env_kwargs'] = env_kwargs
    init_camera = variant.get('init_camera', None)
    imsize = variant.get('imsize', 84)
    train_vae_variant['generate_vae_dataset_kwargs']['init_camera'] = (
        init_camera
    )
    train_vae_variant['generate_vae_dataset_kwargs']['imsize'] = imsize
    train_vae_variant['imsize'] = imsize
    grill_variant['imsize'] = imsize
    grill_variant['init_camera'] = init_camera


def train_vae_and_update_variant(variant):
    from rlkit.core import logger
    grill_variant = variant['grill_variant']
    train_vae_variant = variant['train_vae_variant']
    if grill_variant.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae, vae_train_data, vae_test_data = train_vae(train_vae_variant,
                                                       return_data=True)
        if grill_variant.get('save_vae_data', False):
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output(
            'vae_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        grill_variant['vae_path'] = vae  # just pass the VAE directly
    else:
        if grill_variant.get('save_vae_data', False):
            vae_train_data, vae_test_data, info = generate_vae_dataset(
                train_vae_variant['generate_vae_dataset_kwargs']
            )
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data


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
                    import cv2
                    img = img.reshape(3, imsize, imsize).transpose()
                    img = img[::-1, :, ::-1]
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                    # radius = input('waiting...')
            print("done making training data", filename, time.time() - now)
            np.save(filename, dataset)
            # np.save(filename[:-4] + 'labels.npy', np.array(labels))

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


def get_presampled_goals_path(path=''):
    """
    :param path: if relative, this will rpe
    :param config: One of a few options:
        - string: the path to the
        - tuple of two strings: the first string specifies the 'mode' and the
            second string specifies extra parameters to that mode
        - None: return None
    :return:  Path to the presampled goals, or None.
    """
    if not path:
        return path
    if path[0] == '/':
        return path
    else:
        import multiworld.envs.mujoco as mwmj
        return osp.join(osp.dirname(mwmj.__file__), path)


def get_envs(variant):
    from multiworld.core.image_env import ImageEnv
    from rlkit.envs.vae_wrappers import VAEWrappedEnv, ConditionalVAEWrappedEnv
    from rlkit.util.io import load_local_or_remote_file
    from rlkit.torch.vae.conditional_conv_vae import CVAE, CDVAE, ACE, CADVAE, DeltaCVAE

    render = variant.get('render', False)
    vae_path = variant.get("vae_path", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)
    presample_goals = variant.get('presample_goals', False)
    presample_image_goals_only = variant.get('presample_image_goals_only', False)
    presampled_goals_path = get_presampled_goals_path(
        variant.get('presampled_goals_path', None))
    vae = load_local_or_remote_file(vae_path) if type(vae_path) is str else vae_path
    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    if not do_state_exp:
        if isinstance(env, ImageEnv):
            image_env = env
        else:
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )
        if presample_goals:
            """
            This will fail for online-parallel as presampled_goals will not be
            serialized. Also don't use this for online-vae.
            """
            if presampled_goals_path is None:
                image_env.non_presampled_goal_img_is_garbage = True
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    env=vae_env,
                    env_id=variant.get('env_id', None),
                    **variant['goal_generation_kwargs']
                )
                del vae_env
            else:
                presampled_goals = load_local_or_remote_file(
                    presampled_goals_path
                ).item()
            del image_env
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
                presampled_goals=presampled_goals,
                **variant.get('image_env_kwargs', {})
            )
            vae_env = VAEWrappedEnv(
                image_env,
                vae,
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                reward_params=reward_params,
                presampled_goals = presampled_goals,
                **variant.get('vae_wrapped_env_kwargs', {})
            )
            print("Presampling all goals only")
        else:
            if type(vae) is CVAE or type(vae) is CDVAE or type(vae) is ACE or type(vae) is CADVAE or type(vae) is DeltaCVAE:
                vae_env = ConditionalVAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
            else:
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
            if presample_image_goals_only:
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    image_env=vae_env.wrapped_env,
                    **variant['goal_generation_kwargs']
                )
                image_env.set_presampled_goals(presampled_goals)
                print("Presampling image goals only")
            else:
                print("Not using presampled goals")

        env = vae_env

    return env


def get_exploration_strategy(variant, env):
    from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
    from rlkit.exploration_strategies.ou_strategy import OUStrategy
    from rlkit.exploration_strategies.noop import NoopStrategy

    exploration_type = variant['exploration_type']
    exploration_noise = variant.get('exploration_noise', 0.1)
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=exploration_noise,
        )
    elif exploration_type == 'noop':
        es = NoopStrategy(
            action_space=env.action_space
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    return es


def grill_preprocess_variant(variant):
    if variant.get("do_state_exp", False):
        variant['observation_key'] = 'state_observation'
        variant['desired_goal_key'] = 'state_desired_goal'
        variant['achieved_goal_key'] = 'state_acheived_goal'


def grill_her_td3_experiment(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.torch.her.her_td3 import HerTd3
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    her_kwargs = algo_kwargs['her_kwargs']
    her_kwargs['observation_key'] = observation_key
    her_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        env.vae.to(ptu.device)

    algorithm.train()


def grill_her_twin_sac_experiment(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    # from rlkit.torch.her.her_twin_sac import HerTwinSAC
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)
    max_path_length = variant['max_path_length']
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']
    # algo_kwargs['replay_buffer'] = replay_buffer
    # base_kwargs = algo_kwargs['base_kwargs']
    # base_kwargs['training_env'] = env
    # base_kwargs['render'] = variant["render"]
    # base_kwargs['render_during_eval'] = variant["render"]
    # her_kwargs = algo_kwargs['her_kwargs']
    # her_kwargs['observation_key'] = observation_key
    # her_kwargs['desired_goal_key'] = desired_goal_key
    # algorithm = HerTwinSAC(
    #     env,
    #     qf1=qf1,
    #     qf2=qf2,
    #     vf=vf,
    #     target_vf=target_vf,
    #     policy=policy,
    #     exploration_policy=exploration_policy,
    #     **variant['algo_kwargs']
    # )

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=variant['evaluation_goal_sampling_mode'],
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=variant['exploration_goal_sampling_mode'],
    )

    algorithm = TorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()


def grill_tdm_td3_experiment(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.core import logger
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.state_distance.tdm_networks import TdmQf, TdmPolicy
    from rlkit.state_distance.tdm_td3 import TdmTd3
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size

    vectorized = 'vectorized' in env.reward_type
    norm_order = env.norm_order
    variant['algo_kwargs']['tdm_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['norm_order'] = norm_order

    qf1 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    variant['replay_buffer_kwargs']['vectorized'] = vectorized
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = TdmTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        env.vae.to(ptu.device)
    if variant.get("save_video", True):
        logdir = logger.get_snapshot_dir()
        policy.train(False)
        rollout_function = rf.create_rollout_function(
            rf.tdm_rollout,
            init_tau=algorithm.max_tau,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)
    algorithm.train()


def grill_her_td3_experiment_online_vae_exploring(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.torch.her.online_vae_joint_algo import OnlineVaeHerJointAlgo
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.torch.td3.td3 import TD3
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    exploring_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    exploring_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    exploring_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    exploring_exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=exploring_policy,
    )

    vae = env.vae
    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    if variant.get('use_replay_buffer_goals', False):
        env.replay_buffer = replay_buffer
        env.use_replay_buffer_goals = True

    vae_trainer_kwargs = variant.get('vae_trainer_kwargs')
    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'],
                       **vae_trainer_kwargs)

    control_algorithm = TD3(
        env=env,
        training_env=env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    exploring_algorithm = TD3(
        env=env,
        training_env=env,
        qf1=exploring_qf1,
        qf2=exploring_qf2,
        policy=exploring_policy,
        exploration_policy=exploring_exploration_policy,
        **variant['algo_kwargs']
    )

    assert 'vae_training_schedule' not in variant,\
        "Just put it in joint_algo_kwargs"
    algorithm = OnlineVaeHerJointAlgo(
        vae=vae,
        vae_trainer=t,
        env=env,
        training_env=env,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        algo1=control_algorithm,
        algo2=exploring_algorithm,
        algo1_prefix="Control_",
        algo2_prefix="VAE_Exploration_",
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['joint_algo_kwargs']
    )


    algorithm.to(ptu.device)
    vae.to(ptu.device)
    if variant.get("save_video", True):
        policy.train(False)
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)
    algorithm.train()


def grill_her_twin_sac_experiment_online_vae(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer

    grill_preprocess_variant(variant)
    env = get_envs(variant)

    uniform_dataset_fn = variant.get('generate_uniform_dataset_fn', None)
    if uniform_dataset_fn:
        uniform_dataset=uniform_dataset_fn(
            **variant['generate_uniform_dataset_kwargs']
        )
    else:
        uniform_dataset=None

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    vae = env.vae

    replay_buffer_class = variant.get("replay_buffer_class", OnlineVaeRelabelingBuffer)
    replay_buffer = replay_buffer_class(
        vae=env.vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    vae_trainer_class = variant.get("vae_trainer_class", ConvVAETrainer)
    vae_trainer = vae_trainer_class(
        env.vae,
        **variant['online_vae_trainer_kwargs']
    )
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    max_path_length = variant['max_path_length']

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=variant['evaluation_goal_sampling_mode'],
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=variant['exploration_goal_sampling_mode'],
    )

    algorithm = OnlineVaeAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        uniform_dataset=uniform_dataset,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        video_func = VideoSaveFunction(
            env,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)
    if variant['custom_goal_sampler'] == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()


def grill_her_td3_experiment_online_vae(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.torch.td3.td3 import TD3
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.exploration_strategies.gaussian_and_epsilon import \
        GaussianAndEpsilonStrategy

    grill_preprocess_variant(variant)
    env = get_envs(variant)

    uniform_dataset_fn = variant.get('generate_uniform_dataset_fn', None)
    if uniform_dataset_fn:
        uniform_dataset=uniform_dataset_fn(
            **variant['generate_uniform_dataset_kwargs']
        )
    else:
        uniform_dataset=None

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        # **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        # **variant['policy_kwargs']
    )

    es = GaussianAndEpsilonStrategy(
        action_space=env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    vae = env.vae

    replay_buffer_class = variant.get("replay_buffer_class", OnlineVaeRelabelingBuffer)
    replay_buffer = replay_buffer_class(
        vae=env.vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    vae_trainer_class = variant.get("vae_trainer_class", ConvVAETrainer)
    vae_trainer = vae_trainer_class(
        env.vae,
        **variant['online_vae_trainer_kwargs']
    )
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    max_path_length = variant['max_path_length']

    trainer = TD3(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['td3_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=variant['evaluation_goal_sampling_mode'],
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        env,
        expl_policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=variant['exploration_goal_sampling_mode'],
    )

    algorithm = OnlineVaeAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        uniform_dataset=uniform_dataset,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        video_func = VideoSaveFunction(
            env,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)
    if variant['custom_goal_sampler'] == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()


def grill_her_td3_experiment_offpolicy_online_vae(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.torch.td3.td3 import TD3
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.exploration_strategies.gaussian_and_epsilon import \
        GaussianAndEpsilonStrategy
    from rlkit.torch.vae.online_vae_offpolicy_algorithm import OnlineVaeOffpolicyAlgorithm

    grill_preprocess_variant(variant)
    env = get_envs(variant)

    uniform_dataset_fn = variant.get('generate_uniform_dataset_fn', None)
    if uniform_dataset_fn:
        uniform_dataset=uniform_dataset_fn(
            **variant['generate_uniform_dataset_kwargs']
        )
    else:
        uniform_dataset=None

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        # **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        # **variant['policy_kwargs']
    )

    es = GaussianAndEpsilonStrategy(
        action_space=env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    vae = env.vae

    replay_buffer_class = variant.get("replay_buffer_class", OnlineVaeRelabelingBuffer)
    replay_buffer = replay_buffer_class(
        vae=env.vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,

        **variant['replay_buffer_kwargs']
    )
    replay_buffer.representation_size = vae.representation_size

    vae_trainer_class = variant.get("vae_trainer_class", ConvVAETrainer)
    vae_trainer = vae_trainer_class(
        env.vae,
        **variant['online_vae_trainer_kwargs']
    )
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    max_path_length = variant['max_path_length']

    trainer = TD3(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['td3_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=variant['evaluation_goal_sampling_mode'],
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        env,
        expl_policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=variant['exploration_goal_sampling_mode'],
    )

    algorithm = OnlineVaeOffpolicyAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        uniform_dataset=uniform_dataset,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        video_func = VideoSaveFunction(
            env,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)
    if variant['custom_goal_sampler'] == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)

    algorithm.pretrain()
    algorithm.train()


def active_representation_learning_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.arl.active_representation_learning_algorithm import \
        ActiveRepresentationLearningAlgorithm
    from rlkit.torch.arl.representation_wrappers import RepresentationWrappedEnv
    from multiworld.core.image_env import ImageEnv
    from rlkit.samplers.data_collector import MdpPathCollector

    grill_preprocess_variant(variant)

    model_class = variant.get('model_class')
    model_kwargs = variant.get('model_kwargs')

    model = model_class(**model_kwargs)
    model.representation_size = 4
    model.imsize = 48
    variant["vae_path"] = model

    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    env = variant["env_class"](**variant['env_kwargs'])
    image_env = ImageEnv(
        env,
        variant.get('imsize'),
        init_camera=init_camera,
        transpose=True,
        normalize=True,
    )
    env = RepresentationWrappedEnv(
        image_env,
        model,
    )

    uniform_dataset_fn = variant.get('generate_uniform_dataset_fn', None)
    if uniform_dataset_fn:
        uniform_dataset=uniform_dataset_fn(
            **variant['generate_uniform_dataset_kwargs']
        )
    else:
        uniform_dataset=None

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = env.observation_space.spaces[observation_key].low.size
    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    vae = env.vae

    replay_buffer = ObsDictReplayBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )

    model_trainer_class = variant.get('model_trainer_class')
    model_trainer_kwargs = variant.get('model_trainer_kwargs')
    model_trainer = model_trainer_class(
        model,
        **model_trainer_kwargs,
    )
    # vae_trainer = ConvVAETrainer(
    #     env.vae,
    #     **variant['online_vae_trainer_kwargs']
    # )
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    max_path_length = variant['max_path_length']

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    # trainer = HERTrainer(trainer)
    eval_path_collector = MdpPathCollector(
        env,
        MakeDeterministic(policy),
        # max_path_length,
        # observation_key=observation_key,
        # desired_goal_key=desired_goal_key,
    )
    expl_path_collector = MdpPathCollector(
        env,
        policy,
        # max_path_length,
        # observation_key=observation_key,
        # desired_goal_key=desired_goal_key,
    )

    algorithm = ActiveRepresentationLearningAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        model=model,
        model_trainer=model_trainer,
        uniform_dataset=uniform_dataset,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()


def grill_tdm_td3_experiment_online_vae(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.state_distance.tdm_networks import TdmQf, TdmPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.torch.online_vae.online_vae_tdm_td3 import OnlineVaeTdmTd3
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)
    vae_trainer_kwargs = variant.get('vae_trainer_kwargs')
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size

    vectorized = 'vectorized' in env.reward_type
    variant['algo_kwargs']['tdm_td3_kwargs']['tdm_kwargs'][
        'vectorized'] = vectorized

    norm_order = env.norm_order
    # variant['algo_kwargs']['tdm_td3_kwargs']['tdm_kwargs'][
    #     'norm_order'] = norm_order

    qf1 = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=norm_order,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=norm_order,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    vae = env.vae

    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']['tdm_td3_kwargs']
    td3_kwargs = algo_kwargs['td3_kwargs']
    td3_kwargs['training_env'] = env
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algo_kwargs["replay_buffer"] = replay_buffer

    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'],
                       **vae_trainer_kwargs)
    render = variant["render"]
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    algorithm = OnlineVaeTdmTd3(
        online_vae_kwargs=dict(
            vae=vae,
            vae_trainer=t,
            **variant['algo_kwargs']['online_vae_kwargs']
        ),
        tdm_td3_kwargs=dict(
            env=env,
            qf1=qf1,
            qf2=qf2,
            policy=policy,
            exploration_policy=exploration_policy,
            **variant['algo_kwargs']['tdm_td3_kwargs']
        ),
    )

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    if variant.get("save_video", True):
        policy.train(False)
        rollout_function = rf.create_rollout_function(
            rf.tdm_rollout,
            init_tau=algorithm._sample_max_tau_for_rollout(),
            decrement_tau=algorithm.cycle_taus_for_rollout,
            cycle_tau=algorithm.cycle_taus_for_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        env.vae.to(ptu.device)

    algorithm.train()


def grill_tdm_twin_sac_experiment(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.state_distance.tdm_networks import (
        TdmQf, TdmVf,
        StochasticTdmPolicy,
    )
    from rlkit.state_distance.tdm_twin_sac import TdmTwinSAC
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size

    vectorized = 'vectorized' in env.reward_type
    norm_order = env.norm_order
    variant['algo_kwargs']['tdm_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['vectorized'] = vectorized
    variant['vf_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['norm_order'] = norm_order
    variant['vf_kwargs']['norm_order'] = norm_order

    qf1 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    vf = TdmVf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        **variant['vf_kwargs']
    )
    policy = StochasticTdmPolicy(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    variant['replay_buffer_kwargs']['vectorized'] = vectorized
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = TdmTwinSAC(
        env,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        policy=policy,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        rollout_function = rf.create_rollout_function(
            rf.tdm_rollout,
            init_tau=algorithm._sample_max_tau_for_rollout(),
            decrement_tau=algorithm.cycle_taus_for_rollout,
            cycle_tau=algorithm.cycle_taus_for_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        env.vae.to(ptu.device)

    algorithm.train()


# def grill_her_td3_experiment_online_vae(variant):
#     import rlkit.samplers.rollout_functions as rf
#     import rlkit.torch.pytorch_util as ptu
#     from rlkit.data_management.online_vae_replay_buffer import \
#         OnlineVaeRelabelingBuffer
#     from rlkit.exploration_strategies.base import (
#         PolicyWrappedWithExplorationStrategy
#     )
#     from rlkit.torch.her.online_vae_her_td3 import OnlineVaeHerTd3
#     from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
#     from rlkit.torch.vae.vae_trainer import ConvVAETrainer
#     grill_preprocess_variant(variant)
#     env = get_envs(variant)
#     es = get_exploration_strategy(variant, env)
#     observation_key = variant.get('observation_key', 'latent_observation')
#     desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
#     achieved_goal_key = desired_goal_key.replace("desired", "achieved")
#     obs_dim = (
#             env.observation_space.spaces[observation_key].low.size
#             + env.observation_space.spaces[desired_goal_key].low.size
#     )
#     action_dim = env.action_space.low.size
#     qf1 = FlattenMlp(
#         input_size=obs_dim + action_dim,
#         output_size=1,
#         **variant['qf_kwargs'],
#     )
#     qf2 = FlattenMlp(
#         input_size=obs_dim + action_dim,
#         output_size=1,
#         **variant['qf_kwargs'],
#     )
#     policy = TanhMlpPolicy(
#         input_size=obs_dim,
#         output_size=action_dim,
#         **variant['policy_kwargs'],
#     )
#     target_qf1 = FlattenMlp(
#         input_size=obs_dim + action_dim,
#         output_size=1,
#         **variant['qf_kwargs'],
#     )
#     target_qf2 = FlattenMlp(
#         input_size=obs_dim + action_dim,
#         output_size=1,
#         **variant['qf_kwargs'],
#     )
#     target_policy = TanhMlpPolicy(
#         input_size=obs_dim,
#         output_size=action_dim,
#         **variant['policy_kwargs'],
#     )
#     exploration_policy = PolicyWrappedWithExplorationStrategy(
#         exploration_strategy=es,
#         policy=policy,
#     )

#     vae = env.vae
#     vae.action_dim = action_dim

#     replay_buffer = OnlineVaeRelabelingBuffer(
#         vae=vae,
#         env=env,
#         observation_key=observation_key,
#         desired_goal_key=desired_goal_key,
#         achieved_goal_key=achieved_goal_key,
#         **variant['replay_buffer_kwargs']
#     )
#     variant["algo_kwargs"]["base_kwargs"]["replay_buffer"] = replay_buffer
#     if variant.get('use_replay_buffer_goals', False):
#         env.replay_buffer = replay_buffer
#         env.use_replay_buffer_goals = True

#     vae_trainer_kwargs = variant.get('vae_trainer_kwargs')
#     t = ConvVAETrainer(variant['vae_train_data'],
#                        variant['vae_test_data'],
#                        vae,
#                        beta=variant['online_vae_beta'],
#                        **vae_trainer_kwargs)
#     render = variant["render"]
#     assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
#     algorithm = OnlineVaeHerTd3(
#         online_vae_kwargs=dict(
#             vae=vae,
#             vae_trainer=t,
#             **variant['algo_kwargs']['online_vae_kwargs']
#         ),
#         base_kwargs=dict(
#             env=env,
#             training_env=env,
#             policy=policy,
#             exploration_policy=exploration_policy,
#             render=render,
#             render_during_eval=render,
#             **variant['algo_kwargs']['base_kwargs'],
#         ),
#         her_kwargs=dict(
#             observation_key=observation_key,
#             desired_goal_key=desired_goal_key,
#         ),
#         td3_kwargs=dict(
#             **variant['algo_kwargs']['td3_kwargs'],
#             qf1=qf1,
#             qf2=qf2,
#             target_qf1=target_qf1,
#             target_qf2=target_qf2,
#             target_policy=target_policy,
#         )
#     )


#     algorithm.to(ptu.device)
#     vae.to(ptu.device)
#     if variant.get("save_video", True):
#         rollout_function = rf.create_rollout_function(
#             rf.multitask_rollout,
#             max_path_length=algorithm.max_path_length,
#             observation_key=algorithm.observation_key,
#             desired_goal_key=algorithm.desired_goal_key,
#         )
#         video_func = get_video_save_func(
#             rollout_function,
#             env,
#             algorithm.eval_policy,
#             variant,
#         )
#         algorithm.post_train_funcs.append(video_func)
#     algorithm.train()


def grill_her_td3_experiment_online_vae_exploring(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.torch.her.online_vae_joint_algo import OnlineVaeHerJointAlgo
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.torch.td3.td3 import TD3
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    exploring_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    exploring_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    exploring_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    exploring_exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=exploring_policy,
    )

    vae = env.vae
    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    if variant.get('use_replay_buffer_goals', False):
        env.replay_buffer = replay_buffer
        env.use_replay_buffer_goals = True

    vae_trainer_kwargs = variant.get('vae_trainer_kwargs')
    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'],
                       **vae_trainer_kwargs)

    control_algorithm = TD3(
        env=env,
        training_env=env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    exploring_algorithm = TD3(
        env=env,
        training_env=env,
        qf1=exploring_qf1,
        qf2=exploring_qf2,
        policy=exploring_policy,
        exploration_policy=exploring_exploration_policy,
        **variant['algo_kwargs']
    )

    assert 'vae_training_schedule' not in variant,\
        "Just put it in joint_algo_kwargs"
    algorithm = OnlineVaeHerJointAlgo(
        vae=vae,
        vae_trainer=t,
        env=env,
        training_env=env,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        algo1=control_algorithm,
        algo2=exploring_algorithm,
        algo1_prefix="Control_",
        algo2_prefix="VAE_Exploration_",
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['joint_algo_kwargs']
    )


    algorithm.to(ptu.device)
    vae.to(ptu.device)
    if variant.get("save_video", True):
        policy.train(False)
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)
    algorithm.train()

def HER_baseline_her_td3_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.torch.her.her_td3 import HerTd3
    from rlkit.torch.networks import MergedCNN, CNNPolicy
    import torch
    from multiworld.core.image_env import ImageEnv
    from rlkit.util.io import load_local_or_remote_file

    init_camera = variant.get("init_camera", None)
    presample_goals = variant.get('presample_goals', False)
    presampled_goals_path = get_presampled_goals_path(
        variant.get('presampled_goals_path', None))

    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    image_env = ImageEnv(
        env,
        variant.get('imsize'),
        reward_type='image_sparse',
        init_camera=init_camera,
        transpose=True,
        normalize=True,
    )
    if presample_goals:
        if presampled_goals_path is None:
            image_env.non_presampled_goal_img_is_garbage = True
            presampled_goals = variant['generate_goal_dataset_fctn'](
                env=image_env,
                **variant['goal_generation_kwargs']
            )
        else:
            presampled_goals = load_local_or_remote_file(
                presampled_goals_path
            ).item()
        del image_env
        env = ImageEnv(
            env,
            variant.get('imsize'),
            reward_type='image_distance',
            init_camera=init_camera,
            transpose=True,
            normalize=True,
            presampled_goals=presampled_goals,
        )
    else:
        env = image_env

    es = get_exploration_strategy(variant, env)

    observation_key = variant.get('observation_key', 'image_observation')
    desired_goal_key = variant.get('desired_goal_key', 'image_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    imsize=variant['imsize']
    action_dim = env.action_space.low.size
    qf1 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )
    qf2 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )

    policy = CNNPolicy(input_width=imsize,
                       input_height=imsize,
                       added_fc_input_size=0,
                       output_size=action_dim,
                       input_channels=3 * 2,
                       output_activation=torch.tanh,
                       **variant['cnn_params'],
                       )
    target_qf1 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )
    target_qf2 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )

    target_policy = CNNPolicy(input_width=imsize,
                       input_height=imsize,
                       added_fc_input_size=0,
                       output_size=action_dim,
                       input_channels=3 * 2,
                       output_activation=torch.tanh,
                       **variant['cnn_params'],
                       )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    her_kwargs = algo_kwargs['her_kwargs']
    her_kwargs['observation_key'] = observation_key
    her_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    algorithm.train()

def HER_baseline_twin_sac_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.torch.her.her_twin_sac import HerTwinSAC
    from rlkit.torch.sac.policies import TanhCNNGaussianPolicy
    from rlkit.torch.networks import MergedCNN, CNN
    import torch
    from multiworld.core.image_env import ImageEnv
    from rlkit.util.io import load_local_or_remote_file

    init_camera = variant.get("init_camera", None)
    presample_goals = variant.get('presample_goals', False)
    presampled_goals_path = get_presampled_goals_path(
        variant.get('presampled_goals_path', None))

    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    image_env = ImageEnv(
        env,
        variant.get('imsize'),
        reward_type='image_sparse',
        init_camera=init_camera,
        transpose=True,
        normalize=True,
    )
    if presample_goals:
        if presampled_goals_path is None:
            image_env.non_presampled_goal_img_is_garbage = True
            presampled_goals = variant['generate_goal_dataset_fctn'](
                env=image_env,
                **variant['goal_generation_kwargs']
            )
        else:
            presampled_goals = load_local_or_remote_file(
                presampled_goals_path
            ).item()
        del image_env
        env = ImageEnv(
            env,
            variant.get('imsize'),
            reward_type='image_distance',
            init_camera=init_camera,
            transpose=True,
            normalize=True,
            presampled_goals=presampled_goals,
        )
    else:
        env = image_env
    es = get_exploration_strategy(variant, env)

    observation_key = variant.get('observation_key', 'image_observation')
    desired_goal_key = variant.get('desired_goal_key', 'image_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    imsize=variant['imsize']
    action_dim = env.action_space.low.size
    qf1 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )
    qf2 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )

    policy = TanhCNNGaussianPolicy(input_width=imsize,
                       input_height=imsize,
                       added_fc_input_size=0,
                       output_size=action_dim,
                       input_channels=3 * 2,
                       output_activation=torch.tanh,
                       **variant['cnn_params'],
                       )

    vf = CNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    **variant['cnn_params']
                    )
    target_vf = CNN(input_width=imsize,
             input_height=imsize,
             output_size=1,
             input_channels=3 * 2,
             **variant['cnn_params']
             )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    her_kwargs = algo_kwargs['her_kwargs']
    her_kwargs['observation_key'] = observation_key
    her_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = HerTwinSAC(
        env,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        target_vf=target_vf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    algorithm.train()


def get_state_experiment_video_save_function(rollout_function, env, policy, variant):
    from multiworld.core.image_env import ImageEnv
    from rlkit.core import logger
    from rlkit.envs.vae_wrappers import temporary_mode
    from rlkit.visualization.video import dump_video
    logdir = logger.get_snapshot_dir()
    save_period = variant.get('save_video_period', 50)
    do_state_exp = variant.get("do_state_exp", False)
    dump_video_kwargs = variant.get("dump_video_kwargs", dict())
    if do_state_exp:
        imsize = variant.get('imsize')
        dump_video_kwargs['imsize'] = imsize
        image_env = ImageEnv(
            env,
            imsize,
            init_camera=variant.get('init_camera', None),
            transpose=True,
            normalize=True,
        )

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                dump_video(image_env, policy, filename, rollout_function,
                           **dump_video_kwargs)
    else:
        image_env = env
        dump_video_kwargs['imsize'] = env.imsize

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_env',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
                filename = osp.join(logdir,
                                    'video_{epoch}_vae.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_vae',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
    return save_video
