import pickle
import time

import cv2
import numpy as np
import os.path as osp

import rlkit.samplers.rollout_functions as rf
import rlkit.torch.pytorch_util as ptu
from multiworld.core.image_env import ImageEnv
from rlkit.core import logger
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.images.camera import (
    sawyer_init_camera_zoomed_in,
)
from rlkit.util.io import local_path_from_s3_or_local_path
from rlkit.util.ml_util import PiecewiseLinearSchedule
from rlkit.visualization.video import dump_video
from rlkit.torch.her.her_td3 import HerTd3
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer

from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.her.her_twin_sac import HerTwinSac
from rlkit.torch.her.her_sac import HerSac

def grill_her_td3_full_experiment(variant):
    grill_her_full_experiment(variant, mode='td3')

def grill_her_twin_sac_full_experiment(variant):
    grill_her_full_experiment(variant, mode='twin-sac')

def grill_her_sac_full_experiment(variant):
    grill_her_full_experiment(variant, mode='sac')

def grill_her_full_experiment(variant, mode='td3'):
    train_vae_variant = variant['train_vae_variant']
    grill_variant = variant['grill_variant']
    env_class = variant['env_class']
    env_kwargs = variant['env_kwargs']
    init_camera = variant['init_camera']
    train_vae_variant['generate_vae_dataset_kwargs']['env_class'] = env_class
    train_vae_variant['generate_vae_dataset_kwargs']['env_kwargs'] = env_kwargs
    train_vae_variant['generate_vae_dataset_kwargs']['init_camera'] = init_camera
    grill_variant['env_class'] = env_class
    grill_variant['env_kwargs'] = env_kwargs
    grill_variant['init_camera'] = init_camera
    if 'vae_paths' not in grill_variant:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae = train_vae(train_vae_variant)
        rdim = train_vae_variant['representation_size']
        vae_file = logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output(
            'vae_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        grill_variant['vae_paths'] = {
            str(rdim): vae_file,
        }
        grill_variant['rdim'] = str(rdim)
    if mode == 'td3':
        grill_her_td3_experiment(variant['grill_variant'])
    elif mode == 'twin-sac':
        grill_her_twin_sac_experiment(variant['grill_variant'])
    elif mode == 'sac':
        grill_her_sac_experiment(variant['grill_variant'])


def train_vae(variant):
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = generate_vae_dataset(
        **variant['generate_vae_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    m = ConvVAE(representation_size, input_channels=3)
    if ptu.gpu_enabled():
        m.to(ptu.device)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
            save_scatterplot=should_save_imgs,
            save_vae=False,
        )
        if should_save_imgs:
            t.dump_samples(epoch)
    return m


def generate_vae_dataset(
        env_class,
        N=10000,
        test_p=0.9,
        use_cached=True,
        imsize=84,
        show=False,
        init_camera=sawyer_init_camera_zoomed_in,
        dataset_path=None,
        env_kwargs=None,
        oracle_dataset=False,
        n_random_steps=100,
):
    if env_kwargs is None:
        env_kwargs = {}
    filename = "/tmp/{}_{}_{}_oracle{}.npy".format(
        env_class.__name__,
        str(N),
        init_camera.__name__,
        oracle_dataset,
    )
    info = {}
    if dataset_path is not None:
        filename = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(filename)
        N = dataset.shape[0]
    elif use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        now = time.time()
        env = env_class(**env_kwargs)
        env = ImageEnv(
            env,
            imsize,
            init_camera=init_camera,
            transpose=True,
            normalize=True,
        )
        env.reset()
        info['env'] = env

        dataset = np.zeros((N, imsize * imsize * 3))
        for i in range(N):
            if oracle_dataset:
                goal = env.sample_goal()
                env.set_to_goal(goal)
            else:
                env.reset()
                for _ in range(n_random_steps):
                    obs = env.step(env.action_space.sample())[0]
            obs = env.step(env.action_space.sample())[0]
            img = obs['image_observation']
            dataset[i, :] = img
            if show:
                img = img.reshape(3, 84, 84).transpose()
                img = img[::-1, :, ::-1]
                cv2.imshow('img', img)
                cv2.waitKey(1)
                # radius = input('waiting...')
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


def grill_her_td3_experiment(variant):
    env = variant["env_class"](**variant['env_kwargs'])

    render = variant["render"]

    rdim = variant["rdim"]
    vae_path = variant["vae_paths"][str(rdim)]
    reward_params = variant.get("reward_params", dict())

    init_camera = variant.get("init_camera", None)
    if init_camera is None:
        camera_name = "topview"
    else:
        camera_name = None

    env = ImageEnv(
        env,
        84,
        init_camera=init_camera,
        camera_name=camera_name,
        transpose=True,
        normalize=True,
    )

    env = VAEWrappedEnv(
        env,
        vae_path,
        decode_goals=render,
        render_goals=render,
        render_rollouts=render,
        reward_params=reward_params,
        **variant.get('vae_wrapped_env_kwargs', {})
    )

    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    exploration_noise = variant.get('exploration_noise', 0.1)
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
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
    else:
        raise Exception("Invalid type: " + exploration_type)
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
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    training_mode = variant.get("training_mode", "train")
    testing_mode = variant.get("testing_mode", "test")

    testing_env = pickle.loads(pickle.dumps(env))
    testing_env.mode(testing_mode)

    training_env = pickle.loads(pickle.dumps(env))
    training_env.mode(training_mode)

    relabeling_env = pickle.loads(pickle.dumps(env))
    relabeling_env.mode(training_mode)
    relabeling_env.disable_render()

    video_vae_env = pickle.loads(pickle.dumps(env))
    video_vae_env.mode("video_vae")
    video_goal_env = pickle.loads(pickle.dumps(env))
    video_goal_env.mode("video_env")


    replay_buffer = ObsDictRelabelingBuffer(
        env=relabeling_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    algorithm = HerTd3(
        testing_env,
        training_env=training_env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        render=render,
        render_during_eval=render,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['algo_kwargs']
    )

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.to(ptu.device)
        for e in [testing_env, training_env, video_vae_env, video_goal_env]:
            e.vae.to(ptu.device)

    algorithm.train()

    if variant.get("save_video", True):
        logdir = logger.get_snapshot_dir()
        policy.train(False)
        filename = osp.join(logdir, 'video_final_env.mp4')
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        dump_video(video_goal_env, policy, filename, rollout_function)
        filename = osp.join(logdir, 'video_final_vae.mp4')
        dump_video(video_vae_env, policy, filename, rollout_function)

def grill_her_twin_sac_experiment(variant):
    env = variant["env_class"](**variant['env_kwargs'])

    render = variant["render"]

    rdim = variant["rdim"]
    vae_path = variant["vae_paths"][str(rdim)]
    reward_params = variant.get("reward_params", dict())

    init_camera = variant.get("init_camera", None)
    if init_camera is None:
        camera_name = "topview"
    else:
        camera_name = None

    env = ImageEnv(
        env,
        84,
        init_camera=init_camera,
        camera_name=camera_name,
        transpose=True,
        normalize=True,
    )

    env = VAEWrappedEnv(
        env,
        vae_path,
        decode_goals=render,
        render_goals=render,
        render_rollouts=render,
        reward_params=reward_params,
        **variant.get('vae_wrapped_env_kwargs', {})
    )

    if variant['normalize']:
        env = NormalizedBoxEnv(env)
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
    vf = ConcatMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    training_mode = variant.get("training_mode", "train")
    testing_mode = variant.get("testing_mode", "test")

    testing_env = pickle.loads(pickle.dumps(env))
    testing_env.mode(testing_mode)

    training_env = pickle.loads(pickle.dumps(env))
    training_env.mode(training_mode)

    relabeling_env = pickle.loads(pickle.dumps(env))
    relabeling_env.mode(training_mode)
    relabeling_env.disable_render()

    video_vae_env = pickle.loads(pickle.dumps(env))
    video_vae_env.mode("video_vae")
    video_goal_env = pickle.loads(pickle.dumps(env))
    video_goal_env.mode("video_env")


    replay_buffer = ObsDictRelabelingBuffer(
        env=relabeling_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    algorithm = HerTwinSac(
        testing_env,
        training_env=training_env,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        policy=policy,
        render=render,
        render_during_eval=render,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['algo_kwargs']
    )

    if ptu.gpu_enabled():
        print("using GPU")
        qf1.to(ptu.device)
        qf2.to(ptu.device)
        vf.to(ptu.device)
        policy.to(ptu.device)
        algorithm.to(ptu.device)
        for e in [testing_env, training_env, video_vae_env, video_goal_env]:
            e.vae.to(ptu.device)

    algorithm.train()

    if variant.get("save_video", True):
        logdir = logger.get_snapshot_dir()
        policy.train(False)
        filename = osp.join(logdir, 'video_final_env.mp4')
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        dump_video(video_goal_env, policy, filename, rollout_function)
        filename = osp.join(logdir, 'video_final_vae.mp4')
        dump_video(video_vae_env, policy, filename, rollout_function)

def grill_her_sac_experiment(variant):
    env = variant["env_class"](**variant['env_kwargs'])

    render = variant["render"]

    rdim = variant["rdim"]
    vae_path = variant["vae_paths"][str(rdim)]
    reward_params = variant.get("reward_params", dict())

    init_camera = variant.get("init_camera", None)
    if init_camera is None:
        camera_name = "topview"
    else:
        camera_name = None

    env = ImageEnv(
        env,
        84,
        init_camera=init_camera,
        camera_name=camera_name,
        transpose=True,
        normalize=True,
    )

    env = VAEWrappedEnv(
        env,
        vae_path,
        decode_goals=render,
        render_goals=render,
        render_rollouts=render,
        reward_params=reward_params,
        **variant.get('vae_wrapped_env_kwargs', {})
    )

    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
        + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    vf = ConcatMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    training_mode = variant.get("training_mode", "train")
    testing_mode = variant.get("testing_mode", "test")

    testing_env = pickle.loads(pickle.dumps(env))
    testing_env.mode(testing_mode)

    training_env = pickle.loads(pickle.dumps(env))
    training_env.mode(training_mode)

    relabeling_env = pickle.loads(pickle.dumps(env))
    relabeling_env.mode(training_mode)
    relabeling_env.disable_render()

    video_vae_env = pickle.loads(pickle.dumps(env))
    video_vae_env.mode("video_vae")
    video_goal_env = pickle.loads(pickle.dumps(env))
    video_goal_env.mode("video_env")


    replay_buffer = ObsDictRelabelingBuffer(
        env=relabeling_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    algorithm = HerSac(
        testing_env,
        training_env=training_env,
        qf=qf,
        vf=vf,
        policy=policy,
        render=render,
        render_during_eval=render,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['algo_kwargs']
    )

    if ptu.gpu_enabled():
        print("using GPU")
        qf.to(ptu.device)
        vf.to(ptu.device)
        policy.to(ptu.device)
        algorithm.to(ptu.device)
        for e in [testing_env, training_env, video_vae_env, video_goal_env]:
            e.vae.to(ptu.device)

    algorithm.train()

    if variant.get("save_video", True):
        logdir = logger.get_snapshot_dir()
        policy.train(False)
        filename = osp.join(logdir, 'video_final_env.mp4')
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        dump_video(video_goal_env, policy, filename, rollout_function)
        filename = osp.join(logdir, 'video_final_vae.mp4')
        dump_video(video_vae_env, policy, filename, rollout_function)
