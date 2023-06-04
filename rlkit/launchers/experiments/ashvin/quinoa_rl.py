import gym
from rlkit.data_management.awr_env_replay_buffer import AWREnvReplayBuffer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv, StackObservationEnv, RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.visualization.video import save_paths, VideoSaveFunction

from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.core.image_env import ImageEnv
from multiworld.core.gym_to_multi_env import GymToMultiEnv

from rlkit.launchers.experiments.ashvin.rfeatures.encoder_wrapped_env import EncoderWrappedEnv
from rlkit.launchers.experiments.ashvin.rfeatures.rfeatures_model import TimestepPredictionModel

import torch
import numpy as np
from torchvision.utils import save_image

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
from rlkit.util.io import load_local_or_remote_file
import pickle
from rlkit.torch.sac.quinoa import QuinoaTrainer

ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'HalfCheetah-v2'
    },
    'hopper': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Hopper-v2'
    },
    'humanoid': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Humanoid-v2'
    },
    'inv-double-pendulum': {  # 2 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'InvertedDoublePendulum-v2'
    },
    'pendulum': {  # 2 DoF
        'num_expl_steps_per_train_loop': 200,
        'max_path_length': 200,
        'min_num_steps_before_training': 2000,
        'target_update_period': 200,
        'env_id':'Pendulum-v2'
    },
    'ant': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Ant-v2'
    },
    'walker': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Walker2d-v2'
    },
    'swimmer': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Swimmer-v2'
    },

    'pen-v0': {
        'env_id': 'pen-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },
    'door-v0': {
        'env_id': 'door-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },
    'relocate-v0': {
        'env_id': 'relocate-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },
    'hammer-v0': {
        'env_id': 'hammer-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },

    'pen-sparse-v0': {
        'env_id': 'pen-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/pen2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/pen_bc_sparse1.npy",
            # path="demos/icml2020/hand/pen_bc_sparse2.npy",
            # path="demos/icml2020/hand/pen_bc_sparse3.npy",
            path="demos/icml2020/hand/pen_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-sparse-v0': {
        'env_id': 'door-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/door2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/door_bc_sparse1.npy",
            # path="demos/icml2020/hand/door_bc_sparse3.npy",
            path="demos/icml2020/hand/door_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-sparse-v0': {
        'env_id': 'relocate-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/relocate2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/relocate_bc_sparse1.npy",
            path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'hammer-sparse-v0': {
        'env_id': 'hammer-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/hammer2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/hammer_bc_sparse1.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
}

def compute_hand_sparse_reward(next_obs, reward, done, info):
    return info['goal_achieved'] - 1

def encoder_wrapped_env(variant):
    representation_size = 128
    output_classes = 20

    model_class = variant.get('model_class', TimestepPredictionModel)
    model = model_class(
        representation_size,
        # decoder_output_activation=decoder_activation,
        output_classes=output_classes,
        **variant['model_kwargs'],
    )
    # model = torch.nn.DataParallel(model)

    model_path = variant.get("model_path")
    # model = load_local_or_remote_file(model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(ptu.device)
    model.eval()

    traj = np.load(variant.get("desired_trajectory"), allow_pickle=True)[0]

    goal_image = traj["observations"][-1]["image_observation"]
    goal_image = goal_image.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
    # goal_image = goal_image.reshape(1, 300, 500, 3).transpose([0, 3, 1, 2]) / 255.0 # BECAUSE RLBENCH DEMOS ARENT IMAGE_ENV WRAPPED
    # goal_image = goal_image[:, :, :240, 60:500]
    goal_image = goal_image[:, :, 60:, 60:500]
    goal_image_pt = ptu.from_numpy(goal_image)
    save_image(goal_image_pt.data.cpu(), 'gitignore/goal.png', nrow=1)
    goal_latent = model.encode(goal_image_pt).detach().cpu().numpy().flatten()

    initial_image = traj["observations"][0]["image_observation"]
    initial_image = initial_image.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
    # initial_image = initial_image.reshape(1, 300, 500, 3).transpose([0, 3, 1, 2]) / 255.0
    # initial_image = initial_image[:, :, :240, 60:500]
    initial_image = initial_image[:, :, 60:, 60:500]
    initial_image_pt = ptu.from_numpy(initial_image)
    save_image(initial_image_pt.data.cpu(), 'gitignore/initial.png', nrow=1)
    initial_latent = model.encode(initial_image_pt).detach().cpu().numpy().flatten()

    # Move these to td3_bc and bc_v3 (or at least type for reward_params)
    reward_params = dict(
        goal_latent=goal_latent,
        initial_latent=initial_latent,
        type=variant["reward_params_type"],
    )

    config_params = variant.get("config_params")

    env = variant['env_class'](**variant['env_kwargs'])
    env = ImageEnv(env,
        recompute_reward=False,
        transpose=True,
        image_length=450000,
        reward_type="image_distance",
        # init_camera=sawyer_pusher_camera_upright_v2,
    )
    env = EncoderWrappedEnv(
        env,
        model,
        reward_params,
        config_params,
        **variant.get("encoder_wrapped_env_kwargs", dict())
    )
    env = FlatGoalEnv(env, obs_keys=["state_observation", ])

    return env

def resume(variant):
    data = load_local_or_remote_file(variant.get("pretrained_algorithm_path"), map_location="cuda")
    algo = data['algorithm']

    algo.num_epochs = variant['num_epochs']

    post_pretrain_hyperparams = variant["trainer_kwargs"].get("post_pretrain_hyperparams", {})
    algo.trainer.set_algorithm_weights(**post_pretrain_hyperparams)

    algo.train()

def process_args(variant):
    if variant.get("debug", False):
        variant['max_path_length'] = 50
        variant['batch_size'] = 5
        variant['num_epochs'] = 5
        variant['num_eval_steps_per_epoch'] = 100
        variant['num_expl_steps_per_train_loop'] = 100
        variant['num_trains_per_train_loop'] = 10
        variant['min_num_steps_before_training'] = 100
        variant['min_num_steps_before_training'] = 100
        variant['trainer_kwargs']['bc_num_pretrain_steps'] = 10
        variant['trainer_kwargs']['q_num_pretrain1_steps'] = 10
        variant['trainer_kwargs']['q_num_pretrain2_steps'] = 10

def experiment(variant):
    if variant.get("pretrained_algorithm_path", False):
        resume(variant)
        return

    if 'env' in variant:
        env_params = ENV_PARAMS[variant['env']]
        variant.update(env_params)

        if 'env_id' in env_params:
            if env_params['env_id'] in ['pen-v0', 'pen-sparse-v0', 'door-v0', 'relocate-v0', 'hammer-v0',
                                        'pen-sparse-v0', 'door-sparse-v0', 'relocate-sparse-v0', 'hammer-sparse-v0']:
                import mj_envs
            expl_env = gym.make(env_params['env_id'])
            eval_env = gym.make(env_params['env_id'])
        else:
            expl_env = NormalizedBoxEnv(variant['env_class']())
            eval_env = NormalizedBoxEnv(variant['env_class']())

        if variant.get('sparse_reward', False):
            expl_env = RewardWrapperEnv(expl_env, compute_hand_sparse_reward)
            eval_env = RewardWrapperEnv(eval_env, compute_hand_sparse_reward)

        if variant.get('add_env_demos', False):
            variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])

        if variant.get('add_env_offpolicy_data', False):
            variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])
    else:
        expl_env = encoder_wrapped_env(variant)
        eval_env = encoder_wrapped_env(variant)

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    stack_obs = path_loader_kwargs.get("stack_obs", 1)
    if stack_obs > 1:
        expl_env = StackObservationEnv(expl_env, stack_obs=stack_obs)
        eval_env = StackObservationEnv(eval_env, stack_obs=stack_obs)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

    M = variant['layer_size']
    vf_kwargs = variant.get("vf_kwargs", {})
    vf1 = ConcatMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M, M],
        **vf_kwargs
    )
    target_vf1 = ConcatMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M, M],
        **vf_kwargs
    )
    policy_class = variant.get("policy_class", TanhGaussianPolicy)
    policy_kwargs = variant['policy_kwargs']
    policy = policy_class(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **policy_kwargs,
    )
    target_policy = policy_class(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **policy_kwargs,
    )

    buffer_policy_class = variant.get("buffer_policy_class", policy_class)
    buffer_policy = buffer_policy_class(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant.get("buffer_policy_kwargs", policy_kwargs),
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    expl_policy = policy
    exploration_kwargs =  variant.get('exploration_kwargs', {})
    if exploration_kwargs:
        if exploration_kwargs.get("deterministic_exploration", False):
            expl_policy = MakeDeterministic(policy)

        exploration_strategy = exploration_kwargs.get("strategy", None)
        if exploration_strategy is None:
            pass
        elif exploration_strategy == 'ou':
            es = OUStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        elif exploration_strategy == 'gauss_eps':
            es = GaussianAndEpsilonStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],  # constant sigma
                epsilon=0,
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        else:
            error

    if variant.get('replay_buffer_class', EnvReplayBuffer) == AWREnvReplayBuffer:
        main_replay_buffer_kwargs = variant['replay_buffer_kwargs']
        main_replay_buffer_kwargs['env'] = expl_env
        main_replay_buffer_kwargs['qf1'] = qf1
        main_replay_buffer_kwargs['qf2'] = qf2
        main_replay_buffer_kwargs['policy'] = policy
    else:
        main_replay_buffer_kwargs=dict(
            max_replay_buffer_size=variant['replay_buffer_size'],
            env=expl_env,
        )
    replay_buffer_kwargs = dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )

    replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        **main_replay_buffer_kwargs,
    )
    if variant.get('use_validation_buffer', False):
        train_replay_buffer = replay_buffer
        validation_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
            **main_replay_buffer_kwargs,
        )
        replay_buffer = SplitReplayBuffer(train_replay_buffer, validation_replay_buffer, 0.9)

    trainer_class = variant.get("trainer_class", QuinoaTrainer)
    trainer = trainer_class(
        env=eval_env,
        policy=policy,
        vf1=vf1,
        target_policy=target_policy,
        target_vf1=target_vf1,
        buffer_policy=buffer_policy,
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            policy,
        )
        algorithm = TorchOnlineRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )
    else:
        expl_path_collector = MdpPathCollector(
            expl_env,
            expl_policy,
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )
    algorithm.to(ptu.device)

    demo_train_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )
    demo_test_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )

    if variant.get("save_video", False):
        if variant.get("presampled_goals", None):
            variant['image_env_kwargs']['presampled_goals'] = load_local_or_remote_file(variant['presampled_goals']).item()
        image_eval_env = ImageEnv(GymToMultiEnv(eval_env), **variant["image_env_kwargs"])
        image_eval_path_collector = ObsDictPathCollector(
            image_eval_env,
            eval_policy,
            observation_key="state_observation",
        )
        image_expl_env = ImageEnv(GymToMultiEnv(expl_env), **variant["image_env_kwargs"])
        image_expl_path_collector = ObsDictPathCollector(
            image_expl_env,
            expl_policy,
            observation_key="state_observation",
        )
        video_func = VideoSaveFunction(
            image_eval_env,
            variant,
            image_expl_path_collector,
            image_eval_path_collector,
        )
        algorithm.post_train_funcs.append(video_func)
    if variant.get('save_paths', False):
        algorithm.post_train_funcs.append(save_paths)
    if variant.get('load_demos', False):
        path_loader_class = variant.get('path_loader_class', MDPPathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos()
    if variant.get('save_initial_buffers', False):
        buffers = dict(
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
        )
        buffer_path = osp.join(logger.get_snapshot_dir(), 'buffers.p')
        pickle.dump(buffers, open(buffer_path, "wb"))
    if variant.get('pretrain_policy', False):
        trainer.pretrain_policy_with_bc()
    if variant.get('pretrain_rl', False):
        trainer.pretrain_q_with_bc_data()
    if variant.get('save_pretrained_algorithm', False):
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, "wb"))
        torch.save(data, open(p_path, "wb"))
    if variant.get('train_rl', True):
        algorithm.train()
