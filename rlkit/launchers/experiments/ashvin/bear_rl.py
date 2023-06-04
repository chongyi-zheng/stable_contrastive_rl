import gym
from rlkit.data_management.awr_env_replay_buffer import AWREnvReplayBuffer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv, StackObservationEnv, RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, PolicyFromQ
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
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

import copy
import torch.nn as nn
from rlkit.samplers.data_collector import MdpPathCollector # , CustomMdpPathCollector
from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.torch.networks import MlpQf, TanhMlpPolicy
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.lvm.bear_vae import VAEPolicy
from rlkit.torch.sac.bear import BEARTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'env_id':'HalfCheetah-v2',
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'HalfCheetah-v2',
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/hc_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/hc_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'ant': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Ant-v2',
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/ant_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/ant_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'walker': {  # 6 DoF
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_id':'Walker2d-v2',
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/walker_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/walker_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
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
        'env_id': 'pen-binary-v0',
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
        'env_id': 'door-binary-v0',
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
        'env_id': 'relocate-binary-v0',
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
        'env_id': 'hammer-binary-v0',
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
        variant['trainer_kwargs']['num_pretrain_steps'] = min(10, variant['trainer_kwargs'].get('num_pretrain_steps', 0))


def experiment(variant):
    import mj_envs

    env_params = ENV_PARAMS.get(variant.get('env'), {})
    variant.update(env_params)
    env_name = variant.get("env", None)
    env_id = variant.get('env_id', None)
    env_class = variant.get('env_class', None)

    if env_name in [
        'pen-v0', 'pen-sparse-v0', 'pen-notermination-v0', 'pen-binary-v0',
        'door-v0', 'door-sparse-v0', 'door-binary-v0',
        'relocate-v0', 'relocate-sparse-v0', 'relocate-binary-v0',
        'hammer-v0', 'hammer-sparse-v0', 'hammer-binary-v0',
    ]:
        import mj_envs
        expl_env = gym.make(env_params.get('env_id', env_name))
        eval_env = gym.make(env_params.get('env_id', env_name))
    elif env_name in [ # D4RL envs
        "maze2d-open-v0", "maze2d-umaze-v0", "maze2d-medium-v0", "maze2d-large-v0",
        "maze2d-open-dense-v0", "maze2d-umaze-dense-v0", "maze2d-medium-dense-v0", "maze2d-large-dense-v0",
        "antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-diverse-v0",
        "antmaze-medium-play-v0", "antmaze-large-diverse-v0", "antmaze-large-play-v0",
        "pen-human-v0", "pen-cloned-v0", "pen-expert-v0", "hammer-human-v0", "hammer-cloned-v0", "hammer-expert-v0",
        "door-human-v0", "door-cloned-v0", "door-expert-v0", "relocate-human-v0", "relocate-cloned-v0", "relocate-expert-v0",
        "halfcheetah-random-v0", "halfcheetah-medium-v0", "halfcheetah-expert-v0", "halfcheetah-mixed-v0", "halfcheetah-medium-expert-v0",
        "walker2d-random-v0", "walker2d-medium-v0", "walker2d-expert-v0", "walker2d-mixed-v0", "walker2d-medium-expert-v0",
        "hopper-random-v0", "hopper-medium-v0", "hopper-expert-v0", "hopper-mixed-v0", "hopper-medium-expert-v0"
    ]:
        import d4rl
        expl_env = gym.make(env_name)
        eval_env = gym.make(env_name)
    elif env_id:
        expl_env = NormalizedBoxEnv(gym.make(env_id))
        eval_env = NormalizedBoxEnv(gym.make(env_id))
    elif env_class:
        expl_env = NormalizedBoxEnv(env_class())
        eval_env = NormalizedBoxEnv(env_class())
    else:
        expl_env = NormalizedBoxEnv(variant['env']())
        eval_env = NormalizedBoxEnv(variant['env']())

    # expl_env = gym.make(variant['env'])
    # eval_env = gym.make(variant['env'])

    if variant.get('add_env_demos', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])

    if variant.get('add_env_offpolicy_data', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

    action_dim = int(np.prod(eval_env.action_space.shape))
    state_dim = obs_dim = np.prod(expl_env.observation_space.shape)
    M = 256

    qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
    qf_kwargs['output_size'] = 1
    qf_kwargs['input_size'] = action_dim + state_dim
    qf1 = MlpQf(**qf_kwargs)
    qf2 = MlpQf(**qf_kwargs)

    target_qf_kwargs = copy.deepcopy(qf_kwargs)
    target_qf1 = MlpQf(**target_qf_kwargs)
    target_qf2 = MlpQf(**target_qf_kwargs)

    policy_class = variant.get("policy_class", TanhGaussianPolicy)
    policy_kwargs = copy.deepcopy(variant['policy_kwargs'])
    policy_kwargs['action_dim'] = action_dim
    policy_kwargs['obs_dim'] = state_dim
    policy = policy_class(**policy_kwargs)

    vae_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=action_dim * 2,
    )

    # vae_eval_path_collector = MdpPathCollector(
    #     eval_env,
    #     vae_policy,
    #     # max_num_epoch_paths_saved=5,
    #     # save_images=False,
    # )


    replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )
    demo_train_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )
    demo_test_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )

    trainer_class = variant.get("trainer_class", BEARTrainer)
    trainer = trainer_class(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae=vae_policy,
        replay_buffer=replay_buffer,
        **variant['trainer_kwargs']
    )

    expl_path_collector = MdpPathCollector(
        expl_env,
        # policy,
        trainer.eval_policy, # PolicyFromQ(qf1, policy),
        **variant['expl_path_collector_kwargs']
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        # save_images=False,
        # MakeDeterministic(policy),
        trainer.eval_policy, # PolicyFromQ(qf1, policy),
        **variant['eval_path_collector_kwargs']
    )

    path_loader_class = variant.get('path_loader_class', MDPPathLoader)
    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    path_loader = path_loader_class(trainer,
                                    replay_buffer=replay_buffer,
                                    demo_train_buffer=demo_train_replay_buffer,
                                    demo_test_buffer=demo_test_replay_buffer,
                                    **path_loader_kwargs,
                                    # demo_off_policy_path=variant['data_path'],
                                    )
    # path_loader.load_bear_demos(pickled=False)
    path_loader.load_demos()
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        # vae_evaluation_data_collector=vae_eval_path_collector,
        replay_buffer=replay_buffer,
        # q_learning_alg=True,
        # batch_rl=variant['batch_rl'],
        **variant['algo_kwargs']
    )


    algorithm.to(ptu.device)
    trainer.pretrain_q_with_bc_data(256)
    algorithm.train()
