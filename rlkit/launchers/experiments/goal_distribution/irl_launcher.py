import copy
from functools import partial

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
)
from rlkit.envs.contextual import ContextualEnv, delete_info
from rlkit.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    AddImageDistribution,
    GoalConditionedDiagnosticsToContextualDiagnostics,
    IndexIntoAchievedGoal,
)
from rlkit.torch.irl.irl_trainer import IRLTrainer, IRLRewardFn
from rlkit.envs.images import (
    EnvRenderer, InsertImagesEnv, GymEnvRenderer
)
from rlkit.envs.images.env_renderer import GymSimRenderer
from rlkit.launchers.contextual.rig.rig_launcher import get_gym_env
from rlkit.launchers.contextual.util import (
    get_save_video_function,
)
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from rlkit.torch.networks import ConcatMlp, Mlp
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, JointTrainer
from rlkit.util.hyperparameter import recursive_dictionary_update

from rlkit.util.io import (
    load_local_or_remote_file, sync_down_folder, get_absolute_path
)
from rlkit.envs.contextual.latent_distributions import (
    AddLatentDistribution,
    PriorDistribution,
)
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)

from rlkit.demos.source.dict_to_mdp_path_loader import (
    DictToMDPPathLoader,
    EncoderDictToMDPPathLoader
)
from rlkit.envs.make_env import make
from multiworld.core.gym_to_multi_env import GymToMultiEnv


ENV_PARAMS = {
    'HalfCheetah-v2': {
        'max_path_length': 1000,
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
    'Ant-v2': {
        'max_path_length': 1000,
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
    'Walker2d-v2': {
        'max_path_length': 1000,
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

    'SawyerRigGrasp-v0': {
        'env_id': 'SawyerRigGrasp-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 50,
        # 'num_epochs': 1000,
    },

    'pen-binary-v0': {
        'env_id': 'pen-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/pen2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_pen-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/pen_bc_sparse1.npy",
            # path="demos/icml2020/hand/pen_bc_sparse2.npy",
            # path="demos/icml2020/hand/pen_bc_sparse3.npy",
            # path="demos/icml2020/hand/pen_bc_sparse4.npy",
            path="demos/icml2020/hand/pen_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-binary-v0': {
        'env_id': 'door-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/door2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_door-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/door_bc_sparse1.npy",
            # path="demos/icml2020/hand/door_bc_sparse3.npy",
            path="demos/icml2020/hand/door_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-binary-v0': {
        'env_id': 'relocate-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/relocate2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_relocate-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/relocate_bc_sparse1.npy",
            path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
}

def process_args(variant):
    if variant.get("debug", False):
        variant["algo_kwargs"]=dict(
            num_epochs=5,
            batch_size=10,
            num_eval_steps_per_epoch=10,
            num_expl_steps_per_train_loop=10,
            num_trains_per_train_loop=10, #4000,
            min_num_steps_before_training=10,
            eval_epoch_freq=1,
        )

    env_id = variant.get("env_id", None)
    if env_id:
        env_params = ENV_PARAMS.get(env_id, {})
        recursive_dictionary_update(variant, env_params)

def irl_experiment(
        max_path_length,
        contextual_replay_buffer_kwargs,
        trainer_kwargs,
        algo_kwargs,
        qf_kwargs=None,
        policy_kwargs=None,
        # env settings
        env_id=None,
        env_class=None,
        env_kwargs=None,
        observation_key='latent_observation',
        context_keys=None,
        path_loader_class=EncoderDictToMDPPathLoader,
        path_loader_kwargs=None,
        renderer_kwargs=None,
        reward_trainer_kwargs=None,
        save_video=True,
        save_video_period=50,
        save_env_in_snapshot=True,
        dump_video_kwargs=None,
        # re-loading
        ckpt=None,
        ckpt_epoch=None,
        seedid=0,
        debug=False,
        normalize_env=False,
        add_env_demos=False,
        add_env_offpolicy_data=False,
        env_demo_path=None,
        env_offpolicy_data_path=None,
        score_fn_class=Mlp,
        score_fn_kwargs=None,
        use_oracle_reward=False,
        reward_fn_class=IRLRewardFn,
        reward_trainer_class=IRLTrainer,
):
    if renderer_kwargs is None:
        renderer_kwargs = {}
    if dump_video_kwargs is None:
        dump_video_kwargs = {}
    if policy_kwargs is None:
        policy_kwargs = {}
    if qf_kwargs is None:
        qf_kwargs = {}
    if reward_trainer_kwargs is None:
        reward_trainer_kwargs = {}
    if path_loader_kwargs is None:
        path_loader_kwargs = {}
    if score_fn_kwargs is None:
        score_fn_kwargs = {}
    if context_keys is None:
        context_keys = []
    if debug:
        algo_kwargs=dict(
            num_epochs=5,
            batch_size=10,
            num_eval_steps_per_epoch=10,
            num_expl_steps_per_train_loop=10,
            num_trains_per_train_loop=10, #4000,
            min_num_steps_before_training=10,
            eval_epoch_freq=1,
        )
        max_path_length=2

    env_params = ENV_PARAMS.get(env_id, {})
    if add_env_demos:
        demo_paths = path_loader_kwargs.setdefault("demo_paths", [])
        demo_paths.append(env_demo_path)
    if add_env_offpolicy_data:
        demo_paths = path_loader_kwargs.setdefault("demo_paths", [])
        demo_paths.append(env_offpolicy_data_path)

    reward_fn = reward_fn_class(None, context_keys)

    def contextual_env_distrib_and_reward(mode='expl'):
        assert mode in ['expl', 'eval']
        env = make(env_id, env_class, env_kwargs, normalize_env)
        env = GymToMultiEnv(env)
        # env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)

        no_goal_distribution = PriorDistribution(
            representation_size=0,
            key="no_goal",
        )
        contextual_reward_fn = None
        env = ContextualEnv(
            env,
            context_distribution=no_goal_distribution,
            reward_fn=contextual_reward_fn,
            observation_key=observation_key,
            # contextual_diagnostics_fns=[state_diag_fn],
            update_env_info_fn=None,
        )
        return env, no_goal_distribution, contextual_reward_fn

    env, context_distrib, _ = contextual_env_distrib_and_reward(
        mode='expl')
    eval_env, eval_context_distrib, _ = contextual_env_distrib_and_reward(
        mode='eval')

    keys = [observation_key] + context_keys
    obs_dim = sum(env.observation_space.spaces[key].low.size for key in keys)

    action_dim = env.action_space.low.size

    def create_qf():
        return ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )

    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **policy_kwargs
    )
    expl_policy = policy
    eval_policy = MakeDeterministic(policy)

    def concat_context_to_obs(batch, replay_buffer=None, obs_dict=None,
                              next_obs_dict=None, new_contexts=None):
        obs = batch['observations']
        next_obs = batch['next_observations']
        batch['observations'] = np.concatenate([obs, ] + context_keys, axis=1)
        batch['next_observations'] = np.concatenate([next_obs, ] + context_keys, axis=1)
        return batch

    if 'observation_keys' not in contextual_replay_buffer_kwargs:
        contextual_replay_buffer_kwargs['observation_keys'] = []
    observation_keys = contextual_replay_buffer_kwargs['observation_keys']
    if observation_key not in observation_keys:
        observation_keys.append(observation_key)

    replay_buffer_reward_fn = None if use_oracle_reward else reward_fn
    def create_replay_buffer():
        return ContextualRelabelingReplayBuffer(
            env=env,
            context_keys=context_keys,
            context_distribution=context_distrib,
            sample_context_from_obs_dict_fn=None,
            reward_fn=replay_buffer_reward_fn,
            post_process_batch_fn=concat_context_to_obs,
            **contextual_replay_buffer_kwargs
        )
    replay_buffer = create_replay_buffer()
    demo_train_buffer = create_replay_buffer()
    demo_test_buffer = create_replay_buffer()

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **trainer_kwargs
    )
    path_loader = path_loader_class(trainer,
        replay_buffer=replay_buffer,
        demo_train_buffer=demo_train_buffer,
        demo_test_buffer=demo_test_buffer,
        **path_loader_kwargs
    )
    path_loader.load_demos()

    score_fn = score_fn_class(
        input_size=obs_dim,
        output_size=1,
        **score_fn_kwargs
    )
    reward_fn.score_fn = score_fn

    vice_trainer = reward_trainer_class(
        score_fn,
        demo_train_buffer,
        policy,
        **reward_trainer_kwargs
    )

    def create_path_collector(
            env,
            policy,
            mode='expl',
    ):
        return ContextualPathCollector(
            env,
            policy,
            observation_key=observation_key,
            context_keys_for_policy=context_keys,
            save_env_in_snapshot=save_env_in_snapshot,
        )

    expl_path_collector = create_path_collector(env, expl_policy, mode='expl')
    eval_path_collector = create_path_collector(eval_env, eval_policy,
                                                mode='eval')

    joint_trainer = JointTrainer(dict(
        rl=trainer,
        reward=vice_trainer,
    ))

    algorithm = TorchBatchRLAlgorithm(
        trainer=joint_trainer,
        exploration_env=env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        # reward_trainer=vice_trainer,
        **algo_kwargs
    )
    algorithm.to(ptu.device)

    if save_video:
        # renderer = GymEnvRenderer(**renderer_kwargs)
        renderer = GymSimRenderer(**renderer_kwargs)
        # import ipdb; ipdb.set_trace()
        def add_images(env, state_distribution):
            state_env = env.env
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_distribution,
                image_goal_key='image_desired_goal',
                renderer=renderer,
            )
            img_env = InsertImagesEnv(state_env, renderers={
                'image_observation': renderer,
            })
            context_env = ContextualEnv(
                img_env,
                context_distribution=image_goal_distribution,
                reward_fn=reward_fn,
                observation_key=observation_key,
                update_env_info_fn=None,
            )
            return context_env

        # img_eval_env = add_images(eval_env, eval_context_distrib)
        img_eval_env = InsertImagesEnv(eval_env, renderers={
            'image_observation': renderer,
        })

        video_path_collector = create_path_collector(img_eval_env,
                                                     eval_policy,
                                                     mode='eval')
        rollout_function = video_path_collector._rollout_fn
        eval_video_func = get_save_video_function(
            rollout_function,
            img_eval_env,
            eval_policy,
            tag="eval",
            imsize=renderer_kwargs['width'],
            image_format='CHW',
            save_video_period=save_video_period,
            horizon=max_path_length,
            keys_to_show=["image_observation",],
            **dump_video_kwargs
        )
        algorithm.post_train_funcs.append(eval_video_func)

        # img_expl_env = add_images(env, context_distrib)
        img_expl_env = InsertImagesEnv(env, renderers={
            'image_observation': renderer,
        })
        video_path_collector = create_path_collector(img_expl_env,
                                                     expl_policy,
                                                     mode='expl')
        rollout_function = video_path_collector._rollout_fn
        expl_video_func = get_save_video_function(
            rollout_function,
            img_expl_env,
            expl_policy,
            tag="expl",
            imsize=renderer_kwargs['width'],
            image_format='CHW',
            save_video_period=save_video_period,
            horizon=max_path_length,
            keys_to_show=["image_observation",],
            **dump_video_kwargs
        )
        algorithm.post_train_funcs.append(expl_video_func)

    algorithm.train()
