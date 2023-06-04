from rlkit.envs.wrappers import StackObservationEnv, RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.samplers.data_collector.path_collector import GoalConditionedPathCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.networks.cnn import ConcatCNN
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.util.io import load_local_or_remote_file
from rlkit.envs.encoder_wrappers import VQVAEWrappedEnv
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.visualization.video import save_paths
import numpy as np
import torch
from rlkit.visualization.video import save_paths, VideoSaveFunction, RIGVideoSaveFunction
from rlkit.envs.images import Renderer, InsertImageEnv, EnvRenderer
from rlkit.launchers.contextual.util import (
    get_save_video_function,
    get_gym_env,
)

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
from rlkit.util.io import load_local_or_remote_file
from rlkit.launchers.contextual.rig.rig_launcher import StateImageGoalDiagnosticsFn
from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
from rlkit.data_management.wrappers.concat_to_obs_wrapper import \
        ConcatToObsWrapper
from rlkit.envs.reward_mask_wrapper import DiscreteDistribution, RewardMaskWrapper

from functools import partial
import rlkit.samplers.rollout_functions as rf

from rlkit.envs.contextual import ContextualEnv
import pickle

from rlkit.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    AddImageDistribution,
    GoalConditionedDiagnosticsToContextualDiagnostics,
    IndexIntoAchievedGoal,
)


from torch.utils import data
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from rlkit.envs.contextual import ContextualEnv
from rlkit.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    AddImageDistribution,
    PresampledPathDistribution,
)
from rlkit.envs.contextual.latent_distributions import (
    AddLatentDistribution,
    PriorDistribution,
)
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.core.eval_util import create_stats_ordered_dict
from collections import OrderedDict

from multiworld.core.image_env import ImageEnv, unormalize_image
import multiworld

from rlkit.launchers.contextual.rig.model_train_launcher import train_vae

def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = pickle.load(open(local_path, "rb"))
    # vae = torch.load(local_path, map_location='cpu')
    print("loaded", local_path)
    vae.to("cuda")
    return vae

def compute_hand_sparse_reward(next_obs, reward, done, info):
    return info['goal_achieved'] - 1

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
        variant['trainer_kwargs']['bc_num_pretrain_steps'] = min(10, variant['trainer_kwargs'].get('bc_num_pretrain_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain1_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain1_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain2_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain2_steps', 0))

def experiment(variant):
    render = variant.get("render", False)
    debug = variant.get("debug", False)
    vae_path = variant.get("vae_path", False)

    process_args(variant)

    env_class = variant.get("env_class")
    env_kwargs = variant.get("env_kwargs")
    env_id = variant.get("env_id")
    # expl_env = env_class(**env_kwargs)
    # eval_env = env_class(**env_kwargs)
    expl_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
    eval_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
    env = eval_env

    if variant.get('sparse_reward', False):
        expl_env = RewardWrapperEnv(expl_env, compute_hand_sparse_reward)
        eval_env = RewardWrapperEnv(eval_env, compute_hand_sparse_reward)

    if variant.get("vae_path", False):
        vae = load_local_or_remote_file(vae_path)
        variant['path_loader_kwargs']['model_path'] = vae_path
        renderer = EnvRenderer(**variant.get("renderer_kwargs", {}))
        expl_env = VQVAEWrappedEnv(InsertImageEnv(expl_env, renderer=renderer), vae,
                reward_params=variant.get("reward_params", {}),
                **variant.get('vae_wrapped_env_kwargs', {}))
        eval_env = VQVAEWrappedEnv(InsertImageEnv(eval_env, renderer=renderer), vae,
                reward_params=variant.get("reward_params", {}),
                **variant.get('vae_wrapped_env_kwargs', {}))
        env = eval_env
        variant['path_loader_kwargs']['env'] = env

    if variant.get('add_env_demos', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])

    if variant.get('add_env_offpolicy_data', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

    if variant.get("use_masks", False):
        mask_wrapper_kwargs = variant.get("mask_wrapper_kwargs", dict())

        expl_mask_distribution_kwargs = variant["expl_mask_distribution_kwargs"]
        expl_mask_distribution = DiscreteDistribution(**expl_mask_distribution_kwargs)
        expl_env = RewardMaskWrapper(env, expl_mask_distribution, **mask_wrapper_kwargs)

        eval_mask_distribution_kwargs = variant["eval_mask_distribution_kwargs"]
        eval_mask_distribution = DiscreteDistribution(**eval_mask_distribution_kwargs)
        eval_env = RewardMaskWrapper(env, eval_mask_distribution, **mask_wrapper_kwargs)
        env = eval_env

    if variant.get("pretrained_algorithm_path", False):
        resume(variant)
        return

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    stack_obs = path_loader_kwargs.get("stack_obs", 1)
    if stack_obs > 1:
        expl_env = StackObservationEnv(expl_env, stack_obs=stack_obs)
        eval_env = StackObservationEnv(eval_env, stack_obs=stack_obs)

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = variant.get('achieved_goal_key', 'latent_achieved_goal')

    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = eval_env.action_space.low.size

    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

    replay_buffer_kwargs=dict(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
    )
    replay_buffer_kwargs.update(variant.get('replay_buffer_kwargs', dict()))
    replay_buffer = ConcatToObsWrapper(
        ObsDictRelabelingBuffer(**replay_buffer_kwargs),
        ["resampled_goals", ],
    )
    replay_buffer_kwargs.update(variant.get('demo_replay_buffer_kwargs', dict()))
    demo_train_buffer = ConcatToObsWrapper(
        ObsDictRelabelingBuffer(**replay_buffer_kwargs),
        ["resampled_goals", ],
    )
    demo_test_buffer = ConcatToObsWrapper(
        ObsDictRelabelingBuffer(**replay_buffer_kwargs),
        ["resampled_goals", ],
    )

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy_class = variant.get("policy_class", TanhGaussianPolicy)
    policy = policy_class(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs'],
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

    trainer = AWACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
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
        eval_path_collector = GoalConditionedPathCollector(
            eval_env,
            MakeDeterministic(policy),
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            render=render,
            goal_sampling_mode=variant.get("goal_sampling_mode", None),
        )
        expl_path_collector = GoalConditionedPathCollector(
            expl_env,
            policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            render=render,
            goal_sampling_mode=variant.get("goal_sampling_mode", None),
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

    if variant.get("save_video", True):
        video_func = VideoSaveFunction(
            env,
            variant,
        )
        #algorithm.post_train_funcs.append(video_func)
        algorithm.post_train_funcs.append(video_func)

    # if variant.get("save_video", False):
    #     from rlkit.visualization.video import VideoSaveFunction
    #     renderer_kwargs = variant.get("renderer_kwargs", {})
    #     save_video_kwargs = variant.get("save_video_kwargs", {})

    #     def get_video_func(
    #         env,
    #         policy,
    #         tag,
    #     ):
    #         renderer = EnvRenderer(**renderer_kwargs)
    #         state_goal_distribution = GoalDictDistributionFromMultitaskEnv(
    #             env,
    #             desired_goal_keys=[desired_goal_key],
    #         )
    #         image_goal_distribution = AddImageDistribution(
    #             env=env,
    #             base_distribution=state_goal_distribution,
    #             image_goal_key='image_desired_goal',
    #             renderer=renderer,
    #         )
    #         img_env = InsertImageEnv(env, renderer=renderer)
    #         rollout_function = partial(
    #             rf.multitask_rollout,
    #             max_path_length=variant['max_path_length'],
    #             observation_key=observation_key,
    #             desired_goal_key=desired_goal_key,
    #             return_dict_obs=True,
    #         )
    #         reward_fn = ContextualRewardFnFromMultitaskEnv(
    #             env=env,
    #             achieved_goal_from_observation=IndexIntoAchievedGoal(observation_key),
    #             desired_goal_key=desired_goal_key,
    #             achieved_goal_key="state_achieved_goal",
    #         )
    #         contextual_env = ContextualEnv(
    #             img_env,
    #             context_distribution=image_goal_distribution,
    #             reward_fn=reward_fn,
    #             observation_key=observation_key,
    #         )
    #         video_func = get_save_video_function(
    #             rollout_function,
    #             contextual_env,
    #             policy,
    #             tag=tag,
    #             imsize=renderer.width,
    #             image_format='CWH',
    #             **save_video_kwargs
    #         )
    #         return video_func
        # expl_video_func = get_video_func(expl_env, expl_policy, "expl")
        # eval_video_func = get_video_func(eval_env, MakeDeterministic(policy), "eval")
        # algorithm.post_train_funcs.append(eval_video_func)
        # algorithm.post_train_funcs.append(expl_video_func)

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
    if variant.get('pretrain_policy', False):
        trainer.pretrain_policy_with_bc(
            policy,
            demo_train_buffer,
            demo_test_buffer,
            trainer.bc_num_pretrain_steps,
        )
    if variant.get('pretrain_rl', False):
        trainer.pretrain_q_with_bc_data()

    if variant.get('save_pretrained_algorithm', False):
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, "wb"))
        torch.save(data, open(p_path, "wb"))

    algorithm.train()



class RewardFn:
    def __init__(self,
            env,
            observation_key,
            desired_goal_key,
            reward_type='dense',
            epsilon=1.0
            ):
        self.env = env
        self.reward_type = reward_type
        self.epsilon = epsilon
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key

    def process(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def __call__(self, states, actions, next_states, contexts):
        s = next_states[self.observation_key]
        c = contexts[self.desired_goal_key]

        if self.reward_type == 'dense':
            reward = -np.linalg.norm(s - c, axis=1)
        elif self.reward_type == 'sparse':
            success = np.linalg.norm(s - c, axis=1) < self.epsilon
            reward = success - 1
        elif self.reward_type == 'wrapped_env':
            reward = self.env.compute_reward(states, actions, next_states, contexts)
        else:
            raise ValueError(self.reward_type)
        return reward


class GraspingRewardFn:
    def __call__(self, states, actions, next_states, contexts):
        height = next_states['state_observation'][:, 10]
        # reward = (height + 0.345) / 0.16 - 1
        reward = (height > -0.3) - 1
        return reward


def awac_rig_experiment(
        max_path_length,
        qf_kwargs,
        trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        train_vae_kwargs,
        policy_class=TanhGaussianPolicy,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        reward_kwargs=None,
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        state_observation_key='state_observation',
        state_goal_key='state_desired_goal',
        image_goal_key='image_desired_goal',

        path_loader_class=MDPPathLoader,
        demo_replay_buffer_kwargs=None,
        path_loader_kwargs=None,
        env_demo_path='',
        env_offpolicy_data_path='',

        debug=False,
        epsilon=1.0,
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,

        add_env_demos=False,
        add_env_offpolicy_data=False,
        save_paths=False,
        load_demos=False,
        pretrain_policy=False,
        pretrain_rl=False,
        save_pretrained_algorithm=False,

        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
        imsize=84,
        pretrained_vae_path="",
        presampled_goals_path="",
        init_camera=None,

        qf_class=ConcatMlp,
    ):

    #Kwarg Definitions
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if demo_replay_buffer_kwargs is None:
        demo_replay_buffer_kwargs = {}
    if path_loader_kwargs is None:
        path_loader_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}

    if debug:
        max_path_length = 5
        algo_kwargs['batch_size'] = 5
        algo_kwargs['num_epochs'] = 5
        algo_kwargs['num_eval_steps_per_epoch'] = 100
        algo_kwargs['num_expl_steps_per_train_loop'] = 100
        algo_kwargs['num_trains_per_train_loop'] = 10
        algo_kwargs['min_num_steps_before_training'] = 100
        algo_kwargs['min_num_steps_before_training'] = 100
        trainer_kwargs['bc_num_pretrain_steps'] = min(10, trainer_kwargs.get('bc_num_pretrain_steps', 0))
        trainer_kwargs['q_num_pretrain1_steps'] = min(10, trainer_kwargs.get('q_num_pretrain1_steps', 0))
        trainer_kwargs['q_num_pretrain2_steps'] = min(10, trainer_kwargs.get('q_num_pretrain2_steps', 0))

    #Enviorment Wrapping
    renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)
    def contextual_env_distrib_and_reward(
            env_id, env_class, env_kwargs, goal_sampling_mode, presampled_goals_path
    ):
        state_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)
        img_env = InsertImageEnv(state_env, renderer=renderer)

        # encoded_env = EncoderWrappedEnv(
        #     img_env,
        #     model,
        #     dict(image_observation="latent_observation", ),
        # )
        # if goal_sampling_mode == "vae_prior":
        #     latent_goal_distribution = PriorDistribution(
        #         model.representation_size,
        #         desired_goal_key,
        #     )
        #     diagnostics = StateImageGoalDiagnosticsFn({}, )
        # elif goal_sampling_mode == "presampled":
        #     diagnostics = state_env.get_contextual_diagnostics
        #     image_goal_distribution = PresampledPathDistribution(
        #         presampled_goals_path,
        #     )

        #     latent_goal_distribution = AddLatentDistribution(
        #         image_goal_distribution,
        #         image_goal_key,
        #         desired_goal_key,
        #         model,
        #     )
        # elif goal_sampling_mode == "reset_of_env":
        #     state_goal_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        #     state_goal_distribution = GoalDictDistributionFromMultitaskEnv(
        #         state_goal_env,
        #         desired_goal_keys=[state_goal_key],
        #     )
        #     image_goal_distribution = AddImageDistribution(
        #         env=state_env,
        #         base_distribution=state_goal_distribution,
        #         image_goal_key=image_goal_key,
        #         renderer=renderer,
        #     )
        #     latent_goal_distribution = AddLatentDistribution(
        #         image_goal_distribution,
        #         image_goal_key,
        #         desired_goal_key,
        #         model,
        #     )
        #     no_goal_distribution = PriorDistribution(
        #         representation_size=0,
        #         key="no_goal",
        #     )
        #     diagnostics = state_goal_env.get_contextual_diagnostics
        # else:
        #     error
        diagnostics = StateImageGoalDiagnosticsFn({}, )
        no_goal_distribution = PriorDistribution(
            representation_size=0,
            key="no_goal",
        )

        reward_fn = GraspingRewardFn(
            # img_env, # state_env,
            # observation_key=observation_key,
            # desired_goal_key=desired_goal_key,
            # **reward_kwargs
        )

        env = ContextualEnv(
            img_env, # state_env,
            context_distribution=no_goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=[diagnostics],
        )
        return env, no_goal_distribution, reward_fn

    #VAE Setup
    if pretrained_vae_path:
        model = load_local_or_remote_file(pretrained_vae_path)
    else:
        model = train_vae(train_vae_kwargs, env_kwargs, env_id, env_class, imsize, init_camera)
    path_loader_kwargs['model_path'] = pretrained_vae_path

    #Enviorment Definitions
    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        env_id, env_class, env_kwargs, exploration_goal_sampling_mode, presampled_goals_path
    )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        env_id, env_class, env_kwargs, evaluation_goal_sampling_mode, presampled_goals_path
    )
    path_loader_kwargs['env'] = eval_env

    #AWAC Code
    if add_env_demos:
        path_loader_kwargs["demo_paths"].append(env_demo_path)
    if add_env_offpolicy_data:
        path_loader_kwargs["demo_paths"].append(env_offpolicy_data_path)


    #Key Setting
    context_key = desired_goal_key
    obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[context_key].low.size
    )
    action_dim = expl_env.action_space.low.size

    state_rewards = reward_kwargs.get('reward_type', 'dense') == 'wrapped_env'
    # if state_rewards:
    #     mapper = RemapKeyFn({context_key: observation_key, state_goal_key: state_observation_key})
    #     obs_keys = [state_observation_key, observation_key]
    #     cont_keys = [state_goal_key, context_key]
    # else:
    mapper = RemapKeyFn({context_key: observation_key})
    obs_keys = [observation_key]
    cont_keys = [context_key]

    #Replay Buffer
    def concat_context_to_obs(batch, replay_buffer, obs_dict, next_obs_dict, new_contexts):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch
    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=cont_keys,
        observation_keys_to_save=obs_keys,
        observation_key=observation_key,
        context_distribution=expl_context_distrib,
        sample_context_from_obs_dict_fn=mapper,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs
    )
    replay_buffer_kwargs.update(demo_replay_buffer_kwargs)
    demo_train_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=cont_keys,
        observation_keys_to_save=obs_keys,
        observation_key=observation_key,
        context_distribution=expl_context_distrib,
        sample_context_from_obs_dict_fn=mapper,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs
    )
    demo_test_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=cont_keys,
        observation_keys_to_save=obs_keys,
        observation_key=observation_key,
        context_distribution=expl_context_distrib,
        sample_context_from_obs_dict_fn=mapper,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs
    )


    #Neural Network Architecture
    def create_qf():
        # return ConcatMlp(
        #     input_size=obs_dim + action_dim,
        #     output_size=1,
        #     **qf_kwargs
        # )
        if qf_class is ConcatMlp:
            qf_kwargs["input_size"] = obs_dim + action_dim
        if qf_class is ConcatCNN:
            qf_kwargs["added_fc_input_size"] = action_dim
        return qf_class(
            output_size=1,
            **qf_kwargs
        )
    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()

    policy = policy_class(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **policy_kwargs,
    )

    #Path Collectors
    eval_path_collector = ContextualPathCollector(
        eval_env,
        MakeDeterministic(policy),
        observation_key=observation_key,
        context_keys_for_policy=[context_key, ],
    )
    exploration_policy = create_exploration_policy(
        expl_env, policy, **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key,
        context_keys_for_policy=[context_key, ],
    )

    #Algorithm
    trainer = AWACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **trainer_kwargs
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs
    )


    algorithm.to(ptu.device)

    #Video Saving
    if save_video:

        expl_video_func = RIGVideoSaveFunction(
            model,
            expl_path_collector,
            "train",
            # decode_goal_image_key="image_decoded_goal",
            # reconstruction_key="image_reconstruction",
            rows=2,
            columns=5,
            unnormalize=True,
            imsize=imsize,
            image_format=renderer.output_image_format,
            **save_video_kwargs
        )
        algorithm.post_train_funcs.append(expl_video_func)

        eval_video_func = RIGVideoSaveFunction(
            model,
            eval_path_collector,
            "eval",
            # goal_image_key=image_goal_key,
            # decode_goal_image_key="image_decoded_goal",
            # reconstruction_key="image_reconstruction",
            num_imgs=4,
            rows=2,
            columns=5,
            unnormalize=True,
            imsize=imsize,
            image_format=renderer.output_image_format,
            **save_video_kwargs
        )
        algorithm.post_train_funcs.append(eval_video_func)

    #AWAC CODE
    if save_paths:
        algorithm.post_train_funcs.append(save_paths)

    if load_demos:
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            # reward_fn=eval_reward, # omit reward because its recomputed later
            **path_loader_kwargs
        )
        path_loader.load_demos()
    if pretrain_policy:
        trainer.pretrain_policy_with_bc(
            policy,
            demo_train_buffer,
            demo_test_buffer,
            trainer.bc_num_pretrain_steps,
        )
    if pretrain_rl:
        trainer.pretrain_q_with_bc_data()

    if save_pretrained_algorithm:
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, "wb"))
        torch.save(data, open(p_path, "wb"))

    algorithm.train()
