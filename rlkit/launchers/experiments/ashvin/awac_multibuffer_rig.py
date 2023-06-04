from rlkit.envs.wrappers import StackObservationEnv, RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.samplers.data_collector.path_collector import GoalConditionedPathCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.networks.cnn import ConcatCNN
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, GaussianPolicy, GaussianMixturePolicy, GaussianCNNPolicy, GaussianTwoChannelCNNPolicy
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.util.io import load_local_or_remote_file
from rlkit.envs.encoder_wrappers import VQVAEWrappedEnv
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.sac.iql_trainer import IQLTrainer
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

from torch.distributions import kl_divergence
import rlkit.pythonplusplus as ppp

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
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
    AddManualImageDistribution,
    GoalConditionedDiagnosticsToContextualDiagnostics,
    IndexIntoAchievedGoal,
    PresampledPathDistribution
)

from torch.utils import data
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from rlkit.data_management.online_offline_split_replay_buffer import (
    OnlineOfflineSplitReplayBuffer,
)
from rlkit.envs.contextual import ContextualEnv
from rlkit.envs.contextual.latent_distributions import (
    AmortizedConditionalPriorDistribution,
    PresampledPriorDistribution,
    ConditionalPriorDistribution,
    AmortizedPriorDistribution,
    AddDecodedImageDistribution,
    AddLatentDistribution,
    AddConditionalLatentDistribution,
    PriorDistribution,
    PresamplePriorDistribution,
    PresampledConditionalPriorDistribution,
)
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv, ConditionalEncoderWrappedEnv, PresamplingEncoderWrappedEnv
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.core.eval_util import create_stats_ordered_dict
from collections import OrderedDict

from multiworld.core.image_env import ImageEnv, unormalize_image
import multiworld

from rlkit.torch.grill.common import train_vae
from torchvision.utils import save_image

from rlkit.data_management.dataset_logger_fn import DatasetLoggerFn, run_bc_batch

import copy


class RewardFn:
    def __init__(self,
                 env,
                 obs_type='latent',
                 reward_type='dense',
                 epsilon=1.0,
                 ):
        if obs_type == 'latent':
            self.observation_key = 'latent_observation'
            self.desired_goal_key = 'latent_desired_goal'
        elif obs_type == 'state':
            self.observation_key = 'state_observation'
            self.desired_goal_key = 'state_desired_goal'
        self.env = env
        self.reward_type = reward_type

        self.epsilon = epsilon

    def set_model(self, model):
        # IGNORES model! we use different models for reward, obs
        if self.reward_type == "disco":
            pretrained_vae_path = "/home/ashvin/raw_vae_for_ashvin.pt"
            vae = load_local_or_remote_file(pretrained_vae_path)
            self.model = vae
            self.model.eval()
            self.model.to(ptu.device)
            base_path = '/home/ashvin/ros_ws/src/sawyer_control/src/'
            train_sets = [ptu.from_numpy(t) for t in np.load(
                base_path + 'train_sets.npy')]
            eval_sets = [ptu.from_numpy(t) for t in np.load(
                base_path + 'eval_sets.npy')]

            set_images = train_sets[0]  # 0 = closed door, 1 = open door
            prior_c = self.model.encoder_c(set_images)
            c = prior_c.mean
            self.prior = self.model.prior_z_given_c(c)

    def process(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def __call__(self, states, actions, next_states, contexts):
        s = self.process(next_states[self.observation_key])
        c = self.process(contexts[self.desired_goal_key])

        if self.reward_type == 'dense':
            reward = -np.linalg.norm(s - c, axis=1)
        elif self.reward_type == 'sparse':
            success = np.linalg.norm(s - c, axis=1) < self.epsilon
            reward = success - 1
        elif self.reward_type == 'wrapped_env':
            reward = self.env.compute_reward(states, actions, next_states,
                                             contexts)
        elif self.reward_type == 'disco':
            # def crop(img):
            #     import cv2
            #     img = cv2.resize(img, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
            #     Cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     img = Cimg.transpose(2,0,1).reshape(1, 3, 48, 48)
            #     return img / 256
            if 'image_observation' in states:  # on-policy
                x = ptu.from_numpy(states['image_observation'])
            else:
                error
                x = ptu.from_numpy(crop(states['hires_image_observation']))
            # save_image(x, "reward_test.png")
            q_z = self.model.q_zs_given_independent_xs(x)
            def reward_fn(q_z): return -kl_divergence(q_z, self.prior)
            reward = ptu.get_numpy(reward_fn(q_z))
        else:
            raise ValueError(self.reward_type)
        return reward


def compute_hand_sparse_reward(next_obs, reward, done, info):
    return info['goal_achieved'] - 1


def resume(variant):
    data = load_local_or_remote_file(variant.get(
        "pretrained_algorithm_path"), map_location="cuda")
    algo = data['algorithm']

    algo.num_epochs = variant['num_epochs']

    post_pretrain_hyperparams = variant["trainer_kwargs"].get(
        "post_pretrain_hyperparams", {})
    algo.trainer.set_algorithm_weights(**post_pretrain_hyperparams)

    algo.train()


def process_args(variant):
    if variant.get("debug", False):
        # variant['max_path_length'] = 5
        # variant['num_presample'] = 50
        # variant.get('algo_kwargs', {}).update(dict(
        #     batch_size=5,
        #     start_epoch=-2,
        #     num_epochs=5,
        #     num_eval_steps_per_epoch=50,
        #     num_expl_steps_per_train_loop=50,
        #     num_trains_per_train_loop=10,
        #     min_num_steps_before_training=50,
        # ))
        # variant['trainer_kwargs']['bc_num_pretrain_steps'] = min(10, variant['trainer_kwargs'].get('bc_num_pretrain_steps', 0))
        # variant['trainer_kwargs']['q_num_pretrain1_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain1_steps', 0))
        # variant['trainer_kwargs']['q_num_pretrain2_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain2_steps', 0))
        # variant.get('train_vae_kwargs', {}).update(dict(
        #     num_epochs=1,
        #     train_pixelcnn_kwargs=dict(
        #         num_epochs=1,
        #         data_size=10,
        #         num_train_batches_per_epoch=2,
        #         num_test_batches_per_epoch=2,
        #     ),
        # ))
        paths = variant['path_loader_kwargs'].get('paths', [])
        if len(paths) > 1:
            variant['path_loader_kwargs']['paths'] = [paths[0], paths[-1]]


def awac_rig_experiment(
        max_path_length,
        qf_kwargs,
        trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        train_vae_kwargs,
        online_offline_split_replay_buffer_kwargs=None,
        image=False,
        online_offline_split=False,
        policy_class=TanhGaussianPolicy,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        reward_kwargs=None,
        encoder_wrapper=EncoderWrappedEnv,
        observation_key='latent_observation',
        observation_keys=['latent_observation'],
        observation_key_reward_fn='latent_observation',
        desired_goal_key='latent_desired_goal',
        desired_goal_key_reward_fn='latent_desired_goal',
        state_observation_key='state_observation',
        state_goal_key='state_desired_goal',
        image_goal_key='image_desired_goal',
        reset_keys_map=None,

        path_loader_class=MDPPathLoader,
        demo_replay_buffer_kwargs=None,
        extra_replay_buffers_kwargs=None,
        path_loader_kwargs=None,
        env_demo_path='',
        env_offpolicy_data_path='',

        debug=False,
        epsilon=1.0,
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        deterministc_eval=True,
        training_goal_sampling_mode=None,

        add_env_demos=False,
        add_env_offpolicy_data=False,
        pickle_paths=False,
        load_demos=False,
        pretrain_policy=False,
        pretrain_rl=False,
        save_pretrained_algorithm=False,
        train_model_func=train_vae,

        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
        imsize=84,
        pretrained_vae_path="",
        input_representation="",
        goal_representation="",
        presampled_goal_kwargs=None,
        presampled_goals_path="",
        ccvae_or_cbigan_exp=False,
        num_presample=5000,
        init_camera=None,
        qf_class=ConcatMlp,

        trainer_class=AWACTrainer,

    # ICLR 2020 SPECIFIC
    num_pybullet_objects=None,
    pretrained_algo_path=None,
    seed=None,
    **kwargs
):
    #Kwarg Definitions
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if reset_keys_map is None:
        reset_keys_map = {}
    if demo_replay_buffer_kwargs is None:
        demo_replay_buffer_kwargs = {}
    if extra_replay_buffers_kwargs is None:
        extra_replay_buffers_kwargs = []
    if presampled_goal_kwargs is None:
        presampled_goal_kwargs = \
            {'eval_goals': '', 'expl_goals': '', 'training_goals': ''}
    if path_loader_kwargs is None:
        path_loader_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}
    if online_offline_split_replay_buffer_kwargs is None:
        online_offline_split_replay_buffer_kwargs = {}

    #Enviorment Wrapping
    renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)

    def contextual_env_distrib_and_reward(
            env, encoder_wrapper, goal_sampling_mode, presampled_goals_path, num_presample
    ):
        state_env = env
        renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)
        img_env = InsertImageEnv(state_env, renderer=renderer)

        encoded_env = encoder_wrapper(
            img_env,
            model,
            step_keys_map=dict(image_observation="latent_observation"),
            reset_keys_map=reset_keys_map,
        )
        if goal_sampling_mode == "vae_prior":
            latent_goal_distribution = PriorDistribution(
                model.representation_size,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )

        elif goal_sampling_mode == "none":
            latent_goal_distribution = PriorDistribution(
                0,
                desired_goal_key,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )

        elif goal_sampling_mode == "amortized_vae_prior":
            latent_goal_distribution = AmortizedPriorDistribution(
                model,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                num_presample=num_presample,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )

        elif goal_sampling_mode == 'conditional_vae_prior':
            latent_goal_distribution = ConditionalPriorDistribution(
                model,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key
            )
            if image:
                latent_goal_distribution = AddDecodedImageDistribution(
                    latent_goal_distribution,
                    desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                    image_goal_key,
                    model,
                )
            diagnostics = StateImageGoalDiagnosticsFn({}, )
        elif goal_sampling_mode == 'presampled_conditional_prior':
            latent_goal_distribution = PresampledConditionalPriorDistribution(
                model,
                desired_goal_key
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )
        elif goal_sampling_mode == "amortized_conditional_vae_prior":
            latent_goal_distribution = AmortizedConditionalPriorDistribution(
                model,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                num_presample=num_presample,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )
        elif goal_sampling_mode == "presampled_images":
            diagnostics = state_env.get_contextual_diagnostics
            image_goal_distribution = PresampledPathDistribution(
                presampled_goals_path,
                model.representation_size,
            )

            #TEMP#
            if ccvae_or_cbigan_exp:
                add_distrib = AddConditionalLatentDistribution
            else:
                add_distrib = AddLatentDistribution
            #TEMP#

            #AddLatentDistribution
            latent_goal_distribution = add_distrib(
                image_goal_distribution,
                image_goal_key,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                model,
            )
        elif goal_sampling_mode == "presample_latents":
            diagnostics = state_env.get_contextual_diagnostics
            latent_goal_distribution = PresamplePriorDistribution(
                model,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                state_env,
                num_presample=num_presample,
            )
            if image:
                latent_goal_distribution = AddDecodedImageDistribution(
                    latent_goal_distribution,
                    desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                    image_goal_key,
                    model,
                )
        elif goal_sampling_mode == "presampled_latents":
            diagnostics = state_env.get_contextual_diagnostics
            latent_goal_distribution = PresampledPriorDistribution(
                presampled_goals_path,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
            )
        elif goal_sampling_mode == "reset_of_env":
            state_goal_env = get_gym_env(
                env_id, env_class=env_class, env_kwargs=env_kwargs)
            state_goal_distribution = GoalDictDistributionFromMultitaskEnv(
                state_goal_env,
                desired_goal_keys=[state_goal_key],
            )
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_goal_distribution,
                image_goal_key=image_goal_key,
                renderer=renderer,
            )
            latent_goal_distribution = AddLatentDistribution(
                image_goal_distribution,
                image_goal_key,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                model,
            )
            diagnostics = state_goal_env.get_contextual_diagnostics
        elif goal_sampling_mode == "manual_reset_of_env":
            state_goal_env = get_gym_env(
                env_id, env_class=env_class, env_kwargs=env_kwargs)
            state_goal_distribution = GoalDictDistributionFromMultitaskEnv(
                state_goal_env,
                desired_goal_keys=[state_goal_key],
            )
            image_goal_distribution = AddManualImageDistribution(
                env=state_env,
                base_distribution=state_goal_distribution,
                image_goal_key=image_goal_key,
                renderer=renderer,
            )
            latent_goal_distribution = AddLatentDistribution(
                image_goal_distribution,
                image_goal_key,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                model,
            )
            diagnostics = state_goal_env.get_contextual_diagnostics
        else:
            print("goal_sampling_mode", goal_sampling_mode)
            error

        reward_fn = RewardFn(
            state_env,
            **reward_kwargs
        )
        reward_fn.set_model(model)
        env = ContextualEnv(
            encoded_env,
            context_distribution=latent_goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=[diagnostics],
        )
        return env, latent_goal_distribution, reward_fn

    if pretrained_vae_path:
        model = load_local_or_remote_file(pretrained_vae_path)
        path_loader_kwargs['model_path'] = pretrained_vae_path
    else:
        model = train_model_func(train_vae_kwargs)
        path_loader_kwargs['model'] = model
    model.to(ptu.device)

    expl_env_kwargs = env_kwargs.copy()
    expl_env_kwargs['expl'] = True

    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['expl'] = False

    #Environment Definitions
    state_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        state_env, encoder_wrapper, exploration_goal_sampling_mode,
        presampled_goal_kwargs['expl_goals'], num_presample
    )
    training_env, training_context_distrib, training_reward = contextual_env_distrib_and_reward(
        state_env, encoder_wrapper, training_goal_sampling_mode,
        presampled_goal_kwargs['expl_goals'], num_presample
    )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        state_env, encoder_wrapper, evaluation_goal_sampling_mode,
        presampled_goal_kwargs['eval_goals'], num_presample
    )
    path_loader_kwargs['env'] = eval_env

    #AWAC Code
    if add_env_demos:
        path_loader_kwargs["paths"].append(env_demo_path)
    if add_env_offpolicy_data:
        path_loader_kwargs["paths"].append(env_offpolicy_data_path)

    #Key Setting
    context_key = desired_goal_key
    obs_dim = (
        expl_env.observation_space.spaces[observation_key].low.size
        + expl_env.observation_space.spaces[context_key].low.size
    )
    action_dim = expl_env.action_space.low.size

    state_rewards = reward_kwargs.get('reward_type', 'dense') == 'wrapped_env'
    if desired_goal_key_reward_fn:
        if state_rewards:
            mapper = RemapKeyFn({context_key: observation_key, state_goal_key: state_observation_key,
                                desired_goal_key_reward_fn: observation_key_reward_fn})
            obs_keys = [state_observation_key,
                        observation_key, observation_key_reward_fn]
            cont_keys = [state_goal_key, context_key,
                         desired_goal_key_reward_fn]
        else:
            mapper = RemapKeyFn(
                {context_key: observation_key, desired_goal_key_reward_fn: observation_key_reward_fn})
            obs_keys = [observation_key, observation_key_reward_fn] + \
                list(reset_keys_map.values())
            cont_keys = [context_key, desired_goal_key_reward_fn]
    else:
        if state_rewards:
            mapper = RemapKeyFn(
                {context_key: observation_key, state_goal_key: state_observation_key})
            obs_keys = [state_observation_key, observation_key]
            cont_keys = [state_goal_key, context_key]
        else:
            mapper = RemapKeyFn({context_key: observation_key})
            obs_keys = [observation_key] + list(reset_keys_map.values())
            cont_keys = [context_key]

    #tmp robot code#
    #purpose: hardcoded to presample goals per image
    if exploration_goal_sampling_mode == 'presampled_conditional_prior':
        obs_keys.append('presampled_latent_goals')

    #Replay Buffer
    def concat_context_to_obs(batch, replay_buffer, obs_dict, next_obs_dict, new_contexts):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate(
            [next_obs, context], axis=1)
        return batch

    def make_replay_buffer(**kwargs):
        return ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys,
            observation_key=observation_key,
            observation_key_reward_fn=observation_key_reward_fn,
            #observation_keys=observation_keys,
            context_distribution=training_context_distrib,  # expl_context_distrib,
            sample_context_from_obs_dict_fn=mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            **kwargs
        )

    if online_offline_split:
        online_replay_buffer = make_replay_buffer(
            **online_offline_split_replay_buffer_kwargs['online_replay_buffer_kwargs'])
        offline_replay_buffer = make_replay_buffer(
            **online_offline_split_replay_buffer_kwargs['offline_replay_buffer_kwargs'])
        replay_buffer = OnlineOfflineSplitReplayBuffer(
            offline_replay_buffer,
            online_replay_buffer,
            **online_offline_split_replay_buffer_kwargs
        )
    else:
        replay_buffer = make_replay_buffer(**replay_buffer_kwargs)

    replay_buffers = dict(
        replay=replay_buffer,
    )

    replay_buffer_kwargs.update(demo_replay_buffer_kwargs)

    for extra_replay_buffer in extra_replay_buffers_kwargs:
        kwargs = copy.deepcopy(replay_buffer_kwargs)
        kwargs.update(demo_replay_buffer_kwargs)
        kwargs.update(extra_replay_buffers_kwargs[extra_replay_buffer])
        buffer = make_replay_buffer(**kwargs)
        replay_buffers[extra_replay_buffer] = buffer

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

    if pretrained_algo_path:
        data = load_local_or_remote_file(
            pretrained_algo_path, map_location="cuda")
        qf1 = data['trainer/qf1']
        qf2 = data['trainer/qf2']
        target_qf1 = data['trainer/target_qf1']
        target_qf2 = data['trainer/target_qf2']
        policy = data['trainer/policy']
        vf = data['trainer/vf']
    else:
        qf1 = create_qf()
        qf2 = create_qf()
        target_qf1 = create_qf()
        target_qf2 = create_qf()

        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )

        vf_kwargs = dict(hidden_sizes=[256, 256, ],)
        vf = ConcatMlp(
            input_size=obs_dim,
            output_size=1,
            **vf_kwargs
        )

    #Path Collectors
    if deterministc_eval:
        eval_policy = MakeDeterministic(policy)
    else:
        eval_policy = policy
    eval_path_collector = ContextualPathCollector(
        eval_env,
        eval_policy,
        observation_keys=[observation_key, ],
        context_keys_for_policy=[context_key, ],
    )
    exploration_policy = create_exploration_policy(
        expl_env, policy, **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_keys=[observation_key, ],
        context_keys_for_policy=[context_key, ],
    )

    #Algorithm
    trainer = trainer_class(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vf=vf,
        model=model,
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

    #AWAC CODE
    if pickle_paths:
        algorithm.pre_train_funcs.append(save_paths)

    #Video Saving
    if save_video:
        expl_video_func = RIGVideoSaveFunction(
            model,
            expl_path_collector,
            "train",
            decode_goal_image_key="image_decoded_goal",
            reconstruction_key="image_reconstruction",
            rows=2,
            columns=4,
            unnormalize=True,
            imsize=imsize,
            image_format=renderer.output_image_format,
            **save_video_kwargs
        )
        algorithm.pre_train_funcs.append(expl_video_func)

        eval_video_func = RIGVideoSaveFunction(
            model,
            eval_path_collector,
            "eval",
            goal_image_key=image_goal_key,
            decode_goal_image_key="image_decoded_goal",
            reconstruction_key="image_reconstruction",
            num_imgs=4,
            rows=2,
            columns=4,
            unnormalize=True,
            imsize=imsize,
            image_format=renderer.output_image_format,
            **save_video_kwargs
        )
        algorithm.pre_train_funcs.append(eval_video_func)

    if load_demos:
        default_path_loading_kwargs = dict(
            recompute_reward=True,
            normalize_img=False,
        )
        paths = path_loader_kwargs.pop('paths')
        path_loader = path_loader_class(
            replay_buffers,
            paths,
            reward_fn=eval_reward,
            **path_loader_kwargs
        )
        path_loader.load_paths()

        for buffer_name in replay_buffers:
            buffer = replay_buffers[buffer_name]
            log_fn = DatasetLoggerFn(
                buffer, run_bc_batch, prefix="%s/" % buffer_name, policy=policy)
            algorithm.pre_train_diag_funcs.append(log_fn)

    algorithm.train()
