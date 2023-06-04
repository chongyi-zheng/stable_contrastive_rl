import os.path as osp

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from rlkit.envs.contextual import ContextualEnv
from rlkit.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    AddImageDistribution,
    GoalConditionedDiagnosticsToContextualDiagnostics,
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
from rlkit.visualization.video import dump_video, RIGVideoSaveFunction
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.core import logger
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv
from rlkit.util.io import load_local_or_remote_file
from rlkit.core.eval_util import create_stats_ordered_dict

from collections import OrderedDict

from rlkit.launchers.contextual.rig.model_train_launcher import train_vae


class DistanceRewardFn:
    def __init__(self, observation_key, desired_goal_key):
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key

    def __call__(self, states, actions, next_states, contexts):
        s = next_states[self.observation_key]
        c = contexts[self.desired_goal_key]
        return -np.linalg.norm(s - c, axis=1)


class StateImageGoalDiagnosticsFn:
    def __init__(self, state_to_goal_keys_map):
        self.state_to_goal_keys_map = state_to_goal_keys_map

    def __call__(self, paths, contexts):
        diagnostics = OrderedDict()
        for state_key in self.state_to_goal_keys_map:
            goal_key = self.state_to_goal_keys_map[state_key]
            values = []
            for i in range(len(paths)):
                state = paths[i]["observations"][-1][state_key]
                goal = contexts[i][goal_key]
                distance = np.linalg.norm(state - goal)
                values.append(distance)
            diagnostics_key = goal_key + "/final/distance"
            diagnostics.update(create_stats_ordered_dict(
                diagnostics_key,
                values,
            ))
        return diagnostics


def rig_experiment(
        max_path_length,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        train_vae_kwargs,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        state_goal_key='state_desired_goal',
        state_observation_key='state_observation',
        image_goal_key='image_desired_goal',
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
        imsize=48,
        pretrained_vae_path="",
        init_camera=None,
):
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}

    renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)

    def contextual_env_distrib_and_reward(
            env_id, env_class, env_kwargs, goal_sampling_mode
    ):
        state_env = get_gym_env(
            env_id, env_class=env_class, env_kwargs=env_kwargs)

        renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)
        img_env = InsertImageEnv(state_env, renderer=renderer)

        encoded_env = EncoderWrappedEnv(
            img_env,
            model,
            dict(image_observation="latent_observation", ),
        )
        if goal_sampling_mode == "vae_prior":
            latent_goal_distribution = PriorDistribution(
                model.representation_size,
                desired_goal_key,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )
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
                desired_goal_key,
                model,
            )
            if hasattr(state_goal_env, 'goal_conditioned_diagnostics'):
                diagnostics = (
                    GoalConditionedDiagnosticsToContextualDiagnostics(
                        state_goal_env.goal_conditioned_diagnostics,
                        desired_goal_key=state_goal_key,
                        observation_key=state_observation_key,
                    )
                )
            else:
                state_goal_env.get_contextual_diagnostics
                diagnostics = state_goal_env.get_contextual_diagnostics
        else:
            raise NotImplementedError(
                'unknown goal sampling method: %s' % goal_sampling_mode
            )

        reward_fn = DistanceRewardFn(
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )

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
    else:
        model = train_vae(train_vae_kwargs, env_kwargs,
                          env_id, env_class, imsize, init_camera)

    expl_env, expl_context_distrib, expl_reward = (
        contextual_env_distrib_and_reward(
            env_id, env_class, env_kwargs, exploration_goal_sampling_mode
        )
    )
    eval_env, eval_context_distrib, eval_reward = (
        contextual_env_distrib_and_reward(
            env_id, env_class, env_kwargs, evaluation_goal_sampling_mode
        )
    )
    context_key = desired_goal_key

    obs_dim = (
        expl_env.observation_space.spaces[observation_key].low.size
        + expl_env.observation_space.spaces[context_key].low.size
    )
    action_dim = expl_env.action_space.low.size

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

    def concat_context_to_obs(batch, *args, **kwargs):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate(
            [next_obs, context], axis=1)
        return batch
    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=[context_key],
        observation_keys_to_save=[observation_key],
        observation_key=observation_key,
        context_distribution=expl_context_distrib,
        sample_context_from_obs_dict_fn=RemapKeyFn(
            {context_key: observation_key}),
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs
    )
    trainer = SACTrainer(
        env=expl_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs
    )

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

    if save_video:
        expl_video_func = RIGVideoSaveFunction(
            model,
            expl_path_collector,
            "train",
            decode_goal_image_key="image_decoded_goal",
            reconstruction_key="image_reconstruction",
            rows=2,
            columns=5,
            unnormalize=True,
            # max_path_length=200,
            imsize=48,
            image_format=renderer.output_image_format,
            **save_video_kwargs
        )
        algorithm.post_train_funcs.append(expl_video_func)

        eval_video_func = RIGVideoSaveFunction(
            model,
            eval_path_collector,
            "eval",
            goal_image_key=image_goal_key,
            decode_goal_image_key="image_decoded_goal",
            reconstruction_key="image_reconstruction",
            num_imgs=4,
            rows=2,
            columns=5,
            unnormalize=True,
            # max_path_length=200,
            imsize=48,
            image_format=renderer.output_image_format,
            **save_video_kwargs
        )
        algorithm.post_train_funcs.append(eval_video_func)

    algorithm.train()


def get_save_video_function(
        rollout_function,
        env,
        policy,
        save_video_period=10,
        imsize=48,
        tag="",
        **dump_video_kwargs
):
    logdir = logger.get_snapshot_dir()

    def save_video(algo, epoch):
        if epoch % save_video_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(
                logdir,
                'video_{}_{epoch}_env.mp4'.format(tag, epoch=epoch),
            )
            dump_video(env, policy, filename, rollout_function,
                       imsize=imsize, **dump_video_kwargs)
    return save_video


def get_gym_env(env_id, env_class=None, env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}

    assert env_id or env_class
    if env_id:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(env_id)
    else:
        env = env_class(**env_kwargs)
    return env


def process_args(variant):
    debug = variant.pop("debug", False)
    if debug:
        train_vae_kwargs = variant["train_vae_kwargs"]
        train_vae_kwargs["num_epochs"] = 1
        train_vae_kwargs["algo_kwargs"]["batch_size"] = 7
        train_vae_kwargs["generate_vae_dataset_kwargs"]["batch_size"] = 7
        train_vae_kwargs["generate_vae_dataset_kwargs"]["N"] = 100
        variant["max_path_length"] = 50
        algo_kwargs = variant["algo_kwargs"]
        algo_kwargs["batch_size"] = 2
        algo_kwargs["num_eval_steps_per_epoch"] = 500
        algo_kwargs["num_expl_steps_per_train_loop"] = 500
        algo_kwargs["num_trains_per_train_loop"] = 50
        algo_kwargs["min_num_steps_before_training"] = 500
