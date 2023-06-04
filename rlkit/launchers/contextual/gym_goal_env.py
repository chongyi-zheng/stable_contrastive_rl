from functools import partial

import numpy as np
from gym.envs import robotics

import rlkit.samplers.rollout_functions as rf
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from rlkit.envs.contextual import (
    ContextualEnv, delete_info,
)
from rlkit.envs.contextual.goal_conditioned import (
    ThresholdDistanceReward,
    L2Distance,
    IndexIntoAchievedGoal,
)
from rlkit.envs.contextual.gym_goal_envs import \
    (
    GoalDictDistributionFromGymGoalEnv,
    GenericGoalConditionedContextualDiagnostics,
)
from rlkit.envs.images import InsertImageEnv
from rlkit.envs.images.env_renderer import GymEnvRenderer
from rlkit.launchers.contextual.util import (
    get_save_video_function,
    get_gym_env,
)
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from rlkit.torch.networks.mlp import ConcatMlp
from rlkit.torch.sac.policies import (
    MakeDeterministic,
)
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def sac_on_gym_goal_env_experiment(
        max_path_length,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        observation_key='observation',
        desired_goal_key='desired_goal',
        achieved_goal_key='achieved_goal',
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
):
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}
    context_key = desired_goal_key
    sample_context_from_obs_dict_fn = RemapKeyFn({context_key: achieved_goal_key})

    def contextual_env_distrib_and_reward(
            env_id, env_class, env_kwargs, goal_sampling_mode
    ):
        env = get_gym_env(
            env_id, env_class=env_class, env_kwargs=env_kwargs,
            unwrap_timed_envs=True,
        )
        env.goal_sampling_mode = goal_sampling_mode
        goal_distribution = GoalDictDistributionFromGymGoalEnv(
            env,
            desired_goal_key=desired_goal_key,
        )
        distance_fn = L2Distance(
            achieved_goal_from_observation=IndexIntoAchievedGoal(
                achieved_goal_key,
            ),
            desired_goal_key=desired_goal_key,
        )
        if (
                isinstance(env, robotics.FetchReachEnv)
                or isinstance(env, robotics.FetchPushEnv)
                or isinstance(env, robotics.FetchPickAndPlaceEnv)
                or isinstance(env, robotics.FetchSlideEnv)
        ):
            success_threshold = 0.05
        else:
            raise TypeError("I don't know the success threshold of env ", env)
        reward_fn = ThresholdDistanceReward(distance_fn, success_threshold)
        diag_fn = GenericGoalConditionedContextualDiagnostics(
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
            success_threshold=success_threshold,
        )
        env = ContextualEnv(
            env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=[diag_fn],
            update_env_info_fn=delete_info,
        )
        return env, goal_distribution, reward_fn


    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        env_id, env_class, env_kwargs, exploration_goal_sampling_mode
    )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        env_id, env_class, env_kwargs, evaluation_goal_sampling_mode
    )

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
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=[context_key],
        observation_keys_to_save=[observation_key, achieved_goal_key],
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=sample_context_from_obs_dict_fn,
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
        context_keys_for_policy=[context_key],
    )
    exploration_policy = create_exploration_policy(
        policy=policy, env=expl_env, **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key,
        context_keys_for_policy=[context_key],
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

        # Setting the goal like this is discourage, but the Fetch environment
        # are designed to visualize the goals by setting their goal parameter.
        def set_goal_for_visualization(env, policy, o):
            goal = o[desired_goal_key]
            print(goal)
            env.unwrapped.goal = goal

        rollout_function = partial(
            rf.contextual_rollout,
            max_path_length=max_path_length,
            observation_key=observation_key,
            context_keys_for_policy=[context_key],
            reset_callback=set_goal_for_visualization,
        )
        renderer = GymEnvRenderer(**renderer_kwargs)

        def add_images(env, context_distribution):
            state_env = env.env
            img_env = InsertImageEnv(
                state_env, renderer=renderer, image_key='image_observation',
            )
            return ContextualEnv(
                img_env,
                context_distribution=context_distribution,
                reward_fn=eval_reward,
                observation_key=observation_key,
                update_env_info_fn=delete_info,
            )
        img_eval_env = add_images(eval_env, eval_context_distrib)
        img_expl_env = add_images(expl_env, expl_context_distrib)
        eval_video_func = get_save_video_function(
            rollout_function,
            img_eval_env,
            MakeDeterministic(policy),
            tag="eval",
            imsize=renderer.image_chw[1],
            image_format=renderer.output_image_format,
            keys_to_show=['image_observation'],
            **save_video_kwargs
        )
        expl_video_func = get_save_video_function(
            rollout_function,
            img_expl_env,
            exploration_policy,
            tag="train",
            imsize=renderer.image_chw[1],
            image_format=renderer.output_image_format,
            keys_to_show=['image_observation'],
            **save_video_kwargs
        )

        algorithm.post_train_funcs.append(eval_video_func)
        algorithm.post_train_funcs.append(expl_video_func)

    algorithm.train()
