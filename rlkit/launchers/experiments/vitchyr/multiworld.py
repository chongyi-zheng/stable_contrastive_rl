import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import (
    ObsDictRelabelingBuffer
)
from rlkit.samplers.data_collector.path_collector import (
    GoalConditionedPathCollector,
)
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def relabeling_tsac_experiment(variant):
    if 'presample_goals' in variant:
        raise NotImplementedError()
    if 'env_id' in variant:
        eval_env = gym.make(variant['env_id'])
        expl_env = gym.make(variant['env_id'])
    else:
        eval_env = variant['env_class'](**variant['env_kwargs'])
        expl_env = variant['env_class'](**variant['env_kwargs'])

    observation_key = variant['observation_key']
    desired_goal_key = variant['desired_goal_key']
    if variant.get('normalize', False):
        raise NotImplementedError()

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    max_path_length = variant['max_path_length']
    eval_policy = MakeDeterministic(policy)
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        eval_policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    # if variant.get("save_video", False):
    #     rollout_function = rf.create_rollout_function(
    #         rf.multitask_rollout,
    #         max_path_length=algorithm.max_path_length,
    #         observation_key=algorithm.observation_key,
    #         desired_goal_key=algorithm.desired_goal_key,
    #     )
    #     video_func = get_video_save_func(
    #         rollout_function,
    #         env,
    #         policy,
    #         variant,
    #     )
    #     algorithm.post_epoch_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()
