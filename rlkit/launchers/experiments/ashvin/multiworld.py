"""This file contains experiments for training RL algorithms on multiworld
(observation dict) environments.
"""

def td3_experiment(variant):
    import gym
    import multiworld.envs.mujoco
    import multiworld.envs.pygame
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
    from rlkit.exploration_strategies.ou_strategy import OUStrategy
    from rlkit.torch.grill.launcher import get_state_experiment_video_save_function
    from rlkit.torch.her.her_td3 import HerTd3
    from rlkit.torch.td3.td3 import TD3
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.data_management.obs_dict_replay_buffer import (
        ObsDictReplayBuffer
    )
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.samplers.data_collector.path_collector import ObsDictPathCollector

    if 'env_id' in variant:
        eval_env = gym.make(variant['env_id'])
        expl_env = gym.make(variant['env_id'])
    else:
        eval_env_kwargs = variant.get('eval_env_kwargs', variant['env_kwargs'])
        eval_env = variant['env_class'](**eval_env_kwargs)
        expl_env = variant['env_class'](**variant['env_kwargs'])

    observation_key = variant['observation_key']
    # desired_goal_key = variant['desired_goal_key']
    # variant['algo_kwargs']['her_kwargs']['observation_key'] = observation_key
    # variant['algo_kwargs']['her_kwargs']['desired_goal_key'] = desired_goal_key
    if variant.get('normalize', False):
        raise NotImplementedError()

    # achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    replay_buffer = ObsDictReplayBuffer(
        env=eval_env,
        observation_key=observation_key,
        # desired_goal_key=desired_goal_key,
        # achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=eval_env.action_space,
            **variant['es_kwargs']
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=eval_env.action_space,
            **variant['es_kwargs'],
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=eval_env.action_space,
            **variant['es_kwargs'],
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
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
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
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
    target_policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    trainer = TD3(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    eval_path_collector = ObsDictPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        # render=True,
        # desired_goal_key=desired_goal_key,
    )
    expl_path_collector = ObsDictPathCollector(
        expl_env,
        expl_policy,
        observation_key=observation_key,
        # render=True,
        # desired_goal_key=desired_goal_key,
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
    #         observation_key=observation_key,
    #         desired_goal_key=algorithm.desired_goal_key,
    #     )
    #     video_func = get_state_experiment_video_save_function(
    #         rollout_function,
    #         env,
    #         policy,
    #         variant,
    #     )
    #     algorithm.post_epoch_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()


def her_td3_experiment(variant):
    import gym

    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import \
        PolicyWrappedWithExplorationStrategy
    from rlkit.exploration_strategies.gaussian_and_epsilon import \
        GaussianAndEpsilonStrategy
    from rlkit.launchers.launcher_util import setup_logger
    from rlkit.samplers.data_collector import GoalConditionedPathCollector
    from rlkit.torch.her.her import HERTrainer
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.torch.td3.td3 import TD3
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    import rlkit.samplers.rollout_functions as rf
    from rlkit.torch.grill.launcher import get_state_experiment_video_save_function

    if 'env_id' in variant:
        eval_env = gym.make(variant['env_id'])
        expl_env = gym.make(variant['env_id'])
    else:
        eval_env_kwargs = variant.get('eval_env_kwargs', variant['env_kwargs'])
        eval_env = variant['env_class'](**eval_env_kwargs)
        expl_env = variant['env_class'](**variant['env_kwargs'])

    observation_key = 'state_observation'
    desired_goal_key = 'state_desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    es = GaussianAndEpsilonStrategy(
        action_space=expl_env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    obs_dim = expl_env.observation_space.spaces['observation'].low.size
    goal_dim = expl_env.observation_space.spaces['desired_goal'].low.size
    action_dim = expl_env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    trainer = TD3(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        expl_policy,
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

    if variant.get("save_video", False):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
        video_func = get_state_experiment_video_save_function(
            rollout_function,
            eval_env,
            policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)

    algorithm.to(ptu.device)
    algorithm.train()

