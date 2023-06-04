from multiworld.core.image_env import ImageEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.util.io import load_local_or_remote_file
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.demos.td3_bc import TD3BCTrainer
from rlkit.torch.td3.td3 import TD3
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.visualization.video import VideoSaveFunction


def state_td3bc_experiment(variant):
    if variant.get('env_id', None):
        import gym
        import multiworld
        multiworld.register_all_envs()
        eval_env = gym.make(variant['env_id'])
        expl_env = gym.make(variant['env_id'])
    else:
        eval_env_kwargs = variant.get('eval_env_kwargs', variant['env_kwargs'])
        eval_env = variant['env_class'](**eval_env_kwargs)
        expl_env = variant['env_class'](**variant['env_kwargs'])

    observation_key = 'state_observation'
    desired_goal_key = 'state_desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    es_strat =  variant.get('es', 'ou')
    if es_strat == 'ou':
        es = OUStrategy(
            action_space=expl_env.action_space,
            max_sigma=variant['exploration_noise'],
            min_sigma=variant['exploration_noise'],
        )
    elif es_strat == 'gauss_eps':
        es = GaussianAndEpsilonStrategy(
            action_space=expl_env.action_space,
            max_sigma=.2,
            min_sigma=.2,  # constant sigma
            epsilon=.3,
        )
    else:
        raise ValueError("invalid exploration strategy provided")
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
    demo_train_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        max_size=variant['replay_buffer_kwargs']['max_size']
    )
    demo_test_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        max_size=variant['replay_buffer_kwargs']['max_size'],
    )
    if variant.get('td3_bc', True):
        td3_trainer = TD3BCTrainer(
            env=expl_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            **variant['td3_bc_trainer_kwargs']
        )
    else:
        td3_trainer = TD3(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            **variant['td3_trainer_kwargs']
        )
    trainer = HERTrainer(td3_trainer)
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

    if variant.get("save_video", True):
        if variant.get("presampled_goals", None):
            variant['image_env_kwargs']['presampled_goals'] = load_local_or_remote_file(variant['presampled_goals']).item()
        image_eval_env = ImageEnv(eval_env, **variant["image_env_kwargs"])
        image_eval_path_collector = GoalConditionedPathCollector(
            image_eval_env,
            policy,
            observation_key='state_observation',
            desired_goal_key='state_desired_goal',
        )
        image_expl_env = ImageEnv(expl_env, **variant["image_env_kwargs"])
        image_expl_path_collector = GoalConditionedPathCollector(
            image_expl_env,
            expl_policy,
            observation_key='state_observation',
            desired_goal_key='state_desired_goal',
        )
        video_func = VideoSaveFunction(
            image_eval_env,
            variant,
            image_expl_path_collector,
            image_eval_path_collector,
        )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    if variant.get('load_demos', False):
        td3_trainer.load_demos()
    if variant.get('pretrain_policy', False):
        td3_trainer.pretrain_policy_with_bc()
    if variant.get('pretrain_rl', False):
        td3_trainer.pretrain_q_with_bc_data()
    algorithm.train()
