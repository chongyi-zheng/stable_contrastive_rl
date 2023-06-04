import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon import \
    GaussianAndEpsilonStrategy
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
# from rlkit.torch.td3.td3 import TD3
from rlkit.demos.td3_bc import TD3BCTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv

from multiworld.core.image_env import ImageEnv
# from multiworld.envs.real_world.sawyer.sawyer_reaching import SawyerReachXYZEnv
# from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv

# import rlkit.util.hyperparameter as hyp

from rlkit.visualization.video import VideoSaveFunction


# from rlkit.launchers.experiments.ashvin.rfeatures.rfeatures_trainer import TimePredictionTrainer

def state_td3bc_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    env = ImageEnv(env,
        recompute_reward=False,
        transpose=True,
        image_length=3 * 84 * 84,
        reward_type="not_used",
        # init_camera=sawyer_pusher_camera_upright_v2,
    )

    expl_env = env # variant['env_class'](**variant['env_kwargs'])
    eval_env = env # variant['env_class'](**variant['env_kwargs'])

    observation_key = 'state_observation'
    desired_goal_key = 'state_desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    es = GaussianAndEpsilonStrategy(
        action_space=expl_env.action_space,
        **variant["exploration_kwargs"],
    )
    obs_dim = expl_env.wrapped_env.observation_space.spaces['observation'].low.size
    goal_dim = expl_env.wrapped_env.observation_space.spaces['desired_goal'].low.size
    action_dim = expl_env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        # output_activation=TorchMaxClamp(0.0),
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        # output_activation=TorchMaxClamp(0.0),
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        # output_activation=TorchMaxClamp(0.0),
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        # output_activation=TorchMaxClamp(0.0),
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
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    demo_test_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    td3bc_trainer = TD3BCTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        replay_buffer=replay_buffer,
        demo_train_buffer=demo_train_buffer,
        demo_test_buffer=demo_test_buffer,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    trainer = HERTrainer(td3bc_trainer)
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
        video_func = VideoSaveFunction(
            env,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)

    td3bc_trainer.load_demos()
    td3bc_trainer.pretrain_policy_with_bc()
    td3bc_trainer.pretrain_q_with_bc_data()

    algorithm.train()
