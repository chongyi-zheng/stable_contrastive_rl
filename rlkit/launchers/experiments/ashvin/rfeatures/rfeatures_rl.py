import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon import \
    GaussianAndEpsilonStrategy
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy, CNNPolicy
# from rlkit.torch.td3.td3 import TD3
from rlkit.demos.td3_bc import TD3BCTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv

from multiworld.core.image_env import ImageEnv

# import rlkit.util.hyperparameter as hyp
from rlkit.launchers.experiments.ashvin.rfeatures.encoder_wrapped_env import EncoderWrappedEnv

import torch

from rlkit.launchers.experiments.ashvin.rfeatures.rfeatures_model import TimestepPredictionModel
import numpy as np

from rlkit.visualization.video import VideoSaveFunction

# from rlkit.launchers.experiments.ashvin.rfeatures.rfeatures_trainer import TimePredictionTrainer

from torchvision.utils import save_image

def encoder_wrapped_td3bc_experiment(variant):
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
    save_image(goal_image_pt.data.cpu(), 'demos/goal.png', nrow=1)
    goal_latent = model.encode(goal_image_pt).detach().cpu().numpy().flatten()

    initial_image = traj["observations"][0]["image_observation"]
    initial_image = initial_image.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
    # initial_image = initial_image.reshape(1, 300, 500, 3).transpose([0, 3, 1, 2]) / 255.0
    # initial_image = initial_image[:, :, :240, 60:500]
    initial_image = initial_image[:, :, 60:, 60:500]
    initial_image_pt = ptu.from_numpy(initial_image)
    save_image(initial_image_pt.data.cpu(), 'demos/initial.png', nrow=1)
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

    expl_env = env # variant['env_class'](**variant['env_kwargs'])
    eval_env = env # variant['env_class'](**variant['env_kwargs'])

    observation_key = variant.get("observation_key", 'state_observation')
    # one of 'state_observation', 'latent_observation', 'concat_observation'
    desired_goal_key = 'latent_desired_goal'

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    es = GaussianAndEpsilonStrategy(
        action_space=expl_env.action_space,
        **variant["exploration_kwargs"],
    )
    obs_dim = expl_env.observation_space.spaces[observation_key].low.size
    goal_dim = expl_env.observation_space.spaces[desired_goal_key].low.size
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

    # Support for CNNPolicy based policy/target policy
    # Defaults to TanhMlpPolicy unless cnn_params is supplied in variant
    if 'cnn_params' in variant.keys():
        imsize = 48
        policy = CNNPolicy(input_width=imsize,
                           input_height=imsize,
                           output_size=action_dim,
                           input_channels=3,
                           **variant['cnn_params'],
                           output_activation=torch.tanh,
        )
        target_policy = CNNPolicy(input_width=imsize,
                           input_height=imsize,
                           output_size=action_dim,
                           input_channels=3,
                           **variant['cnn_params'],
                           output_activation=torch.tanh,
        )
    else:
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