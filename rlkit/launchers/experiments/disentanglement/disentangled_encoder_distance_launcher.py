from functools import partial

import rlkit.samplers.rollout_functions as rf
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Identity
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.launchers.experiments.disentanglement.util import (
    get_extra_imgs,
    get_save_video_function,
    add_heatmap_imgs_to_o_dict,
)
from rlkit.samplers.data_collector import (
    GoalConditionedPathCollector,
)
from rlkit.torch.disentanglement.networks import DisentangledMlpQf
from rlkit.torch.disentanglement.encoder_wrapped_env import (
    EncoderWrappedEnv, EncoderFromNetwork,
)
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import (
    TanhGaussianPolicy,
    MakeDeterministic,
)
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def use_disentangled_encoder_distance(variant):
    _use_disentangled_encoder_distance(**variant)


def _use_disentangled_encoder_distance(
        max_path_length,
        encoder_kwargs,
        disentangled_qf_kwargs,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        evaluation_goal_sampling_mode,
        exploration_goal_sampling_mode,
        algo_kwargs,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        encoder_key_prefix='encoder',
        encoder_input_prefix='state',
        latent_dim=2,
        reward_mode=EncoderWrappedEnv.ENCODER_DISTANCE_REWARD,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        save_vf_heatmap=True,
        **kwargs
):
    if save_video_kwargs is None:
        save_video_kwargs = {}
    if env_kwargs is None:
        env_kwargs = {}
    assert env_id or env_class
    vectorized = (
            reward_mode == EncoderWrappedEnv.VECTORIZED_ENCODER_DISTANCE_REWARD
    )

    if env_id:
        import gym
        import multiworld
        multiworld.register_all_envs()
        raw_train_env = gym.make(env_id)
        raw_eval_env = gym.make(env_id)
    else:
        raw_eval_env = env_class(**env_kwargs)
        raw_train_env = env_class(**env_kwargs)

    raw_train_env.goal_sampling_mode = exploration_goal_sampling_mode
    raw_eval_env.goal_sampling_mode = evaluation_goal_sampling_mode

    raw_obs_dim = (
            raw_train_env.observation_space.spaces['state_observation'].low.size
    )
    action_dim = raw_train_env.action_space.low.size

    encoder = ConcatMlp(
        input_size=raw_obs_dim,
        output_size=latent_dim,
        **encoder_kwargs
    )
    encoder = Identity()
    encoder.input_size = raw_obs_dim
    encoder.output_size = raw_obs_dim

    np_encoder = EncoderFromNetwork(encoder)
    train_env = EncoderWrappedEnv(
        raw_train_env, np_encoder, encoder_input_prefix,
        key_prefix=encoder_key_prefix,
        reward_mode=reward_mode,
    )
    eval_env = EncoderWrappedEnv(
        raw_eval_env, np_encoder, encoder_input_prefix,
        key_prefix=encoder_key_prefix,
        reward_mode=reward_mode,
    )
    observation_key = '{}_observation'.format(encoder_key_prefix)
    desired_goal_key = '{}_desired_goal'.format(encoder_key_prefix)
    achieved_goal_key = '{}_achieved_goal'.format(encoder_key_prefix)
    obs_dim = train_env.observation_space.spaces[observation_key].low.size
    goal_dim = train_env.observation_space.spaces[desired_goal_key].low.size

    def make_qf():
        return DisentangledMlpQf(
            encoder=encoder,
            preprocess_obs_dim=obs_dim,
            action_dim=action_dim,
            qf_kwargs=qf_kwargs,
            vectorized=vectorized,
            **disentangled_qf_kwargs
        )
    qf1 = make_qf()
    qf2 = make_qf()
    target_qf1 = make_qf()
    target_qf2 = make_qf()

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
        **policy_kwargs
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=train_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        vectorized=vectorized,
        **replay_buffer_kwargs
    )
    sac_trainer = SACTrainer(
        env=train_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs
    )
    trainer = HERTrainer(sac_trainer)

    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode='env',
    )
    expl_path_collector = GoalConditionedPathCollector(
        train_env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode='env',
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=train_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs
    )
    algorithm.to(ptu.device)

    if save_video:
        def v_function(obs):
            action = policy.get_actions(obs)
            obs, action = ptu.from_numpy(obs), ptu.from_numpy(action)
            return qf1(obs, action, return_individual_q_vals=True)

        add_heatmap = partial(
            add_heatmap_imgs_to_o_dict,
            v_function=v_function,
            vectorized=vectorized,
        )
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=max_path_length,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            full_o_postprocess_func=add_heatmap if save_vf_heatmap else None,
        )
        img_keys = ['v_vals'] + [
            'v_vals_dim_{}'.format(dim) for dim
            in range(latent_dim)
        ]
        eval_video_func = get_save_video_function(
            rollout_function,
            eval_env,
            MakeDeterministic(policy),
            get_extra_imgs=partial(get_extra_imgs, img_keys=img_keys),
            tag="eval",
            **save_video_kwargs
        )
        train_video_func = get_save_video_function(
            rollout_function,
            train_env,
            policy,
            get_extra_imgs=partial(get_extra_imgs, img_keys=img_keys),
            tag="train",
            **save_video_kwargs
        )
        algorithm.post_train_funcs.append(eval_video_func)
        algorithm.post_train_funcs.append(train_video_func)
    algorithm.train()
