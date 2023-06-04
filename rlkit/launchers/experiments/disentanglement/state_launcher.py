from functools import partial

from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.launchers.experiments.disentanglement.util import (
    get_extra_imgs,
    get_save_video_function,
    add_heatmap_imgs_to_o_dict,
)
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.disentanglement.networks import DisentangledMlpQf
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks.basic import Detach
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer


def her_twin_sac_experiment_v2(variant):
    her_sac_experiment(**variant)


def her_sac_experiment(
        max_path_length,
        qf_kwargs,
        twin_sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        evaluation_goal_sampling_mode,
        exploration_goal_sampling_mode,
        algo_kwargs,
        save_video=True,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        # Video parameters
        save_video_kwargs=None,
        exploration_policy_kwargs=None,
        **kwargs
):
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    if not save_video_kwargs:
        save_video_kwargs = {}

    if env_kwargs is None:
        env_kwargs = {}

    assert env_id or env_class
    if env_id:
        import gym
        import multiworld
        multiworld.register_all_envs()
        train_env = gym.make(env_id)
        eval_env = gym.make(env_id)
    else:
        eval_env = env_class(**env_kwargs)
        train_env = env_class(**env_kwargs)

    obs_dim = (
            train_env.observation_space.spaces[observation_key].low.size
            + train_env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = train_env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **policy_kwargs
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=train_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **replay_buffer_kwargs
    )
    trainer = SACTrainer(
        env=train_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **twin_sac_trainer_kwargs
    )
    trainer = HERTrainer(trainer)

    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=evaluation_goal_sampling_mode,
    )
    exploration_policy = create_exploration_policy(
        train_env, policy, **exploration_policy_kwargs)
    expl_path_collector = GoalConditionedPathCollector(
        train_env,
        exploration_policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=exploration_goal_sampling_mode,
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
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=max_path_length,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            return_dict_obs=True,
        )
        eval_video_func = get_save_video_function(
            rollout_function,
            eval_env,
            MakeDeterministic(policy),
            tag="eval",
            **save_video_kwargs
        )
        train_video_func = get_save_video_function(
            rollout_function,
            train_env,
            exploration_policy,
            tag="expl",
            **save_video_kwargs
        )

        # algorithm.post_train_funcs.append(plot_buffer_function(
            # save_video_period, 'state_achieved_goal'))
        # algorithm.post_train_funcs.append(plot_buffer_function(
            # save_video_period, 'state_desired_goal'))
        algorithm.post_train_funcs.append(eval_video_func)
        algorithm.post_train_funcs.append(train_video_func)

    algorithm.train()


def disentangled_her_twin_sac_experiment_v2(variant):
    _disentangled_her_twin_sac_experiment_v2(**variant)


def _disentangled_her_twin_sac_experiment_v2(
        max_path_length,
        encoder_kwargs,
        disentangled_qf_kwargs,
        qf_kwargs,
        twin_sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        evaluation_goal_sampling_mode,
        exploration_goal_sampling_mode,
        algo_kwargs,
        save_video=True,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        # Video parameters
        latent_dim=2,
        save_video_kwargs=None,
        **kwargs
):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

    if save_video_kwargs is None:
        save_video_kwargs = {}

    if env_kwargs is None:
        env_kwargs = {}
    assert env_id or env_class

    if env_id:
        import gym
        import multiworld
        multiworld.register_all_envs()
        train_env = gym.make(env_id)
        eval_env = gym.make(env_id)
    else:
        eval_env = env_class(**env_kwargs)
        train_env = env_class(**env_kwargs)

    obs_dim = train_env.observation_space.spaces[observation_key].low.size
    goal_dim = train_env.observation_space.spaces[desired_goal_key].low.size
    action_dim = train_env.action_space.low.size

    encoder = ConcatMlp(
        input_size=goal_dim,
        output_size=latent_dim,
        **encoder_kwargs
    )

    qf1 = DisentangledMlpQf(
        encoder=encoder,
        preprocess_obs_dim=obs_dim,
        action_dim=action_dim,
        qf_kwargs=qf_kwargs,
        **disentangled_qf_kwargs
    )
    qf2 = DisentangledMlpQf(
        encoder=encoder,
        preprocess_obs_dim=obs_dim,
        action_dim=action_dim,
        qf_kwargs=qf_kwargs,
        **disentangled_qf_kwargs
    )
    target_qf1 = DisentangledMlpQf(
        encoder=Detach(encoder),
        preprocess_obs_dim=obs_dim,
        action_dim=action_dim,
        qf_kwargs=qf_kwargs,
        **disentangled_qf_kwargs
    )
    target_qf2 = DisentangledMlpQf(
        encoder=Detach(encoder),
        preprocess_obs_dim=obs_dim,
        action_dim=action_dim,
        qf_kwargs=qf_kwargs,
        **disentangled_qf_kwargs
    )

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
        **replay_buffer_kwargs
    )
    sac_trainer = SACTrainer(
        env=train_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **twin_sac_trainer_kwargs
    )
    trainer = HERTrainer(sac_trainer)

    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=evaluation_goal_sampling_mode,
    )
    expl_path_collector = GoalConditionedPathCollector(
        train_env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=exploration_goal_sampling_mode,
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=train_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs,
    )
    algorithm.to(ptu.device)

    if save_video:
        save_vf_heatmap = save_video_kwargs.get('save_vf_heatmap', True)

        def v_function(obs):
            action = policy.get_actions(obs)
            obs, action = ptu.from_numpy(obs), ptu.from_numpy(action)
            return qf1(obs, action, return_individual_q_vals=True)
        add_heatmap = partial(add_heatmap_imgs_to_o_dict, v_function=v_function)
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
            tag="eval",
            get_extra_imgs=partial(get_extra_imgs, img_keys=img_keys),
            **save_video_kwargs
        )
        train_video_func = get_save_video_function(
            rollout_function,
            train_env,
            policy,
            tag="train",
            get_extra_imgs=partial(get_extra_imgs, img_keys=img_keys),
            **save_video_kwargs
        )
        decoder = ConcatMlp(
            input_size=obs_dim,
            output_size=obs_dim,
            hidden_sizes=[128, 128],
        )
        decoder.to(ptu.device)

        # algorithm.post_train_funcs.append(train_decoder(variant, encoder, decoder))
        # algorithm.post_train_funcs.append(plot_encoder_function(variant, encoder))
        # algorithm.post_train_funcs.append(plot_buffer_function(
            # save_video_period, 'state_achieved_goal'))
        # algorithm.post_train_funcs.append(plot_buffer_function(
            # save_video_period, 'state_desired_goal'))
        algorithm.post_train_funcs.append(eval_video_func)
        algorithm.post_train_funcs.append(train_video_func)



    algorithm.train()