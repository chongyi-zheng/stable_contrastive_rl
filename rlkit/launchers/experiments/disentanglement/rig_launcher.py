from functools import partial

import rlkit.samplers.rollout_functions as rf
import rlkit.torch.pytorch_util as ptu

from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.launchers.experiments.disentanglement.train_vae_launcher import \
    get_n_train_vae
from rlkit.launchers.experiments.disentanglement.util import (
    get_extra_imgs,
    get_save_video_function,
    add_heatmap_imgs_to_o_dict,
    add_heatmap_img_to_o_dict,
)
from rlkit.samplers.data_collector import (
    VAEWrappedEnvPathCollector
)
from rlkit.util.io import load_local_or_remote_file
from rlkit.torch.disentanglement.networks import DisentangledMlpQf
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.policies import (
    TanhGaussianPolicy,
    MakeDeterministic
)
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.envs.vae_wrappers import VAEWrappedEnv


def disentangled_grill_her_twin_sac_experiment(variant):
    _disentangled_grill_her_twin_sac_experiment(**variant)


def _disentangled_grill_her_twin_sac_experiment(
        max_path_length,
        encoder_kwargs,
        disentangled_qf_kwargs,
        qf_kwargs,
        twin_sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        vae_evaluation_goal_sampling_mode,
        vae_exploration_goal_sampling_mode,
        base_env_evaluation_goal_sampling_mode,
        base_env_exploration_goal_sampling_mode,
        algo_kwargs,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        latent_dim=2,
        vae_wrapped_env_kwargs=None,
        vae_path=None,
        vae_n_vae_training_kwargs=None,
        vectorized=False,
        save_video=True,
        save_video_kwargs=None,
        have_no_disentangled_encoder=False,
        **kwargs
):
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

    train_env.goal_sampling_mode = base_env_exploration_goal_sampling_mode
    eval_env.goal_sampling_mode = base_env_evaluation_goal_sampling_mode

    if vae_path:
        vae = load_local_or_remote_file(vae_path)
    else:
        vae = get_n_train_vae(latent_dim=latent_dim, env=eval_env,
                              **vae_n_vae_training_kwargs)

    train_env = VAEWrappedEnv(
        train_env,
        vae,
        imsize=train_env.imsize,
        **vae_wrapped_env_kwargs
    )
    eval_env = VAEWrappedEnv(
        eval_env,
        vae,
        imsize=train_env.imsize,
        **vae_wrapped_env_kwargs
    )

    obs_dim = train_env.observation_space.spaces[observation_key].low.size
    goal_dim = train_env.observation_space.spaces[desired_goal_key].low.size
    action_dim = train_env.action_space.low.size

    encoder = ConcatMlp(
        input_size=obs_dim,
        output_size=latent_dim,
        **encoder_kwargs
    )

    def make_qf():
        if have_no_disentangled_encoder:
            return ConcatMlp(
                input_size=obs_dim + goal_dim + action_dim,
                output_size=1,
                **qf_kwargs,
            )
        else:
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
        **twin_sac_trainer_kwargs
    )
    trainer = HERTrainer(sac_trainer)

    eval_path_collector = VAEWrappedEnvPathCollector(
        eval_env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=vae_evaluation_goal_sampling_mode,
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        train_env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=vae_exploration_goal_sampling_mode,
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

        if have_no_disentangled_encoder:
            def v_function(obs):
                action = policy.get_actions(obs)
                obs, action = ptu.from_numpy(obs), ptu.from_numpy(action)
                return qf1(obs, action)
            add_heatmap = partial(
                add_heatmap_img_to_o_dict, v_function=v_function)
        else:
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
