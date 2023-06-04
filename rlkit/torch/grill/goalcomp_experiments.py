from rlkit.samplers.data_collector import VAEWrappedEnvPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm

from rlkit.visualization.video import VideoSaveFunction

from rlkit.torch.grill.common import *


def grill_her_twin_sac_online_vae_full_experiment(variant):
    variant['grill_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_her_twin_sac_experiment_online_vae(variant['grill_variant'])


def grill_her_twin_sac_experiment_online_vae(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer

    grill_preprocess_variant(variant)
    env = get_envs(variant)

    uniform_dataset_fn = variant.get('generate_uniform_dataset_fn', None)
    if uniform_dataset_fn:
        uniform_dataset=uniform_dataset_fn(
            **variant['generate_uniform_dataset_kwargs']
        )
    else:
        uniform_dataset=None

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    vae = env.vae

    replay_buffer_class = variant.get("replay_buffer_class", OnlineVaeRelabelingBuffer)
    replay_buffer = replay_buffer_class(
        vae=env.vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    vae_trainer_class = variant.get("vae_trainer_class", ConvVAETrainer)
    vae_trainer = vae_trainer_class(
        env.vae,
        **variant['online_vae_trainer_kwargs']
    )
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    max_path_length = variant['max_path_length']

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=variant['evaluation_goal_sampling_mode'],
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=variant['exploration_goal_sampling_mode'],
    )

    algorithm = OnlineVaeAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        uniform_dataset=uniform_dataset,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        video_func = VideoSaveFunction(
            env,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)
    if variant['custom_goal_sampler'] == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()
