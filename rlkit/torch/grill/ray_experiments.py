def full_experiment_variant_preprocess(variant):
    train_vae_variant = variant['train_vae_variant']
    grill_variant = variant['grill_variant']
    if 'env_id' in variant:
        assert 'env_class' not in variant
        env_id = variant['env_id']
        grill_variant['env_id'] = env_id
        train_vae_variant['generate_vae_dataset_kwargs']['env_id'] = env_id
    else:
        env_class = variant['env_class']
        env_kwargs = variant['env_kwargs']
        train_vae_variant['generate_vae_dataset_kwargs']['env_class'] = (
            env_class
        )
        train_vae_variant['generate_vae_dataset_kwargs']['env_kwargs'] = (
            env_kwargs
        )
        grill_variant['env_class'] = env_class
        grill_variant['env_kwargs'] = env_kwargs
    init_camera = variant.get('init_camera', None)
    imsize = variant.get('imsize', 84)
    train_vae_variant['generate_vae_dataset_kwargs']['init_camera'] = (
        init_camera
    )
    train_vae_variant['generate_vae_dataset_kwargs']['imsize'] = imsize
    train_vae_variant['imsize'] = imsize
    grill_variant['imsize'] = imsize
    grill_variant['init_camera'] = init_camera

def grill_her_twin_sac_online_vae_full_experiment():
    return [
        (train_vae_and_update_variant, 'vae_progress.csv'),
        (grill_her_twin_sac_experiment_online_vae, 'progress.csv'),
    ]

def grill_her_sac_full_experiment():
    return [(grill_her_sac_experiment, 'progress.csv')]

def grill_her_sac_experiment(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
    from rlkit.torch.sac.sac import SACTrainer
    from rlkit.torch.her.her import HERTrainer
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.samplers.data_collector import GoalConditionedPathCollector
    from rlkit.torch.grill.launcher import (
        grill_preprocess_variant,
        get_envs,
        get_exploration_strategy
    )

    full_experiment_variant_preprocess(variant)
    variant = variant['grill_variant']
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    eval_policy = MakeDeterministic(policy)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = GoalConditionedPathCollector(
        env,
        eval_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedPathCollector(
        env,
        exploration_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    return algorithm

def grill_her_twin_sac_experiment_online_vae(vae_exp, variant):

    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.torch.grill.launcher import grill_preprocess_variant, get_envs
    from rlkit.torch.sac.sac import SACTrainer
    from rlkit.torch.her.her import HERTrainer
    from rlkit.samplers.data_collector import (
        GoalConditionedPathCollector,
        VAEWrappedEnvPathCollector,
    )
    from rlkit.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm
    from rlkit.torch.sac.policies import MakeDeterministic

    variant = variant['grill_variant']
    vae = vae_exp.vae_trainer.model
    variant['vae_path'] = vae
    grill_preprocess_variant(variant)
    env = get_envs(variant)

    uniform_dataset_fn = variant.get('generate_uniform_dataset_fn', None)
    if uniform_dataset_fn:
        uniform_dataset = uniform_dataset_fn(
            **variant['generate_uniform_dataset_kwargs']
        )
    else:
        uniform_dataset = None
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
    env.vae = vae

    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    vae_trainer = ConvVAETrainer(
        variant['vae_train_data'],
        variant['vae_test_data'],
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

    if variant['custom_goal_sampler'] == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    return algorithm


def vae_experiment(variant):
    return train_vae_and_update_variant(variant)


def train_vae_and_update_variant(variant):
    from rlkit.torch.grill.launcher import generate_vae_dataset
    full_experiment_variant_preprocess(variant)
    grill_variant = variant['grill_variant']
    train_vae_variant = variant['train_vae_variant']
    if grill_variant.get('vae_path', None) is None:
        vae_experiment, train_data, test_data = train_vae(**train_vae_variant)
        grill_variant['vae_train_data'] = train_data
        grill_variant['vae_test_data'] = test_data
    else:
        #TODO: steven add preloaded vae support
        if grill_variant.get('save_vae_data', False):
            vae_train_data, vae_test_data, info = generate_vae_dataset(
                train_vae_variant['generate_vae_dataset_kwargs']
            )
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data
    return vae_experiment


def train_vae(beta, representation_size, imsize, num_epochs, save_period,
              generate_vae_dataset_fctn=None, beta_schedule_kwargs=None,
              decoder_activation=None, vae_kwargs=None,
              generate_vae_dataset_kwargs=None, algo_kwargs=None,
              use_spatial_auto_encoder=False, vae_class=None,
              dump_skew_debug_plots=False):
    from rlkit.util.ml_util import PiecewiseLinearSchedule
    from rlkit.torch.vae.conv_vae import (
        ConvVAE,
        SpatialAutoEncoder,
        AutoEncoder,
    )
    import rlkit.torch.vae.conv_vae as conv_vae
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.torch.vae.vae_experiment import VAEExperiment
    from rlkit.pythonplusplus import identity
    from rlkit.torch.grill.launcher import generate_vae_dataset
    import torch
    if vae_kwargs is None:
        vae_kwargs = {}
    if generate_vae_dataset_kwargs is None:
        generate_vae_dataset_kwargs = {}
    if algo_kwargs is None:
        algo_kwargs = {}
    if generate_vae_dataset_fctn is None:
        generate_vae_dataset_fctn = generate_vae_dataset
    if vae_class is None:
        vae_class = ConvVAE
    if beta_schedule_kwargs is not None:
        beta_schedule = PiecewiseLinearSchedule(**beta_schedule_kwargs)
    else:
        beta_schedule = None
    if decoder_activation == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
    architecture = vae_kwargs.get('architecture', None)
    if not architecture and imsize == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and imsize == 48:
        architecture = conv_vae.imsize48_default_architecture
    vae_kwargs['architecture'] = architecture
    vae_kwargs['imsize'] = imsize

    if algo_kwargs.get('is_auto_encoder', False):
        m = AutoEncoder(representation_size,
                        decoder_output_activation=decoder_activation,
                        **vae_kwargs)
    elif use_spatial_auto_encoder:
        m = SpatialAutoEncoder(
            representation_size,
            decoder_output_activation=decoder_activation,
            **vae_kwargs)
    else:
        m = vae_class(representation_size,
                      decoder_output_activation=decoder_activation,
                      **vae_kwargs)
    train_data, test_data, info = generate_vae_dataset_fctn(
        generate_vae_dataset_kwargs
    )
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **algo_kwargs)
    vae_exp = VAEExperiment(t, num_epochs, save_period, dump_skew_debug_plots)
    return vae_exp, train_data, test_data
