from rlkit.samplers.data_collector import VAEWrappedEnvPathCollector
from rlkit.visualization.video import VideoSaveFunction
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm

from rlkit.launchers.rl_exp_launcher_util import (
    preprocess_rl_variant,
    get_envs,
    get_exploration_strategy,
    get_presampled_goals_path,
    get_video_save_func,
)


def td3_experiment_online_vae(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.torch.td3.td3 import TD3
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.exploration_strategies.gaussian_and_epsilon import \
        GaussianAndEpsilonStrategy

    preprocess_rl_variant(variant)
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
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        # **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        # **variant['policy_kwargs']
    )

    es = GaussianAndEpsilonStrategy(
        action_space=env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
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

    trainer = TD3(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['td3_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        variant['evaluation_goal_sampling_mode'],
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        variant['exploration_goal_sampling_mode'],
        env,
        expl_policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
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


def twin_sac_experiment_online_vae(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer

    preprocess_rl_variant(variant)
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
        variant['evaluation_goal_sampling_mode'],
        env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        variant['exploration_goal_sampling_mode'],
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
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


def td3_experiment_online_vae_exploring(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.torch.her.online_vae_joint_algo import OnlineVaeHerJointAlgo
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.torch.td3.td3 import TD3
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    preprocess_rl_variant(variant)
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
        **variant['qf_kwargs'],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    exploring_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    exploring_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    exploring_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    exploring_exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=exploring_policy,
    )

    vae = env.vae
    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    if variant.get('use_replay_buffer_goals', False):
        env.replay_buffer = replay_buffer
        env.use_replay_buffer_goals = True

    vae_trainer_kwargs = variant.get('vae_trainer_kwargs')
    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'],
                       **vae_trainer_kwargs)

    control_algorithm = TD3(
        env=env,
        training_env=env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    exploring_algorithm = TD3(
        env=env,
        training_env=env,
        qf1=exploring_qf1,
        qf2=exploring_qf2,
        policy=exploring_policy,
        exploration_policy=exploring_exploration_policy,
        **variant['algo_kwargs']
    )

    assert 'vae_training_schedule' not in variant,\
        "Just put it in joint_algo_kwargs"
    algorithm = OnlineVaeHerJointAlgo(
        vae=vae,
        vae_trainer=t,
        env=env,
        training_env=env,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        algo1=control_algorithm,
        algo2=exploring_algorithm,
        algo1_prefix="Control_",
        algo2_prefix="VAE_Exploration_",
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['joint_algo_kwargs']
    )


    algorithm.to(ptu.device)
    vae.to(ptu.device)
    if variant.get("save_video", True):
        policy.train(False)
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)
    algorithm.train()


def td3_experiment_offpolicy_online_vae(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.torch.td3.td3 import TD3
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.exploration_strategies.gaussian_and_epsilon import \
        GaussianAndEpsilonStrategy
    from rlkit.torch.vae.online_vae_offpolicy_algorithm import OnlineVaeOffpolicyAlgorithm

    preprocess_rl_variant(variant)
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
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        # **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
        # **variant['policy_kwargs']
    )

    es = GaussianAndEpsilonStrategy(
        action_space=env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
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
    replay_buffer.representation_size = vae.representation_size

    vae_trainer_class = variant.get("vae_trainer_class", ConvVAETrainer)
    vae_trainer = vae_trainer_class(
        env.vae,
        **variant['online_vae_trainer_kwargs']
    )
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    max_path_length = variant['max_path_length']

    trainer = TD3(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['td3_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        variant['evaluation_goal_sampling_mode'],
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        variant['exploration_goal_sampling_mode'],
        env,
        expl_policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    algorithm = OnlineVaeOffpolicyAlgorithm(
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

    algorithm.pretrain()
    algorithm.train()


def tdm_td3_experiment(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.core import logger
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.state_distance.tdm_networks import TdmQf, TdmPolicy
    from rlkit.state_distance.tdm_td3 import TdmTd3
    preprocess_rl_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size

    vectorized = 'vectorized' in env.reward_type
    norm_order = env.norm_order
    variant['algo_kwargs']['tdm_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['norm_order'] = norm_order

    qf1 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    variant['replay_buffer_kwargs']['vectorized'] = vectorized
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = TdmTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        env.vae.to(ptu.device)
    if variant.get("save_video", True):
        logdir = logger.get_snapshot_dir()
        policy.train(False)
        rollout_function = rf.create_rollout_function(
            rf.tdm_rollout,
            init_tau=algorithm.max_tau,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)
    algorithm.train()


def tdm_td3_experiment_online_vae(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.state_distance.tdm_networks import TdmQf, TdmPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.torch.online_vae.online_vae_tdm_td3 import OnlineVaeTdmTd3
    preprocess_rl_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)
    vae_trainer_kwargs = variant.get('vae_trainer_kwargs')
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size

    vectorized = 'vectorized' in env.reward_type
    variant['algo_kwargs']['tdm_td3_kwargs']['tdm_kwargs'][
        'vectorized'] = vectorized

    norm_order = env.norm_order
    # variant['algo_kwargs']['tdm_td3_kwargs']['tdm_kwargs'][
    #     'norm_order'] = norm_order

    qf1 = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=norm_order,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=norm_order,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    vae = env.vae

    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']['tdm_td3_kwargs']
    td3_kwargs = algo_kwargs['td3_kwargs']
    td3_kwargs['training_env'] = env
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algo_kwargs["replay_buffer"] = replay_buffer

    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'],
                       **vae_trainer_kwargs)
    render = variant["render"]
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    algorithm = OnlineVaeTdmTd3(
        online_vae_kwargs=dict(
            vae=vae,
            vae_trainer=t,
            **variant['algo_kwargs']['online_vae_kwargs']
        ),
        tdm_td3_kwargs=dict(
            env=env,
            qf1=qf1,
            qf2=qf2,
            policy=policy,
            exploration_policy=exploration_policy,
            **variant['algo_kwargs']['tdm_td3_kwargs']
        ),
    )

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    if variant.get("save_video", True):
        policy.train(False)
        rollout_function = rf.create_rollout_function(
            rf.tdm_rollout,
            init_tau=algorithm._sample_max_tau_for_rollout(),
            decrement_tau=algorithm.cycle_taus_for_rollout,
            cycle_tau=algorithm.cycle_taus_for_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        env.vae.to(ptu.device)

    algorithm.train()


def tdm_twin_sac_experiment(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.state_distance.tdm_networks import (
        TdmQf, TdmVf,
        StochasticTdmPolicy,
    )
    from rlkit.state_distance.tdm_twin_sac import TdmTwinSAC
    preprocess_rl_variant(variant)
    env = get_envs(variant)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size

    vectorized = 'vectorized' in env.reward_type
    norm_order = env.norm_order
    variant['algo_kwargs']['tdm_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['vectorized'] = vectorized
    variant['vf_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['norm_order'] = norm_order
    variant['vf_kwargs']['norm_order'] = norm_order

    qf1 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    vf = TdmVf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        **variant['vf_kwargs']
    )
    policy = StochasticTdmPolicy(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    variant['replay_buffer_kwargs']['vectorized'] = vectorized
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = TdmTwinSAC(
        env,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        policy=policy,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        rollout_function = rf.create_rollout_function(
            rf.tdm_rollout,
            init_tau=algorithm._sample_max_tau_for_rollout(),
            decrement_tau=algorithm.cycle_taus_for_rollout,
            cycle_tau=algorithm.cycle_taus_for_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        env.vae.to(ptu.device)

    algorithm.train()


def active_representation_learning_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.arl.active_representation_learning_algorithm import \
        ActiveRepresentationLearningAlgorithm
    from rlkit.torch.arl.representation_wrappers import RepresentationWrappedEnv
    from multiworld.core.image_env import ImageEnv
    from rlkit.samplers.data_collector import MdpPathCollector

    preprocess_rl_variant(variant)

    model_class = variant.get('model_class')
    model_kwargs = variant.get('model_kwargs')

    model = model_class(**model_kwargs)
    model.representation_size = 4
    model.imsize = 48
    variant["vae_path"] = model

    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    env = variant["env_class"](**variant['env_kwargs'])
    image_env = ImageEnv(
        env,
        variant.get('imsize'),
        init_camera=init_camera,
        transpose=True,
        normalize=True,
    )
    env = RepresentationWrappedEnv(
        image_env,
        model,
    )

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
    obs_dim = env.observation_space.spaces[observation_key].low.size
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

    replay_buffer = ObsDictReplayBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )

    model_trainer_class = variant.get('model_trainer_class')
    model_trainer_kwargs = variant.get('model_trainer_kwargs')
    model_trainer = model_trainer_class(
        model,
        **model_trainer_kwargs,
    )
    # vae_trainer = ConvVAETrainer(
    #     env.vae,
    #     **variant['online_vae_trainer_kwargs']
    # )
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
    # trainer = HERTrainer(trainer)
    eval_path_collector = MdpPathCollector(
        env,
        MakeDeterministic(policy),
        # max_path_length,
        # observation_key=observation_key,
        # desired_goal_key=desired_goal_key,
    )
    expl_path_collector = MdpPathCollector(
        env,
        policy,
        # max_path_length,
        # observation_key=observation_key,
        # desired_goal_key=desired_goal_key,
    )

    algorithm = ActiveRepresentationLearningAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        model=model,
        model_trainer=model_trainer,
        uniform_dataset=uniform_dataset,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()


def HER_baseline_td3_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.torch.her.her_td3 import HerTd3
    from rlkit.torch.networks import MergedCNN, CNNPolicy
    import torch
    from multiworld.core.image_env import ImageEnv
    from rlkit.util.io import load_local_or_remote_file

    init_camera = variant.get("init_camera", None)
    presample_goals = variant.get('presample_goals', False)
    presampled_goals_path = get_presampled_goals_path(
        variant.get('presampled_goals_path', None))

    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    image_env = ImageEnv(
        env,
        variant.get('imsize'),
        reward_type='image_sparse',
        init_camera=init_camera,
        transpose=True,
        normalize=True,
    )
    if presample_goals:
        if presampled_goals_path is None:
            image_env.non_presampled_goal_img_is_garbage = True
            presampled_goals = variant['generate_goal_dataset_fctn'](
                env=image_env,
                **variant['goal_generation_kwargs']
            )
        else:
            presampled_goals = load_local_or_remote_file(
                presampled_goals_path
            ).item()
        del image_env
        env = ImageEnv(
            env,
            variant.get('imsize'),
            reward_type='image_distance',
            init_camera=init_camera,
            transpose=True,
            normalize=True,
            presampled_goals=presampled_goals,
        )
    else:
        env = image_env

    es = get_exploration_strategy(variant, env)

    observation_key = variant.get('observation_key', 'image_observation')
    desired_goal_key = variant.get('desired_goal_key', 'image_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    imsize=variant['imsize']
    action_dim = env.action_space.low.size
    qf1 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )
    qf2 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )

    policy = CNNPolicy(input_width=imsize,
                       input_height=imsize,
                       added_fc_input_size=0,
                       output_size=action_dim,
                       input_channels=3 * 2,
                       output_activation=torch.tanh,
                       **variant['cnn_params'],
                       )
    target_qf1 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )
    target_qf2 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )

    target_policy = CNNPolicy(input_width=imsize,
                       input_height=imsize,
                       added_fc_input_size=0,
                       output_size=action_dim,
                       input_channels=3 * 2,
                       output_activation=torch.tanh,
                       **variant['cnn_params'],
                       )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    her_kwargs = algo_kwargs['her_kwargs']
    her_kwargs['observation_key'] = observation_key
    her_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    algorithm.train()


def HER_baseline_twin_sac_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.torch.her.her_twin_sac import HerTwinSAC
    from rlkit.torch.sac.policies import TanhCNNGaussianPolicy
    from rlkit.torch.networks import MergedCNN, CNN
    import torch
    from multiworld.core.image_env import ImageEnv
    from rlkit.util.io import load_local_or_remote_file

    init_camera = variant.get("init_camera", None)
    presample_goals = variant.get('presample_goals', False)
    presampled_goals_path = get_presampled_goals_path(
        variant.get('presampled_goals_path', None))

    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    image_env = ImageEnv(
        env,
        variant.get('imsize'),
        reward_type='image_sparse',
        init_camera=init_camera,
        transpose=True,
        normalize=True,
    )
    if presample_goals:
        if presampled_goals_path is None:
            image_env.non_presampled_goal_img_is_garbage = True
            presampled_goals = variant['generate_goal_dataset_fctn'](
                env=image_env,
                **variant['goal_generation_kwargs']
            )
        else:
            presampled_goals = load_local_or_remote_file(
                presampled_goals_path
            ).item()
        del image_env
        env = ImageEnv(
            env,
            variant.get('imsize'),
            reward_type='image_distance',
            init_camera=init_camera,
            transpose=True,
            normalize=True,
            presampled_goals=presampled_goals,
        )
    else:
        env = image_env
    es = get_exploration_strategy(variant, env)

    observation_key = variant.get('observation_key', 'image_observation')
    desired_goal_key = variant.get('desired_goal_key', 'image_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    imsize=variant['imsize']
    action_dim = env.action_space.low.size
    qf1 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )
    qf2 = MergedCNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    added_fc_input_size=action_dim,
                    **variant['cnn_params']
                    )

    policy = TanhCNNGaussianPolicy(input_width=imsize,
                       input_height=imsize,
                       added_fc_input_size=0,
                       output_size=action_dim,
                       input_channels=3 * 2,
                       output_activation=torch.tanh,
                       **variant['cnn_params'],
                       )

    vf = CNN(input_width=imsize,
                    input_height=imsize,
                    output_size=1,
                    input_channels=3 * 2,
                    **variant['cnn_params']
                    )
    target_vf = CNN(input_width=imsize,
             input_height=imsize,
             output_size=1,
             input_channels=3 * 2,
             **variant['cnn_params']
             )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    her_kwargs = algo_kwargs['her_kwargs']
    her_kwargs['observation_key'] = observation_key
    her_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = HerTwinSAC(
        env,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        target_vf=target_vf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    algorithm.train()
