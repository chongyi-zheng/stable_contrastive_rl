import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.meta_rl_algorithm import MetaRLAlgorithm
from rlkit.core.simple_offline_rl_algorithm import (
    OfflineMetaRLAlgorithm,
)
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.envs.pearl_envs import ENVS, register_pearl_envs
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.util.io import load_local_or_remote_file
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.pearl.agent import PEARLAgent
from rlkit.torch.pearl.networks import MlpEncoder, DummyMlpEncoder, MlpDecoder
from rlkit.torch.pearl.launcher_util import (
    policy_class_from_str,
    load_buffer_onto_algo,
)
from rlkit.torch.pearl.path_collector import PearlPathCollector
from rlkit.torch.pearl.pearl_awac import PearlAwacTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
)


def pearl_awac_launcher_simple(
        trainer_kwargs=None,
        algo_kwargs=None,
        qf_kwargs=None,
        policy_kwargs=None,
        context_encoder_kwargs=None,
        context_decoder_kwargs=None,
        env_name=None,
        env_params=None,
        path_loader_kwargs=None,
        latent_dim=None,
        policy_class="TanhGaussianPolicy",
        # video/debug
        debug=False,
        use_dummy_encoder=False,
        networks_ignore_context=False,
        use_ground_truth_context=False,
        # Pre-train params
        pretrain_rl=False,
        pretrain_offline_algo_kwargs=None,
        pretrain_buffer_kwargs=None,
        load_buffer_kwargs=None,
        saved_tasks_path=None,
        # PEARL
        n_train_tasks=0,
        n_eval_tasks=0,
        use_data_collectors=False,
        use_next_obs_in_context=False,
):
    pretrain_buffer_kwargs = pretrain_buffer_kwargs or {}
    context_decoder_kwargs = context_decoder_kwargs or {}
    pretrain_offline_algo_kwargs = pretrain_offline_algo_kwargs or {}
    register_pearl_envs()
    env_params = env_params or {}
    context_encoder_kwargs = context_encoder_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    path_loader_kwargs = path_loader_kwargs or {}

    base_env = ENVS[env_name](**env_params)
    if saved_tasks_path:
        task_data = load_local_or_remote_file(
            saved_tasks_path, file_type='joblib')
        tasks = task_data['tasks']
        train_task_idxs = task_data['train_task_indices']
        eval_task_idxs = task_data['eval_task_indices']
        base_env.tasks = tasks
        task_indices = base_env.get_all_task_idx()
    else:
        tasks = base_env.tasks
        task_indices = base_env.get_all_task_idx()
        train_task_idxs = list(task_indices[:n_train_tasks])
        eval_task_idxs = list(task_indices[-n_eval_tasks:])
    train_tasks = [tasks[i] for i in train_task_idxs]
    eval_tasks = [tasks[i] for i in eval_task_idxs]
    if use_ground_truth_context:
        latent_dim = len(train_tasks[0])
    expl_env = NormalizedBoxEnv(base_env)

    reward_dim = 1

    if debug:
        algo_kwargs['max_path_length'] = 50
        algo_kwargs['batch_size'] = 5
        algo_kwargs['num_epochs'] = 5
        algo_kwargs['num_eval_steps_per_epoch'] = 100
        algo_kwargs['num_expl_steps_per_train_loop'] = 100
        algo_kwargs['num_trains_per_train_loop'] = 10
        algo_kwargs['min_num_steps_before_training'] = 100

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    if use_next_obs_in_context:
        context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim
    else:
        context_encoder_input_dim = obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2

    def create_qf():
        return ConcatMlp(
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
            **qf_kwargs
        )

    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()

    if isinstance(policy_class, str):
        policy_class = policy_class_from_str(policy_class)
    policy = policy_class(
        obs_dim=obs_dim + latent_dim,
        action_dim=action_dim,
        **policy_kwargs,
    )
    encoder_class = DummyMlpEncoder if use_dummy_encoder else MlpEncoder
    context_encoder = encoder_class(
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        hidden_sizes=[200, 200, 200],
        use_ground_truth_context=use_ground_truth_context,
        **context_encoder_kwargs
    )
    context_decoder = MlpDecoder(
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
        **context_decoder_kwargs
    )
    reward_predictor = ConcatMlp(
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
        hidden_sizes=[200, 200, 200],
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        reward_predictor,
        use_next_obs_in_context=use_next_obs_in_context,
        _debug_ignore_context=networks_ignore_context,
        _debug_use_ground_truth_context=use_ground_truth_context,
    )
    trainer = PearlAwacTrainer(
        agent=agent,
        env=expl_env,
        latent_dim=latent_dim,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        reward_predictor=reward_predictor,
        context_encoder=context_encoder,
        context_decoder=context_decoder,
        _debug_ignore_context=networks_ignore_context,
        _debug_use_ground_truth_context=use_ground_truth_context,
        **trainer_kwargs
    )
    if use_data_collectors:
        eval_env = NormalizedBoxEnv(ENVS[env_name](**env_params))
        # raise NotImplementedError()
        eval_env.tasks = expl_env.tasks  # does this work?
        eval_policy = MakeDeterministic(policy)
        eval_path_collector = PearlPathCollector(eval_env, eval_policy)
        expl_policy = policy
        expl_path_collector = PearlPathCollector(expl_env, expl_policy)
        algo_kwargs.pop('save_replay_buffer')
        algo_kwargs.pop('num_iterations_with_reward_supervision')
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100,
            **algo_kwargs
        )
    else:
        algorithm = MetaRLAlgorithm(
            agent=agent,
            env=expl_env,
            trainer=trainer,
            # exploration_env=expl_env,
            # evaluation_env=eval_env,
            # exploration_data_collector=expl_path_collector,
            # evaluation_data_collector=eval_path_collector,
            train_task_indices=train_task_idxs,
            eval_task_indices=eval_task_idxs,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            # nets=[agent, qf1, qf2, vf],
            # latent_dim=latent_dim,
            use_next_obs_in_context=use_next_obs_in_context,
            use_ground_truth_context=use_ground_truth_context,
            **algo_kwargs
        )

    if load_buffer_kwargs:
        load_buffer_onto_algo(
            algorithm.replay_buffer,
            algorithm.enc_replay_buffer,
            **load_buffer_kwargs)
    if path_loader_kwargs:
        replay_buffer = algorithm.replay_buffer.task_buffers[0]
        enc_replay_buffer = algorithm.enc_replay_buffer.task_buffers[0]
        demo_test_buffer = EnvReplayBuffer(
            env=expl_env, **pretrain_buffer_kwargs)
        path_loader = MDPPathLoader(
            trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=enc_replay_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos()

    pretrain_algo = OfflineMetaRLAlgorithm(
        replay_buffer=algorithm.replay_buffer,
        task_embedding_replay_buffer=algorithm.enc_replay_buffer,
        trainer=trainer,
        train_tasks=train_task_idxs,
        **pretrain_offline_algo_kwargs
    )
    pretrain_algo.to(ptu.device)
    if pretrain_rl:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain.csv', relative_to_snapshot_dir=True
        )
        pretrain_algo.train()
        logger.remove_tabular_output(
            'pretrain.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True,
        )
    algorithm.to(ptu.device)

    algorithm.train()


