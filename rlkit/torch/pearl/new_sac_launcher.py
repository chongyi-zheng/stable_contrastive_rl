# import roboverse
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.multitask_replay_buffer import MultiTaskReplayBuffer
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.envs.pearl_envs import ENVS, register_pearl_envs
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.data_collector.joint_path_collector import \
    JointPathCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.pearl.agent import PEARLAgent, MakePEARLAgentDeterministic
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.pearl.diagnostics import get_diagnostics
from rlkit.torch.pearl.networks import MlpEncoder
from rlkit.torch.pearl.path_collector import PearlPathCollector
from rlkit.torch.pearl.pearl_algorithm import PearlAlgorithm
from rlkit.torch.pearl.pearl_trainer import PEARLSoftActorCriticTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy


def pearl_sac_experiment(
        qf_kwargs=None,
        vf_kwargs=None,
        trainer_kwargs=None,
        algo_kwargs=None,
        context_encoder_kwargs=None,
        policy_kwargs=None,
        policy_path=None,
        normalize_env=True,
        env_name=None,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        env_params=None,
        add_env_demos=False,
        path_loader_kwargs=None,
        env_demo_path=None,
        env_offpolicy_data_path=None,
        add_env_offpolicy_data=False,
        exploration_kwargs=None,
        expl_path_collector_kwargs=None,
        pearl_buffer_kwargs=None,
        name_to_eval_path_collector_kwargs=None,
        name_to_expl_path_collector_kwargs=None,
        replay_buffer_class=EnvReplayBuffer,
        replay_buffer_kwargs=None,
        use_validation_buffer=False,
        pretrain_policy=False,
        pretrain_rl=False,
        train_rl=False,
        path_loader_class=MDPPathLoader,
        latent_dim=None,
        # video/debug
        debug=False,
        save_video=False,
        presampled_goals=None,
        renderer_kwargs=None,
        image_env_kwargs=None,
        save_paths=False,
        load_demos=False,
        load_env_dataset_demos=False,
        save_initial_buffers=False,
        save_pretrained_algorithm=False,
        ignore_overlapping_train_and_test=False,
        _debug_do_not_sqrt=False,
        # PEARL
        n_train_tasks=0,
        n_eval_tasks=0,
        path_to_weights=None,
        util_params=None,
        use_data_collectors=False,
        use_next_obs_in_context=False,
):
    register_pearl_envs()
    env_kwargs = env_kwargs or {}
    env_params = env_params or {}
    path_loader_kwargs = path_loader_kwargs or {}
    exploration_kwargs = exploration_kwargs or {}
    replay_buffer_kwargs = replay_buffer_kwargs or {}
    context_encoder_kwargs = context_encoder_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    base_env = ENVS[env_name](**env_params)
    expl_env = NormalizedBoxEnv(base_env)
    eval_env = NormalizedBoxEnv(ENVS[env_name](**env_params))
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
    action_dim = eval_env.action_space.low.size

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
    vf = ConcatMlp(
        input_size=obs_dim + latent_dim,
        output_size=1,
        **vf_kwargs
    )

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + latent_dim,
        action_dim=action_dim,
        **policy_kwargs,
    )
    context_encoder = MlpEncoder(
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        **context_encoder_kwargs
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
        _debug_do_not_sqrt=_debug_do_not_sqrt,
    )
    trainer = PEARLSoftActorCriticTrainer(
        latent_dim=latent_dim,
        agent=agent,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        reward_predictor=reward_predictor,
        context_encoder=context_encoder,
        **trainer_kwargs
    )
    task_indices = expl_env.get_all_task_idx()
    train_task_indices = task_indices[:n_train_tasks]
    test_task_indices = task_indices[-n_eval_tasks:]
    if (
            n_train_tasks + n_eval_tasks > len(task_indices)
            and not ignore_overlapping_train_and_test
    ):
        raise ValueError("Your test and train overlap!")
    eval_policy = MakePEARLAgentDeterministic(agent)
    expl_policy = agent

    replay_buffer = MultiTaskReplayBuffer(
        env=expl_env,
        task_indices=task_indices,
        **replay_buffer_kwargs
    )
    enc_replay_buffer = MultiTaskReplayBuffer(
        env=expl_env,
        task_indices=task_indices,
        **replay_buffer_kwargs
    )
    pearl_replay_buffer = PearlReplayBuffer(
        replay_buffer,
        enc_replay_buffer,
        train_task_indices=train_task_indices,
        **pearl_buffer_kwargs
    )

    eval_path_collectors = {
        'train/' + name: PearlPathCollector(
            eval_env, eval_policy, train_task_indices, pearl_replay_buffer, **kwargs)
        for name, kwargs in name_to_eval_path_collector_kwargs.items()
    }
    eval_path_collectors.update({
        'test/' + name: PearlPathCollector(
            eval_env, eval_policy, test_task_indices,
            pearl_replay_buffer,
            **kwargs)
        for name, kwargs in name_to_eval_path_collector_kwargs.items()
    })
    eval_path_collector = JointPathCollector(eval_path_collectors)
    expl_path_collector = JointPathCollector({
        name: PearlPathCollector(
            expl_env, expl_policy, train_task_indices,
            pearl_replay_buffer,
            **kwargs)
        for name, kwargs in name_to_expl_path_collector_kwargs.items()
    })

    diagnostic_fns = get_diagnostics(base_env)
    algorithm = PearlAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=pearl_replay_buffer,
        train_task_indices=train_task_indices,
        test_task_indices=test_task_indices,
        evaluation_get_diagnostic_functions=diagnostic_fns,
        exploration_get_diagnostic_functions=diagnostic_fns,
        **algo_kwargs
    )

    def check_data_collection_settings():
        if (
                algorithm.num_expl_steps_per_train_loop % (
                len(name_to_expl_path_collector_kwargs) * algorithm.max_path_length) != 0
        ):
            raise ValueError("# of exploration steps should divide nicely")
        if (
                algorithm.num_eval_steps_per_epoch % (len(name_to_eval_path_collector_kwargs) * algorithm.max_path_length) != 0
        ):
            raise ValueError("# of eval steps should divide nicely")
    check_data_collection_settings()

    algorithm.to(ptu.device)
    algorithm.train()
