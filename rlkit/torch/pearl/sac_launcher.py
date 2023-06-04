# import roboverse
import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.meta_rl_algorithm import MetaRLAlgorithm
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.envs.pearl_envs import ENVS, register_pearl_envs
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.pearl.agent import PEARLAgent
from rlkit.torch.pearl.encoder import MlpEncoder
from rlkit.torch.pearl.path_collector import PearlPathCollector
from rlkit.torch.pearl.pearl_trainer import PEARLSoftActorCriticTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
)


def pearl_experiment(
        qf_kwargs=None,
        vf_kwargs=None,
        trainer_kwargs=None,
        algo_kwargs=None,
        context_encoder_kwargs=None,
        policy_class=None,
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
    # expl_env = make(env_id, env_class, env_kwargs, normalize_env)
    # eval_env = make(env_id, env_class, env_kwargs, normalize_env)
    expl_env = NormalizedBoxEnv(ENVS[env_name](**env_params))
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

    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

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
    vf = ConcatMlp(
        input_size=obs_dim + latent_dim,
        output_size=1,
        **vf_kwargs
    )
    # target_qf1 = create_qf()
    # target_qf2 = create_qf()

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + latent_dim,
        action_dim=action_dim,
        **policy_kwargs,
    )
    context_encoder = MlpEncoder(
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        hidden_sizes=[200, 200, 200],
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
        # target_qf1=target_qf1,
        # target_qf2=target_qf2,
        **trainer_kwargs
    )
    tasks = expl_env.get_all_task_idx()
    if use_data_collectors:
        eval_policy = MakeDeterministic(policy)
        eval_path_collector = PearlPathCollector(eval_env, eval_policy)
        expl_policy = policy
        expl_path_collector = PearlPathCollector(expl_env, expl_policy)
        # algorithm = TorchBatchRLAlgorithm(
        #     trainer=trainer,
        #     exploration_env=expl_env,
        #     evaluation_env=eval_env,
        #     exploration_data_collector=expl_path_collector,
        #     evaluation_data_collector=eval_path_collector,
        #     **algo_kwargs
        # )
        algorithm = MetaRLAlgorithm(
            agent=agent,
            env=expl_env,
            trainer=trainer,
            # exploration_env=expl_env,
            # evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            train_task_indices=list(tasks[:n_train_tasks]),
            eval_task_indices=list(tasks[-n_eval_tasks:]),
            # nets=[agent, qf1, qf2, vf],
            # latent_dim=latent_dim,
            use_next_obs_in_context=use_next_obs_in_context,
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
            train_task_indices=list(tasks[:n_train_tasks]),
            eval_task_indices=list(tasks[-n_eval_tasks:]),
            # nets=[agent, qf1, qf2, vf],
            # latent_dim=latent_dim,
            use_next_obs_in_context=use_next_obs_in_context,
            **algo_kwargs
        )
    saved_path = logger.save_extra_data(
        data=dict(
            tasks=expl_env.tasks,
            train_task_indices=list(tasks[:n_train_tasks]),
            eval_task_indices=list(tasks[-n_eval_tasks:]),
        ),
        file_name='tasks',
    )
    print('saved tasks to', saved_path)

    algorithm.to(ptu.device)

    algorithm.train()
