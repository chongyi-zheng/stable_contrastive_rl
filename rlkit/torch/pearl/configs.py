# default PEARL experiment settings
# all experiments should modify these settings only as needed
default_config = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=5, # dimension of the latent context vector
    net_size=300, # number of units per FC layer in each network
    path_to_weights=None, # path to pre-trained weights to load into networks
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
    ),
    algo_params=dict(
        meta_batch=16, # number of tasks to average the gradient across
        num_iterations=500, # number of data sampling / training iterates
        num_initial_steps=2000, # number of transitions collected per task before training
        num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=400, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=400, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=2000, # number of meta-gradient steps taken per iteration
        num_evals=2, # number of independent evals
        num_steps_per_eval=600,  # nuumber of transitions to eval on
        batch_size=256, # number of transitions in the RL batch
        embedding_batch_size=64, # number of transitions in the context batch
        embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3e-4,
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        use_information_bottleneck=True, # False makes latent context deterministic
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=1, # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False, # recurrent or permutation-invariant encoder
        dump_eval_paths=False, # whether to save evaluation trajectories
    ),
    util_params=dict(
        base_log_dir='output',
        use_gpu=True,
        gpu_id=0,
        debug=False, # debugging triggers printing and writes logs to debug directory
        docker=False, # TODO docker is not yet supported
    )
)


default_trainer_config = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=5, # dimension of the latent context vector
    net_size=300, # number of units per FC layer in each network
    path_to_weights=None, # path to pre-trained weights to load into networks
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
    ),
    trainer_kwargs=dict(
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3e-4,
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        use_information_bottleneck=True, # False makes latent context deterministic
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        recurrent=False, # recurrent or permutation-invariant encoder
        discount=0.99, # RL discount factor
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
    ),
    algo_params=dict(
        meta_batch=16, # number of tasks to average the gradient across
        num_iterations=500, # number of data sampling / training iterates
        num_initial_steps=2000, # number of transitions collected per task before training
        num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=400, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=400, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=2000, # number of meta-gradient steps taken per iteration
        num_evals=2, # number of independent evals
        num_steps_per_eval=600,  # nuumber of transitions to eval on
        batch_size=256, # number of transitions in the RL batch
        embedding_batch_size=64, # number of transitions in the context batch
        embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200, # max path length for this environment
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=1, # how many exploration trajs to collect before beginning posterior sampling at test time
        dump_eval_paths=False, # whether to save evaluation trajectories
    ),
    util_params=dict(
        base_log_dir='output',
        use_gpu=True,
        gpu_id=0,
        debug=False, # debugging triggers printing and writes logs to debug directory
        docker=False, # TODO docker is not yet supported
    )
)


default_awac_trainer_config = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=5,  # dimension of the latent context vector
    env_params=dict(
        n_tasks=2,  # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
    ),
    qf_kwargs=dict(
        hidden_sizes=[300, 300, 300],
    ),
    policy_kwargs=dict(
        hidden_sizes=[300, 300, 300],
    ),
    trainer_kwargs=dict(
        soft_target_tau=0.005,  # for SAC target network update
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        context_lr=3e-4,
        kl_lambda=.1,  # weight on KL divergence term in encoder loss
        beta=1,
        alpha=0,
        rl_weight=1.0,
        use_awr_update=True,
        use_reparam_update=False,
        use_automatic_entropy_tuning=False,
        use_information_bottleneck=True, # False makes latent context deterministic
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        recurrent=False, # recurrent or permutation-invariant encoder
        discount=0.99, # RL discount factor
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        reparam_weight=0.0,
        awr_weight=1.0,
        bc_weight=0.0,
        compute_bc=False,
        awr_use_mle_for_vf=False,
        awr_sample_actions=False,
        awr_min_q=True,
    ),
    algo_params=dict(
        meta_batch=16, # number of tasks to average the gradient across
        num_iterations=500, # number of data sampling / training iterates
        num_initial_steps=2000, # number of transitions collected per task before training
        num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=400, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=400, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=2000, # number of meta-gradient steps taken per iteration
        num_evals=2, # number of independent evals
        num_steps_per_eval=600,  # nuumber of transitions to eval on
        batch_size=256, # number of transitions in the RL batch
        embedding_batch_size=64, # number of transitions in the context batch
        embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200, # max path length for this environment
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=1, # how many exploration trajs to collect before beginning posterior sampling at test time
        dump_eval_paths=False, # whether to save evaluation trajectories
    ),
    pretrain_buffer_kwargs=dict(
        max_replay_buffer_size=int(1e6),
    ),
)
