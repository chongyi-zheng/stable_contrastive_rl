def process_args(variant):
    if variant.get("debug", False):
        variant.update(dict(
            num_pretraining_steps = int(1e3),
            max_steps = int(1e3),
        ))
        variant.get('config', {}).update(dict(
            replay_buffer_size = int(1e6),
        ))

def pr_experiment(variant):
    import os

    import numpy as np
    import tqdm
    import ml_collections
    from tensorboardX import SummaryWriter

    from jaxrl.agents import AWACLearner, SACLearner
    from learner import Learner
    from jaxrl.datasets import ReplayBuffer
    from jaxrl.datasets.dataset_utils import make_env_and_dataset
    from jaxrl.evaluation import evaluate
    from jaxrl.utils import make_env
    from jaxrl.datasets.awac_dataset import AWACDataset

    from rlkit.core import logger
    from jaxrl.datasets.dataset import Batch

    variant = ml_collections.ConfigDict(variant)
    kwargs = variant.config

    seed = variant.seedid # seed
    save_dir = logger.get_snapshot_dir()

    summary_writer = SummaryWriter(
        os.path.join(save_dir, 'tb', str(seed)))

    if variant.save_video:
        video_train_folder = os.path.join(save_dir, 'video', 'train')
        video_eval_folder = os.path.join(save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env, dataset = make_env_and_dataset(variant.env_name, seed,
                                        variant.dataset_name, video_train_folder, )

    eval_env = make_env(variant.env_name, seed + 42, video_eval_folder, )

    np.random.seed(seed)

    kwargs = dict(variant.config)
    algo = kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    if algo == 'sac':
        agent = SACLearner(seed,
                           env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'awac':
        agent = AWACLearner(seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'pr':
        max_reward = max(0.0, np.max(dataset.rewards))
        agent = Learner(seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis],
                            # max_reward,
                            **kwargs)
    else:
        raise NotImplementedError()

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or variant.max_steps)
    replay_buffer.initialize_with_dataset(dataset, variant.init_dataset_size)

    # demo_train_dataset = AWACDataset(variant.env_name, dataset_names=('awac_demo',))
    # demo_train_buffer = ReplayBuffer(env.observation_space, action_dim,
    #                              replay_buffer_size or variant.max_steps)
    # demo_train_buffer.initialize_with_dataset(demo_train_dataset, variant.init_dataset_size)

    # demo_test_dataset = AWACDataset(variant.env_name, dataset_names=('awac_demo',), validation_split=True)
    # demo_test_buffer = ReplayBuffer(env.observation_space, action_dim,
    #                              replay_buffer_size or variant.max_steps)
    # demo_test_buffer.initialize_with_dataset(demo_test_dataset, variant.init_dataset_size)

    eval_returns = []
    observation, done = env.reset(), False

    action_noise_scale = variant.get("action_noise_scale", 0)
    exploration_temperature = variant.get("exploration_temperature", 1.0)

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(range(1 - variant.num_pretraining_steps,
                             variant.max_steps + 1),
                       smoothing=0.1,
                       disable=not variant.tqdm):
        if i == 1 - variant.num_pretraining_steps or i % 100000 == 0:
            agent.actor.save(os.path.join(save_dir, 'itr_%d.jax' % i))

        if i >= 1:
            action = agent.sample_actions(observation, temperature=exploration_temperature)
            noise = np.random.normal(size=action.shape) * action_noise_scale
            action = np.clip(action + noise, -1, 1)
            next_observation, reward, done, info = env.step(action)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(observation, action, reward, mask,
                                 float(done), next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'training/{k}', v,
                                              info['total']['timesteps'])
        else:
            info = {}
            info['total'] = {'timesteps': i}

        batch = replay_buffer.sample(variant.batch_size)
        if 'antmaze' in variant.env_name:
            batch = Batch(observations=batch.observations,
                     actions=batch.actions,
                     rewards=batch.rewards - 1,
                     masks=batch.masks,
                     next_observations=batch.next_observations)
        update_info = agent.update(batch)

        # demo_train_batch = demo_train_buffer.sample(variant.batch_size)
        # demo_test_batch = demo_test_buffer.sample(variant.batch_size)
        # update_info = {
        #     **update_info,
        #     **agent.eval(demo_train_batch, "train/",),
        #     **agent.eval(demo_test_batch, "test/",)
        # }

        if i % variant.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % variant.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, variant.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return'], eval_stats['success']))
            np.savetxt(os.path.join(save_dir, 'progress.csv'),
                       eval_returns,
                       fmt=['%d', '%.1f', '%.1f'],
                       delimiter=",", header="expl/num steps total,eval/Average Returns,eval/success")
