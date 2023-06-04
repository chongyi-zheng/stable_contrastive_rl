import numpy as np

def compute_obs_mean_std(
        env,
        N=100,
        oracle_dataset=True, #does it make sense to not use oracle data?
        observation_key='observation',
        n_random_steps=50,
):
    env.reset()
    obs_dim = env.observation_space.spaces[observation_key].low.size
    dataset = np.zeros((N, obs_dim))
    for i in range(N):
        if oracle_dataset:
            goal = env.sample_goal()
            env.set_to_goal(goal)
        else:
            env.reset()
            for _ in range(n_random_steps):
                env.step(env.action_space.sample())
        obs = env._get_obs()
        obs = obs[observation_key]
        dataset[i, :] = obs
        print('Step '+str(i))
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    return mean, std