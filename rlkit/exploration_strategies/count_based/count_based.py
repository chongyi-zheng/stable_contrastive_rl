import numpy as np
from collections import Counter
from rlkit.exploration_strategies.count_based.compute_obs_mean_std import compute_obs_mean_std
from rlkit.util.np_util import bin2int, softmax


class CountExploration:
    def __init__(self,
                 env,
                 hash_dim=16,
                 observation_key='observation',
                 num_samples=5000,
                 normalize_obs=False,
                 obs_mean=None,
                 obs_std=None,
                 hashing_matrix=None,
                 ):
        self.obs_dim = env.observation_space.spaces[observation_key].low.size
        self.hash_dim = hash_dim
        if hashing_matrix is not None:
            self.hashing_matrix = hashing_matrix
        else:
            self.hashing_matrix = np.reshape(np.random.normal(0, 1, self.obs_dim * self.hash_dim),
                                             (self.obs_dim, self.hash_dim))
        self.counts = Counter()
        obs_dim = env.observation_space.spaces[observation_key].low.size
        if normalize_obs and (obs_mean is None or obs_std is None):
            obs_mean, obs_std = compute_obs_mean_std(env, N=num_samples, observation_key=observation_key)
        elif not normalize_obs:
            obs_mean, obs_std = np.zeros(obs_dim), np.ones(obs_dim)
        else:
            raise NotImplementedError('invalid normalization params')
        self.obs_mean = obs_mean
        self.obs_std = obs_std + .00001
        self.env = env
        self.observation_key = observation_key

    def increment_counts(self, observations):
        bins = self._observations_to_bins(observations)
        for b in bins:
            self.counts[b] += 1

    def get_counts(self, observations):
        bins = self._observations_to_bins(observations)
        return np.array([self.counts[bin] for bin in bins], dtype=np.float32)

    def _observations_to_bins(self, observations):
        observations = np.divide(observations - self.obs_mean, self.obs_std)
        mul = np.dot(observations, self.hashing_matrix)
        sn = np.where(mul > 0, 1, 0)
        code = bin2int(sn.T)
        if code.shape == ():
            code = np.array([code])
        return code

    def compute_count_based_reward(self, observations):
        new_obs_counts = self.get_counts(observations)
        new_rewards = ((new_obs_counts + .0001) ** (-1 / 2)).reshape(-1, 1)
        return new_rewards

    def clear_counter(self):
        self.counts = Counter()


class CountExplorationCountGoalSampler(CountExploration):
    '''
    Steps:
    1. take in a bunch of randomly sampled goals
    2. compute the count_based reward for those goals
    3. compute softmax prob dist from those rewards
    4. use the softmax dist to pick one of the goals you had sampled originally

    goal space has to be equal to obs space ie use achieved goals to hash
    '''

    def __init__(self,
                 theta=1.0,
                 replay_buffer=None,
                 goal_key='desired_goal',
                 num_count_based_goals=100,
                 use_softmax=True,
                 **kwargs
                 ):
        self.theta = theta
        self.goal_key = goal_key
        self.num_count_based_goals = num_count_based_goals
        self.use_softmax = use_softmax
        self.replay_buffer = replay_buffer
        super().__init__(**kwargs)

    def get_count_based_goal(self):
        if len(self.counts.keys()) == 0:
            # initially sample a random goal
            return self.env.sample_goal()
        if self.replay_buffer is not None:
            indices = self.replay_buffer._sample_indices(self.num_count_based_goals)
            goals = self.replay_buffer._next_obs[self.observation_key][indices]
        else:
            goals = self.env.sample_goals(self.num_count_based_goals)[self.goal_key]
        count_based_rewards = self.compute_count_based_reward(goals)
        if self.use_softmax:
            probabilities = softmax(count_based_rewards, self.theta).reshape(-1)
        else:
            probabilities = np.ones(self.num_count_based_goals) * 1 / self.num_count_based_goals
        idxs = np.array(list(range(self.num_count_based_goals)))
        idx = np.random.choice(idxs, p=probabilities)
        return goals[idx]

