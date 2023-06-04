import numpy as np

from rlkit.policies.base import Policy

class SubgoalPolicyWrapper(Policy):
    def __init__(
            self,
            wrapped_policy,
            env,
            episode_length,
            num_subgoals_per_episode=2,
    ):
        self._wrapped_policy = wrapped_policy
        self._env = env
        self.episode_length = episode_length
        self.num_subgoals_per_episode = num_subgoals_per_episode
        self.subepisode_length = self.episode_length // self.num_subgoals_per_episode
        self.episode_timer = 0
        self.episode_goal = None
        self.episode_subgoals = None

    def get_action(self, observation):
        ob, goal = np.split(observation, 2)
        if self.episode_goal is None:
            self.episode_goal = goal
            self.episode_subgoals = self._env.generate_expert_subgoals(ob, goal, self.num_subgoals_per_episode)
            self._env.update_subgoals(self.episode_subgoals)

        curr_subgoal_idx = self.episode_timer // self.subepisode_length
        self._env.update_subgoals(self.episode_subgoals[curr_subgoal_idx:])

        self.episode_timer += 1
        subgoal = self.episode_subgoals[curr_subgoal_idx]
        return self._wrapped_policy.get_action(np.hstack((ob, subgoal)))

    def reset(self):
        # print("reset")
        self._wrapped_policy.reset()
        self.episode_timer = 0
        self.episode_goal = None
        self.episode_subgoals = None

    def __call__(self, *input, **kwargs):
        return self._wrapped_policy(*input, **kwargs)

    def __getattr__(self, attr):
        if attr == '_wrapped_policy':
            raise AttributeError()
        return getattr(self._wrapped_policy, attr)