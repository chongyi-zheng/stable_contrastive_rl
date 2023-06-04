from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from gym.spaces import Dict, Discrete
import numpy as np

class CLearningReplayBuffer(ObsDictReplayBuffer):
    """
    Like ObsDictRelabelingBuffer but return goals/future states/random states separately
    """

    def __init__(
            self,
            max_size,
            env,
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
            goal_keys=None,
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            ob_keys_to_save=None,
            use_multitask_rewards=True,
            recompute_rewards=True,
            **kwargs
    ):
        """

        :param max_size:
        :param env:
        :param fraction_goals_rollout_goals: Default, no her
        :param fraction_resampled_goals_env_goals:  of the resampled
        goals, what fraction are sampled from the env. Reset are sampled from future.
        :param ob_keys_to_save: List of keys to save
        """
        if ob_keys_to_save is None:
            ob_keys_to_save = []
        else:  # in case it's a tuple
            ob_keys_to_save = list(ob_keys_to_save)
        if desired_goal_key not in ob_keys_to_save:
            ob_keys_to_save.append(desired_goal_key)
        if achieved_goal_key not in ob_keys_to_save:
            ob_keys_to_save.append(achieved_goal_key)
        if goal_keys is not None:
            for goal_key in goal_keys:
                if goal_key not in ob_keys_to_save:
                    ob_keys_to_save.append(goal_key)
                    # TODO: fix hack. Necessary for future-style relabeling
                    ob_keys_to_save.append(
                        goal_key.replace('desired', 'achieved')
                    )
        super().__init__(
            max_size,
            env,
            ob_keys_to_save=ob_keys_to_save,
            **kwargs
        )
        if goal_keys is None:
            self.goal_keys = [k for k in ob_keys_to_save if 'desired' in k]
        else:
            logging.warning("""
            Are you sure you want to set the goal keys manually?
            You're less likely to get bugs by setting it automatically.
            In particular, relabeling will ONLY relabel the keys in goal_keys,
            which may break if you have wrapped environments.
            For details, ask @vitchyr.
            """)
            self.goal_keys = list(goal_keys)
        assert isinstance(env.observation_space, Dict)
        assert 0 <= fraction_goals_rollout_goals
        assert 0 <= fraction_goals_env_goals
        assert 0 <= fraction_goals_rollout_goals + fraction_goals_env_goals
        assert fraction_goals_rollout_goals + fraction_goals_env_goals <= 1
        self.fraction_goals_rollout_goals = fraction_goals_rollout_goals
        self.fraction_goals_env_goals = fraction_goals_env_goals
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key
        self.recompute_rewards = recompute_rewards
        self.use_multitask_rewards = use_multitask_rewards

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        num_env_goals = int(batch_size * self.fraction_goals_env_goals)
        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_future_goals = batch_size - (num_env_goals + num_rollout_goals)
        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_env_goals > 0:
            env_goals = self._sample_goals_from_env(num_env_goals)
            last_env_goal_idx = num_rollout_goals + num_env_goals

            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx] = \
                    env_goals[goal_key]
                new_next_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx] = \
                    env_goals[goal_key]
        if num_future_goals > 0:
            future_obs_idxs = self._get_future_obs_indices(indices[-num_future_goals:])
            for goal_key in self.goal_keys:
                achieved_k = goal_key.replace('desired', 'achieved')
                new_obs_dict[goal_key][-num_future_goals:] = (
                    self._next_obs[achieved_k][future_obs_idxs]
                )
                new_next_obs_dict[goal_key][-num_future_goals:] = (
                    self._next_obs[achieved_k][future_obs_idxs]
                )

        future_states_dict = {}
        future_obs_idxs = self._get_future_obs_indices(indices)
        future_states_dict = self._batch_obs_dict(future_obs_idxs)

        random_indices = self._sample_indices(batch_size)
        random_states_dict = self._batch_obs_dict(random_indices)

        new_actions = self._actions[indices]

        if len(self.observation_keys) == 1:
            obs_key = self.observation_keys[0]
            new_obs = new_obs_dict[obs_key]
            new_next_obs = new_obs_dict[obs_key]
            future_obs = future_states_dict[obs_key]
            random_obs = random_states_dict[obs_key]
        else:
            new_obs = tuple(new_obs_dict[k] for k in self.observation_keys)
            new_next_obs = tuple(new_next_obs_dict[k] for k in self.observation_keys)

        resampled_goals = new_next_obs_dict[self.desired_goal_key]
        rewards = self._rewards[indices]

        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': rewards,
            'terminals': self._terminals[indices],
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
            'random_obs': random_obs,
            'future_obs': future_obs,
        }
        return batch
