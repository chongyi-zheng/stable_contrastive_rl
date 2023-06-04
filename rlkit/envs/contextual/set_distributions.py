from collections import defaultdict
from typing import List

import numpy as np
from gym.spaces import Box, Space, Discrete, MultiBinary

from rlkit.core.distribution import DictDistribution
from rlkit.envs.contextual.contextual_env import (
    ContextualDiagnosticsFn,
    Path,
    Context,
    Diagnostics,
    ContextualRewardFn,
)
from rlkit.util import np_util
from rlkit.core import eval_util
from rlkit.torch.sets.set_projection import Set
from rlkit.torch.sets import set_vae_trainer
from rlkit.torch.vae.vae import VAE
import rlkit.torch.pytorch_util as ptu


class ObjectSpace(Space):  # placeholder space
    pass


class GoalDictDistributionFromSet(DictDistribution):
    def __init__(
            self,
            set,
            desired_goal_keys=('desired_goal',),
    ):
        self.set = set
        self._desired_goal_keys = desired_goal_keys

        set_space = Box(
            -10 * np.ones(set.shape[1:]),
            10 * np.ones(set.shape[1:]),
            dtype=np.float32,
        )
        self._spaces = {
            k: set_space
            for k in self._desired_goal_keys
        }

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.set), batch_size)
        sampled_data = self.set[indices]
        return {
            k: sampled_data
            for k in self._desired_goal_keys
        }

    @property
    def spaces(self):
        return self._spaces


class LatentGoalDictDistributionFromSet(DictDistribution):
    def __init__(
            self,
            sets: List[Set],
            vae: VAE,
            data_key: str,
            cycle_for_batch_size_1=False,
    ):
        self.sets = sets
        self.vae = vae
        self.data_key = data_key
        self.mean_key = 'latent_mean'
        self.covariance_key = 'latent_covariance'
        self.description_key = 'set_description'
        self.set_index_key = 'set_index'
        self.set_embedding_key = 'set_embedding'
        self._num_sets = len(sets)
        self.cycle_for_batch_size_1 = cycle_for_batch_size_1

        set_space = Box(
            -10 * np.ones(vae.representation_size),
            10 * np.ones(vae.representation_size),
            dtype=np.float32,
        )
        self._spaces = {
            self.mean_key: set_space,
            self.covariance_key: set_space,
            self.description_key: ObjectSpace(),
            self.set_index_key: Discrete(len(sets)),
            self.set_embedding_key: MultiBinary(len(sets)),
        }
        self.means = None
        self.covariances = None
        self.descriptions = [set.description for set in sets]
        self.update_encodings()
        self._current_idx = 0

    def sample(self, batch_size: int):
        if batch_size == 1 and self.cycle_for_batch_size_1:
            indices = np.array([self._current_idx])
            self._current_idx = (self._current_idx + 1) % len(self.sets)
        else:
            indices = np.random.choice(len(self.sets), batch_size)
        sampled_means = self.means[indices]
        sampled_covariances = self.covariances[indices]
        sampled_descriptions = [self.descriptions[i] for i in indices]
        return {
            self.mean_key: sampled_means,
            self.covariance_key: sampled_covariances,
            self.description_key: sampled_descriptions,
            self.set_index_key: indices,
            self.set_embedding_key: np_util.onehot(indices, self._num_sets)
        }

    def update_encodings(self):
        means = []
        covariances = []
        for set in self.sets:
            sampled_data = set.example_dict[self.data_key]
            posteriors = self.vae.encoder(ptu.from_numpy(sampled_data))
            learned_prior = set_vae_trainer.compute_prior(posteriors)
            means.append(ptu.get_numpy(learned_prior.mean)[0])
            covariances.append(ptu.get_numpy(learned_prior.variance)[0])
        self.means = np.array(means)
        self.covariances = np.array(covariances)

    @property
    def spaces(self):
        return self._spaces


class OracleRIGMeanSetter(LatentGoalDictDistributionFromSet):
    def __init__(self, *args, env, renderer, use_random_goal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.renderer = renderer
        self.use_random_goal = use_random_goal

    def sample(self, batch_size: int, init_obs=None):
        if batch_size != 1:
            raise NotImplementedError()

        if batch_size == 1 and self.cycle_for_batch_size_1:
            indices = np.array([self._current_idx])
            self._current_idx = (self._current_idx + 1) % len(self.sets)
        else:
            indices = np.random.choice(len(self.sets), batch_size)
        if self.use_random_goal:
            set = self.sets[indices[0]]
            all_goal_imgs = set.example_dict['example_image']
            example_i = np.random.randint(0, len(all_goal_imgs))
            goal_image = all_goal_imgs[example_i]
        else:
            orig_state = init_obs['state_observation']
            set = self.sets[indices[0]]
            goal_state = set.description(orig_state.copy())
            self.env._set_positions(goal_state)
            goal_image = self.renderer(self.env)
            self.env._set_positions(orig_state)

        posterior = self.vae.encoder(ptu.from_numpy(goal_image[None]))
        sampled_means = ptu.get_numpy(posterior.mean)
        sampled_covariances = ptu.get_numpy(posterior.variance)
        sampled_descriptions = [self.descriptions[i] for i in indices]
        return {
            self.mean_key: sampled_means,
            self.covariance_key: sampled_covariances,
            self.description_key: sampled_descriptions,
            self.set_index_key: indices,
        }

    def update_encodings(self):
        means = []
        covariances = []
        for set in self.sets:
            sampled_data = set.example_dict[self.data_key]
            posteriors = self.vae.encoder(ptu.from_numpy(sampled_data))
            learned_prior = set_vae_trainer.compute_prior(posteriors)
            means.append(ptu.get_numpy(learned_prior.mean)[0])
            covariances.append(ptu.get_numpy(learned_prior.variance)[0])
        self.means = np.array(means)
        self.covariances = np.array(covariances)

    @property
    def spaces(self):
        return self._spaces


class SetDiagnostics(ContextualDiagnosticsFn):
    # use a class rather than function for serialization
    def __init__(
            self,
            set_description_key: str,
            set_index_key: str,
            observation_key: str,
    ):
        self._set_description_key = set_description_key
        self._set_index_key = set_index_key
        self._observation_key = observation_key

    def __call__(self, paths: List[Path],
                 contexts: List[Context]) -> Diagnostics:
        # set_descriptions = [c[self._set_description_key] for c in contexts]
        # set_indices = [c[self._set_index_key] for c in contexts]

        stat_to_lists = defaultdict(list)
        # for path, set_idx, set_description in zip(
        #         paths, set_indices, set_descriptions
        # ):
        for path, context in zip(paths, contexts):
            set_description = context[self._set_description_key]
            set_idx = context[self._set_index_key]
            distances_to_set = []
            for obs_dict in path['observations']:
                state = obs_dict[self._observation_key]
                distances_to_set.append(
                    set_description.distance_to_set(state)
                )

            stat_name = 'set{}/{}'.format(
                set_idx,
                set_description.describe()
            )
            stat_to_lists[stat_name].append(distances_to_set)

        return eval_util.diagnostics_from_paths_statistics(stat_to_lists)


class SetReward(ContextualRewardFn):
    # use a class rather than function for serialization
    def __init__(
            self,
            sets: List[Set],
            set_index_key: str,
            observation_key: str,
            batched=True,
    ):
        if not isinstance(sets, List):
            raise TypeError("The order of the sets is important!")
        self._sets = sets
        self._set_index_key = set_index_key
        self._observation_key = observation_key
        self._batched = batched

    def __call__(self, states, actions, next_states, contexts):
        set_indices = contexts[self._set_index_key]
        x = next_states[self._observation_key]
        distances_to_sets = np.array([
            set.description.distance_to_set(x) for set in self._sets
        ])
        distances = distances_to_sets[set_indices, np.arange(set_indices.size)]

        reward = - distances
        if self._batched:
            reward = reward.reshape(-1, 1)
        return reward
