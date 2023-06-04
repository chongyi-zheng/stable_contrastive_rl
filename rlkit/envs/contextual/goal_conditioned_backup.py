import warnings
from typing import Any, Callable, Dict, List
import random
import numpy as np
import gym
from gym.spaces import Box, Dict
from multiworld.core.multitask_env import MultitaskEnv
from rlkit.util.io import load_local_or_remote_file
from gym.spaces import Box, Dict
from rlkit import pythonplusplus as ppp
from rlkit.core.distribution import DictDistribution
from rlkit.envs.contextual import ContextualRewardFn
from rlkit.envs.contextual.contextual_env import (
    ContextualDiagnosticsFn,
    Path,
    Context,
    Diagnostics,
)
from rlkit.envs.images import EnvRenderer

Observation = Dict
Goal = Any
GoalConditionedDiagnosticsFn = Callable[
    [List[Path], List[Goal]],
    Diagnostics,
]


class GoalDictDistributionFromMultitaskEnv(DictDistribution):
    def __init__(
            self,
            env: MultitaskEnv,
            desired_goal_keys=('desired_goal',),
    ):
        self._env = env
        self._desired_goal_keys = desired_goal_keys
        env_spaces = self._env.observation_space.spaces
        self._spaces = {
            k: env_spaces[k]
            for k in self._desired_goal_keys
        }

    def sample(self, batch_size: int):
        return {
            k: self._env.sample_goals(batch_size)[k]
            for k in self._desired_goal_keys
        }

    @property
    def spaces(self):
        return self._spaces


class AddImageDistribution(DictDistribution):
    def __init__(
            self,
            env: MultitaskEnv,
            base_distribution: DictDistribution,
            renderer: EnvRenderer,
            image_goal_key='image_desired_goal',
            _suppress_warning=False,
    ):
        self._env = env
        self._base_distribution = base_distribution
        img_space = Box(0, 1, renderer.image_shape, dtype=np.float32)
        self._spaces = base_distribution.spaces
        self._spaces[image_goal_key] = img_space
        self._image_goal_key = image_goal_key
        self._renderer = renderer
        self._suppress_warning = _suppress_warning

    def sample(self, batch_size: int):
        if batch_size > 1 and not self._suppress_warning:
            warnings.warn(
                "Sampling many goals is slow. Consider using "
                "PresampledImageAndStateDistribution"
            )
        contexts = self._base_distribution.sample(batch_size)
        images = []
        for i in range(batch_size):
            goal = ppp.treemap(lambda x: x[i], contexts, atomic_type=np.ndarray)
            env_state = self._env.get_env_state()
            self._env.set_to_goal(goal)
            img_goal = self._renderer(self._env)
            self._env.set_env_state(env_state)
            images.append(img_goal)

        contexts[self._image_goal_key] = np.array(images)
        return contexts

    @property
    def spaces(self):
        return self._spaces


class AddManualImageDistribution(DictDistribution):
    def __init__(
            self,
            env: MultitaskEnv,
            base_distribution: DictDistribution,
            renderer: EnvRenderer,
            image_goal_key='image_desired_goal',
            _suppress_warning=False,
    ):
        self._env = env
        self._base_distribution = base_distribution
        img_space = Box(0, 1, renderer.image_shape, dtype=np.float32)
        self._spaces = base_distribution.spaces
        self._spaces[image_goal_key] = img_space
        self._image_goal_key = image_goal_key
        self._renderer = renderer
        self._suppress_warning = _suppress_warning

    def sample(self, batch_size: int):
        if batch_size > 1 and not self._suppress_warning:
            warnings.warn(
                "Sampling many goals is slow. Consider using "
                "PresampledImageAndStateDistribution"
            )
        contexts = self._base_distribution.sample(batch_size)
        images = []
        for i in range(batch_size):
            goal = ppp.treemap(lambda x: x[i], contexts, atomic_type=np.ndarray)
            env_state = self._env.get_env_state()
            inp = input("set the goal image now")
            # self._env.set_to_goal(goal)
            img_goal = self._renderer(self._env)
            inp = input("reset the robot now")
            # self._env.set_env_state(env_state)
            images.append(img_goal)

        contexts[self._image_goal_key] = np.array(images)
        return contexts

    @property
    def spaces(self):
        return self._spaces


class PresampledDistribution(DictDistribution):
    def __init__(
            self,
            slow_sampler: DictDistribution,
            num_presampled_goals,
    ):
        self._sampler = slow_sampler
        self._num_presampled_goals = num_presampled_goals
        self._presampled_goals = self._sampler.sample(num_presampled_goals)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._num_presampled_goals, batch_size)
        sampled_goals = {
            k: v[idx] for k, v in self._presampled_goals.items()
        }
        return sampled_goals

    @property
    def spaces(self):
        return self._sampler.spaces

class PresampledPathDistribution(DictDistribution):
    def __init__(
            self,
            datapath,
            representation_size,
    ):  
        self._presampled_goals = load_local_or_remote_file(datapath)
        try: self._presampled_goals = self._presampled_goals.item()
        except: pass
        # self._num_presampled_goals = self._presampled_goals[list(self._presampled_goals)[0]].shape[0]
        self._num_presampled_goals = len(self._presampled_goals['image_desired_goal'])
        self._set_spaces()

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._num_presampled_goals, batch_size)
        sampled_goals = {
            k: v[idx] for k, v in self._presampled_goals.items()
        }
        return sampled_goals

    def _set_spaces(self):
        pairs = []
        for key in self._presampled_goals:
            dim = self._presampled_goals[key][0].shape[0]
            box = gym.spaces.Box(-np.ones(dim), np.ones(dim))
            pairs.append((key, box))
        self.observation_space = Dict(pairs)

    @property
    def spaces(self):
        return self.observation_space.spaces


class ContextualRewardFnFromMultitaskEnv(ContextualRewardFn):
    def __init__(
            self,
            env: MultitaskEnv,
            achieved_goal_from_observation: Callable[[Observation], Goal],
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            additional_obs_keys=None,
            additional_context_keys=None,
    ):
        self._env = env
        self._desired_goal_key = desired_goal_key
        self._achieved_goal_key = achieved_goal_key
        self._achieved_goal_from_observation = achieved_goal_from_observation

        self._additional_obs_keys = additional_obs_keys
        self._additional_context_keys = additional_context_keys

    def __call__(self, states, actions, next_states, contexts):
        del states

        achieved = self._achieved_goal_from_observation(next_states)

        obs = {
            self._achieved_goal_key: achieved,
            self._desired_goal_key: contexts[self._desired_goal_key],
        }

        if self._additional_obs_keys is not None:
            for key in self._additional_obs_keys:
                obs[key] = next_states[key]
        if self._additional_context_keys is not None:
            for key in self._additional_context_keys:
                obs[key] = contexts[key]

        return self._env.compute_rewards(actions, obs)


class IndexIntoAchievedGoal(object):
    def __init__(self, key):
        self._key = key

    def __call__(self, observations):
        return observations[self._key]


class L2Distance(ContextualRewardFn):
    def __init__(
            self,
            achieved_goal_from_observation: Callable[[Observation], Goal],
            desired_goal_key='desired_goal',
            dimension_weights=None,
    ):
        self._desired_goal_key = desired_goal_key
        self._achieved_goal_from_observation = achieved_goal_from_observation
        self._dimension_weights = dimension_weights

    def __call__(self, states, actions, next_states, contexts):
        del states
        achieved = self._achieved_goal_from_observation(next_states)
        desired = contexts[self._desired_goal_key]
        if self._dimension_weights is None:
            distance = np.linalg.norm(achieved - desired, axis=-1)
        else:
            difference = achieved - desired
            weighted_difference = (
                    self._dimension_weights[None, :]
                    * difference
            )
            distance = np.linalg.norm(weighted_difference, axis=-1)
        return distance


class NegativeL2Distance(ContextualRewardFn):
    def __init__(
            self,
            achieved_goal_from_observation: Callable[[Observation], Goal],
            desired_goal_key='desired_goal',
            dimension_weights=None,
    ):
        self.distance_fn = L2Distance(
            achieved_goal_from_observation,
            desired_goal_key=desired_goal_key,
            dimension_weights=dimension_weights,
        )

    def __call__(self, states, actions, next_states, contexts):
        distance = self.distance_fn(states, actions, next_states, contexts)
        return - distance


class ThresholdDistanceReward(ContextualRewardFn):
    def __init__(self, distance_fn: ContextualRewardFn, threshold):
        self._distance_fn = distance_fn
        self._distance_threshold = threshold

    def __call__(self, states, actions, next_states, contexts):
        distance = self._distance_fn(states, actions, next_states, contexts)
        return -(distance > self._distance_threshold).astype(np.float32)

class GoalConditionedDiagnosticsToContextualDiagnostics():
    # use a class rather than function for serialization
    def __init__(
            self,
            goal_conditioned_diagnostics,
            desired_goal_key: str,
            observation_key: str,
    ):
        self._goal_conditioned_diagnostics = goal_conditioned_diagnostics
        self._desired_goal_key = desired_goal_key
        self._observation_key = observation_key

#    def __call__(self, paths: List[Path],
#                 contexts: List[Context]) -> Diagnostics:
    def __call__(self, paths, contexts):
        goals = [c[self._desired_goal_key] for c in contexts]
        non_contextual_paths = [self._remove_context(p) for p in paths]
        return self._goal_conditioned_diagnostics(non_contextual_paths, goals)

    def _remove_context(self, path):
        new_path = path.copy()
        new_path['observations'] = np.array([
            o[self._observation_key] for o in path['observations']
        ])
        new_path['next_observations'] = np.array([
            o[self._observation_key] for o in path['next_observations']
        ])
        new_path.pop('full_observations', None)
        new_path.pop('full_next_observations', None)
        return new_path
