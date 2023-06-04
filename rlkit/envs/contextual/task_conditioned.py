from functools import partial

import numpy as np

from rlkit.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
)
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)

from gym.spaces import Box
from rlkit.samplers.rollout_functions import contextual_rollout

class TaskGoalDictDistributionFromMultitaskEnv(
        GoalDictDistributionFromMultitaskEnv):
    def __init__(
            self,
            *args,
            task_key='task_id',
            task_ids=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.task_key = task_key
        self._spaces[task_key] = Box(
            low=np.zeros(1),
            high=np.ones(1))
        self.task_ids = np.array(task_ids)

    def sample(self, batch_size: int):
        goals = super().sample(batch_size)
        idxs = np.random.choice(len(self.task_ids), batch_size)
        goals[self.task_key] = self.task_ids[idxs].reshape(-1, 1)
        return goals

class TaskPathCollector(ContextualPathCollector):
    def __init__(
            self,
            *args,
            task_key=None,
            max_path_length=100,
            task_ids=None,
            rotate_freq=0.0,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rotate_freq = rotate_freq
        self.rollout_tasks = []

        def obs_processor(o):
            if len(self.rollout_tasks) > 0:
                task = self.rollout_tasks[0]
                self.rollout_tasks = self.rollout_tasks[1:]
                o[task_key] = task
                self._env._rollout_context_batch[task_key] = task[None]

            combined_obs = [o[self._observation_key]]
            for k in self._context_keys_for_policy:
                combined_obs.append(o[k])
            return np.concatenate(combined_obs, axis=0)

        def reset_postprocess_func():
            rotate = (np.random.uniform() < self.rotate_freq)
            self.rollout_tasks = []
            if rotate:
                num_steps_per_task = max_path_length // len(task_ids)
                self.rollout_tasks = np.ones((max_path_length, 1)) * (len(task_ids) - 1)
                for (idx, id) in enumerate(task_ids):
                    start = idx * num_steps_per_task
                    end = start + num_steps_per_task
                    self.rollout_tasks[start:end] = id

        self._rollout_fn = partial(
            contextual_rollout,
            context_keys_for_policy=self._context_keys_for_policy,
            observation_key=self._observation_key,
            obs_processor=obs_processor,
            reset_postprocess_func=reset_postprocess_func,
        )