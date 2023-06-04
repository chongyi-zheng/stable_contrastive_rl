import numpy as np

from rlkit.policies.base import Policy
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.pearl.sampler import rollout
from rlkit.torch.pearl.agent import PEARLAgent


class PearlPathCollector(MdpPathCollector):
    def __init__(
            self,
            env: PEARLAgent,
            policy: Policy,
            task_indices,
            replay_buffer: PearlReplayBuffer,
            rollout_fn=rollout,
            sample_initial_context=False,
            accum_context_across_rollouts=False,
            **kwargs
    ):
        super().__init__(
            env, policy,
            rollout_fn=rollout_fn,
            **kwargs
        )
        self.replay_buffer = replay_buffer
        self.task_indices = task_indices
        self._rollout_kwargs = kwargs
        self._sample_initial_context = sample_initial_context
        self.accum_context_across_rollouts = accum_context_across_rollouts

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            task_idx=None,
    ):
        task_idx = task_idx or np.random.choice(self.task_indices)
        if self._sample_initial_context:
            # TODO: fix hack and consolidate where init context is sampled
            try:
                initial_context = self.replay_buffer.sample_context(task_idx)
                initial_context = ptu.from_numpy(initial_context)
            except ValueError:
                # this is needed for just the first loop where we need to fill the replay buffer without setting the replay buffer
                initial_context = None
        else:
            initial_context = None
        self._env.reset_task(task_idx)

        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
                )
            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                initial_context=initial_context,
            )
            if self.accum_context_across_rollouts:
                initial_context = path['context']
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            rollout_kwargs=self._rollout_kwargs,
            task_indices=self.task_indices,
        )
        return snapshot
