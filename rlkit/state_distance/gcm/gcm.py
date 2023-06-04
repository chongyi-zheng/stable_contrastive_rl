import abc

import numpy as np

from rlkit.data_management.path_builder import PathBuilder
from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.util.ml_util import ConstantSchedule
from rlkit.state_distance.exploration import MakeUniversal
from rlkit.state_distance.rollout_util import MultigoalSimplePathSampler, \
    multitask_rollout
from rlkit.state_distance.util import merge_into_flat_obs
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.core import np_to_pytorch_batch


class GoalConditionedModel(TorchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            max_tau=10,
            epoch_max_tau_schedule=None,
            sample_train_goals_from='replay_buffer',
            sample_rollout_goals_from='environment',
            cycle_taus_for_rollout=False,
    ):
        """
        :param max_tau: Maximum tau (planning horizon) to train with.
        :param epoch_max_tau_schedule: A schedule for the maximum planning
        horizon tau.
        :param sample_train_goals_from: Sampling strategy for goals used in
        training. Can be one of the following strings:
            - environment: Sample from the environment
            - replay_buffer: Sample from the replay_buffer
            - her: Sample from a HER-based replay_buffer
        :param sample_rollout_goals_from: Sampling strategy for goals used
        during rollout. Can be one of the following strings:
            - environment: Sample from the environment
            - replay_buffer: Sample from the replay_buffer
            - fixed: Do no resample the goal. Just use the one in the
            environment.
        :param vectorized: Train the QF in vectorized form?
        :param cycle_taus_for_rollout: Decrement the tau passed into the
        policy during rollout?
        """
        assert sample_train_goals_from in ['environment', 'replay_buffer',
                                           'her']
        assert sample_rollout_goals_from in ['environment', 'replay_buffer',
                                             'fixed']
        if epoch_max_tau_schedule is None:
            epoch_max_tau_schedule = ConstantSchedule(max_tau)

        self.max_tau = max_tau
        self.epoch_max_tau_schedule = epoch_max_tau_schedule
        self.sample_train_goals_from = sample_train_goals_from
        self.sample_rollout_goals_from = sample_rollout_goals_from
        self.cycle_taus_for_rollout = cycle_taus_for_rollout
        self._current_path_goal = None
        self._rollout_tau = self.max_tau

        self.policy = MakeUniversal(self.policy)
        self.eval_policy = MakeUniversal(self.eval_policy)
        self.exploration_policy = MakeUniversal(self.exploration_policy)
        self.eval_sampler = MultigoalSimplePathSampler(
            env=self.env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
            discount_sampling_function=self._sample_max_tau_for_rollout,
            goal_sampling_function=self._sample_goal_for_rollout,
            cycle_taus_for_rollout=self.cycle_taus_for_rollout,
        )
        if self.collection_mode == 'online-parallel':
            # TODO(murtaza): What happens to the eval env?
            # see `eval_sampler` definition above.

            self.training_env = RemoteRolloutEnv(
                env=self.env,
                policy=self.eval_policy,
                exploration_policy=self.exploration_policy,
                max_path_length=self.max_path_length,
                normalize_env=self.normalize_env,
                rollout_function=self.rollout,
            )

    def rollout(self, env, policy, max_path_length):
        goal = env.sample_goal_for_rollout()
        return multitask_rollout(
            env,
            agent=policy,
            goal=goal,
            discount=self.max_tau,
            max_path_length=max_path_length,
            decrement_discount=self.cycle_taus_for_rollout,
            cycle_tau=self.cycle_taus_for_rollout,
        )

    def _start_epoch(self, epoch):
        self.max_tau = self.epoch_max_tau_schedule.get_value(epoch)
        super()._start_epoch(epoch)

    def get_batch(self, training=True):
        if self.replay_buffer_is_split:
            replay_buffer = self.replay_buffer.get_replay_buffer(training)
        else:
            replay_buffer = self.replay_buffer
        batch = replay_buffer.random_batch(self.batch_size)

        """
        Update the goal states/rewards
        """
        num_steps_left = np.random.randint(
            0, self.max_tau + 1, (self.batch_size, 1)
        )
        terminals = 1 - (1 - batch['terminals']) * (num_steps_left != 0)
        batch['terminals'] = terminals

        obs = batch['observations']
        next_obs = batch['next_observations']
        if self.sample_train_goals_from == 'her':
            goals = batch['goals']
        else:
            goals = self._sample_goals_for_training()
        goal_differences = np.abs(
            self.env.convert_obs_to_goals(next_obs)
            # - self.env.convert_obs_to_goals(obs)
            - goals
        )
        batch['goal_differences'] = goal_differences * self.reward_scale
        batch['goals'] = goals

        """
        Update the observations
        """
        batch['observations'] = merge_into_flat_obs(
            obs=batch['observations'],
            goals=batch['goals'],
            num_steps_left=num_steps_left,
        )
        batch['next_observations'] = merge_into_flat_obs(
            obs=batch['next_observations'],
            goals=batch['goals'],
            num_steps_left=num_steps_left-1,
        )

        return np_to_pytorch_batch(batch)

    @property
    def train_buffer(self):
        if self.replay_buffer_is_split:
            return self.replay_buffer.get_replay_buffer(trainig=True)
        else:
            return self.replay_buffer

    def _sample_goals_for_training(self):
        if self.sample_train_goals_from == 'environment':
            return self.env.sample_goals(self.batch_size)
        elif self.sample_train_goals_from == 'replay_buffer':
            batch = self.train_buffer.random_batch(self.batch_size)
            obs = batch['observations']
            return self.env.convert_obs_to_goals(obs)
        elif self.sample_train_goals_from == 'her':
            raise Exception("Take samples from replay buffer.")
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_train_goals_from
            ))

    def _sample_goal_for_rollout(self):
        if self.sample_rollout_goals_from == 'environment':
            return self.env.sample_goal_for_rollout()
        elif self.sample_rollout_goals_from == 'replay_buffer':
            batch = self.train_buffer.random_batch(1)
            obs = batch['observations']
            goal_state = self.env.convert_obs_to_goals(obs)[0]
            return self.env.modify_goal_for_rollout(goal_state)
        elif self.sample_rollout_goals_from == 'fixed':
            return self.env.multitask_goal
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_rollout_goals_from
            ))

    def _sample_max_tau_for_rollout(self):
        return np.random.randint(0, self.max_tau + 1)

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self._current_path_goal = self._sample_goal_for_rollout()
        self.training_env.set_goal(self._current_path_goal)
        self.exploration_policy.set_goal(self._current_path_goal)
        self._rollout_tau = self.max_tau
        self.exploration_policy.set_tau(self._rollout_tau)
        return self.training_env.reset()

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            goals=self._current_path_goal,
        )
        if self.cycle_taus_for_rollout:
            self._rollout_tau -= 1
            if self._rollout_tau < 0:
                self._rollout_tau = self.max_tau
            self.exploration_policy.set_tau(self._rollout_tau)

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _handle_path(self, path):
        self._n_rollouts_total += 1
        self.replay_buffer.add_path(path)
