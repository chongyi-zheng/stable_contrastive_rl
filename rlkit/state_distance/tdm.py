"""
New implementation of state distance q learning.
"""
import abc

import numpy as np

from rlkit.data_management.path_builder import PathBuilder
from rlkit.util.ml_util import ConstantSchedule
from rlkit.util.np_util import truncated_geometric
from rlkit.samplers.rollout_functions import (
    create_rollout_function,
    tdm_rollout,
)
from rlkit.state_distance.rollout_util import MultigoalSimplePathSampler
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class TemporalDifferenceModel(TorchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            max_tau=10,
            epoch_max_tau_schedule=None,
            vectorized=True,
            cycle_taus_for_rollout=True,
            dense_rewards=False,
            finite_horizon=True,
            tau_sample_strategy='uniform',
            goal_reached_epsilon=1e-3,
            terminate_when_goal_reached=False,
            truncated_geom_factor=2.,
            square_distance=False,
            goal_weights=None,
            normalize_distance=False,
            observation_key=None,
            desired_goal_key=None,
    ):
        """

        :param max_tau: Maximum tau (planning horizon) to train with.
        :param epoch_max_tau_schedule: A schedule for the maximum planning
        horizon tau.
        :param vectorized: Train the QF in vectorized form?
        :param cycle_taus_for_rollout: Decrement the tau passed into the
        policy during rollout?
        :param dense_rewards: If True, always give rewards. Otherwise,
        only give rewards when the episode terminates.
        :param finite_horizon: If True, use a finite horizon formulation:
        give the time as input to the Q-function and terminate.
        :param tau_sample_strategy: Sampling strategy for taus used
        during training. Can be one of the following strings:
            - no_resampling: Do not resample the tau. Use the one from rollout.
            - uniform: Sample uniformly from [0, max_tau]
            - truncated_geometric: Sample from a truncated geometric
            distribution, truncated at max_tau.
            - all_valid: Always use all 0 to max_tau values
        :param goal_reached_epsilon: Epsilon used to determine if the goal
        has been reached. Used by `indicator` version of `reward_type` and when
        `terminate_whe_goal_reached` is True.
        :param terminate_when_goal_reached: Do you terminate when you have
        reached the goal?
        :param goal_weights: None or the weights for the different goal
        dimensions. These weights are used to compute the distances to the goal.
        """
        assert tau_sample_strategy in [
            'no_resampling',
            'uniform',
            'truncated_geometric',
            'all_valid',
        ]
        if epoch_max_tau_schedule is None:
            epoch_max_tau_schedule = ConstantSchedule(max_tau)

        if not finite_horizon:
            max_tau = 0
            epoch_max_tau_schedule = ConstantSchedule(max_tau)
            cycle_taus_for_rollout = False

        self.max_tau = max_tau
        self.epoch_max_tau_schedule = epoch_max_tau_schedule
        self.vectorized = vectorized
        self.cycle_taus_for_rollout = cycle_taus_for_rollout
        self.dense_rewards = dense_rewards
        self.finite_horizon = finite_horizon
        self.tau_sample_strategy = tau_sample_strategy
        self.goal_reached_epsilon = goal_reached_epsilon
        self.terminate_when_goal_reached = terminate_when_goal_reached
        self.square_distance = square_distance
        self._rollout_tau = np.array([self.max_tau])
        self.truncated_geom_factor = float(truncated_geom_factor)
        self.goal_weights = goal_weights
        if self.goal_weights is not None:
            # In case they were passed in as (e.g.) tuples or list
            self.goal_weights = np.array(self.goal_weights)
            assert self.goal_weights.size == self.env.goal_dim
        self.normalize_distance = normalize_distance

        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key

        self.eval_sampler = MultigoalSimplePathSampler(
            env=self.env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
            tau_sampling_function=self._sample_max_tau_for_rollout,
            cycle_taus_for_rollout=self.cycle_taus_for_rollout,
            render=self.render_during_eval,
            observation_key=self.observation_key,
            desired_goal_key=self.desired_goal_key,
        )
        self.pretrain_obs = None

        # Serializing this eval_rollout_function creates an infinite loop for
        # some reason. Calling cloudpickle.dumps(self.eval_rollout_function) will
        # literally fill your entire RAM/swap.
        #
        # self.eval_rollout_function = create_rollout_function(
        #     tau_sampling_tdm_rollout,
        #     tau_sampler=self._sample_max_tau_for_rollout(),
        #     cycle_tau=self.cycle_taus_for_rollout,
        #     decrement_tau=self.cycle_taus_for_rollout,
        #     observation_key=self.observation_key,
        #     desired_goal_key=self.desired_goal_key,
        # )

    @property
    def train_rollout_function(self):
        return create_rollout_function(
            tdm_rollout,
            init_tau=self.max_tau,
            cycle_tau=self.cycle_taus_for_rollout,
            decrement_tau=self.cycle_taus_for_rollout,
            observation_key=self.observation_key,
            desired_goal_key=self.desired_goal_key,
        )

    @property
    def eval_rollout_function(self):
        return create_rollout_function(
            tdm_rollout,
            init_tau=self.max_tau,
            cycle_tau=self.cycle_taus_for_rollout,
            decrement_tau=self.cycle_taus_for_rollout,
            observation_key=self.observation_key,
            desired_goal_key=self.desired_goal_key,
        )

    def _start_epoch(self, epoch):
        self.max_tau = self.epoch_max_tau_schedule.get_value(epoch)
        super()._start_epoch(epoch)

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)
        """
        Update the goal states/rewards
        """
        num_steps_left = self._sample_taus_for_training(batch)
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']
        rewards = batch['rewards']
        terminals = batch['terminals']


        if self.tau_sample_strategy == 'all_valid':
            obs = np.repeat(obs, self.max_tau + 1, 0)
            actions = np.repeat(actions, self.max_tau + 1, 0)
            next_obs = np.repeat(next_obs, self.max_tau + 1, 0)
            goals = np.repeat(goals, self.max_tau + 1, 0)
            rewards = np.repeat(rewards, self.max_tau + 1, 0)
            terminals = np.repeat(terminals, self.max_tau + 1, 0)

        if self.finite_horizon:
            terminals = 1 - (1 - terminals) * (num_steps_left != 0)
        if self.terminate_when_goal_reached:
            diff = self.env.convert_obs_to_goals(next_obs) - goals
            goal_not_reached = (
                    np.linalg.norm(diff, axis=1, keepdims=True)
                    > self.goal_reached_epsilon
            )
            terminals = 1 - (1 - terminals) * goal_not_reached

        if not self.dense_rewards:
            rewards = rewards * terminals

        """
        Update the batch
        """
        batch['rewards'] = rewards
        batch['terminals'] = terminals
        batch['actions'] = actions
        batch['num_steps_left'] = num_steps_left
        batch['goals'] = goals
        batch['observations'] = obs
        batch['next_observations'] = next_obs

        return np_to_pytorch_batch(batch)

    def _sample_taus_for_training(self, batch):
        if self.finite_horizon:
            if self.tau_sample_strategy == 'uniform':
                num_steps_left = np.random.randint(
                    0, self.max_tau + 1, (self.batch_size, 1)
                )
            elif self.tau_sample_strategy == 'truncated_geometric':
                num_steps_left = truncated_geometric(
                    p=self.truncated_geom_factor / self.max_tau,
                    truncate_threshold=self.max_tau,
                    size=(self.batch_size, 1),
                    new_value=0
                )
            elif self.tau_sample_strategy == 'no_resampling':
                num_steps_left = batch['num_steps_left']
            elif self.tau_sample_strategy == 'all_valid':
                num_steps_left = np.tile(
                    np.arange(0, self.max_tau + 1),
                    self.batch_size
                )
                num_steps_left = np.expand_dims(num_steps_left, 1)
            else:
                raise TypeError("Invalid tau_sample_strategy: {}".format(
                    self.tau_sample_strategy
                ))
        else:
            num_steps_left = np.zeros((self.batch_size, 1))
        return num_steps_left

    def _sample_max_tau_for_rollout(self):
        if self.finite_horizon:
            return self.max_tau
        else:
            return 0

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self._rollout_tau = np.array([self.max_tau])
        obs = self.training_env.reset()
        return obs

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
        if self.vectorized:
            # since rl_algorithm wraps scalar rews with an extra dim, we want to undo this for vectorized rews
            reward = reward[0]
            assert reward.shape == (1,)
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            num_steps_left=self._rollout_tau,
        )
        if self.cycle_taus_for_rollout:
            self._rollout_tau -= 1
            if self._rollout_tau[0] < 0:
                self._rollout_tau = np.array([self.max_tau])

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation[self.observation_key],
            observation[self.desired_goal_key],
            self._rollout_tau,
        )

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
        self._exploration_paths.append(path)

    def evaluate(self, epoch, eval_paths=None):
        self.eval_statistics['Max Tau'] = self.max_tau
        super().evaluate(epoch, eval_paths=eval_paths)
