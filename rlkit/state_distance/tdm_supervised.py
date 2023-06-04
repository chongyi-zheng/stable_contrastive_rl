from collections import OrderedDict

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.state_distance.tdm import TemporalDifferenceModel
import torch.optim as optim
import torch.nn as nn

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class TdmSupervised(TemporalDifferenceModel, TorchRLAlgorithm):
    def __init__(
            self,
            env,
            exploration_policy,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            loss_fn=None,
            policy_learning_rate=1e-3,
            optimizer_class=optim.Adam,
            policy_criterion='MSE',
            replay_buffer=None,
    ):
        TorchRLAlgorithm.__init__(
            self,
            env,
            exploration_policy,
            **base_kwargs
        )
        super().__init__(**tdm_kwargs)
        self.policy = policy
        self.loss_fn=loss_fn
        self.policy_learning_rate = policy_learning_rate
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )
        if policy_criterion=='MSE':
            self.policy_criterion = nn.MSELoss()
        elif policy_criterion=='Huber':
            self.policy_criterion = HuberLoss()
        self.eval_policy = self.policy
        self.replay_buffer = replay_buffer

    @property
    def networks(self):
        return [
            self.policy,
        ]

    def get_batch(self):
        batch = self.replay_buffer.random_batch_random_tau(self.batch_size, self.max_tau)
        """
        Update the goal states/rewards
        """
        num_steps_left = self._sample_taus_for_training(batch)
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['training_goals']
        rewards = self._compute_rewards_np(
            batch, obs, actions, next_obs, goals
        )
        terminals = batch['terminals']

        #not too sure what this code does
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

    def _do_training(self):
        batch = self.get_batch()
        obs = batch['observations']
        actions = batch['actions']
        num_steps_left = batch['num_steps_left']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        # import ipdb; ipdb.set_trace()
        policy_actions = self.policy(
            obs, self.env.convert_obs_to_goals(next_obs), num_steps_left, return_preactivations=False,
        )
        policy_loss = self.policy_criterion(policy_actions, actions)
        """
        Update Networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.eval_statistics is None:
            """
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def evaluate(self, epoch):
        statistics = OrderedDict()
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        super().evaluate(epoch)

    def pretrain(self):
        pass

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            policy=self.eval_policy,
            trained_policy=self.policy,
            exploration_policy=self.exploration_policy,
        )
        return snapshot

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation,
            self._current_path_goal,
            self._rollout_tau,
        )