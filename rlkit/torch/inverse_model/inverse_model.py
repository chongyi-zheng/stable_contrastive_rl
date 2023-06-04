from collections import OrderedDict

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.data_management.path_builder import PathBuilder
from rlkit.core.eval_util import create_stats_ordered_dict
import torch.optim as optim
import torch.nn as nn
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.inverse_model.rollout_util import InverseModelSimplePathSampler
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class Inverse_Model(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            exploration_policy,
            base_kwargs,
            policy=None,
            loss_fn=None,
            policy_learning_rate=1e-3,
            optimizer_class=optim.Adam,
            policy_criterion='MSE',
            replay_buffer=None,
            time_horizon=None,
    ):

        TorchRLAlgorithm.__init__(
            self,
            env,
            exploration_policy,
            **base_kwargs
        )
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

        if time_horizon==None:
            self.time_horizon = self.max_path_length
        else:
            self.time_horizon = time_horizon
        self.eval_policy = self.policy
        self._current_path_goal = None
        self.replay_buffer = replay_buffer
        self._rollout_tau = np.array([self.max_path_length])
        #TEMPORARY SAMPLER:
        self.eval_sampler = InverseModelSimplePathSampler(
            env=self.env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
        )

    @property
    def networks(self):
        return [
            self.policy,
        ]

    def get_batch(self):
        batch = self.replay_buffer.random_batch_random_tau(self.batch_size, max_tau=self.time_horizon)
        return np_to_pytorch_batch(batch)

    def _do_training(self):
        batch = self.get_batch()
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        inputs = torch.cat((obs, self.env.convert_obs_to_goals(next_obs)), dim=1)
        policy_actions = self.policy(inputs)
        policy_loss = self.policy_criterion(policy_actions, actions)
        """
        Update Networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
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

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self._current_path_goal = self.training_env.sample_goal_for_rollout()
        self.training_env.set_goal(self._current_path_goal)
        self._rollout_tau = np.array([self.max_path_length])
        return self.training_env.reset()

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        inputs = np.hstack((observation, self._current_path_goal))
        return self.exploration_policy.get_action(inputs)

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
            num_steps_left=self._rollout_tau,
            goals=self._current_path_goal,
        )
        self._rollout_tau -= 1
        if self._rollout_tau[0] < 0:
            self._rollout_tau = np.array([self.max_path_length])

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            # self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _handle_path(self, path):
        self._n_rollouts_total += 1
        self.replay_buffer.add_path(path)

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