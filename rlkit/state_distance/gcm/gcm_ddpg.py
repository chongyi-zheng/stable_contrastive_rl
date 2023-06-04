from collections import OrderedDict

import numpy as np
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.state_distance.gcm import GoalConditionedModel
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class GcmDdpg(GoalConditionedModel):
    def __init__(
            self,
            env,
            gcm,
            policy,
            exploration_policy,
            gcm_kwargs,
            base_kwargs,

            gcm_criterion=None,
            policy_learning_rate=1e-4,
            gcm_learning_rate=1e-3,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
    ):
        TorchRLAlgorithm.__init__(
            self,
            env,
            exploration_policy=exploration_policy,
            eval_policy=policy,
            **base_kwargs
        )
        if gcm_criterion is None:
            gcm_criterion = nn.MSELoss()
        self.gcm = gcm
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.gcm_learning_rate = gcm_learning_rate
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.gcm_criterion = gcm_criterion

        self.target_gcm = self.gcm.copy()
        self.target_policy = self.policy.copy()
        self.gcm_optimizer = optim.Adam(
            self.gcm.parameters(),
            lr=self.gcm_learning_rate,
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)
        super().__init__(**gcm_kwargs)

    def _do_training(self):
        batch = self.get_batch(training=True)
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goal_differences = batch['goal_differences']
        goals = batch['goals']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs)
        # future_goals_predicted = (
            # self.env.convert_obs_to_goals(obs) + self.gcm(obs, policy_actions)
        # )
        # policy_loss = ((future_goals_predicted-goals)**2).sum(dim=1).mean()
        policy_loss = self.gcm(obs, policy_actions).sum(dim=1).mean()

        """
        GCM operations.
        """
        next_actions = self.target_policy(next_obs)
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        target_difference = self.target_gcm(
            next_obs,
            next_actions,
        )
        gcm_target = goal_differences + (1. - terminals) * target_difference
        gcm_target = gcm_target.detach()
        gcm_pred = self.gcm(obs, actions)
        bellman_errors = (gcm_pred - gcm_target) ** 2
        gcm_loss = self.gcm_criterion(gcm_pred, gcm_target)

        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.gcm_optimizer.zero_grad()
        gcm_loss.backward()
        self.gcm_optimizer.step()

        self._update_target_networks()

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['GCM Loss'] = np.mean(ptu.get_numpy(gcm_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors',
                ptu.get_numpy(bellman_errors),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'GCM Predictions',
                ptu.get_numpy(gcm_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'GCM Targets',
                ptu.get_numpy(gcm_target),
            ))

    def _update_target_networks(self):
        if self.use_soft_update:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.gcm, self.target_gcm, self.tau)
        else:
            if self._n_env_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.gcm, self.target_gcm)
                ptu.copy_model_params_from_to(self.policy, self.target_policy)

    @property
    def networks(self):
        return [
            self.gcm,
            self.policy,
            self.target_gcm,
            self.target_policy,
        ]
