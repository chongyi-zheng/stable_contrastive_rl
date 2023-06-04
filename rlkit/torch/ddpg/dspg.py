import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class DeepStochasticPolicyGradient(TorchRLAlgorithm):
    """
    Same thing as DDPG but instead of using

      $$\nabla_\theta \pi_\theta(s) = \nabla_a Q(s, a)|_{a = \pi_\theta(s)} \nabla_\theta \pi_\theta(s)$$

    for the policy gradient, use the likelihood ration policy gradient

      $$\nabla_\theta \log p(a | s) = \sum_i 2 \nabla_\theta \mu_\theta(s)_i (a_i - \mu_\theta(s)_i)(Q(s, a) - V(s))$$

    where we effectively assume the policy output is the mean of a Gaussian with
    std `sample_std`.
    """
    def __init__(
            self,
            env,
            qf,
            vf,
            policy,
            exploration_policy,

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0,
            qf_criterion=None,
            vf_learning_rate=1e-3,
            vf_criterion=None,
            optimizer_class=optim.Adam,

            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            sample_std=0.1,

            **kwargs
    ):
        super().__init__(
            env,
            exploration_policy,
            eval_policy=policy,
            **kwargs
        )
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        if vf_criterion is None:
            vf_criterion = nn.MSELoss()
        self.qf = qf
        self.vf = vf
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.qf_criterion = qf_criterion
        self.vf_learning_rate = vf_learning_rate
        self.vf_criterion = vf_criterion

        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.sample_std = sample_std

        self.target_vf = self.vf.copy()
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=self.qf_learning_rate,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=self.vf_learning_rate,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )
        self.eval_statistics = None

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs)
        sampled_actions = ptu.Variable(
            torch.randn(policy_actions.shape)
        ) * self.sample_std + policy_actions
        sampled_actions = sampled_actions.detach()
        deviations = (policy_actions - sampled_actions)**2
        avg_deviations = deviations.mean(dim=1, keepdim=True)
        policy_loss = (
                avg_deviations
                * (self.qf(obs, sampled_actions) - self.target_vf(obs))
        ).mean()

        """
        Qf operations.
        """
        target_q_values = self.target_vf(next_obs)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
        q_pred = self.qf(obs, actions)
        bellman_errors = (q_pred - q_target) ** 2
        qf_loss = self.qf_criterion(q_pred, q_target)

        """
        Vf operations.
        """
        v_target = self.qf(obs, self.policy(obs)).detach()
        v_pred = self.vf(obs)
        vf_loss = self.vf_criterion(v_pred, v_target)

        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self._update_target_networks()

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Targets',
                ptu.get_numpy(v_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors',
                ptu.get_numpy(bellman_errors),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def _update_target_networks(self):
        if self.use_soft_update:
            ptu.soft_update_from_to(self.vf, self.target_vf, self.tau)
        else:
            if self._n_env_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.vf, self.target_vf)

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot['policy'] = self.policy
        snapshot['qf'] = self.qf
        snapshot['vf'] = self.vf
        snapshot['target_vf'] = self.target_vf
        snapshot['exploration_policy'] = self.exploration_policy
        snapshot['batch_size'] = self.batch_size
        return snapshot

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.vf,
            self.target_vf,
        ]
