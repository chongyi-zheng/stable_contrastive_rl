from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.core.logging import add_prefix
from rlkit.core.loss import LossStatistics
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.sac.sac import SACTrainer, SACLosses


def convex_sum(a, b, weight_a):
    return a * weight_a + b * (1-weight_a)


class DisentangedTrainer(SACTrainer):
    def __init__(self, *args, single_loss_weight, single_uses_mean_not_sum=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_loss_weight = single_loss_weight
        self.single_uses_mean_not_sum = single_uses_mean_not_sum

    def compute_loss(
            self, batch, skip_statistics=False,
    ) -> Tuple[SACLosses, LossStatistics]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = (
                self.reward_scale * rewards
                + (1. - terminals) * self.discount * target_q_values
        ).detach()
        vec_qf1_loss = self.qf_criterion(q1_pred, q_target)
        vec_qf2_loss = self.qf_criterion(q2_pred, q_target)

        if self.single_uses_mean_not_sum:
            single_q_target = q_target.mean(dim=1)
            single_qf1_loss = self.qf_criterion(
                q1_pred.mean(dim=1),
                single_q_target,
            )
            single_qf2_loss = self.qf_criterion(
                q2_pred.mean(dim=1),
                single_q_target,
            )
        else:
            single_q_target = q_target.sum(dim=1)
            single_qf1_loss = self.qf_criterion(
                q1_pred.sum(dim=1),
                single_q_target,
            )
            single_qf2_loss = self.qf_criterion(
                q2_pred.sum(dim=1),
                single_q_target,
            )

        qf1_loss = convex_sum(single_qf1_loss, vec_qf1_loss, self.single_loss_weight)
        qf2_loss = convex_sum(single_qf2_loss, vec_qf2_loss, self.single_loss_weight)

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            policy_loss = (log_pi - q_new_actions).mean()

            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['QF1 Loss single'] = np.mean(ptu.get_numpy(single_qf1_loss))
            eval_statistics['QF2 Loss single'] = np.mean(ptu.get_numpy(single_qf2_loss))
            eval_statistics['QF1 Loss vec'] = np.mean(ptu.get_numpy(vec_qf1_loss))
            eval_statistics['QF2 Loss vec'] = np.mean(ptu.get_numpy(vec_qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

        return SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        ), eval_statistics
