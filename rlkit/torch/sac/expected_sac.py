from collections import OrderedDict

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.sac.sac import SoftActorCritic

EXACT = 'exact'
MEAN_ACTION = 'mean_action'
SAMPLE = 'sample'


class ExpectedSAC(SoftActorCritic):
    """
    Compute

    E_{a \sim \pi(. | s)}[Q(s, a) - \log \pi(a | s)]

    in closed form
    """

    def __init__(
            self,
            *args,
            expected_qf_estim_strategy='exact',
            expected_log_pi_estim_strategy='exact',
            **kwargs
    ):
        """

        :param args:
        :param expected_qf_estim_strategy: String describing how to estimate
            E[Q(s, A)]:
                'exact': estimate exactly by convolving Q with Gaussian
                'mean_action': estimate with Q(s, E[A])
                'sample': estimate with one sample of Q(s, A)
        :param expected_log_pi_estim_strategy: String describing how to
            estimate E[log \pi(A | s)]
                'exact': compute in closed form
                'mean_action': estimate with log \pi(E[A] | s)
                'sample': estimate with one sample of log \pi(A | s)
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        assert expected_qf_estim_strategy in [EXACT, MEAN_ACTION, SAMPLE]
        assert expected_log_pi_estim_strategy in [EXACT, MEAN_ACTION, SAMPLE]
        self.expected_qf_estim_strategy = expected_qf_estim_strategy
        self.expected_log_pi_estim_strategy = expected_log_pi_estim_strategy

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        q_pred = self.qf(obs, actions)
        v_pred = self.vf(obs)
        # Make sure policy accounts for squashing functions like tanh correctly!
        (
            new_actions, policy_mean, policy_log_std, log_pi, entropy,
            policy_stds, log_pi_mean
        ) = self.policy(
            obs,
            return_log_prob=True,
            return_entropy=(
                self.expected_log_pi_estim_strategy == EXACT
            ),
            return_log_prob_of_mean=(
                self.expected_log_pi_estim_strategy == MEAN_ACTION
            ),
        )
        expected_log_pi = - entropy

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        """
        VF Loss
        """
        q_new_actions = self.qf(obs, new_actions)
        if self.expected_qf_estim_strategy == EXACT:
            expected_q = self.qf(obs, policy_mean, action_stds=policy_stds)
        elif self.expected_qf_estim_strategy == MEAN_ACTION:
            expected_q = self.qf(obs, policy_mean)
        elif self.expected_qf_estim_strategy == SAMPLE:
            expected_q = q_new_actions
        else:
            raise TypeError("Invalid E[Q(s, a)] estimation strategy: {}".format(
                self.expected_qf_estim_strategy
            ))
        if self.expected_log_pi_estim_strategy == EXACT:
            expected_log_pi_target = expected_log_pi
        elif self.expected_log_pi_estim_strategy == MEAN_ACTION:
            expected_log_pi_target = log_pi_mean
        elif self.expected_log_pi_estim_strategy == SAMPLE:
            expected_log_pi_target = log_pi
        else:
            raise TypeError(
                "Invalid E[log pi(a|s)] estimation strategy: {}".format(
                    self.expected_log_pi_estim_strategy
                )
            )
        v_target = expected_q - expected_log_pi_target
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Policy Loss
        """
        # paper says to do + but Tuomas said that's a typo. Do Q - V.
        log_policy_target = q_new_actions - v_pred
        policy_loss = (
            log_pi * (log_pi - log_policy_target).detach()
        ).mean()
        policy_reg_loss = self.policy_reg_weight * (
            (policy_mean ** 2).mean()
            + (policy_log_std ** 2).mean()
        )
        policy_loss = policy_loss + policy_reg_loss

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._update_target_network()

        """
        Save some statistics for eval
        """
        self.eval_statistics = OrderedDict()
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
            'V Predictions',
            ptu.get_numpy(v_pred),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Log Pis',
            ptu.get_numpy(log_pi),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Policy mu',
            ptu.get_numpy(policy_mean),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Policy log std',
            ptu.get_numpy(policy_log_std),
        ))
