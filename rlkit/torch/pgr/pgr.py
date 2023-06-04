from collections import OrderedDict, namedtuple
from typing import Union, Tuple, MutableMapping, Optional
from numbers import Number

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction
from torch.distributions import Bernoulli
from torch.distributions.kl import kl_divergence
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.util.ml_util import ScalarSchedule, ConstantSchedule
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix

PGRLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss',
)


class PGRTrainer(TorchTrainer, LossFunction):
    NORMAL_REWARD = 'normal'
    DISCOUNTED_REWARD = 'discounted'
    DISCOUNTED_PLUS_TIME_KL = 'discounted_plus_time_kl'

    LEARNED_DISCOUNT = 'learned'
    PRIOR_DISCOUNT = 'prior'
    COMPUTED_DISCOUNT = 'computed_from_qr'
    COMPUTED_DISCOUNT_NO_PRIOR = 'computed_from_qr_no_prior'

    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            reward_type,

            discount_type=None,
            discount_model=None,
            discount=0.99,
            prior_discount_weight_schedule: Optional[ScalarSchedule] = None,
            multiply_bootstrap_by_prior_discount=False,
            upper_bound_discount_by_prior=False,
            reward_scale=1.0,
            reward_tracking_momentum=0.999,
            auto_init_qf_bias=False,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        """
        :param env:
        :param policy:
        :param qf1:
        :param qf2:
        :param target_qf1:
        :param target_qf2:
        :param reward_type:
        :param discount_type:
        :param discount_model:
        :param discount:
        :param prior_discount_weight_schedule:
        At epoch i, use the discount
            discount = c_i * prior_discount + (1-c_i) posterior_discount
        :param multiply_bootstrap_by_prior_discount:
        If true, when you compute the discount prior, always multiply it by the
        prior discount in addition to the normal prior discount.
        :param upper_bound_discount_by_prior:
        Always upper-bound the discount by the prior.
        :param reward_scale:
        :param reward_tracking_momentum:
        :param policy_lr:
        :param qf_lr:
        :param optimizer_class:
        :param soft_target_tau:
        :param target_update_period:
        :param plotter:
        :param render_eval_paths:
        :param use_automatic_entropy_tuning:
        :param target_entropy:
        """
        if reward_type not in {
            self.NORMAL_REWARD,
            self.DISCOUNTED_REWARD,
            self.DISCOUNTED_PLUS_TIME_KL,
        }:
            raise ValueError("Invalid reward type: {}".format(reward_type))
        if discount_type is None:  # preserve old behavior
            if reward_type == self.DISCOUNTED_PLUS_TIME_KL:
                discount_type = self.COMPUTED_DISCOUNT
            else:
                discount_type = self.PRIOR_DISCOUNT

        if discount_type not in {
            self.PRIOR_DISCOUNT,
            self.LEARNED_DISCOUNT,
            self.COMPUTED_DISCOUNT,
            self.COMPUTED_DISCOUNT_NO_PRIOR,
        }:
            raise ValueError("Invalid discount type: {}".format(
                discount_type
            ))
        if (
                reward_type == self.LEARNED_DISCOUNT
                and discount_model is None
        ):
            raise ValueError(
                "Need to set discount_model for using mode {}".format(
                    reward_type
                )
            )
        if not isinstance(reward_scale, Number) and reward_scale not in {
            'auto_normalize_by_max_magnitude',
            'auto_normalize_by_max_magnitude_times_10',
            'auto_normalize_by_max_magnitude_times_100',
            'auto_normalize_by_max_magnitude_times_invsig_prior',
            'auto_normalize_by_mean_magnitude',
        }:
            raise ValueError("Invalid reward_scale type: {}".format(
                reward_scale
            ))

        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.reward_type = reward_type
        self.discount_type = discount_type
        if prior_discount_weight_schedule is None:
            prior_discount_weight_schedule = ConstantSchedule(0.)
        self._prior_discount_weight_schedule = prior_discount_weight_schedule
        self._multiply_bootstrap_by_prior_discount = (
            multiply_bootstrap_by_prior_discount
        )
        self._upper_bound_discount_by_prior = (
            upper_bound_discount_by_prior
        )
        self._auto_init_qf_bias = auto_init_qf_bias
        self._current_epoch = 0

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount_model = discount_model
        self.discount = discount
        self.prior_on_discount = Bernoulli(self.discount)
        self._reward_scale = reward_scale
        self.reward_tracking_momentum = reward_tracking_momentum
        self._reward_normalizer = ptu.from_numpy(np.array(1))
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self._qfs_were_initialized = False

    def train_from_torch(self, batch):
        if self._need_to_update_eval_statistics:
            losses, stats = self.compute_loss(batch, return_statistics=True)
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
            self.eval_statistics = stats
        else:
            losses = self.compute_loss(batch)

        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

        if self._reward_scale == 'auto_normalize_by_mean_magnitude':
            rewards = batch['rewards']
            self._reward_normalizer = (
                    self._reward_normalizer * self.reward_tracking_momentum
                    + rewards.abs().mean() * (1 - self.reward_tracking_momentum)
            )
        elif self._reward_scale in {
            'auto_normalize_by_max_magnitude',
            'auto_normalize_by_max_magnitude_times_10',
            'auto_normalize_by_max_magnitude_times_100',
            'auto_normalize_by_max_magnitude_times_invsig_prior',
        }:
            rewards = batch['rewards']
            self._reward_normalizer = (
                    self._reward_normalizer * self.reward_tracking_momentum
                    + rewards.abs().max() * (1 - self.reward_tracking_momentum)
            )
        elif isinstance(self._reward_scale, Number):
            pass
        else:
            raise NotImplementedError()

        if self._num_train_steps % self.target_update_period == 0:
            self.update_target_networks()
        self._num_train_steps += 1

        if self._need_to_update_eval_statistics:
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    def compute_loss(
            self, batch, return_statistics=False,
    ) -> Union[PGRLosses, Tuple[PGRLosses, MutableMapping]]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        eval_statistics = OrderedDict()

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (
                    log_pi + self.target_entropy).detach()).mean()
        else:
            alpha_loss = 0

        alpha = self.get_alpha()

        if not self._qfs_were_initialized and self._auto_init_qf_bias:
            average_value = (rewards - alpha * log_pi).mean()
            self.qf1.last_fc.bias.data = average_value
            self.qf2.last_fc.bias.data = average_value
            self._qfs_were_initialized = False

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )

        """
        QF Loss
        """
        bootstrap_value, q1_pred, q2_pred, bootstrap_log_pi_term = (
            self.get_bootstrap_stats(
                obs,
                actions,
                next_obs,
            ))
        # Use the unscaled bootstrap values/rewards so that the weight on the
        # the Q-value/reward has the correct scale relative to the other terms
        raw_discount = self.get_discount_factor(
            bootstrap_value,
            rewards,
            obs,
            actions,
        )
        discount = (
                self._weight_on_prior_discount * self.discount
                + (1 - self._weight_on_prior_discount) * raw_discount
        )
        q_target = self._compute_target_q_value(
            discount,
            rewards,
            terminals,
            bootstrap_value,
            eval_statistics,
            return_statistics,
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Save some statistics for eval
        """
        if return_statistics:
            eval_statistics.update(create_stats_ordered_dict(
                'rewards',
                ptu.get_numpy(rewards),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'bootstrap log pi',
                ptu.get_numpy(bootstrap_log_pi_term),
            ))
            if isinstance(discount, torch.Tensor):
                eval_statistics.update(create_stats_ordered_dict(
                    'discount factor',
                    ptu.get_numpy(raw_discount),
                ))
            else:
                eval_statistics.update(create_stats_ordered_dict(
                    'discount factor',
                    np.array([raw_discount]),
                ))
            if isinstance(discount, torch.Tensor):
                eval_statistics.update(create_stats_ordered_dict(
                    'used discount factor',
                    ptu.get_numpy(discount),
                ))
            else:
                eval_statistics.update(create_stats_ordered_dict(
                    'used discount factor',
                    np.array([discount]),
                ))
            eval_statistics[
                'weight on prior discount'] = self._weight_on_prior_discount
            reward_scale = self.reward_scale
            if isinstance(reward_scale, torch.Tensor):
                reward_scale = ptu.get_numpy(reward_scale)
            eval_statistics['reward scale'] = reward_scale
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            eval_statistics['Policy Q-only Loss'] = np.mean(ptu.get_numpy(
                -q_new_actions
            ))
            eval_statistics['Policy entropy-only Loss'] = np.mean(ptu.get_numpy(
                alpha * log_pi
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
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        losses = PGRLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )
        if return_statistics:
            return losses, eval_statistics
        else:
            return losses

    def get_bootstrap_stats(self, obs, actions, next_obs):
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        alpha = self.get_alpha()
        bootstrap_log_pi_term = - alpha * new_log_pi
        bootstrap_value = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) + bootstrap_log_pi_term
        return bootstrap_value, q1_pred, q2_pred, bootstrap_log_pi_term

    @property
    def reward_scale(self):
        if self._reward_scale == 'auto_normalize_by_max_magnitude':
            return 1. / self._reward_normalizer
        elif self._reward_scale == 'auto_normalize_by_max_magnitude_times_10':
            return 10. / self._reward_normalizer
        elif self._reward_scale == 'auto_normalize_by_max_magnitude_times_100':
            return 100. / self._reward_normalizer
        elif self._reward_scale == 'auto_normalize_by_max_magnitude_times_invsig_prior':
            return (np.log(self.discount) - np.log(1 - self.discount)
                    ) / self._reward_normalizer
        elif self._reward_scale == 'auto_normalize_by_mean_magnitude':
            return 1. / self._reward_normalizer
        elif isinstance(self._reward_scale, Number):
            return self._reward_scale
        else:
            raise ValueError(self._reward_scale)

    def _compute_target_q_value(
            self,
            discount,
            rewards,
            terminals,
            bootstrap_value,
            statistics_log,
            update_statistics,
    ):
        scaled_rewards = rewards * self.reward_scale
        del rewards
        if self.reward_type == self.NORMAL_REWARD:
            reward_target = scaled_rewards
        elif self.reward_type == self.DISCOUNTED_REWARD:
            reward_target = scaled_rewards * (1 - discount)
        elif self.reward_type == self.DISCOUNTED_PLUS_TIME_KL:
            kl_reward = kl_divergence(
                Bernoulli(discount),
                self.prior_on_discount,
            )
            reward_target = (
                    scaled_rewards * (1 - discount) + kl_reward
            )
            if update_statistics:
                statistics_log.update(create_stats_ordered_dict(
                    'time_kl_reward',
                    ptu.get_numpy(kl_reward),
                ))
                statistics_log.update(create_stats_ordered_dict(
                    'inferred_discount',
                    ptu.get_numpy(discount),
                ))
        else:
            raise ValueError("Unknown update type".format(self.reward_type))
        if self._multiply_bootstrap_by_prior_discount:
            bootstrap_target = (
                (1. - terminals) * discount * bootstrap_value * self.discount
            )
        else:
            bootstrap_target = (
                (1. - terminals) * discount * bootstrap_value
            )
        q_target = reward_target + bootstrap_target
        return q_target

    def get_discount_factor(
            self, bootstrap_value, unscaled_reward, obs, action
    ):
        # TODO: train a separate Q-value for the log-pi terms so that the reward
        # scale matches
        prior_discount = self.discount  # rename for readability
        if self.discount_type == self.PRIOR_DISCOUNT:
            discount = prior_discount
        elif self.discount_type == self.LEARNED_DISCOUNT:
            discount = self.discount_model(obs, action)
        elif self.discount_type == self.COMPUTED_DISCOUNT:
            # large reward or tiny prior ==> small current discount
            discount = torch.sigmoid(
                bootstrap_value
                - unscaled_reward * self.reward_scale
                + np.log(prior_discount / (1 - prior_discount))
            ).detach()
        elif self.discount_type == self.COMPUTED_DISCOUNT_NO_PRIOR:
            # large reward or tiny prior ==> small current discount
            discount = torch.sigmoid(
                bootstrap_value
                - unscaled_reward * self.reward_scale
            ).detach()
        else:
            raise ValueError("Unknown discount type".format(
                self.discount_type
            ))
        if self._upper_bound_discount_by_prior and discount is not prior_discount:
            discount = torch.clamp(discount, max=prior_discount)
        return discount

    def get_alpha(self):
        if self.use_automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
        return alpha

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    @property
    def _weight_on_prior_discount(self):
        return self._prior_discount_weight_schedule.get_value(
            self._current_epoch
        )

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self._current_epoch = epoch + 1

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
