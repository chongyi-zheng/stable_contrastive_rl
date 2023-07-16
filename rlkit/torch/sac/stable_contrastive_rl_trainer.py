from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix

from rlkit.utils.data_augmentation import AUG_TO_FUNC


class StableContrastiveRLTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf,
            target_qf=None,
            discount=0.99,
            lr=3e-4,
            gradient_clipping=None,  # TODO
            optimizer_class=optim.Adam,
            update_period=1,
            soft_target_tau=1e-2,
            target_update_period=1,
            use_td=False,
            use_td_cpc=False,
            entropy_coefficient=None,
            target_entropy=None,
            bc_coef=0.05,
            augment_order=[],
            augment_probability=0.0,

            * args,
            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf = qf
        self.target_qf = target_qf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.gradient_clipping = gradient_clipping
        self.use_td = use_td
        self.use_td_cpc = use_td_cpc
        self.entropy_coefficient = entropy_coefficient
        self.adaptive_entropy_coefficient = entropy_coefficient is None
        self.target_entropy = target_entropy
        self.bc_coef = bc_coef
        self.discount = discount
        self.update_period = update_period
        self.augment_probability = augment_probability

        if self.adaptive_entropy_coefficient:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=lr,
            )

        self.qf_criterion = nn.BCEWithLogitsLoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=lr,
        )

        ptu.copy_model_params_from_to(self.qf, self.target_qf)

        self.eval_statistics = OrderedDict()
        self.n_train_steps_total = 0

        self.need_to_update_eval_statistics = {
            'train/': True,
            'eval/': True,
        }

        self.augment_stack = None
        self.augment_funcs = {}
        if augment_probability > 0:
            self.augment_funcs = {}
            for aug_name in augment_order:
                assert aug_name in AUG_TO_FUNC, 'invalid data aug string'
                self.augment_funcs[aug_name] = AUG_TO_FUNC[aug_name]

    def augment(self, batch, train=True):
        augmented_batch = dict()
        for key, value in batch.items():
            augmented_batch[key] = value

        if (train and self.augment_probability > 0 and
                torch.rand(1) < self.augment_probability and
                batch['observations'].shape[0] > 0):
            width = self.policy.input_width
            height = self.policy.input_height
            channel = self.policy.input_channels

            img_obs = batch['observations'].reshape(
                -1, channel, width, height)
            next_img_obs = batch['next_observations'].reshape(
                -1, channel, width, height)
            img_goal = batch['contexts'].reshape(
                -1, channel, width, height)

            # transpose to (B, C, H, W)
            aug_img_obs = img_obs.permute(0, 1, 3, 2)
            aug_img_goal = img_goal.permute(0, 1, 3, 2)
            aug_next_img_obs = next_img_obs.permute(0, 1, 3, 2)

            for aug, func in self.augment_funcs.items():
                # apply same augmentation
                aug_img_obs_goal = func(torch.cat([aug_img_obs, aug_img_goal], dim=1))
                aug_img_obs, aug_img_goal = aug_img_obs_goal[:, :channel], aug_img_obs_goal[:, channel:]
                aug_next_img_obs = func(aug_next_img_obs)

            # transpose to (B, C, W, H)
            aug_img_obs = aug_img_obs.reshape([-1, channel, height, width])
            aug_img_goal = aug_img_goal.reshape([-1, channel, height, width])
            aug_next_img_obs = aug_next_img_obs.reshape([-1, channel, height, width])

            augmented_batch['augmented_observations'] = aug_img_obs.flatten(1)
            augmented_batch['augmented_next_observations'] = aug_next_img_obs.flatten(1)
            augmented_batch['augmented_contexts'] = aug_img_goal.flatten(1)
        else:
            augmented_batch['augmented_observations'] = augmented_batch['observations']
            augmented_batch['augmented_next_observations'] = augmented_batch['next_observations']
            augmented_batch['augmented_contexts'] = augmented_batch['contexts']

        return augmented_batch

    def train_from_torch(self, batch, train=True):
        if train:
            for net in self.networks:
                net.train(True)
        else:
            for net in self.networks:
                net.train(False)

        batch['observations'] = batch['observations'] / 255.0
        batch['next_observations'] = batch['next_observations'] / 255.0
        batch['contexts'] = batch['contexts'] / 255.0

        batch = self.augment(batch, train=train)

        reward = batch['rewards']
        terminal = batch['terminals']
        action = batch['actions']

        obs = batch['observations']
        next_obs = batch['next_observations']
        goal = batch['contexts']

        aug_obs = batch['augmented_observations']
        aug_goal = batch['augmented_contexts']

        batch_size = obs.shape[0]
        new_goal = goal

        if self.use_td or self.use_td_cpc:
            new_goal = next_obs
        I = torch.eye(batch_size, device=ptu.device)
        logits, sa_repr, g_repr, sa_repr_norm, g_repr_norm = self.qf(
            torch.cat([obs, new_goal], -1), action, repr=True)

        # compute classifier accuracies
        logits_log = logits.mean(-1)
        correct = (torch.argmax(logits_log, dim=-1) == torch.argmax(I, dim=-1))
        logits_pos = torch.sum(logits_log * I) / torch.sum(I)
        logits_neg = torch.sum(logits_log * (1 - I)) / torch.sum(1 - I)
        q_pos, q_neg = torch.sum(torch.sigmoid(logits_log) * I) / torch.sum(I), \
                       torch.sum(torch.sigmoid(logits_log) * (1 - I)) / torch.sum(1 - I)
        q_pos_ratio, q_neg_ratio = q_pos / (1 - q_pos), q_neg / (1 - q_neg)
        binary_accuracy = torch.mean(((logits_log > 0) == I).float())
        categorical_accuracy = torch.mean(correct.float())

        if self.use_td:
            # Make sure to use the twin Q trick.
            assert len(logits.shape) == 3

            # we evaluate the next-state Q function using random goals
            goal_indices = torch.roll(
                torch.arange(batch_size, dtype=torch.int64), -1)

            random_goal = new_goal[goal_indices]

            next_s_rand_g = torch.cat([next_obs, random_goal], -1)

            next_dist = self.policy(next_s_rand_g)
            next_action = next_dist.rsample()

            next_q = self.target_qf(
                next_s_rand_g, next_action)

            next_q = torch.sigmoid(next_q)
            next_v = torch.min(next_q, dim=-1)[0].detach()
            next_v = torch.diag(next_v)
            w = next_v / (1 - next_v)
            w_clipping = 20.0
            w = torch.clamp(w, min=0.0, max=w_clipping)

            # (B, B, 2) --> (B, 2), computes diagonal of each twin Q.
            pos_logits = torch.diagonal(logits).permute(1, 0)
            loss_pos = self.qf_criterion(
                pos_logits, ptu.ones_like(pos_logits))

            neg_logits = logits[torch.arange(batch_size), goal_indices]
            loss_neg1 = w[:, None] * self.qf_criterion(
                neg_logits, ptu.ones_like(neg_logits))
            loss_neg2 = self.qf_criterion(
                neg_logits, ptu.zeros_like(neg_logits))

            qf_loss = (1 - self.discount) * loss_pos + \
                      self.discount * loss_neg1 + loss_neg2
            qf_loss = torch.mean(qf_loss)
        elif self.use_td_cpc:
            w = ptu.zeros(1)
            assert not self.use_td

            next_s_new_g = torch.cat([next_obs, new_goal], -1)
            next_dist = self.policy(next_s_new_g)
            next_action = next_dist.rsample()

            next_logits = self.target_qf(
                next_s_new_g, next_action)
            next_logits = torch.min(next_logits, dim=-1)[0].detach()

            logit_max = torch.max(next_logits, dim=1, keepdim=True)[0]
            unnormalized = torch.exp(next_logits - logit_max) * (1 - I)
            neg_labels = unnormalized / torch.sum(unnormalized, dim=1, keepdim=True)

            # These cross entropy losses should be run with PyTorch >= 1.10
            # https://discuss.pytorch.org/t/cross-entropy-with-logit-targets/134068/3
            ce_loss = nn.CrossEntropyLoss(reduction='none')
            loss_pos = ce_loss(logits, I.unsqueeze(-1).repeat_interleave(logits.shape[-1], dim=-1))
            loss_neg = ce_loss(logits, neg_labels.unsqueeze(-1).repeat_interleave(logits.shape[-1], dim=-1))
            qf_loss = (1 - self.discount) * loss_pos + self.discount * loss_neg

            qf_loss = torch.mean(qf_loss)
        else:  # For the MC losses.
            w = ptu.zeros(1)

            # decrease the weight of negative term to 1 / (B - 1)
            qf_loss_weights = ptu.ones((batch_size, batch_size)) / (batch_size - 1)
            qf_loss_weights[torch.arange(batch_size), torch.arange(batch_size)] = 1
            if len(logits.shape) == 3:
                # logits.shape = (B, B, 2) with 1 term for positive pair
                # and (B - 1) terms for negative pairs in each row
                qf_loss = self.qf_criterion(
                    logits, I.unsqueeze(-1).repeat_interleave(logits.shape[-1], dim=-1)).mean(-1)
            else:
                qf_loss = self.qf_criterion(logits, I)
            qf_loss *= qf_loss_weights

            qf_loss = torch.mean(qf_loss)

        """
        Policy and Alpha Loss
        """
        obs_goal = torch.cat([obs, goal], -1)
        aug_obs_goal = torch.cat([aug_obs, aug_goal], -1)
        dist = self.policy(obs_goal)
        dist_aug = self.policy(aug_obs_goal)
        sampled_action, log_prob = dist.rsample_and_logprob()

        if self.adaptive_entropy_coefficient:
            alpha_loss = -(self.log_alpha.exp() * (
                log_prob + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.entropy_coefficient

        q_action = self.qf(obs_goal, sampled_action)

        if len(q_action.shape) == 3:  # twin q trick
            assert q_action.shape[2] == 2
            q_action = torch.min(q_action, dim=-1)[0]

        actor_q_loss = alpha * log_prob - torch.diag(q_action)

        assert 0.0 <= self.bc_coef <= 1.0
        orig_action = action

        train_mask = ((orig_action * 1E8 % 10)[:, 0] != 4).float()

        gcbc_loss = -train_mask * dist.log_prob(orig_action)
        gcbc_val_loss = -(1.0 - train_mask) * dist.log_prob(orig_action)
        aug_gcbc_loss = -train_mask * dist_aug.log_prob(orig_action)
        aug_gcbc_val_loss = -(1.0 - train_mask) * dist_aug.log_prob(orig_action)

        actor_loss = self.bc_coef * aug_gcbc_loss + (1 - self.bc_coef) * actor_q_loss

        gcbc_loss_log = torch.sum(gcbc_loss) / torch.sum(train_mask)
        aug_gcbc_loss_log = torch.sum(aug_gcbc_loss) / torch.sum(train_mask)
        if torch.sum(1 - train_mask) > 0:
            gcbc_val_loss_log = torch.sum(gcbc_val_loss) / torch.sum(1 - train_mask)
            aug_gcbc_val_loss_log = torch.sum(aug_gcbc_val_loss) / torch.sum(1 - train_mask)
        else:
            gcbc_val_loss_log = ptu.zeros(1)
            aug_gcbc_val_loss_log = ptu.zeros(1)

        actor_loss = torch.mean(actor_loss)

        if train:
            """
            Optimization.
            """
            if self.n_train_steps_total % self.update_period == 0:
                if self.adaptive_entropy_coefficient:
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    if (self.gradient_clipping is not None and
                            self.gradient_clipping > 0):
                        torch.nn.utils.clip_grad_norm(
                            [self.log_alpha], self.gradient_clipping)
                    self.alpha_optimizer.step()

                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                if (self.gradient_clipping is not None and
                        self.gradient_clipping > 0):
                    torch.nn.utils.clip_grad_norm(
                        self.policy.parameters(), self.gradient_clipping)
                self.policy_optimizer.step()

                self.qf_optimizer.zero_grad()
                qf_loss.backward()
                if (self.gradient_clipping is not None and
                        self.gradient_clipping > 0):
                    torch.nn.utils.clip_grad_norm(
                        self.qf.parameters(), self.gradient_clipping)
                self.qf_optimizer.step()

            """
            Soft Updates
            """
            if self.n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.qf, self.target_qf, self.soft_target_tau
                )

        """
        Save some statistics for eval
        """
        if train:
            prefix = 'train/'
        else:
            prefix = 'eval/'

        if self.need_to_update_eval_statistics[prefix]:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics[prefix + 'QF Loss'] = np.mean(
                ptu.get_numpy(qf_loss))
            self.eval_statistics[prefix + 'Policy Loss'] = np.mean(
                ptu.get_numpy(actor_loss))
            self.eval_statistics[prefix + 'Policy Loss/Actor Q Loss'] = np.mean(
                ptu.get_numpy(actor_q_loss))
            self.eval_statistics[prefix + 'Policy Loss/GCBC Loss'] = np.mean(
                ptu.get_numpy(gcbc_loss_log))
            self.eval_statistics[prefix + 'Policy Loss/GCBC Val Loss'] = np.mean(
                ptu.get_numpy(gcbc_val_loss_log))
            self.eval_statistics[prefix + 'Policy Loss/Augmented GCBC Loss'] = np.mean(
                ptu.get_numpy(aug_gcbc_loss_log))
            self.eval_statistics[prefix + 'Policy Loss/Augmented GCBC Val Loss'] = np.mean(
                ptu.get_numpy(aug_gcbc_val_loss_log))

            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Policy Mean',
                ptu.get_numpy(dist.mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Policy STD',
                ptu.get_numpy(dist.stddev),
            ))

            # critic statistics
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'qf/sa_repr_norm',
                ptu.get_numpy(sa_repr_norm),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'qf/g_repr_norm',
                ptu.get_numpy(g_repr_norm),
            ))

            if self.qf.repr_norm:
                self.eval_statistics[prefix + 'qf/repr_log_scale'] = np.mean(
                    ptu.get_numpy(self.qf.repr_log_scale))

            self.eval_statistics[prefix + 'qf/logits_pos'] = np.mean(
                ptu.get_numpy(logits_pos))
            self.eval_statistics[prefix + 'qf/logits_neg'] = np.mean(
                ptu.get_numpy(logits_neg))
            self.eval_statistics[prefix + 'qf/q_pos_ratio'] = np.mean(
                ptu.get_numpy(q_pos_ratio))
            self.eval_statistics[prefix + 'qf/q_neg_ratio'] = np.mean(
                ptu.get_numpy(q_neg_ratio))
            self.eval_statistics[prefix + 'qf/binary_accuracy'] = np.mean(
                ptu.get_numpy(binary_accuracy))
            self.eval_statistics[prefix + 'qf/categorical_accuracy'] = np.mean(
                ptu.get_numpy(categorical_accuracy))

            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'logits',
                ptu.get_numpy(logits),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'w',
                ptu.get_numpy(w),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'reward',
                ptu.get_numpy(reward),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'terminal',
                ptu.get_numpy(terminal),
            ))
            actor_statistics = add_prefix(
                dist.get_diagnostics(), prefix + 'policy/')
            self.eval_statistics.update(actor_statistics)

            if self.entropy_coefficient is not None:
                self.eval_statistics[prefix + 'alpha'] = alpha
            else:
                self.eval_statistics[prefix + 'alpha'] = np.mean(
                    ptu.get_numpy(alpha))
            if self.adaptive_entropy_coefficient:
                self.eval_statistics[prefix + 'Alpha Loss'] = np.mean(
                    ptu.get_numpy(alpha_loss))

        if train:
            self.n_train_steps_total += 1

        self.need_to_update_eval_statistics[prefix] = False

        for net in self.networks:
            net.train(False)

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        for key in self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics[key] = True

    @property
    def networks(self):
        nets = [
            self.policy,
            self.qf,
            self.target_qf,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf=self.qf,
            target_qf=self.target_qf,
        )
