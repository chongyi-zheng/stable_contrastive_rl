from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from rlkit.torch.core import np_to_pytorch_batch
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd
import time
from rlkit.core import logger

class BEARTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            vae,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            pretraining_env_logging_period=100000,
            pretraining_logging_period=1000,
            do_pretrain_rollouts=False,
            num_pretrain_steps=100000,
            replay_buffer=None,

            # BEAR specific params
            mode='auto',
            kernel_choice='laplacian',
            policy_update_style=0,
            mmd_sigma=10.0,
            target_mmd_thresh=0.05,
            online_target_mmd_thresh=None,
            num_samples_mmd_match=4,
            with_grad_penalty_v1=False,
            with_grad_penalty_v2=False,
            grad_coefficient_policy=0.0,
            grad_coefficient_q=0.0,
            use_target_nets=False,
            policy_update_delay=100,
            start_epoch_grad_penalty=0,
            num_steps_policy_update_only=50,
            bc_pretrain_steps=20000,
            target_update_method='default',
            use_adv_weighting=False,
            positive_reward=False,

    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vae = vae
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

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
        self.vae_optimizer = optimizer_class(
            self.vae.parameters(),
            lr=3e-4,
        )

        self.mode = mode
        if self.mode == 'auto':
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=1e-3,
            )
        self.mmd_sigma = mmd_sigma
        self.kernel_choice = kernel_choice
        self.num_samples_mmd_match = num_samples_mmd_match
        self.policy_update_style = policy_update_style
        self.target_mmd_thresh = target_mmd_thresh
        if online_target_mmd_thresh is not None:
            self.online_target_mmd_thresh = online_target_mmd_thresh
        else:
            self.online_target_mmd_thresh = target_mmd_thresh

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self._with_gradient_penalty_v1 = with_grad_penalty_v1
        self._with_gradient_penalty_v2 = with_grad_penalty_v2
        self._grad_coefficient_q = grad_coefficient_q
        self._grad_coefficient_policy = grad_coefficient_policy
        self._use_target_nets = use_target_nets
        self._policy_delay_update = policy_update_delay
        self._num_policy_steps = num_steps_policy_update_only
        self._start_epoch_grad_penalty = start_epoch_grad_penalty
        self._bc_pretrain_steps = bc_pretrain_steps
        self._target_update_method = target_update_method
        self._use_adv_weighting = use_adv_weighting
        self._positive_reward = positive_reward

        if self._target_update_method == 'distillation':
            self.target_qf1_opt = optimizer_class(
                self.target_qf1.parameters(), lr=qf_lr
            )
            self.target_qf2_opt = optimizer_class(
                self.target_qf2.parameters(), lr=qf_lr
            )

        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0

        self.pretraining_env_logging_period = pretraining_env_logging_period
        self.pretraining_logging_period = pretraining_logging_period
        self.do_pretrain_rollouts = do_pretrain_rollouts
        self.num_pretrain_steps = num_pretrain_steps
        self.replay_buffer=replay_buffer

        if not self._use_target_nets:
            self.target_qf1 = qf1
            self.target_qf2 = qf2

    def eval_q_custom(self, custom_policy, data_batch, q_function=None):
        if q_function is None:
            q_function = self.qf1

        obs = data_batch['observations']
        # Evaluate policy Loss
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        q_new_actions = q_function(obs, new_obs_actions)
        return float(q_new_actions.mean().detach().cpu().numpy())

    def adv_mmd_loss_laplacian(self, samples1, samples2, adv1, sigma=0.2):
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)
        diff_x_x = torch.mean(adv1 * adv1.permute(0, 2, 1) * (
                    -(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean(
            adv1 * (-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(),
            dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(
            1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def adv_mmd_loss_gaussian(self, samples1, samples2, adv1, sigma=0.2):
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)
        diff_x_x = torch.mean(adv1 * adv1.permute(0, 2, 1) * (
                    -(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean(
            adv1 * (-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(),
            dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(
            1)  # B x N x N x d
        diff_y_y = torch.mean(
            (-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(
            1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(
            1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(),
                              dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(
            1)  # B x N x N x d
        diff_x_x = torch.mean(
            (-(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean(
            (-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(
            1)  # B x N x N x d
        diff_y_y = torch.mean(
            (-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def do_rollouts(self):
        total_ret = 0
        for _ in range(20):
            o = self.env.reset()
            ret = 0
            for _ in range(1000):
                a, _ = self.policy.get_action(o)
                o, r, done, info = self.env.step(a)
                ret += r
                if done:
                    break
            total_ret += ret
        return total_ret

    def pretrain_q_with_bc_data(self, batch_size):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain_q.csv', relative_to_snapshot_dir=True
        )

        self.update_policy = True
        # then train policy and Q function together
        prev_time = time.time()
        for i in range(self.num_pretrain_steps):
            self.eval_statistics = dict()
            if i % self.pretraining_logging_period == 0:
                self._need_to_update_eval_statistics=True
            train_data = self.replay_buffer.random_batch(batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            # goals = train_data['resampled_goals']
            train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
            train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
            self.train_from_torch(train_data)
            if self.do_pretrain_rollouts and i % self.pretraining_env_logging_period == 0:
                total_ret = self.do_rollouts()
                print("Return at step {} : {}".format(i, total_ret/20))

            if i%self.pretraining_logging_period==0:
                if self.do_pretrain_rollouts:
                    self.eval_statistics["pretrain_bc/avg_return"] = total_ret / 20
                self.eval_statistics["batch"] = i
                self.eval_statistics["epoch_time"] = time.time()-prev_time
                logger.record_dict(self.eval_statistics)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)
                prev_time = time.time()

        logger.remove_tabular_output(
            'pretrain_q.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )

        self._need_to_update_eval_statistics = True
        self.eval_statistics = dict()

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']
        if self._positive_reward:
            rewards += 2.5  # Make rewards positive
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Behavior clone a policy
        """
        z_dist = self.vae.encoder(obs, actions)
        mean, std = z_dist.mean, z_dist.stddev
        z = z_dist.rsample()
        recon = self.vae.decoder(obs, z).rsample()
        recon_loss = self.qf_criterion(recon, actions)
        kl_loss = -0.5 * (
                    1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        """
        Critic Training
        """
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            # Duplicate state 10 times (10 is a hyperparameter chosen by BCQ)
            state_rep = next_obs.unsqueeze(1).repeat(1, 10, 1).view(
                next_obs.shape[0] * 10, next_obs.shape[1])

            # Compute value of perturbed actions sampled from the VAE
            action_rep = self.policy(state_rep).sample()
            target_qf1 = self.target_qf1(state_rep, action_rep)
            target_qf2 = self.target_qf2(state_rep, action_rep)

            # Soft Clipped Double Q-learning
            target_Q = 0.75 * torch.min(target_qf1,
                                        target_qf2) + 0.25 * torch.max(
                target_qf1, target_qf2)
            target_Q = target_Q.view(next_obs.shape[0], -1).max(1)[0].view(-1,
                                                                           1)
            target_Q = self.reward_scale * rewards + (
                        1.0 - terminals) * self.discount * target_Q

        qf1_pred = self.qf1(obs, actions)
        qf2_pred = self.qf2(obs, actions)

        qf1_loss = (qf1_pred - target_Q.detach()).pow(2).mean()
        qf2_loss = (qf2_pred - target_Q.detach()).pow(2).mean()

        """
        Actor Training
        """
        sampled_actions, raw_sampled_actions = self.vae.decode_multiple(obs,
                                                                        num_decode=self.num_samples_mmd_match)
        actor_samples, raw_actor_actions = self.policy(
            obs.unsqueeze(1).repeat(1, self.num_samples_mmd_match, 1).view(-1,
                                                                           obs.shape[
                                                                               1]),
            ).rsample_with_pretanh()
        actor_samples = actor_samples.view(obs.shape[0],
                                           self.num_samples_mmd_match,
                                           actions.shape[1])
        raw_actor_actions = raw_actor_actions.view(obs.shape[0],
                                                   self.num_samples_mmd_match,
                                                   actions.shape[1])

        if self._use_adv_weighting:
            # import ipdb; ipdb.set_trace()
            qf1_orig = self.qf1(
                obs.unsqueeze(1).repeat(1, self.num_samples_mmd_match, 1).view(
                    -1, obs.shape[1]),
                sampled_actions.view(-1, actions.shape[1])
            )

            # Get the target_q for next state
            state_rep = next_obs.unsqueeze(1).repeat(1, 10, 1).view(
                next_obs.shape[0] * 10, next_obs.shape[1])
            action_rep = self.policy(state_rep)[0]
            target_q = torch.min(
                self.qf1(state_rep, action_rep),
                self.qf1(state_rep, action_rep),
            )
            target_q = target_q.view(next_obs.shape[0], -1).max(1)[0].view(-1,
                                                                           1)
            target_q = self.reward_scale * rewards + (
                        1.0 - terminals) * self.discount * target_q

            adv_mmd = (qf1_orig.view(obs.shape[0], self.num_samples_mmd_match,
                                     1) - target_q.unsqueeze(1))
            adv_mmd_not_clipped = adv_mmd
            adv_mmd = adv_mmd.exp().clamp_(max=10.0, min=1.0)
            adv_mmd = adv_mmd.detach()

        if self.kernel_choice == 'laplacian':
            if self._use_adv_weighting:
                mmd_loss = self.adv_mmd_loss_laplacian(raw_sampled_actions,
                                                       raw_actor_actions,
                                                       adv_mmd,
                                                       sigma=self.mmd_sigma)
            else:
                mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions,
                                                   raw_actor_actions,
                                                   sigma=self.mmd_sigma)
        elif self.kernel_choice == 'gaussian':
            if self._use_adv_weighting:
                mmd_loss = self.adv_mmd_loss_gaussian(raw_sampled_actions,
                                                      raw_actor_actions,
                                                      adv_mmd,
                                                      sigma=self.mmd_sigma)
            else:
                mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions,
                                                  raw_actor_actions,
                                                  sigma=self.mmd_sigma)

        action_divergence = ((sampled_actions - actor_samples) ** 2).sum(-1)
        raw_action_divergence = (
                    (raw_sampled_actions - raw_actor_actions) ** 2).sum(-1)

        q_val1 = self.qf1(obs, actor_samples[:, 0, :])
        q_val2 = self.qf2(obs, actor_samples[:, 0, :])

        if self.policy_update_style == '0':
            policy_loss = torch.min(q_val1, q_val2)[:, 0]
        elif self.policy_update_style == '1':
            policy_loss = 0.5 * (q_val1 + q_val2)[:, 0]

        if self._n_train_steps_total >= self._bc_pretrain_steps:
            # Now we can update the policy
            if self.mode == 'auto':
                policy_loss = (-policy_loss + self.log_alpha.exp() * (
                            mmd_loss - self.online_target_mmd_thresh)).mean()
            else:
                policy_loss = (-policy_loss + 100 * mmd_loss).mean()
        else:
            if self.mode == 'auto':
                policy_loss = (self.log_alpha.exp() * (
                            mmd_loss - self.target_mmd_thresh)).mean()
            else:
                policy_loss = 100 * mmd_loss.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        if self.mode == 'auto':
            policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        if self.mode == 'auto':
            self.alpha_optimizer.zero_grad()
            (-policy_loss).backward()
            self.alpha_optimizer.step()
            self.log_alpha.data.clamp_(min=-5.0, max=10.0)

        """
        Update networks
        """
        if self._use_target_nets:
            if self._target_update_method == 'default':
                if self._n_train_steps_total % self.target_update_period == 0:
                    ptu.soft_update_from_to(
                        self.qf1, self.target_qf1, self.soft_target_tau
                    )
                    ptu.soft_update_from_to(
                        self.qf2, self.target_qf2, self.soft_target_tau
                    )

        """
        Some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics[
                'Num Policy Updates'] = self._num_policy_update_steps
            if (
                    self._with_gradient_penalty_v1 or self._with_gradient_penalty_v2) and (
                    self._current_epoch > self._start_epoch_grad_penalty + 1):
                self.eval_statistics['Grad QF1 Loss'] = np.mean(
                    ptu.get_numpy(grad_qf1_square) * self._grad_coefficient_q)
                self.eval_statistics['Grad QF2 Loss'] = np.mean(
                    ptu.get_numpy(grad_qf2_square) * self._grad_coefficient_q)
                self.eval_statistics['Grad Policy Loss'] = np.mean(
                    ptu.get_numpy(grad_square) * self._grad_coefficient_policy)
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(qf1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(qf2_pred),
            ))
            if self._use_adv_weighting:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Adv MMD',
                    ptu.get_numpy(adv_mmd_not_clipped)
                ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(target_Q),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'MMD Loss',
                ptu.get_numpy(mmd_loss)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Action Divergence',
                ptu.get_numpy(action_divergence)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Raw Action Divergence',
                ptu.get_numpy(raw_action_divergence)
            ))
            if self.mode == 'auto':
                self.eval_statistics['Alpha'] = self.log_alpha.exp().item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.vae
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            vae=self.vae,
        )
