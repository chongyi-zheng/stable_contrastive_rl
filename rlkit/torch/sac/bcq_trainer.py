"""Implementation of BCQ, adapted from
https://github.com/sfujim/BCQ/blob/master/continuous_BCQ/BCQ.py
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from rlkit.torch.sac.policies import MakeDeterministic
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core import logger
from rlkit.core.logging import add_prefix
from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule
import torch.nn.functional as F
from rlkit.torch.networks import LinearTransform
import time
from rlkit.torch.distributions import Delta
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)

def repeat_interleave(input, repeats, dim=None):
    return torch.cat([input] * repeats, dim=dim)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.phi = phi


    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, qf1, qf2):
        super().__init__()
        self.qf1 = qf1
        self.qf2 = qf2

    def forward(self, state, action):
        q1 = self.qf1(state, action)
        q2 = self.qf2(state, action)
        return q1, q2

    def q1(self, state, action):
        return self.qf1(state, action)


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std


    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


class BCQPolicyFromQ(TorchStochasticPolicy):
    def __init__(
            self,
            qf,
            vae,
            actor,
            num_samples=100,
            **kwargs
    ):
        super().__init__()
        self.qf = qf
        self.vae = vae
        self.actor = actor
        self.num_samples = num_samples

    def forward(self, obs):
        with torch.no_grad():
            obs = obs.reshape(1, -1).repeat(100, 1).to(ptu.device)
            action = self.actor(obs, self.vae.decode(obs))
            q1 = self.qf(obs, action)
            ind = q1.argmax(0)
        return Delta(action[ind])


class BCQTrainer(TorchTrainer):
    def __init__(self,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        discount=0.99,
        tau=0.005,
        lmbda=0.75,
        phi=0.05,
        bc_num_pretrain_steps=0,
        q_num_pretrain1_steps=0,
        q_num_pretrain2_steps=0,
        num_pretrain_steps=0,
        bc_batch_size=128,
        pretraining_logging_period=1000,
        *args,
        **kwargs
    ):
        print("ignoring args", args)
        print("ignoring kwargs", kwargs)
        action_dim = env.action_space.shape[0]
        latent_dim = action_dim * 2
        state_dim = env.observation_space.shape[0]
        max_action = 1
        device = ptu.device

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain1_steps = q_num_pretrain1_steps
        self.q_num_pretrain2_steps = q_num_pretrain2_steps
        self.num_pretrain_steps = num_pretrain_steps
        self.bc_batch_size = bc_batch_size
        self.pretraining_logging_period = pretraining_logging_period
        self._num_train_steps = 0
        self._n_train_steps_total = 0

        self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(qf1, qf2)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device

        self.eval_policy = BCQPolicyFromQ(self.qf1, self.vae, self.actor)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten()

    def get_batch_from_buffer(self, replay_buffer, batch_size):
        batch = replay_buffer.random_batch(batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch

    def run_bc_batch(self, replay_buffer, policy):
        batch = self.get_batch_from_buffer(replay_buffer, self.bc_batch_size)
        o = batch["observations"]
        u = batch["actions"]
        # g = batch["resampled_goals"]
        # og = torch.cat((o, g), dim=1)
        og = o
        # pred_u, *_ = self.policy(og)
        dist = policy(og)
        pred_u, log_pi = dist.rsample_and_logprob()
        stats = dist.get_diagnostics()

        mse = (pred_u - u) ** 2
        mse_loss = mse.mean()

        policy_logpp = dist.log_prob(u, )
        logp_loss = -policy_logpp.mean()
        policy_loss = logp_loss

        return policy_loss, logp_loss, mse_loss, stats

    def pretrain_policy_with_bc(self, policy, train_buffer, test_buffer, steps, label="policy", ):
        return

    def pretrain_q_with_bc_data(self, batch_size):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain_q.csv', relative_to_snapshot_dir=True
        )

        prev_time = time.time()
        for i in range(self.num_pretrain_steps):
            self.eval_statistics = dict()
            if i % self.pretraining_logging_period == 0:
                self._need_to_update_eval_statistics=True
            train_data = self.replay_buffer.random_batch(self.bc_batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            # goals = train_data['resampled_goals']
            train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
            train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
            self.train_from_torch(train_data, pretrain=True)

            if i%self.pretraining_logging_period==0:
                self.eval_statistics["batch"] = i
                self.eval_statistics["epoch_time"] = time.time()-prev_time
                stats_with_prefix = add_prefix(self.eval_statistics, prefix="trainer/")
                logger.record_dict(stats_with_prefix)
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

    def set_algorithm_weights(
        self,
        **kwargs
    ):
        for key in kwargs:
            self.__dict__[key] = kwargs[key]

    def test_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        weights = batch.get('weights', None)
        if self.reward_transform:
            rewards = self.reward_transform(rewards)

        if self.terminal_transform:
            terminals = self.terminal_transform(terminals)

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        policy_mle = dist.mle_estimate()

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        qf1_new_actions = self.qf1(obs, new_obs_actions)
        qf2_new_actions = self.qf2(obs, new_obs_actions)
        q_new_actions = torch.min(
            qf1_new_actions,
            qf2_new_actions,
        )

        policy_loss = (log_pi - q_new_actions).mean()

        self.eval_statistics['validation/QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
        self.eval_statistics['validation/QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
        self.eval_statistics['validation/Policy Loss'] = np.mean(ptu.get_numpy(
            policy_loss
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'validation/Q1 Predictions',
            ptu.get_numpy(q1_pred),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'validation/Q2 Predictions',
            ptu.get_numpy(q2_pred),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'validation/Q Targets',
            ptu.get_numpy(q_target),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'validation/Log Pis',
            ptu.get_numpy(log_pi),
        ))
        policy_statistics = add_prefix(dist.get_diagnostics(), "validation/policy/")
        self.eval_statistics.update(policy_statistics)

    def train_from_torch(self, batch, train=True, pretrain=False,):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        weights = batch.get('weights', None)

        state, action, next_state, reward, not_done = obs, actions, next_obs, rewards, terminals
        batch_size = len(rewards)

        # Variational Auto-Encoder Training
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()


        # Critic Training
        with torch.no_grad():
            # Duplicate next state 10 times
            next_state = repeat_interleave(next_state, 10, 0)

            # Compute value of perturbed actions sampled from the VAE
            target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

            # Soft Clipped Double Q-learning
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
            # Take max over each action sampled from the VAE
            target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Pertubation Model / Action Training
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.actor(state, sampled_actions)

        # Update through DPG
        actor_loss = -self.critic.q1(state, perturbed_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        """
        Some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(current_Q1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(current_Q2),
            ))
            self.eval_statistics.update({
                'Critic Loss': ptu.get_numpy(critic_loss)
            })
            self.eval_statistics.update({
                'Actor Loss': ptu.get_numpy(actor_loss)
            })


        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.policy,
            self.actor,
            self.actor_target,
            self.critic,
            self.critic_target,
        ]
        # if self.buffer_policy:
            # nets.append(self.buffer_policy)
        return nets

    def get_snapshot(self):
        return dict(
            # policy=self.policy,
            # qf1=self.qf1,
            # qf2=self.qf2,
            # target_qf1=self.qf1,
            # target_qf2=self.qf2,
            # buffer_policy=self.buffer_policy,
        )
