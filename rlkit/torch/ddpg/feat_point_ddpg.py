import numpy as np
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
import torch
from torch.autograd import Variable
from torch import nn as nn
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.ddpg.ddpg import DDPG
from PIL import Image



class FeatPointDDPG(DDPG):

    def __init__(
            self,
            ae,
            history_length,

            extra_fc_size=0,
            imsize=64,
            downsampled_size=32,
            ae_learning_rate=1e-3,
            optimizer_class=optim.Adam,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.ae = ae
        self.ae_criterion = nn.MSELoss()
        self.ae_optimizer = optimizer_class(
            self.ae.parameters(),
            lr=ae_learning_rate
        )
        self.imsize = imsize
        self.downsampled_size = downsampled_size
        self.history_length = history_length
        self.extra_fc_size = extra_fc_size
        self.input_length = self.imsize**2 * self.history_length + self.extra_fc_size

    def train_ae(self, batch):
        obs = batch['observations']
        downsampled = batch['downsampled']
        # get the first image of history
        downsampled = downsampled.narrow(start=0,
                                         length=self.downsampled_size**2,
                                         dimension=1)
        obs = obs.narrow(start=0,
                         length=self.imsize**2,
                         dimension=1)

        reconstructed = self.ae(obs)
        self.ae_optimizer.zero_grad()
        loss = self.ae_criterion(reconstructed, downsampled)
        loss.backward()
        return loss

    def get_latent_obs(self, batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        image_obs, fc_obs = self.env.split_obs(obs)
        next_image_obs, next_fc_obs = self.env.split_obs(next_obs)

        latent_obs = self.ae.history_encoder(image_obs, self.history_length)
        next_latent_obs = self.ae.history_encoder(next_image_obs, self.history_length)

        if fc_obs is not None:
            latent_obs = torch.cat((latent_obs, fc_obs), dim=1)
            next_latent_obs = torch.cat((next_latent_obs, next_fc_obs), dim=1)

        return latent_obs, next_latent_obs

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        """
        autoencoder training
        """
        ae_loss = self.train_ae(batch)
        obs, next_obs = self.get_latent_obs(batch)
        # Convert observation to latent
        obs = obs.detach()
        next_obs = next_obs.detach()
        """
        Policy operations.
        """
        if self.policy_pre_activation_weight > 0:
            policy_actions, pre_tanh_value = self.policy(
                obs, return_preactivations=True,
            )
            pre_activation_policy_loss = (
                (pre_tanh_value ** 2).sum(dim=1).mean()
            )
            q_output = self.qf(obs, policy_actions)
            raw_policy_loss = - q_output.mean()
            policy_loss = (
                    raw_policy_loss +
                    pre_activation_policy_loss * self.policy_pre_activation_weight
            )
        else:
            policy_actions = self.policy(obs)
            q_output = self.qf(obs, policy_actions)
            raw_policy_loss = policy_loss = - q_output.mean()

        """
        Critic operations.
        """

        next_actions = self.target_policy(next_obs)
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
        )
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
        q_target = torch.clamp(q_target, self.min_q_value, self.max_q_value)
        # Hack for ICLR rebuttal
        if hasattr(self, 'reward_type') and self.reward_type == 'indicator':
            q_target = torch.clamp(q_target,
                                   -self.reward_scale / (1 - self.discount), 0)
        q_pred = self.qf(obs, actions)
        bellman_errors = (q_pred - q_target) ** 2
        raw_qf_loss = self.qf_criterion(q_pred, q_target)

        if self.residual_gradient_weight > 0:
            residual_next_actions = self.policy(next_obs)
            # speed up computation by not backpropping these gradients
            residual_next_actions.detach()
            residual_target_q_values = self.qf(
                next_obs,
                residual_next_actions,
            )
            residual_q_target = (
                self.reward_scale * rewards
                + (1. - terminals) * self.discount * residual_target_q_values
            )
            residual_bellman_errors = (q_pred - residual_q_target) ** 2
            # noinspection PyUnresolvedReferences
            residual_qf_loss = residual_bellman_errors.mean()
            raw_qf_loss = (
                    self.residual_gradient_weight * residual_qf_loss
                    + (1 - self.residual_gradient_weight) * raw_qf_loss
            )

        if self.qf_weight_decay > 0:
            reg_loss = self.qf_weight_decay * sum(
                torch.sum(param ** 2)
                for param in self.qf.regularizable_parameters()
            )
            qf_loss = raw_qf_loss + reg_loss
        else:
            qf_loss = raw_qf_loss

        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self._update_target_networks()

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['Autoencoder Reconstruction Loss'] = np.mean(ptu.get_numpy(ae_loss))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['Raw Policy Loss'] = np.mean(ptu.get_numpy(
                raw_policy_loss
            ))
            self.eval_statistics['Preactivation Policy Loss'] = (
                    self.eval_statistics['Policy Loss'] -
                    self.eval_statistics['Raw Policy Loss']
            )
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors',
                ptu.get_numpy(bellman_errors),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))
