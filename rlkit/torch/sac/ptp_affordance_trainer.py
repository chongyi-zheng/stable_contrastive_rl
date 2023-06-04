from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
# from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
# from rlkit.core.logging import add_prefix
from rlkit.torch.networks import LinearTransform
from rlkit.util.augment_util import create_aug_stack

from rlkit.experimental.kuanfang.networks.encoding_networks import EncodingGaussianPolicy  # NOQA
from rlkit.experimental.kuanfang.networks.encoding_networks import EncodingGaussianPolicyV2  # NOQA


class PTPAffordanceTrainer(TorchTrainer):
    def __init__(
            self,
            env,

            affordance,
            obs_encoder,
            obs_dim,
            vqvae=None,

            policy=None,
            qf1=None,
            qf2=None,
            vf=None,
            plan_vf=None,
            target_qf1=None,
            target_qf2=None,

            kld_weight=1.0,
            affordance_pred_weight=10000.,
            affordance_beta=1.0,

            lr=3e-4,
            gradient_clipping=None,  # TODO
            optimizer_class=optim.Adam,

            reward_transform_class=None,
            reward_transform_kwargs=None,
            terminal_transform_class=None,
            terminal_transform_kwargs=None,

            fraction_generated_goals=0.0,
            end_to_end=False,  # TODO
            affordance_weight=1.0,

            augment_params=dict(),
            augment_order=[],
            augment_probability=0.0,

            use_estimated_logvar=False,
            use_sampled_encodings=False,
            noise_level=None,

            goal_is_encoded=False,
            use_obs_encoder=True,

            * args,
            **kwargs
    ):
        super().__init__()
        self.env = env

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vf = vf
        self.plan_vf = plan_vf

        self.affordance = affordance
        self.obs_encoder = obs_encoder

        self.vqvae = vqvae

        self.gradient_clipping = gradient_clipping

        self.obs_dim = obs_dim
        self.kld_weight = kld_weight

        self.parameters = (
            list(self.affordance.parameters())
        )
        # Remove duplicated parameters.
        self.parameters = list(set(self.parameters))

        self.optimizer = optimizer_class(
            self.parameters,
            lr=lr,
        )

        self.affordance_pred_weight = affordance_pred_weight
        self.affordance_beta = affordance_beta

        self.pred_loss_fn = torch.nn.MSELoss(
            reduction='none').to(ptu.device)

        self.use_estimated_logvar = use_estimated_logvar

        self.eval_statistics = OrderedDict()
        self.n_train_steps_total = 0

        self.fraction_generated_goals = fraction_generated_goals

        self.need_to_update_eval_statistics = {
            'train/': True,
            'eval/': True,
        }

        self.reward_transform_class = reward_transform_class or LinearTransform
        self.reward_transform_kwargs = reward_transform_kwargs or dict(
            m=1, b=0)
        self.terminal_transform_class = (
            terminal_transform_class or LinearTransform)
        self.terminal_transform_kwargs = terminal_transform_kwargs or dict(
            m=1, b=0)
        self.reward_transform = self.reward_transform_class(
            **self.reward_transform_kwargs)
        self.terminal_transform = self.terminal_transform_class(
            **self.terminal_transform_kwargs)

        self.end_to_end = end_to_end
        self.affordance_weight = affordance_weight

        # Image augmentation.
        self.augment_probability = augment_probability
        if augment_probability > 0:
            width = self.obs_encoder.input_width
            height = self.obs_encoder.input_height
            self.augment_stack = create_aug_stack(
                augment_order, augment_params, size=(width, height)
            )
        else:
            self.augment_stack = None

        self.use_sampled_encodings = use_sampled_encodings
        self.noise_level = noise_level

        self.goal_is_encoded = goal_is_encoded
        self.use_obs_encoder = use_obs_encoder

    def _vqvae_encode(self, obs):
        obs = obs - 0.5
        obs = obs.reshape(
            (-1, 3, 48, 48))
        obs = obs.permute([0, 1, 3, 2])
        obs = self.vqvae.encode(obs)
        return obs

    def _vqvae_decode(self, obs):
        obs = self.vqvae.decode(obs)
        obs = obs + 0.5
        obs = torch.clamp(obs, 0, 1)
        obs = obs.permute([0, 1, 3, 2])
        return obs

    def set_augment_params(self, img):
        if torch.rand(1) < self.augment_probability:
            self.augment_stack.set_params(img)
        else:
            self.augment_stack.set_default_params(img)

    def augment(self, batch):
        if (self.augment_probability > 0 and
                batch['observations'].shape[0] > 0):
            width = self.obs_encoder.input_width
            height = self.obs_encoder.input_height

            obs = batch['observations'].reshape(
                -1, 3, width, height)
            next_obs = batch['next_observations'].reshape(
                -1, 3, width, height)
            context = batch['contexts'].reshape(
                -1, 3, width, height)

            self.set_augment_params(obs)
            batch['observations'] = self.augment_stack(obs)
            batch['next_observations'] = self.augment_stack(next_obs)
            batch['contexts'] = self.augment_stack(context)

    def _compute_affordance_loss(
            self, h0, h1, h1_mu=None, h1_logvar=None, weights=None):

        # TODO: Maybe always fix h1 to prevent the encoding from collapsing.
        if not self.end_to_end:
            h0 = h0.detach()

        h1 = h1.detach()

        if self.noise_level is None:
            input_h0 = h0
            input_h1 = h1
        else:
            input_h0 = h0 + self.noise_level * torch.randn(
                h0.size()).to(ptu.device)
            input_h1 = h1 + self.noise_level * torch.randn(
                h0.size()).to(ptu.device)

        (u_mu, u_logvar), u, h1_pred = self.affordance(input_h1, cond=input_h0)

        batch_size = h0.shape[0]

        if self.use_estimated_logvar:
            loss_pred = (
                self.pred_loss_fn(
                    h1_pred.view(batch_size, -1),
                    h1_mu.view(batch_size, -1))
                / torch.exp(h1_logvar)
            ).mean(-1)

            loss_pred = loss_pred / 500.

        else:
            loss_pred = self.pred_loss_fn(
                h1_pred.view(batch_size, -1),
                h1.view(batch_size, -1)).mean(-1)

        kld = - 0.5 * torch.sum(
            1 + u_logvar - u_mu.pow(2) - u_logvar.exp(), dim=-1)

        loss = (
            self.affordance_pred_weight * loss_pred +
            self.affordance_beta * kld
        )

        if weights is not None:
            loss = torch.mean(loss * weights) / (torch.mean(weights) + 1e-8)
            kld = torch.mean(kld * weights) / (torch.mean(weights) + 1e-8)
            loss_pred = torch.mean(
                loss_pred * weights) / (torch.mean(weights) + 1e-8)
        else:
            loss = loss.mean()
            kld = kld.mean()
            loss_pred = loss_pred.mean()

        extra = {
            'kld': kld,
            'loss_pred': loss_pred,

            'h0': h0,
            'h1': h1,
            'h1_pred': h1_pred,

            'u': u,
            'u_mu': u_mu,
            'u_logvar': u_logvar,

        }

        return loss, extra

    def train_from_torch(self, batch, train=True):

        if train:
            self.augment(batch)
            for net in self.networks:
                net.train(True)
        else:
            for net in self.networks:
                net.train(False)

        obs = batch['observations']
        next_obs = batch['next_observations']
        goal = batch['contexts']

        """
        Obs Encoder
        """
        if self.goal_is_encoded or not self.use_obs_encoder:
            obs_feat = obs
            obs_feat_mu = obs
            obs_feat_logvar = torch.zeros_like(obs)

            next_obs_feat = next_obs
            next_obs_feat_mu = next_obs
            next_obs_feat_logvar = torch.zeros_like(next_obs)

            goal_feat = goal
            goal_feat_mu = goal
            goal_feat_logvar = torch.zeros_like(goal)

            if not self.use_obs_encoder:
                # TODO: Get the shape from vqvae.
                obs_feat = obs_feat.view(-1, 5, 12, 12)
                next_obs_feat = next_obs_feat.view(-1, 5, 12, 12)
                goal_feat = goal_feat.view(-1, 5, 12, 12)

                obs_feat_mu = obs_feat_mu.view(-1, 5, 12, 12)
                next_obs_feat_mu = next_obs_feat_mu.view(-1, 5, 12, 12)
                goal_feat_mu = goal_feat_mu.view(-1, 5, 12, 12)

                obs_feat_logvar = obs_feat_logvar.view(-1, 5, 12, 12)
                next_obs_feat_logvar = next_obs_feat_logvar.view(-1, 5, 12, 12)
                goal_feat_logvar = goal_feat_logvar.view(-1, 5, 12, 12)

        else:
            obs_feat, (obs_feat_mu, obs_feat_logvar) = self.obs_encoder(
                obs, training=True)
            next_obs_feat, (next_obs_feat_mu, next_obs_feat_logvar) = self.obs_encoder(  # NOQA
                next_obs, training=True)
            goal_feat, (goal_feat_mu, goal_feat_logvar) = self.obs_encoder(
                goal, training=True)

        # Fix the encoder.
        obs_feat = obs_feat.detach()
        goal_feat = goal_feat.detach()
        next_obs_feat = next_obs_feat.detach()
        obs_feat_mu = obs_feat_mu.detach()
        obs_feat_logvar = obs_feat_logvar.detach()
        next_obs_feat_mu = next_obs_feat_mu.detach()
        next_obs_feat_logvar = next_obs_feat_logvar.detach()
        goal_feat_mu = goal_feat_mu.detach()
        goal_feat_logvar = goal_feat_logvar.detach()

        """
        Affordance Loss
        """
        if self.use_sampled_encodings:
            obs_feat_data = obs_feat
            goal_feat_data = goal_feat
        else:
            obs_feat_data = obs_feat_mu
            goal_feat_data = goal_feat_mu

        affordance_loss, affordance_extra = self._compute_affordance_loss(
            h0=obs_feat_data,
            h1=goal_feat_data,
            h1_mu=goal_feat_mu,
            h1_logvar=goal_feat_logvar,
        )

        loss = affordance_loss * self.affordance_weight

        if train:
            """
            Optimization.
            """
            self.optimizer.zero_grad()
            loss.backward()

            if (self.gradient_clipping is not None and
                    self.gradient_clipping > 0):
                torch.nn.utils.clip_grad_norm(
                    self.parameters, self.gradient_clipping)

            self.optimizer.step()

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
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'obs_feat',
                ptu.get_numpy(obs_feat),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'obs_feat_mu',
                ptu.get_numpy(obs_feat_mu),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'obs_feat_logvar',
                ptu.get_numpy(obs_feat_logvar),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'next_obs_feat',
                ptu.get_numpy(next_obs_feat),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'next_obs_feat_mu',
                ptu.get_numpy(next_obs_feat_mu),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'next_obs_feat_logvar',
                ptu.get_numpy(next_obs_feat_logvar),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'goal_feat',
                ptu.get_numpy(goal_feat),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'goal_feat_mu',
                ptu.get_numpy(goal_feat_mu),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'goal_feat_logvar',
                ptu.get_numpy(goal_feat_logvar),
            ))

            # Affordance
            self.eval_statistics[prefix + 'Affordance Loss'] = np.mean(
                ptu.get_numpy(affordance_loss))
            self.eval_statistics[prefix + 'Affordance KLD'] = np.mean(
                ptu.get_numpy(affordance_extra['kld']))
            self.eval_statistics[prefix + 'Affordance Pred Loss'] = np.mean(  # NOQA
                ptu.get_numpy(affordance_extra['loss_pred']))

            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Affordance Encoding',
                ptu.get_numpy(affordance_extra['u']),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Affordance Encoding Mu',
                ptu.get_numpy(affordance_extra['u_mu']),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Affordance Encoding LogVar',
                ptu.get_numpy(affordance_extra['u_logvar']),
            ))

        if train:
            self.n_train_steps_total += 1

        self.need_to_update_eval_statistics[prefix] = False

        # TODO: Debugging
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
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.vf,
            self.plan_vf,
            self.obs_encoder,
            self.affordance,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            vf=self.vf,
            plan_vf=self.plan_vf,
            obs_encoder=self.obs_encoder,
            affordance=self.affordance,
        )
