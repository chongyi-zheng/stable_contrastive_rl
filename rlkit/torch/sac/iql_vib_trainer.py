from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
from rlkit.torch.networks import LinearTransform
from rlkit.util.augment_util import create_aug_stack

from rlkit.experimental.kuanfang.networks.encoding_networks import EncodingGaussianPolicy  # NOQA
from rlkit.experimental.kuanfang.networks.encoding_networks import EncodingGaussianPolicyV2  # NOQA


class IQLVIBTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            vf,
            plan_vf,
            affordance,
            obs_encoder,
            obs_dim,
            vqvae=None,

            quantile=0.5,
            target_qf1=None,
            target_qf2=None,
            buffer_policy=None,

            kld_weight=1.0,
            affordance_pred_weight=10000.,
            affordance_beta=1.0,

            discount=0.99,
            reward_scale=1.0,

            lr=3e-4,
            gradient_clipping=None,  # TODO
            optimizer_class=optim.Adam,

            # backprop_from_policy=True,
            bc=False,

            actor_update_period=1,
            critic_update_period=1,

            reward_transform_class=None,
            reward_transform_kwargs=None,
            terminal_transform_class=None,
            terminal_transform_kwargs=None,

            clip_score=None,
            soft_target_tau=1e-2,
            target_update_period=1,
            beta=1.0,
            min_value=None,
            max_value=None,

            goal_is_encoded=False,

            fraction_negative_obs=0.3,
            fraction_negative_goal=0.3,
            end_to_end=False,  # TODO
            affordance_weight=1.0,

            train_encoder=True,

            use_encoding_reward=False,
            encoding_reward_thresh=None,

            augment_params=dict(),
            augment_order=[],
            augment_probability=0.0,
            reencode_augmented_images=False,

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
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.gradient_clipping = gradient_clipping
        self.vf = vf
        self.plan_vf = plan_vf
        self.affordance = affordance
        self.obs_encoder = obs_encoder
        self.buffer_policy = buffer_policy
        self.bc = bc
        # self.backprop_from_policy = backprop_from_policy

        self.vqvae = vqvae

        self.obs_dim = obs_dim
        self.kld_weight = kld_weight

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.parameters = (
            list(self.qf1.parameters()) +
            list(self.qf2.parameters()) +
            list(self.vf.parameters()) +
            list(self.plan_vf.parameters()) +
            list(self.policy.parameters()) +
            list(self.obs_encoder.parameters()) +
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
        self.pred_loss_fn = torch.nn.SmoothL1Loss(
            reduction='none').to(ptu.device)
        self.huber_loss_fn = torch.nn.SmoothL1Loss(
            reduction='none').to(ptu.device)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self.n_train_steps_total = 0

        self.goal_is_encoded = goal_is_encoded
        self.fraction_negative_obs = fraction_negative_obs
        self.fraction_negative_goal = fraction_negative_goal

        self.need_to_update_eval_statistics = {
            'train/': True,
            'eval/': True,
        }

        self.critic_update_period = critic_update_period
        self.actor_update_period = actor_update_period

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

        self.clip_score = clip_score
        self.beta = beta
        self.quantile = quantile
        self.train_encoder = train_encoder

        self.min_value = min_value
        self.max_value = max_value

        self.end_to_end = end_to_end
        self.affordance_weight = affordance_weight

        self.use_encoding_reward = use_encoding_reward
        self.encoding_reward_thresh = encoding_reward_thresh

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

    # def _vqvae_encode(self, obs):
    #     obs = obs - 0.5
    #     obs = obs.reshape(
    #         (-1, 3, 48, 48))
    #     obs = obs.permute([0, 1, 3, 2])
    #     obs = self.vqvae.encode(obs)
    #     return obs
    #
    # def _vqvae_decode(self, obs):
    #     obs = self.vqvae.decode(obs)
    #     obs = obs + 0.5
    #     obs = torch.clamp(obs, 0, 1)
    #     obs = obs.permute([0, 1, 3, 2])
    #     return obs

    def set_augment_params(self, img):
        if torch.rand(1) < self.augment_probability:
            self.augment_stack.set_params(img)
        else:
            self.augment_stack.set_default_params(img)

    def augment(self, batch):
        augmented_batch = dict()
        for key, value in batch.items():
            augmented_batch[key] = value

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
            augmented_batch['observations'] = self.augment_stack(obs)

            augmented_batch['next_observations'] = self.augment_stack(next_obs)

            augmented_batch['contexts'] = self.augment_stack(context)

        return augmented_batch

    def _compute_affordance_loss(self, h0, h1, weights=None):

        # TODO: Maybe always fix h1 to prevent the encoding from collapsing.
        if not self.end_to_end:
            h0 = h0.detach()

        h1 = h1.detach()

        (u_mu, u_logvar), u, h1_pred = self.affordance(h1, cond=h0)

        batch_size = h0.shape[0]
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

    def _compute_plan_vf_loss(self, obs, goal, target_value, weights=None):
        num_samples = target_value.shape[0]

        obs = obs.detach()
        goal = goal.detach()

        # TODO (chongyiz): figure out what this block is doing.
        replace_obs = (
            torch.rand(num_samples, 1) < self.fraction_negative_obs
        ).to(ptu.device)
        # sampled_obs = torch.randn(obs.size()).to(ptu.device)
        sampled_obs = torch.where(
            (torch.rand(num_samples, 1) < 0.5).to(ptu.device),
            torch.randn(obs.size()).to(ptu.device),
            torch.flip(obs, [0]))
        obs = torch.where(
            replace_obs,
            sampled_obs,
            obs)

        replace_goal = (
            torch.rand(num_samples, 1) < self.fraction_negative_goal
        ).to(ptu.device)
        # sampled_goal = torch.randn(goal.size()).to(ptu.device)
        sampled_goal = torch.where(
            (torch.rand(num_samples, 1) < 0.5).to(ptu.device),
            torch.randn(goal.size()).to(ptu.device),
            torch.flip(goal, [0]))
        goal = torch.where(
            replace_goal,
            sampled_goal,
            goal)

        input_h = torch.cat([obs, goal], -1)
        pred_value = self.plan_vf(input_h)

        replace_any = (replace_obs + replace_goal) > 0  # logical_or

        # target_value = torch.where(
        #     replace_any,
        #     torch.ones_like(target_value) * self.min_value,
        #     target_value).detach()
        # assert pred_value.shape == target_value.shape
        # plan_vf_loss = self.pred_loss_fn(pred_value, target_value).mean()

        if self.train_encoder:
            pos_plan_vf_loss = self.pred_loss_fn(pred_value, target_value)
            thresh = -50
            # neg_plan_vf_loss = torch.maximum(pred_value - thresh, 0)
            neg_plan_vf_loss = torch.clamp(pred_value - thresh, min=0.)
            plan_vf_loss = torch.where(
                replace_any,
                neg_plan_vf_loss,
                pos_plan_vf_loss)
            plan_vf_loss = plan_vf_loss.mean()

        extra = {
            'vf_pred': pred_value,
        }

        return plan_vf_loss, extra

    def train_from_torch(self, batch, train=True):

        if train:
            batch = self.augment(batch)
            for net in self.networks:
                net.train(True)
        else:
            for net in self.networks:
                net.train(False)

        reward = batch['rewards']
        terminal = batch['terminals']
        action = batch['actions']

        obs = batch['observations']
        next_obs = batch['next_observations']
        goal = batch['contexts']

        """
        Obs Encoder
        """
        if self.goal_is_encoded:
            obs_feat = obs
            obs_feat_mu = obs
            obs_feat_logvar = torch.zeros_like(obs)

            next_obs_feat = next_obs
            next_obs_feat_mu = next_obs
            next_obs_feat_logvar = torch.zeros_like(next_obs)

            goal_feat = goal
            goal_feat_mu = goal
            goal_feat_logvar = torch.zeros_like(goal)

        else:
            obs_feat, (obs_feat_mu, obs_feat_logvar) = self.obs_encoder(
                obs, training=True)
            next_obs_feat, (next_obs_feat_mu, next_obs_feat_logvar) = self.obs_encoder(  # NOQA
                next_obs, training=True)
            goal_feat, (goal_feat_mu, goal_feat_logvar) = self.obs_encoder(
                goal, training=True)

        if not self.train_encoder:
            obs_feat = obs_feat.detach()
            obs_feat_mu = obs_feat_mu.detach()
            obs_feat_logvar = obs_feat_logvar.detach()
            next_obs_feat = next_obs_feat.detach()
            next_obs_feat_mu = next_obs_feat_mu.detach()
            next_obs_feat_logvar = next_obs_feat_logvar.detach()
            goal_feat = goal_feat.detach()
            goal_feat_mu = goal_feat_mu.detach()
            goal_feat_logvar = goal_feat_logvar.detach()

            obs_feat = obs_feat_mu
            next_obs_feat = next_obs_feat_mu
            goal_feat = goal_feat_mu

        # if self.kld_weight == 0:
        #     obs_feat = obs_feat_mu
        #     next_obs_feat = next_obs_feat_mu
        #     goal_feat = goal_feat_mu

        # if self.reward_transform:
        #     reward = self.reward_transform(reward)
        # if self.terminal_transform:
        #     terminal = self.terminal_transform(terminal)

        # TODO: obs_encoder reward.
        if self.use_encoding_reward:
            similarity = torch.nn.functional.cosine_similarity(
                next_obs_feat_mu, goal_feat_mu).detach()
            success = similarity >= self.encoding_reward_thresh
            reward = success.to(torch.float) - 1.0
            terminal = success.to(torch.float)

            terminal = torch.zeros_like(terminal)  # TODO

            reward = reward.view(-1, 1)
            terminal = terminal.view(-1, 1)

        h = torch.cat([obs_feat, goal_feat], -1)

        target_h = torch.cat([obs_feat_mu, goal_feat_mu], -1)
        target_next_h = torch.cat([next_obs_feat_mu, goal_feat_mu], -1)

        """
        QF Loss
        """
        q1_pred = self.qf1(h, action)
        q2_pred = self.qf2(h, action)
        target_next_vf_pred = self.vf(target_next_h).detach()
        if self.min_value is not None or self.max_value is not None:
            target_next_vf_pred = torch.clamp(
                target_next_vf_pred,
                min=self.min_value,
                max=self.max_value,
            )

        assert reward.shape == terminal.shape
        assert reward.shape == target_next_vf_pred.shape
        q_target = self.reward_scale * reward + \
            (1. - terminal) * self.discount * target_next_vf_pred
        q_target = q_target.detach()
        assert q1_pred.shape == q_target.shape
        assert q2_pred.shape == q_target.shape
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """
        VF Loss
        """
        q_pred = torch.min(
            self.target_qf1(target_h, action),
            self.target_qf2(target_h, action),
        ).detach()
        if self.min_value is not None or self.max_value is not None:
            q_pred = torch.clamp(
                q_pred,
                min=self.min_value,
                max=self.max_value,
            )

        vf_pred = self.vf(h)
        assert vf_pred.shape == q_pred.shape
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + \
            vf_sign * (1 - self.quantile)
        assert vf_weight.shape == vf_err.shape
        vf_loss = (vf_weight * (vf_err ** 2)).mean()

        critic_loss = qf1_loss + qf2_loss + vf_loss

        if self.train_encoder:
            plan_vf_loss, plan_vf_extra = self._compute_plan_vf_loss(
                obs_feat, goal_feat, vf_pred)
            critic_loss += plan_vf_loss

        """
        Affordance Loss
        """
        if self.train_encoder:
            # TODO: Maybe start training the affordance after 2 epochs or so.
            affordance_loss, affordance_extra = self._compute_affordance_loss(
                obs_feat_mu, goal_feat_mu)
            critic_loss += affordance_loss * self.affordance_weight

        """
        Policy Loss
        """
        # TODO: Make sure the policy calls the same state encoder inside.
        if isinstance(self.policy, EncodingGaussianPolicy):
            # if not self.backprop_from_policy:
            #     h = h.detach()

            # TODO
            # noise = goal_feat.data.new(goal_feat.size()).normal_() * 0.1
            # h = torch.cat([obs_feat, goal_feat + noise], -1)

            dist = self.policy(h, encoded_input=True)
        elif isinstance(self.policy, EncodingGaussianPolicyV2):
            _h = torch.cat([obs, goal_feat], -1)
            dist = self.policy(_h, encoded_input=True)
        else:
            raise NotImplementedError
            # dist = self.policy(obs_and_goal)

        actor_logpp = dist.log_prob(action)

        vf_baseline = self.vf(target_h).detach()  # TODO: Debugging.
        assert q_pred.shape == vf_baseline.shape
        adv = q_pred - vf_baseline
        exp_adv = torch.exp(adv / self.beta)
        if self.clip_score is not None:
            exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        adv_weight = ptu.from_numpy(np.ones(exp_adv[:, 0].shape)).detach(
        ) if self.bc else exp_adv[:, 0].detach()
        assert actor_logpp.shape == adv_weight.shape
        actor_loss = (-actor_logpp * adv_weight).mean()

        loss = critic_loss + actor_loss

        """
        VIB
        """
        if self.train_encoder:
            kld = - 0.5 * torch.sum(
                1 + obs_feat_logvar - obs_feat_mu.pow(2)
                - obs_feat_logvar.exp(),
                dim=-1)
            kld = torch.mean(kld)
            loss += kld * self.kld_weight

        """
        MSE Loss
        """
        with torch.no_grad():
            mse_loss = (dist.mean - action) ** 2

        if train:
            """
            Optimization.
            """
            if self.n_train_steps_total % self.critic_update_period == 0:
                self.optimizer.zero_grad()
                loss.backward()

                if (self.gradient_clipping is not None and
                        self.gradient_clipping > 0):
                    torch.nn.utils.clip_grad_norm(
                        self.parameters, self.gradient_clipping)

                self.optimizer.step()

            """
            Soft Updates
            """
            if self.n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.qf1, self.target_qf1, self.soft_target_tau
                )
                ptu.soft_update_from_to(
                    self.qf2, self.target_qf2, self.soft_target_tau
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
            # self.eval_statistics[prefix + 'Obs Min'] = np.mean(
            #     ptu.get_numpy(obs.min()))
            # self.eval_statistics[prefix + 'Obs Max'] = np.mean(
            #     ptu.get_numpy(obs.max()))
            # self.eval_statistics[prefix + 'Next Obs Min'] = np.mean(
            #     ptu.get_numpy(next_obs.min()))
            # self.eval_statistics[prefix + 'Next Obs Max'] = np.mean(
            #     ptu.get_numpy(next_obs.max()))
            # self.eval_statistics[prefix + 'Goal Min'] = np.mean(
            #     ptu.get_numpy(goal.min()))
            # self.eval_statistics[prefix + 'Goal Max'] = np.mean(
            #     ptu.get_numpy(goal.max()))

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

            self.eval_statistics[prefix + 'QF1 Loss'] = np.mean(
                ptu.get_numpy(qf1_loss))
            self.eval_statistics[prefix + 'QF2 Loss'] = np.mean(
                ptu.get_numpy(qf2_loss))
            self.eval_statistics[prefix + 'Policy Loss'] = np.mean(
                ptu.get_numpy(actor_loss))
            self.eval_statistics[prefix + 'MSE'] = np.mean(
                ptu.get_numpy(mse_loss))

            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Policy Mean',
                ptu.get_numpy(dist.mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Policy STD',
                ptu.get_numpy(dist.stddev),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Q Targets',
                ptu.get_numpy(q_target),
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
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Advantage Weights',
                ptu.get_numpy(adv_weight),))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Advantage Score',
                ptu.get_numpy(adv),))

            self.eval_statistics[prefix + 'VF Loss'] = np.mean(
                ptu.get_numpy(vf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'V1 Predictions',
                ptu.get_numpy(vf_pred),))

            if self.train_encoder:
                self.eval_statistics[prefix + 'Plan VF Loss'] = np.mean(
                    ptu.get_numpy(plan_vf_loss))
                self.eval_statistics.update(create_stats_ordered_dict(
                    prefix + 'Plan V1 Predictions',
                    ptu.get_numpy(plan_vf_extra['vf_pred']),))

            self.eval_statistics[prefix + 'beta'] = self.beta
            self.eval_statistics[prefix + 'quantile'] = self.quantile

            if self.train_encoder:
                self.eval_statistics[prefix + 'KLD'] = np.mean(
                    ptu.get_numpy(kld))

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
                    prefix + 'Affordance Encoding Mean',
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

    @ property
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
