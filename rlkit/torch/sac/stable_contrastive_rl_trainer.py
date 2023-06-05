"""
Reference: https://github.com/google-research/google-research/tree/master/contrastive_rl
"""

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
from rlkit.torch.networks import LinearTransform
import rlkit.torch.transforms as transforms
from rlkit.util.augment_util import create_aug_stack

from rlkit.experimental.kuanfang.networks.encoding_networks import EncodingGaussianPolicy  # NOQA
from rlkit.experimental.kuanfang.networks.encoding_networks import EncodingGaussianPolicyV2  # NOQA

from rlkit.experimental.chongyiz.utils.data_augmentation import AUG_TO_FUNC
from rlkit.experimental.chongyiz.utils.learning_rate_scheduler import LRWarmUp


class StableContrastiveRLTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            # behavioral_cloning_policy,
            qf,
            # qf1,
            # qf2,
            # vf,
            # target_vf,
            # plan_vf,
            # affordance,
            # obs_encoder,
            # obs_dim,
            # vqvae=None,

            # quantile=0.5,
            # target_qf1=None,
            # target_qf2=None,
            target_qf=None,
            # buffer_policy=None,

            # kld_weight=1.0,
            # affordance_pred_weight=10000.,
            # affordance_beta=1.0,

            discount=0.99,
            reward_scale=1.0,

            lr=3e-4,
            gradient_clipping=None,  # TODO
            optimizer_class=optim.Adam,
            # critic_lr_warmup=False,

            # backprop_from_policy=True,
            # bc=False,

            # actor_update_period=1,  # (chongyiz): Is this flag used?
            # critic_update_period=1,
            update_period=1,

            # reward_transform_class=None,
            # reward_transform_kwargs=None,
            # terminal_transform_class=None,
            # terminal_transform_kwargs=None,

            # clip_score=None,
            soft_target_tau=1e-2,
            target_update_period=1,
            # beta=1.0,
            # min_value=None,
            # max_value=None,

            # Contrastive RL arguments
            use_td=False,
            vf_ratio_loss=False,
            use_b_squared_td=False,
            use_vf_w=False,
            self_normalized_vf_w=False,
            multiply_batch_size_scale=True,
            add_mc_to_td=False,
            # use_gcbc=False,
            entropy_coefficient=None,
            target_entropy=None,
            # random_goals=0.0,
            bc_coef=0.05,
            # bc_augmentation=False,

            adv_weighted_loss=False,
            actor_q_loss=True,
            bc_train_val_split=False,

            # alignment_alpha=2,
            # alignment_coef=0.0,

            # goal_is_encoded=False,

            # fraction_negative_obs=0.3,
            # fraction_negative_goal=0.3,
            # end_to_end=False,  # TODO
            # affordance_weight=1.0,

            # train_encoder=True,

            # use_encoding_reward=False,
            # encoding_reward_thresh=None,

            # augment_type='default',
            # augment_params=dict(),
            # augment_order=[],
            augment_order=[],
            augment_probability=0.0,
            # reencode_augmented_images=False,
            # same_augment_in_a_batch=True,

            # vip_gcbc=False,
            # r3m_gcbc=False,

            * args,
            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        # (chongyiz): behavioral_cloning_policy is not used now
        # self.behavioral_cloning_policy = behavioral_cloning_policy
        # self.qf1 = qf1
        # self.qf2 = qf2
        # self.target_qf1 = target_qf1
        # self.target_qf2 = target_qf2
        self.qf = qf
        self.target_qf = target_qf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.gradient_clipping = gradient_clipping
        # self.vf = vf
        # self.target_vf = target_vf
        # self.plan_vf = plan_vf
        # self.affordance = affordance
        # self.obs_encoder = obs_encoder
        # self.buffer_policy = buffer_policy
        # self.bc = bc
        # self.backprop_from_policy = backprop_from_policy

        # self.vqvae = vqvae

        # self.obs_dim = obs_dim
        # self.kld_weight = kld_weight

        self.qf_criterion = nn.BCEWithLogitsLoss(reduction='none')
        # self.vf_criterion = nn.MSELoss()
        # self.vf_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # Contrastive RL attributes
        # self.critic_lr_warmup = critic_lr_warmup
        self.use_td = use_td
        # self.vf_ratio_loss = vf_ratio_loss
        # self.use_b_squared_td = use_b_squared_td
        # self.use_vf_w = use_vf_w
        # self.self_normalized_vf_w = self_normalized_vf_w
        # self.multiply_batch_size_scale = multiply_batch_size_scale
        # self.add_mc_to_td = add_mc_to_td
        # self.use_gcbc = use_gcbc
        self.entropy_coefficient = entropy_coefficient
        self.adaptive_entropy_coefficient = entropy_coefficient is None
        self.target_entropy = target_entropy
        # (chongyiz): For the actor update, only use future states as goals in offline setting,
        # i.e. random_goals = 0.0.
        # self.random_goals = random_goals
        self.bc_coef = bc_coef
        # self.bc_augmentation = bc_augmentation

        # self.adv_weighted_loss = adv_weighted_loss
        self.actor_q_loss = actor_q_loss
        # self.bc_train_val_split = bc_train_val_split
        # self.alignment_alpha = alignment_alpha
        # self.alignment_coef = alignment_coef

        # self.vip_gcbc = vip_gcbc
        # self.r3m_gcbc = r3m_gcbc

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

        # self.parameters = (
        #     # list(self.qf1.parameters()) +
        #     # list(self.qf2.parameters()) +
        #     list(self.qf.parameters()) +
        #     # list(self.vf.parameters()) +
        #     # list(self.plan_vf.parameters()) +
        #     list(self.policy.parameters()) +
        #     # list(self.obs_encoder.parameters()) +
        #     [self.log_alpha]
        #     # list(self.affordance.parameters())
        # )
        # # # Remove duplicated parameters.
        # # self.parameters = list(set(self.parameters))
        # #
        # self.optimizer = optimizer_class(
        #     self.parameters,
        #     lr=lr,
        # )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=lr,
        )
        # self.behavioral_cloning_policy_optimizer = optimizer_class(
        #     self.behavioral_cloning_policy.parameters(),
        #     lr=lr,
        # )
        # if self.critic_lr_warmup:
        #     optimizer = optimizer_class(
        #         self.qf.parameters(),
        #         lr=lr,
        #     )
        #     self.qf_optimizer = LRWarmUp(lr, 100_000, optimizer)
        # else:
        #     self.qf_optimizer = optimizer_class(
        #         self.qf.parameters(),
        #         lr=lr,
        #     )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=lr,
        )

        # self.vf_optimizer = optimizer_class(
        #     self.vf.parameters(),
        #     lr=lr,
        # )
        # self.obs_encoder_optimizer = optimizer_class(
        #     self.obs_encoder.parameters(),
        #     lr=lr
        # )

        # self.affordance_pred_weight = affordance_pred_weight
        # self.affordance_beta = affordance_beta
        # self.pred_loss_fn = torch.nn.SmoothL1Loss(
        #     reduction='none').to(ptu.device)
        # self.huber_loss_fn = torch.nn.SmoothL1Loss(
        #     reduction='none').to(ptu.device)

        # initialize target_qf/target_vf weights as the same ones in qf/vf
        ptu.copy_model_params_from_to(self.qf, self.target_qf)
        # ptu.copy_model_params_from_to(self.vf, self.target_vf)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self.n_train_steps_total = 0

        # self.goal_is_encoded = goal_is_encoded
        # self.fraction_negative_obs = fraction_negative_obs
        # self.fraction_negative_goal = fraction_negative_goal

        self.need_to_update_eval_statistics = {
            'train/': True,
            'eval/': True,
        }

        # DELETEME (chongyiz)
        # self.critic_update_period = critic_update_period
        # self.actor_update_period = actor_update_period
        self.update_period = update_period

        # DELETEME (chongyiz)
        # self.reward_transform_class = reward_transform_class or LinearTransform
        # self.reward_transform_kwargs = reward_transform_kwargs or dict(
        #     m=1, b=0)
        # self.terminal_transform_class = (
        #     terminal_transform_class or LinearTransform)
        # self.terminal_transform_kwargs = terminal_transform_kwargs or dict(
        #     m=1, b=0)
        # self.reward_transform = self.reward_transform_class(
        #     **self.reward_transform_kwargs)
        # self.terminal_transform = self.terminal_transform_class(
        #     **self.terminal_transform_kwargs)

        # self.clip_score = clip_score
        # self.beta = beta
        # self.quantile = quantile
        # self.train_encoder = train_encoder

        # self.min_value = min_value
        # self.max_value = max_value

        # self.end_to_end = end_to_end
        # self.affordance_weight = affordance_weight

        # self.use_encoding_reward = use_encoding_reward
        # self.encoding_reward_thresh = encoding_reward_thresh

        # Image augmentation.
        self.augment_probability = augment_probability
        # self.augment_type = augment_type
        self.augment_stack = None
        self.augment_funcs = {}
        if augment_probability > 0:
            # assert vqvae is not None

            # self.augment_stack = create_aug_stack(
            #     augment_order, augment_params,
            #     size=(self.policy.input_width, self.policy.input_height)
            # )
            # self.same_augment_in_a_batch = same_augment_in_a_batch

            self.augment_funcs = {}
            for aug_name in augment_order:
                assert aug_name in AUG_TO_FUNC, 'invalid data aug string'
                self.augment_funcs[aug_name] = AUG_TO_FUNC[aug_name]

    # def set_augment_params(self, img):
    #     if torch.rand(1) < self.augment_probability:
    #         self.augment_stack.set_params(img)
    #     else:
    #         self.augment_stack.set_default_params(img)

    def augment(self, batch, train=True):
        augmented_batch = dict()
        for key, value in batch.items():
            augmented_batch[key] = value

        if (train and self.augment_probability > 0 and
                torch.rand(1) < self.augment_probability and
                batch['observations'].shape[0] > 0):
            width = self.policy.input_width
            height = self.policy.input_height
            channel = self.policy.input_channels // 2

            img_obs = batch['observations'].reshape(
                -1, channel, width, height)
            next_img_obs = batch['next_observations'].reshape(
                -1, channel, width, height)
            img_goal = batch['contexts'].reshape(
                -1, channel, width, height)

            # self.set_augment_params(obs)

            # if self.augment_type == 'default':
            #     # Randomized transformations will apply the same transformation to all the images of a given batch
            #     # Reference: https://pytorch.org/vision/stable/transforms.html
            #     if self.same_augment_in_a_batch:
            #         self.augment_stack.set_params(img_obs)
            #         aug_img_obs = self.augment_stack(img_obs)
            #         aug_next_img_obs = self.augment_stack(next_img_obs)
            #         aug_img_goal = self.augment_stack(img_goal)
            #     else:
            #         # Also see: https://discuss.pytorch.org/t/applying-different-data-augmentation-per-image-in-a-mini-batch/139136
            #         imgs = torch.stack([img_obs, next_img_obs, img_goal], dim=1)
            #
            #         def set_augment_params_and_augment(img):
            #             self.augment_stack.set_params(img)
            #             return self.augment_stack(img)
            #
            #         aug_imgs = transforms.Lambda(lambda xs: torch.stack(
            #             [set_augment_params_and_augment(x) for x in xs]))(imgs)
            #         aug_img_obs = aug_imgs[:, 0]
            #         aug_next_img_obs = aug_imgs[:, 1]
            #         aug_img_goal = aug_imgs[:, 2]
            #

            # # shape of augmented images are (B, C, W, H)
            # elif self.augment_type == 'rad':
            # (chongyiz): RAD augmentation
            # transpose to (B, C, H, W)
            aug_img_obs = img_obs.permute(0, 1, 3, 2)
            aug_img_goal = img_goal.permute(0, 1, 3, 2)
            aug_next_img_obs = next_img_obs.permute(0, 1, 3, 2)

            # if self.same_augment_in_a_batch:
            #     # reshape batch as channels to make sure we apply same augmentation to the entire batch
            #     # aug_img_obs = aug_img_obs.reshape([1, -1, height, width])
            #     # aug_img_goal = aug_img_goal.reshape([1, -1, height, width])
            #     # aug_next_img_obs = aug_next_img_obs.reshape([1, -1, height, width])
            #
            #     for aug, func in self.augment_funcs.items():
            #         # apply same augmentation
            #         aug_img_obs_goal = func(torch.cat([aug_img_obs, aug_img_goal], dim=1))
            #         aug_img_obs, aug_img_goal = aug_img_obs_goal[:, :aug_img_obs_goal.shape[1] // 2], \
            #                                     aug_img_obs_goal[:, aug_img_obs_goal.shape[1] // 2:]
            #         aug_next_img_obs = func(aug_next_img_obs)
            #
            #     aug_img_obs = aug_img_obs.reshape([-1, 3, height, width])
            #     aug_img_goal = aug_img_goal.reshape([-1, 3, height, width])
            #     aug_next_img_obs = aug_next_img_obs.reshape([-1, 3, height, width])
            # else:
            #     for aug, func in self.augment_funcs.items():
            #         # skip crop and cutout augs
            #         # if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
            #         #     continue
            #         # obses = func(obses)
            #         # next_obses = func(next_obses)
            #
            #         # apply same augmentation and
            #         aug_img_obs_goal = func(torch.cat([aug_img_obs, aug_img_goal], dim=0))
            #         aug_img_obs, aug_img_goal = aug_img_obs_goal[:, :3], aug_img_obs_goal[:, 3:]
            #         aug_next_img_obs = func(aug_next_img_obs)
            #
            #     # transpose to (B, C, W, H)
            #     aug_img_obs = aug_img_obs.permute(0, 1, 3, 2)
            #     aug_img_goal = aug_img_goal.permute(0, 1, 3, 2)
            #     aug_next_img_obs = aug_next_img_obs.permute(0, 1, 3, 2)

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

        # torch.cuda.synchronize()
        # critic_loss_1_start_time = time.time()
        if self.use_td:
            new_goal = next_obs
        I = torch.eye(batch_size, device=ptu.device)
        # TODO (chongyiz): implement 3 dim logits
        logits, sa_repr, g_repr, sa_repr_norm, g_repr_norm = self.qf(
            torch.cat([obs, new_goal], -1), action, repr=True)

        # (chongyiz): Compute classifier accuracies
        metric_logits = logits.mean(-1)
        correct = (torch.argmax(metric_logits, dim=-1) == torch.argmax(I, dim=-1))
        logits_pos = torch.sum(metric_logits * I) / torch.sum(I)
        logits_neg = torch.sum(metric_logits * (1 - I)) / torch.sum(1 - I)
        q_pos, q_neg = torch.sum(torch.sigmoid(metric_logits) * I) / torch.sum(I), \
                       torch.sum(torch.sigmoid(metric_logits) * (1 - I)) / torch.sum(1 - I)
        q_pos_ratio, q_neg_ratio = q_pos / (1 - q_pos), q_neg / (1 - q_neg)
        binary_accuracy = torch.mean(((metric_logits > 0) == I).float())
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
            next_v = torch.min(next_q, dim=-1)[0].detach()  # NOQA
            next_v = torch.diag(next_v)
            w = next_v / (1 - next_v)  # NOQA
            w_clipping = 20.0
            w = torch.clamp(w, min=0.0, max=w_clipping)  # NOQA

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

            # Take the mean here so that we can compute the accuracy.
            # logits = torch.mean(logits, dim=-1)
        else:  # For the MC losses.
            w = ptu.zeros(1)

            # decrease the weight of negative term to 1 / (B - 1)
            qf_loss_weights = ptu.ones((batch_size, batch_size)) / (batch_size - 1)
            qf_loss_weights[torch.arange(batch_size), torch.arange(batch_size)] = 1
            if len(logits.shape) == 3:
                # logits.shape = (B, B, 2) with 1 term for positive pair
                # and (B - 1) terms for negative pairs in each row

                qf_loss = self.qf_criterion(
                    logits, I.unsqueeze(-1).repeat(1, 1, 2)).mean(-1)

                # Take the mean here so that we can compute the accuracy.
                # logits = torch.mean(logits, dim=-1)
            else:
                qf_loss = self.qf_criterion(logits, I)
            qf_loss *= qf_loss_weights

            qf_loss = torch.mean(qf_loss)

        """
        Policy and Alpha Loss
        """
        # TODO (chongyiz): implement gcbc
        # if self.use_gcbc:
        #     raise NotImplementedError
        # else:

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

            # ContrastiveQf Statistics
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
