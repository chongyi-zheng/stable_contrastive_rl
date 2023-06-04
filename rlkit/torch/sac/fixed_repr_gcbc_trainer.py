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


class FixedRepresentationGCBCTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            # behavioral_cloning_policy,
            # qf,
            # qf1,
            # qf2,
            # vf,
            # target_vf,
            # plan_vf,
            # affordance,
            # obs_encoder,
            obs_dim,
            vqvae=None,

            # quantile=0.5,
            # target_qf1=None,
            # target_qf2=None,
            # target_qf=None,
            # buffer_policy=None,

            # kld_weight=1.0,
            # affordance_pred_weight=10000.,
            # affordance_beta=1.0,

            # discount=0.99,
            # reward_scale=1.0,

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
            # soft_target_tau=1e-2,
            # target_update_period=1,
            # beta=1.0,
            # min_value=None,
            # max_value=None,

            # Contrastive RL arguments
            # use_td=False,
            # vf_ratio_loss=False,
            # use_b_squared_td=False,
            # use_vf_w=False,
            # self_normalized_vf_w=False,
            # multiply_batch_size_scale=True,
            # add_mc_to_td=False,
            # use_gcbc=False,
            # entropy_coefficient=None,
            # target_entropy=None,
            # random_goals=0.0,
            # bc_coef=0.05,
            bc_augmentation=False,

            # adv_weighted_loss=False,
            # actor_q_loss=True,
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

            augment_type='default',
            augment_params=dict(),
            augment_order=[],
            rad_augment_order=[],
            augment_probability=0.0,
            # reencode_augmented_images=False,
            same_augment_in_a_batch=True,

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
        # self.qf = qf
        # self.target_qf = target_qf
        # self.soft_target_tau = soft_target_tau
        # self.target_update_period = target_update_period
        self.update_period = update_period
        self.gradient_clipping = gradient_clipping
        # self.vf = vf
        # self.target_vf = target_vf
        # self.plan_vf = plan_vf
        # self.affordance = affordance
        # self.obs_encoder = obs_encoder
        # self.buffer_policy = buffer_policy
        # self.bc = bc
        # self.backprop_from_policy = backprop_from_policy

        self.vqvae = vqvae

        self.obs_dim = obs_dim
        # self.kld_weight = kld_weight

        # self.qf_criterion = nn.BCEWithLogitsLoss(reduction='none')
        # self.vf_criterion = nn.MSELoss()
        # self.vf_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # Contrastive RL attributes
        # self.critic_lr_warmup = critic_lr_warmup
        # self.use_td = use_td
        # self.vf_ratio_loss = vf_ratio_loss
        # self.use_b_squared_td = use_b_squared_td
        # self.use_vf_w = use_vf_w
        # self.self_normalized_vf_w = self_normalized_vf_w
        # self.multiply_batch_size_scale = multiply_batch_size_scale
        # self.add_mc_to_td = add_mc_to_td
        # self.use_gcbc = use_gcbc
        # self.entropy_coefficient = entropy_coefficient
        # self.adaptive_entropy_coefficient = entropy_coefficient is None
        # self.target_entropy = target_entropy
        # (chongyiz): For the actor update, only use future states as goals in offline setting,
        # i.e. random_goals = 0.0.
        # self.random_goals = random_goals
        # self.bc_coef = bc_coef
        self.bc_augmentation = bc_augmentation

        # self.adv_weighted_loss = adv_weighted_loss
        # self.actor_q_loss = actor_q_loss
        self.bc_train_val_split = bc_train_val_split
        # self.alignment_alpha = alignment_alpha
        # self.alignment_coef = alignment_coef

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=lr,
        )

        # self.discount = discount
        # self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self.n_train_steps_total = 0

        # self.goal_is_encoded = goal_is_encoded

        self.need_to_update_eval_statistics = {
            'train/': True,
            'eval/': True,
        }

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
        self.augment_type = augment_type
        if augment_probability > 0:
            assert vqvae is not None

            width = self.vqvae.imsize
            height = self.vqvae.imsize
            self.augment_stack = create_aug_stack(
                augment_order, augment_params, size=(width, height)
            )
            self.same_augment_in_a_batch = same_augment_in_a_batch

            self.augment_funcs = {}
            for aug_name in rad_augment_order:
                assert aug_name in AUG_TO_FUNC, 'invalid data aug string'
                self.augment_funcs[aug_name] = AUG_TO_FUNC[aug_name]
        else:
            self.augment_stack = None
            self.augment_funcs = {}


    def augment(self, batch, train=True):
        augmented_batch = dict()
        for key, value in batch.items():
            augmented_batch[key] = value

        if (train and self.augment_probability > 0 and
                torch.rand(1) < self.augment_probability and
                batch['observations'].shape[0] > 0):
            width = self.vqvae.imsize
            height = self.vqvae.imsize

            img_obs = batch['observations'].reshape(
                -1, 3, width, height)
            next_img_obs = batch['next_observations'].reshape(
                -1, 3, width, height)
            img_goal = batch['contexts'].reshape(
                -1, 3, width, height)

            # self.set_augment_params(obs)

            if self.augment_type == 'default':
                # Randomized transformations will apply the same transformation to all the images of a given batch
                # Reference: https://pytorch.org/vision/stable/transforms.html
                if self.same_augment_in_a_batch:
                    self.augment_stack.set_params(img_obs)
                    aug_img_obs = self.augment_stack(img_obs)
                    aug_next_img_obs = self.augment_stack(next_img_obs)
                    aug_img_goal = self.augment_stack(img_goal)
                else:
                    # Also see: https://discuss.pytorch.org/t/applying-different-data-augmentation-per-image-in-a-mini-batch/139136
                    imgs = torch.stack([img_obs, next_img_obs, img_goal], dim=1)

                    def set_augment_params_and_augment(img):
                        self.augment_stack.set_params(img)
                        return self.augment_stack(img)

                    aug_imgs = transforms.Lambda(lambda xs: torch.stack(
                        [set_augment_params_and_augment(x) for x in xs]))(imgs)
                    aug_img_obs = aug_imgs[:, 0]
                    aug_next_img_obs = aug_imgs[:, 1]
                    aug_img_goal = aug_imgs[:, 2]

            # shape of augmented images are (B, C, W, H)
            elif self.augment_type == 'rad':
                # (chongyiz): RAD augmentation
                # transpose to (B, C, H, W)
                aug_img_obs = img_obs.permute(0, 1, 3, 2)
                aug_img_goal = img_goal.permute(0, 1, 3, 2)
                aug_next_img_obs = next_img_obs.permute(0, 1, 3, 2)

                if self.same_augment_in_a_batch:
                    # reshape batch as channels to make sure we apply same augmentation to the entire batch
                    # aug_img_obs = aug_img_obs.reshape([1, -1, height, width])
                    # aug_img_goal = aug_img_goal.reshape([1, -1, height, width])
                    # aug_next_img_obs = aug_next_img_obs.reshape([1, -1, height, width])

                    for aug, func in self.augment_funcs.items():
                        # apply same augmentation
                        aug_img_obs_goal = func(torch.cat([aug_img_obs, aug_img_goal], dim=1))
                        aug_img_obs, aug_img_goal = aug_img_obs_goal[:, :aug_img_obs_goal.shape[1] // 2], \
                                                    aug_img_obs_goal[:, aug_img_obs_goal.shape[1] // 2:]
                        aug_next_img_obs = func(aug_next_img_obs)

                    aug_img_obs = aug_img_obs.reshape([-1, 3, height, width])
                    aug_img_goal = aug_img_goal.reshape([-1, 3, height, width])
                    aug_next_img_obs = aug_next_img_obs.reshape([-1, 3, height, width])
                else:
                    for aug, func in self.augment_funcs.items():
                        # skip crop and cutout augs
                        # if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                        #     continue
                        # obses = func(obses)
                        # next_obses = func(next_obses)

                        # apply same augmentation and
                        aug_img_obs_goal = func(torch.cat([aug_img_obs, aug_img_goal], dim=0))
                        aug_img_obs, aug_img_goal = aug_img_obs_goal[:, :3], aug_img_obs_goal[:, 3:]
                        aug_next_img_obs = func(aug_next_img_obs)

                # transpose to (B, C, W, H)
                aug_img_obs = aug_img_obs.permute(0, 1, 3, 2)
                aug_img_goal = aug_img_goal.permute(0, 1, 3, 2)
                aug_next_img_obs = aug_next_img_obs.permute(0, 1, 3, 2)

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
        aug_next_obs = batch['augmented_next_observations']
        aug_goal = batch['augmented_contexts']

        new_obs = obs
        new_goal = goal
        new_aug_obs = aug_obs
        new_aug_goal = aug_goal

        # new_obs = jnp.concatenate([new_state, new_goal], axis=1)
        # new_h = torch.cat([new_obs_feat, new_goal_feat], -1)
        new_obs_goal = torch.cat([new_obs, new_goal], -1)
        new_aug_obs_goal = torch.cat([new_aug_obs, new_aug_goal], -1)
        # dist_params = networks.policy_network.apply(
        #     policy_params, new_obs)
        # if isinstance(self.policy, EncodingGaussianPolicy):
        #     dist = self.policy(new_h, encoded_input=True)
        # elif isinstance(self.policy, EncodingGaussianPolicyV2):
        #     # (chongyiz): is this correct?
        #     _h = torch.cat([obs, random_goal_feat], -1)
        #     dist = self.policy(_h, encoded_input=True)
        # else:
        #     raise NotImplementedError
        # (chongyiz): sticking with EncodingGaussianPolicy
        # new_dist = self.policy(new_h, encoded_input=True)
        dist = self.policy(new_obs_goal)
        dist_aug = self.policy(new_aug_obs_goal)
        # sampled_action, log_prob = dist.rsample_and_logprob()
        # log_prob = log_prob.unsqueeze(-1)

        # action = networks.sample(dist_params, key)
        # log_prob = networks.log_prob(dist_params, action)
        # q_action = networks.q_network.apply(
        #     q_params, new_obs, action)
        # if len(q_action.shape) == 3:  # twin q trick
        #     assert q_action.shape[2] == 2
        #     q_action = jnp.min(q_action, axis=-1)

        orig_action = action

        if self.bc_train_val_split:
            train_mask = ((orig_action * 1E8 % 10)[:, 0] != 4).float()
        else:
            train_mask = ptu.ones(orig_action.shape[0])

        gcbc_loss = -train_mask * dist.log_prob(orig_action)
        gcbc_val_loss = -(1.0 - train_mask) * dist.log_prob(orig_action)
        aug_gcbc_loss = -train_mask * dist_aug.log_prob(orig_action)
        aug_gcbc_val_loss = -(1.0 - train_mask) * dist_aug.log_prob(orig_action)

        if self.bc_augmentation:
            actor_loss = aug_gcbc_loss
        else:
            actor_loss = gcbc_loss

        metric_gcbc_loss = torch.sum(gcbc_loss) / torch.sum(train_mask)
        metric_aug_gcbc_loss = torch.sum(aug_gcbc_loss) / torch.sum(train_mask)
        if torch.sum(1 - train_mask) > 0:
            metric_gcbc_val_loss = torch.sum(gcbc_val_loss) / torch.sum(1 - train_mask)
            metric_aug_gcbc_val_loss = torch.sum(aug_gcbc_val_loss) / torch.sum(1 - train_mask)
        else:
            metric_gcbc_val_loss = ptu.zeros(1)
            metric_aug_gcbc_val_loss = ptu.zeros(1)

        actor_loss = torch.mean(actor_loss)

        if train:
            """
            Optimization.
            """
            if self.n_train_steps_total % self.update_period == 0:
                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                if (self.gradient_clipping is not None and
                        self.gradient_clipping > 0):
                    torch.nn.utils.clip_grad_norm(
                        self.policy.parameters(), self.gradient_clipping)
                self.policy_optimizer.step()

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
            self.eval_statistics[prefix + 'Policy Loss'] = np.mean(
                ptu.get_numpy(actor_loss))
            self.eval_statistics[prefix + 'Policy Loss/GCBC Loss'] = np.mean(
                ptu.get_numpy(metric_gcbc_loss))
            self.eval_statistics[prefix + 'Policy Loss/GCBC Val Loss'] = np.mean(
                ptu.get_numpy(metric_gcbc_val_loss))
            self.eval_statistics[prefix + 'Policy Loss/Augmented GCBC Loss'] = np.mean(
                ptu.get_numpy(metric_aug_gcbc_loss))
            self.eval_statistics[prefix + 'Policy Loss/Augmented GCBC Val Loss'] = np.mean(
                ptu.get_numpy(metric_aug_gcbc_val_loss))

            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Policy Mean',
                ptu.get_numpy(dist.mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                prefix + 'Policy STD',
                ptu.get_numpy(dist.stddev),
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
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
        )
