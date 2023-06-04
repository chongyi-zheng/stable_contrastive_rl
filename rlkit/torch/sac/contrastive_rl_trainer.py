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


class ContrastiveRLTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            behavioral_cloning_policy,
            qf,
            # qf1,
            # qf2,
            vf,
            target_vf,
            # plan_vf,
            # affordance,
            # obs_encoder,
            obs_dim,
            vqvae=None,

            # quantile=0.5,
            # target_qf1=None,
            # target_qf2=None,
            target_qf=None,
            buffer_policy=None,

            kld_weight=1.0,
            # affordance_pred_weight=10000.,
            # affordance_beta=1.0,

            discount=0.99,
            reward_scale=1.0,

            lr=3e-4,
            gradient_clipping=None,  # TODO
            optimizer_class=optim.Adam,
            critic_lr_warmup=False,

            # backprop_from_policy=True,
            bc=False,

            # actor_update_period=1,  # (chongyiz): Is this flag used?
            # critic_update_period=1,
            update_period=1,

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

            # Contrastive RL arguments
            use_td=False,
            vf_ratio_loss=False,
            use_b_squared_td=False,
            use_vf_w=False,
            self_normalized_vf_w=False,
            multiply_batch_size_scale=True,
            add_mc_to_td=False,
            use_gcbc=False,
            entropy_coefficient=None,
            target_entropy=None,
            random_goals=0.0,
            bc_coef=0.05,
            bc_augmentation=False,

            adv_weighted_loss=False,
            actor_q_loss=True,
            bc_train_val_split=False,

            alignment_alpha=2,
            alignment_coef=0.0,

            goal_is_encoded=False,

            # fraction_negative_obs=0.3,
            # fraction_negative_goal=0.3,
            end_to_end=False,  # TODO
            # affordance_weight=1.0,

            train_encoder=True,

            use_encoding_reward=False,
            encoding_reward_thresh=None,

            augment_type='default',
            augment_params=dict(),
            augment_order=[],
            rad_augment_order=[],
            augment_probability=0.0,
            reencode_augmented_images=False,
            same_augment_in_a_batch=True,

            vip_gcbc=False,
            r3m_gcbc=False,

            * args,
            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        # (chongyiz): behavioral_cloning_policy is not used now
        self.behavioral_cloning_policy = behavioral_cloning_policy
        # self.qf1 = qf1
        # self.qf2 = qf2
        # self.target_qf1 = target_qf1
        # self.target_qf2 = target_qf2
        self.qf = qf
        self.target_qf = target_qf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.gradient_clipping = gradient_clipping
        self.vf = vf
        self.target_vf = target_vf
        # self.plan_vf = plan_vf
        # self.affordance = affordance
        # self.obs_encoder = obs_encoder
        self.buffer_policy = buffer_policy
        self.bc = bc
        # self.backprop_from_policy = backprop_from_policy

        self.vqvae = vqvae

        self.obs_dim = obs_dim
        self.kld_weight = kld_weight

        self.qf_criterion = nn.BCEWithLogitsLoss(reduction='none')
        # self.vf_criterion = nn.MSELoss()
        self.vf_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # Contrastive RL attributes
        self.critic_lr_warmup = critic_lr_warmup
        self.use_td = use_td
        self.vf_ratio_loss = vf_ratio_loss
        self.use_b_squared_td = use_b_squared_td
        self.use_vf_w = use_vf_w
        self.self_normalized_vf_w = self_normalized_vf_w
        self.multiply_batch_size_scale = multiply_batch_size_scale
        self.add_mc_to_td = add_mc_to_td
        self.use_gcbc = use_gcbc
        self.entropy_coefficient = entropy_coefficient
        self.adaptive_entropy_coefficient = entropy_coefficient is None
        self.target_entropy = target_entropy
        # (chongyiz): For the actor update, only use future states as goals in offline setting,
        # i.e. random_goals = 0.0.
        self.random_goals = random_goals
        self.bc_coef = bc_coef
        self.bc_augmentation = bc_augmentation

        self.adv_weighted_loss = adv_weighted_loss
        self.actor_q_loss = actor_q_loss
        self.bc_train_val_split = bc_train_val_split
        self.alignment_alpha = alignment_alpha
        self.alignment_coef = alignment_coef

        self.vip_gcbc = vip_gcbc
        self.r3m_gcbc = r3m_gcbc

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
        if self.critic_lr_warmup:
            optimizer = optimizer_class(
                self.qf.parameters(),
                lr=lr,
            )
            self.qf_optimizer = LRWarmUp(lr, 100_000, optimizer)
        else:
            self.qf_optimizer = optimizer_class(
                self.qf.parameters(),
                lr=lr,
            )

        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=lr,
        )
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
        ptu.copy_model_params_from_to(self.vf, self.target_vf)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self.n_train_steps_total = 0

        self.goal_is_encoded = goal_is_encoded
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

        self.clip_score = clip_score
        self.beta = beta
        # self.quantile = quantile
        # self.train_encoder = train_encoder

        self.min_value = min_value
        self.max_value = max_value

        self.end_to_end = end_to_end
        # self.affordance_weight = affordance_weight

        self.use_encoding_reward = use_encoding_reward
        self.encoding_reward_thresh = encoding_reward_thresh

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

        if self.vip_gcbc:
            # from vip import load_vip
            # self.vip = load_vip()
            # self.vip.eval()
            # self.vip.to(ptu.device)
            #
            # ## DEFINE PREPROCESSING
            # import torchvision.transforms as T
            # from PIL import Image
            #
            # transforms = T.Compose([T.Resize(256),
            #                         T.CenterCrop(48),
            #                         T.ToTensor()])  # ToTensor() divides by 255
            #
            # ## ENCODE IMAGE
            # image = np.random.randint(0, 255, (500, 500, 3))
            # preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 48, 48)
            # preprocessed_image.to(ptu.device)
            # with torch.no_grad():
            #     embedding = self.vip(preprocessed_image * 255.0)  ## vip expects image input to be [0-255]
            # print(embedding.shape)  # [1, 1024]
            pass

        # TODO
        if self.r3m_gcbc:
            raise NotImplementedError

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
            # FIXEME (chongyiz)
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

            # # DEBUG (chongyiz): visualize original and augmented images
            # img_obs_vis = ptu.get_numpy(img_obs.permute(0, 3, 2, 1))
            # img_goal_vis = ptu.get_numpy(img_goal.permute(0, 3, 2, 1))
            # aug_img_obs_vis = ptu.get_numpy(aug_img_obs.permute(0, 3, 2, 1))
            # aug_img_goal_vis = ptu.get_numpy(aug_img_goal.permute(0, 3, 2, 1))
            #
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(nrows=20, ncols=2)
            # fig.set_figheight(4 * 20)
            # fig.set_figwidth(4 * 4)
            #
            # for idx in range(10):
            #     axes[2 * idx, 0].imshow(img_obs_vis[idx])
            #     axes[2 * idx, 0].set_title("Original Observation")
            #     axes[2 * idx, 1].imshow(aug_img_obs_vis[idx])
            #     axes[2 * idx, 1].set_title("Augmented Observation")
            #     axes[2 * idx + 1, 0].imshow(img_goal_vis[idx])
            #     axes[2 * idx + 1, 0].set_title("Original Goal")
            #     axes[2 * idx + 1, 1].imshow(aug_img_goal_vis[idx])
            #     axes[2 * idx + 1, 1].set_title("Augmented Goal")
            #
            # plt.tight_layout()
            # fig_save_path = "/projects/rsalakhugroup/chongyiz/offline_c_learning/railrl_logs_debug/img_encoder_augmentation.png"
            # plt.savefig(fig_save_path)
            # print("Save figure to: {}".format(fig_save_path))
            #
            # exit()

            augmented_batch['augmented_observations'] = aug_img_obs.flatten(1)
            augmented_batch['augmented_next_observations'] = aug_next_img_obs.flatten(1)
            augmented_batch['augmented_contexts'] = aug_img_goal.flatten(1)
        else:
            augmented_batch['augmented_observations'] = augmented_batch['observations']
            augmented_batch['augmented_next_observations'] = augmented_batch['next_observations']
            augmented_batch['augmented_contexts'] = augmented_batch['contexts']

        return augmented_batch

    def compute_vf_loss(self, obs, goal):
        # TODO (chongyiz): ablate other goals
        obs_goal = torch.cat([obs, goal], -1)
        dist = self.policy(obs_goal)
        action = dist.sample()
        target_logits = self.qf(obs_goal, action).detach()
        predicted_logits = self.vf(obs_goal)

        # BCE has the same optimal predictions as MSE
        if self.vf_ratio_loss:
            mse = nn.MSELoss(reduction='none')
            vf_loss = mse(
                predicted_logits,
                torch.sigmoid(target_logits) / (1 - torch.sigmoid(target_logits))
            )
            vf_loss = torch.mean(vf_loss)
        else:
            vf_loss = self.vf_criterion(input=predicted_logits, target=torch.sigmoid(target_logits))
            # vf_loss = self.vf_criterion(predicted_logits, target_logits)
            vf_loss = torch.mean(vf_loss)

        return vf_loss

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

        # """
        # Obs Encoder
        # """
        # if self.goal_is_encoded:
        #     obs_feat = obs
        #     obs_feat_mu = obs
        #     obs_feat_logvar = ptu.zeros_like(obs)
        #
        #     next_obs_feat = next_obs
        #     next_obs_feat_mu = next_obs
        #     next_obs_feat_logvar = ptu.zeros_like(next_obs)
        #
        #     goal_feat = goal
        #     goal_feat_mu = goal
        #     goal_feat_logvar = ptu.zeros_like(goal)
        #
        # else:
        #     obs_feat, (obs_feat_mu, obs_feat_logvar) = self.obs_encoder(
        #         obs, training=True)
        #     next_obs_feat, (next_obs_feat_mu, next_obs_feat_logvar) = self.obs_encoder(  # NOQA
        #         next_obs, training=True)
        #     goal_feat, (goal_feat_mu, goal_feat_logvar) = self.obs_encoder(
        #         goal, training=True)
        #
        # if not self.train_encoder:
        #     obs_feat = obs_feat.detach()
        #     obs_feat_mu = obs_feat_mu.detach()
        #     obs_feat_logvar = obs_feat_logvar.detach()
        #     next_obs_feat = next_obs_feat.detach()
        #     next_obs_feat_mu = next_obs_feat_mu.detach()
        #     next_obs_feat_logvar = next_obs_feat_logvar.detach()
        #     goal_feat = goal_feat.detach()
        #     goal_feat_mu = goal_feat_mu.detach()
        #     goal_feat_logvar = goal_feat_logvar.detach()
        #
        #     obs_feat = obs_feat_mu
        #     next_obs_feat = next_obs_feat_mu
        #     goal_feat = goal_feat_mu

        # if self.kld_weight == 0:
        #     obs_feat = obs_feat_mu
        #     next_obs_feat = next_obs_feat_mu
        #     goal_feat = goal_feat_mu

        # if self.reward_transform:
        #     reward = self.reward_transform(reward)
        # if self.terminal_transform:
        #     terminal = self.terminal_transform(terminal)

        # TODO: obs_encoder reward.
        # if self.use_encoding_reward:
        #     similarity = torch.nn.functional.cosine_similarity(
        #         next_obs_feat_mu, goal_feat_mu).detach()
        #     success = similarity >= self.encoding_reward_thresh
        #     reward = success.to(torch.float) - 1.0
        #     terminal = success.to(torch.float)
        #
        #     terminal = ptu.zeros_like(terminal)  # TODO
        #
        #     reward = reward.view(-1, 1)
        #     terminal = terminal.view(-1, 1)

        # h = torch.cat([obs_feat, goal_feat], -1)
        # # (chongyiz): add next_h
        # next_h = torch.cat([next_obs_feat, goal_feat], -1)
        #
        # target_h = torch.cat([obs_feat_mu, goal_feat_mu], -1)
        # target_next_h = torch.cat([next_obs_feat_mu, goal_feat_mu], -1)

        """
        QF Loss
        """
        # DELEME (chongyiz)
        # q1_pred = self.qf1(h, action)
        # q2_pred = self.qf2(h, action)
        # target_next_vf_pred = self.vf(target_next_h).detach()
        # if self.min_value is not None or self.max_value is not None:
        #     target_next_vf_pred = torch.clamp(
        #         target_next_vf_pred,
        #         min=self.min_value,
        #         max=self.max_value,
        #     )
        #
        # assert reward.shape == terminal.shape
        # assert reward.shape == target_next_vf_pred.shape
        # q_target = self.reward_scale * reward + \
        #     (1. - terminal) * self.discount * target_next_vf_pred
        # q_target = q_target.detach()
        # assert q1_pred.shape == q_target.shape
        # assert q2_pred.shape == q_target.shape
        # qf1_loss = self.qf_criterion(q1_pred, q_target)
        # qf2_loss = self.qf_criterion(q2_pred, q_target)

        batch_size = obs.shape[0]
        new_goal = goal

        # torch.cuda.synchronize()
        # critic_loss_1_start_time = time.time()
        if self.use_td:
            # # TODO (chongyiz): replace representation_size with obs_dim
            # s_h, g_h = torch.split(
            #     h, [self.obs_encoder.representation_size], dim=1)
            # next_s_h, _ = torch.split()

            # TODO (chongyiz): add_mc_to_td
            if self.add_mc_to_td:
                raise NotImplementedError
            else:
                new_goal = next_obs
                # h = torch.cat([obs_feat, new_goal_feat], -1)
        I = torch.eye(batch_size, device=ptu.device)
        # TODO (chongyiz): implement 3 dim logits
        logits, sa_repr, g_repr, sa_repr_norm, g_repr_norm = self.qf(
            torch.cat([obs, new_goal], -1), action, repr=True)
        if self.qf._repr_norm_temp:
            sa_repr = sa_repr * torch.exp(self.qf.repr_log_scale)

        # compute cosine similarities
        # cos = nn.CosineSimilarity(dim=1, eps=1e-32)
        # cos_sims = torch.mean(cos(sa_repr, g_repr))
        # print("Cosine similarity for positive pairs: {}".format(cos_sims))

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

        # (chongyiz): compute align_loss and uniform_loss could be slow
        if self.alignment_coef > 0:
            # compute alignment and uniformity metrics for representation learning
            def compute_align_loss(x, y, alpha=2):
                return (x - y).norm(p=2, dim=1).pow(alpha).mean()

            def compute_uniform_loss(x, t=2):
                return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

            # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html
            # torch.pdist will be faster if the rows are contiguous.
            # TODO (chongyiz): sanity check of positive pairs
            metric_sa_repr = torch.mean(sa_repr, dim=-1).contiguous()
            metric_g_repr = torch.mean(g_repr, dim=-1).contiguous()
            align_loss = compute_align_loss(metric_sa_repr, metric_g_repr, alpha=self.alignment_alpha)
            unif_loss = (compute_uniform_loss(metric_sa_repr) + compute_uniform_loss(metric_g_repr)) / 2
        else:
            align_loss = ptu.zeros(1)
            unif_loss = ptu.zeros(1)
        # torch.cuda.synchronize()
        # cls_acc_end_time = time.time()
        # print("Time to compute classifier accuracies: {} secs".format(cls_acc_end_time - cls_acc_start_time))

        # pos_logits2 = self.qf2(h, action)

        # goal_indices = torch.roll(torch.arange(batch_size, dtype=torch.int64), -1)
        # rand_goal_feat = goal_feat[goal_indices]
        # rand_h = torch.cat([obs_feat, rand_goal_feat], -1)
        # neg_logits1 = self.qf1(rand_h, action)
        # neg_logits2 = self.qf2(rand_h, action)

        if self.use_td:
            # Make sure to use the twin Q trick.
            assert len(logits.shape) == 3

            if self.use_b_squared_td:
                assert self.use_vf_w

                next_v = self.target_vf(torch.cat([next_obs, new_goal], dim=-1))
                if not self.vf_ratio_loss:
                    next_v = torch.sigmoid(next_v)
                # TODO (chongyiz): take the minimum of two V
                next_v = torch.min(next_v, dim=-1)[0].detach()

                # w will be (B, B)
                if self.vf_ratio_loss:
                    w = next_v
                else:
                    w = next_v / (1 - next_v)  # NOQA

                if self.self_normalized_vf_w:
                    # only normalize over negative terms
                    # w = w / torch.mean(w, dim=1, keepdim=True)  # NOQA
                    w[torch.arange(batch_size), torch.arange(batch_size)] = 0
                    w = w / torch.sum(w, dim=1, keepdim=True)  # NOQA
                else:
                    w_clipping = 20.0
                    w = torch.clamp(w, min=0.0, max=w_clipping)  # NOQA
                w = w.unsqueeze(-1).repeat(1, 1, 2)

                targets = self.discount * w / (1 + self.discount * w)
                targets[torch.arange(batch_size), torch.arange(batch_size)] = 1
                if self.multiply_batch_size_scale:
                    qf_loss_weights = (1 + self.discount * w)
                    qf_loss_weights[torch.arange(batch_size), torch.arange(batch_size)] = \
                        (1 - self.discount) * (batch_size - 1)
                else:
                    qf_loss_weights = (1 + self.discount * w) / (batch_size - 1)
                    qf_loss_weights[torch.arange(batch_size), torch.arange(batch_size)] = \
                        (1 - self.discount)

                qf_loss = qf_loss_weights * self.qf_criterion(logits, targets)
                qf_loss = torch.mean(qf_loss)

            else:
                # We evaluate the next-state Q function using random goals
                # random_goal_feat_mu = goal_feat_mu[goal_indices]
                # random_target_next_h = torch.cat(
                #     [next_obs_feat_mu, random_goal_feat_mu], -1)
                goal_indices = torch.roll(
                    torch.arange(batch_size, dtype=torch.int64), -1)

                # (chongyiz): the random goal of w should match the random goal for neg_logit
                # random_goal = goal[goal_indices]
                random_goal = new_goal[goal_indices]

                # next_dist_params = networks.policy_network.apply(
                #     policy_params, transitions.next_observation)
                # if isinstance(self.policy, EncodingGaussianPolicy):
                #     dist = self.policy(random_target_next_h, encoded_input=True)
                # elif isinstance(self.policy, EncodingGaussianPolicyV2):
                #     _h = torch.cat([obs, random_goal_feat], -1)
                #     dist = self.policy(_h, encoded_input=True)
                # else:
                #     raise NotImplementedError
                # TODO (chongyiz): sticking with EncodingGaussianPolicy
                next_s_rand_g = torch.cat([next_obs, random_goal], -1)

                if self.use_vf_w:
                    next_v = self.target_vf(next_s_rand_g)
                    if not self.vf_ratio_loss:
                        next_v = torch.sigmoid(next_v)
                    next_v = torch.min(next_v, dim=-1)[0].detach()
                    next_v = torch.diag(next_v)

                    if self.vf_ratio_loss:
                        w = next_v
                    else:
                        w = next_v / (1 - next_v)  # NOQA
                else:
                    next_dist = self.policy(next_s_rand_g)
                    next_action = next_dist.rsample()

                    next_q = self.target_qf(
                        next_s_rand_g, next_action)
                    # torch.cuda.synchronize()
                    # target_qf_end_time = time.time()
                    # print("Time to compute target_qf for C-learning: {} secs".format(target_qf_end_time - target_qf_start_time))

                    # if self.target_qf._repr_norm_temp:
                    #     target_sa_repr = target_sa_repr * torch.exp(self.target_qf.repr_log_scale)

                    next_q = torch.sigmoid(next_q)
                    next_v = torch.min(next_q, dim=-1)[0].detach()  # NOQA
                    next_v = torch.diag(next_v)
                    w = next_v / (1 - next_v)  # NOQA
                w_clipping = 20.0
                w = torch.clamp(w, min=0.0, max=w_clipping)  # NOQA
                # pos_logits1 = logits1
                # pos_logits2 = logits2
                # qf1_loss_pos = torch.nn.functional.binary_cross_entropy_with_logits(
                #     pos_logits1, ptu.ones_like(pos_logits1))
                # qf2_loss_pos = torch.nn.functional.binary_cross_entropy_with_logits(
                #     pos_logits2, ptu.ones_like(pos_logits2))
                # (B, B, 2) --> (B, 2), computes diagonal of each twin Q.
                pos_logits = torch.diagonal(logits).permute(1, 0)
                loss_pos = self.qf_criterion(
                    pos_logits, ptu.ones_like(pos_logits))

                # neg_logits1 = self.qf1(random_h, action)
                # neg_logits2 = self.qf2(random_h, action)
                # qf1_loss_neg1 = w * torch.nn.functional.binary_cross_entropy_with_logits(
                #     neg_logits1, ptu.ones_like(neg_logits1))
                # qf2_loss_neg1 = w * torch.nn.functional.binary_cross_entropy_with_logits(
                #     neg_logits2, ptu.ones_like(neg_logits2))
                # qf1_loss_neg2 = torch.nn.functional.binary_cross_entropy_with_logits(
                #     neg_logits1, ptu.zeros_like(neg_logits1))
                # qf2_loss_neg2 = torch.nn.functional.binary_cross_entropy_with_logits(
                #     neg_logits2, ptu.zeros_like(neg_logits2))

                neg_logits = logits[torch.arange(batch_size), goal_indices]
                loss_neg1 = w[:, None] * self.qf_criterion(
                    neg_logits, ptu.ones_like(neg_logits))
                loss_neg2 = self.qf_criterion(
                    neg_logits, ptu.zeros_like(neg_logits))

                # TODO (chongyiz): add_mc_to_td
                if self.add_mc_to_td:
                    raise NotImplementedError
                else:
                    # qf1_loss = (1 - self.discount) * qf1_loss_pos + \
                    #            self.discount * qf1_loss_neg1 + qf1_loss_neg2  # NOQA
                    # qf2_loss = (1 - self.discount) * qf2_loss_pos + \
                    #            self.discount * qf2_loss_neg1 + qf2_loss_neg2  # NOQA

                    qf_loss = (1 - self.discount) * loss_pos + \
                              self.discount * loss_neg1 + loss_neg2
                    qf_loss = torch.mean(qf_loss)

            # Take the mean here so that we can compute the accuracy.
            # logits = torch.mean(logits, dim=-1)
        else:  # For the MC losses.
            # qf1_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            #     pos_logits1, ptu.ones_like(pos_logits1)) + \
            #            torch.nn.functional.binary_cross_entropy_with_logits(
            #                neg_logits1, ptu.zeros_like(neg_logits1))
            # qf2_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            #     pos_logits2, ptu.ones_like(pos_logits2)) + \
            #            torch.nn.functional.binary_cross_entropy_with_logits(
            #                neg_logits2, ptu.zeros_like(neg_logits2))

            # for logging
            w = ptu.zeros(1)

            if self.multiply_batch_size_scale:
                # increase the weight of positive term to (B - 1)
                qf_loss_weights = ptu.ones((batch_size, batch_size))
                qf_loss_weights[torch.arange(batch_size), torch.arange(batch_size)] = (batch_size - 1)
            else:
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

        # 'binary_accuracy': jnp.mean((logits > 0) == I),
        # 'categorical_accuracy': jnp.mean(correct),
        # 'logits_pos': logits_pos,
        # 'logits_neg': logits_neg,
        # 'logsumexp': logsumexp.mean(),

        # critic_loss = qf1_loss + qf2_loss

        # """
        # VF Loss
        # """
        # q_pred = torch.min(
        #     self.target_qf1(target_h, action),
        #     self.target_qf2(target_h, action),
        # ).detach()
        # if self.min_value is not None or self.max_value is not None:
        #     q_pred = torch.clamp(
        #         q_pred,
        #         min=self.min_value,
        #         max=self.max_value,
        #     )
        #
        # vf_pred = self.vf(h)
        # assert vf_pred.shape == q_pred.shape
        # vf_err = vf_pred - q_pred
        # vf_sign = (vf_err > 0).float()
        # vf_weight = (1 - vf_sign) * self.quantile + \
        #     vf_sign * (1 - self.quantile)
        # assert vf_weight.shape == vf_err.shape
        # vf_loss = (vf_weight * (vf_err ** 2)).mean()
        #
        # critic_loss = qf1_loss + qf2_loss + vf_loss

        # if self.train_encoder:
        #     plan_vf_loss, plan_vf_extra = self._compute_plan_vf_loss(
        #         obs_feat, goal_feat, vf_pred)
        #     critic_loss += plan_vf_loss

        vf_loss = self.compute_vf_loss(obs.clone(), new_goal.clone())

        # """
        # Affordance Loss
        # """
        # if self.train_encoder:
        #     # TODO: Maybe start training the affordance after 2 epochs or so.
        #     affordance_loss, affordance_extra = self._compute_affordance_loss(
        #         obs_feat_mu, goal_feat_mu)
        #     critic_loss += affordance_loss * self.affordance_weight

        """
        Policy and Alpha Loss
        """
        # DELETEME (chongyiz)
        # # TODO: Make sure the policy calls the same state encoder inside.
        # if isinstance(self.policy, EncodingGaussianPolicy):
        #     # if not self.backprop_from_policy:
        #     #     h = h.detach()
        #
        #     # TODO
        #     # noise = goal_feat.data.new(goal_feat.size()).normal_() * 0.1
        #     # h = torch.cat([obs_feat, goal_feat + noise], -1)
        #
        #     dist = self.policy(h, encoded_input=True)
        # elif isinstance(self.policy, EncodingGaussianPolicyV2):
        #     _h = torch.cat([obs, goal_feat], -1)
        #     dist = self.policy(_h, encoded_input=True)
        # else:
        #     raise NotImplementedError
        #     # dist = self.policy(obs_and_goal)
        #
        # actor_logpp = dist.log_prob(action)
        #
        # vf_baseline = self.vf(target_h).detach()  # TODO: Debugging.
        # assert q_pred.shape == vf_baseline.shape
        # adv = q_pred - vf_baseline
        # exp_adv = torch.exp(adv / self.beta)
        # if self.clip_score is not None:
        #     exp_adv = torch.clamp(exp_adv, max=self.clip_score)
        #
        # adv_weight = ptu.from_numpy(np.ones(exp_adv[:, 0].shape)).detach(
        # ) if self.bc else exp_adv[:, 0].detach()
        # assert actor_logpp.shape == adv_weight.shape
        # actor_loss = (-actor_logpp * adv_weight).mean()
        #
        # loss = critic_loss + actor_loss

        # TODO (chongyiz): implement gcbc
        if self.use_gcbc:
            raise NotImplementedError
        else:
            if self.random_goals == 0.0:
                # new_obs_feat = obs_feat
                # new_goal_feat = goal_feat
                new_obs = obs
                new_goal = goal
                new_aug_obs = aug_obs
                new_aug_goal = aug_goal
            elif self.random_goals == 0.5:
                # (chongyiz): is this correct?
                # new_obs_feat = torch.cat([obs_feat, obs_feat], 0)
                # new_goal_feat = torch.cat
                # TODO (chongyiz): check this
                raise NotImplementedError
                new_obs = torch.cat([obs, obs], 0)  # NOQA
                new_goal = torch.cat([  # NOQA
                    goal, torch.roll(goal, 1, dims=[0])], 0)  # NOQA
                new_aug_obs = torch.cat([aug_obs, aug_obs], 0)
                new_aug_goal = torch.cat([
                    aug_goal, torch.roll(aug_goal, 1, dims=[0])], 0)
            elif self.random_goals == 1.0:
                # new_obs_feat = obs_feat
                # new_goal_feat = goal_feat.roll(1, dim=0)
                # TODO (chongyiz): check this
                raise NotImplementedError
                new_obs = obs
                new_goal = torch.roll(goal, 1, dims=[0])  # NOQA
                new_aug_obs = aug_obs
                new_aug_goal = torch.roll(aug_goal, 1, dims=[0])
            else:
                raise NotImplementedError

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
            sampled_action, log_prob = dist.rsample_and_logprob()
            # log_prob = log_prob.unsqueeze(-1)

            # alpha = self.log_alpha.exp()
            if self.adaptive_entropy_coefficient:
                alpha_loss = -(self.log_alpha.exp() * (
                    log_prob + self.target_entropy).detach()).mean()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = self.entropy_coefficient

            # action = networks.sample(dist_params, key)
            # log_prob = networks.log_prob(dist_params, action)
            # q_action = networks.q_network.apply(
            #     q_params, new_obs, action)
            # if len(q_action.shape) == 3:  # twin q trick
            #     assert q_action.shape[2] == 2
            #     q_action = jnp.min(q_action, axis=-1)

            if self.adv_weighted_loss:
                # TODO (chongyiz): implement advantage weighted loss
                target_q = self.target_qf(new_obs_goal, action)
                if len(target_q.shape) == 3:  # twin q trick
                    assert target_q.shape[2] == 2
                    target_q = torch.min(target_q, dim=-1)[0]

                adv = torch.diag(target_q)  # TODO (chongyiz): implement vf_baseline
                exp_adv = torch.exp(adv / self.beta)
                if self.clip_score is not None:
                    exp_adv = torch.clamp(exp_adv, max=self.clip_score)
                adv_weight = exp_adv.detach()

                actor_logpp = dist.log_prob(action)

                # (chongyiz): we don't need entropy term for adv_weighted loss
                adv_weighted_loss = -actor_logpp * adv_weight
                actor_loss = adv_weighted_loss
            elif self.actor_q_loss:
                q_action = self.qf(new_obs_goal, sampled_action)

                if len(q_action.shape) == 3:  # twin q trick
                    assert q_action.shape[2] == 2
                    q_action = torch.min(q_action, dim=-1)[0]

                actor_q_loss = alpha * log_prob - torch.diag(q_action)
            else:
                raise RuntimeError("Actor loss is not defined!")

            assert 0.0 <= self.bc_coef <= 1.0
            if self.bc_coef > 0:
                orig_action = action
                if self.random_goals == 0.5:
                    orig_action = torch.cat([orig_action, orig_action], 0)

                if self.bc_train_val_split:
                    train_mask = ((orig_action * 1E8 % 10)[:, 0] != 4).float()
                else:
                    train_mask = ptu.ones(orig_action.shape[0])

                gcbc_loss = -train_mask * dist.log_prob(orig_action)
                gcbc_val_loss = -(1.0 - train_mask) * dist.log_prob(orig_action)
                aug_gcbc_loss = -train_mask * dist_aug.log_prob(orig_action)
                aug_gcbc_val_loss = -(1.0 - train_mask) * dist_aug.log_prob(orig_action)
                # else:
                #     gcbc_loss = -dist.log_prob(orig_action)
                #     aug_gcbc_loss = -dist_aug.log_prob(orig_action)
                #
                #     gcbc_val_loss = ptu.zeros(1)
                #     aug_gcbc_val_loss = ptu.zeros(1)

                if self.adv_weighted_loss:
                    if self.bc_augmentation:
                        actor_loss = self.bc_coef * aug_gcbc_loss + (1 - self.bc_coef) * adv_weighted_loss
                    else:
                        actor_loss = self.bc_coef * gcbc_loss + (1 - self.bc_coef) * adv_weighted_loss
                elif self.actor_q_loss:
                    if self.bc_augmentation:
                        actor_loss = self.bc_coef * aug_gcbc_loss + (1 - self.bc_coef) * actor_q_loss
                    else:
                        actor_loss = self.bc_coef * gcbc_loss + (1 - self.bc_coef) * actor_q_loss

                metric_gcbc_loss = torch.sum(gcbc_loss) / torch.sum(train_mask)
                metric_aug_gcbc_loss = torch.sum(aug_gcbc_loss) / torch.sum(train_mask)
                if torch.sum(1 - train_mask) > 0:
                    metric_gcbc_val_loss = torch.sum(gcbc_val_loss) / torch.sum(1 - train_mask)
                    metric_aug_gcbc_val_loss = torch.sum(aug_gcbc_val_loss) / torch.sum(1 - train_mask)
                else:
                    metric_gcbc_val_loss = ptu.zeros(1)
                    metric_aug_gcbc_val_loss = ptu.zeros(1)

        if self.adv_weighted_loss:
            actor_q_loss = ptu.zeros(1)
        if self.actor_q_loss:
            adv_weighted_loss = ptu.zeros(1)
        if self.bc_coef == 0.0:
            metric_gcbc_loss = ptu.zeros(1)
            metric_gcbc_val_loss = ptu.zeros(1)
            metric_aug_gcbc_loss = ptu.zeros(1)
            metric_aug_gcbc_val_loss = ptu.zeros(1)

        actor_loss = torch.mean(actor_loss)

        """
        Behavioral Cloning Loss
        """
        # def behavioral_cloning_loss(behavioral_cloning_policy_params: networks_lib.Params,
        #                             transitions: types.Transition,
        #                             key: networks_lib.PRNGKey):
        #     del key
        #     dist_params = networks.behavioral_cloning_policy_network.apply(
        #         behavioral_cloning_policy_params,
        #         transitions.observation[:, :self._obs_dim])
        #     log_prob = networks.log_prob(dist_params, transitions.action)
        #     bc_loss = -1.0 * jnp.mean(log_prob)
        #
        #     return bc_loss

        # FIXME (chongyiz)
        # bc_dist = self.behavioral_cloning_policy()
        # bc_log_prob = bc_dist.log_prob(action)
        # bc_loss = -1.0 * torch.mean(bc_log_prob)

        metric_bc_loss = ptu.zeros(1)
        metric_bc_val_loss = ptu.zeros(1)

        # loss = alpha_loss + actor_loss + qf_loss

        # DELETEME (chongyiz)
        # if torch.isnan(alpha_loss) or torch.isnan(actor_loss) \
        #         or torch.isnan(qf_loss):
        #     print()

        # """
        # VIB
        # """
        # if self.train_encoder:
        #     kld = - 0.5 * torch.sum(
        #         1 + obs_feat_logvar - obs_feat_mu.pow(2)
        #         - obs_feat_logvar.exp(),
        #         dim=-1)
        #     kld = torch.mean(kld)
        #     vib_loss = kld * self.kld_weight  # (chongyiz): for statistics log
        #     loss += vib_loss

        # """
        # MSE Loss
        # """
        # with torch.no_grad():
        #     mse_loss = (dist.mean - action) ** 2

        if train:
            """
            Optimization.
            """
            if self.n_train_steps_total % self.update_period == 0:
                # self.behavioral_cloning_policy_optimizer.zero_grad()
                # bc_loss.backward()
                # if (self.gradient_clipping is not None and
                #         self.gradient_clipping > 0):
                #     torch.nn.utils.clip_grad_norm(
                #         self.policy.parameters(), self.gradient_clipping)
                # self.behavioral_cloning_policy_optimizer.step()

                # self.optimizer.zero_grad()
                # loss.backward()
                #
                # if (self.gradient_clipping is not None and
                #         self.gradient_clipping > 0):
                #     torch.nn.utils.clip_grad_norm(
                #         self.parameters, self.gradient_clipping)
                #
                # self.optimizer.step()

                if not self.use_gcbc and self.adaptive_entropy_coefficient:
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

                self.vf_optimizer.zero_grad()
                vf_loss.backward()
                if (self.gradient_clipping is not None and
                        self.gradient_clipping > 0):
                    torch.nn.utils.clip_grad_norm(
                        self.vf.parameters(), self.gradient_clipping)
                self.vf_optimizer.step()

                #
                # if self.train_encoder:
                #     self.obs_encoder_optimizer.zero_grad()
                #     vib_loss.backward()
                #     if (self.gradient_clipping is not None and
                #             self.gradient_clipping > 0):
                #         torch.nn.utils.clip_grad_norm(
                #             self.obs_encoder.parameters(), self.gradient_clipping)
                #     self.obs_encoder_optimizer.step()

                # self.optimizer.zero_grad()
                # loss.backward()
                #
                # if (self.gradient_clipping is not None and
                #         self.gradient_clipping > 0):
                #     torch.nn.utils.clip_grad_norm(
                #         self.parameters, self.gradient_clipping)
                #
                # self.optimizer.step()

            """
            Soft Updates
            """
            if self.n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.qf, self.target_qf, self.soft_target_tau
                )
                # ptu.soft_update_from_to(
                #     self.qf2, self.target_qf2, self.soft_target_tau
                # )
                ptu.soft_update_from_to(
                    self.vf, self.target_vf, self.soft_target_tau
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

            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'obs_feat',
            #     ptu.get_numpy(obs_feat),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'obs_feat_mu',
            #     ptu.get_numpy(obs_feat_mu),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'obs_feat_logvar',
            #     ptu.get_numpy(obs_feat_logvar),
            # ))

            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'goal_feat',
            #     ptu.get_numpy(goal_feat),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'goal_feat_mu',
            #     ptu.get_numpy(goal_feat_mu),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'goal_feat_logvar',
            #     ptu.get_numpy(goal_feat_logvar),
            # ))

            # self.eval_statistics[prefix + 'QF1 Loss'] = np.mean(
            #     ptu.get_numpy(qf1_loss))
            # self.eval_statistics[prefix + 'QF2 Loss'] = np.mean(
            #     ptu.get_numpy(qf2_loss))
            self.eval_statistics[prefix + 'QF Loss'] = np.mean(
                ptu.get_numpy(qf_loss))
            self.eval_statistics[prefix + 'VF Loss'] = np.mean(
                ptu.get_numpy(vf_loss))
            self.eval_statistics[prefix + 'Policy Loss'] = np.mean(
                ptu.get_numpy(actor_loss))
            self.eval_statistics[prefix + 'Policy Loss/Advantage Weighted Loss'] = np.mean(
                ptu.get_numpy(adv_weighted_loss))
            self.eval_statistics[prefix + 'Policy Loss/Actor Q Loss'] = np.mean(
                ptu.get_numpy(actor_q_loss))
            self.eval_statistics[prefix + 'Policy Loss/GCBC Loss'] = np.mean(
                ptu.get_numpy(metric_gcbc_loss))
            self.eval_statistics[prefix + 'Policy Loss/GCBC Val Loss'] = np.mean(
                ptu.get_numpy(metric_gcbc_val_loss))
            self.eval_statistics[prefix + 'Policy Loss/Augmented GCBC Loss'] = np.mean(
                ptu.get_numpy(metric_aug_gcbc_loss))
            self.eval_statistics[prefix + 'Policy Loss/Augmented GCBC Val Loss'] = np.mean(
                ptu.get_numpy(metric_aug_gcbc_val_loss))
            self.eval_statistics[prefix + 'Behavioral Cloning Loss'] = np.mean(
                ptu.get_numpy(metric_bc_loss))
            self.eval_statistics[prefix + 'Behavioral Cloning Val Loss'] = np.mean(
                ptu.get_numpy(metric_bc_val_loss))

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
            # if self.use_td:
            #     self.eval_statistics.update(create_stats_ordered_dict(
            #         prefix + 'qf/target_sa_repr_norm',
            #         ptu.get_numpy(target_sa_repr_norm),  # NOQA
            #     ))
            #     self.eval_statistics.update(create_stats_ordered_dict(
            #         prefix + 'qf/target_g_repr_norm',
            #         ptu.get_numpy(target_g_repr_norm),  # NOQA
            #     ))

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
            self.eval_statistics[prefix + 'qf/alignment'] = np.mean(
                ptu.get_numpy(align_loss))
            self.eval_statistics[prefix + 'qf/uniformity'] = np.mean(
                ptu.get_numpy(unif_loss))

            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Q1 Predictions',
            #     ptu.get_numpy(q1_pred),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Q2 Predictions',
            #     ptu.get_numpy(q2_pred),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Q Targets',
            #     ptu.get_numpy(q_target),
            # ))
            # (chongyiz): Contrastive RL statistics
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Positive Logits1',
            #     ptu.get_numpy(pos_logits1),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Positive Logits2',
            #     ptu.get_numpy(pos_logits2),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Negative Logits1',
            #     ptu.get_numpy(neg_logits1),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Negative Logits2',
            #     ptu.get_numpy(neg_logits2),
            # ))
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

            # DELEME (chongyiz)
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Advantage Weights',
            #     ptu.get_numpy(adv_weight),))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Advantage Score',
            #     ptu.get_numpy(adv),))
            #
            # self.eval_statistics[prefix + 'VF Loss'] = np.mean(
            #     ptu.get_numpy(vf_loss))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'V1 Predictions',
            #     ptu.get_numpy(vf_pred),))
            # if self.train_encoder:
            #     self.eval_statistics[prefix + 'Plan VF Loss'] = np.mean(
            #         ptu.get_numpy(plan_vf_loss))
            #     self.eval_statistics.update(create_stats_ordered_dict(
            #         prefix + 'Plan V1 Predictions',
            #         ptu.get_numpy(plan_vf_extra['vf_pred']),))

            if not self.use_gcbc:
                if self.entropy_coefficient is not None:
                    self.eval_statistics[prefix + 'alpha'] = alpha
                else:
                    self.eval_statistics[prefix + 'alpha'] = np.mean(
                        ptu.get_numpy(alpha))
                if self.adaptive_entropy_coefficient:
                    self.eval_statistics[prefix + 'Alpha Loss'] = np.mean(
                        ptu.get_numpy(alpha_loss))

            # self.eval_statistics[prefix + 'beta'] = self.beta
            # self.eval_statistics[prefix + 'quantile'] = self.quantile

            # DELETEME (chongyiz)
            # if self.train_encoder:
            #     self.eval_statistics[prefix + 'KLD'] = np.mean(
            #         ptu.get_numpy(kld))
            #     self.eval_statistics[prefix + 'VIB Loss'] = np.mean(
            #         ptu.get_numpy(vib_loss))

            # DELETEME (chongyiz)
            # # Affordance
            # self.eval_statistics[prefix + 'Affordance Loss'] = np.mean(
            #     ptu.get_numpy(affordance_loss))
            # self.eval_statistics[prefix + 'Affordance KLD'] = np.mean(
            #     ptu.get_numpy(affordance_extra['kld']))
            # self.eval_statistics[prefix + 'Affordance Pred Loss'] = np.mean(  # NOQA
            #     ptu.get_numpy(affordance_extra['loss_pred']))
            #
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Affordance Encoding',
            #     ptu.get_numpy(affordance_extra['u']),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Affordance Encoding Mean',
            #     ptu.get_numpy(affordance_extra['u_mu']),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     prefix + 'Affordance Encoding LogVar',
            #     ptu.get_numpy(affordance_extra['u_logvar']),
            # ))

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
            # self.behavioral_cloning_policy,
            # self.qf1,
            # self.qf2,
            # self.target_qf1,
            # self.target_qf2,
            self.qf,
            self.target_qf,
            self.vf,
            self.target_vf,
            # self.plan_vf,
            # self.obs_encoder,
            # self.affordance,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            # behavioral_cloning_policy=self.behavioral_cloning_policy,
            # qf1=self.qf1,
            # qf2=self.qf2,
            # target_qf1=self.target_qf1,
            # target_qf2=self.target_qf2,
            qf=self.qf,
            target_qf=self.target_qf,
            vf=self.vf,
            target_vf=self.target_vf,
            # plan_vf=self.plan_vf,
            # obs_encoder=self.obs_encoder,
            # affordance=self.affordance,
        )
