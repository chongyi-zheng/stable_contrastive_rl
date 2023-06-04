from functools import partial
import os.path as osp

from gym.spaces import Box
import joblib
import numpy as np

import rlkit.samplers.rollout_functions as rf
import rlkit.torch.pytorch_util as ptu
from rlkit.core.distribution import DictDistribution
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from rlkit.envs.contextual import ContextualEnv
from rlkit.envs.contextual.goal_conditioned import (
    GoalConditionedDiagnosticsToContextualDiagnostics,
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    AddImageDistribution,
    IndexIntoAchievedGoal,
)
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector,
)
from rlkit.visualization.video import dump_video
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.core import logger, eval_util

from rlkit.launchers.contextual.util import (
    get_gym_env,
    get_save_video_function,
)

from rlkit.samplers.rollout_functions import (
    rollout,
)
from rlkit.samplers.rollout_functions import contextual_rollout


def default_masked_reward_fn(obs_dict, actions, next_obs_dict, new_contexts,
                             context_key, mask_key):
    achieved_goals = next_obs_dict['state_achieved_goal']
    desired_goals = new_contexts[context_key]
    mask = new_contexts[mask_key]
    rewards = -np.linalg.norm(mask * (achieved_goals - desired_goals),
                              axis=1)
    return rewards


def one_hot_mask(mask_length, mask_idx):
    mask = np.zeros(mask_length)
    mask[mask_idx] = 1
    return mask

class MaskedGoalDictDistribution(DictDistribution):
    def __init__(
            self,
            dict_distribution: DictDistribution,
            mask_dim=3,
            mask_key='masked_desired_goal',
            distribution_type='random_bit_masks',
            static_mask=None,
    ):
        self.mask_key = mask_key
        self._dict_distribution = dict_distribution
        self._spaces = dict_distribution.spaces

        self._spaces[mask_key] = Box(
            low=np.zeros(mask_dim),
            high=np.ones(mask_dim))
        self.mask_dim = mask_dim
        self.static_mask = static_mask
        self.distribution_type = distribution_type
        assert self.distribution_type in [
            'one_hot_masks', 'random_bit_masks', 'static_mask']
        if self.distribution_type == 'static_mask':
            assert static_mask is not None
        else:
            assert static_mask is None
        self.one_hot_masks = np.array([
            one_hot_mask(mask_dim, mask_idx)
            for mask_idx in range(mask_dim)
        ])


    def sample(self, batch_size: int):
        goals = self._dict_distribution.sample(batch_size)
        if self.distribution_type == 'static_mask':
            goals[self.mask_key] = np.tile(self.static_mask, (batch_size, 1))
        elif self.distribution_type == 'one_hot_masks':
            goals[self.mask_key] = self.one_hot_masks[
                np.random.choice(self.mask_dim, size=batch_size)]
        elif self.distribution_type == 'random_bit_masks':
            goals[self.mask_key] = np.random.choice(
                2, size=(batch_size, self.mask_dim)
            )
        else:
            raise RuntimeError('Invalid distribution type')
        return goals

    @property
    def spaces(self):
        return self._spaces


def generate_revolving_masks(num_masks_to_generate, mask_length,
                             num_steps_per_mask_change):
    num_mask_changes = np.ceil(
        num_masks_to_generate / num_steps_per_mask_change).astype(int)
    if num_mask_changes < mask_length:
        raise AssertionError("Not enough cycles to go through all masks")

    masks = [
        one_hot_mask(mask_length, mask_idx)
        for mask_idx in range(mask_length)
    ]

    rollout_masks = []

    mask_permutation = np.random.permutation(masks)
    for _ in range(num_mask_changes):
        if len(mask_permutation) == 0:
            mask_permutation = np.random.permutation(masks)
        cur_mask = mask_permutation[0]
        for mask in range(num_steps_per_mask_change):
            rollout_masks.append(cur_mask)
        mask_permutation = mask_permutation[1:]
    return rollout_masks


class RotatingMaskingPathCollector(ContextualPathCollector):
    """Changes the one hot masks every num_steps_per_mask_change during a
    rollout. All one hot masks are seen before reuse.
    """
    def __init__(
            self,
            *args,
            mask_key=None,
            mask_length=4,
            num_steps_per_mask_change=None,
            **kwargs
    ):
        self.rollout_masks = []
        super().__init__(*args, **kwargs)

        def obs_processor(o):
            if len(self.rollout_masks) == 0:
                self.rollout_masks = generate_revolving_masks(
                    mask_length * num_steps_per_mask_change, mask_length,
                    num_steps_per_mask_change)

            mask = self.rollout_masks[0]
            self.rollout_masks = self.rollout_masks[1:]
            o[mask_key] = mask

            combined_obs = [o[self._observation_key].flatten()]
            for k in self._context_keys_for_policy:
                combined_obs.append(o[k].flatten())
            return np.concatenate(combined_obs, axis=0)

        self._rollout_fn = partial(
            contextual_rollout,
            context_keys_for_policy=self._context_keys_for_policy,
            observation_key=self._observation_key,
            obs_processor=obs_processor,
        )


def masking_sac_experiment(
        max_path_length,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        env_class=None,
        env_kwargs=None,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
        train_env_id=None,
        eval_env_id=None,

        do_masking=True,
        mask_key="masked_observation",
        masking_eval_steps=200,
        log_mask_diagnostics=True,
        mask_dim=None,
        masking_reward_fn=None,
        masking_for_exploration=True,
        rotate_masks_for_eval=False,
        rotate_masks_for_expl=False,

        mask_distribution=None,
        num_steps_per_mask_change=10,

        tag=None,
):
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}

    context_key = desired_goal_key
    env = get_gym_env(train_env_id, env_class=env_class, env_kwargs=env_kwargs)
    mask_dim = (
        mask_dim or
        env.observation_space.spaces[context_key].low.size
    )

    if not do_masking:
        mask_distribution = 'all_ones'

    assert mask_distribution in ['one_hot_masks', 'random_bit_masks',
                                 'all_ones']
    if mask_distribution == 'all_ones':
        mask_distribution = 'static_mask'

    def contextual_env_distrib_and_reward(
            env_id, env_class, env_kwargs, goal_sampling_mode,
            env_mask_distribution_type,
            static_mask=None,
    ):
        env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        env.goal_sampling_mode = goal_sampling_mode
        use_static_mask = (
            not do_masking or
            env_mask_distribution_type == 'static_mask'
        )
        # Default to all ones mask if static mask isn't defined and should
        # be using static masks.
        if use_static_mask and static_mask is None:
            static_mask = np.ones(mask_dim)
        if not do_masking:
            assert env_mask_distribution_type == 'static_mask'

        goal_distribution = GoalDictDistributionFromMultitaskEnv(
            env,
            desired_goal_keys=[desired_goal_key],
        )
        goal_distribution = MaskedGoalDictDistribution(
            goal_distribution,
            mask_key=mask_key,
            mask_dim=mask_dim,
            distribution_type=env_mask_distribution_type,
            static_mask=static_mask,
        )

        reward_fn = ContextualRewardFnFromMultitaskEnv(
            env=env,
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
            achieved_goal_from_observation=IndexIntoAchievedGoal(
                observation_key
            ),
        )

        if do_masking:
            if masking_reward_fn:
                reward_fn = partial(masking_reward_fn, context_key=context_key,
                                    mask_key=mask_key)
            else:
                reward_fn = partial(default_masked_reward_fn,
                                    context_key=context_key, mask_key=mask_key)

        state_diag_fn = GoalConditionedDiagnosticsToContextualDiagnostics(
            env.goal_conditioned_diagnostics,
            desired_goal_key=desired_goal_key,
            observation_key=observation_key,
        )

        env = ContextualEnv(
            env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=[state_diag_fn],
        )
        return env, goal_distribution, reward_fn


    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        train_env_id, env_class, env_kwargs, exploration_goal_sampling_mode,
        mask_distribution if masking_for_exploration else 'static_mask',
    )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        eval_env_id, env_class, env_kwargs, evaluation_goal_sampling_mode,
        'static_mask'
    )

    # Distribution for relabeling
    relabel_context_distrib = GoalDictDistributionFromMultitaskEnv(
        env,
        desired_goal_keys=[desired_goal_key],
    )
    relabel_context_distrib = MaskedGoalDictDistribution(
        relabel_context_distrib,
        mask_key=mask_key,
        mask_dim=mask_dim,
        distribution_type=mask_distribution,
        static_mask=None if do_masking else np.ones(mask_dim),
    )

    obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[context_key].low.size
            + mask_dim
    )
    action_dim = expl_env.action_space.low.size

    def create_qf():
        return ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )
    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **policy_kwargs
    )

    def context_from_obs_dict_fn(obs_dict):
        achieved_goal = obs_dict['state_achieved_goal']
        # Should the mask be randomized for future relabeling?
        # batch_size = len(achieved_goal)
        # mask = np.random.choice(2, size=(batch_size, mask_dim))
        mask = obs_dict[mask_key]
        return {
            mask_key: mask,
            context_key: achieved_goal,
        }

    def concat_context_to_obs(batch, *args, **kwargs):
        obs = batch['observations']
        next_obs = batch['next_observations']

        context = batch[context_key]
        mask = batch[mask_key]

        batch['observations'] = np.concatenate(
            [obs, context, mask], axis=1)
        batch['next_observations'] = np.concatenate(
            [next_obs, context, mask], axis=1)
        return batch

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=[context_key, mask_key],
        observation_keys_to_save=[observation_key, 'state_achieved_goal'],
        context_distribution=relabel_context_distrib,
        sample_context_from_obs_dict_fn=context_from_obs_dict_fn,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs
    )
    trainer = SACTrainer(
        env=expl_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs
    )

    def create_path_collector(env, policy, is_rotating):
        if is_rotating:
            assert do_masking and mask_distribution == 'one_hot_masks'
            return RotatingMaskingPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=[context_key, mask_key],
                mask_key=mask_key,
                mask_length=mask_dim,
                num_steps_per_mask_change=num_steps_per_mask_change,
            )
        else:
            return ContextualPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=[context_key, mask_key],
            )

    exploration_policy = create_exploration_policy(
        expl_env, policy, **exploration_policy_kwargs)

    eval_path_collector = create_path_collector(
        eval_env, MakeDeterministic(policy), rotate_masks_for_eval)
    expl_path_collector = create_path_collector(
        expl_env, exploration_policy, rotate_masks_for_expl)

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs
    )
    algorithm.to(ptu.device)

    if save_video:
        rollout_function = partial(
            rf.contextual_rollout,
            max_path_length=max_path_length,
            observation_key=observation_key,
            context_keys_for_policy=[context_key, mask_key],
            # Eval on everything for the base video
            obs_processor=lambda o: np.hstack(
                (o[observation_key], o[context_key], np.ones(mask_dim)))
        )
        renderer = EnvRenderer(**renderer_kwargs)

        def add_images(env, state_distribution):
            state_env = env.env
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_distribution,
                image_goal_key='image_desired_goal',
                renderer=renderer,
            )
            img_env = InsertImageEnv(state_env, renderer=renderer)
            return ContextualEnv(
                img_env,
                context_distribution=image_goal_distribution,
                reward_fn=eval_reward,
                observation_key=observation_key,
                # update_env_info_fn=DeleteOldEnvInfo(),
            )
        img_eval_env = add_images(eval_env, eval_context_distrib)
        img_expl_env = add_images(expl_env, expl_context_distrib)
        eval_video_func = get_save_video_function(
            rollout_function,
            img_eval_env,
            MakeDeterministic(policy),
            tag="eval",
            imsize=renderer.image_shape[0],
            image_format='CWH',
            **save_video_kwargs
        )
        expl_video_func = get_save_video_function(
            rollout_function,
            img_expl_env,
            exploration_policy,
            tag="train",
            imsize=renderer.image_shape[0],
            image_format='CWH',
            **save_video_kwargs
        )

        algorithm.post_train_funcs.append(eval_video_func)
        algorithm.post_train_funcs.append(expl_video_func)

    # For diagnostics, evaluate the policy on each individual dimension of the
    # mask.
    masks = []
    collectors = []
    for mask_idx in range(mask_dim):
        mask = np.zeros(mask_dim)
        mask[mask_idx] = 1
        masks.append(mask)

    for mask in masks:
        for_dim_mask = mask if do_masking else np.ones(mask_dim)
        masked_env, _, _ = contextual_env_distrib_and_reward(
            eval_env_id, env_class, env_kwargs, evaluation_goal_sampling_mode,
            'static_mask', static_mask=for_dim_mask
        )

        collector = ContextualPathCollector(
            masked_env,
            MakeDeterministic(policy),
            observation_key=observation_key,
            context_keys_for_policy=[context_key, mask_key],
        )
        collectors.append(collector)
    log_prefixes = [
        'mask_{}/'.format(''.join(mask.astype(int).astype(str)))
        for mask in masks
    ]

    def get_mask_diagnostics(unused):
        from rlkit.core.logging import append_log, add_prefix, OrderedDict
        log = OrderedDict()
        for prefix, collector in zip(log_prefixes, collectors):
            paths = collector.collect_new_paths(
                max_path_length,
                masking_eval_steps,
                discard_incomplete_paths=True,
            )
            generic_info = add_prefix(
                eval_util.get_generic_path_information(paths),
                prefix,
            )
            append_log(log, generic_info)

        for collector in collectors:
            collector.end_epoch(0)
        return log
    if log_mask_diagnostics:
        algorithm._eval_get_diag_fns.append(get_mask_diagnostics)

    algorithm.train()
