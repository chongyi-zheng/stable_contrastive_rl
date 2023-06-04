import copy
from functools import partial

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
)
from rlkit.envs.contextual import ContextualEnv, delete_info
from rlkit.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    AddImageDistribution,
    GoalConditionedDiagnosticsToContextualDiagnostics,
    IndexIntoAchievedGoal,
)
from rlkit.envs.contextual.mask_conditioned import (
    MaskedGoalDictDistributionFromMultitaskEnv,
    MaskPathCollector,
    default_masked_reward_fn,
)
from rlkit.envs.contextual.mask_inference import infer_masks as infer_masks_fn
from rlkit.envs.images import EnvRenderer, InsertImagesEnv
from rlkit.launchers.contextual.rig.rig_launcher import get_gym_env
from rlkit.launchers.contextual.util import (
    get_save_video_function,
)
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def representation_learning_with_goal_distribution_launcher(
        max_path_length,
        contextual_replay_buffer_kwargs,
        sac_trainer_kwargs,
        algo_kwargs,
        qf_kwargs=None,
        policy_kwargs=None,
        # env settings
        env_id=None,
        env_class=None,
        env_kwargs=None,
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        achieved_goal_key='latent_achieved_goal',
        renderer_kwargs=None,
        # mask settings
        mask_variant=None,  # TODO: manually unpack this as well
        mask_conditioned=True,
        mask_format='vector',
        infer_masks=False,
        # rollout
        expl_goal_sampling_mode=None,
        eval_goal_sampling_mode=None,
        eval_rollouts_for_videos=None,
        eval_rollouts_to_log=None,
        # debugging
        log_mask_diagnostics=True,
        log_expl_video=True,
        log_eval_video=True,
        save_video=True,
        save_video_period=50,
        save_env_in_snapshot=True,
        dump_video_kwargs=None,
        # re-loading
        ckpt=None,
        ckpt_epoch=None,
        seedid=0,
):
    if eval_rollouts_to_log is None:
        eval_rollouts_to_log = [
            'atomic',
            'atomic_seq',
            'cumul_seq',
            'full',
        ]
    if renderer_kwargs is None:
        renderer_kwargs = {}
    if dump_video_kwargs is None:
        dump_video_kwargs = {}
    if eval_rollouts_for_videos is None:
        eval_rollouts_for_videos = [
            'atomic',
            'atomic_seq',
            'cumul_seq',
            'full',
        ]
    if mask_variant is None:
        mask_variant = {}
    if policy_kwargs is None:
        policy_kwargs = {}
    if qf_kwargs is None:
        qf_kwargs = {}
    context_key = desired_goal_key
    prev_subtask_weight = mask_variant.get('prev_subtask_weight', None)

    context_post_process_mode = mask_variant.get('context_post_process_mode',
                                                 None)
    if context_post_process_mode in [
        'dilute_prev_subtasks_uniform', 'dilute_prev_subtasks_fixed'
    ]:
        prev_subtask_weight = 0.5
    prev_subtasks_solved = mask_variant.get('prev_subtasks_solved', False)
    max_subtasks_to_focus_on = mask_variant.get(
        'max_subtasks_to_focus_on', None)
    max_subtasks_per_rollout = mask_variant.get(
        'max_subtasks_per_rollout', None)
    mask_groups = mask_variant.get('mask_groups', None)
    rollout_mask_order_for_expl = mask_variant.get(
        'rollout_mask_order_for_expl', 'fixed')
    rollout_mask_order_for_eval = mask_variant.get(
        'rollout_mask_order_for_eval', 'fixed')
    masks = mask_variant.get('masks', None)
    idx_masks = mask_variant.get('idx_masks', None)
    matrix_masks = mask_variant.get('matrix_masks', None)
    train_mask_distr = mask_variant.get('train_mask_distr', None)
    mask_inference_variant = mask_variant.get('mask_inference_variant', {})
    mask_reward_fn = mask_variant.get('reward_fn', default_masked_reward_fn)
    expl_mask_distr = mask_variant['expl_mask_distr']
    eval_mask_distr = mask_variant['eval_mask_distr']
    use_g_for_mean = mask_variant['use_g_for_mean']
    context_post_process_frac = mask_variant.get(
        'context_post_process_frac', 0.50)
    sample_masks_for_relabeling = mask_variant.get(
        'sample_masks_for_relabeling', True)

    if mask_conditioned:
        env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        assert mask_format in ['vector', 'matrix', 'distribution']
        goal_dim = env.observation_space.spaces[context_key].low.size
        if mask_format == 'vector':
            mask_keys = ['mask']
            mask_dims = [(goal_dim,)]
            context_dim = goal_dim + goal_dim
        elif mask_format == 'matrix':
            mask_keys = ['mask']
            mask_dims = [(goal_dim, goal_dim)]
            context_dim = goal_dim + (goal_dim * goal_dim)
        elif mask_format == 'distribution':
            mask_keys = ['mask_mu_w', 'mask_mu_g', 'mask_mu_mat',
                         'mask_sigma_inv']
            mask_dims = [(goal_dim,), (goal_dim,), (goal_dim, goal_dim),
                         (goal_dim, goal_dim)]
            context_dim = goal_dim + (goal_dim * goal_dim)  # mu and sigma_inv
        else:
            raise NotImplementedError

        if infer_masks:
            assert mask_format == 'distribution'
            env_kwargs_copy = copy.deepcopy(env_kwargs)
            env_kwargs_copy['lite_reset'] = True
            infer_masks_env = get_gym_env(env_id, env_class=env_class,
                                          env_kwargs=env_kwargs_copy)

            masks = infer_masks_fn(
                infer_masks_env,
                idx_masks,
                mask_inference_variant,
            )

        context_keys = [context_key] + mask_keys
    else:
        context_keys = [context_key]

    def contextual_env_distrib_and_reward(mode='expl'):
        assert mode in ['expl', 'eval']
        env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)

        if mode == 'expl':
            goal_sampling_mode = expl_goal_sampling_mode
        elif mode == 'eval':
            goal_sampling_mode = eval_goal_sampling_mode
        else:
            goal_sampling_mode = None
        if goal_sampling_mode is not None:
            env.goal_sampling_mode = goal_sampling_mode

        if mask_conditioned:
            context_distrib = MaskedGoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
                mask_keys=mask_keys,
                mask_dims=mask_dims,
                mask_format=mask_format,
                max_subtasks_to_focus_on=max_subtasks_to_focus_on,
                prev_subtask_weight=prev_subtask_weight,
                masks=masks,
                idx_masks=idx_masks,
                matrix_masks=matrix_masks,
                mask_distr=train_mask_distr,
            )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(
                    achieved_goal_key),  # observation_key
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=contextual_replay_buffer_kwargs.get(
                    'observation_keys', None),
                additional_context_keys=mask_keys,
                reward_fn=partial(
                    mask_reward_fn,
                    mask_format=mask_format,
                    use_g_for_mean=use_g_for_mean
                ),
            )
        else:
            context_distrib = GoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
            )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(
                    achieved_goal_key),  # observation_key
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=contextual_replay_buffer_kwargs.get(
                    'observation_keys', None),
            )
        diag_fn = GoalConditionedDiagnosticsToContextualDiagnostics(
            env.goal_conditioned_diagnostics,
            desired_goal_key=desired_goal_key,
            observation_key=observation_key,
        )
        env = ContextualEnv(
            env,
            context_distribution=context_distrib,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=[diag_fn],
            update_env_info_fn=delete_info,
        )
        return env, context_distrib, reward_fn

    env, context_distrib, reward_fn = contextual_env_distrib_and_reward(
        mode='expl')
    eval_env, eval_context_distrib, _ = contextual_env_distrib_and_reward(
        mode='eval')

    if mask_conditioned:
        obs_dim = (
                env.observation_space.spaces[observation_key].low.size
                + context_dim
        )
    else:
        obs_dim = (
                env.observation_space.spaces[observation_key].low.size
                + env.observation_space.spaces[context_key].low.size
        )

    action_dim = env.action_space.low.size

    if ckpt:
        from rlkit.util.io import local_path_from_s3_or_local_path
        import os.path as osp

        if ckpt_epoch is not None:
            epoch = ckpt_epoch
            filename = local_path_from_s3_or_local_path(
                osp.join(ckpt, 'itr_%d.pkl' % epoch))
        else:
            filename = local_path_from_s3_or_local_path(
                osp.join(ckpt, 'params.pkl'))
        print("Loading ckpt from", filename)
        # data = joblib.load(filename)
        data = torch.load(filename, map_location='cuda:1')
        qf1 = data['trainer/qf1']
        qf2 = data['trainer/qf2']
        target_qf1 = data['trainer/target_qf1']
        target_qf2 = data['trainer/target_qf2']
        policy = data['trainer/policy']
        eval_policy = data['evaluation/policy']
        expl_policy = data['exploration/policy']
    else:
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
        expl_policy = policy
        eval_policy = MakeDeterministic(policy)

    def context_from_obs_dict_fn(obs_dict):
        context_dict = {
            context_key: obs_dict[achieved_goal_key],  # observation_key
        }
        if mask_conditioned:
            if sample_masks_for_relabeling:
                batch_size = obs_dict[list(obs_dict.keys())[0]].shape[0]
                sampled_contexts = context_distrib.sample(batch_size)
                for mask_key in mask_keys:
                    context_dict[mask_key] = sampled_contexts[mask_key]
            else:
                for mask_key in mask_keys:
                    context_dict[mask_key] = obs_dict[mask_key]
        return context_dict

    def post_process_mask_fn(obs_dict, context_dict):
        assert mask_conditioned
        pp_context_dict = copy.deepcopy(context_dict)

        assert context_post_process_mode in [
            'prev_subtasks_solved',
            'dilute_prev_subtasks_uniform',
            'dilute_prev_subtasks_fixed',
            'atomic_to_corresp_cumul',
            None
        ]

        if context_post_process_mode in [
            'prev_subtasks_solved',
            'dilute_prev_subtasks_uniform',
            'dilute_prev_subtasks_fixed',
            'atomic_to_corresp_cumul'
        ]:
            frac = context_post_process_frac
            cumul_mask_to_indices = context_distrib.get_cumul_mask_to_indices(
                context_dict['mask']
            )
            for k in cumul_mask_to_indices:
                indices = cumul_mask_to_indices[k]
                subset = np.random.choice(len(indices),
                                          int(len(indices) * frac),
                                          replace=False)
                cumul_mask_to_indices[k] = indices[subset]
        else:
            cumul_mask_to_indices = None

        mode = context_post_process_mode
        if mode in [
            'prev_subtasks_solved', 'dilute_prev_subtasks_uniform',
            'dilute_prev_subtasks_fixed'
        ]:
            cumul_masks = list(cumul_mask_to_indices.keys())
            for i in range(1, len(cumul_masks)):
                curr_mask = cumul_masks[i]
                prev_mask = cumul_masks[i - 1]
                prev_obj_indices = np.where(np.array(prev_mask) > 0)[0]
                indices = cumul_mask_to_indices[curr_mask]
                if mode == 'prev_subtasks_solved':
                    pp_context_dict[context_key][indices][:, prev_obj_indices] = \
                        obs_dict[achieved_goal_key][indices][:,
                        prev_obj_indices]
                elif mode == 'dilute_prev_subtasks_uniform':
                    pp_context_dict['mask'][indices][:, prev_obj_indices] = \
                        np.random.uniform(
                            size=(len(indices), len(prev_obj_indices)))
                elif mode == 'dilute_prev_subtasks_fixed':
                    pp_context_dict['mask'][indices][:, prev_obj_indices] = 0.5
            indices_to_relabel = np.concatenate(
                list(cumul_mask_to_indices.values()))
            orig_masks = obs_dict['mask'][indices_to_relabel]
            atomic_mask_to_subindices = context_distrib.get_atomic_mask_to_indices(
                orig_masks)
            atomic_masks = list(atomic_mask_to_subindices.keys())
            cumul_masks = list(cumul_mask_to_indices.keys())
            for i in range(1, len(atomic_masks)):
                orig_atomic_mask = atomic_masks[i]
                relabeled_cumul_mask = cumul_masks[i]
                subindices = atomic_mask_to_subindices[orig_atomic_mask]
                pp_context_dict['mask'][indices_to_relabel][
                    subindices] = relabeled_cumul_mask

        return pp_context_dict

    # if mask_conditioned:
    #     variant['contextual_replay_buffer_kwargs']['post_process_batch_fn'] = post_process_mask_fn

    def concat_context_to_obs(batch, replay_buffer=None, obs_dict=None,
                              next_obs_dict=None, new_contexts=None):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
        if mask_conditioned:
            if obs_dict is not None and new_contexts is not None:
                updated_contexts = post_process_mask_fn(obs_dict, new_contexts)
                batch.update(updated_contexts)

            if mask_format in ['vector', 'matrix']:
                assert len(mask_keys) == 1
                mask = batch[mask_keys[0]].reshape((len(context), -1))
                batch['observations'] = np.concatenate([obs, context, mask],
                                                       axis=1)
                batch['next_observations'] = np.concatenate(
                    [next_obs, context, mask], axis=1)
            elif mask_format == 'distribution':
                g = context
                mu_w = batch['mask_mu_w']
                mu_g = batch['mask_mu_g']
                mu_A = batch['mask_mu_mat']
                sigma_inv = batch['mask_sigma_inv']
                if use_g_for_mean:
                    mu_w_given_g = g
                else:
                    mu_w_given_g = mu_w + np.squeeze(
                        mu_A @ np.expand_dims(g - mu_g, axis=-1), axis=-1)
                sigma_w_given_g_inv = sigma_inv.reshape((len(context), -1))
                batch['observations'] = np.concatenate(
                    [obs, mu_w_given_g, sigma_w_given_g_inv], axis=1)
                batch['next_observations'] = np.concatenate(
                    [next_obs, mu_w_given_g, sigma_w_given_g_inv], axis=1)
            else:
                raise NotImplementedError
        else:
            batch['observations'] = np.concatenate([obs, context], axis=1)
            batch['next_observations'] = np.concatenate([next_obs, context],
                                                        axis=1)
        return batch

    if 'observation_keys' not in contextual_replay_buffer_kwargs:
        contextual_replay_buffer_kwargs['observation_keys'] = []
    observation_keys = contextual_replay_buffer_kwargs['observation_keys']
    if observation_key not in observation_keys:
        observation_keys.append(observation_key)
    if achieved_goal_key not in observation_keys:
        observation_keys.append(achieved_goal_key)

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=env,
        context_keys=context_keys,
        context_distribution=context_distrib,
        sample_context_from_obs_dict_fn=context_from_obs_dict_fn,
        reward_fn=reward_fn,
        post_process_batch_fn=concat_context_to_obs,
        **contextual_replay_buffer_kwargs
    )

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs
    )

    def create_path_collector(
            env,
            policy,
            mode='expl',
            mask_kwargs=None,
    ):
        if mask_kwargs is None:
            mask_kwargs = {}
        assert mode in ['expl', 'eval']
        if mask_conditioned:
            if 'rollout_mask_order' in mask_kwargs:
                rollout_mask_order = mask_kwargs['rollout_mask_order']
            else:
                if mode == 'expl':
                    rollout_mask_order = rollout_mask_order_for_expl
                elif mode == 'eval':
                    rollout_mask_order = rollout_mask_order_for_eval
                else:
                    raise NotImplementedError

            if 'mask_distr' in mask_kwargs:
                mask_distr = mask_kwargs['mask_distr']
            else:
                if mode == 'expl':
                    mask_distr = expl_mask_distr
                elif mode == 'eval':
                    mask_distr = eval_mask_distr
                else:
                    raise NotImplementedError

            return MaskPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
                concat_context_to_obs_fn=concat_context_to_obs,
                save_env_in_snapshot=save_env_in_snapshot,
                mask_sampler=(
                    context_distrib if mode == 'expl' else eval_context_distrib),
                mask_distr=mask_distr.copy(),
                mask_groups=mask_groups,
                max_path_length=max_path_length,
                rollout_mask_order=rollout_mask_order,
                prev_subtask_weight=prev_subtask_weight,
                prev_subtasks_solved=prev_subtasks_solved,
                max_subtasks_to_focus_on=max_subtasks_to_focus_on,
                max_subtasks_per_rollout=max_subtasks_per_rollout,
            )
        else:
            return ContextualPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
                save_env_in_snapshot=save_env_in_snapshot,
            )

    expl_path_collector = create_path_collector(env, expl_policy, mode='expl')
    eval_path_collector = create_path_collector(eval_env, eval_policy,
                                                mode='eval')

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs
    )

    algorithm.to(ptu.device)

    if save_video:
        renderer = EnvRenderer(**renderer_kwargs)

        def add_images(env, state_distribution):
            state_env = env.env
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_distribution,
                image_goal_key='image_desired_goal',
                renderer=renderer,
            )
            img_env = InsertImagesEnv(state_env, renderers={
                'image_observation': renderer,
            })
            context_env = ContextualEnv(
                img_env,
                context_distribution=image_goal_distribution,
                reward_fn=reward_fn,
                observation_key=observation_key,
                update_env_info_fn=delete_info,
            )
            return context_env

        img_eval_env = add_images(eval_env, eval_context_distrib)

        if log_eval_video:
            video_path_collector = create_path_collector(img_eval_env,
                                                         eval_policy,
                                                         mode='eval')
            rollout_function = video_path_collector._rollout_fn
            eval_video_func = get_save_video_function(
                rollout_function,
                img_eval_env,
                eval_policy,
                tag="eval",
                imsize=renderer_kwargs['width'],
                image_format='CHW',
                save_video_period=save_video_period,
                horizon=max_path_length,
                **dump_video_kwargs
            )
            algorithm.post_train_funcs.append(eval_video_func)

        # additional eval videos for mask conditioned case
        if mask_conditioned:
            if 'cumul_seq' in eval_rollouts_for_videos:
                video_path_collector = create_path_collector(
                    img_eval_env,
                    eval_policy,
                    mode='eval',
                    mask_kwargs=dict(
                        mask_distr=dict(
                            cumul_seq=1.0
                        ),
                    ),
                )
                rollout_function = video_path_collector._rollout_fn
                eval_video_func = get_save_video_function(
                    rollout_function,
                    img_eval_env,
                    eval_policy,
                    tag="eval_cumul" if mask_conditioned else "eval",
                    imsize=renderer_kwargs['width'],
                    image_format='HWC',
                    save_video_period=save_video_period,
                    horizon=max_path_length,
                    **dump_video_kwargs
                )
                algorithm.post_train_funcs.append(eval_video_func)

            if 'full' in eval_rollouts_for_videos:
                video_path_collector = create_path_collector(
                    img_eval_env,
                    eval_policy,
                    mode='eval',
                    mask_kwargs=dict(
                        mask_distr=dict(
                            full=1.0
                        ),
                    ),
                )
                rollout_function = video_path_collector._rollout_fn
                eval_video_func = get_save_video_function(
                    rollout_function,
                    img_eval_env,
                    eval_policy,
                    tag="eval_full",
                    imsize=renderer_kwargs['width'],
                    image_format='HWC',
                    save_video_period=save_video_period,
                    horizon=max_path_length,
                    **dump_video_kwargs
                )
                algorithm.post_train_funcs.append(eval_video_func)

            if 'atomic_seq' in eval_rollouts_for_videos:
                video_path_collector = create_path_collector(
                    img_eval_env,
                    eval_policy,
                    mode='eval',
                    mask_kwargs=dict(
                        mask_distr=dict(
                            atomic_seq=1.0
                        ),
                    ),
                )
                rollout_function = video_path_collector._rollout_fn
                eval_video_func = get_save_video_function(
                    rollout_function,
                    img_eval_env,
                    eval_policy,
                    tag="eval_atomic",
                    imsize=renderer_kwargs['width'],
                    image_format='HWC',
                    save_video_period=save_video_period,
                    horizon=max_path_length,
                    **dump_video_kwargs
                )
                algorithm.post_train_funcs.append(eval_video_func)

        if log_expl_video:
            img_expl_env = add_images(env, context_distrib)
            video_path_collector = create_path_collector(img_expl_env,
                                                         expl_policy,
                                                         mode='expl')
            rollout_function = video_path_collector._rollout_fn
            expl_video_func = get_save_video_function(
                rollout_function,
                img_expl_env,
                expl_policy,
                tag="expl",
                imsize=renderer_kwargs['width'],
                image_format='CHW',
                save_video_period=save_video_period,
                horizon=max_path_length,
                **dump_video_kwargs
            )
            algorithm.post_train_funcs.append(expl_video_func)

    if mask_conditioned and log_mask_diagnostics:
        collectors = []
        log_prefixes = []

        default_list = [
            'atomic',
            'atomic_seq',
            'cumul_seq',
            'full',
        ]
        for key in eval_rollouts_to_log:
            assert key in default_list

        if 'atomic' in eval_rollouts_to_log:
            num_masks = len(eval_path_collector.mask_groups)
            for mask_id in range(num_masks):
                mask_kwargs = dict(
                    rollout_mask_order=[mask_id],
                    mask_distr=dict(
                        atomic_seq=1.0,
                    ),
                )
                collector = create_path_collector(eval_env, eval_policy,
                                                  mode='eval',
                                                  mask_kwargs=mask_kwargs)
                collectors.append(collector)
            log_prefixes += [
                'mask_{}/'.format(''.join(str(mask_id)))
                for mask_id in range(num_masks)
            ]

        # full mask
        if 'full' in eval_rollouts_to_log:
            mask_kwargs = dict(
                mask_distr=dict(
                    full=1.0,
                ),
            )
            collector = create_path_collector(eval_env, eval_policy,
                                              mode='eval',
                                              mask_kwargs=mask_kwargs)
            collectors.append(collector)
            log_prefixes.append('mask_full/')

        # cumulative, sequential mask
        if 'cumul_seq' in eval_rollouts_to_log:
            mask_kwargs = dict(
                rollout_mask_order='fixed',
                mask_distr=dict(
                    cumul_seq=1.0,
                ),
            )
            collector = create_path_collector(eval_env, eval_policy,
                                              mode='eval',
                                              mask_kwargs=mask_kwargs)
            collectors.append(collector)
            log_prefixes.append('mask_cumul_seq/')

        # atomic, sequential mask
        if 'atomic_seq' in eval_rollouts_to_log:
            mask_kwargs = dict(
                rollout_mask_order='fixed',
                mask_distr=dict(
                    atomic_seq=1.0,
                ),
            )
            collector = create_path_collector(eval_env, eval_policy,
                                              mode='eval',
                                              mask_kwargs=mask_kwargs)
            collectors.append(collector)
            log_prefixes.append('mask_atomic_seq/')

        def get_mask_diagnostics(unused):
            from rlkit.core.logging import append_log, add_prefix, OrderedDict
            log = OrderedDict()
            for prefix, collector in zip(log_prefixes, collectors):
                paths = collector.collect_new_paths(
                    max_path_length,
                    max_path_length,  # masking_eval_steps,
                    discard_incomplete_paths=True,
                )
                # old_path_info = eval_util.get_generic_path_information(paths)
                old_path_info = eval_env.get_diagnostics(paths)

                keys_to_keep = []
                for key in old_path_info.keys():
                    if ('env_infos' in key) and ('final' in key) and (
                            'Mean' in key):
                        keys_to_keep.append(key)
                path_info = OrderedDict()
                for key in keys_to_keep:
                    path_info[key] = old_path_info[key]

                generic_info = add_prefix(
                    path_info,
                    prefix,
                )
                append_log(log, generic_info)

            for collector in collectors:
                collector.end_epoch(0)
            return log

        algorithm._eval_get_diag_fns.append(get_mask_diagnostics)
    algorithm.train()
