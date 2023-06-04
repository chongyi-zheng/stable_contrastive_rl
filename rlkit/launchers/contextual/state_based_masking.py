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
    MaskDictDistribution,
    MaskPathCollector,
    ContextualMaskingRewardFn,
)
from rlkit.launchers.sets.mask_inference import get_mask_params
from rlkit.launchers.sets.example_set_gen import gen_example_sets

from rlkit.envs.images import EnvRenderer, InsertImagesEnv
from rlkit.launchers.contextual.util import (
    get_save_video_function,
)
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)

from rlkit.launchers.rl_exp_launcher_util import (
    preprocess_rl_variant,
    get_envs,
    create_exploration_policy,
)

import copy
import torch
from functools import partial
import numpy as np

def rl_context_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.td3.td3 import TD3 as TD3Trainer
    from rlkit.torch.sac.sac import SACTrainer
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.sac.policies import MakeDeterministic

    preprocess_rl_variant(variant)
    max_path_length = variant['max_path_length']
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = variant.get('achieved_goal_key', 'latent_achieved_goal')

    contextual_mdp = variant.get('contextual_mdp', True)
    print("contextual_mdp:", contextual_mdp)

    mask_variant = variant.get('mask_variant', {})
    mask_conditioned = mask_variant.get('mask_conditioned', False)
    print("mask_conditioned:", mask_conditioned)

    if mask_conditioned:
        assert contextual_mdp

    if 'sac' in variant['algorithm'].lower():
        rl_algo = 'sac'
    elif 'td3' in variant['algorithm'].lower():
        rl_algo = 'td3'
    else:
        raise NotImplementedError
    print("RL algorithm:", rl_algo)

    ### load the example dataset, if running checkpoints ###
    if 'ckpt' in variant:
        import os.path as osp
        example_set_variant = variant.get('example_set_variant', dict())
        example_set_variant['use_cache'] = True
        example_set_variant['cache_path'] = osp.join(variant['ckpt'], 'example_dataset.npy')

    if mask_conditioned:
        env = get_envs(variant)
        mask_format = mask_variant['param_variant']['mask_format']
        assert mask_format in ['vector', 'matrix', 'distribution', 'cond_distribution']
        goal_dim = env.observation_space.spaces[desired_goal_key].low.size
        if mask_format in ['vector']:
            context_dim_for_networks = goal_dim + goal_dim
        elif mask_format in ['matrix', 'distribution', 'cond_distribution']:
            context_dim_for_networks = goal_dim + (goal_dim * goal_dim)
        else:
            raise TypeError

        if 'ckpt' in variant:
            from rlkit.util.io import local_path_from_s3_or_local_path
            import os.path as osp

            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'masks.npy'))
            masks = np.load(filename, allow_pickle=True)[()]
        else:
            masks = get_mask_params(
                env=env,
                example_set_variant=variant['example_set_variant'],
                param_variant=mask_variant['param_variant'],
            )

        mask_keys = list(masks.keys())
        context_keys = [desired_goal_key] + mask_keys
    else:
        context_keys = [desired_goal_key]


    def contextual_env_distrib_and_reward(mode='expl'):
        assert mode in ['expl', 'eval']
        env = get_envs(variant)

        if mode == 'expl':
            goal_sampling_mode = variant.get('expl_goal_sampling_mode', None)
        elif mode == 'eval':
            goal_sampling_mode = variant.get('eval_goal_sampling_mode', None)
        if goal_sampling_mode not in [None, 'example_set']:
            env.goal_sampling_mode = goal_sampling_mode

        mask_ids_for_training = mask_variant.get('mask_ids_for_training', None)

        if mask_conditioned:
            context_distrib = MaskDictDistribution(
                env,
                desired_goal_keys=[desired_goal_key],
                mask_format=mask_format,
                masks=masks,
                max_subtasks_to_focus_on=mask_variant.get('max_subtasks_to_focus_on', None),
                prev_subtask_weight=mask_variant.get('prev_subtask_weight', None),
                mask_distr=mask_variant.get('train_mask_distr', None),
                mask_ids=mask_ids_for_training,
            )
            reward_fn = ContextualMaskingRewardFn(
                achieved_goal_from_observation=IndexIntoAchievedGoal(achieved_goal_key),
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                mask_keys=mask_keys,
                mask_format=mask_format,
                use_g_for_mean=mask_variant['use_g_for_mean'],
                use_squared_reward=mask_variant.get('use_squared_reward', False),
            )
        else:
            if goal_sampling_mode == 'example_set':
                example_dataset = gen_example_sets(get_envs(variant), variant['example_set_variant'])
                assert len(example_dataset['list_of_waypoints']) == 1
                from rlkit.envs.contextual.set_distributions import GoalDictDistributionFromSet
                context_distrib = GoalDictDistributionFromSet(
                    example_dataset['list_of_waypoints'][0],
                    desired_goal_keys=[desired_goal_key],
                )
            else:
                context_distrib = GoalDictDistributionFromMultitaskEnv(
                    env,
                    desired_goal_keys=[desired_goal_key],
                )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(achieved_goal_key),
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=variant['contextual_replay_buffer_kwargs'].get('observation_keys', None),
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
            update_env_info_fn=delete_info if not variant.get('keep_env_infos', False) else None,
        )
        return env, context_distrib, reward_fn

    env, context_distrib, reward_fn = contextual_env_distrib_and_reward(mode='expl')
    eval_env, eval_context_distrib, _ = contextual_env_distrib_and_reward(mode='eval')

    if mask_conditioned:
        obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + context_dim_for_networks
        )
    elif contextual_mdp:
        obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
        )
    else:
        obs_dim = env.observation_space.spaces[observation_key].low.size

    action_dim = env.action_space.low.size

    if 'ckpt' in variant and 'ckpt_epoch' in variant:
        from rlkit.util.io import local_path_from_s3_or_local_path
        import os.path as osp

        ckpt_epoch = variant['ckpt_epoch']
        if ckpt_epoch is not None:
            epoch = variant['ckpt_epoch']
            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'itr_%d.pkl' % epoch))
        else:
            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'params.pkl'))
        print("Loading ckpt from", filename)
        data = torch.load(filename)
        qf1 = data['trainer/qf1']
        qf2 = data['trainer/qf2']
        target_qf1 = data['trainer/target_qf1']
        target_qf2 = data['trainer/target_qf2']
        policy = data['trainer/policy']
        eval_policy = data['evaluation/policy']
        expl_policy = data['exploration/policy']
    else:
        qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **variant['qf_kwargs']
        )
        qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **variant['qf_kwargs']
        )
        target_qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **variant['qf_kwargs']
        )
        target_qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **variant['qf_kwargs']
        )
        if rl_algo == 'td3':
            policy = TanhMlpPolicy(
                input_size=obs_dim,
                output_size=action_dim,
                **variant['policy_kwargs']
            )
            target_policy = TanhMlpPolicy(
                input_size=obs_dim,
                output_size=action_dim,
                **variant['policy_kwargs']
            )
            expl_policy = create_exploration_policy(
                env, policy,
                exploration_version=variant['exploration_type'],
                exploration_noise=variant['exploration_noise'],
            )
            eval_policy = policy
        elif rl_algo == 'sac':
            policy = TanhGaussianPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                **variant['policy_kwargs']
            )
            expl_policy = policy
            eval_policy = MakeDeterministic(policy)

    post_process_mask_fn = partial(
        full_post_process_mask_fn,
        mask_conditioned=mask_conditioned,
        mask_variant=mask_variant,
        context_distrib=context_distrib,
        context_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
    )

    def context_from_obs_dict_fn(obs_dict):
        context_dict = {
            desired_goal_key: obs_dict[achieved_goal_key]
        }

        if mask_conditioned:
            sample_masks_for_relabeling = mask_variant.get('sample_masks_for_relabeling', True)
            if sample_masks_for_relabeling:
                batch_size = next(iter(obs_dict.values())).shape[0]
                sampled_contexts = context_distrib.sample(batch_size)
                for mask_key in mask_keys:
                    context_dict[mask_key] = sampled_contexts[mask_key]
            else:
                for mask_key in mask_keys:
                    context_dict[mask_key] = obs_dict[mask_key]

        return context_dict

    def concat_context_to_obs(batch, replay_buffer=None, obs_dict=None, next_obs_dict=None, new_contexts=None):
        obs = batch['observations']
        next_obs = batch['next_observations']
        batch_size = obs.shape[0]
        if mask_conditioned:
            if obs_dict is not None and new_contexts is not None:
                if not mask_variant.get('relabel_masks', True):
                    for k in mask_keys:
                        new_contexts[k] = next_obs_dict[k][:]
                    batch.update(new_contexts)
                if not mask_variant.get('relabel_goals', True):
                    new_contexts[desired_goal_key] = next_obs_dict[desired_goal_key][:]
                    batch.update(new_contexts)

                new_contexts = post_process_mask_fn(obs_dict, new_contexts)
                batch.update(new_contexts)

            if mask_format in ['vector', 'matrix']:
                goal = batch[desired_goal_key]
                mask = batch['mask'].reshape((batch_size, -1))
                batch['observations'] = np.concatenate([obs, goal, mask], axis=1)
                batch['next_observations'] = np.concatenate([next_obs, goal, mask], axis=1)
            elif mask_format == 'distribution':
                goal = batch[desired_goal_key]
                sigma_inv = batch['mask_sigma_inv'].reshape((batch_size, -1))
                batch['observations'] = np.concatenate([obs, goal, sigma_inv], axis=1)
                batch['next_observations'] = np.concatenate([next_obs, goal, sigma_inv], axis=1)
            elif mask_format == 'cond_distribution':
                goal = batch[desired_goal_key]
                mu_w = batch['mask_mu_w']
                mu_g = batch['mask_mu_g']
                mu_A = batch['mask_mu_mat']
                sigma_inv = batch['mask_sigma_inv']
                if mask_variant['use_g_for_mean']:
                    mu_w_given_g = goal
                else:
                    mu_w_given_g = mu_w + np.squeeze(mu_A @ np.expand_dims(goal - mu_g, axis=-1), axis=-1)
                sigma_w_given_g_inv = sigma_inv.reshape((batch_size, -1))
                batch['observations'] = np.concatenate([obs, mu_w_given_g, sigma_w_given_g_inv], axis=1)
                batch['next_observations'] = np.concatenate([next_obs, mu_w_given_g, sigma_w_given_g_inv], axis=1)
            else:
                raise NotImplementedError
        elif contextual_mdp:
            goal = batch[desired_goal_key]
            batch['observations'] = np.concatenate([obs, goal], axis=1)
            batch['next_observations'] = np.concatenate([next_obs, goal], axis=1)
        else:
            batch['observations'] = obs
            batch['next_observations'] = next_obs

        return batch

    if 'observation_keys' not in variant['contextual_replay_buffer_kwargs']:
        variant['contextual_replay_buffer_kwargs']['observation_keys'] = []
    observation_keys = variant['contextual_replay_buffer_kwargs']['observation_keys']
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
        **variant['contextual_replay_buffer_kwargs']
    )

    if rl_algo == 'td3':
        trainer = TD3Trainer(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            **variant['td3_trainer_kwargs']
        )
    elif rl_algo == 'sac':
        trainer = SACTrainer(
            env=env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['sac_trainer_kwargs']
        )

    def create_path_collector(
            env,
            policy,
            mode='expl',
            mask_kwargs={},
    ):
        assert mode in ['expl', 'eval']

        save_env_in_snapshot = variant.get('save_env_in_snapshot', True)

        if mask_conditioned:
            if 'rollout_mask_order' in mask_kwargs:
                rollout_mask_order = mask_kwargs['rollout_mask_order']
            else:
                if mode == 'expl':
                    rollout_mask_order = mask_variant.get('rollout_mask_order_for_expl', 'fixed')
                elif mode == 'eval':
                    rollout_mask_order = mask_variant.get('rollout_mask_order_for_eval', 'fixed')
                else:
                    raise TypeError

            if 'mask_distr' in mask_kwargs:
                mask_distr = mask_kwargs['mask_distr']
            else:
                if mode == 'expl':
                    mask_distr = mask_variant['expl_mask_distr']
                elif mode == 'eval':
                    mask_distr = mask_variant['eval_mask_distr']
                else:
                    raise TypeError

            if 'mask_ids' in mask_kwargs:
                mask_ids = mask_kwargs['mask_ids']
            else:
                if mode == 'expl':
                    mask_ids = mask_variant.get('mask_ids_for_expl', None)
                elif mode == 'eval':
                    mask_ids = mask_variant.get('mask_ids_for_eval', None)
                else:
                    raise TypeError

            prev_subtask_weight = mask_variant.get('prev_subtask_weight', None)
            max_subtasks_to_focus_on = mask_variant.get('max_subtasks_to_focus_on', None)
            max_subtasks_per_rollout = mask_variant.get('max_subtasks_per_rollout', None)

            mode = mask_variant.get('context_post_process_mode', None)
            if mode in ['dilute_prev_subtasks_uniform', 'dilute_prev_subtasks_fixed']:
                prev_subtask_weight = 0.5

            return MaskPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
                concat_context_to_obs_fn=concat_context_to_obs,
                save_env_in_snapshot=save_env_in_snapshot,
                mask_sampler=(context_distrib if mode=='expl' else eval_context_distrib),
                mask_distr=mask_distr.copy(),
                mask_ids=mask_ids,
                max_path_length=max_path_length,
                rollout_mask_order=rollout_mask_order,
                prev_subtask_weight=prev_subtask_weight,
                max_subtasks_to_focus_on=max_subtasks_to_focus_on,
                max_subtasks_per_rollout=max_subtasks_per_rollout,
            )
        elif contextual_mdp:
            return ContextualPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
                save_env_in_snapshot=save_env_in_snapshot,
            )
        else:
            return ContextualPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=[],
                save_env_in_snapshot=save_env_in_snapshot,
            )

    expl_path_collector = create_path_collector(env, expl_policy, mode='expl')
    eval_path_collector = create_path_collector(eval_env, eval_policy, mode='eval')

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)

    if variant.get("save_video", True):
        save_period = variant.get('save_video_period', 50)
        dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        dump_video_kwargs['horizon'] = max_path_length

        renderer = EnvRenderer(**variant.get('renderer_kwargs', {}))

        def add_images(env, state_distribution):
            state_env = env.env
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_distribution,
                image_goal_key='image_desired_goal',
                renderer=renderer,
            )
            img_env = InsertImagesEnv(state_env, renderers={
                'image_observation' : renderer,
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

        if variant.get('log_eval_video', True):
            video_path_collector = create_path_collector(img_eval_env, eval_policy, mode='eval')
            rollout_function = video_path_collector._rollout_fn
            eval_video_func = get_save_video_function(
                rollout_function,
                img_eval_env,
                eval_policy,
                tag="eval",
                imsize=variant['renderer_kwargs']['width'],
                image_format='CHW',
                save_video_period=save_period,
                **dump_video_kwargs
            )
            algorithm.post_train_funcs.append(eval_video_func)

        # additional eval videos for mask conditioned case
        if mask_conditioned:
            default_list = [
                'atomic',
                'atomic_seq',
                'cumul_seq',
                'full',
            ]
            eval_rollouts_for_videos = mask_variant.get('eval_rollouts_for_videos', default_list)
            for key in eval_rollouts_for_videos:
                assert key in default_list

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
                    imsize=variant['renderer_kwargs']['width'],
                    image_format='HWC',
                    save_video_period=save_period,
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
                    imsize=variant['renderer_kwargs']['width'],
                    image_format='HWC',
                    save_video_period=save_period,
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
                    imsize=variant['renderer_kwargs']['width'],
                    image_format='HWC',
                    save_video_period=save_period,
                    **dump_video_kwargs
                )
                algorithm.post_train_funcs.append(eval_video_func)

        if variant.get('log_expl_video', True) and not variant['algo_kwargs'].get('eval_only', False):
            img_expl_env = add_images(env, context_distrib)
            video_path_collector = create_path_collector(img_expl_env, expl_policy, mode='expl')
            rollout_function = video_path_collector._rollout_fn
            expl_video_func = get_save_video_function(
                rollout_function,
                img_expl_env,
                expl_policy,
                tag="expl",
                imsize=variant['renderer_kwargs']['width'],
                image_format='CHW',
                save_video_period=save_period,
                **dump_video_kwargs
            )
            algorithm.post_train_funcs.append(expl_video_func)

    addl_collectors = []
    addl_log_prefixes = []
    if mask_conditioned and mask_variant.get('log_mask_diagnostics', True):
        default_list = [
            'atomic',
            'atomic_seq',
            'cumul_seq',
            'full',
        ]
        eval_rollouts_to_log = mask_variant.get('eval_rollouts_to_log', default_list)
        for key in eval_rollouts_to_log:
            assert key in default_list

        # atomic masks
        if 'atomic' in eval_rollouts_to_log:
            for mask_id in eval_path_collector.mask_ids:
                mask_kwargs=dict(
                    mask_ids=[mask_id],
                    mask_distr=dict(
                        atomic=1.0,
                    ),
                )
                collector = create_path_collector(eval_env, eval_policy, mode='eval', mask_kwargs=mask_kwargs)
                addl_collectors.append(collector)
            addl_log_prefixes += [
                'mask_{}/'.format(''.join(str(mask_id)))
                for mask_id in eval_path_collector.mask_ids
            ]

        # full mask
        if 'full' in eval_rollouts_to_log:
            mask_kwargs=dict(
                mask_distr=dict(
                    full=1.0,
                ),
            )
            collector = create_path_collector(eval_env, eval_policy, mode='eval', mask_kwargs=mask_kwargs)
            addl_collectors.append(collector)
            addl_log_prefixes.append('mask_full/')

        # cumulative, sequential mask
        if 'cumul_seq' in eval_rollouts_to_log:
            mask_kwargs=dict(
                rollout_mask_order='fixed',
                mask_distr=dict(
                    cumul_seq=1.0,
                ),
            )
            collector = create_path_collector(eval_env, eval_policy, mode='eval', mask_kwargs=mask_kwargs)
            addl_collectors.append(collector)
            addl_log_prefixes.append('mask_cumul_seq/')

        # atomic, sequential mask
        if 'atomic_seq' in eval_rollouts_to_log:
            mask_kwargs=dict(
                rollout_mask_order='fixed',
                mask_distr=dict(
                    atomic_seq=1.0,
                ),
            )
            collector = create_path_collector(eval_env, eval_policy, mode='eval', mask_kwargs=mask_kwargs)
            addl_collectors.append(collector)
            addl_log_prefixes.append('mask_atomic_seq/')

        def get_mask_diagnostics(unused):
            from rlkit.core.logging import append_log, add_prefix, OrderedDict
            log = OrderedDict()
            for prefix, collector in zip(addl_log_prefixes, addl_collectors):
                paths = collector.collect_new_paths(
                    max_path_length,
                    variant['algo_kwargs']['num_eval_steps_per_epoch'],
                    discard_incomplete_paths=True,
                )
                old_path_info = eval_env.get_diagnostics(paths)

                keys_to_keep = []
                for key in old_path_info.keys():
                    if ('env_infos' in key) and ('final' in key) and ('Mean' in key):
                        keys_to_keep.append(key)
                path_info = OrderedDict()
                for key in keys_to_keep:
                    path_info[key] = old_path_info[key]

                generic_info = add_prefix(
                    path_info,
                    prefix,
                )
                append_log(log, generic_info)

            for collector in addl_collectors:
                collector.end_epoch(0)
            return log

        algorithm._eval_get_diag_fns.append(get_mask_diagnostics)
        
    if 'ckpt' in variant:
        from rlkit.util.io import local_path_from_s3_or_local_path
        import os.path as osp
        assert variant['algo_kwargs'].get('eval_only', False)

        def update_networks(algo, epoch):
            if 'ckpt_epoch' in variant:
                return

            if epoch % algo._eval_epoch_freq == 0:
                filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'itr_%d.pkl' % epoch))
                print("Loading ckpt from", filename)
                data = torch.load(filename)#, map_location='cuda:1')
                eval_policy = data['evaluation/policy']
                eval_policy.to(ptu.device)
                algo.eval_data_collector._policy = eval_policy
                for collector in addl_collectors:
                    collector._policy = eval_policy

        algorithm.post_train_funcs.insert(0, update_networks)

    algorithm.train()

def full_post_process_mask_fn(
        obs_dict, context_dict,
        mask_conditioned,
        mask_variant,
        context_distrib,
        context_key,
        achieved_goal_key,
):
    assert mask_conditioned
    mode = mask_variant.get('context_post_process_mode', None)
    assert mode in [
        'prev_subtasks_solved',
        'dilute_prev_subtasks_uniform',
        'dilute_prev_subtasks_fixed',
        'atomic_to_corresp_cumul',
        None
    ]
    if mode is None:
        return context_dict

    if mode in [
        'prev_subtasks_solved',
        'dilute_prev_subtasks_uniform',
        'dilute_prev_subtasks_fixed',
        'atomic_to_corresp_cumul'
    ]:
        frac = mask_variant.get('context_post_process_frac', 0.50)
        cumul_mask_to_indices = context_distrib.get_cumul_mask_to_indices(context_dict['mask'])
        for k in cumul_mask_to_indices:
            indices = cumul_mask_to_indices[k]
            subset = np.random.choice(len(indices), int(len(indices)*frac), replace=False)
            cumul_mask_to_indices[k] = indices[subset]
    else:
        cumul_mask_to_indices = None
    pp_context_dict = copy.deepcopy(context_dict)

    if mode in ['prev_subtasks_solved', 'dilute_prev_subtasks_uniform', 'dilute_prev_subtasks_fixed']:
        cumul_masks = list(cumul_mask_to_indices.keys())
        for i in range(1, len(cumul_masks)):
            curr_mask = cumul_masks[i]
            prev_mask = cumul_masks[i-1]
            prev_obj_indices = np.where(np.array(prev_mask) > 0)[0]
            indices = cumul_mask_to_indices[curr_mask]
            if mode == 'prev_subtasks_solved':
                pp_context_dict[context_key][indices][:,prev_obj_indices] = \
                    obs_dict[achieved_goal_key][indices][:,prev_obj_indices]
            elif mode == 'dilute_prev_subtasks_uniform':
                pp_context_dict['mask'][indices][:, prev_obj_indices] = \
                    np.random.uniform(size=(len(indices), len(prev_obj_indices)))
            elif mode == 'dilute_prev_subtasks_fixed':
                pp_context_dict['mask'][indices][:, prev_obj_indices] = 0.5
        indices_to_relabel = np.concatenate(list(cumul_mask_to_indices.values()))
        orig_masks = obs_dict['mask'][indices_to_relabel]
        atomic_mask_to_subindices = context_distrib.get_atomic_mask_to_indices(orig_masks)
        atomic_masks = list(atomic_mask_to_subindices.keys())
        cumul_masks = list(cumul_mask_to_indices.keys())
        for i in range(1, len(atomic_masks)):
            orig_atomic_mask = atomic_masks[i]
            relabeled_cumul_mask = cumul_masks[i]
            subindices = atomic_mask_to_subindices[orig_atomic_mask]
            pp_context_dict['mask'][indices_to_relabel][subindices] = relabeled_cumul_mask

    return pp_context_dict
