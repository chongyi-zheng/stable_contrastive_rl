import os.path as osp
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from gym.wrappers import ClipAction

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.data_management.contextual_replay_buffer import (
    RemapKeyFn,
)
from rlkit.envs.contextual.goal_conditioned import (
    PresampledPathDistribution,
)
from rlkit.envs.images import EnvRenderer
from rlkit.envs.images import InsertImageEnv
from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader

from rlkit.torch.sac.stable_contrastive_rl_trainer import StableContrastiveRLTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.util.io import load_local_or_remote_file
from rlkit.visualization.video import RIGVideoSaveFunction
from rlkit.visualization.video import save_paths as save_paths_fn
from rlkit.samplers.data_collector.contextual_path_collector import ContextualPathCollector
from rlkit.samplers.rollout_functions import contextual_rollout

from rlkit.envs.reward_fns import GoalReachingRewardFn
from rlkit.envs.contextual_env import ContextualEnv
from rlkit.utils.logging import logger as logging

from rlkit.learning.contextual_replay_buffer import ContextualRelabelingReplayBuffer
from rlkit.learning.online_offline_split_replay_buffer import OnlineOfflineSplitReplayBuffer
from rlkit.networks.contrastive_networks import ContrastiveQf
from rlkit.networks.gaussian_policy import (
    GaussianCNNPolicy,
    GaussianTwoChannelCNNPolicy
)


state_obs_key = 'state_observation'
state_goal_key = 'state_desired_goal'
state_init_key = 'initial_state_observation'

image_obs_key = 'image_observation'
image_goal_key = 'image_desired_goal'
image_init_key = 'initial_image_observation'

latent_obs_key = 'latent_observation'
latent_goal_key = 'latent_desired_goal'
latent_init_key = 'initial_latent_observation'


def process_args(variant):
    # Maybe adjust the arguments for debugging purposes.
    if variant.get('debug', False):
        if variant['algo_kwargs']['start_epoch'] == 0:
            start_epoch = 0
        else:
            start_epoch = -5

        variant.get('algo_kwargs', {}).update(dict(
            batch_size=32,
            start_epoch=start_epoch,
            num_epochs=5,
            num_eval_steps_per_epoch=variant['max_path_length'],
            num_expl_steps_per_train_loop=variant['max_path_length'],
            num_trains_per_train_loop=2,
            num_online_trains_per_train_loop=2,
            min_num_steps_before_training=2,
        ))
        variant['num_video_columns'] = 1
        variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = int(5E2)
        variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = int(5E2)
        demo_paths = variant['path_loader_kwargs'].get('demo_paths', [])
        if len(demo_paths) > 1:
            variant['path_loader_kwargs']['demo_paths'] = [demo_paths[0]]


def stable_contrastive_rl_experiment(
        max_path_length,
        qf_kwargs,
        trainer_kwargs,
        replay_buffer_kwargs,
        online_offline_split_replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        network_type,
        use_image=False,
        online_offline_split=False,
        qf_class=ContrastiveQf,
        policy_class=None,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        reward_kwargs=None,

        path_loader_kwargs=None,
        env_demo_path='',
        env_offpolicy_data_path='',

        exploration_policy_kwargs=None,
        exploration_goal_sampling_mode=None,
        evaluation_goal_sampling_mode=None,
        training_goal_sampling_mode=None,

        add_env_demos=False,
        add_env_offpolicy_data=False,
        save_paths=True,
        load_demos=False,
        save_pretrained_algorithm=False,

        # Video parameters
        save_video=True,
        save_video_pickle=False,
        expl_save_video_kwargs=None,
        eval_save_video_kwargs=None,

        renderer_kwargs=None,
        imsize=84,
        pretrained_rl_path=None,
        presampled_goal_kwargs=None,
        num_video_columns=8,
        init_camera=None,
        **kwargs
):
    # Kwarg Definitions
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if presampled_goal_kwargs is None:
        presampled_goal_kwargs = \
            {'eval_goals': '', 'expl_goals': '', 'training_goals': ''}
    if path_loader_kwargs is None:
        path_loader_kwargs = {}
    if not expl_save_video_kwargs:
        expl_save_video_kwargs = {}
    if not eval_save_video_kwargs:
        eval_save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}

    if use_image:
        obs_type = 'image'
    else:
        raise NotImplementedError

    obs_key = '%s_observation' % obs_type
    goal_key = '%s_desired_goal' % obs_type

    # Observation keys for the reward function.
    obs_key_reward_fn = None
    goal_key_reward_fn = None
    if obs_type != reward_kwargs['obs_type']:
        raise ValueError

    ########################################
    # Enviorments
    ########################################
    logging.info('Creating the environment...')
    renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)

    def contextual_env_distrib_and_reward(
        env_id,
        env_class,
        env_kwargs,
        goal_sampling_mode,
        presampled_goals_path,
        reward_kwargs,
    ):
        state_env = get_gym_env(
            env_id,
            env_class=env_class,
            env_kwargs=env_kwargs,
        )
        state_env = ClipAction(state_env)
        renderer = EnvRenderer(
            init_camera=init_camera,
            **renderer_kwargs)

        env = InsertImageEnv(
            state_env,
            renderer=renderer)

        if goal_sampling_mode == 'presampled_images':
            diagnostics = state_env.get_contextual_diagnostics
            context_distribution = PresampledPathDistribution(
                presampled_goals_path, None, initialize_encodings=False)
        else:
            raise NotImplementedError

        reward_fn = GoalReachingRewardFn(
            state_env.env,
            **reward_kwargs
        )

        contextual_env = ContextualEnv(
            env,
            context_distribution=context_distribution,
            reward_fn=reward_fn,
            observation_key=obs_key,
            contextual_diagnostics_fns=[diagnostics] if not isinstance(
                diagnostics, list) else diagnostics,
        )

        return contextual_env, context_distribution, reward_fn

    # Environment Definitions
    presampled_goal_kwargs['expl_goals'] = (
        presampled_goal_kwargs['eval_goals'])
    presampled_goal_kwargs['expl_goals_kwargs'] = (
        presampled_goal_kwargs['eval_goals_kwargs'])

    logging.info('use_image: %s', use_image)

    logging.info('Preparing the [exploration] env and contextual distrib...')
    logging.info('sampling mode: %r', exploration_goal_sampling_mode)
    logging.info('presampled goals: %r',
                 presampled_goal_kwargs['expl_goals'])
    logging.info('presampled goals kwargs: %r',
                 presampled_goal_kwargs['expl_goals_kwargs'],
                 )
    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        env_id,
        env_class,
        env_kwargs,
        exploration_goal_sampling_mode,
        presampled_goal_kwargs['expl_goals'],
        reward_kwargs,
    )

    logging.info('Preparing the [evaluation] env and contextual distrib...')
    logging.info('Preparing the eval env and contextual distrib...')
    logging.info('sampling mode: %r', evaluation_goal_sampling_mode)
    logging.info('presampled goals: %r',
                 presampled_goal_kwargs['eval_goals'])
    logging.info('presampled goals kwargs: %r',
                 presampled_goal_kwargs['eval_goals_kwargs'],
                 )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        env_id,
        env_class,
        env_kwargs,
        evaluation_goal_sampling_mode,
        presampled_goal_kwargs['eval_goals'],
        reward_kwargs,
    )

    logging.info('Preparing the [training] env and contextual distrib...')
    logging.info('sampling mode: %r', training_goal_sampling_mode)
    logging.info('presampled goals: %r',
                 presampled_goal_kwargs['training_goals'])
    logging.info('presampled goals kwargs: %r',
                 presampled_goal_kwargs['training_goals_kwargs'],
                 )
    _, training_context_distrib, _ = contextual_env_distrib_and_reward(
        env_id,
        env_class,
        env_kwargs,
        training_goal_sampling_mode,
        presampled_goal_kwargs['training_goals'],
        reward_kwargs,
    )

    # Key Setting
    assert (
        expl_env.observation_space.spaces[obs_key].low.size
        == expl_env.observation_space.spaces[goal_key].low.size)
    obs_dim = expl_env.observation_space.spaces[obs_key].low.size
    goal_dim = expl_env.observation_space.spaces[goal_key].low.size
    action_dim = expl_env.action_space.low.size

    ########################################
    # Neural Network Architecture
    ########################################
    logging.info('Creating the models...')

    if pretrained_rl_path is not None:
        logging.info('Loading pretrained RL from: %s', pretrained_rl_path)
        rl_model_dict = load_local_or_remote_file(pretrained_rl_path)

        qf = rl_model_dict['trainer/qf']
        target_qf = rl_model_dict['trainer/target_qf']
        policy = rl_model_dict['trainer/policy']
    else:
        def create_qf():
            assert network_type == "contrastive_cnn"

            qf_kwargs['use_image_obs'] = True
            qf_kwargs['imsize'] = imsize
            qf_kwargs['action_dim'] = action_dim

            qf_kwargs['hidden_init'] = partial(
                nn.init.xavier_uniform_,
                gain=nn.init.calculate_gain('relu'))

            return qf_class(
                **qf_kwargs
            )

        def create_policy():
            policy_kwargs['hidden_init'] = partial(
                nn.init.xavier_uniform_,
                gain=nn.init.calculate_gain('relu'))
            policy_layer_norm = policy_kwargs.pop('layer_norm')

            if policy_class == GaussianCNNPolicy:
                assert policy_kwargs['output_activation'] is None
                policy_kwargs['input_width'] = imsize
                policy_kwargs['input_height'] = imsize
                policy_kwargs['input_channels'] = 6
                policy_kwargs['kernel_sizes'] = [8, 4, 3]
                policy_kwargs['n_channels'] = [32, 64, 64]
                policy_kwargs['strides'] = [4, 2, 1]
                policy_kwargs['paddings'] = [2, 1, 1]
                policy_kwargs['conv_normalization_type'] = 'layer' if policy_layer_norm else 'none'
                policy_kwargs['fc_normalization_type'] = 'layer' if policy_layer_norm else 'none'
            elif policy_class == GaussianTwoChannelCNNPolicy:
                assert policy_kwargs['output_activation'] is None
                policy_kwargs['input_width'] = imsize
                policy_kwargs['input_height'] = imsize
                policy_kwargs['input_channels'] = 3
                policy_kwargs['kernel_sizes'] = [8, 4, 3]
                policy_kwargs['n_channels'] = [32, 64, 64]
                policy_kwargs['strides'] = [4, 2, 1]
                policy_kwargs['paddings'] = [2, 1, 1]
                policy_kwargs['conv_normalization_type'] = 'layer' if policy_layer_norm else 'none'
                policy_kwargs['fc_normalization_type'] = 'layer' if policy_layer_norm else 'none'
            else:
                raise NotImplementedError

            return policy_class(
                obs_dim=None,
                action_dim=action_dim,
                **policy_kwargs,
            )

        qf = create_qf()
        target_qf = create_qf()
        policy = create_policy()

    ########################################
    # Replay Buffer
    ########################################
    logging.info('Creating the replay buffer...')

    context2obs = {goal_key: obs_key}
    cont_keys = [goal_key]

    obs_keys_to_save = []
    cont_keys_to_save = []

    if reward_kwargs['obs_type'] != obs_type and (
            expl_reward.obs_key == 'latent'):
        assert obs_key_reward_fn == expl_reward.obs_key
        assert goal_key_reward_fn == expl_reward.goal_key
        context2obs[goal_key_reward_fn] = obs_key_reward_fn
        obs_keys_to_save.append(obs_key_reward_fn)
        cont_keys.append(goal_key_reward_fn)

    if reward_kwargs.get('reward_type', 'dense') == 'highlevel':
        context2obs[state_goal_key] = state_obs_key
        obs_keys_to_save.append(state_obs_key)
        cont_keys_to_save.append(state_goal_key)

    context2obs_mapper = RemapKeyFn(context2obs)

    def concat_context_to_obs(batch,
                              replay_buffer,
                              obs_dict,
                              next_obs_dict,
                              new_contexts):
        obs = batch['observations']
        if type(obs) is tuple:
            obs = np.concatenate(obs, axis=1)

        next_obs = batch['next_observations']
        if type(next_obs) is tuple:
            next_obs = np.concatenate(next_obs, axis=1)

        context = new_contexts[goal_key]

        batch['observations'] = obs
        batch['next_observations'] = next_obs
        batch['contexts'] = context

        return batch

    online_replay_buffer_kwargs = online_offline_split_replay_buffer_kwargs[
        'online_replay_buffer_kwargs']
    offline_replay_buffer_kwargs = online_offline_split_replay_buffer_kwargs[
        'offline_replay_buffer_kwargs']

    if not online_offline_split:
        replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys_to_save,
            observation_key=obs_key,
            context_distribution=training_context_distrib,
            sample_context_from_obs_dict_fn=context2obs_mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            context_keys_to_save=cont_keys_to_save,
            **replay_buffer_kwargs
        )
    else:
        online_replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys_to_save,
            observation_key=obs_key,
            context_distribution=training_context_distrib,
            sample_context_from_obs_dict_fn=context2obs_mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            context_keys_to_save=cont_keys_to_save,
            **online_replay_buffer_kwargs,
        )
        offline_replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys_to_save,
            observation_key=obs_key,
            context_distribution=training_context_distrib,
            sample_context_from_obs_dict_fn=context2obs_mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            context_keys_to_save=cont_keys_to_save,
            **offline_replay_buffer_kwargs
        )
        replay_buffer = OnlineOfflineSplitReplayBuffer(
            offline_replay_buffer,
            online_replay_buffer,
            **online_offline_split_replay_buffer_kwargs
        )

        logging.info('online_replay_buffer_kwargs: %r',
                     online_replay_buffer_kwargs)
        logging.info('offline_replay_buffer_kwargs: %r',
                     offline_replay_buffer_kwargs)
        logging.info('online_offline_split_replay_buffer_kwargs: %r',
                     online_offline_split_replay_buffer_kwargs)

    eval_replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=cont_keys,
        observation_keys_to_save=obs_keys_to_save,
        observation_key=obs_key,
        context_distribution=training_context_distrib,
        sample_context_from_obs_dict_fn=context2obs_mapper,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        context_keys_to_save=cont_keys_to_save,
        **replay_buffer_kwargs
    )

    ########################################
    # Path Collectors
    ########################################
    logging.info('Creating the path collector...')

    path_collector_observation_keys = [obs_key]
    path_collector_context_keys_for_policy = [goal_key]

    def path_collector_obs_processor(o):
        combined_obs = []

        for k in path_collector_observation_keys:
            combined_obs.append(o[k])

        for k in path_collector_context_keys_for_policy:
            combined_obs.append(o[k])

        return np.concatenate(combined_obs, axis=0)

    eval_policy = policy
    eval_path_collector = ContextualPathCollector(
        eval_env,
        eval_policy,
        observation_keys=path_collector_observation_keys,
        context_keys_for_policy=path_collector_context_keys_for_policy,
        obs_processor=path_collector_obs_processor,
        rollout=contextual_rollout,
    )

    expl_policy = create_exploration_policy(
        expl_env,
        policy,
        **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        expl_policy,
        observation_keys=path_collector_observation_keys,
        context_keys_for_policy=path_collector_context_keys_for_policy,
        obs_processor=path_collector_obs_processor,
        rollout=contextual_rollout,
    )

    ########################################
    # Training Algorithm
    ########################################
    logging.info('Creating the training algorithm...')

    print('trainer_kwargs: ')
    print(trainer_kwargs)

    trainer = StableContrastiveRLTrainer(
        env=eval_env,
        policy=policy,
        qf=qf,
        target_qf=target_qf,
        **trainer_kwargs
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_replay_buffer=eval_replay_buffer,  # TODO
        max_path_length=max_path_length,
        **algo_kwargs
    )

    algorithm.to(ptu.device)

    if save_paths:
        algorithm.post_train_funcs.append(save_paths_fn)

    if save_video:
        assert (num_video_columns * max_path_length <=
                algo_kwargs['num_expl_steps_per_train_loop'])

        expl_save_video_kwargs['include_final_goal'] = True
        eval_save_video_kwargs['include_final_goal'] = True

        expl_video_func = RIGVideoSaveFunction(
            None,
            expl_path_collector,
            'train',
            image_goal_key=image_goal_key,
            rows=2,
            columns=num_video_columns,
            imsize=imsize,
            image_format=renderer.output_image_format,
            unnormalize=True,
            dump_pickle=save_video_pickle,
            dump_only_init_and_goal=True,
            **expl_save_video_kwargs
        )
        algorithm.post_train_funcs.append(expl_video_func)

        if algo_kwargs['num_eval_steps_per_epoch'] > 0:
            eval_video_func = RIGVideoSaveFunction(
                None,
                eval_path_collector,
                'eval',
                image_goal_key=image_goal_key,
                rows=2,
                columns=num_video_columns,
                imsize=imsize,
                image_format=renderer.output_image_format,
                unnormalize=True,
                dump_pickle=save_video_pickle,
                dump_only_init_and_goal=True,
                **eval_save_video_kwargs
            )
            algorithm.post_train_funcs.append(eval_video_func)

    if online_offline_split:
        replay_buffer.set_online_mode(False)

    if load_demos:
        if add_env_demos:
            path_loader_kwargs['demo_paths'].append(env_demo_path)

        if add_env_offpolicy_data:
            path_loader_kwargs['demo_paths'].append(env_offpolicy_data_path)

        demo_train_buffer = None
        demo_test_buffer = eval_replay_buffer

        path_loader_class = DictToMDPPathLoader

        path_loader = path_loader_class(
            env=eval_env,
            trainer=trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            reward_fn=eval_reward,
            **path_loader_kwargs
        )
        path_loader.load_demos()

        if not path_loader_kwargs['add_demos_to_replay_buffer']:
            path_loader.dump_paths()
            exit()

    if online_offline_split:
        replay_buffer.set_online_mode(True)

    if save_pretrained_algorithm:
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, 'wb'))
        torch.save(data, open(p_path, 'wb'))

    logging.info('Start training..')
    algorithm.train()
