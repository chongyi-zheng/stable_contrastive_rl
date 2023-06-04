import os.path as osp
import sys  # NOQA
from collections import OrderedDict  # NOQA
from functools import partial

import gin  # NOQA
import numpy as np
import torch
from gym.wrappers import ClipAction

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.data_management.contextual_replay_buffer import (
    RemapKeyFn,
)
from rlkit.data_management.online_offline_split_replay_buffer import (
    OnlineOfflineSplitReplayBuffer,
)
from rlkit.envs.contextual.goal_conditioned import (
    PresampledPathDistribution,
)
from rlkit.envs.contextual.latent_distributions import (  # NOQA
    AmortizedConditionalPriorDistribution,
    PresampledPriorDistribution,
    ConditionalPriorDistribution,
    AmortizedPriorDistribution,
    AddDecodedImageDistribution,
    AddLatentDistribution,
    PriorDistribution,
    PresamplePriorDistribution,
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv
from rlkit.envs.images import EnvRenderer
from rlkit.envs.images import InsertImageEnv
from rlkit.demos.source.mdp_path_loader import MDPPathLoader  # NOQA
from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader  # NOQA
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader  # NOQA
from rlkit.torch.networks import ConcatMlp, Mlp
# from rlkit.torch.networks.cnn import ConcatCNN
# from rlkit.torch.networks.cnn import ConcatTwoChannelCNN
from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.sac.policies import MakeDeterministic  # NOQA
from rlkit.torch.sac.iql_vib_trainer import IQLVIBTrainer  # NOQA
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.launchers.contextual.rig.rig_launcher import StateImageGoalDiagnosticsFn  # NOQA
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.util.io import load_local_or_remote_file
from rlkit.visualization.video import RIGVideoSaveFunction
from rlkit.visualization.video import save_paths as save_paths_fn
from rlkit.samplers.data_collector.contextual_path_collector import ContextualPathCollector  # NOQA
from rlkit.samplers.rollout_functions import contextual_rollout

from rlkit.experimental.kuanfang.envs.reward_fns import GoalReachingRewardFn
from rlkit.experimental.kuanfang.envs.contextual_env import ContextualEnv
from rlkit.experimental.kuanfang.envs.contextual_env import SubgoalContextualEnv  # NOQA
from rlkit.experimental.kuanfang.envs.contextual_env import NonEpisodicSubgoalContextualEnv  # NOQA
from rlkit.experimental.kuanfang.learning.contextual_replay_buffer import ContextualRelabelingReplayBuffer  # NOQA
from rlkit.experimental.kuanfang.planning.random_planner import RandomPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.mppi_planner import MppiPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.rb_mppi_planner import RbMppiPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.planner import HierarchicalPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.scripted_planner import ScriptedPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.scripted_planner import RandomChoicePlanner  # NOQA
from rlkit.experimental.kuanfang.utils.logging import logger as logging
from rlkit.experimental.kuanfang.utils import io_util

from rlkit.experimental.kuanfang.vae import affordance_networks
from rlkit.experimental.kuanfang.networks.encoding_networks import ObsEncoder  # NOQA
from rlkit.experimental.kuanfang.networks.encoding_networks import VariationalObsEncoder  # NOQA
from rlkit.experimental.kuanfang.networks.encoding_networks import CNNVariationalObsEncoder  # NOQA
from rlkit.experimental.kuanfang.networks.encoding_networks import ResNetVariationalObsEncoder  # NOQA
from rlkit.experimental.kuanfang.networks.encoding_networks import VqvaeVariationalObsEncoder  # NOQA
from rlkit.experimental.kuanfang.networks.encoding_networks import EncodingGaussianPolicy  # NOQA


PLANNER_CTORS = {
    'random': RandomPlanner,
    'mppi': MppiPlanner,
    'rb': RbMppiPlanner,
    'hierarchical': partial(HierarchicalPlanner,
                            sub_planner_ctor=MppiPlanner,
                            num_levels=3,
                            min_dt=20,
                            )
}


state_obs_key = 'state_observation'
state_goal_key = 'state_desired_goal'
state_init_key = 'initial_state_observation'

image_obs_key = 'image_observation'
image_goal_key = 'image_desired_goal'
image_init_key = 'initial_image_observation'

latent_obs_key = 'latent_observation'
latent_goal_key = 'latent_desired_goal'
latent_init_key = 'initial_latent_observation'

vib_obs_key = 'vib_observation'
vib_goal_key = 'vib_desired_goal'
vib_init_key = 'initial_vib_observation'


def process_args(variant):
    # Maybe adjust the arguments for debugging purposes.
    if variant.get('debug', False):
        variant['max_path_length'] = 5
        # variant['max_path_length'] = 100
        # variant['num_presample'] = 50
        # variant['num_presample'] = 32
        if variant['algo_kwargs']['start_epoch'] == 0:
            start_epoch = 0
        else:
            start_epoch = -5

        variant.get('algo_kwargs', {}).update(dict(
            batch_size=32,
            start_epoch=start_epoch,
            num_epochs=5,
            num_eval_steps_per_epoch=variant['max_path_length'] * 5,
            num_expl_steps_per_train_loop=variant['max_path_length'] * 5,
            num_trains_per_train_loop=2,
            num_online_trains_per_train_loop=2,
            min_num_steps_before_training=2,
        ))
        variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = int(5E2)  # NOQA
        variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = int(5E2)  # NOQA
        demo_paths = variant['path_loader_kwargs'].get('demo_paths', [])
        if len(demo_paths) > 1:
            variant['path_loader_kwargs']['demo_paths'] = [demo_paths[0]]


def ptp_experiment(  # NOQA
        max_path_length,
        qf_kwargs,
        vf_kwargs,
        obs_encoder_kwargs,
        trainer_kwargs,
        replay_buffer_kwargs,
        online_offline_split_replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        obs_encoding_dim,
        affordance_encoding_dim,
        network_type,   # TODO
        use_image=False,
        finetune_with_obs_encoder=False,
        online_offline_split=False,
        policy_class=None,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        reward_kwargs=None,

        path_loader_kwargs=None,
        env_demo_path='',
        env_offpolicy_data_path='',

        debug=False,
        epsilon=1.0,
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        training_goal_sampling_mode=None,

        add_env_demos=False,
        add_env_offpolicy_data=False,
        save_paths=True,
        load_demos=False,
        pretrain_policy=False,
        pretrain_rl=False,
        save_pretrained_algorithm=False,

        trainer_type='iql',
        network_version=None,

        # Video parameters
        save_video=True,
        save_video_pickle=False,
        expl_save_video_kwargs=None,
        eval_save_video_kwargs=None,

        renderer_kwargs=None,
        imsize=84,
        pretrained_vae_path='',
        pretrained_rl_path=None,
        input_representation='',
        goal_representation='',
        presampled_goal_kwargs=None,
        presampled_goals_path='',
        num_presample=50,
        num_video_columns=8,
        init_camera=None,
        # vf_type='mlp',
        qf_class=ConcatMlp,
        vf_class=Mlp,
        env_type=None,  # For plotting
        seed=None,
        multiple_goals_eval_seeds=None,

        expl_reset_interval=0,

        use_expl_planner=True,
        expl_planner_type='hierarchical',
        expl_planner_kwargs=None,
        expl_planner_scripted_goals=None,
        expl_contextual_env_kwargs=None,

        use_eval_planner=False,
        eval_planner_type=None,
        eval_planner_kwargs=None,
        eval_planner_scripted_goals=None,
        eval_contextual_env_kwargs=None,

        augment_order=[],
        augment_params=dict(),
        augment_probability=0.0,

        use_encoder_in_policy=False,
        fix_encoder_online=False,

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

    if not expl_planner_kwargs:
        expl_planner_kwargs = {}
    if not expl_contextual_env_kwargs:
        expl_contextual_env_kwargs = {}

    if not eval_planner_kwargs:
        eval_planner_kwargs = {}
    if not eval_contextual_env_kwargs:
        eval_contextual_env_kwargs = {}

    if finetune_with_obs_encoder:
        obs_type = 'vib'
        reward_kwargs['obs_type'] = 'vib'
    else:
        if use_image:
            obs_type = 'image'
        else:
            obs_type = 'latent'

    obs_key = '%s_observation' % (obs_type)
    goal_key = '%s_desired_goal' % (obs_type)

    # Observation keys for the reward function.
    obs_key_reward_fn = None
    goal_key_reward_fn = None
    if obs_type == reward_kwargs['obs_type']:
        pass
    elif obs_type == 'image' and reward_kwargs['obs_type'] == 'latent':
        obs_key_reward_fn = 'latent_observation'
        goal_key_reward_fn = 'latent_desired_goal'
    else:
        raise ValueError

    use_vqvae = (not use_image or reward_kwargs['obs_type'] == 'latent')

    ########################################
    # VQVAE
    ########################################
    model = io_util.load_model(pretrained_vae_path)
    vqvae = model['vqvae']

    if finetune_with_obs_encoder:
        logging.info('Loading pretrained RL from: %s', pretrained_rl_path)
        rl_model_dict = load_local_or_remote_file(pretrained_rl_path)
        obs_encoder = rl_model_dict['trainer/obs_encoder']
        model['obs_encoder'] = obs_encoder
        # The lines below should be removed from future commits.
        obs_encoder.representation_size = obs_encoder.output_dim
        obs_encoder.input_channels = obs_encoder._vqvae.input_channels
        obs_encoder.imsize = obs_encoder._vqvae.imsize
    else:
        obs_encoder = None

    ########################################
    # Enviorments
    ########################################
    logging.info('Creating the environment...')
    renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)

    def create_planner(
        planner_type,
        planner_kwargs,
        planner_scripted_goals,
    ):

        if obs_type == 'image':
            planner_kwargs['encoding_type'] = None
        elif obs_type == 'latent':
            assert use_vqvae
            planner_kwargs['encoding_type'] = 'vqvae'
        elif obs_type == 'vib':
            assert use_vqvae
            planner_kwargs['encoding_type'] = 'vib'
        else:
            raise ValueError('Unrecognized obs_type: %s' %
                             (obs_type))

        if planner_scripted_goals is None:
            planner_scripted_goals = presampled_goal_kwargs['eval_goals']

        if planner_type == 'scripted':
            planner = ScriptedPlanner(
                model,
                path=planner_scripted_goals,
                **planner_kwargs)

        elif planner_type == 'uniform':
            planner = RandomChoicePlanner(
                model,
                path=planner_scripted_goals,
                uniform=True,
                **planner_kwargs)

        elif planner_type == 'closest':
            planner = RandomChoicePlanner(
                model,
                path=planner_scripted_goals,
                uniform=False,
                **planner_kwargs)

        elif planner_type == 'prior':
            planner_kwargs['cost_mode'] = None
            planner = RandomPlanner(
                model,
                **planner_kwargs)

        elif planner_type[:3] == 'gen':
            subtype = planner_type[4:]
            planner_kwargs['cost_mode'] = 'vf_' + subtype
            planner = RandomPlanner(
                model,
                **planner_kwargs)

        else:
            planner_ctor = PLANNER_CTORS[planner_type]
            planner = planner_ctor(model, debug=False, **planner_kwargs)

        return planner

    def contextual_env_distrib_and_reward(
        env_id,
        env_class,
        env_kwargs,
        goal_sampling_mode,
        presampled_goals_path,
        num_presample,
        reward_kwargs,
        presampled_goals_kwargs,

        use_planner,
        planner_type,
        planner_kwargs,
        planner_scripted_goals,
        contextual_env_kwargs,
        reset_interval=0,
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

        if use_vqvae:
            env = EncoderWrappedEnv(
                env,
                vqvae,
                step_keys_map={'image_observation': 'latent_observation'},
                reset_keys_map={'image_observation': 'initial_latent_state'},
            )

        if finetune_with_obs_encoder:
            env = EncoderWrappedEnv(
                env,
                obs_encoder,
                step_keys_map={'latent_observation': 'vib_observation'},
                reset_keys_map={'latent_observation': 'initial_vib_state'},
            )

        if goal_sampling_mode == 'presampled_images':
            diagnostics = state_env.get_contextual_diagnostics
            context_distribution = PresampledPathDistribution(
                presampled_goals_path,
                vqvae.representation_size if use_vqvae else None,
                initialize_encodings=use_vqvae)
            if use_vqvae:
                context_distribution = AddLatentDistribution(
                    context_distribution,
                    input_key=image_goal_key,
                    output_key=latent_goal_key,
                    model=vqvae)
            if finetune_with_obs_encoder:
                context_distribution = AddLatentDistribution(
                    context_distribution,
                    input_key=latent_goal_key,
                    output_key=vib_goal_key,
                    model=obs_encoder)
        else:
            raise NotImplementedError

        reward_fn = GoalReachingRewardFn(
            state_env,
            **reward_kwargs
        )

        if not isinstance(diagnostics, list):
            contextual_diagnostics_fns = [diagnostics]
        else:
            contextual_diagnostics_fns = diagnostics

        if use_planner:

            planner = create_planner(
                planner_type=planner_type,
                planner_kwargs=planner_kwargs,
                planner_scripted_goals=planner_scripted_goals)

            if reset_interval <= 0:
                contextual_env = SubgoalContextualEnv(
                    env,
                    context_distribution=context_distribution,
                    reward_fn=reward_fn,
                    observation_key=obs_key,
                    goal_key=goal_key,
                    goal_key_reward_fn=goal_key_reward_fn,
                    contextual_diagnostics_fns=contextual_diagnostics_fns,
                    planner=planner,
                    **contextual_env_kwargs,
                )

            else:
                if obs_type == 'latent':
                    assert use_vqvae
                    context_distribution = AddLatentDistribution(
                        context_distribution,
                        input_key=image_init_key,
                        output_key=latent_init_key,
                        model=vqvae)

                contextual_env = NonEpisodicSubgoalContextualEnv(
                    env,
                    context_distribution=context_distribution,
                    reward_fn=reward_fn,
                    observation_key=obs_key,
                    initial_state_key=latent_init_key,
                    goal_key=goal_key,
                    goal_key_reward_fn=goal_key_reward_fn,
                    contextual_diagnostics_fns=contextual_diagnostics_fns,
                    planner=planner,
                    reset_interval=reset_interval,
                    **contextual_env_kwargs,
                )

        else:
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

    # TODO(kuanfang): Use exactly the same arguments as in the eval_env.
    exploration_goal_sampling_mode = evaluation_goal_sampling_mode
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
    logging.info('num_presample: %d', num_presample)
    expl_env, expl_context_distrib, expl_reward = (
        contextual_env_distrib_and_reward(
            env_id,
            env_class,
            env_kwargs,
            exploration_goal_sampling_mode,
            presampled_goal_kwargs['expl_goals'],
            num_presample,
            reward_kwargs=reward_kwargs,
            presampled_goals_kwargs=(
                presampled_goal_kwargs['expl_goals_kwargs']),
            use_planner=use_expl_planner,
            planner_type=expl_planner_type,
            planner_kwargs=expl_planner_kwargs,
            planner_scripted_goals=expl_planner_scripted_goals,
            contextual_env_kwargs=expl_contextual_env_kwargs,
            reset_interval=expl_reset_interval,
        ))

    logging.info('Preparing the [evaluation] env and contextual distrib...')
    logging.info('Preparing the eval env and contextual distrib...')
    logging.info('sampling mode: %r', evaluation_goal_sampling_mode)
    logging.info('presampled goals: %r',
                 presampled_goal_kwargs['eval_goals'])
    logging.info('presampled goals kwargs: %r',
                 presampled_goal_kwargs['eval_goals_kwargs'],
                 )
    logging.info('num_presample: %d', num_presample)
    eval_env, eval_context_distrib, eval_reward = (
        contextual_env_distrib_and_reward(
            env_id,
            env_class,
            env_kwargs,
            evaluation_goal_sampling_mode,
            presampled_goal_kwargs['eval_goals'],
            num_presample,
            reward_kwargs=reward_kwargs,
            presampled_goals_kwargs=(
                presampled_goal_kwargs['eval_goals_kwargs']),
            use_planner=use_eval_planner,
            planner_type=eval_planner_type,
            planner_kwargs=eval_planner_kwargs,
            planner_scripted_goals=eval_planner_scripted_goals,
            contextual_env_kwargs=eval_contextual_env_kwargs,
        ))

    logging.info('Preparing the [training] env and contextual distrib...')
    logging.info('sampling mode: %r', training_goal_sampling_mode)
    logging.info('presampled goals: %r',
                 presampled_goal_kwargs['training_goals'])
    logging.info('presampled goals kwargs: %r',
                 presampled_goal_kwargs['training_goals_kwargs'],
                 )
    logging.info('num_presample: %d', num_presample)
    _, training_context_distrib, _ = (  # TODO(kuanfang)
        contextual_env_distrib_and_reward(
            env_id,
            env_class,
            env_kwargs,
            training_goal_sampling_mode,
            presampled_goal_kwargs['training_goals'],
            num_presample,
            reward_kwargs=reward_kwargs,
            presampled_goals_kwargs=(
                presampled_goal_kwargs['training_goals_kwargs']),
            use_planner=False,
            planner_type=None,
            planner_kwargs=None,
            planner_scripted_goals=None,
            contextual_env_kwargs=None,
        ))

    # Key Setting
    assert (
        expl_env.observation_space.spaces[obs_key].low.size
        == expl_env.observation_space.spaces[goal_key].low.size)
    obs_dim = expl_env.observation_space.spaces[obs_key].low.size
    action_dim = expl_env.action_space.low.size

    ########################################
    # Neural Network Architecture
    ########################################
    logging.info('Creating the models...')

    # if trainer_type == 'vib':
    #     if expl_planner_type not in ['scripted']:
    #         assert eval_planner_type not in ['scripted']
    #         trainer_kwargs['goal_is_encoded'] = True
    #         policy_kwargs['goal_is_encoded'] = True

    if pretrained_rl_path is not None:
        logging.info('Loading pretrained RL from: %s', pretrained_rl_path)
        rl_model_dict = load_local_or_remote_file(pretrained_rl_path)
        if obs_encoder is None:
            # Has been loaded
            obs_encoder = rl_model_dict['trainer/obs_encoder']
        affordance = rl_model_dict['trainer/affordance']
        qf1 = rl_model_dict['trainer/qf1']
        qf2 = rl_model_dict['trainer/qf2']
        target_qf1 = rl_model_dict['trainer/target_qf1']
        target_qf2 = rl_model_dict['trainer/target_qf2']
        vf = rl_model_dict['trainer/vf']
        plan_vf = rl_model_dict['trainer/plan_vf']
        policy = rl_model_dict['trainer/policy']
        if 'std' in policy_kwargs and policy_kwargs['std'] is not None:
            policy.std = policy_kwargs['std']
            policy.log_std = np.log(policy.std)
        policy.obs_encoder = obs_encoder  # TODO

        # if trainer_type == 'vib':
        #     if expl_planner_type not in ['scripted']:
        #         assert eval_planner_type not in ['scripted']
        #         policy._goal_is_encoded = True

        if finetune_with_obs_encoder:
            policy._always_use_encoded_input = True
            policy._goal_is_encoded = True
            trainer_kwargs['goal_is_encoded'] = True
            policy_kwargs['goal_is_encoded'] = True

    else:
        if use_image:
            # obs_encoder_class = CNNVariationalObsEncoder
            obs_encoder_kwargs['input_width'] = imsize
            obs_encoder_kwargs['input_height'] = imsize
            obs_encoder_kwargs['input_channels'] = 3
            obs_encoder_kwargs['output_dim'] = obs_encoding_dim

            # TODO(Debug)
            if network_type == 'cnn_v1':
                obs_encoder_class = CNNVariationalObsEncoder
            elif network_type == 'cnn_v2':
                obs_encoder_class = CNNVariationalObsEncoder
                obs_encoder_kwargs['kernel_sizes'] = [3, 3, 3]
                obs_encoder_kwargs['n_channels'] = [64, 64, 64]
                obs_encoder_kwargs['strides'] = [1, 1, 1]
                obs_encoder_kwargs['paddings'] = [1, 1, 1]
                obs_encoder_kwargs['pool_sizes'] = [2, 2, 1]
                obs_encoder_kwargs['pool_strides'] = [2, 2, 1]
                obs_encoder_kwargs['pool_paddings'] = [0, 0, 0]
                obs_encoder_kwargs['embedding_dim'] = 5
                obs_encoder_kwargs['fc_hidden_sizes'] = [128, 128]
            elif network_type == 'cnn_v3':
                obs_encoder_class = CNNVariationalObsEncoder
                obs_encoder_kwargs['kernel_sizes'] = [3, 3, 3, 3, 3, 3]
                obs_encoder_kwargs['n_channels'] = [64, 64, 64, 64, 64, 64]
                obs_encoder_kwargs['strides'] = [1, 1, 1, 1, 1, 1]
                obs_encoder_kwargs['paddings'] = [1, 1, 1, 1, 1, 1]
                obs_encoder_kwargs['pool_sizes'] = [1, 2, 1, 2, 1, 1]
                obs_encoder_kwargs['pool_strides'] = [1, 2, 1, 2, 1, 1]
                obs_encoder_kwargs['pool_paddings'] = [0, 0, 0, 0, 0, 0]
                obs_encoder_kwargs['embedding_dim'] = None
            elif network_type == 'resnet_finetune':
                obs_encoder_class = ResNetVariationalObsEncoder
                obs_encoder_kwargs['fixed'] = False
            elif network_type == 'resnet_fixed':
                obs_encoder_class = ResNetVariationalObsEncoder
                obs_encoder_kwargs['fixed'] = True
                obs_encoder_kwargs['fc_hidden_sizes'] = [128, 128]
            elif network_type == 'resnet_bottleneck64':
                obs_encoder_class = ResNetVariationalObsEncoder
                obs_encoder_kwargs['fixed'] = True
                obs_encoder_kwargs['bottleneck_dim'] = 64
                obs_encoder_kwargs['fc_hidden_sizes'] = [128, 128]
            elif network_type == 'resnet_bottleneck16':
                obs_encoder_class = ResNetVariationalObsEncoder
                obs_encoder_kwargs['fixed'] = True
                obs_encoder_kwargs['bottleneck_dim'] = 16
                obs_encoder_kwargs['fc_hidden_sizes'] = [128, 128]
            elif network_type == 'vqvae':
                obs_encoder_class = VqvaeVariationalObsEncoder
                obs_encoder_kwargs['vqvae'] = vqvae
            else:
                raise ValueError('Unrecognized network_type: %s' %
                                 (network_type))

        else:
            if network_type is None or network_type == 'none':
                obs_encoder_class = VariationalObsEncoder
                obs_encoder_kwargs['input_dim'] = obs_dim
                obs_encoder_kwargs['output_dim'] = obs_encoding_dim
            elif network_type == 'vqvae':
                obs_encoder_class = VqvaeVariationalObsEncoder
                obs_encoder_kwargs['vqvae'] = vqvae
                obs_encoder_kwargs['input_width'] = imsize
                obs_encoder_kwargs['input_height'] = imsize
                obs_encoder_kwargs['input_channels'] = 3
                obs_encoder_kwargs['output_dim'] = obs_encoding_dim
            else:
                raise ValueError('Unrecognized network_type: %s' %
                                 (network_type))

        print('obs_encoder_class: ', obs_encoder_class)
        print('obs_encoder_kwargs: ', obs_encoder_kwargs)

        obs_encoder = obs_encoder_class(
            **obs_encoder_kwargs,
        )

        def create_qf():
            if qf_class in [ConcatMlp]:
                qf_kwargs['input_size'] = obs_encoding_dim * 2 + action_dim
                qf_kwargs['output_size'] = 1
            else:
                qf_kwargs['obs_dim'] = obs_encoding_dim
                qf_kwargs['action_dim'] = action_dim

            return qf_class(
                **qf_kwargs
            )
        qf1 = create_qf()
        qf2 = create_qf()
        target_qf1 = create_qf()
        target_qf2 = create_qf()

        def create_vf():
            if vf_class in [Mlp]:
                vf_kwargs['input_size'] = obs_encoding_dim * 2
                vf_kwargs['output_size'] = 1
            else:
                vf_kwargs['obs_dim'] = obs_encoding_dim

            return vf_class(
                **vf_kwargs
            )
        vf = create_vf()
        plan_vf = create_vf()

        if policy_class is GaussianPolicy:
            assert policy_kwargs['output_activation'] is None

        if use_encoder_in_policy:
            policy_kwargs['obs_encoder'] = obs_encoder
            # assert policy_class == EncodingGaussianPolicy

        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )

        affordance = affordance_networks.SimpleCcVae(
            data_dim=obs_encoding_dim,
            z_dim=affordance_encoding_dim,
        ).to(ptu.device)

    if use_expl_planner:
        expl_env.set_vf(plan_vf)

    if use_eval_planner:
        eval_env.set_vf(plan_vf)

    if trainer_type == 'vib':
        model['affordance'] = affordance
        model['vf'] = plan_vf
        model['qf1'] = qf1
        model['qf2'] = qf2

        # TODO(kuanfang): Debugging!
        if use_expl_planner and expl_planner_type not in ['scripted']:
            # expl_env.set_model(obs_encoder, affordance)
            expl_env.set_model(model)

        if use_eval_planner and eval_planner_type not in ['scripted']:
            eval_env.set_model(model)

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
    else:
        if reward_kwargs['obs_type'] == 'latent':
            obs_keys_to_save.extend(['initial_latent_state'])

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

        # batch['observations'] = np.concatenate(
        #     [obs, context], axis=1)
        # batch['next_observations'] = np.concatenate(
        #     [next_obs, context], axis=1)

        batch['observations'] = obs
        batch['next_observations'] = next_obs
        batch['contexts'] = context

        return batch

    online_replay_buffer_kwargs = online_offline_split_replay_buffer_kwargs[
        'online_replay_buffer_kwargs']
    offline_replay_buffer_kwargs = online_offline_split_replay_buffer_kwargs[
        'offline_replay_buffer_kwargs']

    if (replay_buffer_kwargs['fraction_perturbed_context'] > 0.0 or
            replay_buffer_kwargs['fraction_foresight_context'] > 0.0):
        assert NotImplementedError
        # for rb_kwargs in [
        #         replay_buffer_kwargs,
        #         online_replay_buffer_kwargs,
        #         offline_replay_buffer_kwargs]:
        #     # Use the VQVAE-based affordance.
        #     # rb_kwargs['vqvae'] = vqvae
        #     # rb_kwargs['affordance'] = model['affordance']
        #     rb_kwargs['noise_level'] = 0.5

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
            imsize=imsize,
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
            imsize=imsize,
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
            imsize=imsize,
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
        imsize=imsize,
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

    # eval_policy = MakeDeterministic(policy)
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

    if trainer_kwargs['use_online_beta']:
        if algo_kwargs['start_epoch'] == 0:
            trainer_kwargs['beta'] = trainer_kwargs['beta_online']

    if trainer_kwargs['use_online_quantile']:
        if algo_kwargs['start_epoch'] == 0:
            trainer_kwargs['quantile'] = trainer_kwargs['quantile_online']

    if fix_encoder_online:
        if algo_kwargs['start_epoch'] == 0:
            trainer_kwargs['train_encoder'] = False

    if trainer_kwargs['use_encoding_reward_online']:
        if algo_kwargs['start_epoch'] == 0:
            trainer_kwargs['use_encoding_reward'] = True

    if trainer_type == 'iql':
        raise NotImplementedError

    elif trainer_type == 'vib':
        print('trainer_kwargs: ')
        print(trainer_kwargs)

        trainer = IQLVIBTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            vf=vf,
            plan_vf=plan_vf,
            obs_encoder=obs_encoder,
            obs_dim=obs_dim,
            affordance=affordance,
            vqvae=vqvae,
            **trainer_kwargs
        )

    else:
        raise NotImplementedError

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

    if trainer_kwargs['use_online_beta']:
        def switch_beta(self, epoch):
            if epoch == -1:
                self.trainer.beta = trainer_kwargs['beta_online']
        algorithm.post_epoch_funcs.append(switch_beta)

    if trainer_kwargs['use_online_quantile']:
        def switch_quantile(self, epoch):
            if epoch == -1:
                print('Switching quantile from %f to %f' % (
                    self.trainer.quantile,
                    trainer_kwargs['quantile_online']))
                self.trainer.quantile = trainer_kwargs['quantile_online']
        algorithm.post_epoch_funcs.append(switch_quantile)

    elif trainer_kwargs['use_anneal_beta']:
        def switch_beta(self, epoch):
            if (epoch != algo_kwargs['start_epoch'] and
                    (epoch - algo_kwargs['start_epoch'])
                    % trainer_kwargs['anneal_beta_every'] == 0 and
                    self.trainer.beta * trainer_kwargs['anneal_beta_by']
                    >= trainer_kwargs['anneal_beta_stop_at']):
                self.trainer.beta *= trainer_kwargs['anneal_beta_by']
        algorithm.post_epoch_funcs.append(switch_beta)

    if fix_encoder_online:
        def fix_encoder(self, epoch):
            if epoch == -1:
                print('Fixing the encoder')
            if epoch >= -1:
                self.trainer.train_encoder = False
        algorithm.post_epoch_funcs.append(fix_encoder)

    if trainer_kwargs['use_encoding_reward_online']:
        def switch_to_encoding_reward(self, epoch):
            if epoch >= -1:
                self.trainer.use_encoding_reward = True
        algorithm.post_epoch_funcs.append(switch_to_encoding_reward)

    algorithm.to(ptu.device)

    if save_paths:
        algorithm.post_train_funcs.append(save_paths_fn)

    if save_video:
        assert (num_video_columns * max_path_length <=
                algo_kwargs['num_expl_steps_per_train_loop'])

        expl_save_video_kwargs['include_final_goal'] = use_expl_planner
        eval_save_video_kwargs['include_final_goal'] = use_eval_planner

        if use_vqvae and not finetune_with_obs_encoder:
            expl_save_video_kwargs['decode_image_goal_key'] = 'image_decoded_goal'  # NOQA
            eval_save_video_kwargs['decode_image_goal_key'] = 'image_decoded_goal'  # NOQA

        expl_video_func = RIGVideoSaveFunction(
            vqvae,
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
                vqvae,
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

        if use_vqvae:
            path_loader_kwargs['model'] = model
            path_loader_class = EncoderDictToMDPPathLoader
        else:
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

    if online_offline_split:
        replay_buffer.set_online_mode(True)

    if save_pretrained_algorithm:
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, 'wb'))
        torch.save(data, open(p_path, 'wb'))

    logging.info('Start training...')
    algorithm.train()
