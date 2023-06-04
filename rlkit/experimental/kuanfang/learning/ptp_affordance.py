import os.path  # NOQA
import sys  # NOQA
from collections import OrderedDict  # NOQA
from functools import partial  # NOQA

import gin  # NOQA
import numpy as np
import torch  # NOQA
from gym.wrappers import ClipAction

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger  # NOQA
from rlkit.data_management.contextual_replay_buffer import (
    RemapKeyFn,
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
from rlkit.torch.sac.ptp_affordance_trainer import PTPAffordanceTrainer  # NOQA
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.util.io import load_local_or_remote_file

from rlkit.experimental.kuanfang.envs.reward_fns import GoalReachingRewardFn
from rlkit.experimental.kuanfang.envs.contextual_env import ContextualEnv
from rlkit.experimental.kuanfang.envs.contextual_env import SubgoalContextualEnv  # NOQA
from rlkit.experimental.kuanfang.envs.contextual_env import NonEpisodicSubgoalContextualEnv  # NOQA
from rlkit.experimental.kuanfang.learning.affordance_replay_buffer import AffordanceReplayBuffer  # NOQA
from rlkit.experimental.kuanfang.utils.logging import logger as logging
from rlkit.experimental.kuanfang.utils import io_util

from rlkit.experimental.kuanfang.vae import affordance_networks

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
        # variant['max_path_length'] = 5
        variant.get('algo_kwargs', {}).update(dict(
            batch_size=128,
            start_epoch=-5,
            # start_epoch=0,
            num_epochs=5,
            num_trains_per_train_loop=2,
        ))
        demo_paths = variant['path_loader_kwargs'].get('demo_paths', [])
        if len(demo_paths) > 1:
            variant['path_loader_kwargs']['demo_paths'] = [demo_paths[0]]


def ptp_affordance_experiment(  # NOQA
        max_path_length,
        qf_kwargs,
        vf_kwargs,
        obs_encoder_kwargs,
        affordance_kwargs,
        trainer_kwargs,
        replay_buffer_kwargs,
        algo_kwargs,
        obs_encoding_dim,
        affordance_encoding_dim,
        network_type,   # TODO
        use_image=False,
        finetune_with_obs_encoder=True,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        reward_kwargs=None,
        policy_kwargs=None,

        reset_keys_map=None,

        path_loader_kwargs=None,
        env_demo_path='',

        debug=False,
        epsilon=1.0,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        training_goal_sampling_mode=None,

        add_env_demos=False,
        save_paths=True,
        pretrain_rl=False,
        save_pretrained_algorithm=False,

        renderer_kwargs=None,
        imsize=84,
        pretrained_vae_path=None,
        pretrained_rl_path=None,
        presampled_goal_kwargs=None,
        presampled_goals_path=None,
        num_presample=50,
        init_camera=None,
        qf_class=ConcatMlp,
        vf_class=Mlp,
        env_type=None,  # For plotting
        seed=None,
        multiple_goals_eval_seeds=None,

        augment_order=[],
        augment_params=dict(),
        augment_probability=0.0,

        fix_encoder_online=False,

        **kwargs
):
    # Kwarg Definitions
    if reset_keys_map is None:
        reset_keys_map = {}
    if presampled_goal_kwargs is None:
        presampled_goal_kwargs = \
            {'eval_goals': '', 'expl_goals': '', 'training_goals': ''}
    if path_loader_kwargs is None:
        path_loader_kwargs = {}

    if finetune_with_obs_encoder:
        assert augment_probability == 0.0
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
    if obs_type == 'image' and reward_kwargs['obs_type'] == 'latent':
        obs_key_reward_fn = 'latent_observation'
        goal_key_reward_fn = 'latent_desired_goal'

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

    ########################################
    # Enviorments
    ########################################
    logging.info('Creating the environment...')

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
            vqvae = model['vqvae']
            env = EncoderWrappedEnv(
                env,
                vqvae,
                step_keys_map=dict(image_observation='latent_observation'),
                reset_keys_map=reset_keys_map,
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
    logging.info('use_image: %s', use_image)

    logging.info('Preparing the [training] env and contextual distrib...')
    logging.info('sampling mode: %r', training_goal_sampling_mode)
    logging.info('presampled goals: %r',
                 presampled_goal_kwargs['eval_goals'])
    logging.info('presampled goals kwargs: %r',
                 presampled_goal_kwargs['training_goals_kwargs'],
                 )
    logging.info('num_presample: %d', num_presample)
    env, training_context_distrib, reward_fn = (  # TODO(kuanfang)
        contextual_env_distrib_and_reward(
            env_id,
            env_class,
            env_kwargs,
            training_goal_sampling_mode,
            presampled_goal_kwargs['eval_goals'],
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
        env.observation_space.spaces[obs_key].low.size
        == env.observation_space.spaces[goal_key].low.size)
    obs_dim = env.observation_space.spaces[obs_key].low.size

    ########################################
    # Neural Network Architecture
    ########################################
    logging.info('Creating the models...')

    assert pretrained_rl_path is not None
    logging.info('Loading pretrained RL from: %s', pretrained_rl_path)
    rl_model_dict = load_local_or_remote_file(pretrained_rl_path)
    # affordance = rl_model_dict['trainer/affordance']
    qf1 = rl_model_dict['trainer/qf1']
    qf2 = rl_model_dict['trainer/qf2']
    target_qf1 = rl_model_dict['trainer/target_qf1']
    target_qf2 = rl_model_dict['trainer/target_qf2']
    vf = rl_model_dict['trainer/vf']
    plan_vf = rl_model_dict['trainer/plan_vf']

    if not finetune_with_obs_encoder:
        obs_encoder = rl_model_dict['trainer/obs_encoder']

    policy = rl_model_dict['trainer/policy']
    if 'std' in policy_kwargs and policy_kwargs['std'] is not None:
        policy.std = policy_kwargs['std']
        policy.log_std = np.log(policy.std)
    policy.obs_encoder = obs_encoder  # TODO

    if finetune_with_obs_encoder:
        policy._always_use_encoded_input = True
        policy._goal_is_encoded = True
        trainer_kwargs['use_obs_encoder'] = True
        trainer_kwargs['goal_is_encoded'] = True
        policy_kwargs['goal_is_encoded'] = True
    else:
        trainer_kwargs['use_obs_encoder'] = False

    ########################################
    # Replay Buffer
    ########################################
    logging.info('Creating the replay buffer...')

    context2obs = {goal_key: obs_key}
    cont_keys = [goal_key]

    obs_keys_to_save = []
    cont_keys_to_save = []

    if reward_kwargs['obs_type'] != obs_type:
        assert obs_key_reward_fn == reward_fn.obs_key
        assert goal_key_reward_fn == reward_fn.goal_key
        context2obs[goal_key_reward_fn] = obs_key_reward_fn
        obs_keys_to_save.append(obs_key_reward_fn)
        cont_keys.append(goal_key_reward_fn)

    if reward_kwargs.get('reward_type', 'dense') == 'highlevel':
        context2obs[state_goal_key] = state_obs_key
        obs_keys_to_save.append(state_obs_key)
        cont_keys_to_save.append(state_goal_key)
    else:
        if reward_kwargs['obs_type'] == 'latent':
            obs_keys_to_save.extend(list(reset_keys_map.values()))

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

    replay_buffer = AffordanceReplayBuffer(
        env=env,
        context_keys=cont_keys,
        observation_keys_to_save=obs_keys_to_save,
        observation_key=obs_key,
        context_distribution=training_context_distrib,
        sample_context_from_obs_dict_fn=context2obs_mapper,
        reward_fn=reward_fn,
        post_process_batch_fn=concat_context_to_obs,
        context_keys_to_save=cont_keys_to_save,
        imsize=imsize,
        **replay_buffer_kwargs
    )

    eval_replay_buffer = AffordanceReplayBuffer(
        env=env,
        context_keys=cont_keys,
        observation_keys_to_save=obs_keys_to_save,
        observation_key=obs_key,
        context_distribution=training_context_distrib,
        sample_context_from_obs_dict_fn=context2obs_mapper,
        reward_fn=reward_fn,
        post_process_batch_fn=concat_context_to_obs,
        context_keys_to_save=cont_keys_to_save,
        imsize=imsize,
        **replay_buffer_kwargs
    )

    ########################################
    # Training Algorithm
    ########################################
    logging.info('Creating the training algorithm...')

    if finetune_with_obs_encoder:
        # TODO: Debugging!!!
        # affordance_ctor = affordance_networks.SimpleCcVae
        affordance_ctor = affordance_networks.DeltaSimpleCcVae
        affordance = affordance_ctor(
            data_dim=obs_encoding_dim,
            z_dim=affordance_encoding_dim,
            **affordance_kwargs,
        )

    else:
        affordance = affordance_networks.CcVae(
            data_channels=5,
            z_dim=affordance_encoding_dim,
        ).to(ptu.device)

    trainer = PTPAffordanceTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vf=vf,
        plan_vf=plan_vf,
        obs_encoder=obs_encoder,
        affordance=affordance,
        vqvae=vqvae,
        obs_dim=obs_dim,
        **trainer_kwargs
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,

        # No need to use environments or path collectors.
        exploration_env=None,
        evaluation_env=None,
        exploration_data_collector=None,
        evaluation_data_collector=None,

        replay_buffer=replay_buffer,
        eval_replay_buffer=eval_replay_buffer,  # TODO
        max_path_length=max_path_length,
        **algo_kwargs
    )

    algorithm.to(ptu.device)

    if add_env_demos:
        path_loader_kwargs['demo_paths'].append(env_demo_path)

    demo_train_buffer = None
    demo_test_buffer = eval_replay_buffer

    if use_vqvae:
        path_loader_kwargs['model'] = model
        path_loader_class = EncoderDictToMDPPathLoader
    else:
        path_loader_class = DictToMDPPathLoader

    path_loader = path_loader_class(
        env=env,
        trainer=trainer,
        replay_buffer=replay_buffer,
        demo_train_buffer=demo_train_buffer,
        demo_test_buffer=demo_test_buffer,
        reward_fn=reward_fn,
        **path_loader_kwargs
    )
    path_loader.load_demos()

    logging.info('Start training...')
    algorithm.train()
