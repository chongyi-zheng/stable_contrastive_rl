import os.path as osp
from collections import OrderedDict
from functools import partial

import gin
import numpy as np
import torch
from gym.wrappers import ClipAction

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.data_management.contextual_replay_buffer import (
    # ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from rlkit.data_management.online_offline_split_replay_buffer import (
    OnlineOfflineSplitReplayBuffer,
)
from rlkit.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    AddImageDistribution,
    PresampledPathDistribution,
)
from rlkit.envs.contextual.latent_distributions import (
    AmortizedConditionalPriorDistribution,
    PresampledPriorDistribution,
    ConditionalPriorDistribution,
    AmortizedPriorDistribution,
    AddDecodedImageDistribution,
    AddLatentDistribution,
    AddGripperStateDistribution,
    PriorDistribution,
    PresamplePriorDistribution,
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv
from rlkit.envs.gripper_state_wrapper import GripperStateWrappedEnv
from rlkit.envs.gripper_state_wrapper import process_gripper_state
from rlkit.envs.images import EnvRenderer
from rlkit.envs.images import InsertImageEnv
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.torch.networks import ConcatMlp, Mlp
from rlkit.torch.networks.cnn import ConcatCNN
from rlkit.torch.networks.cnn import ConcatTwoChannelCNN
from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.iql_trainer import IQLTrainer
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.launchers.contextual.rig.rig_launcher import StateImageGoalDiagnosticsFn
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.util.io import load_local_or_remote_file
from rlkit.visualization.video import RIGVideoSaveFunction
from rlkit.visualization.video import save_paths as save_paths_fn
from rlkit.samplers.data_collector.contextual_path_collector import ContextualPathCollector
from rlkit.samplers.rollout_functions import contextual_rollout

from rlkit.experimental.kuanfang.envs.contextual_env import ContextualEnv
from rlkit.experimental.kuanfang.envs.contextual_env import SubgoalContextualEnv  # NOQA
from rlkit.experimental.kuanfang.envs.contextual_env import NonEpisodicSubgoalContextualEnv  # NOQA
from rlkit.experimental.kuanfang.learning.contextual_replay_buffer import ContextualRelabelingReplayBuffer  # NOQA
from rlkit.experimental.kuanfang.planning.random_planner import RandomPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.mppi_planner import MppiPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.planner import HierarchicalPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.scripted_planner import ScriptedPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.scripted_planner import RandomChoicePlanner  # NOQA
from rlkit.experimental.kuanfang.utils.logging import logger as logging
from rlkit.experimental.kuanfang.utils import io_util


PLANNER_CTORS = {
    'random': RandomPlanner,
    'mppi': MppiPlanner,
    'hierarchical': partial(HierarchicalPlanner,
                            sub_planner_ctor=MppiPlanner,
                            num_levels=3,
                            min_dt=20,
                            ),
}


class RewardFn:
    def __init__(self,
                 env,
                 obs_type='latent',
                 reward_type='dense',
                 epsilon=1.0,
                 use_pretrained_reward_classifier_path=False,
                 pretrained_reward_classifier_path='',
                 ):

        if obs_type == 'latent':
            self.obs_key = 'latent_observation'
            self.goal_key = 'latent_desired_goal'
        elif obs_type == 'state':
            self.obs_key = 'state_observation'
            self.goal_key = 'state_desired_goal'

        self.env = env
        self.reward_type = reward_type
        self.epsilon = epsilon

        if reward_type == 'classifier':
            self.reward_classifier = load_local_or_remote_file(
                pretrained_reward_classifier_path)
            self.sigmoid = torch.nn.Sigmoid()

    def process(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def __call__(self, states, actions, next_states, contexts):
        s = self.process(next_states[self.obs_key])
        c = self.process(contexts[self.goal_key])

        terminal = np.zeros((s.shape[0], ), dtype=np.uint8)

        if self.reward_type == 'dense':
            reward = -np.linalg.norm(s - c, axis=1)

        elif self.reward_type == 'sparse':
            success = np.linalg.norm(s - c, axis=1) < self.epsilon
            reward = success - 1

        elif self.reward_type == 'progress':
            s_tm1 = self.process(states[self.obs_key])
            sd_tm1 = np.square(np.linalg.norm(s_tm1 - c, axis=1))
            sd_t = np.square(np.linalg.norm(s - c, axis=1))
            reward = sd_tm1 - sd_t

        elif self.reward_type == 'highlevel':
            reward = self.env.compute_reward(
                states, actions, next_states, contexts)

        elif self.reward_type == 'classifier':
            s = ptu.from_numpy(s)
            s = s.view(s.shape[0], 5, 12, 12)
            c = ptu.from_numpy(c)
            c = c.view(c.shape[0], 5, 12, 12)
            pred = self.sigmoid(self.reward_classifier(s, c))
            pred = ptu.get_numpy(pred)[..., 0]
            reward = pred - 1.0

        elif self.reward_type in ['sp', 'sparse_progress']:
            success = np.linalg.norm(s - c, axis=1) < self.epsilon
            sparse_reward = success - 1

            s_tm1 = self.process(states[self.obs_key])
            sd_tm1 = np.square(np.linalg.norm(s_tm1 - c, axis=1))
            sd_t = np.square(np.linalg.norm(s - c, axis=1))
            progress_reward = sd_tm1 - sd_t

            reward = sparse_reward + 0.1 * progress_reward

        else:
            raise ValueError(self.reward_type)

        return reward, terminal


def process_args(variant):
    # Maybe adjust the arguments for debugging purposes.
    if variant.get('debug', False):
        # variant['max_path_length'] = 5
        # variant['num_presample'] = 50
        # variant['num_presample'] = 32
        variant.get('algo_kwargs', {}).update(dict(
            batch_size=32,
            start_epoch=-5,
            # start_epoch=0,
            num_epochs=5,
            num_eval_steps_per_epoch=variant['max_path_length'],  # * 5,
            num_expl_steps_per_train_loop=variant['max_path_length'],  # * 5,
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


def add_gripper_state_obs(
    rollout
):
    def wrapper(*args, **kwargs):
        paths = rollout(*args, **kwargs)
        for i in range(paths['observations'].shape[0]):
            d = paths['observations'][i]
            d['gripper_state_observation'] = process_gripper_state(
                d['state_observation'])
            d['gripper_state_desired_goal'] = process_gripper_state(
                d['state_desired_goal'])

        for i in range(paths['next_observations'].shape[0]):
            d = paths['next_observations'][i]
            d['gripper_state_observation'] = process_gripper_state(
                d['state_observation'])
            d['gripper_state_desired_goal'] = process_gripper_state(
                d['state_desired_goal'])
        return paths
    return wrapper


def ptp_experiment(
        max_path_length,
        qf_kwargs,
        vf_kwargs,
        trainer_kwargs,
        replay_buffer_kwargs,
        online_offline_split_replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        use_image=False,
        online_offline_split=False,
        policy_class=None,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        reward_kwargs=None,
        encoder_wrapper=EncoderWrappedEnv,
        observation_key='latent_observation',
        observation_keys=['latent_observation'],
        observation_key_reward_fn=None,
        init_key='initial_latent_state',
        goal_key='latent_desired_goal',
        goal_key_reward_fn=None,
        state_observation_key='state_observation',
        gripper_observation_key='gripper_state_observation',
        state_goal_key='state_desired_goal',
        image_goal_key='image_desired_goal',
        image_init_key='initial_image_observation',
        gripper_goal_key='gripper_state_desired_goal',
        reset_keys_map=None,
        use_gripper_observation=False,

        path_loader_class=EncoderDictToMDPPathLoader,
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
        pretrained_rl_path='',
        use_pretrained_rl_path=False,
        input_representation='',
        goal_representation='',
        presampled_goal_kwargs=None,
        presampled_goals_path='',
        num_presample=50,
        num_video_columns=8,
        init_camera=None,
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

        **kwargs
):
    # Kwarg Definitions
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if reset_keys_map is None:
        reset_keys_map = {}
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

    # Enviorment Wrapping
    logging.info('Creating the environment...')
    renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)

    if goal_key_reward_fn is not None:
        distrib_goal_key = goal_key_reward_fn
    else:
        distrib_goal_key = goal_key

    def contextual_env_distrib_and_reward(
        env_id,
        env_class,
        env_kwargs,
        encoder_wrapper,
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
        vqvae = model['vqvae']
        state_env = get_gym_env(
            env_id,
            env_class=env_class,
            env_kwargs=env_kwargs,
        )
        state_env = ClipAction(state_env)
        renderer = EnvRenderer(
            init_camera=init_camera,
            **renderer_kwargs)
        img_env = InsertImageEnv(
            state_env,
            renderer=renderer)
        encoded_env = encoder_wrapper(
            img_env,
            vqvae,
            step_keys_map=dict(image_observation='latent_observation'),
            reset_keys_map=reset_keys_map,
        )
        if use_gripper_observation:
            encoded_env = GripperStateWrappedEnv(
                encoded_env,
                state_observation_key,
                step_keys_map=dict(
                    gripper_state_observation='gripper_state_observation')
            )

        if goal_sampling_mode == 'vae_prior':
            latent_goal_distribution = PriorDistribution(
                vqvae.representation_size,
                goal_key,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )

        elif goal_sampling_mode == 'amortized_vae_prior':
            latent_goal_distribution = AmortizedPriorDistribution(
                vqvae,
                distrib_goal_key,
                num_presample=num_presample,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )

        elif goal_sampling_mode == 'conditional_vae_prior':
            latent_goal_distribution = ConditionalPriorDistribution(
                vqvae,
                distrib_goal_key,
            )
            if use_image:
                latent_goal_distribution = AddDecodedImageDistribution(
                    latent_goal_distribution,
                    distrib_goal_key,
                    image_goal_key,
                    vqvae,
                )
            diagnostics = StateImageGoalDiagnosticsFn({}, )
        elif goal_sampling_mode == 'amortized_conditional_vae_prior':
            latent_goal_distribution = AmortizedConditionalPriorDistribution(
                vqvae,
                distrib_goal_key,
                num_presample=num_presample,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )

        elif goal_sampling_mode == 'presampled_images':
            diagnostics = state_env.get_contextual_diagnostics

            image_goal_distribution = PresampledPathDistribution(
                presampled_goals_path,
                vqvae.representation_size,
            )

            # Representation Check
            add_distrib = AddLatentDistribution

            # AddLatentDistribution
            latent_goal_distribution = add_distrib(
                image_goal_distribution,
                input_key=image_goal_key,
                output_key=distrib_goal_key,
                model=vqvae,
            )

        elif goal_sampling_mode == 'multiple_goals_not_done_presampled_images':
            diagnostics = state_env.get_contextual_diagnostics
            image_goal_distribution = MultipleGoalsNotDonePresampledPathDistribution(
                presampled_goals_path,
                vqvae.representation_size,
                encoded_env,
                multiple_goals_eval_seeds,
            )

            # Representation Check
            add_distrib = AddLatentDistribution

            # AddLatentDistribution
            latent_goal_distribution = add_distrib(
                image_goal_distribution,
                image_goal_key,
                distrib_goal_key,
                vqvae,
            )
        elif goal_sampling_mode == 'presample_latents':
            diagnostics = StateImageGoalDiagnosticsFn({}, )
            # diagnostics = state_env.get_contextual_diagnostics
            latent_goal_distribution = PresamplePriorDistribution(
                model,
                distrib_goal_key,
                state_env,
                num_presample=num_presample,
                affordance_type='cc_vae',
            )
            if use_image:
                latent_goal_distribution = AddDecodedImageDistribution(
                    latent_goal_distribution,
                    distrib_goal_key,
                    image_goal_key,
                    vqvae,
                )
        elif goal_sampling_mode == 'presampled_latents':
            diagnostics = state_env.get_contextual_diagnostics
            latent_goal_distribution = PresampledPriorDistribution(
                presampled_goals_path,
                distrib_goal_key,
            )
        elif goal_sampling_mode == 'reset_of_env':
            state_goal_env = get_gym_env(
                env_id, env_class=env_class, env_kwargs=env_kwargs)
            state_goal_distribution = GoalDictDistributionFromMultitaskEnv(
                state_goal_env,
                goal_keys=[state_goal_key],
            )
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_goal_distribution,
                image_goal_key=image_goal_key,
                renderer=renderer,
            )
            latent_goal_distribution = AddLatentDistribution(
                image_goal_distribution,
                image_goal_key,
                distrib_goal_key,
                vqvae,
            )
            diagnostics = state_goal_env.get_contextual_diagnostics
        else:
            raise ValueError

        # if use_gripper_observation:
        #     latent_goal_distribution = AddGripperStateDistribution(
        #         latent_goal_distribution,
        #         state_goal_key,
        #         GRIPPER_GOAL_Key,
        #     )

        reward_fn = RewardFn(
            state_env,
            **reward_kwargs
        )

        if not isinstance(diagnostics, list):
            contextual_diagnostics_fns = [diagnostics]
        else:
            contextual_diagnostics_fns = diagnostics

        if use_planner:

            if observation_key == 'image_observation':
                assert init_key == 'initial_image_observation'
                assert goal_key == 'image_desired_goal'
                planner_kwargs['encoding_type'] = None
                # use_encoding = False

            elif observation_key == 'latent_observation':
                assert init_key == 'initial_latent_state'
                assert goal_key == 'latent_desired_goal'
                planner_kwargs['encoding_type'] = 'vqvae'
                # use_encoding = False

                latent_goal_distribution = AddLatentDistribution(
                    latent_goal_distribution,
                    input_key=image_init_key,
                    output_key=init_key,
                    model=vqvae,
                )

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

            if reset_interval <= 0:
                env = SubgoalContextualEnv(
                    encoded_env,
                    context_distribution=latent_goal_distribution,
                    reward_fn=reward_fn,

                    observation_key=observation_key,
                    goal_key=goal_key,
                    use_encoding=False,

                    contextual_diagnostics_fns=contextual_diagnostics_fns,
                    planner=planner,
                    **contextual_env_kwargs,
                )
            else:
                raise NotImplementedError  # Adapt from ptp_vib
                env = NonEpisodicSubgoalContextualEnv(
                    encoded_env,
                    context_distribution=latent_goal_distribution,
                    reward_fn=reward_fn,
                    observation_key=observation_key,
                    contextual_diagnostics_fns=contextual_diagnostics_fns,
                    # Planning.
                    planner=planner,
                    # Reset-free.
                    reset_interval=reset_interval,
                    # Others.
                    **contextual_env_kwargs,
                )

        else:
            env = ContextualEnv(
                encoded_env,
                context_distribution=latent_goal_distribution,
                reward_fn=reward_fn,
                observation_key=observation_key,
                contextual_diagnostics_fns=[diagnostics] if not isinstance(
                    diagnostics, list) else diagnostics,
            )

        return env, latent_goal_distribution, reward_fn

    model = io_util.load_model(pretrained_vae_path)
    path_loader_kwargs['model'] = model  # ['vqvae']

    # Environment Definitions
    expl_env_kwargs = env_kwargs.copy()
    expl_env_kwargs['expl'] = True

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
            encoder_wrapper,
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
            encoder_wrapper,
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

    compare_reward_kwargs = reward_kwargs.copy()
    compare_reward_kwargs['reward_type'] = 'sparse'
    logging.info('Preparing the [training] env and contextual distrib...')
    logging.info('sampling mode: %r', training_goal_sampling_mode)
    logging.info('presampled goals: %r',
                 presampled_goal_kwargs['training_goals'])
    logging.info('presampled goals kwargs: %r',
                 presampled_goal_kwargs['training_goals_kwargs'],
                 )
    logging.info('num_presample: %d', num_presample)
    _, training_context_distrib, compare_reward = (
        contextual_env_distrib_and_reward(
            env_id,
            env_class,
            env_kwargs,
            encoder_wrapper,
            training_goal_sampling_mode,
            presampled_goal_kwargs['training_goals'],
            num_presample,
            reward_kwargs=compare_reward_kwargs,
            presampled_goals_kwargs=(
                presampled_goal_kwargs['training_goals_kwargs']),
            use_planner=False,
            planner_type=None,
            planner_kwargs=None,
            planner_scripted_goals=None,
            contextual_env_kwargs=None,
        ))

    logging.info('Preparing the IQL code...')

    path_loader_kwargs['env'] = eval_env

    # IQL Code
    if add_env_demos:
        path_loader_kwargs['demo_paths'].append(env_demo_path)
    if add_env_offpolicy_data:
        path_loader_kwargs['demo_paths'].append(env_offpolicy_data_path)

    # Key Setting
    context_key = goal_key
    obs_dim = (
        expl_env.observation_space.spaces[observation_key].low.size
        + expl_env.observation_space.spaces[context_key].low.size
    )
    if use_gripper_observation:
        # obs_dim += 7 * 2
        obs_dim += 7
    action_dim = expl_env.action_space.low.size

    state_rewards = reward_kwargs.get('reward_type', 'dense') == 'highlevel'

    mapper_dict = {context_key: observation_key}
    obs_keys = []  # [observation_key]
    cont_keys = [context_key]
    cont_keys_to_save = []

    if goal_key_reward_fn:
        mapper_dict[goal_key_reward_fn] = observation_key_reward_fn
        if goal_key_reward_fn not in obs_keys:
            obs_keys.append(observation_key_reward_fn)
        if goal_key_reward_fn not in cont_keys:
            cont_keys.append(goal_key_reward_fn)

    obs_keys.append(state_observation_key)
    if state_rewards:
        mapper_dict[state_goal_key] = state_observation_key
        obs_keys.append(state_observation_key)
        cont_keys_to_save.append(state_goal_key)
    else:
        obs_keys.extend(list(reset_keys_map.values()))

    if use_gripper_observation:
        # mapper_dict[gripper_goal_key] = gripper_observation_key
        obs_keys.append(gripper_observation_key)
        # cont_keys.append(gripper_goal_key)

    mapper = RemapKeyFn(mapper_dict)

    # Replay Buffer

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
        if len(new_contexts.keys()) > 1:
            context = np.concatenate(tuple(new_contexts.values()), axis=1)
        else:
            context = batch[context_key]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate(
            [next_obs, context], axis=1)
        return batch

    online_replay_buffer_kwargs = online_offline_split_replay_buffer_kwargs[
        'online_replay_buffer_kwargs']
    offline_replay_buffer_kwargs = online_offline_split_replay_buffer_kwargs[
        'offline_replay_buffer_kwargs']

    for rb_kwargs in [
            online_replay_buffer_kwargs,
            offline_replay_buffer_kwargs]:
        rb_kwargs['fraction_next_context'] = (
            replay_buffer_kwargs['fraction_next_context'])
        rb_kwargs['fraction_future_context'] = (
            replay_buffer_kwargs['fraction_future_context'])
        rb_kwargs['fraction_foresight_context'] = (
            replay_buffer_kwargs['fraction_foresight_context'])
        rb_kwargs['fraction_perturbed_context'] = (
            replay_buffer_kwargs['fraction_perturbed_context'])
        rb_kwargs['fraction_distribution_context'] = (
            replay_buffer_kwargs['fraction_distribution_context'])
        rb_kwargs['max_future_dt'] = (
            replay_buffer_kwargs['max_future_dt'])

    if (replay_buffer_kwargs['fraction_perturbed_context'] > 0.0 or
            replay_buffer_kwargs['fraction_foresight_context'] > 0.0):
        for rb_kwargs in [
                replay_buffer_kwargs,
                online_replay_buffer_kwargs,
                offline_replay_buffer_kwargs]:
            rb_kwargs['vqvae'] = model['vqvae']
            rb_kwargs['affordance'] = model['affordance']
            rb_kwargs['noise_level'] = 0.5

    if online_offline_split:
        online_replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys,
            observation_key=(
                observation_key if observation_keys is None else None),
            observation_key_reward_fn=observation_key_reward_fn,
            observation_keys=observation_keys,
            context_distribution=training_context_distrib,
            sample_context_from_obs_dict_fn=mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            context_keys_to_save=cont_keys_to_save,
            **online_replay_buffer_kwargs,
        )
        offline_replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys,
            observation_key=(
                observation_key if observation_keys is None else None),
            observation_key_reward_fn=observation_key_reward_fn,
            observation_keys=observation_keys,
            context_distribution=training_context_distrib,
            sample_context_from_obs_dict_fn=mapper,
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

    else:
        replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys,
            observation_key=(
                observation_key if observation_keys is None else None),
            observation_key_reward_fn=observation_key_reward_fn,
            observation_keys=observation_keys,
            context_distribution=training_context_distrib,
            sample_context_from_obs_dict_fn=mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            context_keys_to_save=cont_keys_to_save,
            **replay_buffer_kwargs
        )

    if use_pretrained_rl_path:
        logging.info('Loading pretrained RL from: %s', pretrained_rl_path)
        rl_model_dict = load_local_or_remote_file(pretrained_rl_path)
        qf1 = rl_model_dict['trainer/qf1']
        qf2 = rl_model_dict['trainer/qf2']
        target_qf1 = rl_model_dict['trainer/target_qf1']
        target_qf2 = rl_model_dict['trainer/target_qf2']
        vf = rl_model_dict['trainer/vf']
        policy = rl_model_dict['trainer/policy']
        if 'std' in policy_kwargs and policy_kwargs['std'] is not None:
            policy.std = policy_kwargs['std']
            policy.log_std = np.log(policy.std)
    else:
        # Neural Network Architecture
        def create_qf():
            if qf_class is ConcatMlp:
                qf_kwargs['input_size'] = obs_dim + action_dim
            if qf_class is ConcatCNN or qf_class is ConcatTwoChannelCNN:
                qf_kwargs['added_fc_input_size'] = action_dim
            return qf_class(
                output_size=1,
                **qf_kwargs
            )
        qf1 = create_qf()
        qf2 = create_qf()
        target_qf1 = create_qf()
        target_qf2 = create_qf()

        def create_vf():
            if vf_class is Mlp:
                vf_kwargs['input_size'] = obs_dim
            return vf_class(
                output_size=1,
                **vf_kwargs
            )
        vf = create_vf()

        if policy_class is GaussianPolicy:
            assert policy_kwargs['output_activation'] is None

        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )

    # Path Collectors
    path_collector_observation_keys = [
        observation_key, ] if observation_keys is None else observation_keys
    # path_collector_context_keys_for_policy = (
    #     [context_key, ] if not use_gripper_observation
    #     else [context_key, gripper_goal_key])
    path_collector_context_keys_for_policy = [context_key, ]

    def obs_processor(o):
        combined_obs = []

        for k in path_collector_observation_keys:
            if k == gripper_observation_key:
                gripper_state = process_gripper_state(
                    o['state_observation'])
                combined_obs.append(gripper_state)
            else:
                combined_obs.append(o[k])

        for k in path_collector_context_keys_for_policy:
            # if k == gripper_goal_key:
            #     gripper_state = process_gripper_state(
            #         o['state_desired_goal'])
            #     combined_obs.append(gripper_state)
            # else:
            #     combined_obs.append(o[k])
            combined_obs.append(o[k])

        return np.concatenate(combined_obs, axis=0)

    rollout = contextual_rollout
    if use_gripper_observation:
        rollout = add_gripper_state_obs(rollout)

    eval_policy = policy

    eval_path_collector = ContextualPathCollector(
        eval_env,
        eval_policy,
        observation_keys=path_collector_observation_keys,
        context_keys_for_policy=path_collector_context_keys_for_policy,
        obs_processor=obs_processor,
        rollout=rollout,
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
        obs_processor=obs_processor,
        rollout=rollout,
    )

    if trainer_type == 'iql':
        if trainer_kwargs['use_online_beta']:
            if algo_kwargs['start_epoch'] == 0:
                trainer_kwargs['beta'] = trainer_kwargs['beta_online']

        if trainer_kwargs['use_online_quantile']:
            if algo_kwargs['start_epoch'] == 0:
                trainer_kwargs['quantile'] = trainer_kwargs['quantile_online']

    if use_expl_planner:
        expl_env.set_vf(vf)

    if use_eval_planner:
        eval_env.set_vf(vf)

    model['vf'] = vf
    model['qf1'] = qf1
    model['qf2'] = qf2

    if use_expl_planner and expl_planner_type not in ['scripted']:
        expl_env.set_model(model)

    if use_eval_planner and eval_planner_type not in ['scripted']:
        eval_env.set_model(model)

    # Algorithm
    if trainer_type == 'iql':
        trainer = IQLTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            vf=vf,
            **trainer_kwargs
        )

    elif trainer_type == 'sac':
        trainer = SACTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
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
        max_path_length=max_path_length,
        **algo_kwargs
    )

    if trainer_type == 'iql':
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

    algorithm.to(ptu.device)

    # Video Saving
    if save_video:
        assert (num_video_columns * max_path_length <=
                algo_kwargs['num_expl_steps_per_train_loop'])

        expl_save_video_kwargs['include_final_goal'] = use_expl_planner
        eval_save_video_kwargs['include_final_goal'] = use_eval_planner

        expl_save_video_kwargs['decode_image_goal_key'] = 'image_decoded_goal'
        eval_save_video_kwargs['decode_image_goal_key'] = 'image_decoded_goal'

        expl_video_func = RIGVideoSaveFunction(
            model['vqvae'],
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
                model['vqvae'],
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

    # IQL CODE
    if save_paths:
        algorithm.post_train_funcs.append(save_paths_fn)

    if online_offline_split:
        replay_buffer.set_online_mode(False)

    if load_demos:
        demo_train_buffer = None
        demo_test_buffer = None
        path_loader = path_loader_class(trainer,
                                        replay_buffer=replay_buffer,
                                        demo_train_buffer=demo_train_buffer,
                                        demo_test_buffer=demo_test_buffer,
                                        reward_fn=eval_reward,
                                        compare_reward_fn=compare_reward,
                                        **path_loader_kwargs
                                        )
        path_loader.load_demos()

    if save_pretrained_algorithm:
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, 'wb'))
        torch.save(data, open(p_path, 'wb'))

    if online_offline_split:
        replay_buffer.set_online_mode(True)

    logging.info('Start training...')
    algorithm.train()
