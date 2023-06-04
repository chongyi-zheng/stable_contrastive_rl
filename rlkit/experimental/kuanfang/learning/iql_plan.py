import os.path as osp
from collections import OrderedDict  # NOQA
from functools import partial

import gin  # NOQA
import numpy as np
import torch

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
    # NotDonePresampledPathDistribution,
    # MultipleGoalsNotDonePresampledPathDistribution,
    # TwoDistributions,
)
from rlkit.envs.contextual.latent_distributions import (
    AmortizedConditionalPriorDistribution,
    PresampledPriorDistribution,
    ConditionalPriorDistribution,
    AmortizedPriorDistribution,
    AddDecodedImageDistribution,
    AddLatentDistribution,
    # AddGripperStateDistribution,
    PriorDistribution,
    PresamplePriorDistribution,
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv
# from rlkit.envs.gripper_state_wrapper import GripperStateWrappedEnv
from rlkit.envs.gripper_state_wrapper import process_gripper_state
from rlkit.envs.images import EnvRenderer
from rlkit.envs.images import InsertImageEnv
from rlkit.demos.source.mdp_path_loader import MDPPathLoader  # NOQA
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader  # NOQA
from rlkit.torch.networks import ConcatMlp, Mlp
from rlkit.torch.networks.cnn import ConcatCNN
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.iql_trainer import IQLTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.launchers.contextual.rig.rig_launcher import StateImageGoalDiagnosticsFn  # NOQA
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.util.io import load_local_or_remote_file
from rlkit.visualization.video import RIGVideoSaveFunction
from rlkit.visualization.video import save_paths as save_paths_fn
from rlkit.samplers.data_collector.contextual_path_collector import ContextualPathCollector  # NOQA
from rlkit.samplers.rollout_functions import contextual_rollout

from rlkit.experimental.kuanfang.envs.contextual_env import ContextualEnv
from rlkit.experimental.kuanfang.envs.contextual_env import SubgoalContextualEnv  # NOQA
from rlkit.experimental.kuanfang.envs.contextual_env import NonEpisodicSubgoalContextualEnv  # NOQA
from rlkit.experimental.kuanfang.learning.contextual_replay_buffer import ContextualRelabelingReplayBuffer  # NOQA
from rlkit.experimental.kuanfang.planning.random_planner import RandomPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.mppi_planner import MppiPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.planner import HierarchicalPlanner  # NOQA
from rlkit.experimental.kuanfang.planning.scripted_planner import ScriptedPlanner  # NOQA
from rlkit.experimental.kuanfang.utils.logging import logger as logging
from rlkit.experimental.kuanfang.utils import io_util


PLANNER_CTORS = {
    'random': RandomPlanner,
    'mppi': MppiPlanner,
    'hierarchical': partial(HierarchicalPlanner,
                            sub_planner_ctor=MppiPlanner,
                            num_levels=3,
                            min_dt=20,
                            )
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
            self.observation_key = 'latent_observation'
            self.desired_goal_key = 'latent_desired_goal'
        elif obs_type == 'state':
            self.observation_key = 'state_observation'
            self.desired_goal_key = 'state_desired_goal'

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
        s = self.process(next_states[self.observation_key])
        c = self.process(contexts[self.desired_goal_key])

        if self.reward_type == 'dense':
            reward = -np.linalg.norm(s - c, axis=1)

        elif self.reward_type == 'sparse':
            success = np.linalg.norm(s - c, axis=1) < self.epsilon
            reward = success - 1

        elif self.reward_type == 'progress':
            s_tm1 = self.process(states[self.observation_key])
            sd_tm1 = np.square(np.linalg.norm(s_tm1 - c, axis=1))
            sd_t = np.square(np.linalg.norm(s - c, axis=1))
            reward = sd_tm1 - sd_t

        elif self.reward_type == 'wrapped_env':
            reward = self.env.compute_reward(states, actions, next_states,
                                             contexts)

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

            s_tm1 = self.process(states[self.observation_key])
            sd_tm1 = np.square(np.linalg.norm(s_tm1 - c, axis=1))
            sd_t = np.square(np.linalg.norm(s - c, axis=1))
            progress_reward = sd_tm1 - sd_t

            reward = sparse_reward + 0.1 * progress_reward

        else:
            raise ValueError(self.reward_type)

        return reward


def process_args(variant):
    # Maybe adjust the arguments for debugging purposes.
    if variant.get('debug', False):
        # variant['max_path_length'] = 50
        # variant['num_presample'] = 50
        # variant['num_presample'] = 32
        # variant.get('algo_kwargs', {}).update(dict(
        #     batch_size=64,
        #     # start_epoch=-5,
        #     start_epoch=0,
        #     num_epochs=5,
        #     num_eval_steps_per_epoch=variant['max_path_length'] * 5,
        #     num_expl_steps_per_train_loop=variant['max_path_length'] * 5,
        #     num_trains_per_train_loop=2,
        #     num_online_trains_per_train_loop=2,
        #     min_num_steps_before_training=2,
        # ))
        variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = int(5E2)  # NOQA
        variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = int(5E2)  # NOQA
        demo_paths = variant['path_loader_kwargs'].get('demo_paths', [])
        if len(demo_paths) > 1:
            variant['path_loader_kwargs']['demo_paths'] = [demo_paths[0]]


# def gripper_state_contextual_rollout(
#     env,
#     agent,
#     **kwargs
# ):
#     paths = contextual_rollout(
#         env,
#         agent,
#         **kwargs
#     )
#
#     for i in range(paths['observations'].shape[0]):
#         d = paths['observations'][i]
#         d['gripper_state_observation'] = process_gripper_state(
#             d['state_observation'])
#         d['gripper_state_desired_goal'] = process_gripper_state(
#             d['state_desired_goal'])
#
#     for i in range(paths['next_observations'].shape[0]):
#         d = paths['next_observations'][i]
#         d['gripper_state_observation'] = process_gripper_state(
#             d['state_observation'])
#         d['gripper_state_desired_goal'] = process_gripper_state(
#             d['state_desired_goal'])
#
#     return paths


def iql_plan_experiment(  # NOQA
        max_path_length,
        qf_kwargs,
        vf_kwargs,
        trainer_kwargs,
        replay_buffer_kwargs,
        online_offline_split_replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        image=False,
        online_offline_split=False,
        policy_class=TanhGaussianPolicy,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        reward_kwargs=None,
        encoder_wrapper=EncoderWrappedEnv,
        observation_key='latent_observation',
        observation_keys=['latent_observation'],
        observation_key_reward_fn=None,
        desired_goal_key='latent_desired_goal',
        desired_goal_key_reward_fn=None,
        state_observation_key='state_observation',
        gripper_observation_key='gripper_state_observation',
        state_goal_key='state_desired_goal',
        image_goal_key='image_desired_goal',
        image_init_key='initial_image_observation',
        gripper_goal_key='gripper_state_desired_goal',
        reset_keys_map=None,
        gripper_observation=False,

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

        # Video parameters
        save_video=True,
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
        num_presample=50,  # TODO(kuanfang)
        num_video_columns=8,
        init_camera=None,
        qf_class=ConcatMlp,
        vf_class=Mlp,
        env_type=None,  # For plotting
        seed=None,
        multiple_goals_eval_seeds=None,
        subgoal_planning_kwargs=None,
        reset_interval=1,
        use_planning_expl=True,
        use_planning_eval=False,
        planner_type='hierarchical',
        planner_kwargs={},
        **kwargs
):
    assert gripper_observation is False
    # assert use_pretrained_rl_path is True

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
    if not subgoal_planning_kwargs:
        subgoal_planning_kwargs = {}

    # Enviorment Wrapping
    logging.info('Creating the environment...')
    renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)

    if desired_goal_key_reward_fn is not None:
        distrib_desired_goal_key = desired_goal_key_reward_fn
    else:
        distrib_desired_goal_key = desired_goal_key

    def contextual_env_distrib_and_reward(
        env_id,
        env_class,
        env_kwargs,
        encoder_wrapper,
        goal_sampling_mode,
        presampled_goals_path,
        num_presample,
        use_planning,
        reward_kwargs,
        presampled_goals_kwargs,
    ):
        vqvae = model['vqvae']
        # env_kwargs['debug'] = use_planning  # TODO(kuanfang): Debugging.
        state_env = get_gym_env(
            env_id,
            env_class=env_class,
            env_kwargs=env_kwargs,
        )
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
        # if gripper_observation:
        #     encoded_env = GripperStateWrappedEnv(
        #         encoded_env,
        #         state_observation_key,
        #         step_keys_map=dict(
        #             gripper_state_observation='gripper_state_observation')
        #     )

        if goal_sampling_mode == 'vae_prior':
            latent_goal_distribution = PriorDistribution(
                vqvae.representation_size,
                desired_goal_key,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )

        elif goal_sampling_mode == 'amortized_vae_prior':
            latent_goal_distribution = AmortizedPriorDistribution(
                vqvae,
                distrib_desired_goal_key,
                num_presample=num_presample,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )

        elif goal_sampling_mode == 'conditional_vae_prior':
            latent_goal_distribution = ConditionalPriorDistribution(
                vqvae,
                distrib_desired_goal_key,
            )
            if image:
                latent_goal_distribution = AddDecodedImageDistribution(
                    latent_goal_distribution,
                    distrib_desired_goal_key,
                    image_goal_key,
                    vqvae,
                )
            diagnostics = StateImageGoalDiagnosticsFn({}, )
        elif goal_sampling_mode == 'amortized_conditional_vae_prior':
            latent_goal_distribution = AmortizedConditionalPriorDistribution(
                vqvae,
                distrib_desired_goal_key,
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
                output_key=distrib_desired_goal_key,
                model=vqvae,
            )

        elif goal_sampling_mode == 'multiple_goals_not_done_presampled_images':
            diagnostics = state_env.get_contextual_diagnostics
            image_goal_distribution = MultipleGoalsNotDonePresampledPathDistribution(  # NOQA
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
                distrib_desired_goal_key,
                vqvae,
            )
        elif goal_sampling_mode == 'presample_latents':
            diagnostics = StateImageGoalDiagnosticsFn({}, )
            # diagnostics = state_env.get_contextual_diagnostics
            latent_goal_distribution = PresamplePriorDistribution(
                model,
                distrib_desired_goal_key,
                state_env,
                num_presample=num_presample,
                affordance_type='cc_vae',
            )
            if image:
                latent_goal_distribution = AddDecodedImageDistribution(
                    latent_goal_distribution,
                    distrib_desired_goal_key,
                    image_goal_key,
                    vqvae,
                )
        elif goal_sampling_mode == 'presampled_latents':
            diagnostics = state_env.get_contextual_diagnostics
            latent_goal_distribution = PresampledPriorDistribution(
                presampled_goals_path,
                distrib_desired_goal_key,
            )
        elif goal_sampling_mode == 'reset_of_env':
            state_goal_env = get_gym_env(
                env_id, env_class=env_class, env_kwargs=env_kwargs)
            state_goal_distribution = GoalDictDistributionFromMultitaskEnv(
                state_goal_env,
                desired_goal_keys=[state_goal_key],
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
                distrib_desired_goal_key,
                vqvae,
            )
            diagnostics = state_goal_env.get_contextual_diagnostics
        else:
            raise ValueError

        # if gripper_observation:
        #     latent_goal_distribution = AddGripperStateDistribution(
        #         latent_goal_distribution,
        #         state_goal_key,
        #         gripper_goal_key,
        #     )

        reward_fn = RewardFn(
            state_env,
            **reward_kwargs
        )

        if use_planning:

            distrib_init_key = 'initial_latent_state'
            add_distrib = AddLatentDistribution
            latent_goal_distribution = add_distrib(
                latent_goal_distribution,
                input_key=image_init_key,
                output_key=distrib_init_key,
                model=vqvae,
            )

            if planner_type == 'scripted':
                planner = ScriptedPlanner(
                    model,
                    path=presampled_goal_kwargs['eval_goals'],
                    buffer_size=0)
            else:
                assert model['affordance'] is not None
                planner_ctor = PLANNER_CTORS[planner_type]
                planner = planner_ctor(model, debug=False, **planner_kwargs)

            assert reset_interval == 1

            # env = NonEpisodicSubgoalContextualEnv(
            #     encoded_env,
            #     context_distribution=latent_goal_distribution,
            #     # initial_distribution=latent_init_distribution,
            #     initial_distribution=None,
            #
            #     reward_fn=reward_fn,
            #     observation_key=observation_key,
            #     contextual_diagnostics_fns=[diagnostics] if not isinstance(
            #         diagnostics, list) else diagnostics,
            #     # Planning.
            #     planner=planner,
            #     initial_state_key=distrib_init_key,
            #     desired_goal_key=desired_goal_key,
            #     # subgoal_timeout=None,
            #     # subgoal_switch_reward_thresh=None,
            #
            #     reset_interval=reset_interval,
            #
            #     **subgoal_planning_kwargs,
            # )

            env = SubgoalContextualEnv(
                encoded_env,
                context_distribution=latent_goal_distribution,
                reward_fn=reward_fn,
                observation_key=observation_key,
                contextual_diagnostics_fns=[diagnostics] if not isinstance(
                    diagnostics, list) else diagnostics,
                # Planning.
                planner=planner,
                **subgoal_planning_kwargs,
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
    path_loader_kwargs['model'] = model['vqvae']

    # Environment Definitions
    # TODO(kuanfang): What ais expl used for?
    expl_env_kwargs = env_kwargs.copy()
    expl_env_kwargs['expl'] = True
    #
    # eval_env_kwargs = env_kwargs.copy()
    # eval_env_kwargs['expl'] = False

    # TODO(kuanfang): Use exactly the same arguments as in the eval_env.
    exploration_goal_sampling_mode = evaluation_goal_sampling_mode
    presampled_goal_kwargs['expl_goals'] = presampled_goal_kwargs['eval_goals']
    presampled_goal_kwargs['expl_goals_kwargs'] = (
        presampled_goal_kwargs['eval_goals_kwargs'])

    logging.info('image: %s', image)

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
            use_planning=use_planning_expl,
            reward_kwargs=reward_kwargs,
            presampled_goals_kwargs=(
                presampled_goal_kwargs['expl_goals_kwargs']),
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
            use_planning=use_planning_eval,
            reward_kwargs=reward_kwargs,
            presampled_goals_kwargs=(
                presampled_goal_kwargs['eval_goals_kwargs']),
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
    _, training_context_distrib, compare_reward = (  # TODO(kuanfang)
        contextual_env_distrib_and_reward(
            env_id,
            env_class,
            env_kwargs,
            encoder_wrapper,
            training_goal_sampling_mode,
            presampled_goal_kwargs['training_goals'],
            num_presample,
            use_planning=False,
            reward_kwargs=compare_reward_kwargs,
            presampled_goals_kwargs=(
                presampled_goal_kwargs['training_goals_kwargs']),
        ))

    logging.info('Preparing the IQL code...')

    path_loader_kwargs['env'] = eval_env

    # IQL Code
    if add_env_demos:
        path_loader_kwargs['demo_paths'].append(env_demo_path)
    if add_env_offpolicy_data:
        path_loader_kwargs['demo_paths'].append(env_offpolicy_data_path)

    # Key Setting
    context_key = desired_goal_key
    obs_dim = (
        expl_env.observation_space.spaces[observation_key].low.size
        + expl_env.observation_space.spaces[context_key].low.size
    )
    # if gripper_observation:
    #     obs_dim += 6*2
    action_dim = expl_env.action_space.low.size

    state_rewards = reward_kwargs.get('reward_type', 'dense') == 'wrapped_env'

    mapper_dict = {context_key: observation_key}
    obs_keys = [observation_key]
    cont_keys = [context_key]

    if desired_goal_key_reward_fn:
        mapper_dict[desired_goal_key_reward_fn] = observation_key_reward_fn
        obs_keys.append(observation_key_reward_fn)
        cont_keys.append(desired_goal_key_reward_fn)

    obs_keys.append(state_observation_key)
    if state_rewards:
        mapper_dict[state_goal_key] = state_observation_key
        cont_keys.append(state_goal_key)
    else:
        obs_keys.extend(list(reset_keys_map.values()))

    # if gripper_observation:
    #     mapper_dict[gripper_goal_key] = gripper_observation_key
    #     obs_keys.append(gripper_observation_key)
    #     cont_keys.append(gripper_goal_key)

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
            if qf_class is ConcatCNN:
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

        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )

    # Path Collectors
    path_collector_observation_keys = [
        observation_key, ] if observation_keys is None else observation_keys
    # path_collector_context_keys_for_policy = (
    #     [context_key, ] if not gripper_observation
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
            if k == gripper_goal_key:
                gripper_state = process_gripper_state(
                    o['state_desired_goal'])
                combined_obs.append(gripper_state)
            else:
                combined_obs.append(o[k])

        return np.concatenate(combined_obs, axis=0)

    # if gripper_observation:
    #     rollout = gripper_state_contextual_rollout
    # else:
    #     rollout = contextual_rollout
    rollout = contextual_rollout

    eval_path_collector = ContextualPathCollector(
        eval_env,
        MakeDeterministic(policy),
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

    if trainer_kwargs['use_online_beta']:
        if algo_kwargs['start_epoch'] == 0:
            trainer_kwargs['beta'] = trainer_kwargs['beta_online']

    if use_planning_expl:
        expl_env.set_vf(vf)

    # Algorithm
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

    if trainer_kwargs['use_online_beta']:
        def switch_beta(self, epoch):
            if epoch == -1:
                self.trainer.beta = trainer_kwargs['beta_online']
        algorithm.post_epoch_funcs.append(switch_beta)

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

        if use_planning_expl:
            expl_save_video_kwargs['include_final_goal'] = True

        expl_video_func = RIGVideoSaveFunction(
            model['vqvae'],
            expl_path_collector,
            'train',
            decode_goal_image_key='image_decoded_goal',
            reconstruction_key='image_reconstruction',
            rows=2,
            columns=num_video_columns,
            imsize=imsize,
            image_format=renderer.output_image_format,
            unnormalize=True,
            dump_pickle=True,
            dump_only_init_and_goal=True,
            **expl_save_video_kwargs
        )
        algorithm.post_train_funcs.append(expl_video_func)

        if use_planning_eval:
            eval_save_video_kwargs['include_final_goal'] = True
        else:
            eval_save_video_kwargs['goal_image_key'] = image_goal_key

        if algo_kwargs['num_eval_steps_per_epoch'] > 0:
            eval_video_func = RIGVideoSaveFunction(
                model['vqvae'],
                eval_path_collector,
                'eval',
                decode_goal_image_key='image_decoded_goal',
                reconstruction_key='image_reconstruction',
                rows=2,
                columns=num_video_columns,
                imsize=imsize,
                image_format=renderer.output_image_format,
                unnormalize=True,
                dump_pickle=True,
                dump_only_init_and_goal=True,
                **eval_save_video_kwargs
            )
            algorithm.post_train_funcs.append(eval_video_func)

    # IQL CODE
    if save_paths:
        # TODO(kuanfang)
        algorithm.pre_train_funcs.append(save_paths_fn)
        # algorithm.post_train_funcs.append(save_paths_fn)

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
