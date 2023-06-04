from collections import OrderedDict
from functools import partial

import numpy as np
import torch
from gym.envs.robotics import FetchEnv
from matplotlib.ticker import ScalarFormatter
from torch import optim
from torch import nn

import rlkit.samplers.rollout_functions as rf
import rlkit.torch.pytorch_util as ptu
from multiworld.envs.mujoco.cameras import (
    sawyer_door_env_camera_v0,
    sawyer_init_camera_zoomed_in,
    sawyer_pick_and_place_camera,
)
# from multiworld.envs.mujoco.classic_mujoco.hopper import \
    # HopperFullPositionGoalEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    (
    SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ,
)
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_nips import SawyerPushAndReachXYEnv
from multiworld.envs.mujoco.classic_mujoco.ant import (
    AntXYGoalEnv,
    AntFullPositionGoalEnv,
)
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.core.logging import add_prefix
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from rlkit.envs.contextual import (
    ContextualEnv, ContextualRewardFn,
    delete_info,
)
from rlkit.envs.contextual.contextual_env import batchify
from rlkit.envs.contextual.goal_conditioned import (
    AddImageDistribution,
    GoalConditionedDiagnosticsToContextualDiagnostics,
    GoalDictDistributionFromMultitaskEnv,
    PresampledDistribution,
    ThresholdDistanceReward,
    L2Distance,
    IndexIntoAchievedGoal,
    NegativeL2Distance,
)
from rlkit.envs.contextual.gym_goal_envs import (
    GoalDictDistributionFromGymGoalEnv,
    GenericGoalConditionedContextualDiagnostics,
)
from rlkit.envs.images import EnvRenderer, GymEnvRenderer
from rlkit.launchers.contextual.util import (
    get_save_video_function,
    get_gym_env,
)
from rlkit.launchers.experiments.vitchyr.probabilistic_goal_reaching.diagnostics import (
    AntFullPositionGoalEnvDiagnostics,
    SawyerPickAndPlaceEnvAchievedFromObs,
    HopperFullPositionGoalEnvDiagnostics,
)
from rlkit.launchers.experiments.vitchyr.probabilistic_goal_reaching.env import \
    NormalizeAntFullPositionGoalEnv
from rlkit.launchers.experiments.vitchyr.probabilistic_goal_reaching.stochastic_env import \
    NoisyAction
from rlkit.launchers.experiments.vitchyr.probabilistic_goal_reaching.visualize import (
    DynamicsModelEnvRenderer, InsertDebugImagesEnv,
    DynamicNumberEnvRenderer,
    DiscountModelRenderer,
    ProductRenderer,
    ValueRenderer,
)
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.core import eval_util
from rlkit.util.ml_util import create_schedule
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from rlkit.samplers.data_collector.joint_path_collector import \
    JointPathCollector
from rlkit.torch.distributions import (
    Distribution, MultivariateDiagonalNormal,
    IndependentLaplace,
)
from rlkit.torch.networks.basic import MultiInputSequential, Concat
from rlkit.torch.networks.mlp import MultiHeadedMlp, ConcatMlp, ParallelMlp
from rlkit.torch.networks.stochastic.distribution_generator import (
    TanhGaussian, DistributionGenerator,
    Gaussian,
    IndependentLaplaceGen,
)
from rlkit.torch.pgr.dynamics_model import EnsembleToGaussian
from rlkit.torch.pgr.pgr import PGRTrainer
from rlkit.torch.sac.policies import (
    MakeDeterministic,
    PolicyFromDistributionGenerator,
)
from rlkit.torch.supervised_learning.discount_model_trainer import (
    DiscountModelTrainer
)
from rlkit.torch.supervised_learning.dynamics_model_trainer import (
    GenerativeGoalDynamicsModelTrainer
)
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, JointTrainer


class ProbabilisticGoalRewardFn(ContextualRewardFn):
    def __init__(
            self,
            dynamics_model: DistributionGenerator,
            state_key,
            context_key,
            reward_type='log_prob',
            discount_factor=0.99,
    ):
        if reward_type not in {'log_prob', 'prob'}:
            raise ValueError(reward_type)
        self._model = dynamics_model
        self._state_key = state_key
        self._context_key = context_key
        self._reward_type = reward_type
        self._discount_factor = discount_factor

    def __call__(self, states, actions, next_states, contexts):
        states_torch = ptu.from_numpy(states[self._state_key])
        actions_torch = ptu.from_numpy(actions)
        contexts_torch = ptu.from_numpy(contexts[self._context_key])
        dist = self._model(states_torch, actions_torch)
        log_prob_torch = dist.log_prob(contexts_torch)
        log_prob = ptu.get_numpy(log_prob_torch)

        if self._reward_type == 'log_prob':
            reward = log_prob
        elif self._reward_type == 'prob':
            reward = np.exp(log_prob)
        else:
            raise ValueError(self._reward_type)

        return reward


class IsotropicGaussianAroundState(DistributionGenerator):
    def __init__(self, state_to_goal_fn):
        super().__init__()
        self.state_to_goal_fn = state_to_goal_fn or (lambda x: x)

    def forward(self, states, actions) -> Distribution:
        goal = self.state_to_goal_fn(states)  # hack
        return MultivariateDiagonalNormal(goal, 1)


class StandardLaplaceAroundState(DistributionGenerator):
    def __init__(self, state_to_goal_fn):
        super().__init__()
        self.state_to_goal_fn = state_to_goal_fn or (lambda x: x)

    def forward(self, states, actions) -> Distribution:
        goal = self.state_to_goal_fn(states)  # hack
        return IndependentLaplace(goal, 1)


class DeltaModel(MultiInputSequential):
    def __init__(
            self,
            *args,
            state_to_goal_fn=None,
            log_std_max=2,
            log_std_min=-2,
            outputted_log_std_is_tanh=False,
            global_log_std_dim=None,
    ):
        super().__init__(*args)
        self.state_to_goal_fn = state_to_goal_fn or (lambda x: x)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.middle_log_std = (log_std_max + log_std_min) / 2
        self.half_range = (log_std_max - log_std_min) / 2
        self.outputted_log_std_is_tanh = outputted_log_std_is_tanh
        if global_log_std_dim is not None:
            global_log_std = torch.zeros(global_log_std_dim)
            self.global_log_std = nn.Parameter(global_log_std)
        else:
            self.global_log_std = None

    def forward(self, state, action):
        if self.global_log_std is None:
            mean, raw_log_std = super().forward(state, action)
            if self.outputted_log_std_is_tanh:
                log_std = raw_log_std * self.half_range + self.middle_log_std
            else:
                log_std = raw_log_std
        else:
            mean = super().forward(state, action)
            log_std = self.global_log_std
        return mean + self.state_to_goal_fn(state), log_std


class DeterministicEnsembleDeltaModel(MultiInputSequential):
    def __init__(
            self,
            *args,
            state_to_goal_fn=None,
    ):
        super().__init__(*args)
        self.state_to_goal_fn = state_to_goal_fn or (lambda x: x)

    def forward(self, state, action):
        delta = super().forward(state, action)
        current_subset = self.state_to_goal_fn(state).unsqueeze(-1)
        return delta + current_subset


def create_goal_dynamics_model(
        ob_space,
        action_space,
        goal_space,
        version,
        state_to_goal_fn,
        model_kwargs,
        delta_model_kwargs,
        ensemble_model_kwargs,
):
    if version == 'fixed_standard_gaussian':
        model = IsotropicGaussianAroundState(
            state_to_goal_fn=state_to_goal_fn,
        )
    elif version == 'fixed_standard_laplace':
        model = IsotropicGaussianAroundState(
            state_to_goal_fn=state_to_goal_fn,
        )
    elif version == 'learned_model':
        ob_dim = ob_space.high.size
        action_dim = action_space.high.size
        goal_dim = goal_space.high.size
        network = DeltaModel(
            Concat(),
            MultiHeadedMlp(
                input_size=ob_dim + action_dim,
                output_sizes=[goal_dim, goal_dim],
                **model_kwargs
            ),
            state_to_goal_fn=state_to_goal_fn,
            **delta_model_kwargs,
        )
        model = Gaussian(network)
    elif version == 'learned_model_laplace':
        ob_dim = ob_space.high.size
        action_dim = action_space.high.size
        goal_dim = goal_space.high.size
        network = DeltaModel(
            Concat(),
            MultiHeadedMlp(
                input_size=ob_dim + action_dim,
                output_sizes=[goal_dim, goal_dim],
                **model_kwargs
            ),
            state_to_goal_fn=state_to_goal_fn,
            **delta_model_kwargs,
        )
        model = IndependentLaplaceGen(network)
    elif version == 'learned_model_gaussian_global_variance':
        ob_dim = ob_space.high.size
        action_dim = action_space.high.size
        goal_dim = goal_space.high.size
        network = DeltaModel(
            Concat(),
            ConcatMlp(
                input_size=ob_dim + action_dim,
                output_size=goal_dim,
                **model_kwargs
            ),
            global_log_std_dim=goal_dim,
            state_to_goal_fn=state_to_goal_fn,
            **delta_model_kwargs,
        )
        model = Gaussian(network)
    elif version == 'learned_model_laplace_global_variance':
        ob_dim = ob_space.high.size
        action_dim = action_space.high.size
        goal_dim = goal_space.high.size
        network = DeltaModel(
            Concat(),
            ConcatMlp(
                input_size=ob_dim + action_dim,
                output_size=goal_dim,
                **model_kwargs
            ),
            global_log_std_dim=goal_dim,
            state_to_goal_fn=state_to_goal_fn,
            **delta_model_kwargs,
        )
        model = IndependentLaplaceGen(network)
    elif version == 'learned_model_ensemble':
        ob_dim = ob_space.high.size
        action_dim = action_space.high.size
        goal_dim = goal_space.high.size

        network = DeterministicEnsembleDeltaModel(
            Concat(),
            ParallelMlp(
                input_size=ob_dim + action_dim,
                output_size_per_mlp=goal_dim,
                **ensemble_model_kwargs
            ),
            state_to_goal_fn=state_to_goal_fn,
        )

        model = EnsembleToGaussian(network)
    else:
        raise ValueError(version)
    return model


class DiscountModel(ConcatMlp):
    def forward(self, *args, return_logits=False):
        logits = super().forward(*args)
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)


def create_discount_model(ob_space, goal_space, action_space, model_kwargs):
    ob_dim = ob_space.high.size
    action_dim = action_space.high.size
    goal_dim = goal_space.high.size
    return DiscountModel(
        input_size=ob_dim + action_dim + goal_dim,
        output_size=1,
        **model_kwargs
    )


class StateToGoalFn(object):
    def __init__(self, stub_env):
        self.env_type = type(stub_env)

    def __call__(self, state):
        if issubclass(self.env_type, FetchEnv):
            return state[..., 3:6]
        elif self.env_type == AntXYGoalEnv:
            return state[..., :2]
        elif self.env_type == AntFullPositionGoalEnv:
            return state[..., :15]
        # elif self.env_type == HopperFullPositionGoalEnv:
        #     return state[..., :6]
        elif self.env_type == SawyerPickAndPlaceEnvYZ:
            return state[..., 1:]
        else:
            return state


def probabilistic_goal_reaching_experiment(
        max_path_length,
        qf_kwargs,
        policy_kwargs,
        pgr_trainer_kwargs,
        replay_buffer_kwargs,
        algo_kwargs,
        env_id,
        discount_factor,
        reward_type,
        # Dynamics model
        dynamics_model_version,
        dynamics_model_config,
        dynamics_delta_model_config=None,
        dynamics_adam_config=None,
        dynamics_ensemble_kwargs=None,
        # Discount model
        learn_discount_model=False,
        discount_adam_config=None,
        discount_model_config=None,
        prior_discount_weight_schedule_kwargs=None,
        # Environment
        env_class=None,
        env_kwargs=None,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        exploration_policy_kwargs=None,
        action_noise_scale=0.,
        num_presampled_goals=4096,
        success_threshold=0.05,
        # Video / visualization parameters
        save_video=True,
        save_video_kwargs=None,
        video_renderer_kwargs=None,
        plot_renderer_kwargs=None,
        eval_env_ids=None,
        # Debugging params
        visualize_dynamics=False,
        visualize_discount_model=False,
        visualize_all_plots=False,
        plot_discount=False,
        plot_reward=False,
        plot_bootstrap_value=False,
        # env specific-params
        normalize_distances_for_full_state_ant=False,
):
    if dynamics_ensemble_kwargs is None:
        dynamics_ensemble_kwargs = {}
    if eval_env_ids is None:
        eval_env_ids = {'eval': env_id}
    if discount_model_config is None:
        discount_model_config = {}
    if dynamics_delta_model_config is None:
        dynamics_delta_model_config = {}
    if dynamics_adam_config is None:
        dynamics_adam_config = {}
    if discount_adam_config is None:
        discount_adam_config = {}
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not video_renderer_kwargs:
        video_renderer_kwargs = {}
    if not plot_renderer_kwargs:
        plot_renderer_kwargs = video_renderer_kwargs.copy()
        plot_renderer_kwargs['dpi'] = 48
    context_key = desired_goal_key

    stub_env = get_gym_env(
        env_id, env_class=env_class, env_kwargs=env_kwargs,
        unwrap_timed_envs=True,
    )
    is_gym_env = (
        isinstance(stub_env, FetchEnv)
        or isinstance(stub_env, AntXYGoalEnv)
        or isinstance(stub_env, AntFullPositionGoalEnv)
        # or isinstance(stub_env, HopperFullPositionGoalEnv)
    )
    is_ant_full_pos = isinstance(stub_env, AntFullPositionGoalEnv)

    if is_gym_env:
        achieved_goal_key = desired_goal_key.replace('desired', 'achieved')
        ob_keys_to_save_in_buffer = [observation_key, achieved_goal_key]
    elif isinstance(stub_env, SawyerPickAndPlaceEnvYZ):
        achieved_goal_key = desired_goal_key.replace('desired', 'achieved')
        ob_keys_to_save_in_buffer = [observation_key, achieved_goal_key]
    else:
        achieved_goal_key = observation_key
        ob_keys_to_save_in_buffer = [observation_key]
    # TODO move all env-specific code to other file
    if isinstance(stub_env, SawyerDoorHookEnv):
        init_camera = sawyer_door_env_camera_v0
    elif isinstance(stub_env, SawyerPushAndReachXYEnv):
        init_camera = sawyer_init_camera_zoomed_in
    elif isinstance(stub_env, SawyerPickAndPlaceEnvYZ):
        init_camera = sawyer_pick_and_place_camera
    else:
        init_camera = None

    full_ob_space = stub_env.observation_space
    action_space = stub_env.action_space
    state_to_goal = StateToGoalFn(stub_env)
    dynamics_model = create_goal_dynamics_model(
        full_ob_space[observation_key],
        action_space,
        full_ob_space[achieved_goal_key],
        dynamics_model_version,
        state_to_goal,
        dynamics_model_config,
        dynamics_delta_model_config,
        ensemble_model_kwargs=dynamics_ensemble_kwargs,
    )
    sample_context_from_obs_dict_fn = RemapKeyFn({context_key: achieved_goal_key})

    def contextual_env_distrib_reward(
            _env_id, _env_class=None, _env_kwargs=None
    ):
        base_env = get_gym_env(
            _env_id, env_class=env_class, env_kwargs=env_kwargs,
            unwrap_timed_envs=True,
        )
        if init_camera:
            base_env.initialize_camera(init_camera)
        if (isinstance(stub_env, AntFullPositionGoalEnv)
                and normalize_distances_for_full_state_ant):
            base_env = NormalizeAntFullPositionGoalEnv(base_env)
            normalize_env = base_env
        else:
            normalize_env = None
        env = NoisyAction(base_env, action_noise_scale)
        diag_fns = []
        if is_gym_env:
            goal_distribution = GoalDictDistributionFromGymGoalEnv(
                env,
                desired_goal_key=desired_goal_key,
            )
            diag_fns.append(GenericGoalConditionedContextualDiagnostics(
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                success_threshold=success_threshold,
            ))
        else:
            goal_distribution = GoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
            )
            diag_fns.append(
                GoalConditionedDiagnosticsToContextualDiagnostics(
                    env.goal_conditioned_diagnostics,
                    desired_goal_key=desired_goal_key,
                    observation_key=observation_key,
                )
            )
        if isinstance(stub_env, AntFullPositionGoalEnv):
            diag_fns.append(
                AntFullPositionGoalEnvDiagnostics(
                    desired_goal_key=desired_goal_key,
                    achieved_goal_key=achieved_goal_key,
                    success_threshold=success_threshold,
                    normalize_env=normalize_env,
                )
            )
        # if isinstance(stub_env, HopperFullPositionGoalEnv):
        #     diag_fns.append(
        #         HopperFullPositionGoalEnvDiagnostics(
        #             desired_goal_key=desired_goal_key,
        #             achieved_goal_key=achieved_goal_key,
        #             success_threshold=success_threshold,
        #         )
        #     )
        achieved_from_ob = IndexIntoAchievedGoal(
            achieved_goal_key,
        )
        if reward_type == 'sparse':
            distance_fn = L2Distance(
                achieved_goal_from_observation=achieved_from_ob,
                desired_goal_key=desired_goal_key,
            )
            reward_fn = ThresholdDistanceReward(distance_fn, success_threshold)
        elif reward_type == 'negative_distance':
            reward_fn = NegativeL2Distance(
                achieved_goal_from_observation=achieved_from_ob,
                desired_goal_key=desired_goal_key,
            )
        else:
            reward_fn = ProbabilisticGoalRewardFn(
                dynamics_model,
                state_key=observation_key,
                context_key=context_key,
                reward_type=reward_type,
                discount_factor=discount_factor,
            )
        goal_distribution = PresampledDistribution(
            goal_distribution, num_presampled_goals)
        final_env = ContextualEnv(
            env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=diag_fns,
            update_env_info_fn=delete_info,
        )
        return final_env, goal_distribution, reward_fn

    expl_env, expl_context_distrib, reward_fn = contextual_env_distrib_reward(
        env_id, env_class, env_kwargs,
    )
    obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[context_key].low.size
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

    def create_policy():
        obs_processor = MultiHeadedMlp(
            input_size=obs_dim,
            output_sizes=[action_dim, action_dim],
            **policy_kwargs
        )
        return PolicyFromDistributionGenerator(
            TanhGaussian(obs_processor)
        )
    policy = create_policy()

    def concat_context_to_obs(batch, replay_buffer, obs_dict, next_obs_dict, new_contexts):
        obs = batch['observations']
        next_obs = batch['next_observations']
        batch['original_observations'] = obs
        batch['original_next_observations'] = next_obs
        context = batch[context_key]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=expl_env,
        context_keys=[context_key],
        observation_keys_to_save=ob_keys_to_save_in_buffer,
        context_distribution=expl_context_distrib,
        sample_context_from_obs_dict_fn=sample_context_from_obs_dict_fn,
        reward_fn=reward_fn,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs
    )

    def create_trainer():
        trainers = OrderedDict()
        if learn_discount_model:
            discount_model = create_discount_model(
                ob_space=stub_env.observation_space[observation_key],
                goal_space=stub_env.observation_space[context_key],
                action_space=stub_env.action_space,
                model_kwargs=discount_model_config)
            optimizer = optim.Adam(
                discount_model.parameters(),
                **discount_adam_config
            )
            discount_trainer = DiscountModelTrainer(
                discount_model,
                optimizer,
                observation_key='observations',
                next_observation_key='original_next_observations',
                goal_key=context_key,
                state_to_goal_fn=state_to_goal,
            )
            trainers['discount_trainer'] = discount_trainer
        else:
            discount_model = None
        if prior_discount_weight_schedule_kwargs is not None:
            schedule = create_schedule(
                **prior_discount_weight_schedule_kwargs
            )
        else:
            schedule = None
        pgr_trainer = PGRTrainer(
            env=expl_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            discount=discount_factor,
            discount_model=discount_model,
            prior_discount_weight_schedule=schedule,
            **pgr_trainer_kwargs
        )
        trainers[''] = pgr_trainer
        optimizers = [
            pgr_trainer.qf1_optimizer,
            pgr_trainer.qf2_optimizer,
            pgr_trainer.alpha_optimizer,
            pgr_trainer.policy_optimizer,
        ]
        if dynamics_model_version in {
            'learned_model',
            'learned_model_ensemble',
            'learned_model_laplace',
            'learned_model_laplace_global_variance',
            'learned_model_gaussian_global_variance',
        }:
            model_opt = optim.Adam(
                dynamics_model.parameters(),
                **dynamics_adam_config
            )
        elif dynamics_model_version in {
            'fixed_standard_laplace',
            'fixed_standard_gaussian',
        }:
            model_opt = None
        else:
            raise NotImplementedError()
        model_trainer = GenerativeGoalDynamicsModelTrainer(
            dynamics_model,
            model_opt,
            state_to_goal=state_to_goal,
            observation_key='original_observations',
            next_observation_key='original_next_observations',
        )
        trainers['dynamics_trainer'] = model_trainer
        optimizers.append(model_opt)
        return JointTrainer(trainers), pgr_trainer

    trainer, pgr_trainer = create_trainer()

    eval_policy = MakeDeterministic(policy)

    def create_eval_path_collector(some_eval_env):
        return ContextualPathCollector(
            some_eval_env,
            eval_policy,
            observation_key=observation_key,
            context_keys_for_policy=[context_key],
        )
    path_collectors = dict()
    eval_env_name_to_env_and_context_distrib = dict()
    for name, extra_env_id in eval_env_ids.items():
        env, context_distrib, _ = contextual_env_distrib_reward(extra_env_id)
        path_collectors[name] = create_eval_path_collector(env)
        eval_env_name_to_env_and_context_distrib[name] = (
            env, context_distrib
        )
    eval_path_collector = JointPathCollector(path_collectors)
    exploration_policy = create_exploration_policy(
        expl_env, policy, **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key,
        context_keys_for_policy=[context_key],
    )

    def get_eval_diagnostics(key_to_paths):
        stats = OrderedDict()
        for eval_env_name, paths in key_to_paths.items():
            env, _ = eval_env_name_to_env_and_context_distrib[eval_env_name]
            stats.update(add_prefix(
                    env.get_diagnostics(paths),
                    eval_env_name,
                    divider='/',
            ))
            stats.update(add_prefix(
                eval_util.get_generic_path_information(paths),
                eval_env_name,
                divider='/',
            ))
        return stats

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=None,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        evaluation_get_diagnostic_functions=[get_eval_diagnostics],
        **algo_kwargs
    )
    algorithm.to(ptu.device)

    if normalize_distances_for_full_state_ant and is_ant_full_pos:
        qpos_weights = expl_env.unwrapped.presampled_qpos.std(axis=0)
    else:
        qpos_weights = None

    if save_video:
        if is_gym_env:
            video_renderer = GymEnvRenderer(**video_renderer_kwargs)

            def set_goal_for_visualization(env, policy, o):
                goal = o[desired_goal_key]
                if normalize_distances_for_full_state_ant and is_ant_full_pos:
                    unnormalized_goal = goal * qpos_weights
                    env.unwrapped.goal = unnormalized_goal
                else:
                    env.unwrapped.goal = goal

            rollout_function = partial(
                rf.contextual_rollout,
                max_path_length=max_path_length,
                observation_key=observation_key,
                context_keys_for_policy=[context_key],
                reset_callback=set_goal_for_visualization,
            )
        else:
            video_renderer = EnvRenderer(**video_renderer_kwargs)
            rollout_function = partial(
                rf.contextual_rollout,
                max_path_length=max_path_length,
                observation_key=observation_key,
                context_keys_for_policy=[context_key],
                reset_callback=None,
            )

        renderers = OrderedDict(
            image_observation=video_renderer,
        )
        state_env = expl_env.env
        state_space = state_env.observation_space[observation_key]
        low = state_space.low.min()
        high = state_space.high.max()
        y = np.linspace(low, high, num=video_renderer.image_chw[1])
        x = np.linspace(low, high, num=video_renderer.image_chw[2])
        all_xy_np = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        all_xy_torch = ptu.from_numpy(all_xy_np)
        num_states = all_xy_torch.shape[0]
        if visualize_dynamics:
            def create_dynamics_visualizer(show_prob, vary_state=False):
                def get_prob(obs_dict, action):
                    obs = obs_dict['state_observation']
                    obs_torch = ptu.from_numpy(obs)[None]
                    action_torch = ptu.from_numpy(action)[None]
                    if vary_state:
                        action_repeated = torch.zeros((num_states, 2))
                        dist = dynamics_model(all_xy_torch, action_repeated)
                        goal = ptu.from_numpy(
                            obs_dict['state_desired_goal'][None])
                        log_probs = dist.log_prob(goal)
                    else:
                        dist = dynamics_model(obs_torch, action_torch)
                        log_probs = dist.log_prob(all_xy_torch)
                    if show_prob:
                        return log_probs.exp()
                    else:
                        return log_probs
                return get_prob

            renderers['log_prob'] = ValueRenderer(
                create_dynamics_visualizer(False), **video_renderer_kwargs
            )
            # renderers['prob'] = ValueRenderer(
            #     create_dynamics_visualizer(True), **video_renderer_kwargs
            # )
            renderers['log_prob_vary_state'] = ValueRenderer(
                create_dynamics_visualizer(False, vary_state=True),
                only_get_image_once_per_episode=True,
                max_out_walls=isinstance(stub_env, PickAndPlaceEnv),
                **video_renderer_kwargs)
            # renderers['prob_vary_state'] = ValueRenderer(
            #     create_dynamics_visualizer(True, vary_state=True),
            #     **video_renderer_kwargs)
        if visualize_discount_model and pgr_trainer.discount_model:
            def get_discount_values(obs, action):
                obs = obs['state_observation']
                obs_torch = ptu.from_numpy(obs)[None]
                combined_obs = torch.cat([
                    obs_torch.repeat(num_states, 1),
                    all_xy_torch,
                ], dim=1)

                action_torch = ptu.from_numpy(action)[None]
                action_repeated = action_torch.repeat(num_states, 1)
                return pgr_trainer.discount_model(combined_obs, action_repeated)
            renderers['discount_model'] = ValueRenderer(
                get_discount_values,
                states_to_eval=all_xy_torch,
                **video_renderer_kwargs)
        if 'log_prob' in renderers and 'discount_model' in renderers:
            renderers['log_prob_time_discount'] = ProductRenderer(
                renderers['discount_model'],
                renderers['log_prob'],
                **video_renderer_kwargs)


        def get_reward(obs_dict, action, next_obs_dict):
            o = batchify(obs_dict)
            a = batchify(action)
            next_o = batchify(next_obs_dict)
            reward = reward_fn(o, a, next_o, next_o)
            return reward[0]

        def get_bootstrap(obs_dict, action, next_obs_dict, return_float=True):
            context_pt = ptu.from_numpy(obs_dict[context_key][None])
            o_pt = ptu.from_numpy(obs_dict[observation_key][None])
            next_o_pt = ptu.from_numpy(next_obs_dict[observation_key][None])
            action_torch = ptu.from_numpy(action[None])
            bootstrap, *_ = pgr_trainer.get_bootstrap_stats(
                torch.cat((o_pt, context_pt), dim=1),
                action_torch,
                torch.cat((next_o_pt, context_pt), dim=1),
            )
            if return_float:
                return ptu.get_numpy(bootstrap)[0, 0]
            else:
                return bootstrap

        def get_discount(obs_dict, action, next_obs_dict):
            bootstrap = get_bootstrap(obs_dict, action, next_obs_dict, return_float=False)
            reward_np = get_reward(obs_dict, action, next_obs_dict)
            reward = ptu.from_numpy(reward_np[None, None])
            context_pt = ptu.from_numpy(obs_dict[context_key][None])
            o_pt = ptu.from_numpy(obs_dict[observation_key][None])
            obs = torch.cat((o_pt, context_pt), dim=1)
            actions = ptu.from_numpy(action[None])
            discount = pgr_trainer.get_discount_factor(
                bootstrap,
                reward,
                obs,
                actions,
            )
            if isinstance(discount, torch.Tensor):
                discount = ptu.get_numpy(discount)[0, 0]
            return np.clip(discount, a_min=1e-3, a_max=1)

        def create_modify_fn(title, set_params=None, scientific=True,):
            def modify(ax):
                ax.set_title(title)
                if set_params:
                    ax.set(**set_params)
                if scientific:
                    scaler = ScalarFormatter(useOffset=True)
                    scaler.set_powerlimits((1, 1))
                    ax.yaxis.set_major_formatter(scaler)
                    ax.ticklabel_format(axis='y', style='sci')
            return modify

        def add_left_margin(fig):
            fig.subplots_adjust(left=0.2)
        if visualize_all_plots or plot_discount:
            renderers['discount'] = DynamicNumberEnvRenderer(
                dynamic_number_fn=get_discount,
                modify_ax_fn=create_modify_fn(
                    title='discount',
                    set_params=dict(
                        # yscale='log',
                        ylim=[-0.05, 1.1],
                    ),
                    # scientific=False,
                ),
                modify_fig_fn=add_left_margin,
                # autoscale_y=False,
                **plot_renderer_kwargs)

        if visualize_all_plots or plot_reward:
            renderers['reward'] = DynamicNumberEnvRenderer(
                dynamic_number_fn=get_reward,
                modify_ax_fn=create_modify_fn(
                    title='reward',
                    # scientific=False,
                ),
                modify_fig_fn=add_left_margin,
                **plot_renderer_kwargs)
        if visualize_all_plots or plot_bootstrap_value:
            renderers['bootstrap-value'] = DynamicNumberEnvRenderer(
                dynamic_number_fn=get_bootstrap,
                modify_ax_fn=create_modify_fn(
                    title='bootstrap value',
                    # scientific=False,
                ),
                modify_fig_fn=add_left_margin,
                **plot_renderer_kwargs)

        def add_images(env, state_distribution):
            state_env = env.env
            if is_gym_env:
                goal_distribution = state_distribution
            else:
                goal_distribution = AddImageDistribution(
                    env=state_env,
                    base_distribution=state_distribution,
                    image_goal_key='image_desired_goal',
                    renderer=video_renderer,
                )
            context_env = ContextualEnv(
                state_env,
                context_distribution=goal_distribution,
                reward_fn=reward_fn,
                observation_key=observation_key,
                update_env_info_fn=delete_info,
            )
            return InsertDebugImagesEnv(
                context_env,
                renderers=renderers,
            )
        img_expl_env = add_images(expl_env, expl_context_distrib)
        if is_gym_env:
            imgs_to_show = list(renderers.keys())
        else:
            imgs_to_show = ['image_desired_goal'] + list(renderers.keys())
        img_formats = [video_renderer.output_image_format]
        img_formats += [r.output_image_format for r in renderers.values()]
        expl_video_func = get_save_video_function(
            rollout_function,
            img_expl_env,
            exploration_policy,
            tag="xplor",
            imsize=video_renderer.image_chw[1],
            image_formats=img_formats,
            keys_to_show=imgs_to_show,
            **save_video_kwargs
        )
        algorithm.post_train_funcs.append(expl_video_func)
        for eval_env_name, (env, context_distrib) in (
                eval_env_name_to_env_and_context_distrib.items()
        ):
            img_eval_env = add_images(env, context_distrib)
            eval_video_func = get_save_video_function(
                rollout_function,
                img_eval_env,
                eval_policy,
                tag=eval_env_name,
                imsize=video_renderer.image_chw[1],
                image_formats=img_formats,
                keys_to_show=imgs_to_show,
                **save_video_kwargs
            )
            algorithm.post_train_funcs.append(eval_video_func)

    algorithm.train()
