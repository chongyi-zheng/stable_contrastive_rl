import pickle

import rlkit.torch.pytorch_util as ptu
from multiworld.core.flat_goal_env import FlatGoalEnv
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.data_management.her_replay_buffer import RelabelingReplayBuffer
from rlkit.envs.multitask.multitask_env import \
    MultitaskEnvToSilentMultitaskEnv, MultiTaskHistoryEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.state_distance.tdm_networks import TdmQf, TdmPolicy
# from rlkit.torch.her.her_twin_sac import HerTwinSac
# from rlkit.torch.her.her_sac import HerSac

from rlkit.state_distance.tdm_td3 import TdmTd3
from rlkit.util.ml_util import IntPiecewiseLinearSchedule


def tdm_td3_experiment(variant):
    variant['env_kwargs'].update(variant['reward_params'])
    env = variant['env_class'](**variant['env_kwargs'])

    multiworld_env = variant.get('multiworld_env', True)

    if multiworld_env is not True:
        env = MultitaskEnvToSilentMultitaskEnv(env)
        if variant["render"]:
            env.pause_on_goal = True

    if variant['normalize']:
        env = NormalizedBoxEnv(env)

    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            **variant['es_kwargs']
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
            **variant['es_kwargs'],
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=0.1,
            **variant['es_kwargs'],
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    if multiworld_env is True:
        obs_dim = env.observation_space.spaces['observation'].low.size
        action_dim = env.action_space.low.size
        goal_dim = env.observation_space.spaces['desired_goal'].low.size
    else:
        obs_dim = action_dim = goal_dim = None
    vectorized = 'vectorized' in env.reward_type
    variant['algo_kwargs']['tdm_kwargs']['vectorized'] = vectorized

    norm_order = env.norm_order
    variant['algo_kwargs']['tdm_kwargs']['norm_order'] = norm_order

    qf1 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        vectorized=vectorized,
        norm_order=norm_order,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        vectorized=vectorized,
        norm_order=norm_order,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    relabeling_env = pickle.loads(pickle.dumps(env))

    algo_kwargs = variant['algo_kwargs']

    if multiworld_env is True:
        observation_key = variant.get('observation_key', 'state_observation')
        desired_goal_key = variant.get('desired_goal_key', 'state_desired_goal')
        achieved_goal_key = variant.get('achieved_goal_key', 'state_achieved_goal')
        replay_buffer = ObsDictRelabelingBuffer(
            env=relabeling_env,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
            vectorized=vectorized,
            **variant['replay_buffer_kwargs']
        )
        algo_kwargs['tdm_kwargs']['observation_key'] = observation_key
        algo_kwargs['tdm_kwargs']['desired_goal_key'] = desired_goal_key
    else:
        replay_buffer = RelabelingReplayBuffer(
            env=relabeling_env,
            **variant['replay_buffer_kwargs']
        )

    # qf_criterion = variant['qf_criterion_class']()
    # algo_kwargs['td3_kwargs']['qf_criterion'] = qf_criterion
    algo_kwargs['td3_kwargs']['training_env'] = env
    if 'tau_schedule_kwargs' in variant:
        tau_schedule = IntPiecewiseLinearSchedule(**variant['tau_schedule_kwargs'])
    else:
        tau_schedule = None
    algo_kwargs['tdm_kwargs']['epoch_max_tau_schedule'] = tau_schedule

    algorithm = TdmTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        qf1.to(ptu.device)
        qf2.to(ptu.device)
        policy.to(ptu.device)
        algorithm.to(ptu.device)
    algorithm.train()