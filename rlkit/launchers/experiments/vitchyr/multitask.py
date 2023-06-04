from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv, \
    MultitaskEnvToSilentMultitaskEnv, MultiTaskHistoryEnv
from rlkit.envs.wrappers import NormalizedBoxEnv, HistoryEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.state_distance.tdm_networks import TdmPolicy, \
    TdmQf
from rlkit.state_distance.tdm_td3 import TdmTd3
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.torch.her.her_td3 import HerTd3
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3
import rlkit.torch.pytorch_util as ptu


def td3_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    env = MultitaskToFlatEnv(env)
    if variant.get('make_silent_env', True):
        env = MultitaskEnvToSilentMultitaskEnv(env)
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=0.1,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
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
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TD3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def her_td3_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    if 'history_len' in variant:
        history_len = variant['history_len']
        env = MultiTaskHistoryEnv(env, history_len=history_len)
    if variant.get('make_silent_env', True):
        env = MultitaskEnvToSilentMultitaskEnv(env)
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            **variant['es_kwargs']
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    goal_dim = env.goal_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = variant['replay_buffer_class'](
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def tdm_td3_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    tdm_normalizer = None
    qf1 = TdmQf(
        env=env,
        vectorized=True,
        tdm_normalizer=tdm_normalizer,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        vectorized=True,
        tdm_normalizer=tdm_normalizer,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        tdm_normalizer=tdm_normalizer,
        **variant['policy_kwargs']
    )
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=0.1,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = variant['replay_buffer_class'](
        env=env,
        **variant['replay_buffer_kwargs']
    )
    qf_criterion = variant['qf_criterion_class']()
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['td3_kwargs']['qf_criterion'] = qf_criterion
    algo_kwargs['tdm_kwargs']['tdm_normalizer'] = tdm_normalizer
    algorithm = TdmTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **algo_kwargs
    )
    algorithm.to(ptu.device)
    algorithm.train()


def ddpg_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    env = MultitaskToFlatEnv(env)
    if variant.get('make_silent_env', True):
        env = MultitaskEnvToSilentMultitaskEnv(env)
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=0.1,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()
