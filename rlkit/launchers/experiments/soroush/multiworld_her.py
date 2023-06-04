import rlkit.torch.pytorch_util as ptu
from multiworld.core.flat_goal_env import FlatGoalEnv
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.envs.multitask.multitask_env import \
    MultitaskEnvToSilentMultitaskEnv, MultiTaskHistoryEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.torch.her.her_td3 import HerTd3
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.her.her_twin_sac import HerTwinSac
from rlkit.torch.her.her_sac import HerSac


def her_td3_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    observation_key = variant.get('observation_key', 'observation')
    desired_goal_key = variant.get('desired_goal_key', 'desired_goal')
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
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
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        qf1.to(ptu.device)
        qf2.to(ptu.device)
        policy.to(ptu.device)
        algorithm.to(ptu.device)
    algorithm.train()

def her_twin_sac_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    observation_key = variant.get('observation_key', 'observation')
    desired_goal_key = variant.get('desired_goal_key', 'desired_goal')
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
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
    vf = ConcatMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    algorithm = HerTwinSac(
        env,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        policy=policy,
        replay_buffer=replay_buffer,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        qf1.to(ptu.device)
        qf2.to(ptu.device)
        vf.to(ptu.device)
        policy.to(ptu.device)
        algorithm.to(ptu.device)
    algorithm.train()

def her_sac_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    observation_key = variant.get('observation_key', 'observation')
    desired_goal_key = variant.get('desired_goal_key', 'desired_goal')
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    qf = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    vf = ConcatMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    algorithm = HerSac(
        env,
        qf=qf,
        vf=vf,
        policy=policy,
        replay_buffer=replay_buffer,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        qf.to(ptu.device)
        vf.to(ptu.device)
        policy.to(ptu.device)
        algorithm.to(ptu.device)
    algorithm.train()