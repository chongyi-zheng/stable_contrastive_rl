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
from rlkit.torch.her.her_exploration_td3 import HerExplorationTd3
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy


def her_td3_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    her_kwargs = variant['algo_kwargs']['her_kwargs']
    observation_key = her_kwargs['observation_key']
    desired_goal_key = her_kwargs['desired_goal_key']
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
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
    num_ensemble_qs = variant.get("num_ensemble_qs", 0)
    ensemble_qs = [ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    ) for _ in range(num_ensemble_qs)]
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    render = variant.get("render", False)
    algorithm = HerExplorationTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        render=render,
        render_during_eval=render,
        ensemble_qs=ensemble_qs,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()
