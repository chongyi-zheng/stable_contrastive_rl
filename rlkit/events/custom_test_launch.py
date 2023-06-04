import argparse
import joblib

import rlkit.torch.pytorch_util as ptu
import rlkit.util.hyperparameter as hyp
from rlkit.data_management.her_replay_buffer import HerReplayBuffer, \
    PrioritizedHerReplayBuffer, SimplePrioritizedHerReplayBuffer
from rlkit.envs.multitask.point2d import MultitaskPoint2DEnv, CustomBeta
from rlkit.envs.multitask.point2d_wall import MultitaskPoint2dWall
from rlkit.events.beta_learning import BetaLearning
from rlkit.events.controllers import BetaQLbfgsController, BetaQMultigoalLbfgs, \
    BetaVMultigoalLbfgs
from rlkit.events.networks import BetaQ, TanhFlattenMlpPolicy, BetaV
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon import \
    GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.policies.simple import RandomPolicy
from rlkit.torch.networks import TanhMlpPolicy


def experiment(variant):
    env = variant['env_class']()
    es = GaussianAndEpsilonStrategy(
    # es = GaussianStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    beta_q = BetaQ(
        env,
        False,
        hidden_sizes=[32, 32],
    )
    beta_q2 = BetaQ(
        env,
        False,
        hidden_sizes=[32, 32],
    )
    # custom_beta = CustomBeta(env)
    beta_v = BetaV(
        env,
        False,
        hidden_sizes=[32, 32],
    )
    policy = TanhFlattenMlpPolicy(
        env,
        hidden_sizes=[32, 32],
    )
    if variant['load_file'] is not None:
        data = joblib.load(variant['load_file'])
        beta_q = data['beta_q']
        beta_q2 = data['beta_q2']
        beta_v = data['beta_v']
        policy = data['policy']

    goal_slice = env.ob_to_goal_slice
    multitask_goal_slice = slice(None)
    controller = BetaVMultigoalLbfgs(
        beta_v,
        env,
        goal_slice=goal_slice,
        goal_reaching_policy=policy,
        multitask_goal_slice=multitask_goal_slice,
        **variant['controller_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        # policy=policy,
        # policy=RandomPolicy(env.action_space),
        policy=controller,
    )
    replay_buffer = SimplePrioritizedHerReplayBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = BetaLearning(
        env,
        exploration_policy=exploration_policy,
        beta_q=beta_q,
        beta_q2=beta_q2,
        # custom_beta=custom_beta,
        beta_v=beta_v,
        policy=policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--mt', type=int, help='max time to goal', default=0)
    args = parser.parse_args()

    n_seeds = 1
    mode = 'local'
    exp_prefix = "dev-beta-learning"

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = "beta-learning-wall-fixed-log-fd-sweep-goal-ep"

    variant = dict(
        load_file=args.file,
        # env_class=MultitaskPoint2DEnv,
        env_class=MultitaskPoint2dWall,
        algo_kwargs=dict(
            num_epochs=200,
            num_steps_per_epoch=100,
            num_steps_per_eval=200,
            num_updates_per_env_step=10,
            max_path_length=25,
            batch_size=128,
            discount=0.,
            prioritized_replay=False,
            # render=True,
            # render_during_eval=True,
            # save_replay_buffer=True,

            policy_and_target_update_period=1,

            finite_horizon=True,
            max_num_steps_left=0,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E5),
            num_goals_to_sample=1,
            max_time_to_next_goal=1,
            fraction_goals_are_rollout_goals=1,
            # resampling_strategy='truncated_geometric',
            # truncated_geom_factor=0.5,
        ),
        controller_kwargs=dict(
            max_cost=128,
            planning_horizon=5,
            use_max_cost=True,
            replan_every_time_step=False,
            only_use_terminal_env_loss=True,
            solver_kwargs={
                'factr': 1e6,
            },
        ),
        es_kwargs=dict(
            epsilon=0.2,
            max_sigma=0.,
        ),
        algorithm='Beta Learning',
        version='Dev',
    )
    search_space = {
        # 'es_kwargs.epsilon': [0.2, 0.5],
        'algo_kwargs.train_with': ['on_policy', 'off_policy', 'both'],
        'algo_kwargs.always_reset_env': [True, False],
        'algo_kwargs.goal_reached_epsilon': [1e-3, 0.1],
        # 'algo_kwargs.train_simultaneously': [False],
        'controller_kwargs.use_max_cost': [True, False],
        # 'controller_kwargs.replan_every_time_step': [True, False],
        # 'controller_kwargs.only_use_terminal_env_loss': [True, False],
        'controller_kwargs.solver_kwargs.factr': [1e6],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for i in range(n_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            if (variant['controller_kwargs']['replan_every_time_step'] and
                variant['controller_kwargs']['only_use_terminal_env_loss']
                ):
                continue
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
                exp_id=exp_id,
                # snapshot_gap=1,
                # snapshot_mode='gap',
            )
