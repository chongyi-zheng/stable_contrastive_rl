"""
Try out this finite-horizon q learning.
"""

import gym
from gym.envs.mujoco import HopperEnv
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.mujoco.discrete_reacher import DiscreteReacherEnv
from rlkit.envs.mujoco.discrete_swimmer import DiscreteSwimmerEnv
from rlkit.envs.mujoco.reacher_env import ReacherEnv
from rlkit.envs.wrappers import DiscretizeEnv
from rlkit.finite_q_learning.finite_horizon_dqn import FiniteHorizonDQN
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import Mlp
import rlkit.util.hyperparameter as hyp
from rlkit.envs.mujoco.hopper_env import HopperEnv


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    env = DiscretizeEnv(env, variant['num_bins'])
    # env = DiscreteReacherEnv(**variant['env_kwargs'])

    qf = Mlp(
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
        **variant['qf_kwargs']
    )
    algorithm = FiniteHorizonDQN(
        env,
        qf,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'fhql-vs-ddqn-hooper-H1000-1000-networks-2'

    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=200,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=1000,
            max_horizon=1000,
            discount=1.,
            random_action_prob=0.05,
        ),
        env_kwargs=dict(
        ),
        algorithm="Finite-Horizon-DQN",
        num_bins=5,
    )

    search_space = {
        # 'algo_kwargs.discount': [0.99],
        # 'algo_kwargs.random_action_prob': [0.05, 0.2],
        'qf_kwargs.hidden_sizes': [[32, 32]],
        # 'env_kwargs.frame_skip': [2, 5],
        # 'env_kwargs.num_bins': [5],
        'env_class': [HopperEnv],
        'num_bins': [3],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )
