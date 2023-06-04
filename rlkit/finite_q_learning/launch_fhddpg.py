"""
Try out this finite-horizon q learning.
"""
from gym.envs.mujoco import InvertedDoublePendulumEnv, InvertedPendulumEnv

import rlkit.util.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.mujoco.hopper_env import HopperEnv
from rlkit.finite_q_learning.finite_horizon_ddpg import FiniteHorizonDDPG
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import Mlp, TanhMlpPolicy, ConcatMlp


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])

    action_dim = env.action_space.low.size
    obs_dim = env.observation_space.low.size

    qf = ConcatMlp(
        input_size=action_dim + obs_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    algorithm = FiniteHorizonDDPG(
        env,
        qf,
        policy,
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
    exp_prefix = 'fh-ddpg-vs-ddpg-pendulums-h100'

    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=100,
            max_horizon=100,
            discount=1.,
            random_action_prob=0.05,
        ),
        env_kwargs=dict(
        ),
        qf_kwargs=dict(
            hidden_sizes=[32, 32],
        ),
        policy_kwargs=dict(
            hidden_sizes=[32, 32],
        ),
        algorithm="Finite-Horizon-DDPG",
        version="Finite-Horizon-DDPG",
    )

    search_space = {
        'qf_kwargs.hidden_sizes': [[32, 32]],
        'env_class': [
            InvertedPendulumEnv,
            InvertedDoublePendulumEnv,
        ],
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
