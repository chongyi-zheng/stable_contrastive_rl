"""See https://github.com/aravindr93/hand_dapg for setup instructions"""

import os.path as osp
import pickle
from rlkit.core import logger
from rlkit.util.io import load_local_or_remote_file
from rlkit.core import logger

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse

ENV_ID_TO_ENV_CLASS = {
    'pen-v0': 'pen',
    'pen-notermination-v0': 'pen',
    'pen-sparse-v0': 'pen',
    'pen-binary-v0': 'pen',
    'door-v0': 'door',
    'door-sparse-v0': 'door',
    'door-binary-v0': 'door',
    'relocate-v0': 'relocate',
    'relocate-sparse-v0': 'relocate',
    'relocate-binary-v0': 'relocate',
    'hammer-v0': 'hammer',
    'hammer-sparse-v0': 'hammer',
    'hammer-binary-v0': 'hammer',
}

ENV_PARAMS = {
    'pen': {
        'max_path_length': 100,
        'demo_file': 'demos/icml2020/dapg/pen-v0_demos.pickle',
    },
    'door': {
        'max_path_length': 200,
        'demo_file': 'demos/icml2020/dapg/door-v0_demos.pickle',
    },
    'relocate': {
        'max_path_length': 200,
        'demo_file': 'demos/icml2020/dapg/relocate-v0_demos.pickle',
    },
    'hammer': {
        'max_path_length': 200,
        'demo_file': 'demos/icml2020/dapg/hammer-v0_demos.pickle',

    },
}

default_job_data = dict(
    algorithm = "DAPG",
    num_cpu = 1,
    save_freq = 25,
    eval_rollouts = 25,
    bc_batch_size = 32,
    bc_epochs = 5,
    bc_learn_rate = 1e-3,
    policy_size = (32, 32),
    vf_batch_size = 64,
    vf_epochs = 2,
    vf_learn_rate = 1e-3,
    rl_step_size = 0.05,
    rl_gamma = 0.995,
    rl_gae = 0.97,
    rl_num_traj = 200,
    rl_num_iter = 100,
    lam_0 = 1e-2,
    lam_1 = 0.95,
    sparse_reward=False,
)

def experiment(variant):
    """
    This is a job script for running NPG/DAPG on hand tasks and other gym envs.
    Note that DAPG generalizes PG and BC init + PG finetuning.
    With appropriate settings of parameters, we can recover the full family.
    """
    import mj_envs

    job_data = default_job_data.copy()
    job_data.update(variant)

    env_params = ENV_PARAMS[variant['env_class']]
    job_data.update(env_params)

    assert 'algorithm' in job_data.keys()
    assert any([job_data['algorithm'] == a for a in ['NPG', 'BCRL', 'DAPG']])

    JOB_DIR = logger.get_snapshot_dir()

    # ===============================================================================
    # Train Loop
    # ===============================================================================

    seed = int(job_data['seedid'])

    e = GymEnv(job_data['env_id'])
    policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=seed)
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                           epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = load_local_or_remote_file(job_data['demo_file'], 'rb')

        bc_agent = BC(demo_paths, policy=policy, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                      lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running BC with expert demonstrations")
        print("========================================")
        bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        if job_data['eval_rollouts'] >= 1:
            score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
            print("Score with behavior cloning = %f" % score[0][0])

    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    # ===============================================================================
    # RL Loop
    # ===============================================================================

    rl_agent = DAPG(e, policy, baseline, demo_paths,
                    normalized_step_size=job_data['rl_step_size'],
                    lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                    seed=seed, save_logs=True
                    )

    print("========================================")
    print("Starting reinforcement learning phase")
    print("========================================")

    ts = timer.time()
    train_agent(job_name=JOB_DIR,
                agent=rl_agent,
                seed=seed,
                niter=job_data['rl_num_iter'],
                gamma=job_data['rl_gamma'],
                gae_lambda=job_data['rl_gae'],
                num_cpu=job_data['num_cpu'],
                sample_mode='trajectories',
                num_traj=job_data['rl_num_traj'],
                save_freq=job_data['save_freq'],
                evaluation_rollouts=job_data['eval_rollouts'])
    print("time taken = %f" % (timer.time()-ts))

def process_args(variant):
    env_class = ENV_ID_TO_ENV_CLASS[variant['env_id']]
    variant["env_class"] = env_class















