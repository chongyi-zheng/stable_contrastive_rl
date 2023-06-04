"""Script runner for AWR
Use this branch: git@github.com:anair13/awr.git

Setup instructions to run on AWS:
1. Add AWR code path to doodad code_dirs_to_mount
2. If using AWS S3 from AWR, make sure you update the following hardcoded paths
in util/io.py:

LOCAL_LOG_DIR = '/home/ashvin/data/s3doodad'
AWS_S3_PATH="s3://s3doodad/doodad/logs"

3. This code has been tested in docker image: "anair17/railrl-hand-tf-v1"

4. To view logs, use:
exps = plot.load_exps(dirs, suppress_output=True,
        progress_filename="log.txt", custom_log_reader=plot.AWRLogReader())
"""

import os.path as osp
import pickle
from rlkit.core import logger
from rlkit.util.io import load_local_or_remote_file
from rlkit.envs.wrappers import RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu

import gym
import numpy as np
import os
import sys
import tensorflow as tf

import learning.awr_agent as awr_agent

import mj_envs

AWR_CONFIGS = {
    "Ant-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.2,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
        "load_offpolicy_data": True,
        "offpolicy_data_sources": [
            dict(
                path="demos/icml2020/mujoco/ant_action_noise_15.npy",
                obs_dict=False,
                is_demo=True,
            ),
            dict(
                path="demos/icml2020/mujoco/ant_off_policy_15_demos_100.npy",
                obs_dict=False,
                is_demo=False,
                train_split=0.9,
            ),
        ],
    },

    "HalfCheetah-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
        "load_offpolicy_data": True,
        "offpolicy_data_sources": [
            dict(
                path="demos/icml2020/mujoco/hc_action_noise_15.npy",
                obs_dict=False,
                is_demo=True,
            ),
            dict(
                path="demos/icml2020/mujoco/hc_off_policy_15_demos_100.npy",
                obs_dict=False,
                is_demo=False,
                train_split=0.9,
            ),
        ],
    },

    "Hopper-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.0001,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "Humanoid-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00001,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "LunarLander-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.0005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_l2_weight": 0.001,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 100000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "Walker2d-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.000025,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
        "load_offpolicy_data": True,
        "offpolicy_data_sources": [
            dict(
                path="demos/icml2020/mujoco/walker_action_noise_15.npy",
                obs_dict=False,
                is_demo=True,
            ),
            dict(
                path="demos/icml2020/mujoco/walker_off_policy_15_demos_100.npy",
                obs_dict=False,
                is_demo=False,
                train_split=0.9,
            ),
        ],
    },

    "rlbench":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 10.0,
    },

    "pen-v0":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "max_path_length": 100,
        # "temp": 1.0,

        "load_offpolicy_data": True,
        "offpolicy_data_sources": [
            dict(
                path="demos/icml2020/hand/pen2_sparse.npy",
                obs_dict=True,
                is_demo=True,
            ),
            dict(
                path="demos/icml2020/hand/pen_bc_sparse4.npy",
                obs_dict=False,
                is_demo=False,
                train_split=0.9,
            ),
        ],
    },

    "door-v0":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "max_path_length": 200,
        # "temp": 100.0,

        "load_offpolicy_data": True,
        "offpolicy_data_sources": [
            dict(
                path="demos/icml2020/hand/door2_sparse.npy",
                obs_dict=True,
                is_demo=True,
            ),
            dict(
                path="demos/icml2020/hand/door_bc_sparse4.npy",
                obs_dict=False,
                is_demo=False,
                train_split=0.9,
            ),
        ],
    },

    "relocate-v0":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "max_path_length": 200,
        # "temp": 100.0,

        "load_offpolicy_data": True,
        "offpolicy_data_sources": [
            dict(
                path="demos/icml2020/hand/relocate2_sparse.npy",
                obs_dict=True,
                is_demo=True,
            ),
            dict(
                path="demos/icml2020/hand/relocate_bc_sparse4.npy",
                obs_dict=False,
                is_demo=False,
                train_split=0.9,
            ),
        ],
    },
}

def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

def build_env(env_id):
    assert(env_id is not ""), "Unspecified environment."
    env = gym.make(env_id)
    return env

def build_agent(env, env_id, agent_configs):
    agent_configs = {}
    if (env_id in AWR_CONFIGS):
        agent_configs.update(AWR_CONFIGS[env_id])

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    agent = awr_agent.AWRAgent(env=env, sess=sess, **agent_configs)

    return agent

default_variant = dict(
    train = True,
    test = True,
    max_iter = 100,
    test_episodes = 32,
    output_iters = 50,
    visualize = False,
    model_file = "",
    agent_configs = {},
)

def experiment(user_variant):
    variant = default_variant.copy()
    variant.update(user_variant)

    if ptu.gpu_enabled():
        enable_gpus("0")

    env_id = variant["env"]
    env = build_env(env_id)

    agent_configs = variant["agent_configs"]
    agent = build_agent(env, env_id, agent_configs)
    agent.visualize = variant["visualize"]
    model_file = variant.get("model_file")
    if (model_file is not ""):
        agent.load_model(model_file)

    log_dir = logger.get_snapshot_dir()
    if (variant["train"]):
        agent.train(max_iter=variant["max_iter"],
                    test_episodes=variant["test_episodes"],
                    output_dir=log_dir,
                    output_iters=variant["output_iters"])
    else:
        agent.eval(num_episodes=variant["test_episodes"])

    return
