import os

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.pearl_envs import ENVS, register_pearl_envs
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.pearl.agent import PEARLAgent
from rlkit.torch.pearl.encoder import MlpEncoder, RecurrentEncoder
from rlkit.torch.pearl.pearl import PEARLSoftActorCritic
from rlkit.torch.sac.policies import TanhGaussianPolicy


def pearl_experiment(variant):
    register_pearl_envs()
    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = ConcatMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = ConcatMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = ConcatMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        # latent_dim=latent_dim,
        action_dim=action_dim,
    )
    reward_predictor = ConcatMlp(
        hidden_sizes=[200, 200, 200],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        reward_predictor,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_task_indices=list(tasks[:variant['n_train_tasks']]),
        eval_task_indices=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))


    # create logging directory
    # TODO support Docker
    # if exp_name == '':
    #     exp_name = variant['env_name']
    # if DEBUG:
    #     exp_id = 'debug'
    #     exp_name = 'dev-' + exp_name
    # else:
    #     exp_id = None
    # experiment_log_dir = setup_logger(
    #     exp_name,
    #     variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])
    #
    # # optionally save eval trajectories as pkl files
    # if variant['algo_params']['dump_eval_paths']:
    #     pickle_dir = experiment_log_dir + '/eval_trajectories'
    #     pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.to(ptu.device)
    algorithm.train()
