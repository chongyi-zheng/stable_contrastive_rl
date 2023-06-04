import argparse

import joblib
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger


def get_best_action(qfs, observation, t):
    obs = ptu.np_to_var(observation[None], requires_grad=False).float()
    q_values = qfs[t](obs).squeeze(0)
    q_values_np = ptu.get_numpy(q_values)
    return q_values_np.argmax()


def finite_horizon_rollout(env, qfs, max_path_length, max_T):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    path_length = 0
    step = 0
    while path_length < max_path_length:
        a = get_best_action(qfs, o, step)
        next_o, r, d, env_info = env.step(a)
        env.render()
        if args.tick:
            import ipdb; ipdb.set_trace()
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append({})
        env_infos.append(env_info)
        path_length += 1
        step = (step + 1) % max_T
        if d:
            break
        o = next_o

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def simulate_policy(args):
    data = joblib.load(args.file)
    qfs = data['qfs']
    env = data['env']
    print("Data loaded")
    if args.pause:
        import ipdb; ipdb.set_trace()
    for qf in qfs:
        qf.train(False)
    paths = []
    while True:
        paths.append(finite_horizon_rollout(
            env,
            qfs,
            max_path_length=args.H,
            max_T=args.mt,
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--mt', type=int, default=20,
                        help='Max time for policy')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--tick', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
