"""
A special script made for running MPC controller.

The way that the MPC controller saves its cost function makes it so that when
the controller is de-serialized, it doesn't work.
"""
import argparse
import uuid

import joblib

from rlkit.core import logger
from rlkit.samplers.util import rollout

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = joblib.load(args.file)
    policy = data['mpc_controller']
    env = data['env']
    print("Policy loaded")
    if args.pause:
        import ipdb; ipdb.set_trace()
    policy.cost_fn = env.cost_fn
    policy.env = env
    if args.T:
        policy.mpc_horizon = args.T
    paths = []
    while True:
        paths.append(rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=True,
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        logger.dump_tabular()

if __name__ == "__main__":
    # For Point2d u-shaped wall
    # import matplotlib.pyplot as plt
    # plt.show()
    # plt.ion()

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--T', type=int,
                        help='Planning horizon')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
