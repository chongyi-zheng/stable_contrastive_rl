"""
Visualize how the errors in a learned dynamics model propagate over time.

Usage:
```
python ../visualize_model_error.py path/to/params.pkl
```
"""
import argparse
import math
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm


from rlkit.policies.simple import RandomPolicy
import rlkit.torch.pytorch_util as ptu


def get_np_prediction(model, state, action):
    state = ptu.np_to_var(np.expand_dims(state, 0))
    action = ptu.np_to_var(np.expand_dims(action, 0))
    delta = model(state, action)
    return ptu.get_numpy(delta.squeeze(0))


def visualize_policy_error(model, env, horizon):
    policy = RandomPolicy(env.action_space)
    actual_state = env.reset()

    predicted_states = []
    actual_states = []

    predicted_state = actual_state
    for _ in range(horizon):
        predicted_states.append(predicted_state.copy())
        actual_states.append(actual_state.copy())

        action, _ = policy.get_action(actual_state)
        delta = get_np_prediction(model, predicted_state, action)
        predicted_state += delta
        actual_state = env.step(action)[0]

    predicted_states = np.array(predicted_states)
    actual_states = np.array(actual_states)
    times = np.arange(horizon)

    num_state_dims = env.observation_space.low.size
    dims = list(range(num_state_dims))
    norm = colors.Normalize(vmin=0, vmax=num_state_dims)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)

    # Plot the predicted and actual values
    plt.subplot(2, 1, 1)
    for dim in dims:
        plt.plot(
            times,
            predicted_states[:, dim],
            '--',
            label='Predicted, Dim {}'.format(dim),
            color=mapper.to_rgba(dim),
        )
        plt.plot(
            times,
            actual_states[:, dim],
            '-',
            label='Actual, Dim {}'.format(dim),
            color=mapper.to_rgba(dim),
        )
    plt.xlabel("Time Steps")
    plt.ylabel("Observation Value")
    plt.legend(loc='best')

    # Plot the predicted and actual value errors
    plt.subplot(2, 1, 2)
    for dim in dims:
        plt.plot(
            times,
            np.abs(predicted_states[:, dim] - actual_states[:, dim]),
            '-',
            label='Dim {}'.format(dim),
            color=mapper.to_rgba(dim),
        )
    plt.xlabel("Time Steps")
    plt.ylabel("|Predicted - Actual| - Absolute Error")
    plt.legend(loc='best')
    plt.show()

    nrows = min(5, num_state_dims)
    ncols = math.ceil(num_state_dims / nrows)
    fig = plt.figure()
    for dim in dims:
        ax = fig.add_subplot(nrows, ncols, dim+1)
        ax.plot(
            times,
            predicted_states[:, dim],
            '--',
            label='Predicted, Dim {}'.format(dim),
        )
        ax.plot(
            times,
            actual_states[:, dim],
            '-',
            label='Actual, Dim {}'.format(dim),
        )
        ax.set_ylabel("Observation Value")
        ax.set_xlabel("Time Steps")
        ax.set_title("Dim {}".format(dim))
        ax_error = ax.twinx()
        ax_error.plot(
            times,
            np.abs(predicted_states[:, dim] - actual_states[:, dim]),
            '.',
            label='Error, Dim {}'.format(dim),
            color='r',
        )
        ax_error.set_ylabel("Error", color='r')
        ax_error.tick_params('y', colors='r')
        ax.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=30, help='Horizon for eval')
    args = parser.parse_args()

    data = joblib.load(args.file)
    model = data['model']
    env = data['env']
    visualize_policy_error(model, env, args.H)
