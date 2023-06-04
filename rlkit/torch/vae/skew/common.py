import math
import numpy as np
from matplotlib import pyplot as plt

from rlkit.visualization import visualization_util as vu


class Dynamics(object):
    def __init__(self, projection, noise):
        self.projection = projection
        self.noise = noise

    def __call__(self, samples):
        new_samples = samples + self.noise * np.random.randn(
            *samples.shape
        )
        return self.projection(new_samples)


def plot_curves(names_and_data, report):
    n_curves = len(names_and_data)
    if n_curves < 4:
        n_cols = n_curves
        n_rows = 1
    else:
        n_cols = n_curves // 2
        n_rows = math.ceil(float(n_curves) / n_cols)

    plt.figure()
    for i, (name, data) in enumerate(names_and_data):
        j = i + 1
        plt.subplot(n_rows, n_cols, j)
        plt.plot(np.array(data))
        plt.title(name)
    fig = plt.gcf()
    img = vu.save_image(fig)
    report.add_image(img, "Final Distribution")


def visualize_samples(
        samples,
        report,
        title="Samples",
):
    plt.figure()
    plt.plot(samples[:, 0], samples[:, 1], '.')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title(title)

    fig = plt.gcf()
    sample_img = vu.save_image(fig)
    report.add_image(sample_img, title)
    return sample_img


def visualize_samples_and_projection(
        samples,
        report,
        post_dynamics_samples=None,
        dynamics=None,
        title="Samples",
):
    assert post_dynamics_samples is not None or dynamics is not None
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(samples[:, 0], samples[:, 1], '.')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title(title)

    if post_dynamics_samples is None:
        post_dynamics_samples = dynamics(samples)
    plt.subplot(1, 2, 2)
    plt.plot(post_dynamics_samples[:, 0], post_dynamics_samples[:, 1], '.')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title("Projected " + title)

    fig = plt.gcf()
    sample_img = vu.save_image(fig)
    report.add_image(sample_img, title)
    return sample_img


def prob_to_weight(prob, skew_config):
    weight_type = skew_config['weight_type']
    min_prob = skew_config['minimum_prob']
    if min_prob:
        prob = np.maximum(prob, min_prob)
    with np.errstate(divide='ignore', invalid='ignore'):
        if weight_type == 'inv_p':
            weights = 1. / prob
        elif weight_type == 'nll':
            weights = - np.log(prob)
        elif weight_type == 'sqrt_inv_p':
            weights = (1. / prob) ** 0.5
        elif weight_type == 'exp':
            exp = skew_config['alpha']
            weights = prob ** exp
        else:
            raise NotImplementedError()
    weights[weights == np.inf] = 0
    weights[weights == -np.inf] = 0
    weights[weights == -np.nan] = 0
    return weights / weights.flatten().sum()