import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from skvideo.io import vwrite

import rlkit.pythonplusplus as ppp
from rlkit.core import logger
from rlkit.visualization import visualization_util as vu
from rlkit.util.html_report import HTMLReport
from rlkit.visualization.visualization_util import gif
from rlkit.torch.vae.skew.common import Dynamics, plot_curves
from rlkit.torch.vae.skew.datasets import project_samples_square_np


def visualize_histogram_samples(
        histogram,
        report,
        dynamics,
        n_vis=1000,
        title="Histogram Samples",
        samples=None,
):
    plt.figure()
    if samples is None:
        generated_samples = histogram.sample(n_vis)
        samples = dynamics(generated_samples)
    plt.plot(
        samples[:, 0],
        samples[:, 1],
        '.',
    )
    xlim, ylim = histogram.get_plot_ranges()
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title(title)

    fig = plt.gcf()
    sample_img = vu.save_image(fig)
    report.add_image(sample_img, title)
    return sample_img


def visualize_samples(epoch, vis_samples_np, histogram,
                      report, dynamics, n_vis=1000,
                      title=None,
                      xlim=(-1.5, 1.5),
                      ylim=(-1.5, 1.5)):
    if title is None:
        title = "Epoch {}".format(epoch)
    plt.figure()
    plt.suptitle(title)
    generated_samples = histogram.sample(n_vis)
    projected_generated_samples = dynamics(generated_samples)
    plt.subplot(3, 1, 1)
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Generated Samples")
    plt.subplot(3, 1, 2)
    plt.plot(projected_generated_samples[:, 0],
             projected_generated_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Projected Generated Samples")
    plt.subplot(3, 1, 3)
    plt.plot(vis_samples_np[:, 0], vis_samples_np[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Original Samples")

    fig = plt.gcf()
    sample_img = vu.save_image(fig)
    report.add_image(sample_img, "Epoch {} Samples".format(epoch))

    return sample_img


class Histogram(object):
    """
    A perfect histogram

    In this code, x = first index (not necessarily left-right for visualization)
    """

    def __init__(self, num_bins, xy_range=((-1, 1), (-1, 1)), init_idx=(0, 0)):
        self.xy_range = xy_range
        self.pvals = np.zeros((num_bins, num_bins))
        self.pvals[init_idx] = 1
        self.num_bins = num_bins
        self.num_bins_total = num_bins*num_bins
        self.uniform_distrib = (
                1. * np.ones(self.num_bins_total) / self.num_bins_total
        )
        bin_centers = np.zeros((self.num_bins, self.num_bins, 2))
        h, xedges, yedges = np.histogram2d(
            np.zeros(1), np.zeros(1),
            bins=self.num_bins,
            range=self.xy_range,
        )
        self.xedges = xedges
        self.yedges = yedges
        for xi in range(self.num_bins):
            x = 0.5 * (xedges[xi] + xedges[xi+1])
            for yi in range(self.num_bins):
                y = 0.5 * (yedges[yi] + yedges[yi+1])
                bin_centers[xi, yi, 0] = x
                bin_centers[xi, yi, 1] = y
        self.bin_centers_flat = bin_centers.reshape(
            self.num_bins_total, 2
        )

    def get_plot_ranges(self):
        xrange, yrange = self.xy_range
        xdelta = (xrange[1] - xrange[0]) / 10.
        ydelta = (yrange[1] - yrange[0]) / 10.
        return (
            (xrange[0]-xdelta, xrange[1]+xdelta),
            (yrange[0]-ydelta, yrange[1]+ydelta)
        )

    def sample(self, n_samples):
        idxs = np.random.choice(
            np.arange(self.num_bins_total),
            size=n_samples,
            # renormalizing to account for rounding errors
            p=self.pvals.flatten() / self.pvals.flatten().sum(),
        )
        samples = self.bin_centers_flat[idxs]
        return samples

    def fit(self, data, weights=None):
        H, *_ = np.histogram2d(
            data[:, 0],
            data[:, 1],
            self.num_bins,
            range=self.xy_range,
            weights=weights,
        )
        if weights is None:
            assert H.flatten().sum() == len(data), "Is the range wrong?"
            self.pvals = H.astype(np.float32) / len(data)
        else:
            # error = H.flatten().sum() == weights.flatten().sum()
            # assert np.abs(error) < 1e-5, "Is the range wrong?"
            self.pvals = H.astype(np.float32) / H.flatten().sum()

    def compute_density(self, data):
        indices = self._get_indices(data)
        return self.pvals.flatten()[indices]

    def _get_indices(self, data):
        x_indices = np.digitize(data[:, 0], self.xedges)
        # Because digitize will make index = len(self.xedges) if the value
        # equals self.xedges[-1], i.e. the value is on the right-most border.
        max_x = len(self.xedges)-1
        x_indices = np.minimum(x_indices, max_x)
        x_indices -= 1
        y_indices = np.digitize(data[:, 1], self.yedges)
        max_y = len(self.yedges)-1
        y_indices = np.minimum(y_indices, max_y)
        y_indices -= 1
        indices = x_indices * self.num_bins + y_indices
        return indices

    def entropy(self):
        return entropy(self.pvals.flatten())

    def max_entropy(self):
        return entropy(self.uniform_distrib)

    def kl_from_uniform(self):
        return entropy(self.uniform_distrib, self.pvals.flatten())

    def tv_to_uniform(self):
        return sum(np.abs(self.pvals - self.uniform_distrib[0]).flatten())


def train_from_variant(variant):
    train(full_variant=variant, **variant)


def train(
        dataset_generator,
        n_start_samples,
        projection=project_samples_square_np,
        n_samples_to_add_per_epoch=1000,
        n_epochs=100,
        save_period=10,
        append_all_data=True,
        full_variant=None,
        dynamics_noise=0,
        num_bins=5,
        weight_type='sqrt_inv_p',
        **kwargs
):
    report = HTMLReport(
        logger.get_snapshot_dir() + '/report.html',
        images_per_row=3,
        )
    dynamics = Dynamics(projection, dynamics_noise)
    if full_variant:
        report.add_header("Variant")
        report.add_text(
            json.dumps(
                ppp.dict_to_safe_json(
                    full_variant,
                    sort=True),
                indent=2,
            )
        )

    orig_train_data = dataset_generator(n_start_samples)
    train_data = orig_train_data

    heatmap_imgs = []
    sample_imgs = []
    entropies = []
    tvs_to_uniform = []
    """
    p_theta = previous iteration's model
    p_new = this iteration's distribution
    """
    p_theta = Histogram(num_bins, weight_type=weight_type)
    for epoch in range(n_epochs):
        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Entropy ', p_theta.entropy())
        logger.record_tabular('KL from uniform', p_theta.kl_from_uniform())
        logger.record_tabular('TV to uniform', p_theta.tv_to_uniform())
        entropies.append(p_theta.entropy())
        tvs_to_uniform.append(p_theta.tv_to_uniform())

        samples = p_theta.sample(n_samples_to_add_per_epoch)
        empirical_samples = dynamics(samples)

        if append_all_data:
            train_data = np.vstack((train_data, empirical_samples))
        else:
            train_data = np.vstack((orig_train_data, empirical_samples))

        if epoch == 0 or (epoch + 1) % save_period == 0:
            report.add_text("Epoch {}".format(epoch))
            heatmap_img = visualize_histogram(epoch, p_theta, report)
            sample_img = visualize_samples(
                epoch, train_data, p_theta, report, dynamics
            )
            heatmap_imgs.append(heatmap_img)
            sample_imgs.append(sample_img)
            report.save()

            from PIL import Image
            Image.fromarray(heatmap_img).save(
                logger.get_snapshot_dir() + '/heatmap{}.png'.format(epoch)
            )
            Image.fromarray(sample_img).save(
                logger.get_snapshot_dir() + '/samples{}.png'.format(epoch)
            )
        weights = p_theta.compute_per_elem_weights(train_data)
        p_new = Histogram(num_bins, weight_type=weight_type)
        p_new.fit(
            train_data,
            weights=weights,
        )
        p_theta = p_new
        logger.dump_tabular()
    plot_curves(
        [
            ("Entropy", entropies),
            ("TVs to Uniform", tvs_to_uniform),
        ],
        report
    )
    report.add_text("Max entropy: {}".format(p_theta.max_entropy()))
    report.save()

    heatmap_video = np.stack(heatmap_imgs)
    sample_video = np.stack(sample_imgs)

    vwrite(
        logger.get_snapshot_dir() + '/heatmaps.mp4',
        heatmap_video,
    )
    vwrite(
        logger.get_snapshot_dir() + '/samples.mp4',
        sample_video,
    )
    try:
        gif(
            logger.get_snapshot_dir() + '/samples.gif',
            sample_video,
        )
        gif(
            logger.get_snapshot_dir() + '/heatmaps.gif',
            heatmap_video,
        )
        report.add_image(
            logger.get_snapshot_dir() + '/samples.gif',
            "Samples GIF",
            is_url=True,
        )
        report.add_image(
            logger.get_snapshot_dir() + '/heatmaps.gif',
            "Heatmaps GIF",
            is_url=True,
        )
        report.save()
    except ImportError as e:
        print(e)
