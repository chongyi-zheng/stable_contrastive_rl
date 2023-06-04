import sys

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from rlkit.visualization import visualization_util as vu
from rlkit.torch.vae.skew.common import prob_to_weight


def visualize_vae_samples(
        epoch, training_data, vae,
        report, dynamics,
        n_vis=1000,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5)
):
    plt.figure()
    plt.suptitle("Epoch {}".format(epoch))
    n_samples = len(training_data)
    skip_factor = max(n_samples // n_vis, 1)
    training_data = training_data[::skip_factor]
    reconstructed_samples = vae.reconstruct(training_data)
    generated_samples = vae.sample(n_vis)
    projected_generated_samples = dynamics(generated_samples)
    plt.subplot(2, 2, 1)
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Generated Samples")
    plt.subplot(2, 2, 2)
    plt.plot(projected_generated_samples[:, 0],
             projected_generated_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Projected Generated Samples")
    plt.subplot(2, 2, 3)
    plt.plot(training_data[:, 0], training_data[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Training Data")
    plt.subplot(2, 2, 4)
    plt.plot(reconstructed_samples[:, 0], reconstructed_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Reconstruction")

    fig = plt.gcf()
    sample_img = vu.save_image(fig)
    report.add_image(sample_img, "Epoch {} Samples".format(epoch))

    return sample_img


def visualize_vae(vae, skew_config, report,
                  resolution=20,
                  title="VAE Heatmap"):
    xlim, ylim = vae.get_plot_ranges()
    show_prob_heatmap(vae, xlim=xlim, ylim=ylim, resolution=resolution)
    fig = plt.gcf()
    prob_heatmap_img = vu.save_image(fig)
    report.add_image(prob_heatmap_img, "Prob " + title)

    show_weight_heatmap(
        vae, skew_config, xlim=xlim, ylim=ylim, resolution=resolution,
    )
    fig = plt.gcf()
    heatmap_img = vu.save_image(fig)
    report.add_image(heatmap_img, "Weight " + title)
    return prob_heatmap_img


def show_weight_heatmap(
        vae, skew_config,
        xlim, ylim,
        resolution=20,
):

    def get_prob_batch(batch):
        prob = vae.compute_density(batch)
        return prob_to_weight(prob, skew_config)

    heat_map = vu.make_heat_map(get_prob_batch, xlim, ylim,
                                resolution=resolution, batch=True)
    vu.plot_heatmap(heat_map)


def show_prob_heatmap(
        vae,
        xlim, ylim,
        resolution=20,
):

    def get_prob_batch(batch):
        return vae.compute_density(batch)

    heat_map = vu.make_heat_map(get_prob_batch, xlim, ylim,
                                resolution=resolution, batch=True)
    vu.plot_heatmap(heat_map)


def visualize_histogram(histogram, skew_config, report, title=""):
    prob = histogram.pvals
    weights = prob_to_weight(prob, skew_config)
    xrange, yrange = histogram.xy_range
    extent = [xrange[0], xrange[1], yrange[0], yrange[1]]
    for name, values in [
        ('Weight Heatmap', weights),
        ('Prob Heatmap', prob),
    ]:
        plt.figure()
        fig = plt.gcf()
        ax = plt.gca()
        values = values.copy()
        values[values == 0] = np.nan
        heatmap_img = ax.imshow(
            np.swapaxes(values, 0, 1),  # imshow uses first axis as y-axis
            extent=extent,
            cmap=plt.get_cmap('plasma'),
            interpolation='nearest',
            aspect='auto',
            origin='bottom',  # <-- Important! By default top left is (0, 0)
            # norm=LogNorm(),
        )
        divider = make_axes_locatable(ax)
        legend_axis = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(heatmap_img, cax=legend_axis, orientation='vertical')
        heatmap_img = vu.save_image(fig)
        if histogram.num_bins < 5:
            pvals_str = np.array2string(histogram.pvals, precision=3)
            report.add_text(pvals_str)
        report.add_image(heatmap_img, "{} {}".format(title, name))
    return heatmap_img


def progressbar(it, prefix="", size=60):
    count = len(it)

    def _show(_i):
        x = int(size * _i / count)
        sys.stdout.write(
            "%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), _i, count))
        sys.stdout.flush()

    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i + 1)
    sys.stdout.write("\n")
    sys.stdout.flush()