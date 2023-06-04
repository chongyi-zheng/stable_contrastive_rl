"""
Skew the dataset so that it turns into generating a uniform distribution.
"""
import json
import time

import numpy as np
from PIL import Image
from skvideo.io import vwrite
from torch import nn as nn
from torch.optim import Adam

import rlkit.pythonplusplus as ppp
import rlkit.torch.vae.skew.skewed_vae as sv
from rlkit.core import logger
from rlkit.util.html_report import HTMLReport
from rlkit.visualization.visualization_util import gif
from rlkit.torch.vae.skew.common import (
    Dynamics, plot_curves,
    visualize_samples,
    prob_to_weight,
)
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.vae.skew.datasets import project_samples_square_np
from rlkit.torch.vae.skew.histogram import Histogram
from rlkit.torch.vae.skew.plotting import (
    visualize_vae_samples,
    visualize_vae,
    visualize_histogram,
    progressbar,
)

K = 6

"""
Plotting
"""


def train_from_variant(variant):
    train(full_variant=variant, **variant)


def train(
        dataset_generator,
        n_start_samples,
        projection=project_samples_square_np,
        n_samples_to_add_per_epoch=1000,
        n_epochs=100,
        z_dim=1,
        hidden_size=32,
        save_period=10,
        append_all_data=True,
        full_variant=None,
        dynamics_noise=0,
        decoder_output_var='learned',
        num_bins=5,
        skew_config=None,
        use_perfect_samples=False,
        use_perfect_density=False,
        vae_reset_period=0,
        vae_kwargs=None,
        use_dataset_generator_first_epoch=True,
        **kwargs
):

    """
    Sanitize Inputs
    """
    assert skew_config is not None
    if not (use_perfect_density and use_perfect_samples):
        assert vae_kwargs is not None
    if vae_kwargs is None:
        vae_kwargs = {}

    report = HTMLReport(
        logger.get_snapshot_dir() + '/report.html',
        images_per_row=10,
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

    vae, decoder, decoder_opt, encoder, encoder_opt = get_vae(
        decoder_output_var,
        hidden_size,
        z_dim,
        vae_kwargs,
    )
    vae.to(ptu.device)

    epochs = []
    losses = []
    kls = []
    log_probs = []
    hist_heatmap_imgs = []
    vae_heatmap_imgs = []
    sample_imgs = []
    entropies = []
    tvs_to_uniform = []
    entropy_gains_from_reweighting = []
    p_theta = Histogram(num_bins)
    p_new = Histogram(num_bins)

    orig_train_data = dataset_generator(n_start_samples)
    train_data = orig_train_data
    start = time.time()
    for epoch in progressbar(range(n_epochs)):
        p_theta = Histogram(num_bins)
        if epoch == 0 and use_dataset_generator_first_epoch:
            vae_samples = dataset_generator(n_samples_to_add_per_epoch)
        else:
            if use_perfect_samples and epoch != 0:
                # Ideally the VAE = p_new, but in practice, it won't be...
                vae_samples = p_new.sample(n_samples_to_add_per_epoch)
            else:
                vae_samples = vae.sample(n_samples_to_add_per_epoch)
        projected_samples = dynamics(vae_samples)
        if append_all_data:
            train_data = np.vstack((train_data, projected_samples))
        else:
            train_data = np.vstack((orig_train_data, projected_samples))

        p_theta.fit(train_data)
        if use_perfect_density:
            prob = p_theta.compute_density(train_data)
        else:
            prob = vae.compute_density(train_data)
        all_weights = prob_to_weight(prob, skew_config)
        p_new.fit(train_data, weights=all_weights)
        if epoch == 0 or (epoch + 1) % save_period == 0:
            epochs.append(epoch)
            report.add_text("Epoch {}".format(epoch))
            hist_heatmap_img = visualize_histogram(p_theta, skew_config, report)
            vae_heatmap_img = visualize_vae(
                vae, skew_config, report,
                resolution=num_bins,
            )
            sample_img = visualize_vae_samples(
                epoch, train_data, vae, report, dynamics,
            )

            visualize_samples(
                p_theta.sample(n_samples_to_add_per_epoch),
                report,
                title="P Theta/RB Samples",
            )
            visualize_samples(
                p_new.sample(n_samples_to_add_per_epoch),
                report,
                title="P Adjusted Samples",
            )
            hist_heatmap_imgs.append(hist_heatmap_img)
            vae_heatmap_imgs.append(vae_heatmap_img)
            sample_imgs.append(sample_img)
            report.save()

            Image.fromarray(hist_heatmap_img).save(
                logger.get_snapshot_dir() + '/hist_heatmap{}.png'.format(epoch)
            )
            Image.fromarray(vae_heatmap_img).save(
                logger.get_snapshot_dir() + '/hist_heatmap{}.png'.format(epoch)
            )
            Image.fromarray(sample_img).save(
                logger.get_snapshot_dir() + '/samples{}.png'.format(epoch)
            )

        """
        train VAE to look like p_new
        """
        if sum(all_weights) == 0:
            all_weights[:] = 1
        if vae_reset_period > 0 and epoch % vae_reset_period == 0:
            vae, decoder, decoder_opt, encoder, encoder_opt = get_vae(
                decoder_output_var,
                hidden_size,
                z_dim,
                vae_kwargs,
            )
            vae.to(ptu.device)
        vae.fit(train_data, weights=all_weights)
        epoch_stats = vae.get_epoch_stats()

        losses.append(np.mean(epoch_stats['losses']))
        kls.append(np.mean(epoch_stats['kls']))
        log_probs.append(np.mean(epoch_stats['log_probs']))
        entropies.append(p_theta.entropy())
        tvs_to_uniform.append(p_theta.tv_to_uniform())
        entropy_gain = p_new.entropy() - p_theta.entropy()
        entropy_gains_from_reweighting.append(entropy_gain)

        for k in sorted(epoch_stats.keys()):
            logger.record_tabular(k, epoch_stats[k])

        logger.record_tabular("Epoch", epoch)
        logger.record_tabular('Entropy ', p_theta.entropy())
        logger.record_tabular('KL from uniform', p_theta.kl_from_uniform())
        logger.record_tabular('TV to uniform', p_theta.tv_to_uniform())
        logger.record_tabular('Entropy gain from reweight', entropy_gain)
        logger.record_tabular('Total Time (s)', time.time() - start)
        logger.dump_tabular()
        logger.save_itr_params(epoch, {
            'vae': vae,
            'train_data': train_data,
            'vae_samples': vae_samples,
            'dynamics': dynamics,
        })

    report.add_header("Training Curves")
    plot_curves(
        [
            ("Training Loss", losses),
            ("KL", kls),
            ("Log Probs", log_probs),
            ("Entropy Gain from Reweighting", entropy_gains_from_reweighting),
        ],
        report,
    )
    plot_curves(
        [
            ("Entropy", entropies),
            ("TV to Uniform", tvs_to_uniform),
        ],
        report,
    )
    report.add_text("Max entropy: {}".format(p_theta.max_entropy()))
    report.save()

    for filename, imgs in [
        ("hist_heatmaps", hist_heatmap_imgs),
        ("vae_heatmaps", vae_heatmap_imgs),
        ("samples", sample_imgs),
    ]:
        video = np.stack(imgs)
        vwrite(
            logger.get_snapshot_dir() + '/{}.mp4'.format(filename),
            video,
        )
        local_gif_file_path = '{}.gif'.format(filename)
        gif_file_path = '{}/{}'.format(
            logger.get_snapshot_dir(),
            local_gif_file_path
        )
        gif(gif_file_path, video)
        report.add_image(local_gif_file_path, txt=filename, is_url=True)
    report.save()


def get_vae(decoder_output_var, hidden_size, z_dim, vae_kwargs):
    encoder = sv.Encoder(
        nn.Linear(2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, z_dim * 2),
    )
    if decoder_output_var == 'learned':
        last_layer = nn.Linear(hidden_size, 4)
    else:
        last_layer = nn.Linear(hidden_size, 2)
    decoder = sv.Decoder(
        nn.Linear(z_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        last_layer,
        output_var=decoder_output_var,
        output_offset=-1,
    )
    encoder_opt = Adam(encoder.parameters())
    decoder_opt = Adam(decoder.parameters())
    vae = sv.VAE(encoder=encoder, decoder=decoder, z_dim=z_dim, **vae_kwargs)
    return vae, decoder, decoder_opt, encoder, encoder_opt


