import torch
from rlkit.core import logger
from rlkit.data_management.dataset import ImageObservationDataset
from rlkit.data_management.images import unnormalize_image
from rlkit.torch.vae.conv_vae import ConvVAE, SpatialAutoEncoder
from rlkit.torch.vae.vae_trainer import ConvVAETrainer


def get_n_train_vae_from_variant(variant):
    get_n_train_vae(**variant)


def get_n_train_vae(
    latent_dim,
    env,
    vae_train_epochs,
    num_image_examples,
    vae_kwargs,
    vae_trainer_kwargs,
    vae_architecture,
    vae_save_period=10,
    vae_test_p=.9,
    decoder_activation='sigmoid',
    vae_class='VAE',
    **kwargs
):
    env.goal_sampling_mode = 'test'
    image_examples = unnormalize_image(
        env.sample_goals(num_image_examples)['desired_goal'])
    n = int(num_image_examples * vae_test_p)
    train_dataset = ImageObservationDataset(image_examples[:n, :])
    test_dataset = ImageObservationDataset(image_examples[n:, :])

    if decoder_activation == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()

    vae_class = vae_class.lower()
    if vae_class == 'VAE'.lower():
        vae_class = ConvVAE
    elif vae_class == 'SpatialVAE'.lower():
        vae_class = SpatialAutoEncoder
    else:
        raise RuntimeError("Invalid VAE Class: {}".format(vae_class))

    vae = vae_class(
        latent_dim,
        architecture=vae_architecture,
        decoder_output_activation=decoder_activation,
        **vae_kwargs)

    trainer = ConvVAETrainer(vae, **vae_trainer_kwargs)

    logger.remove_tabular_output('progress.csv',
                                 relative_to_snapshot_dir=True)
    logger.add_tabular_output('vae_progress.csv', relative_to_snapshot_dir=True)
    for epoch in range(vae_train_epochs):
        should_save_imgs = (epoch % vae_save_period == 0)
        trainer.train_epoch(epoch, train_dataset)
        trainer.test_epoch(epoch, test_dataset)

        if should_save_imgs:
            trainer.dump_reconstructions(epoch)
            trainer.dump_samples(epoch)
        stats = trainer.get_diagnostics()
        for k, v in stats.items():
            logger.record_tabular(k, v)

        logger.dump_tabular()
        trainer.end_epoch(epoch)

        if epoch % 50 == 0:
            logger.save_itr_params(epoch, vae)
    logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
    logger.remove_tabular_output('vae_progress.csv',
                                 relative_to_snapshot_dir=True)
    logger.add_tabular_output('progress.csv',
                              relative_to_snapshot_dir=True)
    return vae