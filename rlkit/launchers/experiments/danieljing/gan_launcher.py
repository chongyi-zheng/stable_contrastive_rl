import numpy as np
from torch.utils import data
from rlkit.torch.grill.common import *

def train_gan(variant, return_data = False):
    from rlkit.torch.gan.dcgan import Generator, Discriminator
    from rlkit.torch.gan.dcgan_trainer import DCGANTrainer

    from rlkit.torch.gan.bigan import Generator, Encoder, Discriminator
    from rlkit.torch.gan.bigan_trainer import BiGANTrainer

    from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch
    import torch.utils.data
    import torchvision.datasets as dset
    from rlkit.data_management.external.bair_dataset import bair_dataset
    import torchvision.transforms as transforms
    from rlkit.data_management.external.bair_dataset.config import BAIR_DATASET_LOCATION

    from rlkit.util.io import sync_down_folder

    if not variant.get('simpusher', False):
        if variant["dataset"] == "bair":
            #train_dataset, test_dataset, info
            dataloader = bair_dataset.generate_dataset(variant['generate_dataset_kwargs'])[0].dataset_loader
            get_data = lambda d: d['x_t']

        if variant["dataset"] == "cifar10":
            local_path = sync_down_folder(variant["dataroot"])
            #local_path = variant["dataroot"]
            dataset = dset.CIFAR10(
                root=local_path, train=True, download=False, transform=transforms.Compose([
                              transforms.ToTensor()
                          ]))
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=variant["batch_size"], shuffle=True, num_workers=variant["num_workers"])
            get_data = lambda d: d[0]

        if variant["dataset"] == "celebfaces":
            local_path = sync_down_folder(variant["dataroot"])
            #local_path = variant["dataroot"]
            dataset = dset.ImageFolder(root=local_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(variant["image_size"]),
                                       transforms.CenterCrop(variant["image_size"]),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=variant["batch_size"],
               shuffle=True, num_workers=variant["num_workers"])
            get_data = lambda d: d[0]

        model_class = variant["gan_class"]
        trainer_class = variant["gan_trainer_class"]
        if trainer_class is DCGANTrainer:
            model = model_class(variant["ngpu"], variant["nc"], variant["latent_size"], variant["ngf"], variant["ndf"])
            trainer = trainer_class(model, variant["lr"], variant["beta"], variant["latent_size"])
        if trainer_class is BiGANTrainer:
            model = model_class(variant["ngpu"], variant["latent_size"], variant["dropout"], variant["output_size"])
            trainer = trainer_class(model, variant["ngpu"], variant["lr"], variant["beta"], variant["latent_size"], variant["generator_threshold"])

    if variant.get('simpusher', False):
        imsize = variant.get('imsize')
        beta = variant["beta"]
        representation_size = variant.get("representation_size")
        use_linear_dynamics = variant.get('use_linear_dynamics', False)
        generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn',
                                                generate_vae_dataset)
        variant['generate_vae_dataset_kwargs']['use_linear_dynamics'] = use_linear_dynamics
        batch_size = variant['algo_kwargs']['batch_size']
        variant['generate_vae_dataset_kwargs']['batch_size'] = batch_size
        train_dataset, test_dataset, info = generate_vae_dataset_fctn(
            variant['generate_vae_dataset_kwargs'])

        trainloader = train_dataset.dataset_loader
        testloader = test_dataset.dataset_loader
        get_data = lambda d: d['x_t'].reshape(128, 3, imsize, imsize)

        if use_linear_dynamics:
            action_dim = train_dataset.data['actions'].shape[2]

        logger.save_extra_data(info)
        logger.get_snapshot_dir()

        if 'context_schedule' in variant:
            schedule = variant['context_schedule']
            if type(schedule) is dict:
                context_schedule = PiecewiseLinearSchedule(**schedule)
            else:
                context_schedule = ConstantSchedule(schedule)
            variant['algo_kwargs']['context_schedule'] = context_schedule

        if variant['algo_kwargs'].get('is_auto_encoder', False):
            model = AutoEncoder(representation_size, **variant['gan_kwargs'])
        elif variant.get('use_spatial_auto_encoder', False):
            model = SpatialAutoEncoder(representation_size, **variant['gan_kwargs'])
        else:
            gan_class = variant['vae_class']
            if use_linear_dynamics:
                model = gan_class(latent_size = representation_size, **variant['gan_kwargs'])
            else:
                model = gan_class(latent_size = representation_size, **variant['gan_kwargs'])
        model.to(ptu.device)

        gan_trainer_class = variant['vae_trainer_class']
        trainer = gan_trainer_class(model, latent_size = representation_size, beta=beta,
                           **variant['algo_kwargs'])
        save_period = variant['save_period']

        dump_skew_debug_plots = variant.get('dump_skew_debug_plots', False)


    for epoch in range(variant['num_epochs']):
        trainer.train_epoch(trainloader, epoch, variant['num_epochs'], get_data)
        #trainer.test_epoch(epoch, test_dataset)
        #dump samples is called in trainer

        stats = trainer.get_stats(epoch)
        for k, v in stats.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        #trainer.end_epoch(epoch)

        if epoch % 50 == 0:
                logger.save_itr_params(epoch, trainer.get_model())

    logger.save_extra_data(trainer.get_model(), 'gan.pkl', mode='pickle')

    if return_data:
        return model, train_dataset, test_dataset

    return model
