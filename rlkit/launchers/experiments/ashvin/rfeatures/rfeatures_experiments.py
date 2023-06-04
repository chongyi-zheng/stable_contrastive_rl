import os.path as osp
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture

import torchvision
import pickle
from torch.utils import data

import rlkit.data_management.external.epic_kitchens_data as epic

from rlkit.data_management.dataset import InfiniteBatchLoader

def get_data(variant):
    import numpy as np
    from multiworld.core.image_env import ImageEnv, unormalize_image
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.external.epic_kitchens_data import EpicTimePredictionDataset
    
    dataset_name = variant.get("dataset_name")
    full_dataset_path = "/private/home/anair17/ashvindev/rlkit/notebooks/outputs/%s_train_trajectories.p" % dataset_name
    train_clips = pickle.load(open(full_dataset_path, "rb"))
    train_dataset = EpicTimePredictionDataset(train_clips, output_classes=variant["output_classes"])

    full_dataset_path = "/private/home/anair17/ashvindev/rlkit/notebooks/outputs/%s_validation_trajectories.p" % dataset_name
    test_clips = pickle.load(open(full_dataset_path, "rb"))
    test_dataset = EpicTimePredictionDataset(test_clips, output_classes=variant["output_classes"])
    
    return train_dataset, test_dataset, {}

def train_rfeatures_model(variant, return_data=False):
    from rlkit.util.ml_util import PiecewiseLinearSchedule
    # from rlkit.torch.vae.conv_vae import (
    #     ConvVAE, ConvResnetVAE
    # )
    import rlkit.torch.vae.conv_vae as conv_vae
    # from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.launchers.experiments.ashvin.rfeatures.rfeatures_model import TimestepPredictionModel
    from rlkit.launchers.experiments.ashvin.rfeatures.rfeatures_trainer import TimePredictionTrainer
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch
    output_classes = variant["output_classes"]
    representation_size = variant["representation_size"]
    batch_size = variant["batch_size"]
    variant['dataset_kwargs']["output_classes"] = output_classes
    train_dataset, test_dataset, info = get_data(
        variant['dataset_kwargs']
    )

    num_train_workers = variant.get("num_train_workers", 0) # 0 uses main process (good for pdb)
    train_dataset_loader = InfiniteBatchLoader(train_dataset, batch_size=batch_size, num_workers=num_train_workers, )
    test_dataset_loader = InfiniteBatchLoader(test_dataset, batch_size=batch_size, )

    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
    architecture = variant['model_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and variant.get('imsize') == 48:
        architecture = conv_vae.imsize48_default_architecture
    variant['model_kwargs']['architecture'] = architecture

    model_class = variant.get('model_class', TimestepPredictionModel)
    model = model_class(
        representation_size,
        decoder_output_activation=decoder_activation,
        output_classes=output_classes,
        **variant['model_kwargs'],
    )
    # model = torch.nn.DataParallel(model)
    model.to(ptu.device)

    variant['trainer_kwargs']['batch_size'] = batch_size
    trainer_class = variant.get('trainer_class', TimePredictionTrainer)
    trainer = trainer_class(
        model,
        **variant['trainer_kwargs'],
    )
    save_period = variant['save_period']

    trainer.dump_trajectory_rewards("initial", dict(train=train_dataset.dataset, test=test_dataset.dataset))

    dump_skew_debug_plots = variant.get('dump_skew_debug_plots', False)
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        trainer.train_epoch(epoch, train_dataset_loader, batches=10)
        trainer.test_epoch(epoch, test_dataset_loader, batches=1)

        if should_save_imgs:
            trainer.dump_reconstructions(epoch)
        
        trainer.dump_trajectory_rewards(epoch, dict(train=train_dataset.dataset, test=test_dataset.dataset), should_save_imgs)

        stats = trainer.get_diagnostics()
        for k, v in stats.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        trainer.end_epoch(epoch)

        if epoch % 50 == 0:
            logger.save_itr_params(epoch, model)
    logger.save_extra_data(model, 'vae.pkl', mode='pickle')
    if return_data:
        return model, train_dataset, test_dataset
    return model
