import torch
import torch.nn as nn
from torchvision import datasets, transforms
from rlkit.torch import pytorch_util as ptu
from os import path as osp
import numpy as np
from torchvision.utils import save_image
import time
from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize
from PIL import Image
import os
from tqdm import tqdm
import pickle
import sys

from torch.utils.data import DataLoader

from rlkit.util.io import load_local_or_remote_file
from rlkit.core import logger
from rlkit.launchers.experiments.patrick.reward_classifier_dataset import RewardClassifierConditionalLatentBlockDataset 
from rlkit.launchers.experiments.patrick.reward_classifier_trainer import RewardClassifierTrainer

"""
data loaders
"""

def train_classifier(
    classifier=None,
    vqvae_path='',
    num_epochs=100,
    dataset_path=None,
    batch_size=32,
    state_indices=range(8, 11),
    done_thres=0.065,
    cond_on_k_after_done=10, 
    positive_k_before_done=0,
    num_train_batches_per_epoch=10,
    num_test_batches_per_epoch=2,
    flip_network_inputs_randomly=False,
):
    assert classifier and vqvae_path != '' and dataset_path

    vqvae = load_local_or_remote_file(vqvae_path)
    vqvae.to(ptu.device)
    vqvae.eval()

    log_dir = logger.get_snapshot_dir()

    def encode_dataset(paths):
        all_data = []
        done_indices = []

        for path in paths:
            demo = load_local_or_remote_file(path)
            num_trajectories = len(demo)
            for data in demo:
                num_timesteps = len(data["observations"])
                im_length = data['observations'][0]['image_observation'].shape[0]
                obs = np.zeros((num_timesteps, im_length))

                done_idx = None
                for i in range(num_timesteps):
                    obs[i] = data['observations'][i]['image_observation']
                    if not done_idx:
                        done = np.linalg.norm(data['observations'][i]['state_observation'][state_indices] - data['observations'][i]['state_desired_goal'][state_indices]) < done_thres
                        if done:
                            done_idx = i
                if not done_idx:
                    done_idx = num_timesteps - 1

                latent = vqvae.encode_np(obs)
                all_data.append(latent)
                done_indices.append(done_idx)

        encodings = np.stack(all_data, axis=0)
        done_indices = np.array(done_indices).reshape(encodings.shape[0], 1)
        return encodings, done_indices

    train_data, train_done_indices = encode_dataset(dataset_path['train'])
    test_data, test_done_indices = encode_dataset(dataset_path['test'])
    dataset = {'train': train_data, 'train_done_indices': train_done_indices, 'test': test_data, 'test_done_indices': test_done_indices}

    def data_loaders(train_data, val_data, batch_size):
        train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
        val_loader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
        return train_loader, val_loader

    training_data = RewardClassifierConditionalLatentBlockDataset(dataset, train=True, cond_on_k_after_done=cond_on_k_after_done, positive_k_before_done=positive_k_before_done, flip_network_inputs_randomly=flip_network_inputs_randomly)
    validation_data = RewardClassifierConditionalLatentBlockDataset(dataset, train=False, cond_on_k_after_done=cond_on_k_after_done, positive_k_before_done=positive_k_before_done, flip_network_inputs_randomly=flip_network_inputs_randomly)
    train_loader, test_loader = data_loaders(
        training_data, validation_data, batch_size
    )
    
    trainer = RewardClassifierTrainer(
        classifier,
        latent_size=vqvae.representation_size,
    )

    print("Starting training")
    logger.remove_tabular_output(
        'progress.csv', relative_to_snapshot_dir=True
    )
    logger.add_tabular_output(
        'reward_classifier_progress.csv', relative_to_snapshot_dir=True
    )
    BEST_LOSS = 999
    for epoch in range(num_epochs):
        trainer.train_epoch(epoch, train_loader, batches=num_train_batches_per_epoch)
        trainer.test_epoch(epoch, test_loader, batches=num_test_batches_per_epoch)

        logger.save_itr_params(epoch, classifier)

        stats = trainer.get_diagnostics()

        cur_loss = stats["test/loss"]
        if cur_loss < BEST_LOSS:
            BEST_LOSS = cur_loss
            logger.save_extra_data(classifier, 'best_reward_classifier', mode='torch')

        for k, v in stats.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        trainer.end_epoch(epoch)
    logger.remove_tabular_output(
        'reward_classifier_progress.csv', relative_to_snapshot_dir=True
    )
    logger.add_tabular_output(
        'progress.csv', relative_to_snapshot_dir=True
    )

    return vqvae