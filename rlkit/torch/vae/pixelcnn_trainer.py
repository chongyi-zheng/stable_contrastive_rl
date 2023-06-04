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
from rlkit.util.io import load_local_or_remote_file
import os
import pickle
import sys

from rlkit.core import logger
from rlkit.core.loss import LossFunction
import collections
from collections import OrderedDict

class PixelCNNTrainer(LossFunction):
    def __init__(
            self,
            model,
            vqvae, # TODO: make more general
            batch_size=32,
            lr=3e-4,
    ):
        self.model = model
        self.vqvae = vqvae
        self.batch_size = batch_size
        self.lr = lr

        self.criterion = nn.CrossEntropyLoss().to(ptu.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # stateful tracking variables, reset every epoch
        self.eval_statistics = collections.defaultdict(list)
        self.eval_data = collections.defaultdict(list)
        self.num_batches = 0

    @property
    def log_dir(self):
        return logger.get_snapshot_dir()

    def train_epoch(self, epoch, data_loader, batches=100):
        start_time = time.time()
        for i, batch in enumerate(data_loader):
            if batches is not None and i >= batches:
                break
            self.train_batch(epoch, batch.to(ptu.device))
        self.eval_statistics["train/epoch_duration"].append(time.time() - start_time)

    def test_epoch(self, epoch, data_loader, batches=10):
        start_time = time.time()
        for i, batch in enumerate(data_loader):
            if batches is not None and i >= batches:
                break
            self.test_batch(epoch, batch.to(ptu.device))
        self.eval_statistics["test/epoch_duration"].append(time.time() - start_time)

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"

        x = batch
        root_len = self.vqvae.root_len
        num_embeddings = self.vqvae.num_embeddings

        cond = self.vqvae.discrete_to_cont(x[:, self.vqvae.discrete_size:]).reshape(x.shape[0], -1)
        x = x[:, :self.vqvae.discrete_size].reshape(-1, root_len, root_len)

        # Train PixelCNN with images
        logits = self.model(x, cond)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = self.criterion(
            logits.view(-1, num_embeddings),
            x.contiguous().view(-1)
        )

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "loss"].append(loss.item())
        self.eval_statistics["num_train_batches"].append(self.num_batches)

        return loss

    def train_batch(self, epoch, batch):
        self.num_batches += 1
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch, epoch, False)
        loss.backward()

        self.optimizer.step()

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.model.eval()
        loss = self.compute_loss(batch, epoch, True)

    def end_epoch(self, epoch):
        self.eval_statistics = collections.defaultdict(list)
        self.test_last_batch = None

    def get_diagnostics(self):
        stats = OrderedDict()
        for k in sorted(self.eval_statistics.keys()):
            stats[k] = np.mean(self.eval_statistics[k])
        return stats

    def dump_samples(self, epoch, data, test=True):
        start_time = time.time()
        suffix = 'test' if test else 'train'

        rand_indices = np.random.choice(data.shape[0], size=(8,))
        data_points = ptu.from_numpy(data[rand_indices, 0]).long().to(ptu.device)

        root_len = self.vqvae.root_len
        input_channels = self.vqvae.input_channels
        imsize = self.vqvae.imsize

        samples = []

        for i in range(8): # TODO: don't hardcode batch size
            env_latent = data_points[i].reshape(1, -1)
            cond = self.vqvae.discrete_to_cont(env_latent).reshape(1, -1)

            samples.append(self.vqvae.decode(cond))

            e_indices = self.model.generate(shape=(root_len, root_len),
                    batch_size=7, cond=cond.repeat(7, 1)).reshape(-1, root_len**2)
            samples.append(self.vqvae.decode(e_indices, cont=False))

        samples = torch.cat(samples, dim=0)
        filename = osp.join(self.log_dir, "cond_sample_{0}_{1}.png".format(suffix, epoch))
        save_image(
            samples.data.view(-1, input_channels, imsize, imsize).transpose(2, 3),
            filename
        )
        self.eval_statistics[suffix + "/sample_duration"].append(time.time() - start_time)
