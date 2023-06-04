import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
import numpy as np
from rlkit.torch.networks import CNN, TwoHeadDCNN, DCNN
from rlkit.torch.vae.vae_base import compute_bernoulli_log_prob, compute_gaussian_log_prob, GaussianLatentVAE
from rlkit.util.ml_util import ConstantSchedule

from torch import optim
from rlkit.data_management.dataset import TripletReplayBufferWrapper

class HingeDistanceModelTrainer(object):
    def __init__(
            self,
            model,
            batch_size=128,
            log_interval=0,
            beta=0.5,
            beta_schedule=None,
            lr=None,
            weight_decay=0,
    ):
        self.model = model
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = beta
        if lr is None:
            if is_auto_encoder:
                lr = 1e-2
            else:
                lr = 1e-3
        self.beta_schedule = beta_schedule
        if self.beta_schedule is None or is_auto_encoder:
            self.beta_schedule = ConstantSchedule(self.beta)
        self.imsize = model.imsize
        model.to(ptu.device)

        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = self.imsize * self.imsize * self.input_channels

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params,
            lr=self.lr,
            weight_decay=weight_decay,
        )

        self.eval_statistics = {}

    def get_diagnostics(self):
        return self.eval_statistics

    def train_epoch(self, epoch, dataset, batches=100, from_rl=False):
        self.model.train()
        losses = []
        dataset = TripletReplayBufferWrapper(dataset, 100)
        for batch_idx in range(batches):
            self.optimizer.zero_grad()
            data = dataset.random_batch(self.batch_size)
            x0, x1, x2 = [data[key] for key in ["x0", "x1", "x2"]]

            z0 = self.model(x0)
            z1 = self.model(x1)
            z2 = self.model(x2)
            d01 = torch.norm(z1 - z0, dim=1)
            d02 = torch.norm(z2 - z0, dim=1)
            d12 = torch.norm(z2 - z1, dim=1)

            loss = torch.mean(F.relu(d01 - d02)) + torch.mean(F.relu(d02 - d12 - d01))

            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

        self.eval_statistics['train/loss'] = np.mean(losses)

    def test_epoch(
            self,
            epoch,
            dataset,
            save_reconstruction=True,
            save_scatterplot=True,
            save_vae=True,
            from_rl=False,
            batches=10,
    ):
        self.model.eval()
        losses = []
        dataset = TripletReplayBufferWrapper(dataset, 100)
        for batch_idx in range(batches):
            data = dataset.random_batch(self.batch_size)
            x0, x1, x2 = [data[key] for key in ["x0", "x1", "x2"]]

            z0 = self.model(x0)
            z1 = self.model(x1)
            z2 = self.model(x2)
            d01 = torch.norm(z1 - z0, dim=1)
            d02 = torch.norm(z2 - z0, dim=1)
            d12 = torch.norm(z2 - z1, dim=1)

            loss = torch.mean(F.relu(d01 - d02)) + torch.mean(F.relu(d02 - d12 - d01))
            losses.append(loss.item())

        self.eval_statistics['train/loss'] = np.mean(losses)
