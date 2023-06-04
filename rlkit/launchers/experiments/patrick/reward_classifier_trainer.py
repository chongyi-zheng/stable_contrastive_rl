import time
import collections
from collections import OrderedDict

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from rlkit.core.loss import LossFunction
from rlkit.core import logger
from rlkit.data_management.images import normalize_image
from rlkit.torch import pytorch_util as ptu

class RewardClassifierTrainer(LossFunction):

    def __init__(
        self,
        classifier,
        lr=3e-4,
        weight_decay=0,
        latent_size=720,
    ):
        self.lr = lr
        self.classifier = classifier
        cls_params = list(self.classifier.parameters())
        self.cls_optimizer = optim.Adam(cls_params,
                                        lr=self.lr,
                                        weight_decay=weight_decay)
        
        self.latent_size = latent_size

        self.num_batches = 0
        self.eval_statistics = collections.defaultdict(list)

    @property
    def log_dir(self):
        return logger.get_snapshot_dir()

    def end_epoch(self, epoch):
        self.eval_statistics = collections.defaultdict(list)
        self.test_last_batch = None

    def get_diagnostics(self):
        stats = OrderedDict()
        for k in sorted(self.eval_statistics.keys()):
            stats[k] = np.mean(self.eval_statistics[k])
        return stats

    def train_epoch(self, epoch, dataloader, batches=100):
        start_time = time.time()
        for b in range(batches):
            batch = next(iter(dataloader))
            self.train_batch(epoch, batch.to(ptu.device))
        self.eval_statistics['train/epoch_duration'].append(
            time.time() - start_time)

    def test_epoch(self, epoch, dataloader, batches=10):
        start_time = time.time()
        for b in range(batches):
            batch = next(iter(dataloader))
            self.test_batch(epoch, batch.to(ptu.device))
        self.eval_statistics['test/epoch_duration'].append(
            time.time() - start_time)

    def train_batch(self, epoch, batch):
        prefix = 'train'

        self.num_batches += 1
        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['num_batches'] = self.num_batches

        self.cls_optimizer.zero_grad()

        obs = batch[:, :self.latent_size]
        goal = batch[:, self.latent_size:2*self.latent_size]
        reward = batch[:, 2*self.latent_size:]
        cls_loss, _ = self._compute_classifier_loss(
            obs, goal, reward, epoch, '%s' % (prefix))

        cls_loss.backward()
        self.cls_optimizer.step()

    def test_batch(self, epoch, batch):
        prefix = 'test'

        obs = batch[:, :self.latent_size]
        goal = batch[:, self.latent_size:2*self.latent_size]
        reward = batch[:, 2*self.latent_size:]
        cls_loss, _ = self._compute_classifier_loss(
            obs, goal, reward, epoch, '%s' % (prefix))

    def compute_loss(self, batch, epoch, prefix):
        return 0.0

    def _compute_classifier_loss(self, obs, goal, reward, epoch, prefix):
        logits = self.classifier(
            torch.cat((obs, goal), dim=1)
        )
        targets = reward

        loss = F.binary_cross_entropy_with_logits(logits, targets)

        preds = (logits > 0).to(torch.float32)
        acc = torch.sum(
            (preds == targets).to(torch.float32)) / float(targets.shape[0])

        extra = {
            'acc': acc,
            'preds': preds,
            'targets': targets,
        }

        self.eval_statistics['%s/%s' % (prefix, 'loss')].append(loss.item())
        self.eval_statistics['%s/%s' % (prefix, 'accuracy')].append(acc.item())

        return loss, extra