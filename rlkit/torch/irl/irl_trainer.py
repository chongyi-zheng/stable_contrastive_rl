import warnings
from typing import Any, Callable, Dict, List

import numpy as np
from gym.spaces import Box, Dict
from multiworld.core.multitask_env import MultitaskEnv

from rlkit import pythonplusplus as ppp
from rlkit.core.distribution import DictDistribution
from rlkit.envs.contextual import ContextualRewardFn
from rlkit.envs.contextual.contextual_env import (
    ContextualDiagnosticsFn,
    Path,
    Context,
    Diagnostics,
)
from rlkit.envs.images import Renderer
from rlkit.core.loss import LossFunction
from rlkit.torch import pytorch_util as ptu

import torch
from torch import optim, nn
import collections
from collections import OrderedDict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

Observation = Dict
Goal = Any
GoalConditionedDiagnosticsFn = Callable[
    [List[Path], List[Goal]],
    Diagnostics,
]


class IRLTrainer(TorchTrainer, LossFunction):
    def __init__(
        self,
        score_fn,
        positives,
        policy,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=128,
        data_split=1,
        train_split=0.9,
    ):
        """positives are a 2D numpy array"""

        self.score_fn = score_fn
        self.positives = positives
        self.policy = policy
        self.batch_size = batch_size
        self.epoch = 0

        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.softmax = torch.nn.Softmax()

        self.lr = lr
        params = list(self.score_fn.parameters())
        self.optimizer = optim.Adam(params,
            lr=self.lr,
            weight_decay=weight_decay,
        )

        # stateful tracking variables, reset every epoch
        self.eval_statistics = collections.defaultdict(list)
        self.eval_data = collections.defaultdict(list)

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"

        positives = self.positives.random_batch(self.batch_size)["observations"]
        P, feature_size = positives.shape
        positives = ptu.from_numpy(positives)
        negatives = batch['observations']
        N, feature_size = negatives.shape

        X = torch.cat((positives, negatives))
        Y = np.zeros((P + N, 2))
        Y[:P, 0] = 1
        Y[P:, 1] = 1

        # X = ptu.from_numpy(X)
        Y = ptu.from_numpy(Y)
        y_pred = self.softmax(self.airl_discriminator_logits(X)) # todo: logsumexp

        loss = self.loss_fn(y_pred, Y)

        y_pred_class = (y_pred > 0).float()
        self.update_with_classification_stats(y_pred_class, Y, prefix)

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "losses"].append(loss.item())

        return loss

    def update_with_classification_stats(self, y_pred, y_label, prefix):
        stats = self.eval_statistics
        stats[prefix + "tp"].append((y_pred * y_label).mean().item())
        stats[prefix + "tn"].append(((1 - y_pred) * (1 - y_label)).mean().item())
        stats[prefix + "fn"].append(((1 - y_pred) * y_label).mean().item())
        stats[prefix + "fp"].append((y_pred * (1 - y_label)).mean().item())

    def airl_discriminator_logits(self, observations):
        log_p = self.score_fn(observations)
        dist = self.policy(observations)
        new_obs_actions, log_pi = dist.sample_and_logprob()
        logits = torch.cat((log_p, log_pi[:, None]), dim=1)
        return logits

    def train_from_torch(self, batch):
        self.score_fn.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(batch, self.epoch, False)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.score_fn.eval()
        self.compute_loss(batch, self.epoch, True)

        self.epoch += 1

    def get_batch(self, test=False):
        if test:
            indices = np.random.randint(self.train_N, self.data_N, self.batch_size)
        else:
            indices = np.random.randint(0, self.train_N, self.batch_size)
        return self.positives[indices, :]

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.score_fn.eval()
        loss = self.compute_loss(batch, epoch, True)

    def end_epoch(self, epoch):
        self.eval_statistics = collections.defaultdict(list)
        self.test_last_batch = None

    def get_diagnostics(self):
        stats = OrderedDict()
        for k in sorted(self.eval_statistics.keys()):
            stats[k] = np.mean(self.eval_statistics[k])
        return stats

    @property
    def networks(self):
        return [
            self.score_fn,
        ]

    def get_snapshot(self):
        return dict(
            score_fn=self.score_fn,
        )


class IRLRewardFn(ContextualRewardFn):
    def __init__(
        self,
        score_fn,
        context_keys=None
    ):
        self.score_fn = score_fn
        self.context_keys = context_keys or []

    def __call__(self, states, actions, next_states, contexts):
        contexts = [contexts[k] for k in self.context_keys]
        full_obs = [next_states["observation"], ] + contexts
        np_obs = np.concatenate(full_obs, axis=1)
        obs = ptu.from_numpy(np_obs)
        r = self.score_fn(obs)
        return ptu.get_numpy(r)


### Structured reward function definitions below

class MahalanobisReward(nn.Module):
    def __init__(self,
        input_size,
        output_size=1,
    ):
        super().__init__()
        self.sigma = nn.Parameter(ptu.ones(input_size, requires_grad=True))
        self.mu = nn.Parameter(ptu.ones(input_size, requires_grad=True))

    def __call__(self, states):
        x = self.sigma * (states - self.mu)
        return -torch.norm(x, dim=1, keepdim=True)
