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

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.irl.irl_trainer import IRLTrainer

Observation = Dict
Goal = Any
GoalConditionedDiagnosticsFn = Callable[
    [List[Path], List[Goal]],
    Diagnostics,
]


class GAILTrainer(IRLTrainer):
    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"

        positives = self.positives.random_batch(self.batch_size)["observations"]
        P, feature_size = positives.shape
        positives = ptu.from_numpy(positives)
        negatives = batch['observations']
        N, feature_size = negatives.shape

        X = torch.cat((positives, negatives))
        Y = np.zeros((P + N, 1))
        Y[:P, 0] = 1
        # Y[P:, 0] = 0

        # X = ptu.from_numpy(X)
        Y = ptu.from_numpy(Y)
        y_pred = self.GAIL_discriminator_logits(X)

        loss = self.loss_fn(y_pred, Y)

        y_pred_class = (y_pred > 0).float()

        self.update_with_classification_stats(y_pred_class, Y, prefix)
        self.eval_statistics.update(create_stats_ordered_dict(
            "y_pred_positives",
            ptu.get_numpy(y_pred[:P]),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            "y_pred_negatives",
            ptu.get_numpy(y_pred[P:]),
        ))

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "losses"].append(loss.item())

        return loss

    def GAIL_discriminator_logits(self, observations):
        log_p = self.score_fn(observations)
        return log_p
