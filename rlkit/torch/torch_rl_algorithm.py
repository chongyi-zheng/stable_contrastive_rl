import abc
from collections import OrderedDict
from typing import Iterable, MutableMapping
from torch import nn as nn

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.offline_rl_algorithm import BatchOfflineRLAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch


OrderedDictType = MutableMapping


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchOfflineBatchRLAlgorithm(BatchOfflineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch, train=True)

    @abc.abstractmethod
    def train_from_torch(self, batch, train=True):
        pass

    def eval(self, np_batch):
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch, train=False)

    # def eval_from_torch(self, batch):
    #     pass

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])


class JointTrainer(Trainer):
    """
    Combine multiple trainers.

    Usage:
    ```
    trainer1 = ...
    trainer2 = ...

    trainers = OrderedDict([
        ('sac', sac_trainer),
        ('vae', vae_trainer),
    ])

    joint_trainer = JointTrainer

    algorithm = RLAlgorithm(trainer=joint_trainer, ...)
    algorithm.train()
    ```

    And then in the logs, the output will be of the fomr:

    ```
    trainer/sac/...
    trainer/vae/...
    ```
    """

    def __init__(self, trainers: OrderedDictType[str, TorchTrainer]):
        super().__init__()
        if len(trainers) == 0:
            raise ValueError("Need at least one trainer")
        self._trainers = trainers

    def train(self, np_batch):
        for trainer in self._trainers.values():
            trainer.train(np_batch)

    @property
    def networks(self):
        for trainer in self._trainers.values():
            for net in trainer.networks:
                yield net

    def end_epoch(self, epoch):
        for trainer in self._trainers.values():
            trainer.end_epoch(epoch)

    def get_snapshot(self):
        snapshot = {}
        for trainer_name, trainer in self._trainers.items():
            for k, v in trainer.get_snapshot().items():
                if trainer_name:
                    new_k = '{}/{}'.format(trainer_name, k)
                    snapshot[new_k] = v
                else:
                    snapshot[k] = v
        return snapshot

    def get_diagnostics(self):
        stats = {}
        for trainer_name, trainer in self._trainers.items():
            for k, v in trainer.get_diagnostics().items():
                if trainer_name:
                    new_k = '{}/{}'.format(trainer_name, k)
                    stats[new_k] = v
                else:
                    stats[k] = v
        return stats
