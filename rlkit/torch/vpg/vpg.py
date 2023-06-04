from collections import OrderedDict

import numpy as np
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.data_management.env_replay_buffer import VPGEnvReplayBuffer
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class VPG(TorchRLAlgorithm):
    """
    Vanilla Policy Gradient
    """

    def __init__(
            self,
            env,
            policy,
            policy_learning_rate=1e-4,
            replay_buffer_class=VPGEnvReplayBuffer,
            **kwargs
    ):
        eval_policy = policy
        super().__init__(
            env,
            policy,
            eval_policy=eval_policy,
            collection_mode='batch',
            **kwargs
        )
        self.replay_buffer = replay_buffer_class(
            self.replay_buffer_size,
            env,
            self.discount,
        )
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)
        self.eval_statistics = None

    def _start_epoch(self, epoch):
        super()._start_epoch(epoch)
        self.replay_buffer.empty_buffer()

    def get_batch(self):
        batch = self.replay_buffer.get_training_data()
        return np_to_pytorch_batch(batch)

    def _do_training(self):
        batch = self.get_batch()
        obs = batch['observations']
        actions = batch['actions']
        returns = batch['returns']
        """
        Policy operations.
        """

        _, means, _, _, _, stds,_, _ = self.policy.forward(obs,)
        log_probs = TanhNormal(means, stds).log_prob(actions)
        log_probs_times_returns = np.multiply(log_probs, returns)
        policy_loss = -1*np.mean(log_probs_times_returns)

        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))

    def _can_train(self):
        return True

    def _can_evaluate(self):
        return True

    def offline_evaluate(self, epoch):
        statistics = OrderedDict()
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def get_epoch_snapshot(self, epoch):
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
            policy=self.eval_policy,
            trained_policy=self.policy,
            exploration_policy=self.exploration_policy,
            batch_size=self.batch_size,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    @property
    def networks(self):
        return [
            self.policy,
        ]
