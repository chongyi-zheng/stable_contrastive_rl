from collections import OrderedDict

import numpy as np
from torch import optim as optim

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core import logger


def create_batch_iterator(num_unique_batches, batch_dict):
    while True:
        for i in range(num_unique_batches):
            yield batch_dict[i]


class SupervisedLearning(object):
    def __init__(
            self,
            env,
            qf,
            replay_buffer,
            num_epochs=100,
            num_batches_per_epoch=100,
            qf_learning_rate=1e-3,
            batch_size=100,
            num_unique_batches=1000,
    ):
        self.qf = qf
        self.replay_buffer = replay_buffer
        self.env = env
        self.num_epochs = num_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.qf_learning_rate = qf_learning_rate
        self.batch_size = batch_size
        self.num_unique_batches = num_unique_batches

        self.qf_optimizer = optim.Adam(self.qf.parameters(),
                                       lr=self.qf_learning_rate)
        self.batch_iterator = None
        self.discount = ptu.Variable(
            ptu.from_numpy(np.zeros((batch_size, 1))).float()
        )
        self.mode_to_batch_iterator = {}

    def train(self):
        self.fix_data_set()
        logger.log("Done creating dataset.")
        num_batches_total = 0
        for epoch in range(self.num_epochs):
            for _ in range(self.num_batches_per_epoch):
                self.qf.train(True)
                self._do_training()
                num_batches_total += 1
            logger.push_prefix('Iteration #%d | ' % epoch)
            self.qf.train(False)
            self.evaluate(epoch)
            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            logger.log("Done evaluating")
            logger.pop_prefix()

    def fix_data_set(self):
        for training in [True, False]:
            replay_buffer = self.replay_buffer.get_replay_buffer(training)
            batch_dict = {}
            for i in range(self.num_unique_batches):
                batch_size = min(
                    replay_buffer.num_steps_can_sample(),
                    self.batch_size
                )
                batch = replay_buffer.random_batch(batch_size)
                goal_states = self.sample_goal_states(batch_size, training)
                new_rewards = self.env.compute_rewards(
                    batch['observations'],
                    batch['actions'],
                    batch['next_observations'],
                    goal_states,
                )
                batch['goal_states'] = goal_states
                batch['rewards'] = new_rewards
                torch_batch = np_to_pytorch_batch(batch)
                batch_dict[i] = torch_batch
            self.mode_to_batch_iterator[training] = create_batch_iterator(
                self.num_unique_batches, batch_dict
            )

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            qf=self.qf,
            replay_buffer=self.replay_buffer,
        )

    def to(self, device=ptu.device):
        self.qf.to(device)

    def _do_training(self):
        batch = self.get_batch()
        train_dict = self.get_train_dict(batch)

        self.qf_optimizer.zero_grad()
        qf_loss = train_dict['QF Loss']
        qf_loss.backward()
        self.qf_optimizer.step()

    def get_batch(self, training=True):
        batch_iterator = self.mode_to_batch_iterator[training]
        batch = next(batch_iterator)
        return batch

    def sample_goal_states(self, batch_size, training):
        replay_buffer = self.replay_buffer.get_replay_buffer(training=training)
        batch = replay_buffer.random_batch(batch_size)
        obs = batch['next_observations']
        return self.env.convert_obs_to_goal_state(obs)

    def evaluate(self, epoch):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        :param exploration_paths: List of dicts, each representing a path.
        """
        statistics = OrderedDict()
        train_batch = self.get_batch(training=True)
        statistics.update(self._statistics_from_batch(train_batch, "Train"))
        validation_batch = self.get_batch(training=False)
        statistics.update(
            self._statistics_from_batch(validation_batch, "Validation")
        )

        statistics['QF Loss Validation - Train Gap'] = (
            statistics['Validation QF Loss Mean']
            - statistics['Train QF Loss Mean']
        )
        statistics['Epoch'] = epoch
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = OrderedDict()

        train_dict = self.get_train_dict(batch)
        for name in [
            'QF Loss',
        ]:
            tensor = train_dict[name]
            statistics_name = "{} {} Mean".format(stat_prefix, name)
            statistics[statistics_name] = np.mean(ptu.get_numpy(tensor))

        for name in [
            'Bellman Errors',
        ]:
            tensor = train_dict[name]
            statistics.update(create_stats_ordered_dict(
                '{} {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
            ))

        return statistics

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        obs = batch['observations']
        actions = batch['actions']
        goal_states = batch['goal_states']

        y_target = self.reward_scale * rewards

        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions, goal_states, self.discount)
        bellman_errors = (y_pred - y_target)**2
        qf_loss = bellman_errors.mean()

        return OrderedDict([
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('QF Loss', qf_loss),
        ])