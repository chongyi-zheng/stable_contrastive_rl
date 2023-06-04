import time
from collections import OrderedDict

import numpy as np
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.util import split_paths
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core.eval_util import get_generic_path_information, \
    get_average_returns, create_stats_ordered_dict
from rlkit.core import logger


class DdpgQfCombiner(object):
    def __init__(
            self,
            env,
            qf1,
            qf2,
            policy,
            replay_buffer1,
            replay_buffer2,
            num_epochs=1000,
            num_steps_per_epoch=1000,
            policy_learning_rate=1e-4,
            batch_size=128,
            num_steps_per_eval=3000,
            max_path_length=300,
            discount=0.99,
    ):
        super().__init__()
        self.env = env
        self.qf1 = qf1
        self.qf2 = qf2
        self.policy = policy
        self.replay_buffer1 = replay_buffer1
        self.replay_buffer2 = replay_buffer2
        self.num_steps_per_epoch = num_steps_per_epoch
        self.num_epochs = num_epochs
        self.policy_learning_rate = policy_learning_rate
        self.batch_size = batch_size
        self.discount = discount

        self.eval_sampler = InPlacePathSampler(
            env=env,
            policy=self.policy,
            max_samples=num_steps_per_eval,
            max_path_length=max_path_length,
        )

        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            logger.push_prefix('Iteration #%d | ' % epoch)

            start_time = time.time()
            for _ in range(self.num_steps_per_epoch):
                batch = self.get_batch()
                train_dict = self.get_train_dict(batch)

                self.policy_optimizer.zero_grad()
                policy_loss = train_dict['Policy Loss']
                policy_loss.backward()
                self.policy_optimizer.step()
            logger.log("Train time: {}".format(time.time() - start_time))

            start_time = time.time()
            self.evaluate(epoch)
            logger.log("Eval time: {}".format(time.time() - start_time))

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            logger.pop_prefix()

    def to(self, device=ptu.device):
        self.policy.to(device)
        self.qf1.to(device)
        self.qf2.to(device)

    def get_batch(self):
        sample_size = self.batch_size // 2
        batch1 = self.replay_buffer1().random_batch(sample_size)
        batch2 = self.replay_buffer2().random_batch(sample_size)
        new_batch = {}
        for k, v in batch1.items():
            new_batch[k] = np.concatenate(
                (
                    v,
                    batch2[k]
                ),
                axis=0,
            )
        return np_to_pytorch_batch(new_batch)

    def get_train_dict(self, batch):
        obs = batch['observations']

        policy_actions = self.policy(obs)
        q_output = self.qf1(obs, policy_actions) + self.qf2(obs, policy_actions)
        policy_loss = - q_output.mean()

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('QF Outputs', q_output),
        ])

    def evaluate(self, epoch):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        """
        statistics = OrderedDict()

        train_batch = self.get_batch()
        statistics.update(self._statistics_from_batch(train_batch, "Train"))

        logger.log("Collecting samples for evaluation")
        test_paths = self._sample_eval_paths()
        statistics.update(get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        statistics.update(self._statistics_from_paths(test_paths, "Test"))
        average_returns = get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns

        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.env.log_diagnostics(test_paths)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def _statistics_from_paths(self, paths, stat_prefix):
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        np_batch = dict(
            rewards=rewards,
            terminals=terminals,
            observations=obs,
            actions=actions,
            next_observations=next_obs,
        )
        batch = np_to_pytorch_batch(np_batch)
        statistics = self._statistics_from_batch(batch, stat_prefix)
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = OrderedDict()

        train_dict = self.get_train_dict(batch)
        for name in [
            'Policy Loss',
        ]:
            tensor = train_dict[name]
            statistics_name = "{} {} Mean".format(stat_prefix, name)
            statistics[statistics_name] = np.mean(ptu.get_numpy(tensor))

        for name in [
            'QF Outputs',
            'Policy Actions',
        ]:
            tensor = train_dict[name]
            statistics.update(create_stats_ordered_dict(
                '{} {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
            ))

        statistics.update(create_stats_ordered_dict(
            "{} Env Actions".format(stat_prefix),
            ptu.get_numpy(batch['actions'])
        ))

        return statistics

    def _sample_eval_paths(self):
        return self.eval_sampler.obtain_samples()

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            policy=self.policy,
            env=self.env,
            algo=self,
        )
