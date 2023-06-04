from collections import OrderedDict

import torch
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.data_management.subtraj_replay_buffer import SubtrajReplayBuffer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.ddpg import DDPG
from rlkit.util import np_util


def flatten_subtraj_batch(subtraj_batch):
    return {
        k: array.view(-1, array.size()[-1])
        for k, array in subtraj_batch.items()
    }


class MultiStepDdpg(DDPG):
    def __init__(self, *args, subtraj_length=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.subtraj_length = subtraj_length
        self.gammas = self.discount * torch.ones(self.subtraj_length)
        discount_factors = torch.cumprod(self.gammas, dim=0)
        self.discount_factors = ptu.Variable(
            discount_factors.view(-1, 1),
            requires_grad=False,
        )
        self.replay_buffer = SplitReplayBuffer(
            SubtrajReplayBuffer(
                max_replay_buffer_size=self.replay_buffer_size,
                env=self.env,
                subtraj_length=self.subtraj_length,
            ),
            SubtrajReplayBuffer(
                max_replay_buffer_size=self.replay_buffer_size,
                env=self.env,
                subtraj_length=self.subtraj_length,
            ),
            fraction_paths_in_train=0.8,
        )

    def get_train_dict(self, subtraj_batch):
        subtraj_rewards = subtraj_batch['rewards']
        subtraj_rewards_np = ptu.get_numpy(subtraj_rewards).squeeze(2)
        returns = np_util.batch_discounted_cumsum(
            subtraj_rewards_np, self.discount
        )
        returns = np.expand_dims(returns, 2)
        returns = np.ascontiguousarray(returns).astype(np.float32)
        returns = ptu.Variable(ptu.from_numpy(returns))
        subtraj_batch['returns'] = returns
        batch = flatten_subtraj_batch(subtraj_batch)
        # rewards = batch['rewards']
        returns = batch['returns']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        policy_actions = self.policy(obs)
        q = self.qf(obs, policy_actions)
        policy_loss = - q.mean()

        """
        Critic operations.
        """
        next_actions = self.policy(next_obs)
        # TODO: try to get this to work
        # next_actions = None
        q_target = self.target_qf(
            next_obs,
            next_actions,
        )
        # y_target = self.reward_scale * rewards + (1. - terminals) * self.discount * v_target
        batch_size = q_target.size()[0]
        discount_factors = self.discount_factors.repeat(
            batch_size // self.subtraj_length, 1,
        )
        y_target = self.reward_scale * returns + (1. - terminals) * discount_factors * q_target
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_pred = self.qf(obs, actions)
        bellman_errors = (y_pred - y_target)**2
        qf_loss = self.qf_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy Actions', policy_actions),
            ('Policy Loss', policy_loss),
            ('Policy Q Values', q),
            ('Target Y', y_target),
            ('Predicted Y', y_pred),
            ('Bellman Errors', bellman_errors),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
            ('QF Loss', qf_loss),
        ])

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = OrderedDict()

        train_dict = self.get_train_dict(batch)
        for name in [
            'QF Loss',
            'Policy Loss',
        ]:
            tensor = train_dict[name]
            statistics_name = "{} {} Mean".format(stat_prefix, name)
            statistics[statistics_name] = np.mean(ptu.get_numpy(tensor))

        for name in [
            'Bellman Errors',
            'Target Y',
            'Predicted Y',
            'Policy Q Values',
        ]:
            tensor = train_dict[name]
            statistics.update(create_stats_ordered_dict(
                '{} {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
            ))

        return statistics

    def _paths_to_np_batch(self, paths):
        eval_replay_buffer = SubtrajReplayBuffer(
            len(paths) * (self.max_path_length + 1),
            self.env,
            self.subtraj_length,
        )
        for path in paths:
            eval_replay_buffer.add_trajectory(path)
        return eval_replay_buffer.get_all_valid_subtrajectories()

    def get_batch(self, training=True):
        replay_buffer = self.replay_buffer.get_replay_buffer(training)
        sample_size = min(
            replay_buffer.num_steps_can_sample(),
            self.batch_size
        )
        batch = replay_buffer.random_batch(sample_size)
        torch_batch = {
            k: ptu.Variable(ptu.from_numpy(array).float(), requires_grad=False)
            for k, array in batch.items()
        }
        rewards = torch_batch['rewards']
        terminals = torch_batch['terminals']
        torch_batch['rewards'] = rewards.unsqueeze(-1)
        torch_batch['terminals'] = terminals.unsqueeze(-1)
        return torch_batch
