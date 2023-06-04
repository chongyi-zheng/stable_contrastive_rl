from collections import OrderedDict

from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.data_management.subtraj_replay_buffer import SubtrajReplayBuffer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.bptt_ddpg import create_torch_subtraj_batch
from rlkit.torch.ddpg import DDPG


class Rdpg(DDPG):
    """
    Recurrent DPG.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert self.num_steps_per_eval >= self.env.horizon, (
            "Cannot evaluate RDPG with such short trajectories"
        )
        self.subtraj_length = self.env.horizon
        self.num_subtrajs_per_batch = self.batch_size // self.subtraj_length
        assert self.num_subtrajs_per_batch > 0, "# subtrajs per batch is 0!"

        self.replay_buffer = SplitReplayBuffer(
            SubtrajReplayBuffer(
                self.replay_buffer,
                self.env,
                self.subtraj_length,
            ),
            SubtrajReplayBuffer(
                self.replay_buffer,
                self.env,
                self.subtraj_length,
            ),
            fraction_paths_in_train=0.8,
        )

    """
    Training functions
    """
    def get_train_dict(self, subtraj_batch):
        """
        :param subtraj_batch: A tensor subtrajectory dict. Basically, it should
        be the output of `create_torch_subtraj_batch`
        :return: Dictionary containing Variables/Tensors for training the
        critic, including intermediate values that might be useful to log.
        """
        rewards = subtraj_batch['rewards']
        terminals = subtraj_batch['terminals']
        obs = subtraj_batch['observations']
        actions = subtraj_batch['actions']
        next_obs = subtraj_batch['next_observations']

        policy_actions, _ = self.policy(obs)
        q_output = self.qf(obs, policy_actions)
        policy_loss = - q_output.mean()

        next_actions, _ = self.target_policy(next_obs)
        target_q_values = self.target_qf(next_obs, next_actions)
        y_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_predicted = self.qf(obs, actions)
        bellman_errors = (y_predicted - y_target) ** 2

        return OrderedDict([
            ('Policy Loss', policy_loss),
            ('New Env Actions', policy_actions),
            ('Target Q Values', target_q_values),
            ('Y target', y_target),
            ('Y predicted', y_predicted),
            ('Bellman Errors', bellman_errors),
            ('QF Loss', bellman_errors.mean()),
        ])

    """
    Eval functions
    """
    def _statistics_from_paths(self, paths, stat_prefix):
        eval_replay_buffer = SubtrajReplayBuffer(
            len(paths) * (self.max_path_length + 1),
            self.env,
            self.subtraj_length,
        )
        for path in paths:
            eval_replay_buffer.add_trajectory(path)
        raw_subtraj_batch = eval_replay_buffer.get_all_valid_subtrajectories()
        assert raw_subtraj_batch is not None
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)

        statistics = self._statistics_from_batch(
            subtraj_batch, stat_prefix=stat_prefix
        )
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics
