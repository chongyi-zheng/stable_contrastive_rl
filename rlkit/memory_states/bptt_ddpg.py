from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.data_management.updatable_subtraj_replay_buffer import (
    UpdatableSubtrajReplayBuffer
)
from rlkit.core.eval_util import get_average_returns, create_stats_ordered_dict
from rlkit.pythonplusplus import batch, ConditionTimer
from rlkit.core.rl_algorithm import RLAlgorithm
from rlkit.torch import pytorch_util as ptu
from rlkit.core import logger, eval_util


# noinspection PyCallingNonCallable
class BpttDdpg(RLAlgorithm):
    """
    BPTT DDPG implemented in pytorch.
    """

    def __init__(
            self,
            env,
            qf,
            policy,
            exploration_strategy,
            subtraj_length,
            tau=0.01,
            use_soft_update=True,
            target_hard_update_period=1000,
            use_action_policy_params_for_entire_policy=False,
            action_policy_optimize_bellman=True,
            write_policy_optimizes='both',
            action_policy_learning_rate=1e-3,
            write_policy_learning_rate=1e-5,
            qf_learning_rate=1e-3,
            bellman_error_loss_weight=10,
            refresh_entire_buffer_period=None,
            save_new_memories_back_to_replay_buffer=True,
            only_use_last_dqdm=False,
            action_policy_weight_decay=0,
            write_policy_weight_decay=0,
            do_not_load_initial_memories=False,
            save_memory_gradients=False,
            **kwargs
    ):
        """
        :param args: arguments to be passed onto super class constructor
        :param qf: Q function to train
        :param policy: Policy trained to optimized the Q function
        :param subtraj_length: Length of the subtrajectories loaded
        :param tau: Soft target tau
        :param use_soft_update: If False, use hard target updates.
        :param target_hard_update_period: Number of environment steps between
        hard updates.
        :param use_action_policy_params_for_entire_policy: If True, train the
        entire policy together, rather than training the action and write parts
        separately.
        :param action_policy_optimize_bellman:
        :param write_policy_optimizes:
        :param action_policy_learning_rate:
        :param write_policy_learning_rate:
        :param qf_learning_rate:
        :param bellman_error_loss_weight:
        :param refresh_entire_buffer_period:
        :param save_new_memories_back_to_replay_buffer:
        :param only_use_last_dqdm: If True, cut the gradients for all dQ/dmemory
        other than the last time step.
        :param action_policy_weight_decay:
        :param do_not_load_initial_memories: If True, always zero-out the
        loaded initial memory.
        :param write_policy_weight_decay:
        :param save_memory_gradients: If True, save and load dL/dmemory.
        :param kwargs: kwargs to pass onto super class constructor
        """
        super().__init__(env, policy, exploration_strategy, **kwargs)
        assert write_policy_optimizes in ['qf', 'bellman', 'both']
        self.qf = qf
        self.policy = policy
        self.subtraj_length = subtraj_length
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.target_hard_update_period = target_hard_update_period
        self.use_action_policy_params_for_entire_policy = (
            use_action_policy_params_for_entire_policy
        )
        self.action_policy_optimize_bellman = action_policy_optimize_bellman
        self.write_policy_optimizes = write_policy_optimizes
        self.action_policy_learning_rate = action_policy_learning_rate
        self.write_policy_learning_rate = write_policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.bellman_error_loss_weight = bellman_error_loss_weight
        self.should_refresh_buffer = ConditionTimer(
            refresh_entire_buffer_period
        )
        self.save_new_memories_back_to_replay_buffer = (
            save_new_memories_back_to_replay_buffer
        )
        self.only_use_last_dqdm = only_use_last_dqdm
        self.action_policy_weight_decay = action_policy_weight_decay
        self.write_policy_weight_decay = write_policy_weight_decay
        self.do_not_load_initial_memories = do_not_load_initial_memories
        self.save_memory_gradients = save_memory_gradients

        """
        Set some params-dependency values
        """
        self.num_subtrajs_per_batch = self.batch_size // self.subtraj_length
        self.train_validation_num_subtrajs_per_batch = (
            self.num_subtrajs_per_batch
        )
        self.action_dim = int(self.env.action_space.flat_dim)
        self.obs_dim = int(self.env.observation_space.flat_dim)
        self.memory_dim = self.env.memory_dim
        self.max_number_trajectories_loaded_at_once = (
            self.num_subtrajs_per_batch
        )

        if not self.save_new_memories_back_to_replay_buffer:
            assert self.should_refresh_buffer.always_false, (
                "If save_new_memories_back_to_replay_buffer is False, "
                "you cannot refresh the replay buffer."
            )

        """
        Create the necessary node objects.
        """
        self.replay_buffer = SplitReplayBuffer(
            UpdatableSubtrajReplayBuffer(
                self.replay_buffer_size,
                self.env,
                self.subtraj_length,
                self.memory_dim,
            ),
            UpdatableSubtrajReplayBuffer(
                self.replay_buffer_size,
                self.env,
                self.subtraj_length,
                self.memory_dim,
            ),
            fraction_paths_in_train=0.8,
        )
        self.target_qf = self.qf.copy()
        self.target_policy = self.policy.copy()

        self.qf_optimizer = optim.Adam(
            self.qf.parameters(), lr=self.qf_learning_rate
        )
        self.action_policy_optimizer = optim.Adam(
            self.policy.action_parameters(),
            lr=self.action_policy_learning_rate,
            weight_decay=self.action_policy_weight_decay,
        )
        self.write_policy_optimizer = optim.Adam(
            self.policy.write_parameters(),
            lr=self.write_policy_learning_rate,
            weight_decay=self.write_policy_weight_decay,
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.action_policy_learning_rate,
            weight_decay=self.action_policy_weight_decay,
        )

        if self.save_memory_gradients:
            self.saved_grads = {}
            self.save_hook = self.create_save_grad_hook('dl_dmemory')

    """
    Training functions
    """

    def _do_training(self, n_steps_total):
        raw_subtraj_batch, start_indices = (
            self.replay_buffer.train_replay_buffer.random_subtrajectories(
                self.num_subtrajs_per_batch
            )
        )
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
        if self.save_memory_gradients:
            subtraj_batch['memories'].requires_grad = True
        self.train_critic(subtraj_batch)
        self.train_policy(subtraj_batch, start_indices)
        if self.use_soft_update:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf, self.target_qf, self.tau)
        else:
            if n_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
                ptu.copy_model_params_from_to(self.policy, self.target_policy)

    def train_critic(self, subtraj_batch):
        critic_dict = self.get_critic_output_dict(subtraj_batch)
        qf_loss = critic_dict['Loss']
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        return qf_loss

    def get_critic_output_dict(self, subtraj_batch):
        """
        :param subtraj_batch: A tensor subtrajectory dict. Basically, it should
        be the output of `create_torch_subtraj_batch`
        :return: Dictionary containing Variables/Tensors for training the
        critic, including intermediate values that might be useful to log.
        """
        flat_batch = flatten_subtraj_batch(subtraj_batch)
        rewards = flat_batch['rewards']
        terminals = flat_batch['terminals']
        obs = flat_batch['env_obs']
        actions = flat_batch['env_actions']
        next_obs = flat_batch['next_env_obs']
        memories = flat_batch['memories']
        writes = flat_batch['writes']
        next_memories = flat_batch['next_memories']

        next_actions, next_writes = self.target_policy.get_flat_output(
            next_obs, next_memories
        )
        target_q_values = self.target_qf(
            next_obs,
            next_memories,
            next_actions,
            next_writes
        )
        y_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_predicted = self.qf(obs, memories, actions, writes)
        bellman_errors = (y_predicted - y_target) ** 2
        return OrderedDict([
            ('Target Q Values', target_q_values),
            ('Y target', y_target),
            ('Y predicted', y_predicted),
            ('Bellman Errors', bellman_errors),
            ('Loss', bellman_errors.mean()),
        ])

    def train_policy(self, subtraj_batch, start_indices):
        policy_dict = self.get_policy_output_dict(subtraj_batch)

        policy_loss = policy_dict['Loss']
        qf_loss = 0
        if self.save_memory_gradients:
            dloss_dlast_writes = subtraj_batch['dloss_dwrites'][:, -1, :]
            new_last_writes = policy_dict['New Writes'][:, -1, :]
            qf_loss += (dloss_dlast_writes * new_last_writes).sum()
        if self.use_action_policy_params_for_entire_policy:
            self.policy_optimizer.zero_grad()
            policy_loss.backward(
                retain_variables=self.action_policy_optimize_bellman
            )
            if self.action_policy_optimize_bellman:
                bellman_errors = policy_dict['Bellman Errors']
                qf_loss += (
                    self.bellman_error_loss_weight * bellman_errors.mean()
                )
                qf_loss.backward()
            self.policy_optimizer.step()
        else:
            self.action_policy_optimizer.zero_grad()
            self.write_policy_optimizer.zero_grad()
            policy_loss.backward(retain_variables=True)

            if self.write_policy_optimizes == 'qf':
                self.write_policy_optimizer.step()
                if self.action_policy_optimize_bellman:
                    bellman_errors = policy_dict['Bellman Errors']
                    qf_loss += (
                        self.bellman_error_loss_weight * bellman_errors.mean()
                    )
                    qf_loss.backward()
                self.action_policy_optimizer.step()
            else:
                if self.write_policy_optimizes == 'bellman':
                    self.write_policy_optimizer.zero_grad()
                if self.action_policy_optimize_bellman:
                    bellman_errors = policy_dict['Bellman Errors']
                    qf_loss += (
                        self.bellman_error_loss_weight * bellman_errors.mean()
                    )
                    qf_loss.backward()
                    self.action_policy_optimizer.step()
                else:
                    self.action_policy_optimizer.step()
                    bellman_errors = policy_dict['Bellman Errors']
                    qf_loss += (
                        self.bellman_error_loss_weight * bellman_errors.mean()
                    )
                    qf_loss.backward()
                self.write_policy_optimizer.step()

        if self.save_new_memories_back_to_replay_buffer:
            self.replay_buffer.train_replay_buffer.update_write_subtrajectories(
                ptu.get_numpy(policy_dict['New Writes']), start_indices
            )
        if self.save_memory_gradients:
            new_dloss_dmemory = ptu.get_numpy(self.saved_grads['dl_dmemory'])
            self.replay_buffer.train_replay_buffer.update_dloss_dmemories_subtrajectories(
                new_dloss_dmemory, start_indices
            )

    def get_policy_output_dict(self, subtraj_batch):
        """
        :param subtraj_batch: A tensor subtrajectory dict. Basically, it should
        be the output of `create_torch_subtraj_batch`
        :return: Dictionary containing Variables/Tensors for training the
        policy, including intermediate values that might be useful to log.
        """
        subtraj_obs = subtraj_batch['env_obs']
        initial_memories = subtraj_batch['memories'][:, 0, :]
        if self.do_not_load_initial_memories:
            initial_memories.data.fill_(0)
        policy_actions, policy_writes = self.policy(subtraj_obs,
                                                    initial_memories)
        if self.subtraj_length > 1:
            new_memories = torch.cat(
                (
                    initial_memories.unsqueeze(1),
                    policy_writes[:, :-1, :],
                ),
                dim=1,
            )
        else:
            new_memories = initial_memories.unsqueeze(1)
        if self.save_memory_gradients:
            new_memories.register_hook(
                self.save_hook
            )
        # TODO(vitchyr): Test this
        if self.only_use_last_dqdm:
            new_memories = new_memories.detach()
            policy_writes = torch.cat(
                (
                    policy_writes[:, :-1, :].detach(),
                    policy_writes[:, -1:, :]
                ),
                dim=1
            )
        subtraj_batch['policy_new_memories'] = new_memories
        subtraj_batch['policy_new_writes'] = policy_writes
        subtraj_batch['policy_new_actions'] = policy_actions

        flat_batch = flatten_subtraj_batch(subtraj_batch)
        flat_obs = flat_batch['env_obs']
        flat_new_memories = flat_batch['policy_new_memories']
        flat_new_actions = flat_batch['policy_new_actions']
        flat_new_writes = flat_batch['policy_new_writes']
        q_output = self.qf(
            flat_obs,
            flat_new_memories,
            flat_new_actions,
            flat_new_writes
        )
        policy_loss = - q_output.mean()

        """
        Train policy to minimize Bellman error as well.
        """
        flat_next_obs = flat_batch['next_env_obs']
        flat_actions = flat_batch['env_actions']
        flat_rewards = flat_batch['rewards']
        flat_terminals = flat_batch['terminals']
        flat_next_memories = flat_new_writes
        flat_next_actions, flat_next_writes = self.policy.get_flat_output(
            flat_next_obs, flat_next_memories
        )
        target_q_values = self.target_qf(
            flat_next_obs,
            flat_next_memories,
            flat_next_actions,
            flat_next_writes
        )
        y_target = (
            self.reward_scale * flat_rewards
            + (1. - flat_terminals) * self.discount * target_q_values
        )
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_predicted = self.qf(flat_obs, flat_new_memories, flat_actions,
                              flat_new_writes)
        bellman_errors = (y_predicted - y_target) ** 2
        # TODO(vitchyr): Still use target policies when minimizing Bellman err?
        return OrderedDict([
            ('Target Q Values', target_q_values),
            ('Y target', y_target),
            ('Y predicted', y_predicted),
            ('Bellman Errors', bellman_errors),
            ('Q Output', q_output),
            ('Loss', policy_loss),
            ('New Env Actions', flat_batch['policy_new_actions']),
            ('New Writes', policy_writes),
        ])

    """
    Eval functions
    """

    def evaluate(self, epoch, exploration_paths):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        :param exploration_paths: List of dicts, each representing a path.
        """
        logger.log("Collecting samples for evaluation")
        paths = self._sample_eval_paths(epoch)
        statistics = OrderedDict()

        statistics.update(self._statistics_from_paths(paths, "Test"))
        statistics.update(self._get_other_statistics())
        statistics.update(self._statistics_from_paths(exploration_paths,
                                                      "Exploration"))

        statistics['AverageReturn'] = get_average_returns(paths)
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.log_diagnostics(paths)

    def _statistics_from_paths(self, paths, stat_prefix):
        eval_replay_buffer = UpdatableSubtrajReplayBuffer(
            len(paths) * (self.max_path_length + 1),
            self.env,
            self.subtraj_length,
            self.memory_dim,
        )
        for path in paths:
            eval_replay_buffer.add_trajectory(path)
        raw_subtraj_batch = eval_replay_buffer.get_all_valid_subtrajectories()
        assert raw_subtraj_batch is not None
        subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
        if self.save_memory_gradients:
            subtraj_batch['memories'].requires_grad = True
        statistics = self._statistics_from_subtraj_batch(
            subtraj_batch, stat_prefix=stat_prefix
        )
        statistics.update(eval_util.get_generic_path_information(
            paths, stat_prefix="Test",
        ))
        env_actions = np.vstack([path["actions"][:self.action_dim] for path in
                                 paths])
        writes = np.vstack([path["actions"][self.action_dim:] for path in
                            paths])
        statistics.update(create_stats_ordered_dict(
            'Env Actions', env_actions, stat_prefix=stat_prefix
        ))
        statistics.update(create_stats_ordered_dict(
            'Writes', writes, stat_prefix=stat_prefix
        ))
        return statistics

    def _statistics_from_subtraj_batch(self, subtraj_batch, stat_prefix=''):
        statistics = OrderedDict()

        critic_dict = self.get_critic_output_dict(subtraj_batch)
        for name, tensor in critic_dict.items():
            statistics.update(create_stats_ordered_dict(
                '{} QF {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
            ))

        policy_dict = self.get_policy_output_dict(subtraj_batch)
        for name, tensor in policy_dict.items():
            statistics.update(create_stats_ordered_dict(
                '{} Policy {}'.format(stat_prefix, name),
                ptu.get_numpy(tensor)
            ))
        return statistics

    def _get_other_statistics(self):
        statistics = OrderedDict()
        for stat_prefix, training in [
            ('Validation', False),
            ('Train', True),
        ]:
            replay_buffer = self.replay_buffer.get_replay_buffer(training=training)
            sample_size = min(
                replay_buffer.num_subtrajs_can_sample(),
                self.train_validation_num_subtrajs_per_batch
            )
            raw_subtraj_batch = replay_buffer.random_subtrajectories(sample_size)[0]
            subtraj_batch = create_torch_subtraj_batch(raw_subtraj_batch)
            if self.save_memory_gradients:
                subtraj_batch['memories'].requires_grad = True
            statistics.update(self._statistics_from_subtraj_batch(
                subtraj_batch, stat_prefix=stat_prefix
            ))
        return statistics

    def _can_evaluate(self, exploration_paths):
        return (
            self.replay_buffer.train_replay_buffer.num_subtrajs_can_sample() >= 1
            and
            self.replay_buffer.validation_replay_buffer.num_subtrajs_can_sample() >= 1
            and len(exploration_paths) > 0
            and any([len(path['terminals']) >= self.subtraj_length
                     for path in exploration_paths])
            # Technically, I should also check that the exploration path has
            # enough subtraj batches, but whatever.
        )

    """
    Random small functions.
    """

    def _can_train(self):
        return (
            self.replay_buffer.train_replay_buffer.num_subtrajs_can_sample()
            >= self.num_subtrajs_per_batch
        )

    def _sample_eval_paths(self, epoch):
        """
        Returns flattened paths.

        :param epoch: Epoch number
        :return: Dictionary with these keys:
            observations: np.ndarray, shape BATCH_SIZE x flat observation dim
            actions: np.ndarray, shape BATCH_SIZE x flat action dim
            rewards: np.ndarray, shape BATCH_SIZE
            terminals: np.ndarray, shape BATCH_SIZE
            agent_infos: unsure
            env_infos: unsure
        """
        # Sampler uses self.batch_size to figure out how many samples to get
        saved_batch_size = self.batch_size
        self.batch_size = self.num_steps_per_eval
        paths = self.eval_sampler.obtain_samples()
        self.batch_size = saved_batch_size
        return paths

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.training_env,
            epoch=epoch,
            policy=self.policy,
            es=self.exploration_strategy,
            qf=self.qf,
        )

    def _handle_rollout_ending(self, n_steps_total):
        if not self._can_train():
            return

        if self.should_refresh_buffer.check(n_steps_total):
            for replay_buffer in [
                    self.replay_buffer.train_replay_buffer,
                    self.replay_buffer.validation_replay_buffer,
            ]:
                for start_traj_indices in batch(
                        replay_buffer.get_all_valid_trajectory_start_indices(),
                        self.max_number_trajectories_loaded_at_once,
                ):
                    raw_subtraj_batch, start_indices = (
                        replay_buffer.get_trajectory_minimal_covering_subsequences(
                            start_traj_indices, self.training_env.horizon
                        )
                    )
                    subtraj_batch = create_torch_subtraj_batch(
                        raw_subtraj_batch
                    )
                    subtraj_obs = subtraj_batch['env_obs']
                    initial_memories = subtraj_batch['memories'][:, 0, :]
                    _, policy_writes = self.policy(
                        subtraj_obs, initial_memories
                    )
                    replay_buffer.update_write_subtrajectories(
                        ptu.get_numpy(policy_writes), start_indices
                    )

    def create_save_grad_hook(self, key):
        def save_grad_hook(grad):
            self.saved_grads[key] = grad
        return save_grad_hook


def flatten_subtraj_batch(subtraj_batch):
    return {
        k: array.view(-1, array.size()[-1])
        for k, array in subtraj_batch.items()
    }


def create_torch_subtraj_batch(subtraj_batch):
    torch_batch = {
        k: Variable(ptu.from_numpy(array).float(), requires_grad=False)
        for k, array in subtraj_batch.items()
    }
    rewards = torch_batch['rewards']
    terminals = torch_batch['terminals']
    torch_batch['rewards'] = rewards.unsqueeze(-1)
    torch_batch['terminals'] = terminals.unsqueeze(-1)
    return torch_batch
