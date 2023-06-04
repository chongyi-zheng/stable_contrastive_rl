from collections import OrderedDict

import torch

from rlkit.memory_states.bptt_ddpg import BpttDdpg


class BpttDdpgRecurrentQ(BpttDdpg):
    """
    Same as BpttDdpg, but with a recurrent Q function
    """
    def get_critic_output_dict(self, subtraj_batch):
        """
        :param subtraj_batch: A tensor subtrajectory dict. Basically, it should
        be the output of `create_torch_subtraj_batch`
        :return: Dictionary containing Variables/Tensors for training the
        critic, including intermediate values that might be useful to log.
        """
        if not self.qf.is_recurrent:
            return super().get_critic_output_dict(subtraj_batch)
        rewards = subtraj_batch['rewards']
        terminals = subtraj_batch['terminals']
        obs = subtraj_batch['env_obs']
        actions = subtraj_batch['env_actions']
        next_obs = subtraj_batch['next_env_obs']
        memories = subtraj_batch['memories']
        writes = subtraj_batch['writes']
        next_memories = subtraj_batch['next_memories']

        next_actions, next_writes = self.target_policy(
            next_obs, next_memories[:, 0, :]
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

    def get_policy_output_dict(self, subtraj_batch):
        """
        :param subtraj_batch: A tensor subtrajectory dict. Basically, it should
        be the output of `create_torch_subtraj_batch`
        :return: Dictionary containing Variables/Tensors for training the
        policy, including intermediate values that might be useful to log.
        """
        if not self.qf.is_recurrent:
            return super().get_policy_output_dict(subtraj_batch)
        subtraj_obs = subtraj_batch['env_obs']
        initial_memories = subtraj_batch['memories'][:, 0, :]
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
        # TODO(vitchyr): should I detach (stop gradients)?
        # I don't think so. If we have dQ/dmemory, why not use it?
        # new_memories = new_memories.detach()

        obs = subtraj_batch['env_obs']
        q_output = self.qf(
            obs,
            new_memories,
            policy_actions,
            policy_writes
        )
        policy_loss = - q_output.mean()

        """
        Train policy to minimize Bellman error as well.
        """
        next_obs = subtraj_batch['next_env_obs']
        actions = subtraj_batch['env_actions']
        rewards = subtraj_batch['rewards']
        terminals = subtraj_batch['terminals']
        next_memories = policy_writes
        next_actions, next_writes = self.target_policy(
            next_obs, next_memories[:, 0, :]
        )
        target_q_values = self.target_qf(
            next_obs,
            next_memories,
            next_actions,
            next_writes
        )
        y_target = (
            self.reward_scale * rewards
            + (1. - terminals) * self.discount * target_q_values
        )
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        y_predicted = self.qf(obs, new_memories, actions, policy_writes)
        bellman_errors = (y_predicted - y_target) ** 2
        # TODO(vitchyr): Still use target policies when minimizing Bellman err?
        return OrderedDict([
            ('Target Q Values', target_q_values),
            ('Y target', y_target),
            ('Y predicted', y_predicted),
            ('Bellman Errors', bellman_errors),
            ('Loss', policy_loss),
            ('New Env Actions', policy_actions),
            ('New Writes', policy_writes),
        ])
