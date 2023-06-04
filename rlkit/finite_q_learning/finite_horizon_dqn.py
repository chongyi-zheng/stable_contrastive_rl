from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from torch.optim import Adam
from torch import nn as nn
import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch
import random


class FiniteHorizonDQN(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            qf,
            random_action_prob=0.2,
            learning_rate=1e-3,
            max_horizon=None,
            **kwargs
    ):
        super().__init__(env, exploration_policy=None, **kwargs)
        if max_horizon is None:
            max_horizon = self.max_path_length
        self.max_horizon = max_horizon
        self.random_action_prob = random_action_prob
        self.qfs = []
        self.qf_optimizers = []
        if self.max_horizon > 50:
            print("Are you sure you want to make {} q networks??".format(
                self.max_horizon
            ))

        for _ in range(self.max_horizon):
            new_qf = qf.copy(copy_parameters=False)
            self.qfs.append(new_qf)
            self.qf_optimizers.append(
                Adam(new_qf.parameters(), lr=learning_rate)
            )
        self.qf_criterion = nn.MSELoss()
        self._rollout_t = 0

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Compute loss
        """
        for t in range(self.max_horizon):
            if t == self.max_horizon - 1:
                q_target = self.reward_scale * rewards
            else:
                target_q_values = self.qfs[t+1](next_obs).detach().max(
                    1, keepdim=True
                )[0]
                q_target = (
                    self.reward_scale * rewards
                    + (1. - terminals) * self.discount * target_q_values
                )
            # actions are one-hot vectors
            q_pred = torch.sum(self.qfs[t](obs) * actions, dim=1, keepdim=True)
            qf_loss = self.qf_criterion(q_pred, q_target.detach())

            """
            Update networks
            """
            self.qf_optimizers[t].zero_grad()
            qf_loss.backward()
            self.qf_optimizers[t].step()

            """
            Save some statistics for eval
            """
            if self.need_to_update_eval_statistics:
                self.eval_statistics['QF {} Loss'.format(t)] = np.mean(
                    ptu.get_numpy(qf_loss)
                )
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q {} Predictions'.format(t),
                    ptu.get_numpy(q_pred),
                ))
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False

    def get_eval_paths(self):
        paths = []
        n_steps_total = 0
        while n_steps_total <= self.num_steps_per_eval:
            path = self.finite_horizon_rollout()
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths

    def finite_horizon_rollout(self):
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        o = self.env.reset()
        next_o = None
        path_length = 0
        step = 0
        while path_length < self.max_path_length:
            a = self._get_best_action(o, step)
            next_o, r, d, env_info = self.env.step(a)
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append({})
            env_infos.append(env_info)
            path_length += 1
            step = (step + 1) % self.max_horizon
            if d:
                break
            o = next_o

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            next_o = np.array([next_o])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
        )

    def _start_new_rollout(self):
        self._rollout_t = 0
        return self.training_env.reset()

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        if random.random() <= self.random_action_prob:
            action = self.env.action_space.sample()
        else:
            action = self._get_best_action(observation, self._rollout_t)
        return action, {}

    def _get_best_action(self, observation, t):
        obs = ptu.np_to_var(observation[None], requires_grad=False).float()
        q_values = self.qfs[t](obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        return q_values_np.argmax()

    def _handle_step(self, *args, **kwargs):
        super()._handle_step(*args, **kwargs)
        assert 0 <= self._rollout_t < self.max_horizon
        self._rollout_t = (self._rollout_t + 1) % self.max_horizon

    def get_epoch_snapshot(self, epoch):
        data_to_save = super().get_epoch_snapshot(epoch)
        data_to_save['qfs'] = self.qfs
        return data_to_save

    @property
    def networks(self):
        return self.qfs
