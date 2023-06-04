from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DSPTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            classifier,
            search_buffer,

            policy_lr=1e-3,
            classifier_lr=1e-3,
            optimizer_class=optim.Adam,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.dsp = dsp
        self.policy = policy
        self.classifier = classifier
        self.search_buffer = search_buffer

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.classifier_criterion = nn.MSELoss()
        self.dsp_optimizer = optimizer_class(
            self.dsp.parameters(),
            lr=policy_lr,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.classisier_optimizer = optimizer_class(
            self.classifier.parameters(),
            lr=classifier_lr,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def initialize_graph(self):
        self.graph = nx.DiGraph()
        self.waypoints = torch.zeros((self.search_buffer.shape[0], \
                                    self.search_buffer.shape[0], \
                                    self.search_buffer.shape[1]))

        for i in range(self.state_size):
            for j in range(self.state_size):
                obs = torch.cat([self.search_buffer[i], self.search_buffer[j]], dim=1)
                actions = self.policy(obs)
                dist = - torch.log(self.classifier(obs, actions))
                self.graph.add_edge(i, j, weight=dist)

        for i in range(self.state_size):
            for j in range(self.state_size):
                path = nx.shortest_path(self.graph, i ,j)
                waypoint_vec = list(path)[1:]
                self.waypoints[i, j] = self.search_buffer[waypoint_vec[0]]

  def get_dsp_batch(self):
    batch_size = 128
    graph_size = self.search_buffer.shape[0]
    batch_indices = np.random.choice((batch_size, 2), graph_size - 1)
    
    s_index, g_index = batch_indices[:, 0], batch_indices[:, 1]
    waypoint = self.waypoints[batch_indices]
    curr_state = self.search_buffer[s_index]
    goal_state = self.search_buffer[g_index]
    
    observation = torch.cat([curr_state, goal_state], dim=1)
    search_obs = torch.cat([curr_state, waypoint], dim=1)
    return observation, search_obs

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Classifier and Policy
        """
        class_actions = self.policy(obs)
        class_prob = self.classifier(obs, actions)
        prob_target = 1 + rewards[:, -1]

        neg_log_prob = - torch.log(self.classifier(obs, class_actions))
        policy_loss = (neg_log_prob).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )
