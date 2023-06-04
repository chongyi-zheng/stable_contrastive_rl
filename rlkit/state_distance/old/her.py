from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F

from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.state_distance.policies import UniversalPolicy
from rlkit.state_distance.rollout_util import MultigoalSimplePathSampler
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.torch.data_management.normalizer import TorchNormalizer
from rlkit.torch.networks import Mlp


class HER(DDPG):
    """
    Questions:
    - Is episode over when the state is reached? Not in their code.
    - Do you give time to the state? Can't tell
    - Do you mean that you use the target policy for eval? Yes
    - "we add the square of the their preactivations to the actor’s cost
        function" is there a weight?
        The code actually penalized the policy actions and not the
        preactivations (see line 261 of ddpg.py)
    - Does the replay buffer size (10^6) mean 10^6 unique states or
    "state + goal states" (since they save new goal states into the replay
    buffer)? state + goal states

    Known differences:
     - Mujoco skip_frame and dt are different
     - I do not scale inputs
    """
    def __init__(
            self,
            env,
            qf,
            policy,
            exploration_policy,
            replay_buffer,
            obs_normalizer: TorchNormalizer=None,
            goal_normalizer: TorchNormalizer=None,
            eval_sampler=None,
            epsilon=1e-4,
            num_steps_per_eval=1000,
            max_path_length=1000,
            terminate_when_goal_reached=False,
            pre_activation_weight=1.,
            **kwargs
    ):
        assert isinstance(replay_buffer, HerReplayBuffer)
        assert eval_sampler is None
        super().__init__(
            env, qf, policy, exploration_policy,
            replay_buffer=replay_buffer,
            eval_sampler=eval_sampler,
            num_steps_per_eval=num_steps_per_eval,
            max_path_length=max_path_length,
            **kwargs
        )
        self.obs_normalizer = obs_normalizer
        self.goal_normalizer = goal_normalizer
        self.eval_sampler = MultigoalSimplePathSampler(
            env=env,
            policy=self.target_policy,
            max_samples=num_steps_per_eval,
            max_path_length=max_path_length,
            tau_sampling_function=self._sample_tau_for_rollout,
            goal_sampling_function=self._sample_goal_for_rollout,
            cycle_taus_for_rollout=False,
        )
        self.epsilon = epsilon
        assert self.qf_weight_decay == 0
        assert self.residual_gradient_weight == 0
        self.terminate_when_goal_reached = terminate_when_goal_reached
        self.pre_activation_weight = pre_activation_weight
        self._current_path_goal = None

    def _sample_goal_for_rollout(self):
        return self.env.sample_goal_for_rollout()

    def _sample_tau_for_rollout(self):
        return 0  # Her does not vary tau.

    def get_batch(self, training=True):
        batch = super().get_batch(training=training)
        diff = torch.abs(
            self.env.convert_obs_to_goals(batch['next_observations'])
            - batch['goals']
        )
        diff_sum = diff.sum(dim=1, keepdim=True)
        goal_not_reached = (diff_sum >= self.epsilon).float()
        batch['rewards'] = - goal_not_reached
        if self.terminate_when_goal_reached:
            batch['terminals'] = 1 - (1 - batch['terminals']) * goal_not_reached
        return batch

    def _do_training(self):
        batch = self.get_batch(training=True)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['goals']

        """
        Policy operations.
        """
        policy_actions, preactivations = self.policy(
            obs, goals, return_preactivations=True,
        )
        pre_activation_policy_loss = self.pre_activation_weight * (
            (preactivations**2).sum(dim=1).mean()
        )
        q_output = self.qf(obs, policy_actions, goals)
        raw_policy_loss = - q_output.mean()
        policy_loss = (
            raw_policy_loss
            + self.pre_activation_weight * pre_activation_policy_loss
        )

        """
        Critic operations.
        """
        next_actions = self.target_policy(next_obs, goals)
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
            goals,
        )
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
        q_target = torch.clamp(q_target, -1./(1-self.discount), 0)
        q_pred = self.qf(obs, actions, goals)
        bellman_errors = (q_pred - q_target)**2
        qf_loss = self.qf_criterion(q_pred, q_target)

        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self._update_target_networks()

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['Raw Policy Loss'] = np.mean(ptu.get_numpy(
                raw_policy_loss
            ))
            self.eval_statistics['Pre-activation Policy Loss'] = np.mean(
                ptu.get_numpy(pre_activation_policy_loss)
            )
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors',
                ptu.get_numpy(bellman_errors),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self._current_path_goal = self._sample_goal_for_rollout()
        self.training_env.set_goal(self._current_path_goal)
        self.exploration_policy.set_goal(self._current_path_goal)
        return self.training_env.reset()

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            num_steps_left=np.array([0]),
            goals=self._current_path_goal,
        )
        if self.obs_normalizer:
            self.obs_normalizer.update(observation)
        if self.goal_normalizer:
            self.goal_normalizer.update(
                self.env.convert_ob_to_goal(observation)
            )

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _handle_path(self, path):
        self._n_rollouts_total += 1
        self.replay_buffer.add_path(path)


class HerQFunction(Mlp):
    def __init__(
            self,
            env,
            obs_normalizer: TorchNormalizer,
            goal_normalizer: TorchNormalizer,
            hidden_sizes,
            **kwargs
    ):
        self.save_init_params(locals())
        observation_dim = int(np.prod(env.observation_space.low.shape))
        action_dim = int(np.prod(env.action_space.low.shape))
        goal_dim = env.goal_dim
        super().__init__(
            hidden_sizes,
            output_size=1,
            input_size=observation_dim + goal_dim + action_dim,
            **kwargs
        )
        self.env = env
        self.obs_normalizer = obs_normalizer
        self.goal_normalizer = goal_normalizer

    def forward(self, obs, action, goals, _ignored_discount=None):
        obs = self.obs_normalizer.normalize(obs)
        goals = self.goal_normalizer.normalize(goals)

        goal_deltas = self.env.convert_obs_to_goals(obs) - goals
        h = torch.cat((obs, action, goal_deltas), dim=1)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        return self.output_activation(self.last_fc(h))


class HerPolicy(Mlp, UniversalPolicy):
    def __init__(
            self,
            env,
            obs_normalizer,
            goal_normalizer,
            hidden_sizes,
            **kwargs
    ):
        self.save_init_params(locals())
        observation_dim = int(np.prod(env.observation_space.low.shape))
        action_dim = int(np.prod(env.action_space.low.shape))
        goal_dim = env.goal_dim
        super().__init__(
            hidden_sizes,
            output_size=action_dim,
            input_size=observation_dim + goal_dim,
            **kwargs
        )
        self.env = env
        self.obs_normalizer = obs_normalizer
        self.goal_normalizer = goal_normalizer

    def forward(self, obs, goals, return_preactivations=False):
        obs = self.obs_normalizer.normalize(obs)
        goals = self.goal_normalizer.normalize(goals)

        goal_deltas = self.env.convert_obs_to_goals(obs) - goals
        h = torch.cat((obs, goal_deltas), dim=1)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        if return_preactivations:
            preactivations = self.last_fc(h)
            return F.tanh(preactivations), preactivations
        else:
            return F.tanh(self.last_fc(h))

    def get_action(self, obs_np):
        obs = ptu.np_to_var(
            np.expand_dims(obs_np, 0)
        )
        action = self.__call__(
            obs,
            self._goal_expanded_torch,
        )
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}
