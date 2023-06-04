import numpy as np
import torch
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.rollout_functions import (
    create_rollout_function,
    multitask_rollout,
)
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.core import np_ify, torch_ify

from rlkit.util.np_util import softmax

class HERExploration(TorchRLAlgorithm):
    """
    This class is for trying different goal-setting strategies for rollouts

    Note: this assumes the env will sample the goal when reset() is called,
    i.e. use a "silent" env.

    Hindsight Experience Replay

    This is a template class that should be the first sub-class, i.e.[

    ```
    class HerDdpg(HER, DDPG):
    ```

    and not

    ```
    class HerDdpg(DDPG, HER):
    ```

    Or if you really want to make DDPG the first subclass, do alternatively:
    ```
    class HerDdpg(DDPG, HER):
        def get_batch(self):
            return HER.get_batch(self)
    ```
    for each function defined below.
    """

    def __init__(
            self,
            observation_key=None,
            desired_goal_key=None,

            rollout_goal_params=None,
    ):
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.rollout_goal_params = rollout_goal_params
        self._rollout_goal = None

        self.train_rollout_function = create_rollout_function(
            multitask_rollout,
            observation_key=self.observation_key,
            desired_goal_key=self.desired_goal_key
        )
        self.eval_rollout_function = self.train_rollout_function

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        # Note: we assume we're using a silent env.
        o = self.training_env.reset()

        rgp = self.rollout_goal_params
        if rgp is None:
            self._rollout_goal = o[self.desired_goal_key]
        elif rgp["strategy"] == "ensemble_qs":
            exploration_temperature = rgp["exploration_temperature"]
            assert len(self.ensemble_qs) > 0
            N = 128
            obs = np.tile(o[self.observation_key], (N, 1))
            proposed_goals = self.training_env.sample_goals(N)[self.desired_goal_key]
            new_obs = np.hstack((obs, proposed_goals))
            actions = torch_ify(self.policy.get_action(new_obs)[0])
            q_values = np.zeros((len(self.ensemble_qs), N))
            for i, q in enumerate(self.ensemble_qs):
                q_values[i, :] = np_ify(q(torch_ify(new_obs), actions)).flatten()
            q_std = q_values.std(axis=0)
            p = softmax(q_std / exploration_temperature)
            ind = np.random.choice(np.arange(N), p=p)
            self._rollout_goal = {}
            self._rollout_goal[self.desired_goal_key] = proposed_goals[ind, :]
        elif rgp["strategy"] == "vae_q":
            pass
        else:
            assert False, "bad rollout goal strategy"

        return o

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
        observation[self.desired_goal_key] = self._rollout_goal
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )

    def _handle_path(self, path):
        self._n_rollouts_total += 1
        self.replay_buffer.add_path(path)
        self._exploration_paths.append(path)

    def get_batch(self):
        batch = super().get_batch()
        obs = batch['observations']
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']
        batch['observations'] = torch.cat((
            obs,
            goals
        ), dim=1)
        batch['next_observations'] = torch.cat((
            next_obs,
            goals
        ), dim=1)
        return batch

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        new_obs = np.hstack((
            observation[self.observation_key],
            self._rollout_goal,
        ))
        return self.exploration_policy.get_action(new_obs)

    def get_eval_paths(self):
        paths = []
        n_steps_total = 0
        while n_steps_total <= self.num_steps_per_eval:
            path = self.eval_multitask_rollout()
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths

    def eval_multitask_rollout(self):
        return self.eval_rollout_function(
            self.env,
            self.policy,
            self.max_path_length,
            animated=self.render_during_eval
        )

