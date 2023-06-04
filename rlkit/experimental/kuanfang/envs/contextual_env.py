import abc
from collections import OrderedDict

import gym
import gym.spaces
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Union, Callable, Any, Dict, List

from rlkit.core.distribution import DictDistribution
from rlkit.torch import pytorch_util as ptu
# from rlkit.util.io import load_local_or_remote_file
from rlkit import pythonplusplus as ppp

from rlkit.experimental.kuanfang.vae import vqvae
from rlkit.experimental.kuanfang.planning.rb_mppi_planner import RbMppiPlanner  # NOQA


Path = Dict
Diagnostics = Dict
Context = Any
ContextualDiagnosticsFn = Callable[
    [List[Path], List[Context]],
    Diagnostics,
]


def batchify(x):
    return ppp.treemap(lambda x: x[None], x, atomic_type=np.ndarray)


def insert_reward(contexutal_env, info, obs, reward, done):
    info['ContextualEnv/old_reward'] = reward
    return info


def delete_info(contexutal_env, info, obs, reward, done):
    return {}


def maybe_flatten_obs(self, obs):
    if len(obs.shape) == 1:
        return obs.reshape(1, -1)
    return obs


class ContextualRewardFn(object, metaclass=abc.ABCMeta):
    """You can also just pass in a function."""

    @abc.abstractmethod
    def __call__(
            self,
            states: dict,
            actions,
            next_states: dict,
            contexts: dict
    ):
        pass


class UnbatchRewardFn(object):
    def __init__(self, reward_fn: ContextualRewardFn):
        self._reward_fn = reward_fn

    def __call__(
            self,
            state: dict,
            action,
            next_state: dict,
            context: dict
    ):
        states = batchify(state)
        actions = batchify(action)
        next_states = batchify(next_state)
        reward, terminal = self._reward_fn(
            states,
            actions,
            next_states,
            context,
            # debug=True,
        )
        return reward[0]


class ContextualEnv(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env,
            context_distribution: DictDistribution,
            reward_fn: ContextualRewardFn,
            observation_key=None,  # for backwards compatibility
            observation_keys=None,
            update_env_info_fn=None,
            contextual_diagnostics_fns: Union[
                None, List[ContextualDiagnosticsFn]] = None,
            unbatched_reward_fn=None,
    ):
        super().__init__(env)

        if contextual_diagnostics_fns is None:
            contextual_diagnostics_fns = []

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("ContextualEnvs require wrapping Dict spaces.")

        spaces = env.observation_space.spaces
        for key, space in context_distribution.spaces.items():
            spaces[key] = space

        self.observation_space = gym.spaces.Dict(spaces)
        self.reward_fn = reward_fn
        self._last_obs = None
        self._update_env_info = update_env_info_fn or insert_reward

        self._curr_context = None

        self._observation_key = observation_key
        del observation_keys

        self._context_distribution = context_distribution
        self._context_keys = list(context_distribution.spaces.keys())

        self._contextual_diagnostics_fns = contextual_diagnostics_fns

        if unbatched_reward_fn is None:
            unbatched_reward_fn = UnbatchRewardFn(reward_fn)

        self.unbatched_reward_fn = unbatched_reward_fn

    def reset(self):
        obs = self.env.reset()
        self._curr_context = self._context_distribution(
            context=obs).sample(1)
        self._add_context_to_obs(obs)
        self._last_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._add_context_to_obs(obs)
        new_reward = self._compute_reward(self._last_obs, action, obs, reward)
        self._last_obs = obs
        info = self._update_env_info(self, info, obs, reward, done)
        return obs, new_reward, done, info

    def _compute_reward(self, state, action, next_state, env_reward=None):
        """Do reshaping for reward_fn, which is implemented for batches."""
        if not self.reward_fn:
            return env_reward
        else:
            return self.unbatched_reward_fn(
                state, action, next_state, self._curr_context)

    def _add_context_to_obs(self, obs):
        for key in self._context_keys:
            obs[key] = self._curr_context[key][0]

    def get_diagnostics(self, paths):
        stats = OrderedDict()
        contexts = [self._get_context(p) for p in paths]
        for fn in self._contextual_diagnostics_fns:
            stats.update(fn(paths, contexts))
        return stats

    def _get_context(self, path):
        first_observation = path['observations'][0]
        return {
            key: first_observation[key] for key in self._context_keys
        }


class SubgoalContextualEnv(ContextualEnv):

    def __init__(
        self,
        env: gym.Env,
        context_distribution: DictDistribution,
        reward_fn: ContextualRewardFn,
        observation_key=None,  # for backwards compatibility
        observation_keys=None,
        goal_key=None,
        goal_key_reward_fn=None,
        use_encoding=None,
        update_env_info_fn=None,
        contextual_diagnostics_fns: Union[
            None, List[ContextualDiagnosticsFn]] = None,
        unbatched_reward_fn=None,
        # Planning.
        planner=None,
        num_planning_steps=None,
        fraction_planning=None,
        subgoal_timeout=None,
        subgoal_reaching_thresh=None,
        buffer_size=20,
        mode=None,
    ):

        super().__init__(
            env=env,
            context_distribution=context_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            observation_keys=observation_keys,
            update_env_info_fn=update_env_info_fn,
            contextual_diagnostics_fns=contextual_diagnostics_fns,
            unbatched_reward_fn=unbatched_reward_fn,
        )

        self._planner = planner
        self._goal_key = goal_key
        self._goal_key_reward_fn = goal_key_reward_fn
        self._use_encoding = use_encoding
        self._num_planning_steps = num_planning_steps
        self._num_subgoals = None
        self._fraction_planning = fraction_planning

        self._subgoal_timeout = subgoal_timeout
        self._subgoal_reaching_thresh = subgoal_reaching_thresh
        self._subgoal_timer = None
        self._subgoal_id = None

        # if self._goal_key == 'image_desired_goal':
        #     assert self._goal_key_reward_fn == 'latent_desired_goal'

        # self._context_keys = [goal_key]
        # assert len(self._context_keys) == 1, 'self._context_keys: %r' % (
        #     self._context_keys)

        # print('self._context_keys: ', self._context_keys)
        # print('self._goal_key: ', self._goal_key)
        # input()

        self._plan_th = None
        self._plan_np = None
        self._latent_plan_np = None

        self._vqvae = planner.vqvae
        self._obs_encoder = self._vqvae
        self._vf = None

        self._subgoals_reached_at = [[]] * self._num_planning_steps
        self._buffer_size = buffer_size
        self._num_episodes = 0

        self._episode_reward = 0.0

    @property
    def planner(self):
        return self._planner

    def set_vf(self, value):
        self._vf = value
        self._planner.vf = value

    def set_model(self, model):
        self._planner.affordance = model['affordance']
        self._planner.vf = model['vf']
        self._planner.qf1 = model['qf1']
        self._planner.qf2 = model['qf2']
        if 'obs_encoder' in model:
            self._obs_encoder = model['obs_encoder']
            self._planner.encoding_type = 'vib'
        else:
            self._planner.encoding_type = 'vqvae'

    def reset(self):
        obs = self.env.reset()

        # Sample the context (initial and goal states)
        self._targ_context = self._context_distribution(
            context=obs).sample(1)

        # Plan and switch to the first subgoal.
        curr_state = self._get_curr_state(obs)
        goal_state = self._get_goal_state(obs)
        self._plan(curr_state, goal_state)
        self._maybe_get_next_subgoal(obs, must=True)

        # Update obs.
        self._add_context_to_obs(obs)
        self._last_obs = obs

        print('Reached subgoals:')
        for subgoal_id in range(self._num_planning_steps):
            self._subgoals_reached_at[subgoal_id] = (
                self._subgoals_reached_at[subgoal_id][
                    -self._buffer_size:])
            count = sum(self._subgoals_reached_at[subgoal_id])
            buffer_size = len(self._subgoals_reached_at[subgoal_id])
            rate = float(count) / float(buffer_size + 1e-8)
            print('id %d: %d (%.3f)'
                  % (subgoal_id, count, rate))

        print('Episode %d, Reward: %.2f'
              % (self._num_episodes, self._episode_reward))
        self._num_episodes += 1
        self._num_steps = 0
        self._episode_reward = 0.0

        if isinstance(self._planner, RbMppiPlanner):
            print('RbMppiPlanner num_candidates: %d' %
                  (self._planner._num_candidates))

        return obs

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        new_reward = self._compute_reward(self._last_obs, action, obs, reward)
        # print('t: %d, reward: %.2f' % (self._num_steps, new_reward))
        info = self._update_env_info(self, info, obs, reward, done)

        # Update obs.
        self._add_context_to_obs(obs)
        self._last_obs = obs

        # Maybe switch to the next subgoal.
        self._subgoal_timer += 1
        self._maybe_get_next_subgoal(obs, reward=new_reward)

        curr_state = self._get_curr_state(obs)
        self._planner.update(curr_state)

        self._num_steps += 1
        self._episode_reward += new_reward

        return obs, new_reward, done, info

    def _get_curr_state(self, obs):
        curr_state = obs[self._observation_key]
        if self._use_encoding:
            curr_state = self._obs_encoder.encode_one_np(curr_state)
        return curr_state

    def _get_goal_state(self, obs):
        goal_state = self._targ_context[self._goal_key][0]
        if self._use_encoding:
            goal_state = self._obs_encoder.encode_one_np(goal_state)
        return goal_state

    def _plan(self, curr_state, goal_state):
        if np.random.rand() > self._fraction_planning:
            plan_np = np.reshape(
                goal_state,
                [1, -1])
            plan_np = np.stack([plan_np] * self._num_planning_steps, 0)
            plan = ptu.from_numpy(plan_np)
            self._num_subgoals = 1

        else:
            curr_state = ptu.from_numpy(curr_state)
            goal_state = ptu.from_numpy(goal_state)

            plan, _ = self._planner(
                curr_state,
                goal_state,
                self._num_planning_steps,
            )
            plan = plan[:, None, ...]
            plan = torch.flatten(plan, start_dim=2)
            plan_np = ptu.get_numpy(plan)

        self._plan_np = plan_np

        latent_plan = self._compute_latent_plan(plan)
        if latent_plan is not None:
            self._latent_plan_np = ptu.get_numpy(latent_plan)

        self._subgoal_timer = 0
        self._num_subgoals = int(plan.shape[0])

    def _compute_latent_plan(self, plan):
        if (self._goal_key == 'image_desired_goal' and
                self.reward_fn.goal_key == 'latent_desired_goal'):
            inputs = plan[:, 0, ...]

            if isinstance(self._obs_encoder, vqvae.VqVae):
                inputs = inputs - 0.5
                inputs = inputs.view(
                    self._num_planning_steps,
                    self._obs_encoder.input_channels,
                    self._obs_encoder.imsize,
                    self._obs_encoder.imsize)
                inputs = inputs.permute(0, 1, 3, 2)
                latent_plan = self._obs_encoder.encode(inputs)
            else:
                latent_plan = self._obs_encoder(inputs)

            latent_plan = latent_plan[:, None, ...]
            return latent_plan
        else:
            return None

    def _maybe_get_next_subgoal(self, obs, must=False, reward=None):
        if must:
            assert self._num_subgoals > 0
            self._subgoal_id = 0
        else:
            curr_state = obs[self.reward_fn.obs_key]
            subgoal_state = self._curr_context[self.reward_fn.goal_key]

            if self._num_subgoals == 0:
                # No more remaining subgoals in the plan.
                return

            if self._subgoal_timer < self._subgoal_timeout:
                if not self._is_subgoal_reached(
                        obs[self.reward_fn.obs_key],
                        self._curr_context[self.reward_fn.goal_key]):
                    return
                else:
                    self._subgoals_reached_at[self._subgoal_id].append(1)
                    print('### reached subgoal %d' % (self._subgoal_id))
            else:
                # Time out for this subgoal.
                dist = np.linalg.norm(curr_state - subgoal_state)
                print('- dist from subgoal %d: %.3f'
                      % (self._subgoal_id, dist))
                self._subgoals_reached_at[self._subgoal_id].append(0)

            self._subgoal_id += 1

        # Swith to the next subgoal.
        self._curr_context = {}
        for key in self._context_keys:
            if key == self._goal_key:
                self._curr_context[key] = self._plan_np[0]
                self._plan_np = self._plan_np[1:]
            elif key == self.reward_fn.goal_key and key != self._goal_key:
                self._curr_context[key] = self._latent_plan_np[0]
                self._latent_plan_np = self._latent_plan_np[1:]
            else:
                self._curr_context[key] = (
                    self._targ_context[key])

        self._num_subgoals -= 1
        self._subgoal_timer = 0

    def _is_subgoal_reached(self, curr_state, goal_state):
        if self._subgoal_reaching_thresh is None:
            return False

        if self._subgoal_reaching_thresh < 0:
            return False

        dist = np.linalg.norm(goal_state - curr_state)

        return dist < self._subgoal_reaching_thresh

    def _plot_state(self, curr_state, subgoal_state, reward):
        dist = np.linalg.norm(curr_state - subgoal_state)
        print('s: ', curr_state.shape, curr_state.mean())
        print('c: ', subgoal_state.shape, subgoal_state.mean())
        print('dist: ', dist, ' epsl: ', self._subgoal_reaching_thresh)
        print('reward: ', reward)

        if self.reward_fn.obs_type == 'latent':
            s_t = self._vqvae.decode_one_np(curr_state)
            g_t = self._vqvae.decode_one_np(subgoal_state)
        else:
            s_t = curr_state
            g_t = subgoal_state

        plt.figure()
        plt.subplot(1, 2, 1)
        image = np.reshape(s_t, [3, 48, 48])
        image = np.transpose(image, (2, 1, 0))
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        image = np.reshape(g_t, [3, 48, 48])
        image = np.transpose(image, (2, 1, 0))
        plt.imshow(image)

        plt.show()

    def _debug_plan(self, h_0, h_g, plan):
        plan = [plan[i] for i in range(plan.shape[0])]

        s = []

        if self._use_encoding:
            h_0 = np.reshape(
                h_0,
                [-1,
                 self._vqvae.embedding_dim *
                 self._vqvae.root_len *
                 self._vqvae.root_len])

            for h_t in [h_0] + plan + [h_g]:
                s_t = self._vqvae.decode_one_np(h_t)
                s.append(s_t)

        else:
            for h_t in [h_0] + plan + [h_g]:
                if not isinstance(h_t, np.ndarray):
                    h_t = ptu.get_numpy(h_t)
                s_t = np.reshape(
                    h_t,
                    [self._vqvae.input_channels,
                     self._vqvae.imsize,
                     self._vqvae.imsize])
                s.append(s_t)

        horizon = len(s)

        plt.figure()
        for t in range(horizon):
            plt.subplot(1, horizon, t + 1)
            image = s[t]
            print(image.shape)
            image = np.transpose(image, (2, 1, 0))
            plt.title('t%02d' % (t))
            plt.imshow(image)

        plt.show()


class RoundTripSubgoalContextualEnv(SubgoalContextualEnv):

    def __init__(
        self,
        env: gym.Env,
        context_distribution: DictDistribution,
        reward_fn: ContextualRewardFn,
        observation_key=None,  # for backwards compatibility
        observation_keys=None,
        update_env_info_fn=None,
        contextual_diagnostics_fns: Union[
            None, List[ContextualDiagnosticsFn]] = None,
        unbatched_reward_fn=None,
        # Planning.
        planner=None,
        num_planning_steps=None,
        fraction_planning=None,
        init_key=None,
        goal_key=None,
        goal_key_reward_fn=None,
        use_encoding=None,
        subgoal_timeout=None,
        subgoal_reaching_thresh=None,
        # Reset-free.
        reset_interval=None,
    ):

        super().__init__(
            env=env,
            context_distribution=context_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            observation_keys=observation_keys,
            update_env_info_fn=update_env_info_fn,
            contextual_diagnostics_fns=contextual_diagnostics_fns,
            unbatched_reward_fn=unbatched_reward_fn,
            planner=planner,
            num_planning_steps=num_planning_steps,
            fraction_planning=fraction_planning,
            goal_key=goal_key,
            goal_key_reward_fn=goal_key_reward_fn,
            subgoal_timeout=subgoal_timeout,
            subgoal_reaching_thresh=subgoal_reaching_thresh,
        )

        self._init_key = init_key

        self._reset_interval = reset_interval
        self._reset_counter = None

    def reset(self):
        if (self._reset_counter is None or
                self._reset_interval == self._reset_counter - 1):
            obs = self.env.reset()
            self._reset_counter = 0
        else:
            obs = self.env.reset_gripper()
            obs = self.env.get_observation()
            self._reset_counter += 1

        self._targ_context = self._context_distribution(
            context=obs).sample(1)

        # Plan and switch to the first subgoal.
        curr_state = self._get_curr_state(obs)
        goal_state = self._get_goal_state(obs)
        self._plan(curr_state, goal_state)
        self._maybe_get_next_subgoal(curr_state, must=True)

        # Update obs.
        self._add_context_to_obs(obs)
        self._last_obs = obs

        return obs

    def _get_goal_state(self, obs):
        # Periodically plan towards the goal state and the initial state.
        if self._reset_counter % 2 == 0:
            goal_key = self._goal_key
        else:
            goal_key = self._init_key

        goal_state = self._targ_context[goal_key][0]

        if self._use_encoding:
            goal_state = self._obs_encoder.encode_one_np(goal_state)

        return goal_state

    def end_epoch(self):
        self._reset_counter = None


class NonEpisodicSubgoalContextualEnv(SubgoalContextualEnv):

    def __init__(
        self,
        env: gym.Env,
        context_distribution: DictDistribution,
        reward_fn: ContextualRewardFn,
        observation_key=None,  # for backwards compatibility
        observation_keys=None,
        init_key=None,
        goal_key=None,
        goal_key_reward_fn=None,
        use_encoding=None,
        update_env_info_fn=None,
        contextual_diagnostics_fns: Union[
            None, List[ContextualDiagnosticsFn]] = None,
        unbatched_reward_fn=None,
        # Planning.
        planner=None,
        num_planning_steps=None,
        fraction_planning=None,
        subgoal_timeout=None,
        subgoal_reaching_thresh=None,
        # Reset-free.
        reset_interval=None,
        mode='d',
    ):

        super().__init__(
            env=env,
            context_distribution=context_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            observation_keys=observation_keys,
            goal_key=goal_key,
            goal_key_reward_fn=goal_key_reward_fn,
            use_encoding=use_encoding,
            update_env_info_fn=update_env_info_fn,
            contextual_diagnostics_fns=contextual_diagnostics_fns,
            unbatched_reward_fn=unbatched_reward_fn,
            planner=planner,
            num_planning_steps=num_planning_steps,
            fraction_planning=fraction_planning,
            subgoal_timeout=subgoal_timeout,
            subgoal_reaching_thresh=subgoal_reaching_thresh,
        )

        self._init_key = init_key
        self._mode = mode

        self._reset_interval = reset_interval
        self._reset_counter = None

        self._vf = None

    def _get_init_state(self):
        init_key = self._init_key
        init_state = self._targ_context[init_key][0]
        if self._use_encoding:
            init_state = self._obs_encoder.encode_one_np(init_state)
        return init_state

    def _get_goal_state(self):
        goal_key = self._goal_key
        goal_state = self._targ_context[goal_key][0]
        if self._use_encoding:
            goal_state = self._obs_encoder.encode_one_np(goal_state)
        return goal_state

    def reset(self):
        if (self._reset_counter is None or
                self._reset_counter == self._reset_interval - 1):
            obs = self.env.reset()
            self._reset_counter = 0
        else:
            self.env.reset_gripper()
            obs = self.env.get_observation()
            self._reset_counter += 1

        self._targ_context = self._context_distribution(
            context=obs).sample(1)

        # Plan and switch to the first subgoal.
        curr_state = self._get_curr_state(obs)
        init_state = self._get_init_state()
        goal_state = self._get_goal_state()
        self._plan(curr_state, init_state, goal_state)

        self._maybe_switch_subgoal(curr_state, reset=True)

        # Update obs.
        self._add_context_to_obs(obs)
        self._last_obs = obs

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        new_reward = self._compute_reward(self._last_obs, action, obs, reward)
        info = self._update_env_info(self, info, obs, reward, done)

        # Update obs.
        self._add_context_to_obs(obs)
        self._last_obs = obs

        # Maybe switch to the next subgoal.
        self._subgoal_timer += 1
        curr_state = self._get_curr_state(obs)
        self._maybe_switch_subgoal(curr_state)

        return obs, new_reward, done, info

    def _plan(self, curr_state, init_state, goal_state):
        curr_state = ptu.from_numpy(curr_state)
        init_state = ptu.from_numpy(init_state)
        goal_state = ptu.from_numpy(goal_state)

        if self._mode in ['r']:
            plan, info = self._planner(
                curr_state,
                goal_state,
                self._num_planning_steps,
            )
            plan = torch.flatten(plan, start_dim=1)

            print('plan.shape: ', plan.shape)
            if 'top_cost' in info:
                print('top_cost: ', ptu.get_numpy(info['top_cost']))

        else:
            forward_subgoals, _ = self._planner(
                init_state,
                goal_state,
                self._num_planning_steps,
            )
            print('forward_subgoals.shape: ', forward_subgoals.shape)
            forward_subgoals = torch.flatten(forward_subgoals, start_dim=1)

            backward_subgoals, _ = self._planner(
                goal_state,
                init_state,
                self._num_planning_steps,
            )
            backward_subgoals = torch.flatten(backward_subgoals, start_dim=1)

            plan = torch.cat([forward_subgoals, backward_subgoals], 0)
            print('forward_subgoals: ', forward_subgoals.shape)
            print('backward_subgoals: ', backward_subgoals.shape)
            print('plan: ', plan.shape)

        print('plan: ', plan.shape)
        self._plan_th = plan
        self._plan_np = ptu.get_numpy(plan)

        latent_plan = self._compute_latent_plan(plan)
        if latent_plan is not None:
            self._latent_plan_np = ptu.get_numpy(latent_plan)

        self._subgoal_id = -1
        self._subgoal_timer = 0
        self._num_subgoals = int(plan.shape[0])

    def _maybe_switch_subgoal(self, curr_state, reset=False):
        if reset:
            assert self._num_subgoals > 0
        else:
            if self._subgoal_timer < self._subgoal_timeout:
                # Time out for this subgoal.
                return

        # Start with the initial state, and keep iterating.
        if self._mode == 'o':
            if reset:
                self._subgoal_id = 0
            else:
                # self._subgoal_id = (
                #     self._subgoal_id + 1) % self._num_subgoals
                self._subgoal_id = min(self._subgoal_id + 1,
                                       self._num_subgoals - 1)

        # Start with the most reachable subgoal, then keep iterating.
        elif self._mode == 'a':
            if reset:
                self._subgoal_id = self._get_most_reachable_subgoal(
                    curr_state, self._plan_th)
            else:
                # self._subgoal_id = (
                #     self._subgoal_id + 1) % self._num_subgoals
                self._subgoal_id = min(self._subgoal_id + 1,
                                       self._num_subgoals - 1)

        # Periodically restart iterating from the most reachable subgoal.
        elif self._mode == 'b':
            if reset or self._reset_timer >= 3:
                self._subgoal_id = self._get_most_reachable_subgoal(
                    curr_state, self._plan_th)
                self._reset_timer = 0
            else:
                self._subgoal_id = (self._subgoal_id + 1) % self._num_subgoals
                self._reset_timer += 1

        # Keep aiming for the most reachable subgoal.
        elif self._mode == 'c':
            self._subgoal_id = self._get_most_reachable_subgoal(
                curr_state, self._plan_th)

        # Uniformly sample a subgoal from the affordance model.
        elif self._mode == 'r':
            init_state = self._get_init_state()
            goal_state = self._get_goal_state()
            self._plan(curr_state, init_state, goal_state)
            self._subgoal_id = 0

        # Shuffle the order.
        elif self._mode == 's':
            init_state = self._get_init_state()
            goal_state = self._get_goal_state()
            self._plan(curr_state, init_state, goal_state)
            self._subgoal_id = np.random.choice(self._num_subgoals)

        else:
            raise ValueError('Unrecognized mode: %r' % (self._mode))

        print('self._subgoal_id: ', self._subgoal_id)

        self._curr_context = self._get_subgoal_context(self._subgoal_id)

        self._subgoal_timer = 0

    def _get_most_reachable_subgoal(self, curr_state, subgoal_states):
        curr_state = ptu.from_numpy(curr_state)
        curr_states = curr_state[None, ...].repeat(self._num_subgoals, 1)

        vf_inputs = torch.cat([
            curr_states.view(-1, 720),
            subgoal_states.view(-1, 720),
        ], 1)
        values = self._vf(vf_inputs).detach()

        subgoal_id = torch.argmax(values).item()

        return subgoal_id

    def _get_subgoal_context(self, subgoal_id):
        # Swith to the next subgoal.
        context = {}
        for key in self._context_keys:
            if key == self._goal_key:
                context[key] = self._plan_np[subgoal_id][None, ...]
            elif key == self.reward_fn.goal_key and key != self._goal_key:
                context[key] = self._latent_plan_np[subgoal_id][None, ...]
            else:
                context[key] = self._targ_context[key]

        return context

    def end_epoch(self):
        self._reset_counter = None
