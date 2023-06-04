"""
Policies to be used with a state-distance Q function.
"""
import abc
from itertools import product

import numpy as np
from scipy import optimize
from torch import nn
from torch import optim

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.core import logger


class UniversalPolicy(ExplorationPolicy, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, observation, goal, tau, **kwargs):
        pass

    def reset(self):
        pass

    def get_param_values(self):
        return None

    def set_param_values(self, param_values):
        return


class SampleBasedUniversalPolicy(
    UniversalPolicy, ExplorationPolicy, metaclass=abc.ABCMeta
):
    def __init__(self, sample_size, env, sample_actions_from_grid=False):
        super().__init__()
        self.sample_size = sample_size
        self.env = env
        self.sample_actions_from_grid = sample_actions_from_grid
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self._goal_batch = None
        self._goal_batch_np = None
        self._tau_batch = None

    def set_goal(self, goal_np):
        super().set_goal(goal_np)
        self._goal_batch = self.expand_np_to_var(goal_np)
        self._goal_batch_np = np.repeat(
            np.expand_dims(goal_np, 0),
            self.sample_size,
            axis=0
        )

    def set_tau(self, tau):
        super().set_tau(tau)
        self._tau_batch = self.expand_np_to_var(np.array([tau]))

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.sample_size,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=False)

    def sample_actions(self):
        if self.sample_actions_from_grid:
            action_dim = self.env.action_dim
            resolution = int(np.power(self.sample_size, 1./action_dim))
            values = []
            for dim in range(action_dim):
                values.append(np.linspace(
                    self.action_low[dim],
                    self.action_high[dim],
                    num=resolution
                ))
            actions = np.array(list(product(*values)))
            if len(actions) < self.sample_size:
                # Add extra actions in case the grid can't perfectly create
                # `self.sample_size` actions. e.g. sample_size is 30, but the
                # grid is 5x5.
                actions = np.concatenate(
                    (
                        actions,
                        self.env.sample_actions(
                            self.sample_size - len(actions)
                        ),
                    ),
                    axis=0,
                )
            return actions
        else:
            return self.env.sample_actions(self.sample_size)

    def sample_states(self):
        return self.env.sample_states(self.sample_size)


class SamplePolicyPartialOptimizer(SampleBasedUniversalPolicy, nn.Module):
    """
    Greedy-action-partial-state implementation.

    Make it sublcass nn.Module so that calls to `train` and `cuda` get
    propagated to the sub-networks

    See https://paper.dropbox.com/doc/State-Distance-QF-Results-Summary-flRwbIxt0bbUbVXVdkKzr
    for details.
    """
    def __init__(self, qf, env, argmax_q, sample_size=100, **kwargs):
        nn.Module.__init__(self)
        super().__init__(sample_size, env, **kwargs)
        self.qf = qf
        self.argmax_q = argmax_q

    def get_action(self, obs):
        obs_pytorch = self.expand_np_to_var(obs)
        # sampled_actions = self.sample_actions()
        # actions = ptu.np_to_var(sampled_actions)
        goals = ptu.np_to_var(
            self.env.sample_irrelevant_goal_dimensions(
                self._goal_np, self.sample_size
            )
        )
        actions = self.argmax_q(
            obs_pytorch,
            goals,
            self._tau_batch,
        )

        q_values = ptu.get_numpy(self.qf(
            obs_pytorch,
            actions,
            goals,
            self.expand_np_to_var(np.array([self._tau_np])),
        ))
        max_i = np.argmax(q_values)
        # return sampled_actions[max_i], {}
        return ptu.get_numpy(actions[max_i]), {}


class SoftOcOneStepRewardPolicy(SampleBasedUniversalPolicy, nn.Module):
    """
    Optimize over goal state

        g* = \argmax_g R(g) + \lambda Q(s, \pi(s), g)
        a = \pi(s, g*)

    Do the argmax by sampling.

    Make it sublcass nn.Module so that calls to `train` and `cuda` get
    propagated to the sub-networks
    """
    def __init__(
            self,
            qf,
            env,
            policy,
            constraint_weight=1,
            sample_size=100,
            verbose=False,
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size, env, **kwargs)
        self.qf = qf
        self.policy = policy
        self.constraint_weight = constraint_weight
        self.verbose = verbose

    def reward(self, state, action, next_state):
        rewards_np = self.env.compute_rewards(
            None,
            None,
            ptu.get_numpy(next_state),
            ptu.get_numpy(self._goal_batch),
        )
        return ptu.np_to_var(rewards_np)

    def get_action(self, obs):
        goal_state_np = self._get_goal_state_np(obs)
        return self._get_np_action(obs, goal_state_np), {}

    def _get_goal_state_np(self, obs):
        sampled_goal_states_np = self.sample_states()
        sampled_goal_states = ptu.np_to_var(sampled_goal_states_np)
        obs = self.expand_np_to_var(obs)
        reward = self.reward(None, None, sampled_goal_states)
        constraint_reward = self.qf(
            obs,
            self.policy(obs, sampled_goal_states, self._tau_batch),
            self.env.convert_obs_to_goal_states_pytorch(sampled_goal_states),
            self._tau_batch,
        )
        if constraint_reward.size()[1] > 1:
            constraint_reward = constraint_reward.sum(dim=1, keepdim=True)
        if self.verbose:
            print("reward mean:", reward.mean())
            print("reward max:", reward.max())
            print("constraint reward mean:", constraint_reward.mean())
            print("constraint reward max:", constraint_reward.max())
        score = (
            reward
            + self.constraint_weight * constraint_reward
        )
        max_i = np.argmax(ptu.get_numpy(score))
        return sampled_goal_states_np[max_i]

    def _get_np_action(self, state_np, goal_state_np):
        return ptu.get_numpy(
            self.policy(
                ptu.np_to_var(np.expand_dims(state_np, 0)),
                ptu.np_to_var(np.expand_dims(goal_state_np, 0)),
                self._tau_expanded_torch,
            ).squeeze(0)
        )


class TerminalRewardSampleOCPolicy(SoftOcOneStepRewardPolicy, nn.Module):
    """
    Want to implement:

        a = \argmax_{a_T} \max_{a_{1:T-1}, s_{1:T+1}} r(s_{T+1})

        s.t.  Q(s_t, a_t, s_g=s_{t+1}, tau=0) = 0, t=1, T

    Softened version of this:

        a = \argmax_{a_T} \max_{a_{1:T-1}, s_{1:T+1}} r(s_{T+1})
         - C * \sum_{t=1}^T Q(s_t, a_t, s_g=s_{t+1}, tau=0)^2

          = \argmax_{a_T} \max_{a_{1:T-1}, s_{1:T+1}} f(a_{1:T}, s_{1:T+1})

    Naive implementation where I just sample a bunch of a's and s's and take
    the max of this function f.

    Make it sublcass nn.Module so that calls to `train` and `cuda` get
    propagated to the sub-networks

    :param obs: np.array, state/observation
    :return: np.array, action to take
    """
    def __init__(
            self,
            qf,
            env,
            horizon,
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(qf, env, **kwargs)
        self.horizon = horizon
        self._tau_batch = self.expand_np_to_var(np.array([0]))

    def get_action(self, obs):
        state = self.expand_np_to_var(obs)
        first_sampled_actions = self.sample_actions()
        action = ptu.np_to_var(first_sampled_actions)
        next_state = ptu.np_to_var(self.env.sample_states(self.sample_size))

        penalties = []
        for i in range(self.horizon):
            constraint_penalty = self.qf(
                state,
                action,
                self.env.convert_obs_to_goal_states_pytorch(next_state),
                self._tau_batch,
            )**2
            penalties.append(
                - self.constraint_weight * constraint_penalty
            )

            action = ptu.np_to_var(
                self.env.sample_actions(self.sample_size)
            )
            state = next_state
            next_state = ptu.np_to_var(self.env.sample_states(self.sample_size))
        reward = self.reward(state, action, next_state)
        final_score = reward + sum(penalties)
        max_i = np.argmax(ptu.get_numpy(final_score))
        return first_sampled_actions[max_i], {}


class ArgmaxQFPolicy(SampleBasedUniversalPolicy, nn.Module):
    """
    pi(s, g) = \argmax_a Q(s, a, g)

    Implemented by initializing a bunch of actions and doing gradient descent on
    them.

    This should be the same as a policy learned in DDPG.
    This is basically a sanity check.
    """
    def __init__(
            self,
            qf,
            env,
            policy,
            sample_size=100,
            learning_rate=1e-1,
            num_gradient_steps=10,
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size, env, **kwargs)
        self.qf = qf
        self.learning_rate = learning_rate
        self.num_gradient_steps = num_gradient_steps

    def get_action(self, obs):
        action_inits = self.sample_actions()
        actions = ptu.np_to_var(action_inits, requires_grad=True)
        obs = self.expand_np_to_var(obs)
        optimizer = optim.Adam([actions], self.learning_rate)
        losses = -self.qf(
            obs,
            actions,
            self._goal_batch,
            self._tau_batch,
        )
        for _ in range(self.num_gradient_steps):
            loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses = -self.qf(
                obs,
                actions,
                self._goal_batch,
                self._tau_batch,
            )
        losses_np = ptu.get_numpy(losses)
        best_action_i = np.argmin(losses_np)
        return ptu.get_numpy(actions[best_action_i, :]), {}


class PseudoModelBasedPolicy(SampleBasedUniversalPolicy, nn.Module):
    """
    1. Sample actions
    2. Optimize over next state (according to a Q function)
    3. Compare next state with desired next state to choose action
    """
    def __init__(
            self,
            qf,
            env,
            sample_size=100,
            learning_rate=1e-1,
            num_gradient_steps=100,
            state_optimizer='adam',
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size, env, **kwargs)
        self.qf = qf
        self.learning_rate = learning_rate
        self.num_optimization_steps = num_gradient_steps
        self.state_optimizer = state_optimizer
        self.observation_dim = self.env.observation_space.low.size

    def get_next_states_np(self, states, actions):
        if self.state_optimizer == 'adam':
            next_states_np = np.zeros((self.sample_size, self.observation_dim))
            next_states = ptu.np_to_var(next_states_np, requires_grad=True)
            optimizer = optim.Adam([next_states], self.learning_rate)

            for _ in range(self.num_optimization_steps):
                losses = -self.qf(
                    states,
                    actions,
                    next_states,
                    self._tau_batch,
                )
                loss = losses.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return ptu.get_numpy(next_states)
        elif self.state_optimizer == 'lbfgs':
            next_states = []
            for i in range(len(states)):
                state = states[i:i+1, :]
                action = actions[i:i+1, :]
                loss_f = self.create_loss(state, action, return_gradient=True)
                results = optimize.fmin_l_bfgs_b(
                    loss_f,
                    np.zeros((1, self.observation_dim)),
                    maxiter=self.num_optimization_steps,
                )
                next_state = results[0]
                next_states.append(next_state)
            next_states = np.array(next_states)
            return next_states
        elif self.state_optimizer == 'fmin':
            next_states = []
            for i in range(len(states)):
                state = states[i:i+1, :]
                action = actions[i:i+1, :]
                loss_f = self.create_loss(state, action)
                results = optimize.fmin(
                    loss_f,
                    np.zeros((1, self.observation_dim)),
                    maxiter=self.num_optimization_steps,
                )
                next_state = results[0]
                next_states.append(next_state)
            next_states = np.array(next_states)
            return next_states
        else:
            raise Exception(
                "Unknown state optimizer mode: {}".format(self.state_optimizer)
            )

    def create_loss(self, state, action, return_gradient=False):
        def f(next_state_np):
            next_state = ptu.np_to_var(
                np.expand_dims(next_state_np, 0),
                requires_grad=True,
            )
            loss = - self.qf(
                state,
                action,
                next_state,
                self._tau_expanded_torch
            )
            loss.backward()
            loss_np = ptu.get_numpy(loss)
            gradient_np = ptu.get_numpy(next_state.grad)
            if return_gradient:
                return loss_np, gradient_np.astype('double')
            else:
                return loss_np
        return f

    def get_action(self, obs):
        sampled_actions = self.sample_actions()
        states = self.expand_np_to_var(obs)
        actions = ptu.np_to_var(sampled_actions)
        next_states = self.get_next_states_np(states, actions)

        distances = np.sum(
            (next_states - self._goal_np)**2,
            axis=1
        )
        best_action = np.argmin(distances)
        return sampled_actions[best_action, :], {}


class StateOnlySdqBasedSqpOcPolicy(UniversalPolicy, nn.Module):
    """
    Implement

        pi(s_1, g) = pi_{distance}(s_1, s_2)

    where pi_{distance} is the SDQL policy and

        s_2 = argmin_{s_2} min_{s_{3:T+1}} ||s_{T+1} - g||_2^2
        subject to Q(s_i, pi_{distance}(s_i, s_{i+1}), s_{i+1}) = 0

    for i = 1, ..., T

    using SLSQP
    """
    def __init__(
            self,
            qf,
            env,
            policy,
            solver_params=None,
            planning_horizon=1,
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.qf = qf
        self.env = env
        self.policy = policy
        self.solver_params = solver_params
        self.planning_horizon = planning_horizon

        self.observation_dim = self.env.observation_space.low.size
        self.last_solution = None
        self.lower_bounds = np.hstack((
            np.tile(self.env.observation_space.low, self.planning_horizon),
        ))
        self.upper_bounds = np.hstack((
            np.tile(self.env.observation_space.high, self.planning_horizon),
        ))
        # TODO(vitchyr): figure out what to do if the state bounds are infinity
        self.lower_bounds = - np.ones_like(self.lower_bounds)
        self.upper_bounds = np.ones_like(self.upper_bounds)
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))
        self.constraints = {
            'type': 'eq',
            'fun': self.constraint_fctn,
            'jac': self.constraint_jacobian,
        }

    def split(self, x):
        """
        :param x: vector passed to optimization (np array or pytorch)
        :return: next_states shape [planning_horizon X obs_dim]
        """
        if isinstance(x, np.ndarray):
            return x.reshape(self.planning_horizon, self.observation_dim)
        else:
            return x.view(
                self.planning_horizon,
                self.observation_dim,
            )

    def cost_function(self, x):
        all_next_states = self.split(x)
        last_state = all_next_states[-1, :]
        return np.sum((last_state - self._goal_np)**2)

    def cost_jacobian(self, x):
        jacobian = np.zeros_like(x)
        all_next_states = self.split(x)
        last_state = all_next_states[-1, :]
        # Assuming the last `self.observation_dim` part of x is the last state
        jacobian[-self.observation_dim:] = (
            2 * (last_state - self._goal_np)
        )
        return jacobian

    def _constraint_fctn(self, x, state, return_grad):
        state = ptu.np_to_var(state)
        x = ptu.np_to_var(x, requires_grad=return_grad)
        all_next_states = self.split(x)

        loss = 0
        state = state.unsqueeze(0)
        for i in range(self.planning_horizon):
            next_state = all_next_states[i:i+1, :]
            action = self.policy(
                state, next_state, self._tau_expanded_torch
            )
            loss += self.qf(
                state, action, next_state, self._tau_expanded_torch
            )
            state = next_state
        if return_grad:
            loss.squeeze(0).backward()
            return ptu.get_numpy(x.grad)
        else:
            return ptu.get_numpy(loss.squeeze(0))[0]

    def constraint_fctn(self, x, state=None):
        return self._constraint_fctn(x, state, False)

    def constraint_jacobian(self, x, state=None):
        return self._constraint_fctn(x, state, True)

    def reset(self):
        self.last_solution = None

    def get_action(self, obs):
        if self.last_solution is None:
            self.last_solution = np.hstack((
                np.tile(obs, self.planning_horizon),
            ))
        self.constraints['args'] = (obs, )
        result = optimize.minimize(
            self.cost_function,
            self.last_solution,
            jac=self.cost_jacobian,
            constraints=self.constraints,
            method='SLSQP',
            options=self.solver_params,
            bounds=self.bounds,
        )
        next_goal_state = result.x[:self.observation_dim]
        action = self.get_np_action(obs, next_goal_state)
        if np.isnan(action).any():
            logger.log("WARNING: SLSQP returned nan. Adding noise to last "
                       "action")
            action = self.last_solution[:self.action_dim] + np.random.uniform(
                self.env.action_space.low,
                self.env.action_space.high,
            ) / 100
        else:
            self.last_solution = result.x
        return action, {}

    def get_np_action(self, state_np, goal_state_np):
        return ptu.get_numpy(
            self.policy(
                ptu.np_to_var(np.expand_dims(state_np, 0)),
                ptu.np_to_var(np.expand_dims(goal_state_np, 0)),
                self._tau_expanded_torch,
            ).squeeze(0)
        )


class UnconstrainedOcWithGoalConditionedModel(SampleBasedUniversalPolicy, nn.Module):
    """
    Make it sublcass nn.Module so that calls to `train` and `cuda` get
    propagated to the sub-networks
    """
    def __init__(
            self,
            goal_conditioned_model,
            env,
            argmax_q,
            sample_size=100,
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size, env, **kwargs)
        self.model = goal_conditioned_model
        self.argmax_q = argmax_q
        self.env = env

    def rewards_np(self, current_obs, states_predicted):
        return ptu.get_numpy(
            self.env.oc_reward(
                states_predicted,
                self._goal_batch,
                current_obs,
            )
        )

    def get_action(self, obs):
        obs_pytorch = self.expand_np_to_var(obs)
        sampled_goal_state = ptu.np_to_var(
            self.env.sample_dimensions_irrelevant_to_oc(
                self._goal_np, obs, self.sample_size
            )
        )
        actions = self.argmax_q(
            obs_pytorch,
            sampled_goal_state,
            self._tau_batch,
        )
        final_state_predicted = self.model(
            obs_pytorch,
            actions,
            sampled_goal_state,
            self._tau_batch,
        ) + obs_pytorch
        rewards = self.rewards_np(obs_pytorch, final_state_predicted)
        max_i = np.argmax(rewards)
        return ptu.get_numpy(actions[max_i]), {}


class UnconstrainedOcWithImplicitModel(SampleBasedUniversalPolicy, nn.Module):
    def __init__(
            self,
            implicit_model,
            env,
            argmax_q,
            sample_size=100,
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(sample_size, env, **kwargs)
        self.implicit_model = implicit_model
        self.argmax_q = argmax_q
        self.env = env

    def rewards_np(self, current_obs, goals):
        return ptu.get_numpy(
            self.env.oc_reward_on_goals(
                goals,
                self._goal_batch,
                current_obs,
            )
        )

    def get_action(self, obs):
        obs_pytorch = self.expand_np_to_var(obs)
        sampled_goal_state = ptu.np_to_var(
            self.env.sample_dimensions_irrelevant_to_oc(
                self._goal_np, obs, self.sample_size
            )
        )
        actions = self.argmax_q(
            obs_pytorch,
            sampled_goal_state,
            self._tau_batch,
        )
        # actions = self.env.sample_actions(self.sample_size)
        # actions = ptu.np_to_var(actions)
        # Implicit models only predict future goals
        final_goal_predicted = self.implicit_model(
            obs_pytorch,
            actions,
            sampled_goal_state,
            self._tau_batch,
            only_return_next_state=True,
        )
        rewards = self.rewards_np(
            obs_pytorch,
            final_goal_predicted
        )
        max_i = np.argmax(rewards)
        return ptu.get_numpy(actions[max_i]), {}
