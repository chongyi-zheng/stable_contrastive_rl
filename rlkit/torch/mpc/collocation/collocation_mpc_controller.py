import numpy as np
import time
import torch
from scipy import optimize
from torch import optim, nn as nn

from rlkit.core.eval_util import get_stat_in_paths, create_stats_ordered_dict
from rlkit.state_distance.policies import UniversalPolicy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, eval_np
import rlkit.core.logger as default_logger


class SlsqpCMC(UniversalPolicy, nn.Module):
    """
    CMC = Collocation MPC Controller

    Implement

        pi(s_1, g) = pi_{distance}(s_1, s_2)

    where pi_{distance} is the SDQL policy and

        s_2 = argmin_{s_2} min_{s_{3:T+1}} ||s_{T+1} - g||_2^2
        subject to C(s_i, pi_{distance}(s_i, s_{i+1}), s_{i+1}) = 0

    for i = 1, ..., T, where C is an implicit model.

    using SLSQP
    """
    def __init__(
            self,
            implicit_model,
            env,
            goal_slice,
            multitask_goal_slice,
            solver_kwargs=None,
            planning_horizon=1,
            use_implicit_model_gradient=False,
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.implicit_model = implicit_model
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.ao_dim = self.action_dim + self.obs_dim
        self.solver_kwargs = solver_kwargs
        self.use_implicit_model_gradient = use_implicit_model_gradient
        self.planning_horizon = planning_horizon

        self.last_solution = None
        self.lower_bounds = np.hstack((
            self.env.action_space.low,
            self.env.observation_space.low
        ))
        self.upper_bounds = np.hstack((
            self.env.action_space.high,
            self.env.observation_space.high
        ))
        self.lower_bounds = np.tile(self.lower_bounds, self.planning_horizon)
        self.upper_bounds = np.tile(self.upper_bounds, self.planning_horizon)
        # TODO(vitchyr): figure out what to do if the state bounds are infinity
        # self.lower_bounds = - np.ones_like(self.lower_bounds)
        # self.upper_bounds = np.ones_like(self.upper_bounds)
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))
        self.constraints = {
            'type': 'eq',
            'fun': self.constraint_fctn,
            'jac': self.constraint_jacobian,
        }

    def split(self, x):
        """
        split into action, next_state
        """
        actions_and_obs = []
        for h in range(self.planning_horizon):
            start_h = h * self.ao_dim
            actions_and_obs.append((
                x[start_h:start_h+self.action_dim],
                x[start_h+self.action_dim:start_h+self.ao_dim],
            ))
        return actions_and_obs

    def _cost_function(self, x, order):
        x = ptu.np_to_var(x, requires_grad=True)
        loss = 0
        for action, next_state in self.split(x):
            next_features_predicted = next_state[self.goal_slice]
            desired_features = ptu.np_to_var(
                self.env.multitask_goal[self.multitask_goal_slice]
                * np.ones(next_features_predicted.shape)
            )
            diff = next_features_predicted - desired_features
            loss += (diff**2).sum()
        if order == 0:
            return ptu.get_numpy(loss)[0]
        elif order == 1:
            loss.squeeze(0).backward()
            return ptu.get_numpy(x.grad)

    def cost_function(self, x):
        return self._cost_function(x, order=0)
        # action, next_state = self.split(x)
        # return self.env.cost_fn(None, action, next_state)

    def cost_jacobian(self, x):
        return self._cost_function(x, order=1)
        # jacobian = np.zeros_like(x)
        # _, next_state = self.split(x)
        # full_gradient = (
        #         2 * (self.env.convert_ob_to_goal(next_state) - self.env.multitask_goal)
        # )
        # jacobian[7:14] = full_gradient[:7]
        # return jacobian

    def _constraint_fctn(self, x, state, order):
        state = ptu.np_to_var(state)
        state = state.unsqueeze(0)
        x = ptu.np_to_var(x, requires_grad=order > 0)
        loss = 0
        for action, next_state in self.split(x):
            action = action[None]
            next_state = next_state[None]

            loss += self.implicit_model(state, action, next_state)
            state = next_state
        if order == 0:
            return ptu.get_numpy(loss.squeeze(0))[0]
        elif order == 1:
            loss.squeeze(0).backward()
            return ptu.get_numpy(x.grad)
        else:
            grad_kwargs = torch.autograd.grad(loss, x, create_graph=True)[0]
            grad_norm = torch.dot(grad_kwargs, grad_kwargs)
            grad_norm.backward()
            return ptu.get_numpy(x.grad)

    def constraint_fctn(self, x, state=None):
        if self.use_implicit_model_gradient:
            grad = self._constraint_fctn(x, state, 1)
            return np.inner(grad, grad)
        else:
            return self._constraint_fctn(x, state, 0)

    def constraint_jacobian(self, x, state=None):
        if self.use_implicit_model_gradient:
            return self._constraint_fctn(x, state, 2)
        else:
            return self._constraint_fctn(x, state, 1)

    def reset(self):
        self.last_solution = None

    def get_action(self, obs):
        if self.last_solution is None:
            init_solution = np.hstack((
                np.zeros(self.action_dim),
                obs,
            ))
            self.last_solution = np.tile(init_solution, self.planning_horizon)
        self.constraints['args'] = (obs, )
        result = optimize.minimize(
            self.cost_function,
            self.last_solution,
            jac=self.cost_jacobian,
            constraints=self.constraints,
            method='SLSQP',
            options=self.solver_kwargs,
            bounds=self.bounds,
        )
        if not result.success:
            print("WARNING: SLSQP Did not succeed. Message is:")
            print(result.message)

        action, _ = self.split(result.x)[0]
        self.last_solution = result.x
        return action, {}


class GradientCMC(UniversalPolicy, nn.Module):
    """
    CMC = Collocation MPC Controller

    Implement

        pi(s_1, g) = pi_{distance}(s_1, s_2)

    where pi_{distance} is the SDQL policy and

        s_2 = argmin_{s_2} min_{s_{3:T+1}} ||s_{T+1} - g||_2^2
        subject to C(s_i, pi_{distance}(s_i, s_{i+1}), s_{i+1}) = 0

    for i = 1, ..., T, where C is an implicit model.

    using gradient descent.

    Each element of "x" through the code represents the vector
    [a_1, s_1, a_2, s_2, ..., a_T, s_T]
    """
    def __init__(
            self,
            implicit_model,
            env,
            goal_slice,
            multitask_goal_slice,
            lagrange_multiplier=1,
            num_particles=1,
            num_grad_steps=10,
            learning_rate=1e-1,
            warm_start=False,
            planning_horizon=1,
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.implicit_model = implicit_model
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.ao_dim = self.action_dim + self.obs_dim
        self.lagrange_multiplier = lagrange_multiplier
        self.planning_horizon = planning_horizon
        self.num_particles = num_particles
        self.num_grad_steps = num_grad_steps
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        self.last_solution = None

    def split(self, x):
        """
        split into action, next_state
        """
        actions_and_obs = []
        for h in range(self.planning_horizon):
            start_h = h * self.ao_dim
            actions_and_obs.append((
                x[:, start_h:start_h+self.action_dim],
                x[:, start_h+self.action_dim:start_h+self.ao_dim],
            ))
        return actions_and_obs

    def _expand_np_to_var(self, array, requires_grad=False):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.num_particles,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=requires_grad)

    def cost_function(self, state, actions, next_states):
        """
        :param x: a PyTorch Variable.
        :return:
        """
        loss = 0
        for i in range(self.planning_horizon):
            slc = slice(i*self.obs_dim, (i+1)*self.obs_dim)
            next_state = next_states[:, slc]
            next_features_predicted = next_state[:, self.goal_slice]
            desired_features = ptu.np_to_var(
                self.env.multitask_goal[self.multitask_goal_slice][None]
                * np.ones(next_features_predicted.shape)
            )
            diff = next_features_predicted - desired_features
            loss += (diff**2).sum(dim=1, keepdim=True)
        return loss

    def constraint_fctn(self, state, actions, next_states):
        """
        :param x: a PyTorch Variable.
        :param state: a PyTorch Variable.
        :return:
        """
        loss = 0
        for i in range(self.planning_horizon):
            next_state = next_states[:, i*self.obs_dim:(i+1)*self.obs_dim]
            action = actions[:, i*self.action_dim:(i+1)*self.action_dim]

            loss -= self.implicit_model(state, action, next_state)
            state = next_state
        return loss

    def sample_actions(self):
        return np.random.uniform(
            self.action_low,
            self.action_high,
            (self.num_particles, self.action_dim)
        )

    def get_action(self, ob):
        if self.last_solution is None or not self.warm_start:
            init_solution = []
            for _ in range(self.planning_horizon):
                init_solution.append(self.sample_actions())
            for _ in range(self.planning_horizon):
                init_solution.append(
                    np.repeat(ob[None], self.num_particles, axis=0)
                )

            self.last_solution = np.hstack(init_solution)

        ob = self._expand_np_to_var(ob)
        x = ptu.np_to_var(self.last_solution, requires_grad=True)

        optimizer = optim.Adam([x], lr=self.learning_rate)
        loss = None
        for i in range(self.num_grad_steps):
            actions = x[:, :self.action_dim * self.planning_horizon]
            actions = torch.clamp(actions, -1, 1)
            next_states = x[:, self.action_dim * self.planning_horizon:]
            loss = (
                self.cost_function(ob, actions, next_states)
                + self.lagrange_multiplier *
                    self.constraint_fctn(ob, actions, next_states)
            )
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            x[:, :self.action_dim * self.planning_horizon].data = torch.clamp(
                x[:, :self.action_dim * self.planning_horizon].data, -1, 1
            )

        self.last_solution = ptu.get_numpy(x)
        if loss is None:
            actions = x[:, :self.action_dim * self.planning_horizon]
            actions = torch.clamp(actions, -1, 1)
            next_states = x[:, self.action_dim * self.planning_horizon:]
            loss = (
                    self.cost_function(ob, actions, next_states)
                    + self.lagrange_multiplier *
                    self.constraint_fctn(ob, actions, next_states)
            )
        loss_np = ptu.get_numpy(loss).sum(axis=1)
        min_i = np.argmin(loss_np)
        action = self.last_solution[min_i, :self.action_dim]
        action = np.clip(action, -1, 1)
        return action, {}


class StateGCMC(GradientCMC):
    """
    Use gradient-based optimization for choosing the next state, but using
    stochastic optimization for choosing the action.
    """
    def get_action(self, ob):
        if self.last_solution is None or not self.warm_start:
            init_solution = []
            for _ in range(self.planning_horizon):
                init_solution.append(
                    np.repeat(ob[None], self.num_particles, axis=0)
                )

            self.last_solution = np.hstack(init_solution)

        ob = self._expand_np_to_var(ob)
        actions_np = np.hstack(
            [self.sample_actions() for _ in range(self.planning_horizon)]
        )
        actions = ptu.np_to_var(actions_np)
        next_states = ptu.np_to_var(self.last_solution, requires_grad=True)

        optimizer = optim.Adam([next_states], lr=self.learning_rate)
        for i in range(self.num_grad_steps):
            constraint_loss = self.constraint_fctn(ob, actions, next_states)
            optimizer.zero_grad()
            constraint_loss.sum().backward()
            optimizer.step()

        final_loss = (
            self.cost_function(ob, actions, next_states)
            + self.lagrange_multiplier *
            self.constraint_fctn(ob, actions, next_states)
        )
        self.last_solution = ptu.get_numpy(next_states)
        final_loss_np = ptu.get_numpy(final_loss).sum(axis=1)
        min_i = np.argmin(final_loss_np)
        action = actions_np[min_i, :self.action_dim]
        return action, {}


class LBfgsBCMC(UniversalPolicy):
    """
    Solve

        min_{a_1:T, s_1:T} \sum_t c(s_t) - \lambda C(s_t, a_t, s_t+!)

    using L-BFGS-boxed where

        c(s_t) = ||s_t - goal||
        C is an implicit model (larger for more feasible transitiosn)

    """
    def __init__(
            self,
            implicit_model,
            env,
            goal_slice,
            multitask_goal_slice,
            planning_horizon=1,
            lagrange_multipler=1,
            warm_start=False,
            solver_kwargs=None,
            only_use_terminal_env_loss=False,
            replan_every_time_step=True,
            tdm_policy=None,
            dynamic_lm=False,
    ):
        super().__init__()
        if solver_kwargs is None:
            solver_kwargs = {}
        self.implicit_model = implicit_model
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.ao_dim = self.action_dim + self.obs_dim
        self.planning_horizon = planning_horizon
        self.lagrange_multipler = lagrange_multipler
        self.warm_start = warm_start
        self.solver_kwargs = solver_kwargs
        self.only_use_terminal_env_loss = only_use_terminal_env_loss
        self.replan_every_time_step = replan_every_time_step
        self.t_in_plan = 0
        self.tdm_policy = tdm_policy
        self.dynamic_lm = dynamic_lm
        self.min_lm = 0.1
        self.max_lm = 1000
        self.error_threshold = 0.5

        self.last_solution = None
        self.best_action_seq = None
        self.best_obs_seq = None
        self.desired_features_torch = None
        self.totals = []
        self.lower_bounds = np.hstack((
            self.env.action_space.low,
            self.env.observation_space.low
        ))
        self.upper_bounds = np.hstack((
            self.env.action_space.high,
            self.env.observation_space.high
        ))
        self.lower_bounds = np.tile(self.lower_bounds, self.planning_horizon)
        self.upper_bounds = np.tile(self.upper_bounds, self.planning_horizon)
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))
        self.forward = 0
        self.backward = 0

    def batchify(self, x, current_ob):
        """
        Convert
            [a1, s2, a2, s3, a3, s4]
        into
            [s1, s2, s3], [a1, a2, a3], [s2, s3, s4]
        """
        obs = []
        actions = []
        next_obs = []
        ob = current_ob
        for h in range(self.planning_horizon):
            start_h = h * self.ao_dim
            next_ob = x[start_h+self.action_dim:start_h+self.ao_dim]
            obs.append(ob)
            actions.append(x[start_h:start_h+self.action_dim])
            next_obs.append(next_ob)
            ob = next_ob
        return (
            torch.stack(obs),
            torch.stack(actions),
            torch.stack(next_obs),
        )

    def _env_cost_function(self, x, current_ob):
        _, _, next_obs = self.batchify(x, current_ob)
        next_features_predicted = next_obs[:, self.goal_slice]
        if self.only_use_terminal_env_loss:
            diff = (
                next_features_predicted[-1] - self.desired_features_torch[-1]
            )
            loss = (diff**2).sum()
        else:
            diff = next_features_predicted - self.desired_features_torch
            loss = (diff**2).sum()
        return loss

    def _feasibility_cost_function(self, x, current_ob):
        obs, actions, next_obs = self.batchify(x, current_ob)
        loss = -self.implicit_model(obs, actions, next_obs).sum()
        return loss

    def cost_function(self, x, current_ob):
        self.forward -= time.time()
        x = ptu.np_to_var(x, requires_grad=True)
        current_ob = ptu.np_to_var(current_ob)
        loss = (
                self.lagrange_multipler
                * self._feasibility_cost_function(x, current_ob)
                + self._env_cost_function(x, current_ob)
        )
        loss_np = ptu.get_numpy(loss)[0].astype(np.float64)
        self.forward += time.time()
        self.backward -= time.time()
        loss.squeeze(0).backward()
        gradient_np = ptu.get_numpy(x.grad).astype(np.float64)
        self.backward += time.time()
        return loss_np, gradient_np

    def reset(self):
        self.last_solution = None

    def get_action(self, current_ob):
        if (
                self.replan_every_time_step
                or self.t_in_plan == self.planning_horizon
                or self.last_solution is None
        ):
            if self.dynamic_lm and self.best_obs_seq is not None:
                error = np.linalg.norm(
                    current_ob - self.best_obs_seq[self.t_in_plan + 1]
                )
                self.update_lagrange_multiplier(error)
            goal = self.env.multitask_goal[self.multitask_goal_slice]
            full_solution = self.replan(current_ob, goal)

            x_torch = ptu.np_to_var(full_solution, requires_grad=True)
            current_ob_torch = ptu.np_to_var(current_ob)

            _, actions, next_obs = self.batchify(x_torch, current_ob_torch)
            self.best_action_seq = np.array([ptu.get_numpy(a) for a in actions])
            self.best_obs_seq = np.array(
                [current_ob] + [ptu.get_numpy(o) for o in next_obs]
            )

            self.last_solution = full_solution
            self.t_in_plan = 0

        tdm_actions = eval_np(
            self.tdm_policy,
            self.best_obs_seq[:-1],
            self.best_obs_seq[1:],
            np.zeros((self.planning_horizon, 1))
        )
        agent_info = dict(
            best_action_seq=self.best_action_seq[self.t_in_plan:],
            # best_action_seq=tdm_actions,
            best_obs_seq=self.best_obs_seq[self.t_in_plan:],
        )
        action = self.best_action_seq[self.t_in_plan]
        # action = tdm_actions[self.t_in_plan]
        self.t_in_plan += 1
        # print("action", action)
        # print("tdm_action", tdm_actions[0])

        return action, agent_info

    def replan(self, current_ob, goal):
        if self.last_solution is None or not self.warm_start:
            solution = []
            for i in range(self.planning_horizon):
                solution.append(self.env.action_space.sample())
                solution.append(current_ob)
            self.last_solution = np.hstack(solution)
        self.desired_features_torch = ptu.np_to_var(
            goal[None].repeat(self.planning_horizon, 0)
        )
        self.forward = self.backward = 0
        start = time.time()
        x, f, d = optimize.fmin_l_bfgs_b(
            self.cost_function,
            self.last_solution,
            args=(current_ob,),
            bounds=self.bounds,
            **self.solver_kwargs
        )
        total = time.time() - start
        self.totals.append(total)
        warnflag = d['warnflag']
        if warnflag != 0:
            if warnflag == 1:
                print("too many function evaluations or too many iterations")
            else:
                print(d['task'])
        return x

    def update_lagrange_multiplier(self, error):
        if error > self.error_threshold:
            self.lagrange_multipler *= 2
        else:
            self.lagrange_multipler *= 0.5
        self.lagrange_multipler = min(self.lagrange_multipler, self.max_lm)
        self.lagrange_multipler = max(self.lagrange_multipler, self.min_lm)


class TdmLBfgsBCMC(LBfgsBCMC):
    """
    Basically the same as LBfgsBCMC but use the goal passed into get_action

    TODO: maybe use num_steps_left to replace t_in_plan?
    """
    def get_action(self, current_ob, goal, num_steps_left):
        if (
                self.replan_every_time_step
                or self.t_in_plan == self.planning_horizon
                or self.last_solution is None
        ):
            if self.dynamic_lm and self.best_obs_seq is not None:
                error = np.linalg.norm(
                    current_ob - self.best_obs_seq[self.t_in_plan + 1]
                )
                self.update_lagrange_multiplier(error)
            full_solution = self.replan(current_ob, goal)

            x_torch = ptu.np_to_var(full_solution, requires_grad=True)
            current_ob_torch = ptu.np_to_var(current_ob)
            _, actions, next_obs = self.batchify(x_torch, current_ob_torch)
            self.best_action_seq = np.array([ptu.get_numpy(a) for a in actions])
            self.best_obs_seq = np.array(
                [current_ob] + [ptu.get_numpy(o) for o in next_obs]
            )

            self.last_solution = full_solution
            self.t_in_plan = 0

        agent_info = dict(
            best_action_seq=self.best_action_seq[self.t_in_plan:],
            best_obs_seq=self.best_obs_seq[self.t_in_plan:],
            lagrange_multiplier=self.lagrange_multipler,
        )
        action = self.best_action_seq[self.t_in_plan]
        self.t_in_plan += 1

        return action, agent_info

    def log_diagnostics(self, paths, logger=default_logger):
        lms = get_stat_in_paths(paths, 'agent_infos', 'lagrange_multiplier')
        for key, value in create_stats_ordered_dict(
            "TDM LBFGS Lagrange Multiplier",
            lms,
        ).items():
            logger.record_tabular(key, value)


class TdmToImplicitModel(PyTorchModule):
    def __init__(self, env, qf, tau):
        super().__init__()
        self.env = env
        self.qf = qf
        self.tau = tau

    def forward(self, states, actions, next_states):
        taus = ptu.np_to_var(
            self.tau * np.ones((states.shape[0], 1))
        )
        goals = self.env.convert_obs_to_goals(next_states)
        return self.qf(
            observations=states,
            actions=actions,
            goals=goals,
            num_steps_left=taus,
        ).sum(1)


class LBfgsBStateOnlyCMC(UniversalPolicy):
    """
    Solve

        min_{s_1:T} \sum_t c(s_t) - \lambda C(s_t, s_t+1)

    using L-BFGS-boxed where

        c(s_t) = ||s_t - goal||
        C is a state feasibility model (larger for more feasible transitions)
            In the code, we call it a value function (vf)

    The actions are selected according to

        a_t = \pi(s_t, s_t+1)

    where \pi is an inverse model
        In the code, we call this a TDM policy.
    """
    def __init__(
            self,
            vf,
            tdm_policy,
            env,
            goal_slice,
            multitask_goal_slice,
            planning_horizon=1,
            lagrange_multipler=1,
            warm_start=False,
            solver_kwargs=None,
            only_use_terminal_env_loss=False,
            replan_every_time_step=True,
            dynamic_lm=True,
    ):
        super().__init__()
        if solver_kwargs is None:
            solver_kwargs = {}
        self.vf = vf
        self.tdm_policy = tdm_policy
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.obs_dim = self.env.observation_space.low.size
        self.planning_horizon = planning_horizon
        self.lagrange_multipler = lagrange_multipler
        self.warm_start = warm_start
        self.solver_kwargs = solver_kwargs
        self.only_use_terminal_env_loss = only_use_terminal_env_loss
        self.replan_every_time_step = replan_every_time_step
        self.t_in_plan = 0
        self.dynamic_lm = dynamic_lm
        self.min_lm = 0.1
        self.max_lm = 1000
        self.error_threshold = 0.5

        self.num_steps_left_pytorch = ptu.np_to_var(
            np.arange(0, self.planning_horizon).reshape(
                self.planning_horizon, 1
            )
        )
        self.last_solution = None
        self.best_action_seq = None
        self.best_obs_seq = None
        self.desired_features_torch = None
        self.totals = []
        self.lower_bounds = self.env.observation_space.low
        self.upper_bounds = self.env.observation_space.high
        self.lower_bounds = np.tile(self.lower_bounds, self.planning_horizon)
        self.upper_bounds = np.tile(self.upper_bounds, self.planning_horizon)
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))
        self.forward = 0
        self.backward = 0

    def batchify(self, x, current_ob):
        """
        Convert
            [s2, s3, s4]
        into
            [s1, s2, s3], [s2, s3, s4]
        """
        obs = []
        next_obs = []
        ob = current_ob
        for h in range(self.planning_horizon):
            next_ob = x[h * self.obs_dim:(h+1) * self.obs_dim]
            obs.append(ob)
            next_obs.append(next_ob)
            ob = next_ob
        return (
            torch.stack(obs),
            torch.stack(next_obs),
        )

    def _env_cost_function(self, x, current_ob):
        _, next_obs = self.batchify(x, current_ob)
        next_features_predicted = next_obs[:, self.goal_slice]
        if self.only_use_terminal_env_loss:
            diff = (
                next_features_predicted[-1] - self.desired_features_torch[-1]
            )
            loss = (diff**2).sum()
        else:
            diff = next_features_predicted - self.desired_features_torch
            loss = (diff**2).sum()
        return loss

    def _feasibility_cost_function(self, x, current_ob):
        obs, next_obs = self.batchify(x, current_ob)
        loss = -self.vf(obs, next_obs, self.num_steps_left_pytorch).sum()
        return loss

    def cost_function(self, x, current_ob):
        self.forward -= time.time()
        x = ptu.np_to_var(x, requires_grad=True)
        current_ob = ptu.np_to_var(current_ob)
        loss = (
                self.lagrange_multipler
                * self._feasibility_cost_function(x, current_ob)
                + self._env_cost_function(x, current_ob)
        )
        loss_np = ptu.get_numpy(loss)[0].astype(np.float64)
        self.forward += time.time()
        self.backward -= time.time()
        loss.squeeze(0).backward()
        gradient_np = ptu.get_numpy(x.grad).astype(np.float64)
        self.backward += time.time()
        return loss_np, gradient_np

    def reset(self):
        self.last_solution = None
        self.best_obs_seq = None
        self.best_action_seq = None

    def get_action(self, current_ob):
        if (
                self.replan_every_time_step
                or self.t_in_plan == self.planning_horizon
                or self.last_solution is None
        ):
            if self.dynamic_lm and self.best_obs_seq is not None:
                error = np.linalg.norm(
                    current_ob - self.best_obs_seq[self.t_in_plan + 1]
                )
                self.update_lagrange_multiplier(error)

            goal = self.env.multitask_goal[self.multitask_goal_slice]
            full_solution = self.replan(current_ob, goal)

            x_torch = ptu.np_to_var(full_solution, requires_grad=True)
            current_ob_torch = ptu.np_to_var(current_ob)
            # feas_loss = self._feasibility_cost_function(
            #     x_torch, current_ob_torch
            # )
            # env_cos = self._env_cost_function(x_torch, current_ob_torch)
            # print("feasibility loss", ptu.get_numpy(feas_loss)[0])
            # print("weighted feasibility loss",
            #       self.lagrange_multipler * ptu.get_numpy(feas_loss)[0])
            # print("env loss", ptu.get_numpy(env_cos)[0])

            obs, next_obs = self.batchify(x_torch, current_ob_torch)
            actions = self.tdm_policy(
                observations=obs,
                goals=next_obs,
                num_steps_left=self.num_steps_left_pytorch,
            )
            self.best_action_seq = np.array([ptu.get_numpy(a) for a in actions])
            self.best_obs_seq = np.array(
                [current_ob] + [ptu.get_numpy(o) for o in next_obs]
            )

            self.last_solution = full_solution
            self.t_in_plan = 0

        agent_info = dict(
            best_action_seq=self.best_action_seq[self.t_in_plan:],
            best_obs_seq=self.best_obs_seq[self.t_in_plan:],
        )
        action = self.best_action_seq[self.t_in_plan]
        self.t_in_plan += 1

        return action, agent_info

    def replan(self, current_ob, goal):
        if self.last_solution is None or not self.warm_start:
            solution = []
            for i in range(self.planning_horizon):
                solution.append(current_ob)
            self.last_solution = np.hstack(solution)
        self.desired_features_torch = ptu.np_to_var(
            goal[None].repeat(self.planning_horizon, 0)
        )
        self.forward = self.backward = 0
        start = time.time()
        x, f, d = optimize.fmin_l_bfgs_b(
            self.cost_function,
            self.last_solution,
            args=(current_ob,),
            bounds=self.bounds,
            **self.solver_kwargs
        )
        total = time.time() - start
        self.totals.append(total)
        # print("total forward: {}".format(self.forward))
        # print("total backward: {}".format(self.backward))
        # print("total: {}".format(total))
        # print("extra: {}".format(total - self.forward - self.backward))
        # print("total mean: {}".format(np.mean(self.totals)))
        warnflag = d['warnflag']
        if warnflag != 0:
            if warnflag == 1:
                print("too many function evaluations or too many iterations")
            else:
                print(d['task'])
        return x

    def update_lagrange_multiplier(self, error):
        if error > self.error_threshold:
            self.lagrange_multipler *= 2
        else:
            self.lagrange_multipler *= 0.5
        self.lagrange_multipler = min(self.lagrange_multipler, self.max_lm)
        self.lagrange_multipler = max(self.lagrange_multipler, self.min_lm)


class TdmLBfgsBStateOnlyCMC(LBfgsBStateOnlyCMC):
    def get_action(self, current_ob, goal, num_steps_left):
        if (
                self.replan_every_time_step
                or self.t_in_plan == self.planning_horizon
                or self.last_solution is None
        ):
            if self.dynamic_lm and self.best_obs_seq is not None:
                error = np.linalg.norm(
                    current_ob - self.best_obs_seq[self.t_in_plan + 1]
                )
                self.update_lagrange_multiplier(error)

            full_solution = self.replan(current_ob, goal)

            x_torch = ptu.np_to_var(full_solution, requires_grad=True)
            current_ob_torch = ptu.np_to_var(current_ob)

            obs, next_obs = self.batchify(x_torch, current_ob_torch)
            actions = self.tdm_policy(
                observations=obs,
                goals=next_obs,
                num_steps_left=self.num_steps_left_pytorch,
            )
            self.best_action_seq = ptu.get_numpy(actions)
            self.best_obs_seq = np.array(
                [current_ob] + [ptu.get_numpy(o) for o in next_obs]
            )

            self.last_solution = full_solution
            self.t_in_plan = 0

        agent_info = dict(
            best_action_seq=self.best_action_seq[self.t_in_plan:],
            best_obs_seq=self.best_obs_seq[self.t_in_plan:],
        )
        action = self.best_action_seq[self.t_in_plan]
        self.t_in_plan += 1

        return action, agent_info
