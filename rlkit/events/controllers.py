import time

import numpy as np
import torch
from scipy import optimize

from rlkit.util.visualization_util import make_heat_map, plot_heatmap
from rlkit.state_distance.policies import UniversalPolicy
from rlkit.torch import pytorch_util as ptu
from torch.optim import Adam


def process_lbfgs_debug_dict(d):
    warnflag = d['warnflag']
    if warnflag != 0:
        if warnflag == 1:
            print("too many function evaluations or too many iterations")
        else:
            print(d['task'])



def fmin_adam_torch(
        batch_torch_f,
        x0_np,
        f_args=None,
        f_kwargs=None,
        lr=1e-3,
        num_steps=100,
):
    if f_args is None:
        f_args = tuple()
    if f_kwargs is None:
        f_kwargs = {}

    x = ptu.np_to_var(x0_np, requires_grad=True)
    optimizer = Adam([x], lr=lr)
    for _ in range(num_steps):
        loss = batch_torch_f(x, *f_args, **f_kwargs).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_values_np = ptu.get_numpy(batch_torch_f(x, *f_args, **f_kwargs))
    final_x_np = ptu.get_numpy(x)
    min_i = np.argmin(final_values_np)
    return final_x_np[min_i], final_values_np[min_i]


class BetaQLbfgsController(UniversalPolicy):
    # TODO: switch to using beta_v rather than beta_q
    """
    Solve

        min_{s_1:T} \sum_t c(s_t) beta(s_t | s_{t-1})

    using L-BFGS-boxed where

        c(s_t) = ||s_t - goal||
        beta(a, b) = prob(reach a | at state b)

    """
    def __init__(
            self,
            beta_q,
            env,
            goal_slice,
            multitask_goal_slice,
            max_cost,
            goal_reaching_policy=None,
            use_oracle_argmax_policy=False,
            oracle_argmax_policy=None,
            use_max_cost=True,
            planning_horizon=1,
            max_num_steps_to_reach_goal=1,
            warm_start=False,
            solver_kwargs=None,
            only_use_terminal_env_loss=False,
            replan_every_time_step=True,
            use_learned_policy=True,
    ):
        if only_use_terminal_env_loss:
            assert not replan_every_time_step, "Otherwise policy will procrastinate."
        super().__init__()
        if solver_kwargs is None:
            solver_kwargs = {}
        self.beta_q = beta_q
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.ao_dim = self.action_dim + self.obs_dim
        self.planning_horizon = planning_horizon
        self.warm_start = warm_start
        self.solver_kwargs = solver_kwargs
        self.only_use_terminal_env_loss = only_use_terminal_env_loss
        self.use_oracle_argmax_policy = use_oracle_argmax_policy
        self.replan_every_time_step = replan_every_time_step
        self.t_in_plan = 0
        self.learned_policy = goal_reaching_policy
        self.oracle_argmax_policy = oracle_argmax_policy
        self.num_steps_left = ptu.np_to_var(
            max_num_steps_to_reach_goal *
            np.ones((self.planning_horizon, 1))
        )
        self.use_max_cost = use_max_cost
        self.max_cost = max_cost
        self.use_learned_policy = use_learned_policy

        self.t = 0
        self._current_goal = None
        self._num_steps_to_reach_goal = max_num_steps_to_reach_goal
        self.max_num_steps_to_reach_goal = max_num_steps_to_reach_goal

        self.upper_tri = ptu.Variable(torch.triu(
            torch.ones(self.planning_horizon, self.planning_horizon),
            1,
        ))
        self.lower_and_diag_tri = 1 - self.upper_tri

        self.last_solution = None
        self.planned_action_seq = None
        self.subgoal_seq = None
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
        # import matplotlib.pyplot as plt
        # self.fig = plt.figure()

        x_values = np.linspace(-1, 1, num=10)
        y_values = np.linspace(-1, 1, num=10)
        x_values_all, y_values_all = np.meshgrid(x_values, y_values)
        x_values_flat = x_values_all.flatten()
        y_values_flat = y_values_all.flatten()
        self.all_actions = np.vstack((x_values_flat, y_values_flat)).T

    @property
    def current_goal(self):
        return self._current_goal

    @property
    def num_steps_to_reach_goal(self):
        return self._num_steps_to_reach_goal

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
            loss = (diff**2).sum(dim=1, keepdim=True)
        return loss

    def _feasibility_probabilities(self, x, current_ob):
        obs, actions, next_obs = self.batchify(x, current_ob)
        # TODO: use dynamics num_steps_left
        return self.beta_q(obs, actions, next_obs, self.num_steps_left)

    def cost_function(self, x, current_ob, verbose=False):
        self.forward -= time.time()
        x = ptu.np_to_var(x, requires_grad=True)
        current_ob = ptu.np_to_var(current_ob)
        env_costs = self._env_cost_function(x, current_ob)
        probabilities = self._feasibility_probabilities(x, current_ob)
        if self.use_max_cost:
            not_reached_cost = self.max_cost
        else:
            not_reached_cost = ((
                current_ob[self.goal_slice] - self.desired_features_torch
            )**2).sum()
        if verbose:
            print("---")
            print("env_costs", env_costs)
            print("not reached cost", not_reached_cost)
            print("probabilities", probabilities)
        if self.only_use_terminal_env_loss:
            final_prob = torch.prod(probabilities)
            loss = env_costs * (final_prob+1) + (1-final_prob) * not_reached_cost
            # if verbose:
            #     print("final prob", final_prob)
        else:
            """
            argmin_s c(s) p(s) + C_max (1-p(s))
             = argmin_s (c(s) - C_max) p(s)
             = argmin_s -log(C_max - c(s)) - log p(s)

            However, doing the cum-probs thing is better
            (i.e. it's a tighter lower bound)
            """
            # loss = -torch.log(
            #     self.planning_horizon * not_reached_cost - env_costs
            # ).sum() - traj_log_prob
            cum_probs = self._compute_cum_prob(probabilities)
            loss = env_costs * cum_probs + (1-cum_probs) * not_reached_cost
            # if verbose:
            #     print("cum_probs", cum_probs)
            loss = loss.sum()
        loss_np = ptu.get_numpy(loss)[0].astype(np.float64)
        self.forward += time.time()
        self.backward -= time.time()
        loss.backward()
        gradient_np = ptu.get_numpy(x.grad).astype(np.float64)
        self.backward += time.time()
        return loss_np, gradient_np

    def _compute_cum_prob(self, probabilities):
        """
        Convert
        [
            a
            b
            c
        ]
        into
        [
            a 0 0
            a b 0
            a b c
        ]
        and then into
        [
            a 1 1
            a b 1
            a b c
        ]
        then take the product across dim 1 to get
        [
            a
            a * b
            a * b * c
        ]
        """
        return (
                self.upper_tri + self.lower_and_diag_tri *
                probabilities.view(1, self.planning_horizon)
        ).prod(dim=1, keepdim=True)

    def reset(self):
        self.last_solution = None

    def get_action(self, current_ob):
        goal = self.env.multitask_goal[self.multitask_goal_slice]
        return self._get_action(current_ob, goal)

    def _get_action(self, current_ob, goal):
        if (
                self.replan_every_time_step
                or self.t_in_plan == self.planning_horizon
                or self.last_solution is None
        ):
            full_solution = self.replan(current_ob, goal)

            x_torch = ptu.np_to_var(full_solution, requires_grad=True)
            current_ob_torch = ptu.np_to_var(current_ob)

            _, actions, next_obs = self.batchify(x_torch, current_ob_torch)
            self.subgoal_seq = np.array(
                [current_ob] + [ptu.get_numpy(o) for o in next_obs]
            )
            self.learned_actions = self.learned_policy.eval_np(
                self.subgoal_seq[:-1],
                self.subgoal_seq[1:],
                np.zeros((self.planning_horizon, 1))
            )
            self.lbfgs_actions = np.array([ptu.get_numpy(a) for a in actions])
            if self.use_learned_policy:
                self.planned_action_seq = self.learned_actions
            else:
                self.planned_action_seq = self.lbfgs_actions

            self.last_solution = full_solution
            self.t_in_plan = 0

        action = self.planned_action_seq[self.t_in_plan]
        new_goal = self.subgoal_seq[self.t_in_plan+1]
        self._current_goal = new_goal
        oracle_qmax_action = self.get_oracle_qmax_action(current_ob,
                                                         new_goal)
        if self.use_oracle_argmax_policy:
            action = oracle_qmax_action

        # self.cost_function(full_solution, current_ob, verbose=True)
        # adam_action = self.choose_action_to_reach_adam(current_ob, new_goal)
        # lbfgs_action_again = self.choose_action_to_reach_lbfgs_again(
        #     current_ob, new_goal
        # )
        # lbfgs_action = self.lbfgs_actions[self.t_in_plan]
        # learned_action = self.learned_actions[self.t_in_plan]
        # print("---")
        # print("learned action", learned_action)
        # print("\terror: {}".format(np.linalg.norm(learned_action-oracle_qmax_ac)))tion
        # print("lbfgs action", lbfgs_action)
        # print("\terror: {}".format(np.linalg.norm(lbfgs_action-oracle_qmax_ac)))tion
        # print("lbfgs again action", lbfgs_action_again)
        # print("\terror: {}".format(np.linalg.norm(lbfgs_action_again-oracle_qmax_ac)))tion
        # print("adam_action", adam_action)
        # print("\terror: {}".format(np.linalg.norm(adam_action-oracle_qmax_ac)))tion
        # print("oracle best action", oracle_action)
        # print("action", action)
        agent_info = dict(
            planned_action_seq=self.planned_action_seq[self.t_in_plan:],
            subgoal_seq=self.subgoal_seq[self.t_in_plan:],
            oracle_qmax_action=oracle_qmax_action,
            learned_action=self.learned_actions[self.t_in_plan],
            lbfgs_action_seq=self.lbfgs_actions,
            learned_action_seq=self.learned_actions,
            full_action_seq=self.planned_action_seq,
            full_obs_seq=self.subgoal_seq,
        )

        self.t_in_plan += 1
        return action, agent_info

    def _action_cost(self, x, current_ob, goal):
        x = ptu.np_to_var(x, requires_grad=True)
        actions = x.unsqueeze(0)
        current_obs = ptu.np_to_var(current_ob[None])
        goals = ptu.np_to_var(goal[None])
        num_steps_left = ptu.np_to_var(np.zeros((1,1)))
        prob_reach = self.beta_q(current_obs, actions, goals, num_steps_left)
        loss = - prob_reach
        loss_np = ptu.get_numpy(prob_reach)[0].astype(np.float64)
        loss.backward()
        gradient_np = ptu.get_numpy(x.grad).astype(np.float64)
        return loss_np, gradient_np

    def choose_action_to_reach_lbfgs_again(self, current_ob, goal):
        init = self.env.action_space.sample()
        action_bounds = list(zip(
            self.env.action_space.low,
            self.env.action_space.high,
        ))
        x, f, d = optimize.fmin_l_bfgs_b(
            self._action_cost,
            init,
            args=(current_ob, goal),
            bounds=action_bounds,
            **self.solver_kwargs
        )
        return x

    def _action_cost_batch(self, actions, current_obs, goals, num_steps_left):
        return - self.beta_q(current_obs, actions, goals, num_steps_left)

    def choose_action_to_reach_adam(self, current_ob, goal):
        n_parts = 100
        x0 = np.vstack([
            self.env.action_space.sample()
            for _ in range(n_parts)
        ])
        current_obs = ptu.np_to_var(current_ob).unsqueeze(0).repeat(n_parts, 1)
        goals = ptu.np_to_var(goal).unsqueeze(0).repeat(n_parts, 1)
        num_steps_left = ptu.np_to_var(np.zeros((n_parts, 1)))
        best_action, _ = fmin_adam_torch(
            self._action_cost_batch,
            x0,
            f_args=(current_obs, goals, num_steps_left),
        )
        return best_action

    def get_oracle_qmax_action(self, current_ob, goal):
        if self.oracle_argmax_policy is None:
            obs = current_ob[None].repeat(100, 0)
            goals = goal[None].repeat(100, 0)
            num_steps_left = np.zeros((100, 1))
            beta_values = self.beta_q.eval_np(
                observations=obs,
                goals=goals,
                actions=self.all_actions,
                num_steps_left=num_steps_left,
            )
            max_i = np.argmax(beta_values)
            return self.all_actions[max_i]
        else:
            return self.oracle_argmax_policy.get_action(current_ob, goal, 0)[0]

    def replan(self, current_ob, goal):
        if self.last_solution is None or not self.warm_start:
            solution = []
            for i in range(self.planning_horizon):
                solution.append(np.zeros(self.action_dim))
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
        self.t1 = np.array([
            1, 0, 1, 0,
            1, 0, 2, 0,
            0, 1, 2, 1,
            0, 1, 2, 2,
        ])
        self.t2 = np.array([
            1, 0, 2, 0,
            0, 1, 2, 1,
            0, 1, 2, 2,
            -1, 1, 1, 3,
        ])
        total = time.time() - start
        self.totals.append(total)
        process_lbfgs_debug_dict(d)
        return x


class BetaQMultigoalLbfgs(BetaQLbfgsController):
    """
    Basically the same as LBfgsBCMC but use the goal passed into get_action

    TODO: maybe use num_steps_left to replace t_in_plan?
    """
    def get_action(self, current_ob, goal, num_steps_left):
        return self._get_action(current_ob, goal)

class BetaMultigoalLbfgs(BetaQMultigoalLbfgs):
    """Temporary for imports to work"""
    pass


class BetaVLbfgsController(UniversalPolicy):
    """
    Solve

        min_{s_1:T} \sum_t c(s_t) beta(s_t | s_{t-1})

    using L-BFGS-boxed where

        c(s_t) = ||s_t - goal||
        beta(a, b) = prob(reach a | at state b)

    Actions are taken by the goal reaching policy.
    """
    def __init__(
            self,
            beta_v,
            env,
            goal_slice,
            multitask_goal_slice,
            max_cost,
            goal_reaching_policy,
            use_oracle_argmax_policy=False,
            oracle_argmax_policy=None,
            use_max_cost=True,
            planning_horizon=1,
            max_num_steps_to_reach_goal=1,
            warm_start=False,
            solver_kwargs=None,
            only_use_terminal_env_loss=False,
            replan_every_time_step=True,
    ):
        if only_use_terminal_env_loss:
            assert not replan_every_time_step, "Otherwise policy will procrastinate."
        super().__init__()
        if solver_kwargs is None:
            solver_kwargs = {}
        self.beta_v = beta_v
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.max_cost = max_cost
        self.goal_reaching_policy = goal_reaching_policy
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.ao_dim = self.action_dim + self.obs_dim
        self.planning_horizon = planning_horizon
        self.warm_start = warm_start
        self.solver_kwargs = solver_kwargs
        self.only_use_terminal_env_loss = only_use_terminal_env_loss
        self.use_oracle_argmax_policy = use_oracle_argmax_policy
        self.replan_every_time_step = replan_every_time_step
        self.t_in_plan = 0
        self.oracle_argmax_policy = oracle_argmax_policy
        self.num_steps_left = ptu.np_to_var(
            max_num_steps_to_reach_goal *
            np.ones((self.planning_horizon, 1))
        )
        self.use_max_cost = use_max_cost

        self.t = 0
        self.current_goal = None
        self.num_steps_to_reach_goal = max_num_steps_to_reach_goal
        self.max_num_steps_to_reach_goal = max_num_steps_to_reach_goal

        self.upper_tri = ptu.Variable(torch.triu(
            torch.ones(self.planning_horizon, self.planning_horizon),
            1,
        ))
        self.lower_and_diag_tri = 1 - self.upper_tri

        self.last_solution = None
        self.planned_action_seq = None
        self.subgoal_seq = None
        self.desired_features_torch = None
        self.totals = []
        self.lower_bounds = np.hstack((
            self.env.observation_space.low
        ))
        self.upper_bounds = np.hstack((
            self.env.observation_space.high
        ))
        self.lower_bounds = np.tile(self.lower_bounds, self.planning_horizon)
        self.upper_bounds = np.tile(self.upper_bounds, self.planning_horizon)
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))

    def batchify(self, x, current_ob):
        """
        Convert
            s1 and [s2, s3, s4]
        into
            [s1, s2, s3], [s2, s3, s4]
        """
        obs = []
        next_obs = []
        ob = current_ob
        for h in range(self.planning_horizon):
            start_i = h * self.obs_dim
            next_ob = x[start_i:start_i+self.obs_dim]
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
            loss = (diff**2).sum(dim=1, keepdim=True)
        return loss

    def _feasibility_probabilities(self, x, current_ob):
        obs, next_obs = self.batchify(x, current_ob)
        # TODO: use dynamics num_steps_left
        return self.beta_v(obs, next_obs, self.num_steps_left)

    def cost_function(self, x, current_ob):
        x = ptu.np_to_var(x, requires_grad=True)
        current_ob = ptu.np_to_var(current_ob)
        env_costs = self._env_cost_function(x, current_ob)
        probabilities = self._feasibility_probabilities(x, current_ob)
        if self.use_max_cost:
            not_reached_cost = self.max_cost
        else:
            not_reached_cost = (
                (current_ob[self.goal_slice] - self.desired_features_torch)**2
            ).sum()
        if self.only_use_terminal_env_loss:
            final_prob = torch.prod(probabilities)
            loss = env_costs * (final_prob+1) + (1-final_prob) * not_reached_cost
        else:
            cum_probs = self._compute_cum_prob(probabilities)
            # Note: broadcasting happens on not_reached_cost
            loss = env_costs * cum_probs + (1-cum_probs) * not_reached_cost
            loss = loss.sum()
        loss_np = ptu.get_numpy(loss)[0].astype(np.float64)
        loss.backward()
        gradient_np = ptu.get_numpy(x.grad).astype(np.float64)
        return loss_np, gradient_np

    def _compute_cum_prob(self, probabilities):
        return (
                self.upper_tri + self.lower_and_diag_tri *
                probabilities.view(1, self.planning_horizon)
        ).prod(dim=1, keepdim=True)

    def reset(self):
        self.last_solution = None

    def get_action(self, current_ob):
        goal = self.env.multitask_goal[self.multitask_goal_slice]
        return self._get_action(current_ob, goal)

    def _get_action(self, current_ob, goal):
        if (
                self.replan_every_time_step
                or self.t_in_plan == self.planning_horizon
                or self.last_solution is None
        ):
            full_solution = self.replan(current_ob, goal)

            x_torch = ptu.np_to_var(full_solution, requires_grad=True)
            current_ob_torch = ptu.np_to_var(current_ob)

            _, next_obs = self.batchify(x_torch, current_ob_torch)
            self.subgoal_seq = np.array(
                [current_ob] + [ptu.get_numpy(o) for o in next_obs]
            )
            self.planned_action_seq = self.goal_reaching_policy.eval_np(
                self.subgoal_seq[:-1],
                self.subgoal_seq[1:],
                np.zeros((self.planning_horizon, 1))
            )
            self.last_solution = full_solution
            self.t_in_plan = 0

        action = self.planned_action_seq[self.t_in_plan]
        new_goal = self.subgoal_seq[self.t_in_plan+1]
        self.current_goal = new_goal
        oracle_qmax_action = self.get_oracle_qmax_action(current_ob,
                                                         new_goal)
        if self.use_oracle_argmax_policy:
            action = oracle_qmax_action

        agent_info = dict(
            planned_action_seq=self.planned_action_seq[self.t_in_plan:],
            subgoal_seq=self.subgoal_seq[self.t_in_plan:],
            oracle_qmax_action=oracle_qmax_action,
            full_action_seq=self.planned_action_seq,
            full_obs_seq=self.subgoal_seq,
        )

        self.t_in_plan += 1
        return action, agent_info

    def get_oracle_qmax_action(self, current_ob, goal):
        if self.oracle_argmax_policy is not None:
            return self.oracle_argmax_policy.get_action(current_ob, goal, 0)[0]

    def replan(self, current_ob, goal):
        if self.last_solution is None or not self.warm_start:
            solution = []
            for i in range(self.planning_horizon):
                solution.append(current_ob)
            self.last_solution = np.hstack(solution)
        self.desired_features_torch = ptu.np_to_var(
            goal[None].repeat(self.planning_horizon, 0)
        )
        x, f, d = optimize.fmin_l_bfgs_b(
            self.cost_function,
            self.last_solution,
            args=(current_ob,),
            bounds=self.bounds,
            **self.solver_kwargs
        )
        process_lbfgs_debug_dict(d)
        return x


class BetaVMultigoalLbfgs(BetaVLbfgsController):
    """
    Basically the same as LBfgsBCMC but use the goal passed into get_action

    TODO: maybe use num_steps_left to replace t_in_plan?
    """
    def get_action(self, current_ob, goal, num_steps_left):
        return self._get_action(current_ob, goal)
