import torch

from rlkit.policies.base import Policy
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
import numpy as np


class BetaQ(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            **flatten_mlp_kwargs
    ):
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            input_size=(
                    self.observation_dim + self.action_dim + self.goal_dim + 1
            ),
            output_size=self.goal_dim if vectorized else 1,
            hidden_activation=torch.tanh,
            output_activation=torch.sigmoid,
            **flatten_mlp_kwargs
        )
        self.env = env
        self.vectorized = vectorized

    def forward(self, observations, actions, goals, num_steps_left, **kwargs):
        flat_inputs = torch.cat(
            (observations, actions, goals, num_steps_left),
            dim=1,
        )
        return super().forward(flat_inputs, **kwargs)

    def create_eval_function(self, obs, goal, num_steps_left):
        def beta_eval(a1, a2):
            actions = np.array([[a1, a2]])
            return self.eval_np(
                observations=np.array([[
                    *obs
                ]]),
                actions=actions,
                goals=np.array([[
                    *goal
                ]]),
                num_steps_left=np.array([[num_steps_left]])
            )[0, 0]
        return beta_eval


class BetaV(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            **flatten_mlp_kwargs
    ):
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            input_size=(
                    self.observation_dim + self.goal_dim + 1
            ),
            output_size=self.goal_dim if vectorized else 1,
            hidden_activation=torch.tanh,
            output_activation=torch.sigmoid,
            **flatten_mlp_kwargs
        )
        self.env = env
        self.vectorized = vectorized

    def forward(self, observations, goals, num_steps_left, **kwargs):
        flat_inputs = torch.cat(
            (observations, goals, num_steps_left),
            dim=1,
        )
        return super().forward(flat_inputs, **kwargs)


class TanhFlattenMlpPolicy(TanhMlpPolicy):
    def __init__(
            self,
            env,
            **kwargs
    ):
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        self.env = env
        super().__init__(
            input_size=self.observation_dim + self.goal_dim + 1,
            output_size=self.action_dim,
            **kwargs
        )

    def forward(
            self,
            observations,
            goals,
            num_steps_left,
            **kwargs
    ):
        flat_input = torch.cat((observations, goals, num_steps_left), dim=1)
        return super().forward(flat_input, **kwargs)

    def get_action(self, ob_np, goal_np, tau_np):
        actions = self.eval_np(
            ob_np[None],
            goal_np[None],
            tau_np[None],
        )
        return actions[0, :], {}


class ArgmaxBetaQPolicy(Policy):
    """
    For debugging, do a grid search to take the argmax.
    """
    def __init__(self, beta_q):
        self.beta_q = beta_q
        x_values = np.linspace(-1, 1, num=10)
        y_values = np.linspace(-1, 1, num=10)
        x_values_all, y_values_all = np.meshgrid(x_values, y_values)
        x_values_flat = x_values_all.flatten()
        y_values_flat = y_values_all.flatten()
        self.all_actions = np.vstack((x_values_flat, y_values_flat)).T

    def get_action(self, observation, goal, num_steps_left):
        obs = observation[None].repeat(100, 0)
        goals = goal[None].repeat(100, 0)
        num_steps_left = num_steps_left * np.ones((100, 1))
        beta_values = self.beta_q.eval_np(
            observations=obs,
            goals=goals,
            actions=self.all_actions,
            num_steps_left=num_steps_left,
        )
        max_i = np.argmax(beta_values)
        return self.all_actions[max_i], {}
