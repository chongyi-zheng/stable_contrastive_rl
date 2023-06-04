import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from rlkit.state_distance.policies import UniversalPolicy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.networks import Mlp
from rlkit.torch.core import PyTorchModule
from rlkit.core import logger


class AmortizedPolicy(PyTorchModule, UniversalPolicy):
    def __init__(
            self,
            goal_reaching_policy,
            goal_chooser,
    ):
        self.save_init_params(locals())
        super().__init__()
        UniversalPolicy.__init__(self)
        self.goal_reaching_policy = goal_reaching_policy
        self.goal_chooser = goal_chooser

    def get_action(self, obs_np):
        obs = ptu.np_to_var(
            np.expand_dims(obs_np, 0)
        )
        goal = self.goal_chooser(obs, self._goal_expanded_torch)
        action = self.goal_reaching_policy(
            obs,
            goal,
            self._tau_expanded_torch,
        )
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}


class ReacherGoalChooser(Mlp):
    def __init__(
            self,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            input_size=6,
            output_size=4,
            **kwargs
        )

    def forward(self, input):
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        output = self.last_fc(h)
        output_theta_pre_activation = output[:, :2]
        theta = np.pi * F.tanh(output_theta_pre_activation)
        output_vel = output[:, 2:]

        return torch.cat(
            (
                torch.cos(theta),
                torch.sin(theta),
                output_vel,
            ),
            dim=1
        )


class UniversalGoalChooser(Mlp):
    def __init__(
            self,
            input_goal_dim,
            output_goal_dim,
            obs_dim,
            reward_function,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            input_size=obs_dim + input_goal_dim,
            output_size=output_goal_dim,
            **kwargs
        )
        self.reward_function = reward_function

    def forward(self, obs, goal):
        goal = goal[:, :7]
        obs = torch.cat((obs, goal), dim=1)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


def train_amortized_goal_chooser(
        goal_chooser,
        goal_conditioned_model,
        argmax_q,
        discount,
        replay_buffer,
        learning_rate=1e-3,
        batch_size=32,
        num_updates=1000,
):
    def get_loss(training=False):
        buffer = replay_buffer.get_replay_buffer(training)
        batch = buffer.random_batch(batch_size)
        obs = ptu.np_to_var(batch['observations'], requires_grad=False)
        goals = ptu.np_to_var(batch['goal_states'], requires_grad=False)
        goal = goal_chooser(obs, goals)
        actions = argmax_q(
            obs,
            goal,
            discount
        )
        final_state_predicted = goal_conditioned_model(
            obs,
            actions,
            goal,
            discount,
        ) + obs
        rewards = goal_chooser.reward_function(final_state_predicted, goals)
        return -rewards.mean()

    discount = ptu.np_to_var(discount * np.ones((batch_size, 1)))
    optimizer = optim.Adam(goal_chooser.parameters(), learning_rate)
    for i in range(num_updates):
        optimizer.zero_grad()
        loss = get_loss()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logger.log("Number updates: {}".format(i))
            logger.log("Train loss: {}".format(
                float(ptu.get_numpy(loss)))
            )
            logger.log("Validation loss: {}".format(
                float(ptu.get_numpy(get_loss(training=False))))
            )
