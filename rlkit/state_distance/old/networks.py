import math

import numpy as np
import torch
from scipy import optimize
from torch import nn as nn
from torch import optim
from torch.nn import functional as F

from rlkit.state_distance.policies import UniversalPolicy
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.networks import Mlp
from rlkit.torch.core import PyTorchModule


class UniversalQfunction(PyTorchModule):
    """
    Represent Q(s, a, s_g, \gamma) with a two-alyer FF network.
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_state_dim,
            obs_hidden_size,
            embed_hidden_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            w_weight_generator=ptu.fanin_init_weights_like,
            b_init_value=0.1,
            bn_input=False,
            dropout=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout = dropout
        next_layer_size = observation_dim + goal_state_dim + 1
        if bn_input:
            self.process_input = nn.BatchNorm1d(next_layer_size)
        else:
            self.process_input = identity

        self.obs_fc = nn.Linear(next_layer_size, obs_hidden_size)
        new_weight = w_weight_generator(self.obs_fc.weight.data)
        self.obs_fc.weight.data.copy_(new_weight)
        self.obs_fc.bias.data.fill_(b_init_value)

        self.embed_fc = nn.Linear(
            obs_hidden_size + action_dim,
            embed_hidden_size,
        )
        new_weight = w_weight_generator(self.embed_fc.weight.data)
        self.embed_fc.weight.data.copy_(new_weight)
        self.embed_fc.bias.data.fill_(b_init_value)

        next_layer_size = obs_hidden_size + action_dim

        if dropout:
            self.obs_dropout = nn.Dropout()
            self.embed_dropout = nn.Dropout()

        self.last_fc = nn.Linear(embed_hidden_size, 1)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action, goal_state, discount):
        h = torch.cat((obs, goal_state, discount), dim=1)
        h = self.process_input(h)
        h = self.hidden_activation(self.obs_fc(h))
        if self.dropout:
            h = self.obs_dropout(h)
        h = torch.cat((h, action), dim=1)
        h = self.hidden_activation(self.embed_fc(h))
        if self.dropout:
            h = self.embed_dropout(h)
        return self.output_activation(self.last_fc(h))


class FlatUniversalQfunction(PyTorchModule):
    """
    Represent Q(s, a, s_g, \gamma) with a two-layer FF network.
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_state_dim,
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            dropout_prob=0,
            output_multiplier=1,
    ):
        if output_activation == F.softplus or output_activation == F.relu:
            assert output_multiplier < 0, "Q function should output negative #s"

        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_prob = dropout_prob
        self.output_multiplier = output_multiplier
        self.dropouts = []
        self.fcs = []
        in_size = observation_dim + goal_state_dim + action_dim + 1

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
            if self.dropout_prob > 0:
                dropout = nn.Dropout(p=self.dropout_prob)
                self.__setattr__("dropout{}".format(i), dropout)
                self.dropouts.append(dropout)

        self.last_fc = nn.Linear(in_size, 1)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action, goal_state, discount):
        h = torch.cat((obs, action, goal_state, discount), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.dropout_prob > 0:
                h = self.dropouts[i](h)
        return self.output_activation(self.last_fc(h)) * self.output_multiplier


class StructuredUniversalQfunction(PyTorchModule):
    """
    Parameterize QF as

    Q(s, a, s_g) = -||f(s, a) - s_g)||^2

    WARNING: this is only valid for when the reward is l2-norm (as opposed to a
    weighted l2-norm)
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_state_dim,
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.relu,
            hidden_init=ptu.fanin_init,
            bn_input=False,
            dropout_prob=0,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.dropout_prob = dropout_prob
        self.dropouts = []
        self.fcs = []
        in_size = observation_dim + action_dim + 1
        if bn_input:
            self.process_input = nn.BatchNorm1d(in_size)
        else:
            self.process_input = identity

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(0)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
            if self.dropout_prob > 0:
                dropout = nn.Dropout(p=self.dropout_prob)
                self.__setattr__("dropout{}".format(i), dropout)
                self.dropouts.append(dropout)

        self.last_fc = nn.Linear(in_size, goal_state_dim)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(
            self,
            obs,
            action,
            goal_state,
            discount,
            only_return_next_state=False,
    ):
        h = torch.cat((obs, action, discount), dim=1)
        h = self.process_input(h)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.dropout_prob > 0:
                h = self.dropouts[i](h)
        next_state = self.last_fc(h)
        if only_return_next_state:
            return next_state
        out = - torch.norm(goal_state - next_state, p=2, dim=1)
        return out.unsqueeze(1)


class DuelingStructuredUniversalQfunction(PyTorchModule):
    """
    Parameterize QF as

    Q(s, a, s_g) = V(s, s_g) + A(s, a, s_g) - A(s, pi(s))

    where

    V(s) = -||f(s, s_g) - s_g)||^2
    A(s, a) = -||f(s, a, s_g) - s_g)||^2
    pi(s) = argmax_a A(s, a)

    WARNING: this is only valid for when the reward is l2-norm (as opposed to a
    weighted l2-norm)
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_state_dim,
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.relu,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super().__init__()

        # Put it in a list so that it does not count as a sub-module
        self.argmax_policy_lst = None
        self.hidden_activation = hidden_activation

        self.v_fcs = []

        in_size = observation_dim + goal_state_dim + 1
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(0)
            self.__setattr__("v_fc{}".format(i), fc)
            self.v_fcs.append(fc)

        self.v_last_fc = nn.Linear(in_size, goal_state_dim)
        self.v_last_fc.weight.data.uniform_(-init_w, init_w)
        self.v_last_fc.bias.data.uniform_(-init_w, init_w)

        self.a_function = StructuredUniversalQfunction(
            observation_dim,
            action_dim,
            goal_state_dim,
            hidden_sizes,
            init_w=init_w,
            hidden_activation=hidden_activation,
            hidden_init=hidden_init,
        )

    def set_argmax_policy(self, argmax_policy):
        self.argmax_policy_lst = [argmax_policy]

    def forward(
            self,
            obs,
            action,
            goal_state,
            discount,
    ):
        a = self.a_function(obs, action, goal_state, discount)
        a_max = self.a_function(
            obs,
            self.argmax_policy_lst[0](obs, goal_state, discount),
            goal_state,
            discount,
        )

        h = torch.cat((obs, discount, goal_state), dim=1)
        for i, fc in enumerate(self.v_fcs):
            h = self.hidden_activation(fc(h))
        next_state = self.v_last_fc(h)
        v = - torch.norm(goal_state - next_state, p=2, dim=1)
        return v.unsqueeze(1) + a - a_max


class GoalStructuredUniversalQfunction(PyTorchModule):
    """
    Parameterize QF as

    Q(s, a, s_g, discount) = - ||f(s, a, s_g, discount) - s_g)||^2

    WARNING: this is only valid for when the reward is l2-norm (as opposed to a
    weighted l2-norm)
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_state_dim,
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.relu,
            hidden_init=ptu.fanin_init,
            bn_input=False,
            dropout_prob=0,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.dropout_prob = dropout_prob
        self.dropouts = []
        self.fcs = []
        in_size = 2 * observation_dim + action_dim + 1
        if bn_input:
            self.process_input = nn.BatchNorm1d(in_size)
        else:
            self.process_input = identity

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(0)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
            if self.dropout_prob > 0:
                dropout = nn.Dropout(p=self.dropout_prob)
                self.__setattr__("dropout{}".format(i), dropout)
                self.dropouts.append(dropout)

        self.last_fc = nn.Linear(in_size, goal_state_dim)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(
            self,
            obs,
            action,
            goal_state,
            discount,
            only_return_next_state=False,
    ):
        h = torch.cat((obs, action, goal_state, discount), dim=1)
        h = self.process_input(h)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.dropout_prob > 0:
                h = self.dropouts[i](h)
        next_state = self.last_fc(h)
        if only_return_next_state:
            return next_state
        out = - torch.norm(goal_state - next_state, p=2, dim=1)
        return out.unsqueeze(1)


class VectorizedGoalStructuredUniversalQfunction(PyTorchModule):
    """
    Parameterize QF as

    Q(s, a, s_g, discount) = - |f(s, a, s_g, discount) - s_g)|

    element-wisze

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            hidden_sizes,
            init_w=3e-3,
            hidden_activation=F.relu,
            hidden_init=ptu.fanin_init,
            bn_input=False,
            dropout_prob=0,
    ):
        # Keeping it as a separate argument to have same interface
        # assert observation_dim == goal_dim
        self.save_init_params(locals())
        super().__init__()

        self.hidden_activation = hidden_activation
        self.dropout_prob = dropout_prob
        self.dropouts = []
        self.fcs = []
        in_size = goal_dim + observation_dim + action_dim + 1
        if bn_input:
            self.process_input = nn.BatchNorm1d(in_size)
        else:
            self.process_input = identity

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(0)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
            if self.dropout_prob > 0:
                dropout = nn.Dropout(p=self.dropout_prob)
                self.__setattr__("dropout{}".format(i), dropout)
                self.dropouts.append(dropout)

        self.last_fc = nn.Linear(in_size, goal_dim)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(
            self,
            obs,
            action,
            goal_state,
            discount,
            only_return_next_state=False,
    ):
        h = torch.cat((obs, action, goal_state, discount), dim=1)
        h = self.process_input(h)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.dropout_prob > 0:
                h = self.dropouts[i](h)
        next_state = self.last_fc(h)
        if only_return_next_state:
            return next_state
        out = - torch.abs(goal_state - next_state)
        return out


class GoalConditionedDeltaModel(Mlp):
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            hidden_sizes,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            output_size=observation_dim,
            input_size=observation_dim + goal_dim + action_dim + 1,
            **kwargs
        )

    def forward(self, obs, action, goal_state, discount):
        h = torch.cat((obs, action, goal_state, discount), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


class TauBinaryGoalConditionedDeltaModel(Mlp):
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            hidden_sizes,
            max_tau,
            **kwargs
    ):
        self.save_init_params(locals())
        self.word_len = math.ceil(np.log2(max_tau))
        super().__init__(
            hidden_sizes,
            output_size=observation_dim,
            input_size=observation_dim + goal_dim + action_dim + self.word_len,
            **kwargs
        )

    def forward(self, obs, action, goal_state, tau):
        """
        tau isn't differentiated through anyways, so I do the conversion in np.

        """
        tau_np = ptu.get_numpy(tau.int().squeeze(1))
        tau_binary_np = (
            (
                ((tau_np[:, None] & (1 << np.arange(self.word_len)))) > 0
            ).astype(float)
        )
        tau_binary = ptu.np_to_var(tau_binary_np)
        h = torch.cat((obs, action, goal_state, tau_binary), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h))


class ModelExtractor(PyTorchModule):
    def __init__(self, qf, discount=0.):
        super().__init__()
        assert isinstance(qf, StructuredUniversalQfunction)
        self.qf = qf
        self.discount = discount

    def forward(self, state, action):
        batch_size = state.size()[0]
        discount = ptu.np_to_var(self.discount + np.zeros((batch_size, 1)))
        return self.qf(state, action, None, discount, True)


class NumpyModelExtractor(PyTorchModule):
    def __init__(
            self,
            qf,
            discount=0.,
            sample_size=100,
            learning_rate=1e-1,
            num_gradient_steps=100,
            state_optimizer='adam',
    ):
        super().__init__()
        self._is_structured_qf = isinstance(qf, StructuredUniversalQfunction)
        self.qf = qf
        self.discount = discount
        self.sample_size = sample_size
        self.learning_rate = learning_rate
        self.num_optimization_steps = num_gradient_steps
        self.state_optimizer = state_optimizer

    def expand_to_sample_size(self, torch_array):
        return torch_array.repeat(self.sample_size, 1)

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.sample_size,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=False)

    def next_state(self, state, action):
        if self._is_structured_qf:
            states = ptu.np_to_var(np.expand_dims(state, 0))
            actions = ptu.np_to_var(np.expand_dims(action, 0))
            discount = ptu.np_to_var(self.discount + np.zeros((1, 1)))
            return ptu.get_numpy(
                self.qf(states, actions, None, discount, True).squeeze(0)
            )

        if self.state_optimizer == 'adam':
            discount = ptu.np_to_var(
                self.discount * np.ones((self.sample_size, 1))
            )
            obs_dim = state.shape[0]
            states = self.expand_np_to_var(state)
            actions = self.expand_np_to_var(action)
            next_states_np = np.zeros((self.sample_size, obs_dim))
            next_states = ptu.np_to_var(next_states_np, requires_grad=True)
            optimizer = optim.Adam([next_states], self.learning_rate)

            for _ in range(self.num_optimization_steps):
                losses = -self.qf(
                    states,
                    actions,
                    next_states,
                    discount,
                )
                loss = losses.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses_np = ptu.get_numpy(losses)
            best_action_i = np.argmin(losses_np)
            return ptu.get_numpy(next_states[best_action_i, :])
        elif self.state_optimizer == 'lbfgs':
            next_states = []
            for i in range(len(states)):
                state = states[i:i+1, :]
                action = actions[i:i+1, :]
                loss_f = self.create_loss(state, action, return_gradient=True)
                results = optimize.fmin_l_bfgs_b(
                    loss_f,
                    np.zeros((1, obs_dim)),
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
                    np.zeros((1, obs_dim)),
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


class NumpyGoalConditionedModelExtractor(PyTorchModule):
    """
    Extract a goal-conditioned model
    """
    def __init__(
            self,
            qf,
    ):
        super().__init__()
        assert (
            isinstance(qf, StructuredUniversalQfunction)
            or isinstance(qf, VectorizedGoalStructuredUniversalQfunction)
        )
        self.qf = qf

    def next_state(self, state, action, goal_state, discount):
        state = ptu.np_to_var(np.expand_dims(state, 0))
        action = ptu.np_to_var(np.expand_dims(action, 0))
        goal_state = ptu.np_to_var(np.expand_dims(goal_state, 0))
        discount = ptu.np_to_var(np.array([[discount]]))
        return ptu.get_numpy(
            self.qf(state, action, goal_state, discount, True).squeeze(0)
        )


class NumpyGoalConditionedDeltaModelExtractor(PyTorchModule):
    """
    Extract a goal-conditioned model
    """
    def __init__(
            self,
            qf,
    ):
        super().__init__()
        assert (
            isinstance(qf, GoalConditionedDeltaModel)
            or isinstance(qf, TauBinaryGoalConditionedDeltaModel)
        )
        self.qf = qf

    def next_state(self, state, action, goal_state, discount):
        state = ptu.np_to_var(np.expand_dims(state, 0))
        action = ptu.np_to_var(np.expand_dims(action, 0))
        goal_state = ptu.np_to_var(np.expand_dims(goal_state, 0))
        discount = ptu.np_to_var(np.array([[discount]]))
        return ptu.get_numpy(
            self.qf(state, action, goal_state, discount) + state
        )[0]


class FFUniversalPolicy(PyTorchModule, UniversalPolicy):
    def __init__(
            self,
            obs_dim,
            action_dim,
            goal_dim,
            fc1_size,
            fc2_size,
            init_w=3e-3,
            b_init_value=0.1,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super().__init__()
        UniversalPolicy.__init__(self)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.hidden_init = hidden_init

        self.fc1 = nn.Linear(obs_dim + goal_dim + 1, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(b_init_value)
        hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(b_init_value)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, goal, discount):
        h = torch.cat((obs, goal, discount), dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs_np):
        action = self.eval_np(
            obs_np[None],
            self._goal_expanded_np,
            self._tau_expanded_np,
        )[0, :]
        return action, {}

    def get_actions(self, observations):
        batch_size = observations.shape[0]
        return self.eval_np(
            observations,
            np.repeat(self._goal_expanded_np, batch_size, 0),
            np.repeat(self._tau_expanded_np, batch_size, 0),
        )
