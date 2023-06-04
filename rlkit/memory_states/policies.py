import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.rnn import LSTMCell


class FlattenLSTMCell(nn.Module):
    def __init__(self, lstm_cell):
        self.lstm_cell = lstm_cell

    def forward(self, input, state):
        hx, cx = torch.split(state, self.memory_dim // 2, dim=1)
        new_hx, new_cx = self.lstm_cell(input, (hx, cx))
        new_state = torch.cat((new_hx, new_cx), dim=1)
        return hx, new_state


class LinearBN(nn.Module):
    """
    Linear then BN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features)
        self.weight = self.fc.weight
        if bias:
            self.bias = self.fc.bias

    def forward(self, input):
        return self.bn(self.fc(input))


class RWACell(PyTorchModule):
    def __init__(
            self,
            input_dim,
            num_units,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.input_dim = input_dim
        self.num_units = num_units

        self.fc_u = nn.Linear(input_dim, num_units)
        self.fc_g = nn.Linear(input_dim + num_units, num_units)
        # Bias term factors when from numerate and denominator
        self.fc_a = nn.Linear(input_dim + num_units, num_units, bias=False)
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal(self.fc_u.weight)
        self.fc_u.bias.data.fill_(0)
        init.kaiming_normal(self.fc_g.weight)
        self.fc_g.bias.data.fill_(0)
        init.kaiming_normal(self.fc_a.weight)

    def forward(self, inputs, state):
        h, n, d, a_max = state

        u = self.fc_u(inputs)
        g = self.fc_g(torch.cat((inputs, h), dim=1))
        z = u * F.tanh(g)
        a = self.fc_a(torch.cat((inputs, h), dim=1))

        # Numerically stable update of numerator and denom
        a_newmax = ptu.maximum_2d(a_max, a)
        exp_diff = torch.exp(a_max-a_newmax)
        weight_scaled = torch.exp(a-a_newmax)
        n_new = n * exp_diff + z * weight_scaled
        d_new = d * exp_diff + weight_scaled
        h_new = F.tanh(n_new / d_new)

        return h_new, n_new, d_new, a_max

    @staticmethod
    def state_num_split():
        return 4


class MemoryPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            fc1_size,
            fc2_size,
            init_w=1e-3,
            cell_class=LSTMCell,
            hidden_init=ptu.fanin_init,
            feed_action_to_memory=False,
            output_activation=F.tanh,
            only_one_fc_for_action=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.hidden_init = hidden_init
        self.feed_action_to_memory = feed_action_to_memory
        self.output_activation = output_activation
        self.only_one_fc_for_action = only_one_fc_for_action

        if self.only_one_fc_for_action:
            self.last_fc = nn.Linear(obs_dim + memory_dim, action_dim)
        else:
            self.fc1 = nn.Linear(obs_dim + memory_dim, fc1_size)
            self.fc2 = nn.Linear(fc1_size, fc2_size)
            self.last_fc = nn.Linear(fc2_size, action_dim)
        self.num_splits_for_rnn_internally = cell_class.state_num_split()
        assert memory_dim % self.num_splits_for_rnn_internally == 0
        if self.feed_action_to_memory:
            cell_input_dim = self.action_dim + self.obs_dim
        else:
            cell_input_dim = self.obs_dim
        self.rnn_cell = cell_class(
            cell_input_dim,
            self.memory_dim // self.num_splits_for_rnn_internally,
        )
        self.init_weights(init_w)

    def init_weights(self, init_w):
        if not self.only_one_fc_for_action:
            self.hidden_init(self.fc1.weight)
            self.fc1.bias.data.fill_(0)
            self.hidden_init(self.fc2.weight)
            self.fc2.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def action_parameters(self):
        if self.only_one_fc_for_action:
            layers = [self.last_fc]
        else:
            layers = [self.fc1, self.fc2, self.last_fc]
        for fc in layers:
            for param in fc.parameters():
                yield param

    def write_parameters(self):
        return self.rnn_cell.parameters()

    def forward_action(self, input_to_action):
        if self.only_one_fc_for_action:
            return self.output_activation(self.last_fc(input_to_action))
        else:
            h1 = F.tanh(self.fc1(input_to_action))
            h2 = F.tanh(self.fc2(h1))
            return self.output_activation(self.last_fc(h2))

    def forward(self, obs, initial_memory):
        """
        :param obs: torch Variable, [batch_size, sequence length, obs dim]
        :param initial_memory: torch Variable, [batch_size, memory dim]
        :return: (actions, writes) tuple
            actions: [batch_size, sequence length, action dim]
            writes: [batch_size, sequence length, memory dim]
        """
        assert len(obs.size()) == 3
        assert len(initial_memory.size()) == 2
        batch_size, subsequence_length = obs.size()[:2]

        subtraj_writes = Variable(
            ptu.FloatTensor(batch_size, subsequence_length, self.memory_dim),
            requires_grad=False
        )
        subtraj_actions = Variable(
            ptu.FloatTensor(batch_size, subsequence_length, self.action_dim),
            requires_grad=False
        )
        if self.feed_action_to_memory:
            if self.num_splits_for_rnn_internally > 1:
                state = torch.split(
                    initial_memory,
                    self.memory_dim // self.num_splits_for_rnn_internally,
                    dim=1,
                )
                for i in range(subsequence_length):
                    current_obs = obs[:, i, :]
                    augmented_state = torch.cat((current_obs,) + state, dim=1)
                    action = self.forward_action(augmented_state)
                    rnn_input = torch.cat([current_obs, action], dim=1)
                    state = self.rnn_cell(rnn_input, state)
                    subtraj_writes[:, i, :] = torch.cat(state, dim=1)
                    subtraj_actions[:, i, :] = action
            else:
                state = initial_memory
                for i in range(subsequence_length):
                    current_obs = obs[:, i, :]
                    augmented_state = torch.cat([current_obs, state], dim=1)
                    action = self.forward_action(augmented_state)
                    rnn_input = torch.cat([current_obs, action], dim=1)
                    state = self.rnn_cell(rnn_input, state)
                    subtraj_writes[:, i, :] = state
                    subtraj_actions[:, i, :] = action
            return subtraj_actions, subtraj_writes

        """
        Create the new writes.
        """
        if self.num_splits_for_rnn_internally > 1:
            state = torch.split(
                initial_memory,
                self.memory_dim // self.num_splits_for_rnn_internally,
                dim=1,
            )
            for i in range(subsequence_length):
                state = self.rnn_cell(obs[:, i, :], state)
                subtraj_writes[:, i, :] = torch.cat(state, dim=1)
        else:
            state = initial_memory
            for i in range(subsequence_length):
                state = self.rnn_cell(obs[:, i, :], state)
                subtraj_writes[:, i, :] = state

        # The reason that using a LSTM doesn't work is that this gives you only
        # the FINAL hx and cx, not all of them :(

        """
        Create the new subtrajectory memories with the initial memories and the
        new writes.
        """
        expanded_init_memory = initial_memory.unsqueeze(1)
        if subsequence_length > 1:
            memories = torch.cat(
                (
                    expanded_init_memory,
                    subtraj_writes[:, :-1, :],
                ),
                dim=1,
            )
        else:
            memories = expanded_init_memory

        """
        Use new memories to create env actions.
        """
        all_subtraj_inputs = torch.cat([obs, memories], dim=2)
        for i in range(subsequence_length):
            augmented_state = all_subtraj_inputs[:, i, :]
            action = self.forward_action(augmented_state)
            subtraj_actions[:, i, :] = action

        return subtraj_actions, subtraj_writes

    def get_action(self, augmented_obs):
        """
        :param augmented_obs: (obs, memories) tuple
            obs: np.ndarray, [obs_dim]
            memories: nd.ndarray, [memory_dim]
        :return: (actions, writes) tuple
            actions: np.ndarray, [action_dim]
            writes: np.ndarray, [writes_dim]
        """
        obs, memory = augmented_obs
        obs = np.expand_dims(obs, axis=0)
        memory = np.expand_dims(memory, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        memory = Variable(ptu.from_numpy(memory).float(), requires_grad=False)
        action, write = self.get_flat_output(obs, memory)
        return (
                   np.squeeze(ptu.get_numpy(action), axis=0),
                   np.squeeze(ptu.get_numpy(write), axis=0),
               ), {}

    def get_flat_output(self, obs, initial_memories):
        """
        Each batch element is processed independently. So, there's no recurrency
        used.

        :param obs: torch Variable, [batch_size X obs_dim]
        :param initial_memories: torch Variable, [batch_size X memory_dim]
        :return: (actions, writes) Tuple
            actions: torch Variable, [batch_size X action_dim]
            writes: torch Variable, [batch_size X writes_dim]
        """
        obs = obs.unsqueeze(1)
        actions, writes = self.__call__(obs, initial_memories)
        return torch.squeeze(actions, dim=1), torch.squeeze(writes, dim=1)


class RecurrentPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_size,
            fc1_size,
            fc2_size,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.hidden_init = hidden_init
        self.rnn = nn.LSTM(
            self.obs_dim,
            self.hidden_size,
            1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.hidden_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.last_fc = nn.Linear(self.fc2_size, self.action_dim)

        self.state = None
        self.init_weights(init_w)
        self.reset()

    def init_weights(self, init_w):
        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, state=None):
        """
        :param obs: torch Variable, [batch_size, sequence length, obs dim]
        :param state: initial state of the RNN, [?, batch_size, hidden_size]
        :return: torch Variable, [batch_size, sequence length, action dim]
        """
        assert len(obs.size()) == 3
        batch_size, subsequence_length = obs.size()[:2]
        if state is None:
            state = self.get_new_state(batch_size)
        rnn_outputs, new_state = self.rnn(obs, state)
        rnn_outputs.contiguous()
        rnn_outputs_flat = rnn_outputs.view(
            batch_size * subsequence_length,
            self.hidden_size,
        )
        h = F.relu(self.fc1(rnn_outputs_flat))
        h = F.relu(self.fc2(h))
        outputs_flat = F.tanh(self.last_fc(h))

        return outputs_flat.view(
            batch_size, subsequence_length, self.action_dim
        ), new_state

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = np.expand_dims(obs, axis=1)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action, state = self.__call__(
            obs, state=self.state,
        )
        self.state = state
        action = action.squeeze(0)
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}

    def get_new_state(self, batch_size):
        cx = Variable(
            ptu.FloatTensor(1, batch_size, self.hidden_size)
        )
        cx.data.fill_(0)
        hx = Variable(
            ptu.FloatTensor(1, batch_size, self.hidden_size)
        )
        hx.data.fill_(0)
        return hx, cx

    def reset(self):
        self.state = self.get_new_state(1)
