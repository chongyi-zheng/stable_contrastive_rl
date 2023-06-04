import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.rnn import BNLSTMCell, LSTM


class FeedForwardDuelingQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            observation_hidden_size,
            embedded_hidden_size,
            init_w=3e-3,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            batchnorm_obs=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size
        self.hidden_init = hidden_init
        self.obs_fc = nn.Linear(obs_dim, observation_hidden_size)

        self.value_embedded_fc = nn.Linear(observation_hidden_size, embedded_hidden_size)
        self.advantage_embedded_fc = nn.Linear(observation_hidden_size + action_dim, embedded_hidden_size)
        # self.advantage_avg = np.zeros(embedded_hidden_size)
        self.advantage_last_fc = nn.Linear(embedded_hidden_size, 1)
        self.value_last_fc = nn.Linear(embedded_hidden_size, 1)

        self.output_activation = output_activation
        self.init_weights(init_w)
        self.batchnorm_obs = batchnorm_obs
        if self.batchnorm_obs:
            self.bn_obs = nn.BatchNorm1d(obs_dim)

    def init_weights(self, init_w):
        self.hidden_init(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)

        self.hidden_init(self.value_embedded_fc.weight)
        self.hidden_init(self.advantage_embedded_fc.weight)
        self.value_embedded_fc.bias.data.fill_(0)
        self.advantage_embedded_fc.bias.data.fill_(0)

        self.advantage_last_fc.weight.data.uniform_(-init_w, init_w)
        self.value_last_fc.weight.data.uniform_(-init_w, init_w)

        self.advantage_last_fc.bias.data.uniform_(-init_w, init_w)
        self.value_last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        if self.batchnorm_obs:
            obs = self.bn_obs(obs)
        h = obs
        h = F.relu(self.obs_fc(h))

        val_input = h
        advantage_input = torch.cat((h, action), dim=1)

        value = F.relu(self.value_embedded_fc(val_input))
        value = self.output_activation(self.value_last_fc(value))

        advantage = F.relu(self.advantage_embedded_fc(advantage_input))
        advantage = self.output_activation(self.advantage_last_fc(advantage))

        # a_average = self._compute_running_average(advantage)
        q = value + advantage
        return q

    def _compute_running_average(self, update):
        avg = self.advantage_avg
        if self.training:
            self.advantage_avg = .9 * self.advantage_avg + .1 * update
        return avg

class MemoryQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            fc1_size,
            fc2_size,
            init_w=3e-3,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            ignore_memory=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.observation_hidden_size = fc1_size
        self.embedded_hidden_size = fc2_size
        self.init_w = init_w
        self.hidden_init = hidden_init
        self.ignore_memory = ignore_memory

        if self.ignore_memory:
            self.obs_fc = nn.Linear(self.obs_dim, self.observation_hidden_size)
            self.embedded_fc = nn.Linear(
                self.observation_hidden_size + self.action_dim,
                fc2_size,
            )
        else:
            self.obs_fc = nn.Linear(obs_dim + memory_dim, fc1_size)
            self.embedded_fc = nn.Linear(
                fc1_size + action_dim + memory_dim,
                fc2_size,
            )
        self.last_fc = nn.Linear(fc2_size, 1)
        self.output_activation = output_activation

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        self.hidden_init(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, memory, action, write):
        if self.ignore_memory:
            obs_embedded = F.relu(self.obs_fc(obs))
            x = torch.cat((obs_embedded, action), dim=1)
            x = F.relu(self.embedded_fc(x))
        else:
            obs_embedded = torch.cat((obs, memory), dim=1)
            obs_embedded = F.relu(self.obs_fc(obs_embedded))
            x = torch.cat((obs_embedded, action, write), dim=1)
            x = F.relu(self.embedded_fc(x))
        return self.output_activation(self.last_fc(x))


class RecurrentQFunction(PyTorchModule):
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

        self.lstm = LSTM(
            BNLSTMCell,
            self.obs_dim + self.action_dim,
            self.hidden_size,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.hidden_size + self.obs_dim, fc1_size)
        self.fc2 = nn.Linear(self.fc1_size + self.action_dim, fc2_size)
        self.last_fc = nn.Linear(self.fc2_size, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        """
        :param obs: torch Variable, [batch_size, sequence length, obs dim]
        :param action: torch Variable, [batch_size, sequence length, action dim]
        :return: torch Variable, [batch_size, sequence length, 1]
        """
        assert len(obs.size()) == 3
        inputs = torch.cat((obs, action), dim=2)
        batch_size, subsequence_length = obs.size()[:2]
        cx = Variable(
            ptu.FloatTensor(1, batch_size, self.hidden_size)
        )
        cx.data.fill_(0)
        hx = Variable(
            ptu.FloatTensor(1, batch_size, self.hidden_size)
        )
        hx.data.fill_(0)
        rnn_outputs, _ = self.lstm(inputs, (hx, cx))
        rnn_outputs.contiguous()
        rnn_outputs_flat = rnn_outputs.view(-1, self.hidden_size)
        obs_flat = obs.view(-1, self.obs_dim)
        action_flat = action.view(-1, self.action_dim)
        h = torch.cat((rnn_outputs_flat, obs_flat), dim=1)
        h = F.relu(self.fc1(h))
        h = torch.cat((h, action_flat), dim=1)
        h = F.relu(self.fc2(h))
        outputs_flat = self.last_fc(h)
        return outputs_flat.view(batch_size, subsequence_length, 1)

    @property
    def is_recurrent(self):
        return True


class RecurrentMemoryQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            hidden_size,
            fc1_size,
            fc2_size,
            init_w=3e-3,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.output_activation = output_activation
        self.hidden_init = hidden_init
        self.rnn = nn.LSTM(
            self.obs_dim + self.action_dim + 2 * self.memory_dim,
            self.hidden_size,
            1,
            batch_first=True,
        )
        self.last_fc = nn.Linear(self.hidden_size, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, memory, action, write):
        """
        :param obs: torch Variable, [batch_size, sequence length, obs dim]
        :param memory: torch Variable, [batch_size, sequence length, memory dim]
        :param action: torch Variable, [batch_size, sequence length, action dim]
        :param write: torch Variable, [batch_size, sequence length, memory dim]
        :return: torch Variable, [batch_size, sequence length, 1]
        """
        rnn_inputs = torch.cat((obs, memory, action, write), dim=2)
        batch_size, subsequence_length, _ = obs.size()
        cx = Variable(
            ptu.FloatTensor(1, batch_size, self.hidden_size)
        )
        cx.data.fill_(0)
        hx = Variable(
            ptu.FloatTensor(1, batch_size, self.hidden_size)
        )
        hx.data.fill_(0)
        state = (hx, cx)
        rnn_outputs, _ = self.rnn(rnn_inputs, state)
        rnn_outputs.contiguous()
        rnn_outputs_flat = rnn_outputs.view(
            batch_size * subsequence_length,
            self.fc1.in_features,
        )
        outputs_flat = self.output_activation(self.last_fc(rnn_outputs_flat))
        return outputs_flat.view(batch_size, subsequence_length, 1)

    @property
    def is_recurrent(self):
        return True
