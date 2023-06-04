import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable

from rlkit.policies.base import ExplorationPolicy
from rlkit.state_distance.experimental_tdm_networks import make_binary_tensor, \
    SeparateFirstLayerMlp
from rlkit.state_distance.util import split_tau
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.sac.policies import TanhGaussianPolicy, LOG_SIG_MIN, \
    LOG_SIG_MAX


class StandardTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            std=None,
            init_w=1e-3,
            max_tau=None,
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim+goal_dim + 1,
            action_dim=action_dim,
            std=std,
            init_w=init_w,
            **kwargs
        )


class OneHotTauTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            max_tau,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.max_tau = max_tau
        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim+max_tau+goal_dim+1,
            action_dim=action_dim,
            init_w=init_w,
            **kwargs
        )

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        obs, taus = split_tau(obs)
        h = obs
        batch_size = h.size()[0]
        y_binary = ptu.FloatTensor(batch_size, self.max_tau + 1)
        y_binary.zero_()
        t = taus.data.long()
        t = torch.clamp(t, min=0)
        y_binary.scatter_(1, t, 1)

        h = torch.cat((
            obs,
            ptu.Variable(y_binary),
        ), dim=1)

        return super().forward(
            obs=h,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            return_entropy=return_entropy,
            return_log_prob_of_mean=return_log_prob_of_mean,
        )


class BinaryTauTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            max_tau,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.max_tau = np.unpackbits(np.array(max_tau, dtype=np.uint8))
        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim + goal_dim+ len(self.max_tau),
            action_dim=action_dim,
            init_w=init_w,
            **kwargs
        )
    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        obs, taus = split_tau(obs)
        batch_size = taus.size()[0]
        y_binary = make_binary_tensor(taus, len(self.max_tau), batch_size)
        h = torch.cat((
            obs,
            ptu.Variable(y_binary),
        ), dim=1)

        return super().forward(
            obs=h,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            return_entropy=return_entropy,
            return_log_prob_of_mean=return_log_prob_of_mean,
        )


class TauVectorTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            max_tau,
            tau_vector_len=0,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        if tau_vector_len == 0:
            self.tau_vector_len = max_tau
        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim + goal_dim + self.tau_vector_len,
            action_dim=action_dim,
            init_w=init_w,
            **kwargs
        )

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False
        ):
        obs, taus = split_tau(obs)
        h=obs
        batch_size = h.size()[0]
        tau_vector = torch.zeros((batch_size, self.tau_vector_len)) + taus.data
        h = torch.cat((
                obs,
                ptu.Variable(tau_vector),
            ), dim=1)

        return super().forward(
            obs=h,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            return_entropy=return_entropy,
            return_log_prob_of_mean=return_log_prob_of_mean,
        )


class TauVectorSeparateFirstLayerTanhGaussianPolicy(SeparateFirstLayerMlp, ExplorationPolicy):

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            max_tau,
            tau_vector_len=0,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        if tau_vector_len == 0:
            self.tau_vector_len = max_tau
        super().__init__(
            hidden_sizes=hidden_sizes,
            first_input_size=obs_dim + goal_dim,
            second_input_size=self.tau_vector_len,
            output_size=action_dim,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        obs, taus = split_tau(obs)
        batch_size = obs.size()[0]
        tau_vector = Variable(torch.zeros((batch_size, self.tau_vector_len)) + taus.data)
        h=obs
        h1 = self.hidden_activation(self.first_input(h))
        h2 = self.hidden_activation(self.second_input(tau_vector))
        h = torch.cat((h1, h2), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                action = tanh_normal.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # Because tanh is invertible, the entropy of a Gaussian and the
            # entropy of the tanh of a Gaussian is the same.
            entropy = entropy.sum(dim=1, keepdim=True)
        if return_log_prob_of_mean:
            tanh_normal = TanhNormal(mean, std)
            mean_action_log_prob = tanh_normal.log_prob(
                torch.tanh(mean),
                pre_tanh_value=mean,
            )
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )