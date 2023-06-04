"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import ConcatMlp
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks.basic import Concat, MultiInputSequential
from rlkit.torch.networks.mlp import ParallelMlp


class DisentangledMlpQf(PyTorchModule):

    def __init__(
            self,
            goal_encoder,
            state_encoder,
            qf_kwargs,
            preprocess_obs_dim,
            action_dim,
            vectorized=False,
            architecture='splice',
            detach_encoder_via_goal=False,
            detach_encoder_via_state=False,
            num_heads=None,
    ):
        """

        :param encoder:
        :param qf_kwargs:
        :param preprocess_obs_dim:
        :param action_dim:
        :param vectorized:
        :param architecture:
         - 'splice': give each Q function a single index into the latent goal
         - 'many_heads': give each Q function the entire latent goal
         - 'single_head': have one Q function that takes in entire latent goal
         - 'huge_single_head': single head but that multiplies the hidden sizes
             by the number of heads in `many_heads`
         - 'single_head_match_many_heads': single head but that multiplies the
             first hidden size by the number of heads in `many_heads`
        """
        super().__init__()
        self.goal_encoder = goal_encoder
        self.state_encoder = state_encoder
        self.preprocess_obs_dim = preprocess_obs_dim
        self.preprocess_goal_dim = goal_encoder.input_size
        self.postprocess_goal_dim = goal_encoder.output_size
        self.vectorized = vectorized
        self._architecture = architecture
        self._detach_encoder_via_goal = detach_encoder_via_goal
        self._detach_encoder_via_state = detach_encoder_via_state

        # We have a qf for each goal dim, described by qf_kwargs.
        self.feature_qfs = nn.ModuleList()
        if architecture == 'splice':
            qf_goal_input_size = 1
        else:
            qf_goal_input_size = self.postprocess_goal_dim
        qf_input_size = (
                state_encoder.output_size + action_dim + qf_goal_input_size
        )
        if architecture == 'single_head':
            self.feature_qfs.append(ConcatMlp(
                input_size=qf_input_size,
                output_size=1,
                **qf_kwargs
            ))
        elif architecture == 'huge_single_head':
            new_qf_kwargs = qf_kwargs.copy()
            hidden_sizes = new_qf_kwargs.pop('hidden_sizes')
            new_hidden_sizes = [
                size * self.postprocess_goal_dim for size in hidden_sizes
            ]
            self.feature_qfs.append(ConcatMlp(
                hidden_sizes=new_hidden_sizes,
                input_size=qf_input_size,
                output_size=1,
                **new_qf_kwargs
            ))
        elif architecture == 'single_head_match_many_heads':
            new_qf_kwargs = qf_kwargs.copy()
            hidden_sizes = new_qf_kwargs.pop('hidden_sizes')
            new_hidden_sizes = [
                hidden_sizes[0] * self.postprocess_goal_dim
            ] + hidden_sizes[1:]
            self.feature_qfs.append(ConcatMlp(
                hidden_sizes=new_hidden_sizes,
                input_size=qf_input_size,
                output_size=1,
                **new_qf_kwargs
            ))
        elif architecture in {'many_heads', 'splice'}:
            if num_heads is None:
                num_heads = self.postprocess_goal_dim
            for _ in range(num_heads):
                self.feature_qfs.append(ConcatMlp(
                    input_size=qf_input_size,
                    output_size=1,
                    **qf_kwargs
                ))
        else:
            raise ValueError(architecture)

    def forward(self, obs, actions, return_individual_q_vals=False, **kwargs):
        obs_and_goal = obs
        # TODO: undo hack. probably just get rid of these variables
        if self.preprocess_obs_dim == self.preprocess_goal_dim:
            obs, goal = obs_and_goal.chunk(2, dim=1)
        else:
            assert obs_and_goal.shape[1] == (
                    self.preprocess_obs_dim + self.preprocess_goal_dim)
            obs = obs_and_goal[:, :self.preprocess_obs_dim]
            goal = obs_and_goal[:, self.preprocess_obs_dim:]

        h_obs = self.state_encoder(obs)
        h_obs = h_obs.detach() if self._detach_encoder_via_state else h_obs
        h_goal = self.goal_encoder(goal)
        h_goal = h_goal.detach() if self._detach_encoder_via_goal else h_goal

        total_q_value = 0
        individual_q_vals = []
        for goal_dim_idx, feature_qf in enumerate(self.feature_qfs):
            if self._architecture == 'splice':
                flat_inputs = torch.cat((
                    h_obs,
                    h_goal[:, goal_dim_idx:goal_dim_idx+1],
                    actions
                ), dim=1)
            else:
                flat_inputs = torch.cat((
                    h_obs,
                    h_goal,
                    actions
                ), dim=1)
            q_idx_value = feature_qf(flat_inputs)
            total_q_value += q_idx_value
            individual_q_vals.append(q_idx_value)

        if self.vectorized:
            total_q_value = torch.cat(individual_q_vals, dim=1)

        if return_individual_q_vals:
            return total_q_value, individual_q_vals
        else:
            return total_q_value


class ParallelDisentangledMlpQf(PyTorchModule):

    def __init__(
            self,
            goal_encoder,
            state_encoder,
            post_encoder_mlp_kwargs,
            preprocess_obs_dim,
            action_dim,
            vectorized=False,
            architecture='splice',
            detach_encoder_via_goal=False,
            detach_encoder_via_state=False,
            num_heads=None,
    ):
        """

        :param encoder:
        :param post_encoder_mlp_kwargs:
        :param preprocess_obs_dim:
        :param action_dim:
        :param vectorized:
        :param architecture:
         - 'splice': give each Q function a single index into the latent goal
         - 'many_heads': give each Q function the entire latent goal
         - 'single_head': have one Q function that takes in entire latent goal
         - 'huge_single_head': single head but that multiplies the hidden sizes
             by the number of heads in `many_heads`
         - 'single_head_match_many_heads': single head but that multiplies the
             first hidden size by the number of heads in `many_heads`
        """
        super().__init__()
        self.goal_encoder = goal_encoder
        self.state_encoder = state_encoder
        self.preprocess_obs_dim = preprocess_obs_dim
        self.preprocess_goal_dim = goal_encoder.input_size
        self.postprocess_goal_dim = goal_encoder.output_size
        self.vectorized = vectorized
        self._architecture = architecture
        self._detach_encoder_via_goal = detach_encoder_via_goal
        self._detach_encoder_via_state = detach_encoder_via_state

        if architecture == 'splice':
            qf_goal_input_size = 1
        else:
            qf_goal_input_size = self.postprocess_goal_dim
        qf_input_size = (
                state_encoder.output_size + action_dim + qf_goal_input_size
        )
        if architecture == 'single_head':
            self.post_encoder_qf = ConcatMlp(
                input_size=qf_input_size,
                output_size=1,
                **post_encoder_mlp_kwargs
            )
        elif architecture == 'huge_single_head':
            new_qf_kwargs = post_encoder_mlp_kwargs.copy()
            hidden_sizes = new_qf_kwargs.pop('hidden_sizes')
            new_hidden_sizes = [
                size * self.postprocess_goal_dim for size in hidden_sizes
            ]
            self.post_encoder_qf = ConcatMlp(
                hidden_sizes=new_hidden_sizes,
                input_size=qf_input_size,
                output_size=1,
                **new_qf_kwargs
            )
        elif architecture == 'single_head_match_many_heads':
            new_qf_kwargs = post_encoder_mlp_kwargs.copy()
            hidden_sizes = new_qf_kwargs.pop('hidden_sizes')
            new_hidden_sizes = [
                                   hidden_sizes[0] * self.postprocess_goal_dim
                               ] + hidden_sizes[1:]
            self.post_encoder_qf = ConcatMlp(
                hidden_sizes=new_hidden_sizes,
                input_size=qf_input_size,
                output_size=1,
                **new_qf_kwargs
            )
        elif architecture == 'many_heads':
            num_heads = num_heads or self.postprocess_goal_dim
            self.post_encoder_qf = MultiInputSequential(
                Concat(),
                ParallelMlp(
                    num_heads=num_heads,
                    input_size=qf_input_size,
                    output_size_per_mlp=1,
                    **post_encoder_mlp_kwargs
                ),
            )
        elif architecture == 'splice':
            self.post_encoder_qf = MultiInputSequential(
                Concat(),
                ParallelMlp(
                    num_heads=self.postprocess_goal_dim,
                    input_size=qf_input_size,
                    output_size_per_mlp=1,
                    input_is_already_expanded=True,
                    **post_encoder_mlp_kwargs
                ),
            )
        else:
            raise ValueError(architecture)

    def forward(self, obs_and_goal, actions, return_individual_q_vals=False, **kwargs):
        obs, goal = obs_and_goal.chunk(2, dim=1)

        h_obs = self.state_encoder(obs)
        h_obs = h_obs.detach() if self._detach_encoder_via_state else h_obs
        h_goal = self.goal_encoder(goal)
        h_goal = h_goal.detach() if self._detach_encoder_via_goal else h_goal

        if self._architecture == 'splice':
            h_obs_expanded = h_obs.repeat(1, self.postprocess_goal_dim).unsqueeze(-1)
            actions = actions.repeat(1, self.postprocess_goal_dim).unsqueeze(-1)
            expanded_inputs = torch.cat((
                h_obs_expanded,
                goal.unsqueeze(1),
                actions,
            ), dim=1)
            individual_q_vals = self.post_encoder_qf(expanded_inputs)
        else:
            flat_inputs = torch.cat((h_obs, h_goal, actions), dim=1)
            individual_q_vals = self.post_encoder_qf(flat_inputs).squeeze(1)

        if self.vectorized:
            return individual_q_vals
        else:
            return individual_q_vals.sum(dim=-1)


class QfMaximizingPolicy(Policy):
    def __init__(
        self,
        qf,
        env,
        num_action_samples=300,
    ):
        self.qf = qf
        self.num_action_samples = num_action_samples
        self.action_lows = env.action_space.low
        self.action_highs = env.action_space.high

    def get_action(self, obs):
        opt_actions, info = self.get_actions(obs[None])
        return opt_actions[0], info

    def get_actions(self, obs):
        obs_tiled = np.repeat(obs, self.num_action_samples, axis=0)
        action_tiled = np.random.uniform(
            low=self.action_lows, high=self.action_highs,
            size=(len(obs_tiled), len(self.action_lows))
        )
        obs_tiled = ptu.from_numpy(obs_tiled)
        action_tiled = ptu.from_numpy(action_tiled)
        q_val_torch = self.qf(obs_tiled, action_tiled)

        # In case it's vectorized, we'll take the largest sum
        q_val_torch = q_val_torch.sum(dim=1)

        # q_val_torch[i][j] is the q_val for obs[i] and random action[j] for
        # that obs (specifically, action_tiled[i * self.num_action_samples + j])
        q_val_torch = q_val_torch.view(len(obs), -1)
        opt_action_idxs = q_val_torch.argmax(dim=1)

        # Defined q_val_torch[i] to be the optimal q_val for obs[i],
        # selected by action_tiled
        opt_actions = ptu.get_numpy(action_tiled[opt_action_idxs])
        return opt_actions, {}

    def to(self, device):
        self.qf.to(device)


class EncodeObsAndGoal(PyTorchModule):
    def __init__(
            self,
            encoder,
            state_dim,
            encode_state=True,
            encode_goal=True,
            detach_encoder_via_goal=False,
            detach_encoder_via_state=False,
    ):
        super().__init__()
        encoder_output_dim = encoder.output_size
        output_dim = 0
        output_dim += encoder_output_dim if encode_state else state_dim
        output_dim += encoder_output_dim if encode_goal else state_dim

        self._encoder = encoder
        self._encode_state = encode_state
        self._encode_goal = encode_goal
        self._detach_encoder_via_goal = detach_encoder_via_goal
        self._detach_encoder_via_state = detach_encoder_via_state
        self.output_size = output_dim

    def forward(self, obs_and_goal, *args, **kwargs):
        obs, goal = obs_and_goal.chunk(2, dim=1)

        h_obs = self._encoder(obs) if self._encode_state else obs
        h_obs = h_obs.detach() if self._detach_encoder_via_state else h_obs
        h_goal = self._encoder(goal) if self._encode_goal else goal
        h_goal = h_goal.detach() if self._detach_encoder_via_goal else h_goal

        return h_obs, h_goal


class DDRArchitecture(PyTorchModule):

    def __init__(
            self,
            encoder: EncodeObsAndGoal,
            qf_kwargs,
            preprocess_obs_dim,
            action_dim,
            encode_state=False,
            vectorized=False,
            architecture='splice',
    ):
        """

        :param encoder:
        :param qf_kwargs:
        :param preprocess_obs_dim:
        :param action_dim:
        :param encode_state:
        :param vectorized:
        :param architecture:
         - 'splice': give each Q function a single index into the latent goal
         - 'many_heads': give each Q function the entire latent goal
         - 'single_head': have one Q function that takes in entire latent goal
         - 'huge_single_head': single head but that multiplies the hidden sizes
             by the number of heads in `many_heads`
         - 'single_head_match_many_heads': single head but that multiplies the
             first hidden size by the number of heads in `many_heads`
        """
        super().__init__()
        self.encoder = encoder
        self.preprocess_obs_dim = preprocess_obs_dim
        self.preprocess_goal_dim = encoder.input_size
        self.postprocess_goal_dim = encoder.output_size
        self.encode_state = encode_state
        self.vectorized = vectorized
        self._architecture = architecture

        # We have a qf for each goal dim, described by qf_kwargs.
        self.feature_qfs = nn.ModuleList()
        if architecture == 'splice':
            qf_goal_input_size = 1
        else:
            qf_goal_input_size = self.postprocess_goal_dim
        if self.encode_state:
            qf_input_size = (
                    self.postprocess_goal_dim + action_dim + qf_goal_input_size
            )
        else:
            qf_input_size = preprocess_obs_dim + action_dim + qf_goal_input_size
        if architecture == 'single_head':
            self.feature_qfs.append(ConcatMlp(
                input_size=qf_input_size,
                output_size=1,
                **qf_kwargs
            ))
        elif architecture == 'huge_single_head':
            new_qf_kwargs = qf_kwargs.copy()
            hidden_sizes = new_qf_kwargs.pop('hidden_sizes')
            new_hidden_sizes = [
                size * self.postprocess_goal_dim for size in hidden_sizes
            ]
            self.feature_qfs.append(ConcatMlp(
                hidden_sizes=new_hidden_sizes,
                input_size=qf_input_size,
                output_size=1,
                **new_qf_kwargs
            ))
        elif architecture == 'single_head_match_many_heads':
            new_qf_kwargs = qf_kwargs.copy()
            hidden_sizes = new_qf_kwargs.pop('hidden_sizes')
            new_hidden_sizes = [
                                   hidden_sizes[0] * self.postprocess_goal_dim
                               ] + hidden_sizes[1:]
            self.feature_qfs.append(ConcatMlp(
                hidden_sizes=new_hidden_sizes,
                input_size=qf_input_size,
                output_size=1,
                **new_qf_kwargs
            ))
        else:
            for _ in range(self.postprocess_goal_dim):
                self.feature_qfs.append(ConcatMlp(
                    input_size=qf_input_size,
                    output_size=1,
                    **qf_kwargs
                ))

    def forward(self, obs, actions, return_individual_q_vals=False, **kwargs):
        obs_and_goal = obs
        h_obs, h_goal = self.encoder(obs)

        total_q_value = 0
        individual_q_vals = []
        for goal_dim_idx, feature_qf in enumerate(self.feature_qfs):
            if self._architecture == 'splice':
                flat_inputs = torch.cat((
                    h_obs,
                    h_goal[:, goal_dim_idx:goal_dim_idx+1],
                    actions
                ), dim=1)
            else:
                flat_inputs = torch.cat((
                    h_obs,
                    h_goal,
                    actions
                ), dim=1)
            q_idx_value = feature_qf(flat_inputs)
            total_q_value += q_idx_value
            individual_q_vals.append(q_idx_value)

        if self.vectorized:
            total_q_value = torch.cat(individual_q_vals, dim=1)

        if return_individual_q_vals:
            return total_q_value, individual_q_vals
        else:
            return total_q_value


class VAE(PyTorchModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = self._decoder.input_size

    def encode(self, x):
        return self._encoder(x)

    def encode_mu(self, x):
        mu, logvar = self._encoder(x)
        return mu

    def decode(self, z):
        return self._decoder(z)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def reconstruct(self, x, use_mean=True, return_latent_params=False):
        mu, logvar = self.encode(x)
        z = mu
        if not use_mean:
            z = self.reparameterize(mu, logvar)
        if return_latent_params:
            return self._decoder(z), mu, logvar
        else:
            return self._decoder(z)

    def logprob(self, x, x_recon):
        return -1 * F.mse_loss(
            x_recon,
            x,
            reduction='mean'
        ) * self._encoder.input_size

    def sample_np(self, batch_size):
        latents = np.random.normal(size=(batch_size, self.latent_dim))
        latents_torch = ptu.from_numpy(latents)
        return ptu.get_numpy(self.decode(latents_torch))

    def forward(self, x):
        return self.reconstruct(x)


class EncoderMuFromEncoderDistribution(PyTorchModule):
    """Requires encoder(x) to produce mean and variance of latent distribution
    """
    def __init__(self, encoder):
        super().__init__()
        self._encoder = encoder
        self.input_size = encoder.input_size
        self.output_size = encoder.output_size

    def forward(self, x):
        mu, _ = self._encoder(x)
        return mu
