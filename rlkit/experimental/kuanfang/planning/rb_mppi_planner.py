import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from rlkit.torch import pytorch_util as ptu
from rlkit.experimental.kuanfang.planning.planner import Planner


class RbMppiPlanner(Planner):

    def __init__(
            self,
            model,

            num_samples=1024,
            num_iters=5,
            temperature=0.1,
            noise_level=[1.0, 0.5, 0.2, 0.1, 0.1],

            representation_size=8,

            # replay_buffer=None,
            use_states_from_rb=True,
            substitute_prob=0.25,

            update_interval=5,
            max_candidates=10000,
            min_candidates=1,
            l2_dist_thresh=5.0,

            replace_final_subgoal=True,

            **kwargs):

        super().__init__(
            model=model,
            **kwargs)

        assert temperature > 0

        self.num_samples = num_samples
        self.num_iters = num_iters
        self.temperature = float(temperature)
        self.noise_level = noise_level

        if representation_size is None:
            representation_size = self.affordance.representation_size
        else:
            pass

        self._replace_final_subgoal = replace_final_subgoal

        self.noise_distrib = MultivariateNormal(
            torch.zeros(representation_size).to(ptu.device),
            torch.eye(representation_size).to(ptu.device))

        self._use_states_from_rb = use_states_from_rb
        self._substitute_prob = substitute_prob

        self._update_interval = update_interval
        self._update_steps = 0

        self._max_candidates = max_candidates
        self._min_candidates = min_candidates
        self._num_candidates = 0
        self._stored_candidates = None

        self._l2_dist_thresh = l2_dist_thresh

    def _perturb(self, z, noise_level):
        num_steps = int(z.shape[0])
        noise = noise_level * self.noise_distrib.sample(
            (self.num_samples, num_steps))
        perturbed_zs = z[None, ...] + noise
        return perturbed_zs

    def _compute_weights(self, costs, temperature):
        min_costs = torch.min(costs)
        diffs = (costs - min_costs)
        numerators = torch.exp(-diffs / temperature)
        weights = numerators / torch.sum(numerators)
        return weights

    def _plan(self, init_state, goal_state, num_steps, input_info=None):
        all_plan = []
        all_cost = []

        init_states = torch.stack([init_state] * self.num_samples, 0)
        goal_states = torch.stack([goal_state] * self.num_samples, 0)

        z = None

        # Recursive prediction.
        for i in range(self.num_iters + 1):
            if i == 0:
                # Initial samples.
                sampled_zs = self._sample_z(self.num_samples, num_steps)

            else:
                # Perturbed samples.
                if isinstance(self.noise_level, list):
                    noise_level = self.noise_level[i - 1]
                else:
                    noise_level = self.noise_level
                sampled_zs = self._perturb(z, noise_level)
                sampled_zs[0, ...] = z   # Copy the previous best.

            # Predict and evaluate.
            if (i == 0 and self._use_states_from_rb and
                    self._num_candidates > self._min_candidates):
                substitute_state, substitute_mask = (
                    self._sample_substitute_state(
                        self.num_samples, num_steps, init_states, goal_states)
                )
                plans = self._predict(sampled_zs, init_states, goal_states,
                                      substitute_state, substitute_mask)
            else:
                plans = self._predict(sampled_zs, init_states, goal_states)

            costs = self._compute_costs(plans, init_states, goal_states,
                                        zs=sampled_zs)

            # Select.
            top_cost, top_ind = torch.min(costs, 0)
            plan = plans[top_ind]
            if self._use_states_from_rb:
                z = self._encode(plan[None], init_states[0:1])[0]
            else:
                z = sampled_zs[top_ind]

            # Replace the last step.
            if self._replace_final_subgoal:
                plan = torch.cat([plan[:-1], goal_state[None]], 0)

            all_plan.append(plan)
            all_cost.append(top_cost)

            # Update z.
            if i < self.num_iters:
                temperature = self.temperature
                weights = self._compute_weights(costs, temperature)
                z = torch.sum(weights.view(-1, 1, 1) * sampled_zs, dim=0)

        info = {
            'top_step': -1,
            'top_cost': top_cost,
            'top_z': z,

            'all_plan': torch.stack(all_plan, 0),
            'all_cost': torch.stack(all_cost, 0),
        }
        return plan, info

    def _sample_substitute_state(
            self, num_samples, num_steps, init_states, goal_states):
        assert self._num_candidates > 0

        if 0:
            stored_candidates = ptu.from_numpy(self._stored_candidates)

            rand_inds = torch.randint(
                0, self._num_candidates, (num_samples, num_steps))

            substitute_state = stored_candidates[rand_inds].to(
                ptu.device)

            substitute_mask = (
                torch.rand(num_samples, num_steps, 1) < self._substitute_prob
            ).to(ptu.device)
            substitute_mask[..., -1, :] = 0

        else:

            init_states = ptu.get_numpy(init_states[0:1])
            goal_states = ptu.get_numpy(goal_states[0:1])
            stored_candidates = self._stored_candidates[:self._num_candidates]

            # Start with NumPy arrays for make implementation easier.
            substitute_mask = np.zeros(
                (num_samples, num_steps, 1)).astype(np.uint8)

            l2_dists_to_init = np.linalg.norm(
                stored_candidates - init_states, axis=-1)
            l2_dists_to_goal = np.linalg.norm(
                stored_candidates - goal_states, axis=-1)

            cand_inds = np.where(
                (l2_dists_to_init > self._l2_dist_thresh) &
                (l2_dists_to_goal > self._l2_dist_thresh)
            )[0]

            if len(cand_inds) > 0:
                # If there are valid candidates, sample.
                sampled_inds = np.random.choice(
                    cand_inds, (num_samples, num_steps))
                sampled_ts = np.random.choice(
                    num_steps - 1, (num_samples,))
                substitute_state = self._stored_candidates[sampled_inds]
                substitute_mask[np.arange(sampled_ts.shape[0]), sampled_ts] = 1

            substitute_mask = np.where(
                torch.rand(num_samples, num_steps, 1) < self._substitute_prob,
                substitute_mask,
                np.zeros_like(substitute_mask))
            print(substitute_mask[..., 0])

            # Convert back to Torch Tensors.
            substitute_state = ptu.from_numpy(substitute_state).to(ptu.device)
            substitute_mask = ptu.from_numpy(substitute_mask).to(ptu.device)
            substitute_mask = substitute_mask.to(torch.uint8)

        return substitute_state, substitute_mask

    @torch.no_grad()
    def _predict(self, z, init_state=None, goal_state=None,
                 substitute_state=None, substitute_mask=None):
        if self._predict_mode == 'affordance':
            assert self.affordance is not None

            num_steps = z.shape[1]

            h_preds = []
            h_t = init_state
            for t in range(num_steps):
                z_t = z[:, t]
                h_pred = self.affordance.decode(z_t, cond=h_t).detach()

                if self.encoding_type == 'vqvae':
                    _, h_pred = self.vqvae.vector_quantizer(h_pred)

                if substitute_state is not None:
                    assert substitute_mask is not None
                    h_pred = torch.where(
                        substitute_mask[:, t],
                        substitute_state[:, t],
                        h_pred)

                h_preds.append(h_pred)
                h_t = h_pred

            plans = torch.stack(h_preds, 1)

        else:
            raise ValueError('Unrecognized prediction mode: %s'
                             % (self._predict_mode))

        return plans

    @torch.no_grad()
    def _encode(self, plan, init_state=None, goal_state=None):
        if self._predict_mode == 'affordance':
            assert self.affordance is not None

            num_steps = plan.shape[1]

            zs = []
            h_tm1 = init_state
            for t in range(num_steps):
                h_t = plan[:, t]
                z_t, _ = self.affordance.encode(h_t, cond=h_tm1)
                z_t = z_t.detach()

                zs.append(z_t)

            zs = torch.stack(zs, 1)

        else:
            raise ValueError('Unrecognized prediction mode: %s'
                             % (self._predict_mode))

        return zs

    def clear(self):
        raise NotImplementedError

    def update(self, state):
        if self._update_steps > 0:
            self._update_steps -= 1
            return
        elif self._update_steps < 0:
            raise ValueError

        # Initialize.
        if self._num_candidates == 0:
            self._stored_candidates = np.zeros(
                (self._max_candidates, state.shape[-1]),
                dtype=state.dtype)

        # Ignore state that is similar to existing ones.
        l2_dists = np.linalg.norm(
            state[None] - self._stored_candidates, axis=-1)
        if l2_dists.min() <= self._l2_dist_thresh:
            return

        # Update.
        if self._num_candidates >= self._max_candidates:
            replace_ind = np.random.choice(self._max_candidates)
        else:
            replace_ind = self._num_candidates
            self._num_candidates += 1

        self._stored_candidates[replace_ind] = state
        self._update_steps = self._update_interval
