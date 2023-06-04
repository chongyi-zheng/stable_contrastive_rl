import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from rlkit.torch import pytorch_util as ptu
from rlkit.experimental.kuanfang.planning.planner import Planner


class MppiPlanner(Planner):

    def __init__(
            self,
            model,
            predict_mode='affordance',

            num_samples=1024,
            num_iters=5,
            temperature=0.1,
            noise_level=[1.0, 0.5, 0.2, 0.1, 0.1],

            representation_size=8,

            replace_final_subgoal=True,

            **kwargs):

        super().__init__(
            model=model,
            predict_mode=predict_mode,
            **kwargs
        )

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
        print('Replace the last final subgoal in MPPI.')

        self.noise_distrib = MultivariateNormal(
            torch.zeros(representation_size).to(ptu.device),
            torch.eye(representation_size).to(ptu.device))

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
            plans = self._predict(sampled_zs, init_states, goal_states)
            costs = self._compute_costs(plans, init_states, goal_states,
                                        zs=sampled_zs)

            # Select.
            top_cost, top_ind = torch.min(costs, 0)
            z = sampled_zs[top_ind]
            plan = plans[top_ind]

            if self._replace_final_subgoal:
                plan = torch.cat([plan[:-1], goal_state[None]], 0)

            all_plan.append(plan)
            all_cost.append(top_cost)

            if self._debug:
                debug_text = (
                    '- top_cost: %.2f, max_cost: %.2f, avg_cost: %.2f'
                    % (costs.min().item(),
                       costs.max().item(),
                       costs.mean().item()))
                if input_info is not None:
                    if 'indent' in input_info:
                        debug_text = input_info['indent'] * '\t' + debug_text
                print(debug_text)

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
