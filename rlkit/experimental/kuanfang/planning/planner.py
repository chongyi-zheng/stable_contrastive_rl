import numpy as np
import torch
import itertools

from rlkit.torch import pytorch_util as ptu
from rlkit.experimental.kuanfang.utils.logging import logger as logging  # NOQA


def l2_distance(h_t, goal_state, start_dim=-3):
    h_t = torch.flatten(h_t, start_dim=start_dim)
    goal_state = torch.flatten(goal_state, start_dim=start_dim)
    return torch.sqrt(torch.sum((goal_state - h_t) ** 2, dim=-1))


def compute_value(vf, plans, init_states, goal_states=None,
                  batched=True, replace_last=True, reverse=False, encoding_type=None):
    # Note: When we use the last step to resemble goal_state, we could feed in
    # the goal_state to replace the last step for computing the value. In this
    # case, the length of the plan should be longer than 1. Otherwise all the
    # values will be computed from (init_states, goal_states), which will be
    # equivalent and meaningless.

    assert vf is not None

    if batched:
        if replace_last:
            plans = plans[:, :-1]

        num_samples = plans.shape[0]
        num_steps = plans.shape[1] + 1
        h0 = torch.cat([init_states[:, None], plans], 1)

        if goal_states is None:
            assert not replace_last
            h1 = plans
        else:
            # Replace the last step with goal_state.
            assert plans.shape[1] > 1, 'The plan should be longer than 1 step.'
            h1 = torch.cat([plans, goal_states[:, None]], 1)

    else:
        if replace_last:
            plans = plans[:-1]

        num_steps = plans.shape[0] + 1
        h0 = torch.cat([init_states[None], plans], 0)

        if goal_states is None:
            assert not replace_last
            h1 = plans
        else:
            # Replace the last step with goal_state.
            assert plans.shape[0] > 1, 'The plan should be longer than 1 step.'
            h1 = torch.cat([plans[:-1], goal_states[None]], 1)

    if encoding_type == 'vqvae':
        B, T, C, H, W = h0.shape
        h0 = h0.view(B, T, -1)
        h1 = h1.view(B, T, -1)

    if reverse:
        vf_inputs = torch.cat([
            h1.view(-1, h1.shape[-1]),
            h0.view(-1, h0.shape[-1]),
        ], 1)
    else:
        vf_inputs = torch.cat([
            h0.view(-1, h0.shape[-1]),
            h1.view(-1, h1.shape[-1]),
        ], 1)

    values = vf(vf_inputs).detach()

    if batched:
        values = values.view(num_samples, num_steps)
    else:
        values = values.view(num_steps)

    return values


def compute_q_value(qf, plans, init_states, goal_states=None,
                    batched=True, replace_last=True, encoding_type=None):
    # Note: When we use the last step to resemble goal_state, we could feed in
    # the goal_state to replace the last step for computing the value. In this
    # case, the length of the plan should be longer than 1. Otherwise all the
    # values will be computed from (init_states, goal_states), which will be
    # equivalent and meaningless.

    assert qf is not None

    if batched:
        if replace_last:
            plans = plans[:, :-1]

        num_samples = plans.shape[0]
        num_steps = plans.shape[1] + 1
        h0 = torch.cat([init_states[:, None], plans], 1)

        if goal_states is None:
            assert not replace_last
            h1 = plans
        else:
            # Replace the last step with goal_state.
            assert plans.shape[1] > 1, 'The plan should be longer than 1 step.'
            h1 = torch.cat([plans, goal_states[:, None]], 1)

    else:
        if replace_last:
            plans = plans[:-1]

        num_steps = plans.shape[0] + 1
        h0 = torch.cat([init_states[None], plans], 0)

        if goal_states is None:
            assert not replace_last
            h1 = plans
        else:
            # Replace the last step with goal_state.
            assert plans.shape[0] > 1, 'The plan should be longer than 1 step.'
            h1 = torch.cat([plans[:-1], goal_states[None]], 1)

    if encoding_type == 'vqvae':
        B, T, C, H, W = h0.shape
        h0 = h0.view(B, T, -1)
        h1 = h1.view(B, T, -1)

    qf_inputs = torch.cat([
        h0.view(-1, h0.shape[-1]),
        h1.view(-1, h1.shape[-1]),
    ], 1)

    batch_size = qf_inputs.shape[0]
    num_action_samples = 128
    min_action = -1.
    max_action = 1.
    actions = min_action + (max_action - min_action) * torch.rand(
        num_action_samples, batch_size, 5).to(ptu.device)

    _qf_inputs = qf_inputs[None].repeat((num_action_samples, 1, 1))
    values = qf(_qf_inputs.view(num_action_samples * batch_size, -1),
                actions.view(num_action_samples * batch_size, -1)
                ).detach()
    values = values.view(num_action_samples, batch_size)

    values = values.mean(dim=0)

    if batched:
        values = values.view(num_samples, num_steps)
    else:
        values = values.view(num_steps)

    return values


def compute_value_v2(vf, plans, init_states, goal_states=None, batched=True):
    assert vf is not None
    if batched:
        num_samples = plans.shape[0]
        num_steps = plans.shape[1]
        h0 = torch.cat([init_states[:, None]], 1)
        h1 = torch.cat([plans[:, 0:1]], 1)
    else:
        num_steps = plans.shape[0]
        h0 = torch.cat([init_states[None], plans[:-1]], 0)
        h1 = plans

    vf_inputs = torch.cat([
        h0.view(-1, embedding_dim*12*12),
        h1.view(-1, embedding_dim*12*12),
    ], 1)
    values = vf(vf_inputs).detach()
    print('values: ', values)
    print('avg_value: %.2f, max_value: %.2f, min_value: %.2f'
          % (
              ptu.get_numpy(torch.mean(values)),
              ptu.get_numpy(torch.max(values)),
              ptu.get_numpy(torch.min(values))
          ))

    if batched:
        values = values.view(num_samples, num_steps)
    else:
        values = values.view(num_steps)

    return values


def preprocess(vqvae, h):
    h = h.view(
        -1,
        vqvae.embedding_dim,
        vqvae.root_len,
        vqvae.root_len)
    return h


def encode(vqvae, init_obs, goal_obs):
    init_state = vqvae.encode(init_obs[None, ...], flatten=False)[0]
    goal_state = vqvae.encode(goal_obs[None, ...], flatten=False)[0]
    return init_state, goal_state


def decode(vqvae, h):
    if isinstance(h, list):
        h = torch.stack(h, 1)

    outer_shape = list(h.shape)[:-3]
    h = h.view(
        -1,
        vqvae.embedding_dim,
        vqvae.root_len,
        vqvae.root_len)
    s = vqvae.decode(h)

    s_shape = outer_shape + list(s.shape[-3:])
    s = s.view(*s_shape)

    return s


def select(zs, plans, costs, to_list=False):
    # Rank the plans.
    if len(costs.shape) == 1:
        costs = costs[:, None]
    min_costs, min_steps = torch.min(costs, 1)
    top_cost, top_ind = torch.min(min_costs, 0)

    top_zs = zs[top_ind, :]
    top_plan = plans[top_ind, :]

    # Optional: Prevent the random actions after achieving the goal.
    top_step = min_steps[top_ind]
    top_plan[top_step:] = top_plan[top_step:top_step + 1]

    if to_list:
        top_plan = list(torch.unbind(top_plan, 0))

    info = {
        'top_ind': top_ind,
        'top_step': top_step,
        'top_cost': top_cost,
    }

    return top_zs, top_plan, info


def nms(data, scores, num_elites, dist_thresh, stepwise_nms=False):
    num_samples = data.shape[0]
    assert num_samples >= num_elites
    if stepwise_nms:
        num_steps = data.shape[1]

    if scores is None:
        # top_indices = torch.zeros((num_samples,), dtype=torch.int64)
        top_indices = torch.arange(0, num_samples, dtype=torch.int64)
    else:
        _, top_indices = torch.topk(scores, num_samples)

    chosen_inds = torch.zeros((num_elites,), dtype=torch.int64)

    valids = torch.ones((num_samples, ), dtype=torch.float32).to(ptu.device)

    num_chosen = 0
    for i_top in range(num_samples):
        if num_chosen >= num_elites:
            break

        this_ind = top_indices[i_top]

        if valids[this_ind] == 0:
            continue

        chosen_inds[num_chosen] = this_ind
        num_chosen += 1

        diffs = data[this_ind][None, ...] - data

        if stepwise_nms:
            diffs = diffs.view(num_samples, num_steps, -1)
            dists = torch.norm(diffs, dim=-1)
        else:
            diffs = diffs.view(num_samples, -1)
            dists = torch.norm(diffs, dim=-1)
            valids = torch.where((dists >= dist_thresh).to(torch.uint8),
                                 valids,
                                 torch.zeros_like(valids))

    return chosen_inds


class Planner(object):

    def __init__(
            self,
            model,
            predict_mode='affordance',
            cost_mode='l2',
            # cost_mode='l2_vf',
            # cost_mode='l2_s',
            # cost_mode='vf',
            # cost_mode='classifier',
            encoding_type=None,
            debug=False,

            initial_collect_episodes=32,
            buffer_size=0,
            max_steps=8,

            prior_weight=0.0,
            values_weight=0.0,

            **kwargs):

        self.encoding_type = encoding_type

        if 'vqvae' in model:
            self.vqvae = model['vqvae']
            # self.encoding_type = 'vqvae'
        else:
            self.vqvae = None

        if 'obs_encoder' in model:
            self.obs_encoder = model['obs_encoder']
        else:
            self.obs_encoder = None

        if 'affordance' in model:
            self.affordance = model['affordance']
        else:
            self.affordance = None

        if 'classifier' in model:
            self.classifier = model['classifier']
        else:
            self.classifier = None

        if 'plan_vf' in model or 'vf' in model:
            self._vf = model['vf']
        else:
            self._vf = None

        if 'qf1' in model:
            self._qf1 = model['qf1']
        else:
            self._qf1 = None

        if 'qf2' in model:
            self._qf2 = model['qf2']
        else:
            self._qf2 = None

        self._predict_mode = predict_mode
        self._cost_mode = cost_mode
        self._debug = debug

        self._max_steps = max_steps
        self._buffer_size = buffer_size
        self._initial_collect_episodes = initial_collect_episodes

        self.sub_planners = []

        self.prior_weight = prior_weight
        self.values_weight = values_weight

        self._buffer = None
        self._buffer_head = 0

        if self._buffer_size > 0:
            if self.affordance is None:
                representation_size = 8
            else:
                try:
                    representation_size = self.affordance.representation_size
                except Exception:
                    representation_size = (
                        self.affordance.networks[0].representation_size)

            self._buffer = np.ones(
                (self._buffer_size,
                 self._max_steps,
                 representation_size),
                dtype=np.float32)

    @property
    def debug(self):
        return self._debug

    @property
    def vf(self):
        return self._vf

    @vf.setter
    def vf(self, value):
        self._vf = value

    @property
    def qf1(self):
        return self._qf1

    @qf1.setter
    def qf1(self, value):
        self._qf1 = value
        for sub_planner in self.sub_planners:
            sub_planner.qf1 = value

    @property
    def qf2(self):
        return self._qf2

    @qf2.setter
    def qf2(self, value):
        self._qf2 = value
        for sub_planner in self.sub_planners:
            sub_planner.qf2 = value

    @property
    def buffer_head(self):
        return self._buffer_head

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def initial_collect_episodes(self):
        return self._initial_collect_episodes

    def _sample_z(self, num_samples, num_steps):

        if (np.random.rand() < 0.1 or
                self._buffer_head < self._initial_collect_episodes):
            self._frac_buffer = 0.0
        else:
            self._frac_buffer = 0.5

        num_buffer_samples = min(int(self._frac_buffer * num_samples),
                                 self._buffer_head)
        num_prior_samples = num_samples - num_buffer_samples

        z1 = self._sample_prior(num_prior_samples, num_steps)
        if num_buffer_samples > 0:
            z2 = self._sample_buffer(num_buffer_samples, num_steps)
            z = torch.cat([z1, z2], 0)
        else:
            z = z1

        return z

    def _sample_prior(self, num_samples, num_steps):
        z = [self.affordance.sample_prior(num_samples)
             for t in range(num_steps)]
        z = np.stack(z, 1)
        z = ptu.from_numpy(z)
        return z

    def _sample_buffer(self, num_samples, num_steps):
        assert num_steps <= self._max_steps

        sampled_inds = np.random.choice(
            self._buffer_head,
            num_samples,
            replace=False)

        z = self._buffer[sampled_inds, :num_steps]
        z = ptu.from_numpy(z)
        return z

    def _add_to_buffer(self, z):
        num_steps = z.shape[0]
        self._buffer[self._buffer_head, :num_steps] = z
        self._buffer_head = (self._buffer_head + 1) % self._buffer_size
        return

    @torch.no_grad()
    def _predict(self, z, init_state=None, goal_state=None):
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

                h_preds.append(h_pred)
                h_t = h_pred

            plans = torch.stack(h_preds, 1)

        elif self._predict_mode == 'leap':
            raise NotImplementedError

        elif self._predict_mode == 'gcp':
            assert self.vqvae is not None
            assert self.affordance is not None

            num_steps = z.shape[1]
            num_levels = int(np.log2(num_steps))

            ts = [0, num_steps]
            t_to_h = {
                0: init_state,
                num_steps: goal_state,
            }
            for _ in range(num_levels):
                ts_copy = ts.copy()
                for t_init, t_goal in pairwise(ts_copy):
                    t_target = int((t_init + t_goal) // 2)
                    z_t = z[:, t_target]
                    h0 = torch.cat((t_to_h[t_init], t_to_h[t_goal]), dim=1)

                    h_pred = self.affordance.decode(z_t, cond=h0)
                    if self.encoding_type == 'vqvae':
                        _, h_pred = self.vqvae.vector_quantizer(h_pred)

                    ts.append(t_target)
                    t_to_h[t_target] = h_pred
                ts.sort()

            ts = ts[1:-1]
            plans = [t_to_h[t] for t in ts]
            plans = torch.stack(plans, 1)
        else:
            raise ValueError('Unrecognized prediction mode: %s'
                             % (self._predict_mode))

        return plans

    def _compute_costs(self, plans, init_states, goal_states, zs=None):

        if self.encoding_type in ['vqvae']:
            start_dim = -3
        else:
            start_dim = -1

        if self._cost_mode is None or self._cost_mode == 'uniform':
            costs = l2_distance(plans[:, -1], goal_states, start_dim)
            costs = torch.ones_like(costs)

        elif self._cost_mode == 'cos':
            costs = -torch.nn.functional.cosine_similarity(
                plans[:, -1], goal_states, -1)

        elif self._cost_mode == 'l2':
            costs = l2_distance(plans[:, -1], goal_states, start_dim)

        elif self._cost_mode == 'l2_vf_ptp':
            l2_dists = l2_distance(plans[:, -1], goal_states, start_dim)
            values = compute_value(self.vf, plans, init_states, goal_states,
                                   replace_last=False, encoding_type=self.encoding_type)
            total_values = values.sum(dim=1)
            if zs is not None:
                z_costs = torch.sum(torch.norm(zs, dim=-1) ** 2, -1)
            else:
                z_costs = ptu.zeros(*l2_dists.shape)
            costs = l2_dists - self.values_weight * \
                total_values + self.prior_weight * z_costs

        elif self._cost_mode == 'l2_vf':
            l2_dists_1 = l2_distance(plans[:, -1], goal_states, start_dim)

            thresh = -15.0
            values = compute_value(self.vf, plans, init_states, goal_states,
                                   replace_last=False, encoding_type=self.encoding_type)
            overage_1 = torch.clamp(values[..., 0:1] - thresh, min=0.0)
            overage_2 = torch.clamp(values[..., -2:-1] - thresh, min=0.0)

            thresh = -30.0
            overage_3 = torch.clamp(values[..., 0:1] - thresh, max=0.0)
            overage_4 = torch.clamp(values[..., -2:-1] - thresh, max=0.0)

            q_values_1 = compute_q_value(
                self.qf1, plans, init_states, goal_states, replace_last=False, encoding_type=self.encoding_type)
            q_values_2 = compute_q_value(
                self.qf2, plans, init_states, goal_states, replace_last=False, encoding_type=self.encoding_type)
            q_values = torch.min(q_values_1, q_values_2)
            advs = values - q_values
            advs = advs[..., :-1]

            costs = (
                1. * l2_dists_1
                - values[..., -1]
                + 1. * (torch.sum(overage_1, 1) + torch.sum(overage_2, 1))
                - 1. * (torch.sum(overage_3, 1) + torch.sum(overage_4, 1))
            )

            z_costs = torch.sum(torch.norm(zs, dim=-1) ** 2, -1)
            costs += 0.1 * z_costs

        elif self._cost_mode in [
                'ablation_clipvalue1_logpu1',
                'ablation_clipvalue0_logpu0',
                'ablation_clipvalue1_logpu0',
                'ablation_clipvalue0_logpu1',
        ]:
            l2_dists_1 = l2_distance(plans[:, -1], goal_states, start_dim)

            thresh = -15.0
            values = compute_value(self.vf, plans, init_states, goal_states,
                                   replace_last=False)
            overage_1 = torch.clamp(values[..., 0:1] - thresh, min=0.0)
            overage_2 = torch.clamp(values[..., -2:-1] - thresh, min=0.0)

            thresh = -30.0
            overage_3 = torch.clamp(values[..., 0:1] - thresh, max=0.0)
            overage_4 = torch.clamp(values[..., -2:-1] - thresh, max=0.0)

            q_values_1 = compute_q_value(
                self.qf1, plans, init_states, goal_states, replace_last=False)
            q_values_2 = compute_q_value(
                self.qf2, plans, init_states, goal_states, replace_last=False)
            q_values = torch.min(q_values_1, q_values_2)
            advs = values - q_values
            advs = advs[..., :-1]

            costs = 1. * l2_dists_1

            if self._cost_mode in [
                    'ablation_clipvalue1_logpu1',
                    'ablation_clipvalue1_logpu0',
            ]:
                costs += 1. * (torch.sum(overage_1, 1) +
                               torch.sum(overage_2, 1))
                costs -= 1. * (torch.sum(overage_3, 1) +
                               torch.sum(overage_4, 1))

            if self._cost_mode in [
                    'ablation_clipvalue1_logpu1',
                    'ablation_clipvalue0_logpu1',
            ]:
                costs += 0.1 * torch.sum(torch.norm(zs, dim=-1) ** 2, -1)

        elif self._cost_mode == 'l2_vf_clamp':
            l2_dists = l2_distance(plans[:, -1], goal_states, start_dim)
            values = compute_value(self.vf, plans, init_states, goal_states)
            thresh = -15
            overage = torch.clamp(values - thresh, max=0.0)
            total_values = overage.sum(dim=1)
            costs = l2_dists - 0.1 * total_values

        elif self._cost_mode == 'l2_vf_mask':
            costs = l2_distance(plans[:, -1], goal_states, start_dim)
            values = compute_value(self.vf, plans, init_states, goal_states)
            masks = (values < -350).any(dim=1)
            costs[masks] = 1e5

        elif self._cost_mode == 'l2_minvf':
            l2_dists = l2_distance(plans[:, -1], goal_states, start_dim)

            thresh = -3.0
            values = compute_value(self.vf, plans, init_states, goal_states)
            # overage = torch.abs(values[..., 0:1] - thresh)
            overage = torch.clamp(values[..., 0:1] - thresh, min=0.0)
            costs = l2_dists + torch.sum(overage, 1)

        elif self._cost_mode == 'vf':
            values = compute_value(self.vf, plans, init_states, goal_states)
            costs = -torch.sum(values, 1)

        elif self._cost_mode == 'vf_over15':
            thresh = -15.0
            values = compute_value(self.vf, plans, init_states, goal_states)
            overage = torch.clamp(values - thresh, max=0.0)
            costs = -torch.sum(overage, 1)

        elif self._cost_mode == 'vf_abs15':
            thresh = -15.0
            values = compute_value(self.vf, plans, init_states, goal_states)
            overage = torch.abs(values - thresh)
            costs = torch.sum(overage, 1)

        elif self._cost_mode == 'vf_overage':
            l2_dists = l2_distance(plans[:, -1], goal_states, start_dim)

            l2_dists_1 = l2_distance(plans[:, :-1], init_states, start_dim)
            l2_dists_2 = l2_distance(plans[:, :-1], goal_states, start_dim)

            l2_thresh = 5.0
            l2_dists_1 = torch.clamp(l2_dists_1 - l2_thresh, max=0.0)
            l2_dists_2 = torch.clamp(l2_dists_2 - l2_thresh, max=0.0)

            values = compute_value(self.vf, plans, init_states, goal_states,
                                   replace_last=False)
            values_r = compute_value(self.vf, plans, init_states, goal_states,
                                     reverse=True,
                                     replace_last=False)

            thresh = -15.0
            overage_0 = torch.clamp(values[..., 0:-1] - thresh, max=0.0)

            thresh = -5.0
            overage_1 = torch.clamp(values[..., 0:-1] - thresh, min=0.0)

            thresh = -15.0
            overage_2 = torch.clamp(values_r[..., 0:-1] - thresh, max=0.0)
            overage_2 *= 0.

            thresh = -5.0
            overage_3 = torch.clamp(values_r[..., 0:-1] - thresh, min=0.0)

            costs = (
                1. * l2_dists

                - 1. * torch.sum(overage_0, 1)
                + 1. * torch.sum(overage_1, 1)
                - 1. * torch.sum(overage_2, 1)
                + 1. * torch.sum(overage_3, 1)

                - 1. * torch.sum(l2_dists_1, 1)
                - 1. * torch.sum(l2_dists_2, 1)
            )

        else:
            raise ValueError

        return costs

    def _process_state(self, state):
        if self.encoding_type == 'vqvae':
            state_shape = list(state.shape[:-3]) + [
                self.vqvae.embedding_dim,
                self.vqvae.root_len,
                self.vqvae.root_len
            ]
            return state.view(*state_shape)
        # elif self.encoding_type == 'vib':
        else:
            return state

    def __call__(self, init_state, goal_state, num_steps, input_info=None):
        init_state = self._process_state(init_state)
        goal_state = self._process_state(goal_state)

        plan, info = self._plan(
            init_state, goal_state, num_steps, input_info)

        if self._buffer_size > 0:
            self._add_to_buffer(ptu.get_numpy(info['top_z']))

        if self._debug:
            if self.vf is not None:
                num_steps = plan.shape[0]
                h0 = torch.cat([init_state[None], plan[:-1]], 0)
                h1 = torch.cat([plan[:-1], goal_state[None]], 0)
                if self.encoding_type == 'vqvae':
                    h0 = h0.view(-1, 720)
                    h1 = h1.view(-1, 720)
                vf_inputs = torch.cat([h0, h1, ], 1)
                values = self.vf(vf_inputs)
                values = values.view(num_steps)
                info['values'] = ptu.get_numpy(values)

        return plan, info

    def _plan(self, init_state, goal_state, num_steps, input_info=None):
        raise NotImplementedError

    def clear(self):
        pass

    def update(self, state):
        pass


class HierarchicalPlanner(Planner):

    def __init__(
            self,
            model,
            sub_planner_ctor,
            multiplier=2,
            num_levels=3,
            min_dt=15,
            **kwargs):

        super().__init__(
            model=model,
            **kwargs)

        self.multiplier = multiplier
        self.num_levels = num_levels

        max_dt = min_dt * (
            self.multiplier ** (self.num_levels - 1))
        self.dts = [max_dt / (self.multiplier ** k)
                    for k in range(self.num_levels)]

        if self.debug:
            print('========================================')
            print('Initialize')
            print('========================================')

        self.sub_planners = []
        for level, dt in enumerate(self.dts):
            sub_model = {}
            for key, value in model.items():
                if key == 'affordance':
                    sub_model[key] = value.networks[level]
                else:
                    sub_model[key] = value

            logging.info('Adding level %d sub_planner of dt %d ...', level, dt)

            sub_planner = sub_planner_ctor(
                sub_model,
                **kwargs)
            self.sub_planners.append(sub_planner)

        self._buffer_size = 0

    @property
    def vf(self):
        return self._vf

    @vf.setter
    def vf(self, value):
        self._vf = value
        for sub_planner in self.sub_planners:
            sub_planner.vf = value

    def _recursive_planning(self,
                            init_state,
                            goal_state,
                            num_steps,
                            level,
                            input_info=None):
        if self.debug:
            print('\t' * level + '========================================')
            print('\t' * level + '| level: %d, num_steps: %d'
                  % (level, num_steps))
            print('\t' * level + '----------------------------------------')

        assert num_steps >= self.multiplier
        debug_info = {'indent': level}
        up_plan, up_info = self.sub_planners[level](
            init_state, goal_state, num_steps, debug_info)

        # Replace the last state in the plan with the goal state.
        down_goal_states = list(torch.unbind(up_plan, 0))[:-1] + [goal_state]

        if self.debug:
            print('\t' * level + '========================================\r')

        # Resursive calls.
        if level >= self.num_levels - 1:
            output_plan = up_plan
            output_info = up_info

        else:
            output_plan = []
            down_plan_list = []
            down_info_list = []

            down_init_state = init_state
            for down_goal_state in down_goal_states:
                down_plan, down_info = self._recursive_planning(
                    init_state=down_init_state,
                    goal_state=down_goal_state,
                    num_steps=self.multiplier,
                    level=level + 1,
                    input_info=up_info)

                # Add the low-level plan to the plan of this level.
                down_plan = list(torch.unbind(down_plan, 0))
                output_plan.extend(down_plan[:-1] + [down_goal_state])

                if self.debug:
                    down_plan_list.append(down_info['up_plan'])
                    down_info_list.append(down_info)

                # Next segment.
                down_init_state = down_goal_state

            output_plan = torch.stack(output_plan, 0)
            output_info = {
                'top_step': down_info['top_step'],
                'top_cost': down_info['top_cost'],

                'up_cost': up_info['top_cost'],
                'down_cost': down_info['top_cost'],
                'down_goal_states': down_goal_states,
            }

            if self.debug:
                output_info['down_plan'] = down_plan_list
                output_info['down_info'] = down_info_list

        if self.debug:
            output_info['up_plan'] = (
                [init_state] + list(torch.unbind(up_plan, 0)) + [goal_state])

        return output_plan, output_info

    def _plan(self, init_state, goal_state, num_steps, input_info=None):
        max_multiplier = self.multiplier ** (self.num_levels - 1)
        assert num_steps % max_multiplier == 0
        divided_num_steps = int(num_steps / max_multiplier)
        plan, info = self._recursive_planning(
            init_state=init_state,
            goal_state=goal_state,
            num_steps=divided_num_steps,
            level=0)
        return plan, info


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
