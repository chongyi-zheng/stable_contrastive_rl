import typing
import numpy as np
from collections import OrderedDict, defaultdict
from numbers import Number

import gym

from rlkit.core.logging import append_log
from rlkit.envs.images import InsertImagesEnv
from rlkit.envs.images.env_renderer import EnvRenderer
from rlkit.envs.images.plot_renderer import TextRenderer, ScrollingPlotRenderer
from rlkit.envs.pearl_envs import HalfCheetahDirEnv
from rlkit.envs.wrappers.flat_to_dict import FlatToDictPolicy
from rlkit.core import eval_util
from rlkit.core.eval_util import create_stats_ordered_dict


def make_named_path_compatible(fn, divider='/'):
    """
    converts a function of type

    def f(paths) -> dictionary

    to

    def f(Dict[str, List]) -> dictionary

    with the dictionary key prefixed

    for example

    ```
    def foo(paths):
        return {
            'num_paths': len(paths)
        }

    paths = [1,2]

    print(foo(paths))
    # prints {'num_paths': 2}

    named_paths = {
        'a': [1,2],
        'b': [1,2],
    }
    new_foo = make_named_path_compatible(foo)

    print(new_foo(paths))
    # prints {'a/num_paths': 2, 'b'/num_paths': 1}
    ```
    """

    def unpacked_fn(named_paths):
        results = OrderedDict()
        for name, paths in named_paths.items():
            new_results = fn(paths)
            append_log(results, new_results, prefix=name, divider=divider)
        return results

    return unpacked_fn


def get_diagnostics(env):
    diagnostics = [
        eval_util.get_generic_path_information,
    ]
    if isinstance(env, HalfCheetahDirEnv):
        diagnostics.append(half_cheetah_dir_diagnostics)
    return [
        make_named_path_compatible(fn) for fn in
        diagnostics
    ]


def half_cheetah_dir_diagnostics(paths):
    statistics = OrderedDict()
    stat_to_lists = defaultdict(list)
    for path in paths:
        for k in ['reward_forward', 'reward_ctrl']:
            stats_for_this_path = []
            for env_info in path['env_infos']:
                stats_for_this_path.append(env_info[k])
            stat_to_lists[k].append(stats_for_this_path)
    for stat_name, stat_list in stat_to_lists.items():
        statistics.update(create_stats_ordered_dict(
            stat_name,
            stat_list,
            always_show_all_stats=True,
        ))
        statistics.update(create_stats_ordered_dict(
            '{}/final'.format(stat_name),
            [s[-1:] for s in stat_list],
            always_show_all_stats=True,
            exclude_max_min=True,
        ))
    return statistics


def task_str(task):
    if isinstance(task, tuple):
        return tuple(task_str(t) for t in task)
    if isinstance(task, str):
        return task
    if isinstance(task, np.ndarray):
        return '{:.2g}'.format(float(task))
    if isinstance(task, Number):
        return '{:.2g}'.format(task)


def format_task(idx, task):
    lines = ['task_idx = {}'.format(idx)]
    if isinstance(task, dict):
        lines.append('task')
        for k, v in task.items():
            # v = (np.cos(v), np.sin(v))
            lines.append('{}: {}'.format(k, task_str(v)))
    else:
        lines.append('task: {}'.format(task))
    return '\n'.join(lines)


class DebugInsertImagesEnv(InsertImagesEnv):
    def __init__(
            self,
            wrapped_env: gym.Env,
            renderers: typing.Dict[str, EnvRenderer],
    ):
        super().__init__(wrapped_env, renderers)
        self._last_reward = None

    def reset_task(self, idx):
        self.wrapped_env.reset_task(idx)
        task = self.wrapped_env.tasks[idx]
        for renderer in self.renderers.values():
            if isinstance(renderer, TextRenderer):
                renderer.set_text(format_task(idx, task))

    def reset(self):
        self._last_reward = None
        for renderer in self.renderers.values():
            if isinstance(renderer, ScrollingPlotRenderer):
                renderer.reset()
        return super().reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._last_reward = reward
        self._update_obs(obs)
        return obs, reward, done, info

    def _update_obs(self, obs):
        for image_key, renderer in self.renderers.items():
            if isinstance(renderer, ScrollingPlotRenderer):
                obs[image_key] = renderer(self._last_reward)
            else:
                obs[image_key] = renderer(self.env)


class FlatToDictPearlPolicy(FlatToDictPolicy):
    def update_context(self, context, inputs):
        o, a, r, no, d, info = inputs
        new_inputs = (
            o[self.observation_key],
            a,
            r,
            no[self.observation_key],
            d,
            info,
        )
        return self._inner.update_context(context, new_inputs)
