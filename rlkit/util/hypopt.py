"""
Utility functions that wrap the hyperopt library.
"""
import json
import pickle
import time
from os.path import join, exists

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from rlkit.pythonplusplus import (
    dot_map_dict_to_nested_dict,
    merge_recursive_dicts,
    dict_to_safe_json,
)


def optimize_and_save(
        base_log_dir,
        function,
        search_space,
        num_rounds=100,
        num_evals_per_round=1,
        load_trials=False,
        trials_filename="trials.pkl",
        search_space_filename="search_space.pkl",
        non_hp_results_filename="non_hp_results.json",
        verbose=False,
        **kwargs
):
    assert num_rounds > 0
    assert num_evals_per_round > 0

    trials = None
    trials_path = join(base_log_dir, trials_filename)
    if load_trials and exists(trials_path):
        with open(trials_path, 'rb') as handle:
            trials = pickle.load(handle)

    start_time = time.time()
    for round in range(num_rounds):
        num_evals = (round+1) * num_evals_per_round
        best_params, min_value, trials, best_variant = optimize(
            function=function,
            search_space=search_space,
            num_evals=num_evals,
            trials=trials,
            **kwargs
        )
        search_space_path = join(base_log_dir, search_space_filename)
        non_hp_results_path = join(base_log_dir, non_hp_results_filename)
        if verbose:
            print("# evaluations so far:", num_evals)
            print("best_params:", best_params)
            print("min_value:", min_value)
            print("best_variant:", best_variant)
            print("Total time elapsed = {}".format(time.time() - start_time))
            print("Saving all results {0}".format(base_log_dir))
        with open(trials_path, 'wb') as handle:
            pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(search_space_path, 'wb') as handle:
            pickle.dump(search_space, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(non_hp_results_path, "w") as f:
            results = {
                'best_params': dict_to_safe_json(best_params),
                'min_value': min_value,
                'best_variant': dict_to_safe_json(best_variant),
            }
            json.dump(results, f, indent=2, sort_keys=True)


def optimize(
        function,
        search_space,
        extra_function_kwargs=None,
        trials=None,
        num_evals=10,
        maximize=False,
        flatten_choice_dictionary=True,
        dotmap_to_nested_dictionary=True,
        max_magnitude=1e10,
):
    """
    Optimize (by default minimize) a function over a search space.

    :param function: Function to optimize over.
    :param search_space: a hyperopt.pyll.base.Apply instance, or a dictionary
    mapping from keyword to Apply node, as in

    ```
    space = {
        'b': hp.uniform('b', -1, 1),
        'c': hp.uniform('c', 10, 11),
        ...
    }
    ```

    A bit hacky, but the `b` in the dictionary key should match the `b` in
    the argument to the `hp.uniform` call. It doesn't *really* need to match, as
    the `b` in the argument gets ignored, but it's good practice.

    See https://github.com/hyperopt/hyperopt/wiki/FMin#2-defining-a-search-space
    for more details on defining spaces.
    :param extra_function_kwargs: Extra kwargs to pass to function.
    :param trials: hyperopt.base.Trials. Its important members are:
        - `trials.trials` - a list of dictionaries representing everything
        about the search
        - `trials.results` - a list of dictionaries returned by 'objective'
        during the search
        - `trials.losses()` - a list of losses (float for each 'ok' trial)
        - `trials.statuses()` - a list of status strings
    :param num_evals: Maximum number of queries to function.
    :param maximize: Default behavior is the minimize the functions. If True,
    maximize the .
    :param dotmap_to_nested_dictionary: If True, convert keys like `a.b` into
    nested dictionaries before passing onto the main function.
    :param flatten_choice_dictionary: If True, flatten the nested dictionary
    caused by creating a choice variable before passing it to the functions.
    :param max_magnitude: Clip the returned value's magnitude to this value.

    A choice variable defined as

    ```
        z=hp.choice('z', [
            ('z_one', {'a': 5}),
            ('z_two', {'b': 4),
        ])
    ```
    will results in a dictionary like
    ```
        {
            `z`: `z_one`,
            `a`: 5,
        }
    ```
    or
    ```
        {
            `z`: `z_two`,
            `a`: 4,
        }
    ```
    that is passed to the optimization function.
    If `flatten_choice_dictionary` is False, you get
    ```
        {
            `z`: (
                'z_one',
                {
                    `a`: 4,
                }
            )
        }

    ```
    which I find a bit messier.

    :return: tuple
     - best_params: Best dictionary passed to function. Does not include the
     `extra_function_kwargs`.
     - minimum: value at the best_variant.
     - trials: updated hyperopt.base.Trials instance.
     - best_variant: Best dictionary over the search space. Similar to
     `best_params`, but this has a type that hyperopt uses.
    """
    if extra_function_kwargs is None:
        extra_function_kwargs = {}

    def wrapped_function(params):
        start_time = time.time()
        if flatten_choice_dictionary:
            params = flatten_hyperopt_choice_dict(params)
        if dotmap_to_nested_dictionary:
            params = dot_map_dict_to_nested_dict(params)
        loss = function(merge_recursive_dicts(
            params,
            extra_function_kwargs,
            ignore_duplicate_keys_in_second_dict=True,
        ))
        if maximize:
            loss = - loss
        if np.isnan(loss):
            loss = max_magnitude
        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': params,
            # -- store other results like this
            'eval_time': time.time() - start_time,
        }

    if trials is None:
        trials = Trials()
    best_variant = fmin(
        wrapped_function,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_evals,
        trials=trials,
    )
    min_index, min_value = min(enumerate(trials.losses()), key=lambda p: p[1])
    min_results = trials.results[min_index]
    best_params = min_results['params']
    return best_params, min_value, trials, best_variant


def flatten_hyperopt_choice_dict(hyperopt_choice_dict):
    """
    Flatten recursive hyperopt choice dictionary. Behavior is undefined if keys
    are repeated.

    A list of (name, value) tuples is treated like a dict.

    :param hyperopt_choice_dict: Potentially recursive dictionary (e.g. dict of dict of dicts).
    :return: One flat dictionary, where elements
    """

    def iter_items():
        for key, value in hyperopt_choice_dict.items():
            # TODO(vitchyr): Figure out the best way to deal with empty dicts
            if isinstance(value, dict) and len(value) > 0:
                for subkey, subvalue in flatten_hyperopt_choice_dict(value).items():
                    yield subkey, subvalue
            elif (isinstance(value, tuple) and len(value) == 2 and
                      isinstance(value[1], dict)):
                yield key, value[0]
                for subkey, subvalue in flatten_hyperopt_choice_dict(value[1]).items():
                    yield subkey, subvalue
            else:
                yield key, value

    new_dict = {}
    for key, value in iter_items():
        if key in new_dict:
            raise Exception("Key defined twice: {}".format(key))
        new_dict[key] = value
    return new_dict


def __example_objective(params):
    loss = 1
    if params['z'] == 'one':
        loss = params['c'] ** 2 - 1
        # Accessing params['b'] would throw an error!
    elif params['z'] == 'two':
        loss = params['b'] ** 2
        loss += params['d']
        # Accessing params['c'] would throw an error!
    loss += params['a']
    loss += params['f']['e']
    loss += params['f']['g']
    loss += len(params['string'])
    return loss


if __name__ == '__main__':
    space1 = dict(
        c=hp.uniform('c', -10, 10),
    )
    space2 = dict(
        b=hp.uniform('b', -10, 10),
        d=hp.uniform('d', -10, 10),
    )
    space = dict(
        a=hp.uniform('a', -1, 1),
        z=hp.choice('z', [
            ('one', space1),
            ('two', space2),
        ]),
        # With hp.choice the second argument MUST be a list of tuples. So
        # you can't do
        #     z=hp.choice('z', {
        #         'one': space1,
        #         'two': space2,
        #     })
        # Not sure why.
    )
    space['f.e'] = 6  # These get converted into dictionaries of dictionaries
    space['f.g'] = 5
    argmin_params, minimum, trials, _ = optimize(
        __example_objective,
        space,
        extra_function_kwargs={'string': "hello"},  # You can pass kwargs
        num_evals=100,
    )

    print("Optimal parameters found:")
    print(argmin_params)
    print("Minimize score found:")
    print(minimum)

    best_params_truth = {
        'z': 'two',
        'b': 0,
        'd': -10,
        'a': -1,
        'f': {
            'e': 6,
            'g': 5,
        },
    }
    print("Optimal parameters possible:")
    print(best_params_truth)
    print("Minimize score possible:")
    best_params_truth['string'] = 'hello'
    print(__example_objective(best_params_truth))
