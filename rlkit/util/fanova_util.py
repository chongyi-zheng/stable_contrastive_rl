from fanova import fANOVA
import numpy as np
from collections import defaultdict, namedtuple

from rlkit.util.data_processing import get_trials
import ConfigSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
)

import rlkit.pythonplusplus as ppp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

FanovaInfo = namedtuple(
    'FanovaInfo', ['f', 'config_space', 'X', 'Y', 'categorical_remapping',
                   'variants']
)


def get_fanova_info(
        base_dir,
        params_to_ignore=('seed', 'exp_id', 'unique_id', 'exp_name'),
        ylabel='AverageReturn',
):
    data_and_variants = get_trials(base_dir)
    experiment_data_list, variants_list = zip(*data_and_variants)
    ylabel = ylabel.replace(' ', '_')
    ylabel = ylabel.replace('-', '')
    if ylabel not in experiment_data_list[0].dtype.names:
        print("Possible ylabels:")
        for name in experiment_data_list[0].dtype.names:
            print(" - {}".format(name))
        raise ValueError("Invalid ylabel: {}".format(ylabel))
    indices_of_experiments_with_data = [
        i for i, exp in enumerate(experiment_data_list) if exp[ylabel].size >= 1
    ]
    if len(indices_of_experiments_with_data) != len(experiment_data_list):
        print("WARNING: Skipping some experiments. Probably because they only "
              "have one data point.")
    valid_experiment_data_list = [
        d for i, d in enumerate(experiment_data_list)
        if i in indices_of_experiments_with_data
    ]
    variants_list = [
        v for i, v in enumerate(variants_list)
        if i in indices_of_experiments_with_data
    ]
    Y = np.array([
        exp[ylabel][-1]
            if exp[ylabel].size > 1
            else np.array(float(exp[ylabel]), dtype=np.double)
        for exp in valid_experiment_data_list
    ])
    filtered_variants_list = remove_keys_with_nonunique_values(
        variants_list, params_to_ignore=params_to_ignore
    )
    filtered_variants_to_values = get_dict_key_to_values(filtered_variants_list)
    names = list(filtered_variants_list[0].keys())
    X_raw = _extract_features(filtered_variants_list, names)
    config_space, X, categorical_remapping = (
        _get_config_space_and_new_features(
            X_raw,
            names,
            filtered_variants_to_values,
        )
    )

    # Not sure why, but config_space shuffles the order of the hyperparameters
    new_name_order = [
        config_space.get_hyperparameter_by_idx(i) for i in range(len(names))
    ]
    new_order = [names.index(name) for name in new_name_order]
    X = [X[i] for i in new_order]
    # X has be [feature_dim X batch_size], but Fanova expects the transpose
    X = np.array(X, dtype=object).T
    return FanovaInfo(
        fANOVA(X, Y, config_space=config_space),
        config_space,
        X,
        Y,
        categorical_remapping,
        variants_list,
    )


def get_dict_key_to_values(dict_list):
    """
    Given a list of dictionaries, return a dictionary. Keys are the set
    of keys in the list of dictionaries. Values are the set of values
    that seen with that key across every dictionary.
    :param all_variants:
    :return:
    """
    dict_key_to_values = defaultdict(set)
    for d in dict_list:
        for k, v in d.items():
            if type(v) == list:
                v = str(v)
            dict_key_to_values[k].add(v)
    return dict_key_to_values


def remove_keys_with_nonunique_values(dict_list, params_to_ignore=None):
    """
    Given a list of dictionaries, remove all keys from the dictionaries where
    all the dictionaries have the same value for that key.
    :param dict_list:
    :param params_to_ignore:
    :return:
    """
    if params_to_ignore is None:
        params_to_ignore = []
    key_to_values = get_dict_key_to_values(dict_list)
    filtered_dicts = []
    for d in dict_list:
        new_d = {
            k: v for k, v in d.items()
            if len(key_to_values[k]) > 1 and k not in params_to_ignore
        }
        filtered_dicts.append(new_d)
    return filtered_dicts


def _extract_features(all_variants, names):
    Xs = []
    for feature_i, name in enumerate(names):
        X = []
        for ex_i, variant in enumerate(all_variants):
            X.append(variant[name])
        Xs.append(X)
    return Xs


def _get_config_space_and_new_features(Xs, names, name_to_possible_values):
    config_space = ConfigSpace.ConfigurationSpace()
    new_Xs = []
    categorical_remappings = {}
    for X, name in zip(Xs, names):
        if ppp.is_numeric(X[0]):
            config_space.add_hyperparameter(
                UniformFloatHyperparameter(name, np.min(X), np.max(X))
            )
            new_Xs.append(X)
        else:
            possible_values = list(name_to_possible_values[name])
            if len(possible_values) > 128:
                raise ValueError("Size of categorical hyperparameter cannot "
                                 "be larger than 128.")
            config_space.add_hyperparameter(
                CategoricalHyperparameter(
                    name,
                    choices=list(range(len(possible_values))),
                    default=0,
                )
            )
            id_map = ppp.IntIdDict()
            categorical_remappings[name] = id_map
            new_Xs.append([id_map[str(x)] for x in X])

    return config_space, new_Xs, categorical_remappings


def is_categorical(f: fANOVA, param_name_or_id):
    if type(param_name_or_id) == int:
        param_name = f.cs.get_hyperparameter_by_idx(param_name_or_id)
    else:
        param_name = param_name_or_id
    param = f.cs.get_hyperparameter(param_name)
    return isinstance(param, CategoricalHyperparameter)


def plot_pairwise_marginal(vis, param_names, resolution=20, show=True):
    """
    Creates a plot of pairwise marginal of a selected parameters

    The version in visualize.pyr is wrong.

    :param param_names: list of strings
        Contains the selected parameters

    :parm resolution: int
        Number of samples to generate from the parameter range as
        values to predict
    """
    grid_list, zz = generate_pairwise_marginal(vis, param_names, resolution)

    display_xx, display_yy = np.meshgrid(grid_list[0], grid_list[1])

    fig = plt.figure()
    ax = Axes3D(fig)

    surface = ax.plot_surface(
        display_xx,
        display_yy,
        zz,
        rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False,
    )
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_zlabel("Performance")
    fig.colorbar(surface, shrink=0.5, aspect=5)
    if show:
        plt.show()
    else:
        return plt


def generate_pairwise_marginal(vis, param_list, resolution=20):
    """
    Creates a plot of pairwise marginal of a selected parameters

    The version in visualize.pyr is wrong.

    :param param_list: list of ints or strings
        Contains the selected parameters

    :param resolution: int
        Number of samples to generate from the parameter range as
        values to predict
    """
    assert len(param_list) == 2, "You have to specify 2 (different) parameters"

    grid_list = []
    param_names = []
    for p in param_list:
        if type(p) == str:
            p = vis.cs.get_idx_by_hyperparameter_name(p)
        lower_bound = vis.cs_params[p].lower
        upper_bound = vis.cs_params[p].upper
        param_names.append(vis.cs_params[p].name)
        grid = np.linspace(lower_bound, upper_bound, resolution)
        grid_list.append(grid)

    zz = np.zeros([resolution * resolution])
    for i, y_value in enumerate(grid_list[1]):
        for j, x_value in enumerate(grid_list[0]):
            zz[i * resolution + j] = vis.fanova.marginal_mean_variance_for_values(param_list, [x_value, y_value])[0]

    zz = np.reshape(zz, [resolution, resolution])

    return grid_list, zz
