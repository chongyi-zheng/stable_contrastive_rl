from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# import os
# import re
# import pprint
from collections import OrderedDict

import yaml
# from easydict import EasyDict as edict


ROOT_DIR = '.'


def update_dict(d, u, check_exist=False):
    for k, v in u.items():
        if check_exist:
            if k not in d.keys():
                raise ValueError('Key `%s` not in the dictionary with keys: %r'
                                 % (k, d.keys()))

        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v, check_exist)
        else:
            d[k] = v
    return d


def convert_key(expression):
    """Converts keys in YAML that reference other keys.
    """
    if (type(expression) is str and len(expression) > 2 and
            expression[1] == '!'):
        expression = eval(expression[2:-1])

    return expression


def ordered_load(stream,
                 loader=yaml.Loader,
                 object_pairs_hook=OrderedDict):
    """Load an ordered dictionary from a yaml file.
    """
    class OrderedLoader(loader):
        pass

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        lambda loader, node: object_pairs_hook(
            loader.construct_pairs(node)))

    return yaml.load(stream, OrderedLoader)


def update_content(arg_dict, input_content, check_exist=False):
    """Update the yaml config with the input content.

    Args:
        input_content: A string in the yaml file format.
    """
    content = ordered_load(input_content)
    for k in content.keys():
        v = convert_key(content[k])
        content[k] = v
    arg_dict = update_dict(arg_dict, content, check_exist)

    return arg_dict


def update_bindings(arg_dict, input_bindings, check_exist=False):
    """Update the yaml config with the input binding.

    Args:
        input_binding: A string in the gin binding format.
    """
    if input_bindings is None:
        pass
    elif ',' in input_bindings:
        bindings = input_bindings.split(',')
        for binding in bindings:
            update_binding(arg_dict, binding, check_exist)
    elif ' ' in input_bindings:
        bindings = input_bindings.split(' ')
        for binding in bindings:
            update_binding(arg_dict, binding, check_exist)
    else:
        update_binding(arg_dict, input_bindings, check_exist)

    return arg_dict


def update_binding(arg_dict, input_binding, check_exist=False):
    """Update the yaml config with the input binding.

    Args:
        input_binding: A string in the gin binding format or a list.
    """
    if input_binding is None:
        pass
    elif isinstance(input_binding, list):
        for item in input_binding:
            update_binding(arg_dict, item, check_exist)
    else:
        items = input_binding.split('=')
        assert len(items) == 2

        key_list = items[0].split('.')
        value = items[1]

        # Add indents.
        for key_ind, key in enumerate(key_list):
            key_list[key_ind] = ''.join(['  '] * key_ind + [key])

        input_content = ': '.join([
            ': \n  '.join(key_list),
            value
        ])

        update_content(arg_dict, input_content, check_exist)

    return arg_dict
