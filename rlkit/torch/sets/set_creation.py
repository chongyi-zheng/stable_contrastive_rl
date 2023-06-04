import pickle
from os import path as osp

from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.envs.pygame import pnp_util
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.util import io
from rlkit.torch.sets.set_projection import Set
from rlkit.launchers.config import LOCAL_LOG_DIR


def create_sets(
    env_id,
    env_class,
    env_kwargs,
    renderer,
    saved_filename=None,
    save_to_filename=None,
    **kwargs
):
    if saved_filename is not None:
        sets = io.load_local_or_remote_file(saved_filename)
    else:
        env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        if isinstance(env, PickAndPlaceEnv):
            sets = sample_pnp_sets(env, renderer, **kwargs)
        else:
            raise NotImplementedError()
    if save_to_filename:
        save(sets, save_to_filename)
    return sets


def sample_pnp_sets(
    env,
    renderer,
    num_sets=1,
    num_samples_per_set=128,
    set_configs=None,
    example_state_key="example_state",
    example_image_key="example_image",
):
    if set_configs is None:
        print(__file__, "WARNING: will deprecate soon")
        set_projections = pnp_util.sample_set_projections(env, num_sets)
    else:
        set_projections = [
            pnp_util.create_set_projection(**set_config)
            for set_config in set_configs
        ]
    sets = []
    for set_projection in set_projections:
        # for set_config in set_configs:
        # set_projection = pnp_util.create_set_projection(**set_config)
        example_dict = pnp_util.sample_examples_with_images(
            env,
            renderer,
            set_projection,
            num_samples_per_set,
            state_key=example_state_key,
            image_key=example_image_key,
        )
        sets.append(Set(example_dict, set_projection))
    return sets


def get_absolute_path(relative_path):
    return osp.join(LOCAL_LOG_DIR, relative_path)


def load(relative_path):
    path = get_absolute_path(relative_path)
    print("loading data from", path)
    return pickle.load(open(path, "rb"))


def save(data, relative_path):
    path = get_absolute_path(relative_path)
    pickle.dump(data, open(path, "wb"))