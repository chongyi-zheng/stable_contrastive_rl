from gym.envs import registration


def get_class_and_kwargs(spec_or_id):
    if isinstance(spec_or_id, registration.EnvSpec):
        spec = spec_or_id
    else:
        spec = registration.spec(spec_or_id)
    return registration.load(spec._entry_point), spec._kwargs