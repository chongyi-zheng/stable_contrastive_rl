from ray import serialization, utils


def set_serialization_mode_to_pickle(cls):
    """
    Whenever this class is serialized by ray, it will default to using pickle
    serialization (__setstate__ and __getstate__)

    WARNING: This will only work if the driver is serializing and workers
    are de-serializing.

    :param cls: class instance or the Class itself
    """
    if cls not in serialization.type_to_class_id:
        serialization.add_class_to_whitelist(
            cls,
            utils.random_string(),
            pickle=True,
        )
