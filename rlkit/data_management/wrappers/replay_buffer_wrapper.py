from rlkit.data_management.replay_buffer import ReplayBuffer

class ProxyBuffer(ReplayBuffer):
    def __init__(self, replay_buffer):
        self._wrapped_buffer = wrapped_buffer

    def add_sample(self, *args, **kwargs):
        self._wrapped_buffer.add_sample(*args, **kwargs)

    def terminate_episode(self, *args, **kwargs):
        self._wrapped_buffer.terminate_episode(*args, **kwargs)

    def num_steps_can_sample(self, *args, **kwargs):
        self._wrapped_buffer.num_steps_can_sample(*args, **kwargs)

    def add_path(self, *args, **kwargs):
        self._wrapped_buffer.add_path(*args, **kwargs)

    def add_paths(self, *args, **kwargs):
        self._wrapped_buffer.add_paths(*args, **kwargs)

    @abc.abstractmethod
    def random_batch(self, *args, **kwargs):
        self._wrapped_buffer.random_batch(*args, **kwargs)

    def get_diagnostics(self, *args, **kwargs):
        return self._wrapped_buffer.get_diagnostics(*args, **kwargs)

    def get_snapshot(self, *args, **kwargs):
        return self._wrapped_buffer.get_snapshot(*args, **kwargs)

    def end_epoch(self, *args, **kwargs):
        return self._wrapped_buffer.end_epoch(*args, **kwargs)

    @property
    def wrapped_buffer(self):
        return self._wrapped_buffer

    def __getattr__(self, attr):
        if attr == '_wrapped_buffer':
            raise AttributeError()
        return getattr(self._wrapped_buffer, attr)
