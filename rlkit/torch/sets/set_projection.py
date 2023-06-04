import typing
import numpy as np


ExampleSet = typing.Dict[str, typing.Any]


class SetProjection(object):
    def __call__(self, state):
        raise NotImplementedError()


class SetDescription(object):
    def distance_to_set(self, state):
        raise NotImplementedError()

    def describe(self):
        raise NotImplementedError()


class ProjectOntoAxis(SetProjection, SetDescription):
    def __init__(self, axis_idx_to_value):
        self._axis_idx_to_value = axis_idx_to_value

    def __call__(self, state):
        new_state = state.copy()
        for idx, value in self._axis_idx_to_value.items():
            new_state[idx] = value
        return new_state

    def distance_to_set(self, states):
        differences = []
        for idx, value in self._axis_idx_to_value.items():
            differences.append(states[..., idx] - value)
        delta_vectors = np.array(differences)
        return np.linalg.norm(delta_vectors, axis=0)

    def describe(self):
        return "distance_to_axes_" + "_".join(
            [
                str(idx)
                for idx in self._axis_idx_to_value
            ]
        )


class MoveAtoB(SetProjection, SetDescription):
    """
    Project onto the set of states where some elements match another element.
    Usage:
    ```
    projection = MoveAtoB({
        0: 2,
        1: 3,
    })
    state = np.array([10, 11, 12, 13, 14])
    new_state = projection(state)
    print(new_state)
    ```
    will output
    ```
    [12, 13, 12, 13, 14]
    ```
    """

    def __init__(self, a_axis_to_b_axis):
        self.a_axis_to_b_axis = a_axis_to_b_axis

    def __call__(self, state):
        new_state = state.copy()
        for a_i, b_i in self.a_axis_to_b_axis.items():
            new_state[a_i] = new_state[b_i]
        return new_state

    def distance_to_set(self, states):
        differences = []
        for a_i, b_i in self.a_axis_to_b_axis.items():
            differences.append(states[..., a_i] - states[..., b_i])
        delta_vectors = np.array(differences)
        return np.linalg.norm(delta_vectors, axis=0)

    def describe(self):
        return "relative_distance_" + "_".join(
            [
                "{}to{}".format(a_i, b_i)
                for a_i, b_i in self.a_axis_to_b_axis.items()
            ]
        )


class Set(object):
    def __init__(
        self, example_dict: ExampleSet, description: SetDescription,
    ):
        self.example_dict = example_dict
        self.description = description
