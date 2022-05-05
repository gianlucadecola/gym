import warnings

import numpy as np

from gym.error import InvalidAction


def validate_action(func):
    def wrapper(self, action, *args, **kwargs):

        # TODO: if we want to enforce the dtype we should remove
        # this and check action_space.contains with the actual action. Current
        # behaviour allow user to pass float64 to an action space of type
        # float32 and vice-versa
        if isinstance(action, np.ndarray) and self.action_space.dtype != action.dtype:
            backward_compatible_action = np.cast[self.action_space.dtype](action)
            warnings.warn(
                f"You are passing an action of dtype {action.dtype} to an environment "
                f"with action space of type {self.action_space.dtype}"
            )
        else:
            backward_compatible_action = action

        if not self.action_space.contains(backward_compatible_action):
            if isinstance(action, np.ndarray):
                action_type = action.dtype
            else:
                action_type = type(action)
            raise InvalidAction(
                f"you passed the action `{action}` with dtype "
                f"{action_type} while the supported action space is "
                f"{self.action_space} with dtype {self.action_space.dtype}"
            )
        return func(self, action, *args, **kwargs)

    return wrapper


def validate_action_discrete(func):
    def wrapper(self, action, *args, **kwargs):
        if self.action_space.contains(action):
            raise InvalidAction(
                f"you passed the action `{action}` with dtype "
                f"{type(action)} while the supported action space is "
                f"{self.action_space} with dtype {self.action_space.dtype}"
            )
        return func(self, action, *args, **kwargs)

    return wrapper
