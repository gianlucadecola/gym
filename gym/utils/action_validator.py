from gym.error import InvalidAction


def validate_action(func):
    def wrapper(self, action, *args, **kwargs):
        if not self.action_space.contains(action):
            raise InvalidAction(
                f"you passed the action `{action}` with dtype "
                f"{type(action)} while the supported action space is "
                f"{self.action_space} with dtype {self.action_space.dtype}"
            )
        return func(self, action, *args, **kwargs)

    return wrapper
