"""Lambda action wrappers that uses jumpy for compatibility with jax (i.e. brax) and numpy environments."""

from typing import Any, Callable
from typing import Tuple as TypingTuple

import jumpy as jp

import gym
from gym import Space
from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Dict, Tuple, apply_function


class lambda_action_v0(gym.ActionWrapper):
    """A wrapper that provides a function to modify the action passed to :meth:`step`.

    Example to convert continuous actions to discrete:
        >>> import gym
        >>> from gym.spaces import Dict
        >>> import numpy as np
        >>> env = gym.make('CarRacing-v2')
        >>> env = lambda_action_v0(env, lambda action, _: np.astype(action, np.int32), None)
        >>> env.action_space
        TODO
        >>> env.reset()
        TODO
        >>> env.step()[0]
        TODO

    Composite action shape:
        >>> env = ExampleEnv(action_space=Dict(TODO))
        >>> env = lambda_action_v0(env, TODO, TODO, TODO)
        >>> env.action_space
        TODO
        >>> env.reset()
        TODO
        >>> env.step()[0]
        TODO
    """

    def __init__(
        self,
        env: gym.Env,
        func: Callable,
        args: FuncArgType[Any],
        action_space: Space = None,
    ):
        """Initialize lambda_action."""
        super().__init__(env)

        self.func = func
        self.func_args = args
        if action_space is None:
            self.action_space = env.action_space
        else:
            self.action_space = action_space

    def action(self, action):
        """Apply function to action."""
        return apply_function(self.action_space, action, self.func, self.func_args)


class clip_actions_v0(lambda_action_v0):
    """A wrapper that clips actions passed to :meth:`step` with an upper and lower bound.

    Basic Example:
        >>> import gym
        >>> env = gym.make("BipedalWalker-v3")
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env = clip_actions_v0(
        ...     env, 
        ...     (
        ...         np.array([-0.5], dtype='float32'), 
        ...         np.array([0.5], dtype='float32'))
        ...     )
        ... )
        >>> env.action_space
        Box([-0.5], [0.5], (4,), float32)

    Clip with only a lower or upper bound:
        >>> env = gym.make(TODO)
        >>> env.action_space
        TODO
        >>> env = clip_actions_v0(env, TODO)
        >>> env.action_space
        TODO

    Composite action space example:
        >>> env = ExampleEnv()
        >>> env.actions_space
        TODO
        >>> env = clip_actions_v0(env, TODO)
        >>> env.action_space
        TODO
    """

    def __init__(self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
        """Constructor for the clip action wrapper.

        Args:
            env: The environment to wrap
            args: The arguments for clipping the action space
        """
        action_space = self._make_clipped_action_space(env, args)

        func = lambda action, args: jp.clip(action, *args)

        super().__init__(env, func, args, action_space)

    def _make_clipped_action_space(
        self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
    ):
        if type(env.action_space) == Box:
            action_space = Box(*args, shape=env.action_space.shape)

        elif type(env.action_space) == Dict:
            assert type(args) == dict
            action_space = Dict()
            
            for k in env.action_space.keys():
                action_space = self._compute_nested_action_space(
                    env.action_space, action_space, k, args
                )
        else:
            action_space = None

        return action_space

    def _compute_nested_action_space(self, env_action_space, action_space, k, args):
        if k not in args:
            action_space[k] = env_action_space[k]
            return action_space

        args = args[k]
        env_action_space = env_action_space[k]

        if type(env_action_space) == Box:
            action_space[k] = Box(*args, shape=env_action_space.shape)

        elif type(env_action_space) == Dict:
            action_space[k] = Dict()
            for m in env_action_space.keys():
                action_space[k] = self._compute_nested_action_space(
                    env_action_space, action_space[k], m, args
                )

        return action_space


class scale_actions_v0(lambda_action_v0):
    """A wrapper that scales actions passed to :meth:`step` with a scale factor.

    Basic Example:
        >>> import gym
        >>> env = gym.make(TODO)
        >>> env.action_space
        TODO
        >>> env = scale_actions_v0(env, TODO, TODO)
        >>> env.action_space
        TODO

    Composite action space example:
        >>> env = ExampleEnv()
        >>> env = scale_actions_v0(env, TODO, TODO)
        >>> env.action_space
        TODO
    """

    def __init__(self, env: gym.Env, args: FuncArgType[float]):
        """Constructor for the scale action wrapper.

        Args:
            env: The environment to wrap
            args: The arguments for scaling the actions
        """
        action_space = None  # TODO, add action space
        super().__init__(env, lambda action, arg: action * arg, args, action_space)
