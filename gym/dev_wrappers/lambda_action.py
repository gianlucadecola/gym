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
        >>> env = lambda_action_v0(env, lambda action, _: action.astype(np.int32), None)
        >>> env.action_space
        TODO
        >>> env.reset()
        TODO
        >>> env.step()[0]
        TODO

    Composite action shape:
        >>> env = ExampleEnv(action_space=Dict(left_arm=Discrete(4), right_arm=Box(0.0, 5.0, (1,)))
        >>> env = lambda_action_v0(
        ...     env,
        ...     lambda action, _: action.astype(np.int32),
        ...     {"right_arm": True},
        ...     None
        ... )
        >>> env.action_space
        TODO
        >>> env.reset()
        TODO
        >>> env.step()
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

    def _transform_dict_space(
        self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
    ):
        """Process the `Dict` space and apply the transformation."""
        action_space = Dict()

        for k in env.action_space.keys():
            self._transform_dict_space_helper(env.action_space, action_space, k, args)
        return action_space

    def _transform_dict_space_helper(
        self,
        env_space: gym.Space,
        space: gym.Space,
        space_key: str,
        args: FuncArgType[TypingTuple[int, int]],
    ):
        """Recursive function to process possibly nested `Dict` spaces."""
        if space_key not in args:
            space[space_key] = env_space[space_key]
            return space

        args = args[space_key]
        env_space = env_space[space_key]

        if isinstance(env_space, Box):
            space[space_key] = Box(*args, shape=env_space.shape)

        elif isinstance(env_space, Dict):
            space[space_key] = Dict()
            for m in env_space.keys():
                space[space_key] = self._transform_dict_space_helper(
                    env_space, space[space_key], m, args
                )
        return space


class clip_actions_v0(lambda_action_v0):
    """A wrapper that clips actions passed to :meth:`step` with an upper and lower bound.

    Basic Example:
        >>> import gym
        >>> env = gym.make("BipedalWalker-v3")
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env = clip_actions_v0(env, (-0.5, 0.5))
        >>> env.action_space
        Box(-0.5, 0.5, (4,), float32)

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
        Dict(body: Dict(head: Box(0.0, 10.0, (1,), float32)), left_arm: Discrete(4), right_arm: Box(0.0, 5.0, (1,), float32))
        >>> args = {"right_arm": (0, 2), "body": {"head": (0, 3)}}
        >>> env = clip_actions_v0(env, args)
        >>> env.action_space
        Dict(body: Dict(head: Box(0.0, 3.0, (1,), float32)), left_arm: Discrete(4), right_arm: Box(0.0, 2.0, (1,), float32))
    """

    def __init__(self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
        """Constructor for the clip action wrapper.

        Args:
            env: The environment to wrap
            args: The arguments for clipping the action space
        """
        if type(env.action_space) == Box:
            action_space = Box(*args, shape=env.action_space.shape)
        elif type(env.action_space) == Dict:
            assert isinstance(args, dict)
            action_space = self._transform_dict_space(env, args)
        else:
            action_space = None

        super().__init__(
            env, lambda action, args: jp.clip(action, *args), args, action_space
        )


class scale_actions_v0(lambda_action_v0):
    """A wrapper that scales actions passed to :meth:`step` with a scale factor.

    Basic Example:
        >>> import gym
        >>> env = gym.make('BipedalWalker-v3')
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
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
        if type(env.action_space) == Box:
            action_space = Box(*args, shape=env.action_space.shape)
            args = (*args, env.action_space.low, env.action_space.high)
        elif type(env.action_space) == Dict:
            assert isinstance(args, dict)
            action_space = self._transform_dict_space(env, args)
            # TODO: recursive function to add low and high to args
            """
            args before: {"body":{"left_arm": (-0.5,0.5)}, ...}
            args after: {"body":{"left_arm": (-0.5,0.5,-1,1)}, ...}
            where -1, 1 was the old action space bound.
            """
            new_args = {}
            for k in args:
                transform_args(env.action_space, new_args, args, k)
            args = new_args
        else:
            action_space = env.action_space

        def func(action, args):
            action_space_low = args[0]
            action_space_high = args[1]
            low = args[2]
            high = args[3]

            return jp.clip(
                low
                + (high - low)
                * (
                    (action - action_space_low) / (action_space_high - action_space_low)
                ),
                low,
                high,
            )

        super().__init__(env, func, args, action_space)


def transform_args(action_space, new_args, args, space_key):
    if space_key not in args:
        return new_args

    args = args[space_key]

    if isinstance(args, dict):
        new_args[space_key] = {}
        for m in args:
            transform_args(action_space[space_key], new_args[space_key], args, m)
    else:
        new_args[space_key] = (
            *args,
            *list(action_space[space_key].low),
            *list(action_space[space_key].high)
            )

    return new_args