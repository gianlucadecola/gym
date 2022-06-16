"""A set of utility functions for lambda wrappers."""
import gym
from gym import Space

from typing import Any, Callable, Optional
from typing import Tuple as TypingTuple

from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
from functools import singledispatch


def extend_args(action_space: Space, extended_args: dict, args: dict, space_key: str):
    """Extend args for rescaling actions.

    Action space args needs to be extended in order
    to correctly rescale the actions.
    i.e. args before: {"body":{"left_arm": (-0.5,0.5)}, ...}
    args after: {"body":{"left_arm": (-0.5,0.5,-1,1)}, ...}
    where -1, 1 was the old action space bound.
    old action space is needed to rescale actions.
    """
    if space_key not in args:
        return extended_args

    args = args[space_key]

    if isinstance(args, dict):
        extended_args[space_key] = {}
        for arg in args:
            extend_args(action_space[space_key], extended_args[space_key], args, arg)
    else:
        assert len(args) == len(action_space[space_key].low) + len(
            action_space[space_key].high
        )
        extended_args[space_key] = (
            *args,
            *list(action_space[space_key].low),
            *list(action_space[space_key].high),
        )

    return extended_args


@singledispatch
def transform_space(space: Space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]) -> Any:
    ...


@transform_space.register(Box)
def _transform_space_Box(_, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
    return Box(*args, shape=env.action_space.shape)


@transform_space.register(Tuple)
def _transform_space_Tuple(space: Tuple, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
    action_space = Tuple([])
    return action_space


@transform_space.register(Dict)
def _transform_space_dict(space: Dict, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
    def _transform_dict_space_helper(
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
                space[space_key] = _transform_dict_space_helper(
                    env_space, space[space_key], m, args
                )
        return space
    
    action_space = Dict()

    for k in env.action_space.keys():
        _transform_dict_space_helper(env.action_space, action_space, k, args)
    return action_space