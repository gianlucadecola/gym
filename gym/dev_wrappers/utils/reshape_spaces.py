"""A set of utility functions for lambda wrappers."""
from copy import deepcopy
from functools import singledispatch
from typing import Any, Sequence
from typing import Tuple as TypingTuple

import numpy as np

import gym
from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Dict, Discrete, Space, Tuple, MultiBinary, MultiDiscrete
from gym.dev_wrappers.utils.utils import is_nestable


@singledispatch
def reshape_space(
    space: Space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
) -> Any:
    """Reshape space with the provided args."""


@reshape_space.register(Box)
def _reshape_space_box(space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
    if not args:
        return space
    return Box(
        np.reshape(space.low, args),
        np.reshape(space.high, args),
        shape=args)


@reshape_space.register(Discrete)
@reshape_space.register(MultiBinary)
@reshape_space.register(MultiDiscrete)
def _reshape_space_not_reshapable(space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
    if args:
        # TODO: raise warning that args has no effect here
        ...
    return space


@reshape_space.register(Tuple)
def _reshape_space_tuple(space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
    assert isinstance(args, Sequence)
    assert len(space) == len(args)

    updated_space = [s for s in space]

    for i, arg in enumerate(args):
        if is_nestable(space[i]):
            reshape_nestable_space(space[i], updated_space, i, args[i], env)
            if isinstance(updated_space[i], list):
                updated_space[i] = Tuple(updated_space[i])
        else:
            updated_space[i] = reshape_space(env.observation_space[i], env, arg)

    return Tuple(updated_space)


@reshape_space.register(Dict)
def _reshape_space_dict(
    space: Dict, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
):
    assert isinstance(args, dict)
    updated_space = deepcopy(env.observation_space)

    for arg in args:
        if is_nestable(space[arg]):
            reshape_nestable_space(space[arg], updated_space, arg, args[arg], env)
        else:
            updated_space[arg] = reshape_space(space[arg], env, args.get(arg))
    return updated_space


@singledispatch
def reshape_nestable_space(
    original_space: gym.Space,
    space: gym.Space,
    space_key: str,
    args: FuncArgType[TypingTuple[int, int]],
    env=None,
):
    """Transform nestable space with the provided args."""


@reshape_nestable_space.register(Dict)
def _reshape_nestable_dict_space(
    original_space: gym.Space,
    updated_space: gym.Space,
    arg: str,
    args: FuncArgType[TypingTuple[int, int]],
    env,
):
    """Recursive function to process possibly nested `Dict` spaces."""
    updated_space = updated_space[arg]

    for arg in args:
        if is_nestable(original_space[arg]):
            reshape_nestable_space(
                original_space[arg], updated_space, arg, args[arg], env
            )
        else:
            updated_space[arg] = reshape_space(
                original_space[arg], env, args.get(arg)
            )


@reshape_nestable_space.register(Tuple)
def _reshape_nestable_tuple_space(
    original_space: gym.Space,
    updated_space: gym.Space,
    idx_to_update: int,
    args: FuncArgType[TypingTuple[int, int]],
    env,
):
    """Recursive function to process possibly nested `Tuple` spaces."""
    updated_space[idx_to_update] = [s for s in original_space]

    if args is None:
        return

    for i, arg in enumerate(args):
        if is_nestable(original_space[i]):
            _reshape_nestable_tuple_space(
                original_space[i], updated_space[idx_to_update], i, args[i], env
            )
        else:
            updated_space[idx_to_update][i] = reshape_space(
                original_space[i], env, arg
            )

    if isinstance(updated_space[idx_to_update], list):
        updated_space[idx_to_update] = Tuple(updated_space[idx_to_update])