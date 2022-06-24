"""A set of utility functions for lambda wrappers."""
from copy import deepcopy
from functools import singledispatch
from typing import Any, Sequence
from typing import Tuple as TypingTuple

import numpy as np

import gym
from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Dict, Discrete, Space, Tuple, MultiBinary, MultiDiscrete
from gym.dev_wrappers.utils.utils import is_nestable, transform_nestable_space


@singledispatch
def reshape_space(
    space: Space, args: FuncArgType[TypingTuple[int, int]], fn
) -> Any:
    """Reshape space with the provided args."""


@reshape_space.register(Box)
def _reshape_space_box(space, args: FuncArgType[TypingTuple[int, int]], fn):
    if not args:
        return space
    return Box(
        np.reshape(space.low, args),
        np.reshape(space.high, args),
        shape=args)


@reshape_space.register(Discrete)
@reshape_space.register(MultiBinary)
@reshape_space.register(MultiDiscrete)
def _reshape_space_not_reshapable(space, args: FuncArgType[TypingTuple[int, int]], fn):
    if args:
        # TODO: raise warning that args has no effect here
        ...
    return space


@singledispatch
def transform_space_bounds(
    space: Space, args: FuncArgType[TypingTuple[int, int]], fn
) -> Any:
    """Transform space bounds with the provided args."""


@transform_space_bounds.register(Box)
def _transform_space_box(space, args: FuncArgType[TypingTuple[int, int]], fn):
    if not args:
        return space
    return Box(*args, shape=space.shape)


@transform_space_bounds.register(Discrete)
def _transform_space_discrete(
    space, args: FuncArgType[TypingTuple[int, int]], fn
):
    if not args:
        return space
    return space


@reshape_space.register(Tuple)
@transform_space_bounds.register(Tuple)
def _transform_space_tuple(
    space: Tuple, args: FuncArgType[TypingTuple[int, int]], fn
):
    assert isinstance(args, Sequence)
    assert len(space) == len(args)

    updated_space = [s for s in space]

    for i, arg in enumerate(args):
        if is_nestable(space[i]):
            transform_nestable_space(space[i], updated_space, i, args[i], fn)
            if isinstance(updated_space[i], list):
                updated_space[i] = Tuple(updated_space[i])
        else:
            updated_space[i] = fn(space[i], arg, fn)

    return Tuple(updated_space)



@reshape_space.register(Dict)
@transform_space_bounds.register(Dict)
def _transform_space_dict(
    space: Dict, args: FuncArgType[TypingTuple[int, int]], fn
):
    assert isinstance(args, dict)
    updated_space = deepcopy(space)

    for arg in args:
        if is_nestable(space[arg]):
            transform_nestable_space(space[arg], updated_space, arg, args[arg], fn)
        else:
            updated_space[arg] = fn(space[arg], args.get(arg), fn)
    return updated_space
