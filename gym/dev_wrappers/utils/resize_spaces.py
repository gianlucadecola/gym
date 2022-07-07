"""A set of utility functions for lambda wrappers."""
import warnings
from functools import singledispatch
from typing import Any, Callable
from typing import Tuple as TypingTuple

import numpy as np
import tinyscaler

from gym.dev_wrappers import FuncArgType
from gym.error import InvalidSpaceOperation
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space


@singledispatch
def resize_space(
    space: Space, args: FuncArgType[TypingTuple[int, int]], fn: Callable
) -> Any:
    """Resize space with the provided args."""


@resize_space.register(Discrete)
@resize_space.register(MultiBinary)
@resize_space.register(MultiDiscrete)
def _resize_space_not_reshapable(
    space, args: FuncArgType[TypingTuple[int, int]], fn: Callable
):
    """Return original space shape for not reshable space.

    Trying to reshape `Discrete`, `Multibinary` and `MultiDiscrete`
    spaces has no effect.
    """
    if args:
        raise InvalidSpaceOperation(
            f"Cannot resize a space of type {type(space)}."
        )
    return space


@resize_space.register(Box)
def _resize_space_box(space, args: FuncArgType[TypingTuple[int, int]], fn: Callable):
    if args is not None:
        return Box(
            tinyscaler.scale(space.low, args, mode='bilinear'),
            tinyscaler.scale(space.high, args, mode='bilinear'),
            shape=args,
            dtype=space.dtype,
        )
    return space