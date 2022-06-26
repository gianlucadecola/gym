"""A set of utility functions for lambda wrappers."""
import tinyscaler
from functools import singledispatch
from typing import Any, Callable
from typing import Tuple as TypingTuple

import numpy as np

import gym
from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Space, Discrete, MultiBinary, MultiDiscrete


@singledispatch
def grayscale_space(
    space: Space, args: FuncArgType[TypingTuple[int, int]], fn: Callable
) -> Any:
    """Make observation space grayscale (i.e. flatten third dimension)."""


@grayscale_space.register(Box)
def _grayscale_space_box(space, args: FuncArgType[TypingTuple[int, int]], fn: Callable):
    if len(space) != 3 and space.shape[-1] != 3:
        """raise if we are not dealing with an image-like space"""
        raise
    # TODO: warning if we are not dealing with uint8
    w, h = space.shape[0], space.shape[1]
    
    return Box(
        0,
        255,
        shape=(w, h),
        dtype=space.dtype
    )


@grayscale_space.register(Discrete)
@grayscale_space.register(MultiBinary)
@grayscale_space.register(MultiDiscrete)
def _grayscale_space_not_reshapable(space, args: FuncArgType[TypingTuple[int, int]], fn: Callable):
    """Return original space shape for not reshable space.
    
    Trying to reshape `Discrete`, `Multibinary` and `MultiDiscrete`
    spaces has no effect.
    """
    if args:
        # TODO: raise warning that args has no effect here
        ...
    return space