"""A set of utility functions for lambda wrappers."""
import tinyscaler
from functools import singledispatch
from typing import Any, Callable
from typing import Tuple as TypingTuple

import numpy as np

import gym
from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Space


@singledispatch
def resize_space(
    space: Space, args: FuncArgType[TypingTuple[int, int]], fn: Callable
) -> Any:
    """Resize space with the provided args."""


@resize_space.register(Box)
def _resize_space_box(space, args: FuncArgType[TypingTuple[int, int]], fn: Callable):
    return Box(
        tinyscaler.scale(space.low, args),
        tinyscaler.scale(space.high, args),
        shape=args,
        dtype=space.dtype
    )