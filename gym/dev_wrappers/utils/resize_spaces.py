"""A set of utility functions for lambda wrappers."""
from copy import deepcopy
from functools import singledispatch
from typing import Any, Sequence
from typing import Tuple as TypingTuple

import numpy as np

import gym
from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Dict, Discrete, Space, Tuple
from gym.dev_wrappers.utils.utils import is_nestable


@singledispatch
def resize_space(
    space: Space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
) -> Any:
    """Resize space with the provided args."""


@resize_space.register(Box)
def _resize_space_box(space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
    ...