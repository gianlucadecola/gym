from copy import deepcopy
from functools import singledispatch
from typing import Any, Sequence
from typing import Tuple as TypingTuple

import gym
from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Dict, Discrete, Space, Tuple
from gym.dev_wrappers.utils.utils import is_nestable

@singledispatch
def transform_space(
    space: Space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
) -> Any:
    """Transform space with the provided args."""


@transform_space.register(Box)
def _transform_space_box(space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
    if not args:
        return space
    return Box(*args, shape=space.shape)


@transform_space.register(Discrete)
def _transform_space_discrete(
    space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
):
    if not args:
        return space
    return space


@transform_space.register(Tuple)
def _transform_space_tuple(
    space: Tuple, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
):
    assert isinstance(args, Sequence)
    assert len(space) == len(args)

    updated_space = [s for s in space]

    for i, arg in enumerate(args):
        if is_nestable(space[i]):
            transform_nestable_space(space[i], updated_space, i, args[i], env)
            if isinstance(updated_space[i], list):
                updated_space[i] = Tuple(updated_space[i])
        else:
            updated_space[i] = transform_space(env.action_space[i], env, arg)

    return Tuple(updated_space)


@transform_space.register(Dict)
def _transform_space_dict(
    space: Dict, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
):
    assert isinstance(args, dict)
    updated_space = deepcopy(env.action_space)

    for arg in args:
        if is_nestable(space[arg]):
            transform_nestable_space(space[arg], updated_space, arg, args[arg], env)
        else:
            updated_space[arg] = transform_space(space[arg], env, args.get(arg))
    return updated_space


@singledispatch
def transform_nestable_space(
    original_space: gym.Space,
    space: gym.Space,
    space_key: str,
    args: FuncArgType[TypingTuple[int, int]],
    env=None,
):
    """Transform nestable space with the provided args."""


@transform_nestable_space.register(Dict)
def _transform_nestable_dict_space(
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
            transform_nestable_space(
                original_space[arg], updated_space, arg, args[arg], env
            )
        else:
            updated_space[arg] = transform_space(
                original_space[arg], env, args.get(arg)
            )


@transform_nestable_space.register(Tuple)
def _transform_nestable_tuple_space(
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
            transform_nestable_space(
                original_space[i], updated_space[idx_to_update], i, args[i], env
            )
        else:
            updated_space[idx_to_update][i] = transform_space(
                original_space[i], env, arg
            )

    if isinstance(updated_space[idx_to_update], list):
        updated_space[idx_to_update] = Tuple(updated_space[idx_to_update])
