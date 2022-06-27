"""A set of utility functions for lambda wrappers."""
from copy import deepcopy
from functools import singledispatch
from typing import Callable, Sequence
from typing import Tuple as TypingTuple

import gym
from gym.dev_wrappers import FuncArgType
from gym.dev_wrappers.utils.grayscale_space import grayscale_space
from gym.dev_wrappers.utils.reshape_space import reshape_space
from gym.dev_wrappers.utils.resize_spaces import resize_space
from gym.dev_wrappers.utils.transform_space_bounds import transform_space_bounds
from gym.spaces import Dict, Space, Tuple


def is_nestable(space: Space):
    """Returns whether the input space can contains other spaces."""
    return isinstance(space, Tuple) or isinstance(space, Dict)


@singledispatch
def extend_args(space: Space, extended_args: dict, args: dict, space_key: str):
    ...


@extend_args.register(Tuple)
def _extend_args_tuple(space: Space, extended_args: list, args: Sequence, space_idx: int):  
    if args[space_idx] is None:
        extended_args.append(None)
        return

    args = args[space_idx]

    if isinstance(args, list):
        if len(extended_args) == space_idx:
            extended_args.append([])
        else:
            extended_args[space_idx] = []
        
        for i, arg in enumerate(args):
            extend_args(space[i], extended_args[i], args, i)

    elif isinstance(args, tuple):
        extended_args.append(
            (
                *args, 
                space[space_idx].low, 
                space[space_idx].high
            )
        )


@extend_args.register(Dict)
def _extend_args_dict(space: Space, extended_args: dict, args: dict, space_key: str):
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
            extend_args(space[space_key], extended_args[space_key], args, arg)
    else:
        assert len(args) == len(space[space_key].low) + len(
            space[space_key].high
        )
        extended_args[space_key] = (
            *args,
            *list(space[space_key].low),
            *list(space[space_key].high),
        )

    return extended_args


@singledispatch
def transform_nestable_space(
    original_space: gym.Space,
    space: gym.Space,
    space_key: str,
    args: FuncArgType[TypingTuple[int, int]],
    fn: Callable,
):
    """Transform nestable space with the provided args."""


@transform_nestable_space.register(Dict)
def _transform_nestable_dict_space(
    original_space: gym.Space,
    updated_space: gym.Space,
    arg: str,
    args: FuncArgType[TypingTuple[int, int]],
    fn: Callable,
):
    """Recursive function to process nested `Dict` spaces."""
    updated_space = updated_space[arg]

    for arg in args:
        if is_nestable(original_space[arg]):
            transform_nestable_space(
                original_space[arg], updated_space, arg, args[arg], fn
            )
        else:
            updated_space[arg] = fn(original_space[arg], args.get(arg), fn)


@transform_nestable_space.register(Tuple)
def _transform_nestable_tuple_space(
    original_space: gym.Space,
    updated_space: gym.Space,
    idx_to_update: int,
    args: FuncArgType[TypingTuple[int, int]],
    fn: Callable,
):
    """Recursive function to process nested `Tuple` spaces."""
    updated_space[idx_to_update] = [s for s in original_space]

    if args is None:
        return

    for i, arg in enumerate(args):
        if is_nestable(original_space[i]):
            transform_nestable_space(
                original_space[i], updated_space[idx_to_update], i, args[i], fn
            )
        else:
            updated_space[idx_to_update][i] = fn(original_space[i], arg, fn)

    if isinstance(updated_space[idx_to_update], list):
        updated_space[idx_to_update] = Tuple(updated_space[idx_to_update])


@grayscale_space.register(Tuple)
@resize_space.register(Tuple)
@reshape_space.register(Tuple)
@transform_space_bounds.register(Tuple)
def _process_space_tuple(
    space: Tuple, args: FuncArgType[TypingTuple[int, int]], fn: Callable
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


@grayscale_space.register(Dict)
@resize_space.register(Dict)
@reshape_space.register(Dict)
@transform_space_bounds.register(Dict)
def _process_space_dict(
    space: Dict, args: FuncArgType[TypingTuple[int, int]], fn: Callable
):
    assert isinstance(args, dict)
    updated_space = deepcopy(space)

    for arg in args:
        if is_nestable(space[arg]):
            transform_nestable_space(space[arg], updated_space, arg, args[arg], fn)
        else:
            updated_space[arg] = fn(space[arg], args.get(arg), fn)
    return updated_space
