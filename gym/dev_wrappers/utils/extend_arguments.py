"""A set of utility functions for lambda wrappers."""
import gym
from functools import singledispatch
from typing import Callable, Sequence, Union
from typing import Tuple as TypingTuple
from gym.dev_wrappers import FuncArgType

from gym.spaces import Dict, Space, Tuple, Box


def is_iterable_args(args: Union[list, dict, tuple]):
    return isinstance(args, list) or isinstance(args, dict)


@singledispatch
def extend_nestable_args(space, updated_space, i, args, fn):
    ...


@singledispatch
def extend_args(space: Space, args: dict, fn: Callable):
    ...


@extend_args.register(Box)
def _extend_args_box(space: Space, args: Sequence, fn: Callable):
    if args is None:
        return (space.low, space.high, space.low, space.high)
    return (*args, space.low, space.high)


@extend_args.register(Tuple)
def _extend_args_tuple(
    space: Tuple, args: FuncArgType[TypingTuple[int, int]], fn: Callable
):
    assert isinstance(args, Sequence)
    assert len(space) == len(args)

    updated_args = [arg for arg in args]

    for i, arg in enumerate(args):
        if is_iterable_args(arg):
            extend_nestable_args(space[i], updated_args, i, args[i], fn)
        else:
            updated_args[i] = fn(space[i], arg, fn)
    return updated_args


@extend_args.register(Dict)
def _extend_args_dict(space: Tuple, args: FuncArgType[TypingTuple[int, int]], fn: Callable):
    """Extend args for rescaling actions.

    Action space args needs to be extended in order
    to correctly rescale the actions.
    i.e. args before: {"body":{"left_arm": (-0.5,0.5)}, ...}
    args after: {"body":{"left_arm": (-0.5,0.5,-1,1)}, ...}
    where -1, 1 was the old action space bound.
    old action space is needed to rescale actions.
    """
    extended_args = {}
  
    for arg in args:
        if is_iterable_args(args):
            extend_nestable_args(space[arg], extended_args[arg], args, arg)
        else:
            extended_args[arg] = fn(space[arg], arg, fn)
    return extended_args


@extend_nestable_args.register(Tuple)
def _extend_nestable_tuple_args(space: Space, extended_args: dict, space_idx: int, args, fn):
    if args[space_idx] is None:
        return

    args = args[space_idx]
    space = space[space_idx]

    if is_iterable_args(args):
        extend_args(space, extended_args[space_idx], fn)
    else:
        extended_args[space_idx] = fn(space[args], args, fn)


@extend_nestable_args.register(Dict)
def _extend_nestable_dict_args(space: Space, extended_args: dict, space_key: int, args, fn):
    if space_key not in args:
        return extended_args

    args = args[space_key]
    space = space[space_key]
  
    if is_iterable_args(args):
        extend_args(space, extended_args[space_key], fn)   
    else:
        extended_args[space_key] = fn(space[args], args, fn)
