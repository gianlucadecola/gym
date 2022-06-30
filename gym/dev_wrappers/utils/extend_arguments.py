"""A set of utility functions for lambda wrappers."""
from functools import singledispatch
from typing import Sequence, Union

from gym.spaces import Dict, Space, Tuple, Box


def is_iterable_args(args: Union[list, dict, tuple]):
    return isinstance(args, list) or isinstance(args, dict)


@singledispatch
def extend_args(space: Space, extended_args: dict, args: dict, space_key: str):
    ...

@extend_args.register(Box)
def _extend_args_box(space: Space, extended_args: list, args: Sequence, space_idx: int):
    return (*args, space.low, space.high)


@extend_args.register(Tuple)
def _extend_args_tuple(space: Space, extended_args: list, args: Sequence, space_idx: int):
    extended_args = [None for _ in space]

    if args[space_idx] is None:
        return

    args = args[space_idx]
    space = space[space_idx]

    if is_iterable_args(args):
        extended_args[space_idx] = [None for _ in space]
        for i in range(len(args)):
            extend_args(space, extended_args[space_idx], args, i)
    else:
        extended_args[space_idx] = (*args, space.low, space.high)


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

    if is_iterable_args(args):
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