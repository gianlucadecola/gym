import pytest

from collections import OrderedDict
from gym.spaces.utils import apply_function
from gym.spaces import Box, Dict, Tuple


@pytest.mark.parametrize(
    ('space', 'fn', 'x', 'args', 'expected_output'),
    (
        (
            Box(-1, 1, (1,)),
            lambda x, arg: x * arg,
            1,
            10,
            10
        ),
        (
            Dict(left_arm=Box(-1, 1, (1,)), right_arm=Box(-1, 1, (1,))),
            lambda x, arg: x * arg,
            {"left_arm": 1, "right_arm": 1},
            {"left_arm": 10, "right_arm": -10},
            OrderedDict([('left_arm', 10), ('right_arm', -10)])
        ),
        (
            Tuple([Box(-1, 1, (1,)), Box(-1, 1, (1,))]),
            lambda x, arg: x * arg,
            [1, 1],
            [10, -10],
            (10, -10)
        )
    )
)
def test_apply_function(space, fn, x, args, expected_output):
    assert apply_function(space, x, fn, args) == expected_output
