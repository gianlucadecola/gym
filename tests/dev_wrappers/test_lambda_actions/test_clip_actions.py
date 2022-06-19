from typing import Iterable, Sequence
import pytest
import gym
import numpy as np

from tests.dev_wrappers.test_lambda_actions import (
    SEED,
    TESTING_DICT_ACTION_SPACE,
    NEW_BOX_LOW, 
    NEW_BOX_HIGH,
    TESTING_NESTED_DICT_ACTION_SPACE,
    TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE,
    NEW_NESTED_BOX_LOW, 
    NEW_NESTED_BOX_HIGH,
    TESTING_TUPLE_ACTION_SPACE,
    TESTING_NESTED_TUPLE_ACTION_SPACE,
    TESTING_DOUBLY_NESTED_TUPLE_ACTION_SPACE,
    TESTING_TUPLE_WITHIN_DICT_ACTION_SPACE,
    TESTING_DICT_WITHIN_TUPLE_ACTION_SPACE,
)

from tests.dev_wrappers.utils import TestingEnv
from gym.dev_wrappers.lambda_action import clip_actions_v0

@pytest.mark.parametrize(
    ("env", "args"),
    (
        [
            gym.make("MountainCarContinuous-v0"),
            (np.array([-0.5], dtype="float32"), np.array([0.5], dtype="float32")),
        ],
        [
            gym.make("BipedalWalker-v3"),
            (
                -0.5,
                0.5,
            ),
        ],
        [
            gym.make("BipedalWalker-v3"),
            (
                np.array([-0.5, -1, -1, -1], dtype="float32"),
                np.array([0.5, 0.5, 1, 1], dtype="float32"),
            ),
        ],
    ),
)
def test_clip_actions_v0_check_action_space(env, args):
    """Tests if the action space of the wrapped env is correctly clipped.

    This tests assert that the action space of the
    wrapped environment is clipped correctly according
    the args parameters.
    """
    action_space_before_wrapping = env.action_space

    wrapped_env = clip_actions_v0(env, args)

    assert np.equal(wrapped_env.action_space.low, args[0]).all()
    assert np.equal(wrapped_env.action_space.high, args[1]).all()
    assert action_space_before_wrapping == wrapped_env.env.action_space


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_DICT_ACTION_SPACE),
            {"box": (NEW_BOX_LOW, NEW_BOX_HIGH)},
            {"box": NEW_BOX_HIGH + 1},
        )
    ],
)
def test_clip_actions_v0_dict_action(env, args, action):
    """Checks Dict action spaces clipping.

    Check whether dictionaries action are
    correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["box"] == NEW_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_NESTED_DICT_ACTION_SPACE),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "nested": {"nested": (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)},
            },
            {
                "box": NEW_BOX_HIGH + 1,
                "discrete": 0,
                "nested": {"nested": NEW_NESTED_BOX_HIGH + 1},
            },   
        ),
        (
            TestingEnv(action_space=TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "nested": {
                    "nested": {
                        "nested": (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)
                    }
                },
            },
            {
                "box": NEW_BOX_HIGH + 1,
                "nested": {"nested": {"nested": NEW_NESTED_BOX_HIGH + 1}},
            },   
        )
    ],
)
def test_clip_actions_v0_nested_dict_action(env, args, action):
    """Checks Nested Dict action spaces clipping.

    Check whether nested dictionaries action are
    correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    nested_action = executed_actions["nested"]
    while isinstance(nested_action, dict):
        nested_action = nested_action["nested"]
    
    assert executed_actions["box"] == NEW_BOX_HIGH
    assert nested_action == NEW_NESTED_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_TUPLE_ACTION_SPACE),
            [None, (NEW_BOX_LOW, NEW_BOX_HIGH)],
            [0, NEW_BOX_HIGH + 1],
        )
    ],
)
def test_clip_actions_v0_tuple_action(env, args, action):
    """Checks Tuple action spaces clipping.

    Check whether tuples action are
    correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert np.alltrue(executed_actions == (0, NEW_BOX_HIGH))


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_NESTED_TUPLE_ACTION_SPACE),
            [
                (NEW_BOX_LOW, NEW_BOX_HIGH),
                [
                    None,
                    (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH),
                ]                
            ],
            [
                NEW_BOX_HIGH + 1,
                [
                    0,
                    NEW_NESTED_BOX_HIGH + 1
                ]
            ],   
        ),
        (
            TestingEnv(action_space=TESTING_DOUBLY_NESTED_TUPLE_ACTION_SPACE),
            [
                (NEW_BOX_LOW, NEW_BOX_HIGH),
                [
                    None,
                    [
                        (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)
                    ]
                ]                
            ],
            [
                NEW_BOX_HIGH + 1,
                [
                    0,
                    [
                        NEW_NESTED_BOX_HIGH + 1
                    ]
                ]
            ],   
        ),
    ],
)
def test_clip_actions_v0_nested_tuple_action(env, args, action):
    """Checks Nested Tuple action spaces clipping.

    Check whether nested tuples actions are
    correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    nested_action = executed_actions[-1]
    while isinstance(nested_action, tuple):
        nested_action = nested_action[-1]
    
    assert executed_actions[0] == NEW_BOX_HIGH
    assert nested_action == NEW_NESTED_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_DICT_WITHIN_TUPLE_ACTION_SPACE),
            [None, {"dict": (NEW_BOX_LOW, NEW_BOX_HIGH)}],
            [0, {"dict": NEW_BOX_HIGH + 1}],
        )
    ],
)
def test_clip_actions_v0_dict_within_tuple(env, args, action):
    """Checks Dict within Tuple action spaces clipping.

    Check whether a dict action space within a tuple action
    space is correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions[1]["dict"] == NEW_BOX_HIGH



@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_TUPLE_WITHIN_DICT_ACTION_SPACE),
            {"tuple": [(NEW_BOX_LOW, NEW_BOX_HIGH)]},
            {"discrete": 0, "tuple": [NEW_BOX_HIGH + 1]},
        )
    ],
)
def test_clip_actions_v0_tuple_within_dict(env, args, action):
    """Checks Tuple within Dict action spaces clipping.

    Check whether a Tuple action space within a Dict action
    space is correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["tuple"][0] == NEW_BOX_HIGH



@pytest.mark.parametrize(
    ("env_name", "args", "action_unclipped_env", "action_clipped_env"),
    (
        [
            "MountainCarContinuous-v0",
            (np.array([-0.5], dtype="float32"), np.array([0.5], dtype="float32")),
            np.array([0.5]),
            np.array([1]),
        ],
        [
            "BipedalWalker-v3",
            (
                -0.5,
                0.5,
            ),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([10, 10, 10, 10]),
        ],
        [
            "BipedalWalker-v3",
            (
                np.array([-0.5, -1, -1, -1], dtype="float32"),
                np.array([0.5, 0.5, 1, 1], dtype="float32"),
            ),
            np.array([0.5, 0.5, 1, 1]),
            np.array([10, 10, 10, 10]),
        ],
    ),
)
def test_clip_actions_v0(env_name, args, action_unclipped_env, action_clipped_env):
    """Tests if actions out of bound are correctly clipped.

    This tests check whether out of bound actions for the wrapped
    environment are correctly clipped.
    """
    env = gym.make(env_name)
    env.reset(seed=SEED)
    obs, _, _, _ = env.step(action_unclipped_env)

    env = gym.make(env_name)
    env.reset(seed=SEED)
    wrapped_env = clip_actions_v0(env, args)
    wrapped_obs, _, _, _ = wrapped_env.step(action_clipped_env)

    assert np.alltrue(obs == wrapped_obs)
