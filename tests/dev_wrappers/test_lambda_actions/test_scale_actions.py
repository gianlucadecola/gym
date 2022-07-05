"""Test suite for scale_actions_v0."""
from typing import Sequence
import numpy as np
import pytest

import gym
from gym.spaces import Dict
from gym.dev_wrappers.lambda_action import scale_actions_v0
from tests.dev_wrappers.test_lambda_actions.mock_data_actions import (
    NUM_ENVS,
    DISCRETE_ACTION,
    BOX_HIGH,
    NESTED_BOX_HIGH,
    NEW_BOX_HIGH,
    NEW_BOX_LOW,
    NEW_NESTED_BOX_HIGH,
    NEW_NESTED_BOX_LOW,
    SEED,
    TESTING_NESTED_DICT_ACTION_SPACE,
    TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE,
    TESTING_TUPLE_ACTION_SPACE,
    TESTING_NESTED_TUPLE_ACTION_SPACE,
    TESTING_DOUBLY_NESTED_TUPLE_ACTION_SPACE
)
from tests.dev_wrappers.utils import TestingEnv


@pytest.mark.parametrize(
    ("env", "args", "action", "scaled_action"),
    [
        (
            gym.make('BipedalWalker-v3'),
            (-0.5, 0.5),
            np.array([1, 1, 1, 1]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ),
    ],
)
def test_scale_actions_v0_box(env, args, action, scaled_action):
    """Test action rescaling.

    Scale action wrapper allow to rescale action
    to a new range.
    Supposed the old action space is of type
    `Box(-1, 1, (1,))` and we rescale to
    `Box(-0.5, 0.5, (1,))`, an action on the wrapped
    environment with value `0.5` will be seen by the
    environment as `1.0`.  TODO: maybe explain this better.
    """
    env.reset(seed=SEED)
    obs, _, _, _ = env.step(action)

    env.reset(seed=SEED)
    wrapped_env = scale_actions_v0(env, args)

    obs_scaled, _, _, _ = wrapped_env.step(scaled_action)

    assert np.alltrue(obs == obs_scaled)


@pytest.mark.parametrize(
    ("env", "args", "action", "scaled_action"),
    [
        (
            gym.vector.make('BipedalWalker-v3', num_envs=NUM_ENVS),
            (-0.5, 0.5),
            np.tile(np.array([1, 1, 1, 1]), (NUM_ENVS, 1)),
            np.tile(np.array([0.5, 0.5, 0.5, 0.5]), (NUM_ENVS, 1)),
        ),
    ],
)
def test_scale_action_v0_within_vector(env, args, action, scaled_action):
    """Tests scale action in vectorized environments.

    Tests if action is correctly rescaled in vectorized environment.
    """
    env.reset(seed=SEED)
    obs, _, _, _ = env.step(action)

    env.reset(seed=SEED)
    wrapped_env = scale_actions_v0(env, args)
    obs_scaled, _, _, _ = wrapped_env.step(scaled_action)

    assert np.alltrue(obs == obs_scaled)


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
                "box": NEW_BOX_HIGH,
                "discrete": DISCRETE_ACTION,
                "nested": {"nested": NEW_NESTED_BOX_HIGH},
            },
        ),
                (
            TestingEnv(action_space=TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "nested": {"nested": {"nested": (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)}},
            },
            {
                "box": NEW_BOX_HIGH,
                "discrete": DISCRETE_ACTION,
                "nested": {"nested": {"nested": NEW_NESTED_BOX_HIGH}},
            },
        )
    ],
)
def test_scale_actions_v0_nested_dict(env, args, action):
    """Test action rescaling for nested `Dict` action spaces."""
    wrapped_env = scale_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["box"] == BOX_HIGH

    nested_action = executed_actions["nested"]
    while isinstance(nested_action, Dict):
        nested_action = nested_action["nested"]
    assert nested_action == NESTED_BOX_HIGH



def test_scale_actions_v0_tuple():
    """Test action rescaling for `Tuple` action spaces."""
    env = TestingEnv(action_space=TESTING_TUPLE_ACTION_SPACE)
    args = [None, (NEW_BOX_LOW, NEW_BOX_HIGH)]
    action = [DISCRETE_ACTION, NEW_BOX_HIGH]

    wrapped_env = scale_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions[0] == action[0]
    assert executed_actions[1] == BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_NESTED_TUPLE_ACTION_SPACE),
            [None, [None, (NEW_BOX_LOW, NEW_BOX_HIGH)]],
            [BOX_HIGH, [DISCRETE_ACTION, NEW_BOX_HIGH]]
        ),
        (
            TestingEnv(action_space=TESTING_DOUBLY_NESTED_TUPLE_ACTION_SPACE),
            [None, [None, [None, (NEW_BOX_LOW, NEW_BOX_HIGH)]]],
            [BOX_HIGH, [DISCRETE_ACTION, [DISCRETE_ACTION, NEW_BOX_HIGH]]]
        )
    ],
)
def test_scale_actions_v0_nested_tuple(env, args, action):
    """Test action rescaling for nested `Tuple` action spaces."""
    wrapped_env = scale_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    nested_action = executed_actions[-1]
    while isinstance(nested_action, Sequence):
        nested_action = nested_action[-1]


    assert executed_actions[0] == BOX_HIGH
    assert nested_action == NESTED_BOX_HIGH
