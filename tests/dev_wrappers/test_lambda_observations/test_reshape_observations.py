import pytest

import gym
from gym.wrappers import reshape_observations_v0
from tests.dev_wrappers.test_lambda_observations.mock_data_observation import (
    NEW_BOX_DIM,
    NEW_BOX_DIM_IMPOSSIBLE,
    NUM_STEPS,
    SEED,
    TESTING_BOX_OBSERVATION_SPACE,
    TESTING_DICT_OBSERVATION_SPACE,
    TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE,
    TESTING_NESTED_DICT_ACTION_SPACE,
    TESTING_TUPLE_OBSERVATION_SPACE,
)
from tests.dev_wrappers.utils import TestingEnv

NUM_ENVS = 3


@pytest.mark.parametrize(
    (
        "env",
        "args",
    ),
    [
        (
            TestingEnv(observation_space=TESTING_BOX_OBSERVATION_SPACE),
            NEW_BOX_DIM,
        ),
        (gym.make("CarRacing-v1"), (96, 48, 6)),  # Box(0, 255, (96, 96, 3), uint8)
    ],
)
def test_reshape_observations_box_v0(env, args):
    """Test correct reshaping of box observation spaces."""
    wrapped_env = reshape_observations_v0(env, args)
    wrapped_env.reset(seed=SEED)

    assert wrapped_env.observation_space.shape == args

    for _ in range(NUM_STEPS):
        action = wrapped_env.action_space.sample()
        obs, *res = wrapped_env.step(action)
        assert obs in wrapped_env.observation_space


def test_reshape_observations_box_impossible_v0():
    """Test wrong new shape raises ValueError.

    A wrong new shape is a shape that can not be
    obtained from the original shape.
    """
    env = TestingEnv(observation_space=TESTING_BOX_OBSERVATION_SPACE)

    with pytest.raises(ValueError):
        reshape_observations_v0(env, NEW_BOX_DIM_IMPOSSIBLE)


@pytest.mark.parametrize(
    (
        "env",
        "args",
    ),
    [
        (
            TestingEnv(observation_space=TESTING_DICT_OBSERVATION_SPACE),
            {"key_1": NEW_BOX_DIM},
        ),
        (
            TestingEnv(observation_space=TESTING_DICT_OBSERVATION_SPACE),
            {"key_1": NEW_BOX_DIM, "key_2": NEW_BOX_DIM},
        ),
        (TestingEnv(observation_space=TESTING_DICT_OBSERVATION_SPACE), {}),
    ],
)
def test_reshape_observations_dict_v0(env, args):
    """Test reshaping `Dict` observation spaces.

    Tests whether `Dict` observation spaces are correctly reshaped.
    Expected behaviour is that the reshape observation
    space matches the shape provided in `args`.
    """
    wrapped_env = reshape_observations_v0(env, args)
    wrapped_env.reset(seed=SEED)

    for k in wrapped_env.observation_space.keys():
        if k in args:
            assert wrapped_env.observation_space[k].shape == args[k]
        else:
            assert (
                wrapped_env.observation_space[k].shape == env.observation_space[k].shape
            )


@pytest.mark.parametrize(
    (
        "env",
        "args",
    ),
    [
        (
            TestingEnv(observation_space=TESTING_NESTED_DICT_ACTION_SPACE),
            {"nested": {"nested": NEW_BOX_DIM}},
        ),
        (
            TestingEnv(observation_space=TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE),
            {"nested": {"nested": {"nested": NEW_BOX_DIM}}},
        ),
    ],
)
def test_reshape_observations_nested_dict_v0(env, args):
    """Test reshaping nested `Dict` observation spaces.

    Tests whether nested `Dict` observation spaces are correctly reshaped.
    Expected behaviour is that the reshape observation
    space matches the shape provided in `args`.
    """
    wrapped_env = reshape_observations_v0(env, args)
    wrapped_env.reset(seed=SEED)

    nested_arg = args["nested"]
    nested_space = wrapped_env.observation_space["nested"]
    while isinstance(nested_arg, dict):
        nested_arg = nested_arg["nested"]
        nested_space = nested_space["nested"]

    assert nested_space.shape == nested_arg


def test_reshape_observations_tuple_v0():
    """Test reshaping `Tuple` observation spaces.

    Tests whether `Tuple` observation spaces are correctly reshaped.
    Expected behaviour is that the reshape observation
    space matches the shape provided.
    """
    env = TestingEnv(observation_space=TESTING_TUPLE_OBSERVATION_SPACE)
    args = [None, NEW_BOX_DIM]

    wrapped_env = reshape_observations_v0(env, args)
    wrapped_env.reset(seed=SEED)

    for i, arg in enumerate(args):
        if arg:
            assert wrapped_env.observation_space[i].shape == arg
        else:
            assert (
                wrapped_env.observation_space[i].shape == env.observation_space[i].shape
            )
