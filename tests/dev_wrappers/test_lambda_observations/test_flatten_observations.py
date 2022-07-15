import operator as op
from functools import reduce

import pytest

import gym
from gym.wrappers import flatten_observations_v0
from tests.dev_wrappers.mock_data import (
    DICT_SPACE,
    DISCRETE_ACTION,
    FLATTENEND_DICT_SIZE,
)
from tests.dev_wrappers.utils import TestingEnv


@pytest.mark.parametrize(
    ("env"),
    [
        gym.make("CarRacing-v2", continuous=False, disable_env_checker=True),
    ],
)
def test_flatten_observation_v0(env):
    """Test correct flattening of observation space."""
    flattened_shape = reduce(op.mul, env.observation_space.shape, 1)
    wrapped_env = flatten_observations_v0(env)
    wrapped_env.reset()

    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert wrapped_env.observation_space.shape[0] == flattened_shape
    assert obs.shape[0] == flattened_shape


@pytest.mark.parametrize(
    ("env", "flattened_size"),
    [(TestingEnv(observation_space=DICT_SPACE), FLATTENEND_DICT_SIZE)],
)
def test_dict_flatten_observation_v0(env, flattened_size):
    wrapped_env = flatten_observations_v0(env)
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert wrapped_env.observation_space.shape[0] == flattened_size
    assert obs.shape[0] == flattened_size
