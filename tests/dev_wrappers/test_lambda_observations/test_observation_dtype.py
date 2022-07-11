import pytest

import gym
import numpy as np
from gym.spaces import Box, Dict
from gym.wrappers import observations_dtype_v0
from tests.dev_wrappers.mock_data import (
    DISCRETE_ACTION,
    NUM_ENVS,
    SEED,
)
from tests.dev_wrappers.utils import TestingEnv


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (gym.make("CartPole-v1"), np.dtype('int32')),
        (gym.make("CartPole-v1"), np.dtype('float32')),
    ]
)
def test_observation_dtype_v0(env, args):
    """Test correct dtype is applied to observation."""
    wrapped_env = observations_dtype_v0(env, args)
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert obs.dtype == args


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (gym.vector.make("CartPole-v1", num_envs=NUM_ENVS), np.dtype('int32')),
        (gym.vector.make("CartPole-v1", num_envs=NUM_ENVS), np.dtype('float32')),
    ]
)
def test_observation_dtype_v0_within_vector(env, args):
    """Test correct dtype is applied to observation in vectorized envs."""
    wrapped_env = observations_dtype_v0(env, args)
    observations, _, _, _ = wrapped_env.step([DISCRETE_ACTION for _ in range(NUM_ENVS)])

    for obs in observations:
        assert obs.dtype == args
