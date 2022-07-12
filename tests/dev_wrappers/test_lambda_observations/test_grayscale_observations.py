import pytest

import gym
import numpy as np
from gym.spaces import Box, Discrete, Dict, Tuple
from gym.wrappers import grayscale_observations_v0
from tests.dev_wrappers.mock_data import (
    DISCRETE_ACTION,
    NUM_ENVS,
)
from tests.dev_wrappers.utils import TestingEnv


@pytest.mark.parametrize(
    ("env"),
    [gym.make("CarRacingDiscrete-v1")]
)
def test_grayscale_observation_v0(env):
    """Test correct transformation of observation in grayscale."""
    wrapped_env = grayscale_observations_v0(env)
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert len(obs.shape) == 2 # height and width. No more color dim


@pytest.mark.parametrize(
    ("env"),
    [gym.vector.make("CarRacingDiscrete-v1", num_envs=NUM_ENVS)]
)
def test_grayscale_observation_v0_vectorenv(env):
    """Test correct transformation of observation in grayscale."""
    wrapped_env = grayscale_observations_v0(env)
    obs, _, _, _ = wrapped_env.step([DISCRETE_ACTION] * NUM_ENVS)

    assert len(obs.shape) == 3 # height and width. No more color dim
    assert obs.shape[0] == NUM_ENVS
