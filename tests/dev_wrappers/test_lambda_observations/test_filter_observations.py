import pytest

import gym
import numpy as np
from gym.spaces import Box, Discrete, Dict, Tuple
from gym.wrappers import filter_observations_v0
from tests.dev_wrappers.test_lambda_observations.mock_data_observation import (
    DISCRETE_VALUE,
    NUM_ENVS,
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


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (
            TestingEnv(observation_space=Dict(obs=Box(-1, 1, ()), time=Discrete(3))),
            {"obs": True, "time": False}
        ),
    ]
    )
def test_dict_filter_observation_v0(env, args):
    """Test correct filtering of `Dict` observation space."""
    wrapped_env = filter_observations_v0(env, args)

    assert wrapped_env.observation_space.get('obs', False)
    assert not wrapped_env.observation_space.get('time', False)

    obs, _, _, _ = wrapped_env.step(0)

    assert obs.get('obs', False)
    assert not obs.get('time', False)


@pytest.mark.parametrize(
    ("env", "args", "filtered_space_size"),
    [
        (
            TestingEnv(observation_space=Tuple([Box(-1, 1, ()), Box(-2, 2, ()), Discrete(3)])),
            [True, False, True],
            2
        ),
    ]
    )
def test_tuple_filter_observation_v0(env, args, filtered_space_size):
    """Test correct filtering of `Tuple` observation space."""
    wrapped_env = filter_observations_v0(env, args)

    assert len(wrapped_env.observation_space) == filtered_space_size

    obs, _, _, _ = wrapped_env.step(DISCRETE_VALUE)

    assert len(obs) == filtered_space_size

    assert isinstance(obs[0], np.ndarray)
    assert isinstance(obs[1], int)
