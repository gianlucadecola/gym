import pytest

import gym
import numpy as np
from gym.wrappers import observations_dtype_v0
from tests.dev_wrappers.mock_data import (
    DISCRETE_ACTION,
    NUM_ENVS,
    DICT_SPACE,
    NESTED_DICT_SPACE,
    DOUBLY_NESTED_DICT_SPACE
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


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (TestingEnv(observation_space=DICT_SPACE), {"box": np.dtype('float32')}),
        (TestingEnv(observation_space=DICT_SPACE), {
            "box": np.dtype('float32'), "box2": np.dtype('float32')
            }),
        (TestingEnv(observation_space=DICT_SPACE), {
            "box": np.dtype('uint8'), "box2": np.dtype('uint8')
            }),
    ]
)
def test_observation_dtype_v0_dict(env, args):
    """Test correct dtype is applied to observation."""
    wrapped_env = observations_dtype_v0(env, args)
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    for subspace in obs:
        if subspace in args:
            assert obs[subspace].dtype == args[subspace]


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (TestingEnv(observation_space=NESTED_DICT_SPACE),
        {"nested": {"nested": np.dtype('int32')}}),
        
        (TestingEnv(observation_space=NESTED_DICT_SPACE),
        {"box": np.dtype('uint8'), "nested": {"nested": np.dtype('int32')}}),
        
        (TestingEnv(observation_space=DOUBLY_NESTED_DICT_SPACE),
        {"nested": {"nested": {"nested": np.dtype('int32')}}}),
    ]
)
def test_observation_dtype_v0_nested_dict(env, args):
    """Test correct dtype is applied to observation."""
    wrapped_env = observations_dtype_v0(env, args)
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    if "box" in args:
        assert obs["box"].dtype == args["box"]

    dict_subspace = obs["nested"]
    dict_args = args["nested"]
    while "nested" in dict_subspace:
        dict_subspace = dict_subspace["nested"]
        dict_args = dict_args["nested"]
    assert dict_subspace.dtype == dict_args