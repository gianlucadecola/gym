import gym
import pytest

from gym.spaces import Dict, Tuple, Box, MultiDiscrete, MultiBinary, Discrete
from gym.wrappers import (
    resize_observations_v0
)
from tests.dev_wrappers.utils import contains_space, TestingEnv
from tests.dev_wrappers.test_lambda_observations.mock_data_observation import (
    SEED,
    NUM_STEPS,
    NEW_BOX_DIM,
    NEW_BOX_DIM_IMPOSSIBLE,
    TESTING_BOX_OBSERVATION_SPACE,
    TESTING_DICT_OBSERVATION_SPACE,
    TESTING_NESTED_DICT_ACTION_SPACE,
    TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE,
    TESTING_TUPLE_OBSERVATION_SPACE
)


@pytest.mark.parametrize(
    ("env", "args",),
    [
        (
            gym.make('CarRacingDiscrete-v1'), # Box(0, 255, (96, 96, 3), uint8)
            (32, 32, 3)
        )      
    ],
)
def test_ressize_observations_box_v0(env, args):
    """Test correct reshaping of box observation spaces."""
    wrapped_env = resize_observations_v0(env, args)
    wrapped_env.reset(seed=SEED)

    assert wrapped_env.observation_space.shape == args

    action = wrapped_env.action_space.sample()
    obs, *res = wrapped_env.step(action)
    assert obs.shape == args


