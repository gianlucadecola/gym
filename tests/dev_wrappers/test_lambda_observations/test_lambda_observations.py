from faulthandler import disable

import py
import gym
import pytest

from gym.spaces import Dict, Tuple, Box, MultiDiscrete, MultiBinary, Discrete
from gym.wrappers import (
    reshape_observations_v0

)
from tests.dev_wrappers.utils import contains_space, TestingEnv
from tests.dev_wrappers.test_lambda_observations.mock_data import (
    SEED,
    NUM_STEPS,
    NEW_BOX_DIM,
    NEW_BOX_DIM_IMPOSSIBLE,
    TESTING_BOX_OBSERVATION_SPACE
)

NUM_ENVS = 3
ENVS = [
    gym.make('CartPole-v1'),    # Box(np.zeros(4), np.ones(4), (4,), float32)
    gym.make('CarRacing-v1'),   # Box(0, 255, (96, 96, 3), uint8)
    gym.make('FrozenLake-v1'),  # Discrete(16)
    gym.make('Blackjack-v1'),   # Tuple(Discrete(32), Discrete(11), Discrete(2))
    gym.vector.make('CartPole-v1', num_envs=NUM_ENVS),
    gym.vector.make('CarRacing-v1', num_envs=NUM_ENVS),
    gym.vector.make('FrozenLake-v1', num_envs=NUM_ENVS),
    gym.vector.make('Blackjack-v1', num_envs=NUM_ENVS),
    TestingEnv(observation_space=MultiDiscrete([5, 3])),
    TestingEnv(observation_space=MultiBinary(10)),
    TestingEnv(observation_space=MultiBinary([3, 4])),
    TestingEnv(observation_space=Dict()),
    TestingEnv(observation_space=Dict()),
    TestingEnv(observation_space=Tuple([]))
]
IMAGE_ENVS = [env for env in ENVS if contains_space(env.observation_space, Box)]
COMPOSITE_ENVS = [env for env in ENVS if isinstance(env.observation_space, (Dict, Tuple))]


@pytest.mark.parametrize(
    ("env", "args",),
    [
        (
            TestingEnv(observation_space=TESTING_BOX_OBSERVATION_SPACE),
            NEW_BOX_DIM,     
        ),
        (
            gym.make('CarRacing-v1'), # Box(0, 255, (96, 96, 3), uint8)
            (96, 48, 6)
        )      
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
        reshape_observations_v0(
            env, 
            NEW_BOX_DIM_IMPOSSIBLE)


@pytest.mark.parametrize(
    ("env", "args",),
    [
        (
            TestingEnv(observation_space=Dict(obs=Box(0, 1, (96, 96, 3)), time=Discrete(10))),
            {"obs": (96, 36, 8)}
        ),
        (
            TestingEnv(observation_space=Dict(obs=Box(0, 1, (96, 96, 3)), obs2=Box(0, 1, (96, 96, 3)))),
            {"obs": (96, 36, 8)}
        ),
        (
            TestingEnv(observation_space=Dict(obs=Box(0, 1, (96, 96, 3)), obs2=Box(0, 1, (96, 96, 3)))),
            {"obs": (96, 36, 8), "obs2": (36, 96, 8)}
        ),

    ],
)
def test_reshape_observations_dict_v0(env, args):
    wrapped_env = reshape_observations_v0(env, args)
    wrapped_env.reset(seed=SEED)

    for k in wrapped_env.observation_space.keys():
        if k in args:
            assert wrapped_env.observation_space[k].shape == args[k]
        else:
            assert wrapped_env.observation_space[k].shape == env.observation_space[k].shape