"""Test lambda reward wrapper."""
import pytest

import gym
from gym.wrappers import clip_rewards_v0, lambda_reward_v0

SEED = 1
NUM_STEPS = 5


@pytest.mark.parametrize(
    (
        "reward_fn",
        "expected_reward",
    ),
    [(lambda r: 2 * r + 1, 3)],
)
def test_lambda_reward(reward_fn, expected_reward):
    env = gym.make("CartPole-v1")
    env = lambda_reward_v0(env, reward_fn)
    env.reset()
    _, rew, _, _ = env.step(0)
    assert rew == expected_reward


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected_reward"),
    [(None, 0.5, 0.5), (0, None, 1), (0, 0.5, 0.5)],
)
def test_clip_reward(lower_bound, upper_bound, expected_reward):
    env = gym.make("CartPole-v1")
    env = clip_rewards_v0(env, lower_bound, upper_bound)
    env.reset()
    _, rew, _, _ = env.step(0)

    assert rew == expected_reward


@pytest.mark.parametrize(("lower_bound", "upper_bound"), [(None, None), (1, -1)])
def test_clip_reward_incorrect_params(lower_bound, upper_bound):
    env = gym.make("CartPole-v1")
    with pytest.raises(Exception):
        env = clip_rewards_v0(env, lower_bound, upper_bound)
