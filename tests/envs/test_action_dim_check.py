import pytest

from gym import envs
from tests.envs.spec_list import SKIP_MUJOCO_WARNING_MESSAGE, skip_mujoco

ENVIRONMENT_IDS = ("HalfCheetah-v2",)


@pytest.mark.skipif(skip_mujoco, reason=SKIP_MUJOCO_WARNING_MESSAGE)
@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_serialize_deserialize(environment_id):
    env = envs.make(environment_id)
    env.reset()

    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step([0.1])

    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step(0.1)


@pytest.mark.parametrize(
    "environment_id",
    (
        "Acrobot-v1",
        "CartPole-v1",
        "MountainCar-v0",
        "Blackjack-v1",
        "CliffWalking-v0",
        "FrozenLake-v1",
        "Taxi-v3",
    ),
)
def test_discrete_actions_out_of_bound(environment_id):
    env = envs.make(environment_id)
    env.reset()

    action_space = env.action_space
    upper_bound = action_space.start + action_space.n - 1

    with pytest.raises(AssertionError):
        env.step(upper_bound + 1)
