from typing import Any, SupportsFloat

from gymnasium import Wrapper
from gymnasium.core import Env, WrapperObsType, WrapperActType


class RecordEpisodeStatisticsWrapper(Wrapper):
    """
    Record episode statistics for logging
    """

    def __init__(self, env: Env):
        """
        Initialize the frame skipping wrapper
        :param env: The environment that's wrapped
        :param env: The number of frames to skip each step
        """
        super().__init__(env)

        self.total_reward = 0
        self.steps = 0
        self.episode = 0

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.steps += 1
        self.total_reward += reward

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        info['steps'] = self.steps
        info['reward'] = self.total_reward
        info['episode'] = self.episode

        self.episode += 1
        self.steps = 0
        self.total_reward = 0

        return obs, info
