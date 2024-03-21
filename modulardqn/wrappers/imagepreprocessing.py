from typing import Tuple

from gymnasium import Wrapper
from gymnasium import Env
import numpy as np
import cv2


class ImagePreprocessingWrapper(Wrapper):

    def __init__(self, env: Env, obs_size: Tuple[int, int, int]):
        super().__init__(env)
        self.num_stack = obs_size[0] if obs_size is not None else 4
        self.size = obs_size[1:] if obs_size is not None else None
        self.frames = []

    def step(self, action):
        assert (len(self.frames) == self.num_stack)

        pixels, reward, terminated, truncated, info = super().step(action)

        if self.size is None:
            self.size = pixels.shape

        # resize obs if needed
        if pixels.shape != self.size:
            pixels = cv2.resize(src=pixels, dsize=self.size)

        self.frames.pop(0)
        self.frames.append(cv2.cvtColor(src=pixels, code=cv2.COLOR_RGB2GRAY))

        return np.array(self.frames), reward, terminated, truncated, info

    def reset(self, *, seed: int = None, options: dict[str, any] = None):
        pixels, info = super().reset(seed=seed, options=options)

        transformed_pixels = cv2.cvtColor(src=pixels, code=cv2.COLOR_RGB2GRAY)

        self.frames = [transformed_pixels for _ in range(self.num_stack)]

        return np.array(self.frames), info
