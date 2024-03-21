from gymnasium.wrappers.transform_reward import TransformReward
from gymnasium import Env
from math import tanh

def clip_reward(env: Env, scale: int = 1):
    """Wrapper to transform all rewards
        - scale == 0:  Rewards greater than 1 are transformed into 1 and rewards smaller than -1 are transformed into -1
        - otherwise:   Rewards are divided by the scale and then transformed with tanh
    """
    if scale == 0:
        return TransformReward(env, lambda r: 1 if r > 1 else -1 if r < -1 else r)
    else:
        return TransformReward(env, lambda r: tanh(r/scale))
