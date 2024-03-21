from gymnasium import Wrapper
from gymnasium.core import Env

class FrameSkippingWrapper(Wrapper):
    """
    Wraps an environment and skips a number of frames while holding the same action.
    """
    
    def __init__(self, env: Env, skip_amount: int = 4):
        """
        Initialize the frame skipping wrapper
        :param env: The environment that's wrapped
        :param env: The number of frames to skip each step
        """
        super().__init__(env)

        if not skip_amount > 0:
            raise ValueError("At least one frame has to be skipped")
        
        self.skip_amount = skip_amount
 
    def step(self, action):
        result = None
        for _ in range(self.skip_amount):
            result = self.env.step(action)
        return result
