import torch


class Logger:
    """Interface for all loggers to be used by the episode log ensuring they provide basic functionality"""

    def __init__(self) -> None:
        pass
    
    def log(self, episodes: int, steps: int, mean_episode_length: int, lr: float, n_updates: int,
            mean_episode_reward: float, average_q: float, epsilon: float, beta: float):
        raise NotImplementedError("Logger did not implement logging!")
