import torch
import numpy as np
import json

from DQN.Logging.cli import CliLogger
from DQN.Logging.wandblogger import WandBLogger


class EpisodeLog:
    """Class keeping track of all episode metrics and logging them regularly to all selected loggers"""

    def __init__(self, log_interval: int, env_id: str = "", wandb_config: dict = None,
                 wandb_tags: list[str] = None) -> None:
        self.episodes = []
        self.log_interval = log_interval

        self.loggers = [CliLogger(max_rows=10)]

        if wandb_config is not None:
            self.loggers.append(WandBLogger(env_id, wandb_config, wandb_tags))

    def update(self, steps: int, lr: float, n_updates: int, episode_reward: float, average_q: float,
               epsilon: float, beta: float):
        """Adds another episode to the list and logs them if log interval has been reached"""

        if len(self.episodes) > 0:
            episode_length = steps - self.episodes[-1][0]
        else:
            episode_length = steps

        episode = (steps, episode_length, lr, n_updates, episode_reward, average_q, epsilon, beta)

        self.episodes.append(episode)

        if len(self.episodes) % self.log_interval == 0:
            self.__log__()

    def watch(self, model: torch.nn.Module):
        pass
        # if wandb.run is not None:
        #    wandb.watch(model)

    def serialize(self, file):
        with open(file=file, encoding='utf-8', mode='a') as file:
            json.dump(self.episodes, file)

    def __log__(self):
        """logs metrics of last log_interval episodes to all selected loggers"""
        episode = self.episodes[-1]
        last_episodes = self.episodes[-self.log_interval:]

        mean_episode_length = np.mean([item[1] for item in last_episodes])
        mean_episode_reward = np.mean([item[4] for item in last_episodes])

        for logger in self.loggers:
            logger.log(
                episodes=len(self.episodes),
                steps=episode[0],
                mean_episode_length=mean_episode_length,
                lr=episode[2],
                n_updates=episode[3],
                mean_episode_reward=mean_episode_reward,
                average_q=episode[5],
                epsilon=episode[6],
                beta=episode[7]
            )
