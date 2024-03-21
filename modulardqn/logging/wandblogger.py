import wandb
import torch

from DQN.Logging.logger import Logger

class WandBLogger(Logger):
    def __init__(self, project: str, config: dict, tags: list[str]):
        """
        Initialize wandb run
        """
        super().__init__()

        if wandb.run is None:
            # replace all occurrences of illegal characters with '-'
            self.run = wandb.init(project=project.translate(str.maketrans(
                {
                    '/': '-',
                    '\\': '-',
                    '#': '-',
                    '?': '-',
                    '%': '-',
                    ':': '-'
                }
            )),
            config=config,
            tags=tags
            )

        # define custom step metric
        wandb.define_metric("Time Steps")

        # define all metrics to use the custom step metric
        wandb.define_metric("Total Time Steps", step_metric="Time Steps")
        wandb.define_metric("Mean Episode Length", step_metric="Time Steps")
        wandb.define_metric("Learning Rate", step_metric="Time Steps")
        wandb.define_metric("Number Of Updates", step_metric="Time Steps")
        wandb.define_metric("Mean Episode Reward", step_metric="Time Steps")
        wandb.define_metric("Average Max Q-Value", step_metric="Time Steps")
        wandb.define_metric("Epsilon", step_metric="Time Steps")

    def log(self,
            episodes: int,
            steps: int,
            mean_episode_length: int,
            lr: float,
            n_updates: int,
            mean_episode_reward: float,
            average_q: float,
            epsilon: float,
            beta: float
            ):
        """
        Log parameters
        """
        wandb.log(
            {
                "Episode": episodes,
                "Time Steps": steps,
                "Mean Episode Length": mean_episode_length,
                "Learning Rate": lr,
                "Number Of Updates": n_updates,
                "Mean Episode Reward": mean_episode_reward,
                "Average Max Q-Value": average_q,
                "Epsilon": epsilon,
                "Beta": beta
            }
        )

    def log_model(self, model: torch.nn.Module):
        """
        Log model (doesnt work currently but I dont understand)
        """
        wandb.log_artifact(model)
