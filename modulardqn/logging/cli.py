from prettytable import PrettyTable

from DQN.Logging.logger import Logger


class CliLogger(Logger):
    """
    Command line interface logger
    """

    def __init__(self, max_rows):
        super().__init__()

        self.max_rows = max_rows
        self.table = None
        self.clear()

    def clear(self):
        """
        Clears the table by creating a new identical one
        """
        self.table = PrettyTable(['Episodes', 'Time Steps', 'Mean Episode Length', 'Learning Rate', 'Number Of Updates',
                                  'Mean Episode Reward', 'Average Max Q-Value', 'Epsilon', 'Beta'])

    def log(self,
            episodes: int,
            steps: int,
            mean_episode_length: int,
            lr: float,
            n_updates: int,
            mean_episode_reward: float,
            average_q: float,
            epsilon: float,
            beta: float) -> None:
        """
        Adds a new row to the logging table and print the table if it reaches max length
        """
        self.table.add_row([episodes, steps, mean_episode_length, f"{lr:.6f}", n_updates, f"{mean_episode_reward:.4f}",
                            f"{average_q:.4f}", f"{epsilon:.2f}", f"{beta:.4f}"])

        if len(self.table.rows) == self.max_rows:
            print(self.table)
            self.clear()

    def print(self):
        """
        Print the current table
        """
        print(self.table)
