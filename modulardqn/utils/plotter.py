import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import torch


class Plotter:
    def __init__(self, num_actions: int, max_steps=120, smooth_over_last=10):
        self.q_values = np.empty(shape=(0, num_actions))
        self.graph_values = np.empty(shape=(0, num_actions))

        self.max_steps = max_steps
        self.smooth_over_last = smooth_over_last

        self.fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
        self.ax = plt.axes(xlim=(0, self.max_steps), ylim=(0, 0))
        plt.xlabel("Steps")
        plt.ylabel("Q-Value")

    def add_q_values(self, q_values: torch.Tensor = None):
        # Use provided q_values or use last ones if random action is performed
        if q_values is not None:
            q_values = q_values.numpy()
        else:
            if self.q_values.size > 0:
                q_values = self.q_values[-1].reshape(1, -1)
            else:
                q_values = np.zeros((1, self.q_values.shape[1]))

        self.q_values = np.append(self.q_values, q_values, axis=0)

        # Smoothen graph by averaging over last values
        mean_values = np.mean(self.q_values[-self.smooth_over_last:], axis=0).reshape(1, -1)
        self.graph_values = np.append(self.graph_values, mean_values, axis=0)

    def __update_plot(self, end_index, *lines: list[Line2D]):
        start_index = max(end_index - self.max_steps, 0)
        xs = np.linspace(start=start_index, stop=end_index, num=end_index - start_index, dtype=int)
        ys = self.graph_values[xs]

        # Update x-axis if necessary
        if end_index > self.max_steps:
            self.ax.set_xlim(start_index, end_index)

        # Update y-axis if necessary
        if ys.size > 0:
            ylim = self.ax.get_ylim()
            if np.max(ys[-1]) > ylim[1]:
                self.ax.set_ylim(ylim[0], np.max(ys[-1]) + 0.1)

            if np.min(ys[-1]) < ylim[0]:
                self.ax.set_ylim(np.min(ys[-1]) - 0.1, ylim[1])

            # Update lines with data to display on current frame
            for index in range(len(lines)):
                lines[index].set_data(xs, ys[:, index])

        return lines

    def save_animation(self, path: str = "graph.mp4", fps: int = 30, action_names: list[str] = None):
        # Creates lines for all actions at once (don't touch)
        lines = self.ax.plot(np.empty((0, self.q_values.shape[1])), np.empty((0, self.q_values.shape[1])))

        # Name actions in legend accordingly if provided in the env's metadata
        if action_names is None:
            action_names = [f"Action {i}" for i in range(self.q_values.shape[1])]

        self.fig.legend(action_names)

        plt.title('Action-Value Function')

        self.animation = animation.FuncAnimation(fig=self.fig, func=self.__update_plot, fargs=(lines),
                                                 frames=self.q_values.shape[0], repeat=False, interval=1_000 / fps,
                                                 blit=True)

        try:
            self.animation.save(path)
            print("Stored graph to graph.mp4")
        except Exception:
            print("Looks like the graph can not be saved properly! This does not work on Windows")
            raise Exception
