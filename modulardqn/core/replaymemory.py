import sys
from collections import deque

import numpy as np
import torch

from DQN.Tools.sumtree import SumSegmentTree, MinSegmentTree


class ReplayMemory:
    def __init__(self, n: int, device, dtype, gamma: float, n_steps: int):
        """
        Initialize replay memory with capacity n
        :param n: replay memory capacity
        """
        self.capacity = n
        self.current_states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.is_not_terminals = []
        self.device = device
        self.index = 0

        self.gamma = gamma
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=n_steps)

        # set default dtype here
        self.dtype = dtype

    def sample(self, size: int):
        """
        Uniformly sample a random experience
        :param size: size of the minibatch
        :return: a list of uniformly random sampled experiences
        """
        indices = self.__sample__(size)

        return self.__create_batch__(indices)

    def store(self, current_state: np.array, action: int, reward: float, next_state: np.array, is_terminal: bool):
        """
        Store an experience in the memory buffer
        Delete the oldest experience if capacity is reached
        :param current_state: state the action performed in
        :param action: action that was performed in the state
        :param reward: reward observed by the agent
        :param next_state: state resulting from the execution of the action
        :param is_terminal: true if the experience leads to a terminal state
        """
        self.n_step_buffer.append((current_state, action, reward, next_state, is_terminal))

        rew, n_o, t = self.__create_n_step_transition__()
        obs, act = self.n_step_buffer[0][:2]

        if len(self) <= self.index:
            self.current_states.append(torch.as_tensor(data=obs, dtype=self.dtype, device='cpu'))
            self.actions.append(act)
            self.rewards.append(torch.as_tensor(data=rew, dtype=torch.float32, device='cpu'))
            self.next_states.append(torch.as_tensor(data=n_o, dtype=self.dtype, device='cpu'))
            self.is_not_terminals.append(torch.as_tensor(data=not t, dtype=torch.float32, device='cpu'))
        else:
            self.current_states[self.index] = torch.as_tensor(data=obs, dtype=self.dtype, device='cpu')
            self.actions[self.index] = act
            self.rewards[self.index] = torch.as_tensor(data=rew, dtype=torch.float32, device='cpu')
            self.next_states[self.index] = torch.as_tensor(data=n_o, dtype=self.dtype, device='cpu')
            self.is_not_terminals[self.index] = torch.as_tensor(data=not t, dtype=torch.float32, device='cpu')

        self.index += 1

        if len(self.current_states) == 1:
            print(self.current_states[-1].shape)
            print(self.current_states[-1].dtype)
            print(f"""Approximate maximum replay memory size: {
                (
                    self.current_states[-1].nbytes +
                    4 +  # size of int
                    8 +  # size of float
                    self.next_states[-1].nbytes +
                    1  # size of bool
                ) * self.capacity * 1e-9
            } gb""")

        if self.index == self.capacity:
            self.index = 0

        return self.n_step_buffer[0]

    def from_indices(self, indices):
        return self.__create_batch__(indices)

    def __create_n_step_transition__(self):
        # info of previous transition
        reward, next_obs, terminal = self.n_step_buffer[-1][-3:]

        # calculate n_step reward
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, t = transition[-3:]

            reward = r + self.gamma * reward * (1 - t)
            next_obs, terminal = (n_o, t) if t else (next_obs, t)

        return reward, next_obs, terminal

    def __create_batch__(self, indices):
        """Create batch using the provided indices"""
        current_states = torch.stack([self.current_states[i].to(device=self.device, non_blocking=True) for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = torch.stack([self.rewards[i].to(device=self.device, non_blocking=True) for i in indices])
        next_states = torch.stack([self.next_states[i].to(device=self.device, non_blocking=True) for i in indices])
        terminals = torch.stack([self.is_not_terminals[i].to(device=self.device, non_blocking=True) for i in indices])

        if current_states.dtype == torch.uint8:
            current_states = torch.div(current_states, 255.0)
            next_states = torch.div(next_states, 255.0)

        return (
            current_states,
            actions,
            rewards,
            next_states,
            terminals,
            indices
        )

    def __sample__(self, size):
        """Sample indices uniformly at random"""
        return np.random.randint(low=0, high=len(self), size=size)

    def __len__(self):
        return len(self.current_states)


class PriorityReplayMemory(ReplayMemory):
    def __init__(self, n: int, device, dtype, alpha: float, beta: float, epsilon: float):
        super().__init__(n, device, dtype, 0, 1)

        self.alpha, self.beta, self.epsilon = alpha, beta, epsilon

        # since capacity must be power of 2 for segment tree, calculate the smallest power of 2 larger or equal to n
        self.sum_tree = SumSegmentTree(n if np.log2(n) == n else 2 ** (int(np.log2(n)) + 1))
        self.min_tree = MinSegmentTree(n if np.log2(n) == n else 2 ** (int(np.log2(n)) + 1))

        self.max_priority = 1.0
        self.tree_index = 0

        # store last sampled indices to use for priority update
        self.indices_buffer = []

    def update_priorities(self, priorities: np.ndarray):
        """Update priorities of last sampled transitions using the buffered indices"""

        priorities = priorities + self.epsilon

        # used buffered indices here
        for index, priority in zip(self.indices_buffer, priorities):
            self.sum_tree[index] = priority ** self.alpha
            self.min_tree[index] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def store(self, current_state: np.array, action: int, reward: float, next_state: np.array, is_terminal: bool):
        super().store(current_state, action, reward, next_state, is_terminal)

        # TODO check if self.tree_index can be replaced with self.index
        # set initial priority of new transitions to max priority
        self.sum_tree[self.tree_index] = self.max_priority ** self.alpha
        self.min_tree[self.tree_index] = self.max_priority ** self.alpha

        self.tree_index = (self.tree_index + 1) % self.capacity

    def __create_batch__(self, indices):
        """Create batch based on proportionally sampled indices, calculate weights used for bias correction"""
        # buffer indices
        self.indices_buffer = indices
        weights = [self.__calculate_weight__(i, self.beta) for i in indices]

        return *super().__create_batch__(indices), torch.as_tensor(weights, dtype=torch.float32, device=self.device)

    def __sample__(self, size):
        """Proportional sampling of indices"""
        indices = []
        p_total = self.sum_tree.sum(0, len(self))
        segment = p_total / size

        for i in range(size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(upperbound)
            indices.append(idx)

        return indices

    def __calculate_weight__(self, index: int, beta: float):
        """Calculate importance sampling weight of the experience at index"""
        # get max weight
        min_priority = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (min_priority * len(self)) ** (-beta)

        # calculate weight at index
        # sum_tree[index] holds p_i^a
        weight = ((self.sum_tree[index] / self.sum_tree.sum()) * len(self)) ** (-beta)
        weight = weight / max_weight

        # return normalized weight
        return weight
