import torch

from torch import nn
from DQN.Tools.NoisyLinear import NoisyLinear


class FeatureNetwork(nn.Module):
    """
    Implements a fully connected 3-layer perceptron using ReLU as activation function
    """

    def __init__(self, num_features: int, num_outputs: int, num_hidden_nodes: int, use_noisy: bool):
        """
        Initializes a new 3-layer mlp
        :param num_features: number of elements in the input vector
        :param num_outputs: number of elements in the output vector
        :param num_hidden_nodes: number of units in the hidden layer
        """
        super().__init__()

        linear_type = NoisyLinear if use_noisy else nn.Linear

        self.model = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=num_hidden_nodes),
            nn.ReLU(),
            linear_type(in_features=num_hidden_nodes, out_features=num_hidden_nodes),
            nn.ReLU(),
            linear_type(in_features=num_hidden_nodes, out_features=num_outputs),
        )

    def forward(self, x: torch.Tensor):
        """
        Realises one forward pass through the network and returns the result
        :param x: input to the network
        :return: result of the forward pass
        """
        return self.model(x)


class NatureCNN(nn.Module):
    """
    Implements the convolutional neural network from the Nature modulardqn paper
    [Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236]
    """

    def __init__(self, num_actions: int, size: int, use_noisy: bool):
        super().__init__()

        assert num_actions > 0
        assert size > 0

        if size != 84 and size != 64:
            raise ValueError("Pixel space must be either of size (64x64) or (84x84)")

        linear_type = NoisyLinear if use_noisy else nn.Linear

        self.model = nn.Sequential(
            NatureModule(size),
            linear_type(512, num_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class DuelingNatureCNN(nn.Module):
    """
    Implements the dueling network architecture with the nature convolutions as feature extractor
    """

    def __init__(self, num_actions: int, size: int, use_noisy: bool):
        super().__init__()
        assert num_actions > 0
        assert size > 0

        if size != 84 and size != 64:
            raise ValueError("Pixel space must be either of size (64x64) or (84x84)")

        self.model = nn.Sequential(
            NatureModule(size),
            DuelingModule(512, 512, num_actions, use_noisy)
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class DuelingFeatureNetwork(nn.Module):
    """
    Implements the dueling network architecture with a linear layer as feature extractor
    """

    def __init__(self, num_features: int, num_outputs: int, num_hidden_nodes: int, use_noisy: bool):
        """
        Initializes a new 3-layer mlp
        :param num_features: number of elements in the input vector
        :param num_outputs: number of elements in the output vector
        :param num_hidden_nodes: number of units in the hidden layer
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=num_hidden_nodes),
            nn.ReLU(),
            DuelingModule(num_hidden_nodes, 128, num_outputs, use_noisy)
        )

    def forward(self, x: torch.Tensor):
        """
        Realises one forward pass through the network and returns the result
        :param x: input to the network
        :return: result of the forward pass
        """
        return self.model(x)


class CategoricalFeatureNetwork(nn.Module):
    def __init__(
            self,
            num_features: int,
            num_actions: int,
            num_hidden_nodes: int,
            atom_size: int,
            use_dueling: bool,
            use_noisy: bool,
            support: torch.Tensor
    ):
        super().__init__()

        self.support = support
        self.atom_size = atom_size
        self.num_actions = num_actions
        self.use_dueling = use_dueling

        linear_type = NoisyLinear if use_noisy else nn.Linear

        if use_dueling:
            self.model = nn.Sequential(
                nn.Linear(num_features, num_hidden_nodes),
                nn.ReLU()
            )

            self.dueling_module = DuelingModule(num_hidden_nodes, num_hidden_nodes, num_actions, use_noisy, atom_size)
        else:
            self.model = nn.Sequential(
                nn.Linear(num_features, num_hidden_nodes),
                nn.ReLU(),
                linear_type(num_hidden_nodes, num_hidden_nodes),
                nn.ReLU(),
                linear_type(num_hidden_nodes, num_actions * atom_size)
            )

    def forward(self, x: torch.Tensor):
        distribution = self.distribution(x)

        # sum over second dim to get one output per action
        return torch.sum(distribution * self.support, dim=2)

    def distribution(self, x: torch.Tensor):
        # calculate a forward pass and reshape it
        if self.use_dueling:
            advantage, value = self.dueling_module.stream(self.model(x))

            advantage = advantage.view(-1, self.num_actions, self.atom_size)
            value = value.view(-1, 1, self.atom_size)

            q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_atoms = self.model(x).view(-1, self.num_actions, self.atom_size)

        # apply elementwise softmax for normalization
        distribution = nn.functional.softmax(q_atoms, dim=-1)

        # avoid NaNs
        distribution = distribution.clamp(min=1e-3)

        return distribution

    def reset_noise(self):
        self.dueling_module.reset_noise()


class CategoricalNatureCNN(nn.Module):
    def __init__(
            self,
            num_features: int,
            num_actions: int,
            num_hidden_nodes: int,
            atom_size: int,
            use_dueling: bool,
            use_noisy: bool,
            support: torch.Tensor
    ):
        super().__init__()

        self.support = support
        self.atom_size = atom_size
        self.num_actions = num_actions
        self.use_dueling = use_dueling

        linear_type = NoisyLinear if use_noisy else nn.Linear

        if use_dueling:
            self.cnn = NatureModule(num_features)
            self.feature_extractor = nn.Sequential(
                self.cnn,
                nn.ReLU()
            )

            self.dueling_module = DuelingModule(512, num_hidden_nodes, num_actions, use_noisy, atom_size)
        else:
            self.cnn = NatureModule(num_features)
            self.feature_extractor = nn.Sequential(
                self.cnn,
                nn.ReLU(),
                linear_type(512, num_actions * atom_size)
            )

    def forward(self, x: torch.Tensor):
        distribution = self.distribution(x)

        # sum over second dim to get one output per action
        return torch.sum(distribution * self.support, dim=2)

    def distribution(self, x: torch.Tensor):
        # calculate a forward pass and reshape it
        if self.use_dueling:
            advantage, value = self.dueling_module.stream(self.feature_extractor(x))

            advantage = advantage.view(-1, self.num_actions, self.atom_size)
            value = value.view(-1, 1, self.atom_size)

            q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_atoms = self.feature_extractor(x).view(-1, self.num_actions, self.atom_size)

        # apply elementwise softmax for normalization
        distribution = nn.functional.softmax(q_atoms, dim=-1)

        # avoid NaNs
        distribution = distribution.clamp(min=1e-3)

        return distribution

    def reset_noise(self):
        self.dueling_module.reset_noise()


class NatureModule(nn.Module):
    """
    Implements the convolutional part of the NatureCNN
    Input is either 84x84x4 of 64x64x4 output is always a 512 dim vector
    """

    def __init__(self, size: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(out_features=512),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class DuelingModule(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, use_noisy: bool, atom_size: int = 1):
        super().__init__()

        self.linear_layers = [
            NoisyLinear(input_size, hidden_size) if use_noisy else nn.Linear(input_size, hidden_size),
            NoisyLinear(hidden_size, output_size * atom_size) if use_noisy else nn.Linear(hidden_size,
                                                                                          output_size * atom_size),
            NoisyLinear(input_size, hidden_size) if use_noisy else nn.Linear(input_size, hidden_size),
            NoisyLinear(hidden_size, atom_size) if use_noisy else nn.Linear(hidden_size, atom_size)
        ]

        self.noisy = use_noisy

        # value stream going from input_size to a single advantage value
        self.advantage_stream = nn.Sequential(
            self.linear_layers[0],
            nn.ReLU(),
            self.linear_layers[1]
        )

        # action stream going from input_size to an action value for each action
        self.value_stream = nn.Sequential(
            self.linear_layers[2],
            nn.ReLU(),
            self.linear_layers[3]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        action_advantages = self.advantage_stream(x)
        state_value = self.value_stream(x)

        # average replacing max operator according to equation (9) of the dueling networks paper
        return state_value + (action_advantages - action_advantages.mean(dim=-1, keepdim=True))

    def stream(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        action_advantages = self.advantage_stream(x)
        state_value = self.value_stream(x)

        return action_advantages, state_value

    def reset_noise(self):
        if self.noisy:
            for layer in self.linear_layers:
                layer.reset_noise()
