from typing import Tuple

import torch
from torch import nn

from DQN.Tools import CategoricalFeatureNetwork, DuelingFeatureNetwork, FeatureNetwork, CategoricalNatureCNN, \
    DuelingNatureCNN, NatureCNN


class QNetwork:
    def __init__(self, model: nn.Module, device, noisy):
        """
        Create a new QNetwork with a given module as neural network
        :param model: Underlying neural network for the QFunction network
        :param device: device to upload the tensors to
        """

        self.noisy = noisy
        self.model = model
        model.to(device)

        # if torch._dynamo.is_dynamo_supported():
        #    print('Compiling model using torch.compile')
        #    self.model = torch.compile(self.model)

        self.device = device

    def forward(self, x: torch.Tensor):
        """
        Compute one forward pass through the network
        :param x: input tensor for the neural network
        :return: result of the forward pass
        """

        return self.model(x)

    def predict_action(self, x: torch.Tensor):
        """
        Compute one forward pass through the network and return the index of the action with the highest activation
        :param x: input tensor for the neural network
        :return: tensor containing only the index of the highest activation action
        """
        return torch.argmax(self.forward(x), dim=-1)

    def predict_q_value(self, x: torch.Tensor):
        """
        Compute one forward pass through the network and return the q-value of the action with the highest activation
        :param x: input tensor for the neural network
        :return: tensor containing only the highest predicted q-value
        """
        return self.forward(x)

    def reset_noisy(self):
        if self.noisy:
            self.model.reset_noise()


def create_networks(
        policy: [str, nn.Module],
        device: torch.device,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        num_hidden_nodes: int,
        atom_size: int = 51,
        support: torch.Tensor = None,
        use_categorical: bool = False,
        use_dueling: bool = False,
        use_noisy: bool = False
) -> Tuple[QNetwork, QNetwork]:
    if isinstance(policy, nn.Module):
        return QNetwork(policy, device, use_noisy), QNetwork(policy, device, use_noisy)
    elif isinstance(policy, str):
        if policy == 'MLP':
            if use_categorical:
                return (
                    QNetwork(
                        CategoricalFeatureNetwork(
                            num_features=obs_shape[0],
                            num_actions=num_actions,
                            num_hidden_nodes=num_hidden_nodes,
                            atom_size=atom_size,
                            use_dueling=use_dueling,
                            use_noisy=use_noisy,
                            support=support
                        ),
                        device=device,
                        noisy=use_noisy
                    ),
                    QNetwork(
                        CategoricalFeatureNetwork(
                            num_features=obs_shape[0],
                            num_actions=num_actions,
                            num_hidden_nodes=num_hidden_nodes,
                            atom_size=atom_size,
                            use_dueling=use_dueling,
                            use_noisy=use_noisy,
                            support=support
                        ),
                        device=device,
                        noisy=use_noisy
                    )
                )
            elif use_dueling:
                return (
                    QNetwork(
                        DuelingFeatureNetwork(
                            num_features=obs_shape[0],
                            num_outputs=num_actions,
                            num_hidden_nodes=num_hidden_nodes,
                            use_noisy=use_noisy
                        ),
                        device=device,
                        noisy=use_noisy
                    ),
                    QNetwork(
                        DuelingFeatureNetwork(
                            num_features=obs_shape[0],
                            num_outputs=num_actions,
                            num_hidden_nodes=num_hidden_nodes,
                            use_noisy=use_noisy
                        ),
                        device=device,
                        noisy=use_noisy
                    )
                )
            else:
                return (
                    QNetwork(
                        FeatureNetwork(
                            num_features=obs_shape[0],
                            num_outputs=num_actions,
                            num_hidden_nodes=num_hidden_nodes,
                            use_noisy=use_noisy
                        ),
                        device=device,
                        noisy=use_noisy
                    ),
                    QNetwork(
                        FeatureNetwork(
                            num_features=obs_shape[0],
                            num_outputs=num_actions,
                            num_hidden_nodes=num_hidden_nodes,
                            use_noisy=use_noisy
                        ),
                        device=device,
                        noisy=use_noisy
                    )
                )
        elif policy == "CNN":
            if use_categorical:
                return (
                    QNetwork(
                        CategoricalNatureCNN(
                            num_features=obs_shape[1],
                            num_actions=num_actions,
                            num_hidden_nodes=num_hidden_nodes,
                            atom_size=atom_size,
                            use_dueling=use_dueling,
                            use_noisy=use_noisy,
                            support=support
                        ),
                        device=device,
                        noisy=use_noisy
                    ),
                    QNetwork(
                        CategoricalNatureCNN(
                            num_features=obs_shape[1],
                            num_actions=num_actions,
                            num_hidden_nodes=num_hidden_nodes,
                            atom_size=atom_size,
                            use_dueling=use_dueling,
                            use_noisy=use_noisy,
                            support=support
                        ),
                        device=device,
                        noisy=use_noisy
                    )
                )
            elif use_dueling:
                return (
                    QNetwork(
                        DuelingNatureCNN(
                            num_actions=num_actions,
                            size=obs_shape[1],
                            use_noisy=use_noisy
                        ),
                        device=device,
                        noisy=use_noisy
                    ),
                    QNetwork(
                        DuelingNatureCNN(
                            num_actions=num_actions,
                            size=obs_shape[1],
                            use_noisy=use_noisy
                        ),
                        device=device,
                        noisy=use_noisy
                    )
                )
            else:
                return (
                    QNetwork(
                        NatureCNN(
                            num_actions=num_actions,
                            size=obs_shape[1],
                            use_noisy=use_noisy
                        ),
                        device=device,
                        noisy=use_noisy
                    ),
                    QNetwork(
                        NatureCNN(
                            num_actions=num_actions,
                            size=obs_shape[1],
                            use_noisy=use_noisy
                        ),
                        device=device,
                        noisy=use_noisy
                    )
                )
        else:
            raise ValueError(f'Unknown policy {policy}, expected MLP or CNN')
    else:
        raise ValueError(f'Unexpected type {policy.__class__}, expected str or torch.nn.Module')
