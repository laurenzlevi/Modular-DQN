import argparse
import os

import gymnasium as gym
import torch

from DQN.Tools import CategoricalNatureCNN, QNetwork
from DQN.Utils.simulation import record
from DQN.Wrappers.imagepreprocessing import ImagePreprocessingWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--epsilon', default=0.00, type=float)
    parser.add_argument('--device', default=None)

    args = parser.parse_args()

    os.makedirs(args.folder, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env = ImagePreprocessingWrapper(gym.make(args.env, render_mode='rgb_array'))
    obs, info = env.reset()

    # categorical parameter
    v_min = 0.0
    v_max = 200.0
    atom_size = 51
    support = torch.linspace(v_min, v_max, atom_size).to(device)

    model = CategoricalNatureCNN(84, 6, 256, atom_size, True, True, support)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = QNetwork(model, device, True)

    epsilon = args.epsilon

    record(env, device, 100_000, model, args.folder, 0, True, epsilon)
