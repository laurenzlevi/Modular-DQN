import gymnasium as gym
from DQN.Wrappers.imagepreprocessing import ImagePreprocessingWrapper
from DQN.Tools.models import QNetwork, NatureCNN
from DQN.Utils.simulation import simulate
import torch

env = ImagePreprocessingWrapper(gym.make('QWOP:QWOP-pixels-v0', render_mode='rgb_array'))
# env = ImagePreprocessingWrapper(gym.make('racingenv:racingenv/Racing-pixels-v1', render_mode='rgb_array'))

observation, info = env.reset()
model = NatureCNN(env.action_space.n, observation.shape[1])
model.load_state_dict(torch.load('./tmp/QWOP_krass.pt', map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('./tmp/racingenv-model.pt', map_location=torch.device('cpu')))

network = QNetwork(model, 'cpu')

simulate(env, 'cpu', 1000, network, './tmp', heatmap_amount=0.5, upscale_output_by=5)
