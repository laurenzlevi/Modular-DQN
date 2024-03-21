import re
from typing import Union, Tuple

import numpy as np
import torch
from prettytable import PrettyTable

from torch import nn

from DQN.Tools import ReplayMemory, PriorityReplayMemory
from DQN.Logging.episodelog import EpisodeLog
from DQN.Tools.QNetwork import create_networks
from DQN.Utils.loss_funcs import make_loss_func
from DQN.Utils.optimizers import make_optimizer

import gymnasium as gym

from DQN.Utils.simulation import record
from enum import IntFlag, auto

from DQN.Wrappers.frameskipping import FrameSkippingWrapper
from DQN.Wrappers.imagepreprocessing import ImagePreprocessingWrapper
from DQN.Wrappers.recordstatistics import RecordEpisodeStatisticsWrapper


class Mode(IntFlag):
    DQN = 0
    DDQN = auto()
    PER = auto()
    N_STEP = auto()
    NOISY = auto()
    DUELING = auto()
    CATEGORICAL = auto()
    RAINBOW = DDQN | PER | N_STEP | NOISY | DUELING | CATEGORICAL


def __create_device__(device) -> torch.device:
    if isinstance(device, str):
        if re.match(r"^(cpu|cuda:?[0-9]*)$", device) is not None:
            return torch.device(device)
        else:
            raise ValueError(f"Invalid device name was passed expected ('cpu' | 'cuda') but was {device}")
    else:
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class DQN:
    def __init__(
            self,
            policy: Union[str, nn.Module],
            env: Union[str, gym.Env],
            device: Union[str, None] = None,
            replay_memory_size: int = 1_000_000,
            frame_skip: int = 1,
            seed: int = None,
            epsilon_max: float = 1.0,
            epsilon_decay_step: float = 0.1,
            epsilon_decay_interval: int = 1_000,
            epsilon_min: float = 0.1,
            train_freq: int = 4,
            optimizer: str = "SGD",
            loss_func: str = 'Huber',
            lr: float = 1e-3,
            gamma: float = 1.0,
            tau: float = 0.01,
            batch_size: int = 32,
            use_wandb: bool = False,
            wandb_tags: list[str] = None,
            num_evaluation_states: int = 1_000,
            log_interval: int = 10,
            mode: int = Mode.DQN,
            store_model: bool = False,
            # priority replay parameter
            alpha: float = 0.5,
            beta: float = 0.7,
            priority_epsilon=1e-6,
            # categorical dqn parameter
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            n_step: int = 1,
            image_size: Tuple[int, int, int] = None,
            **kwargs
    ):
        """
        Initialize modulardqn
        :param policy: either the neural networked used for training or a string containing the policy ("MLP" or "CNN")
        :param env: Environment to learn
        :param device: device used by pytorch, either 'cpu' or 'cuda', if None is provided device is determined automatically
        :param replay_memory_size: Size of replay memory
        :param seed: Seed for random events
        :param epsilon_max: Epsilon probability for epsilon greedy policy, must be in [0.0, 1.0]
        :param lr: learning rate used by the optimizer
        :param gamma: discount factor
        :param batch_size: size of the transition minibatch used for Q-Function updates
        :param num_evaluation_states: Number of states collected at the start for evaluating the Q function later on
        :param use_wandb: Should results be logged to wandb
        :param alpha: prioritization parameter in [0.0, 1.0] (alpha=0 is equal to uniform sampling)
        :param beta: initial beta for bias correction
        :param priority_epsilon: small additional term for priority sampling
        """

        if lr < 0.0:
            raise ValueError("Learning rate must be greater or equal to zero!")

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("Discount factor must be in [0.0, 1.0]!")

        if not 0.0 <= epsilon_max <= 1.0:
            raise ValueError("Epsilon must be be in [0.0, 1.0]!")

        self.device = __create_device__(device)

        if isinstance(env, str):
            self.env = RecordEpisodeStatisticsWrapper(gym.make(id=env, **kwargs))
        else:
            self.env = RecordEpisodeStatisticsWrapper(env)

        if policy == 'MLP':
            self.obs_type = 'features'
        else:
            self.obs_type = 'pixels'
            self.env = ImagePreprocessingWrapper(self.env, image_size)

        if frame_skip > 1:
            self.env = FrameSkippingWrapper(self.env)

        if isinstance(env.observation_space, gym.spaces.Dict) and env.observation_space.is_np_flattenable:
            self.env = gym.wrappers.FlattenObservation(env)

        self.loss_func = make_loss_func(loss_func)

        # get initial observation here to create networks with the right number of inputs
        observation, info = self.env.reset()

        self.mode = mode
        self.seed = seed
        self.epsilon = epsilon_max if Mode.NOISY not in self.mode else 0.0 # set epsilon to zero when noisy layers are activated
        self.epsilon_decay_step = epsilon_decay_step
        self.epsilon_decay_interval = epsilon_decay_interval
        self.epsilon_min = epsilon_min if Mode.NOISY not in self.mode else 0.0
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_evaluation_states = num_evaluation_states
        self.store_model = store_model
        self.train_freq = train_freq

        # categorical parameter
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        self.replay_memory = self.__create_replay_memory__(
            n=replay_memory_size,
            dtype=torch.uint8 if observation.dtype == np.uint8 else torch.float32,
            alpha=alpha,
            beta=beta,
            epsilon=priority_epsilon
        )

        if Mode.N_STEP in self.mode:
            if n_step < 2:
                raise ValueError("n_step memory enabled with n_step size < 2")

            self.n_step_memory = ReplayMemory(
                replay_memory_size,
                self.device,
                torch.uint8 if observation.dtype == np.uint8 else torch.float32,
                self.gamma,
                n_step
            )
            self.n_steps = n_step

        self.policy_network, self.target_network = create_networks(
            policy=policy,
            device=self.device,
            obs_shape=observation.shape,
            num_actions=self.env.action_space.n,
            num_hidden_nodes=128,
            atom_size=self.atom_size,
            support=self.support,
            use_categorical=Mode.CATEGORICAL in self.mode,
            use_dueling=Mode.DUELING in self.mode,
            use_noisy=Mode.NOISY in self.mode
        )

        self.optimizer = make_optimizer(
            optimizer=optimizer,
            parameters=self.policy_network.model.parameters(),
            lr=self.lr
        )

        # Create episode log
        if use_wandb:
            wandb_config = {
                "learning_rate": lr,
                "gamma": gamma,
                "batch_size": batch_size,
                "tau": tau,
                "epsilon": self.epsilon,
                "epsilon decay step": self.epsilon_decay_step,
                "epsilon decay interval": self.epsilon_decay_interval,
                "mode": str(self.mode).removeprefix('Mode.').replace('|', ' ~ ')
            }

            if Mode.PER in self.mode:
                wandb_config["alpha"] = alpha

            if Mode.CATEGORICAL in self.mode:
                wandb_config["atom size"] = atom_size
                wandb_config["v_min"] = v_min
                wandb_config["v_max"] = v_max

            if Mode.N_STEP in self.mode:
                wandb_config["n_step"] = n_step

        else:
            wandb_config = None

        self.episode_log = EpisodeLog(
            log_interval=log_interval,
            env_id=self.env.spec.id,
            wandb_config=wandb_config,
            wandb_tags=wandb_tags
        )

        self.__print_hyperparameter__(observation)

        self.evaluation_states = None

    def __create_replay_memory__(self, n: int, dtype, alpha, beta, epsilon) -> ReplayMemory:
        if Mode.PER in self.mode:
            return PriorityReplayMemory(n, self.device, dtype, alpha, beta, epsilon)
        else:
            return ReplayMemory(n, self.device, dtype, self.gamma, 1)

    def learn(
            self,
            steps: int,
            episodes_trigger: int,
            record_path: str,
            use_progress_bar: bool = False,
            heatmap_opacity: float = 0.0,
            generate_grpahs: bool = False
    ):
        """
        Step limited variant of main loop for modulardqn algorithm
        :param steps: Maximum amount of steps per episode
        :param episodes_trigger: Number of steps between each recording
        :param record_path: Folder to save the videos to
        :return: Evaluation over the number of steps as a list (frequency set by log_evaluation_every_n_steps)
        """

        # just for logging
        n_updates = 0
        average_q = 0

        # just for recording
        max_ep_len = 0

        observation, info = self.env.reset()

        # sync the initial target and policy networks
        self.target_network.model.load_state_dict(self.policy_network.model.state_dict())
        self.policy_network.model.train()
        self.target_network.model.train()

        self.episode_log.watch(self.policy_network.model)

        if use_progress_bar:
            import rich.progress

            progress = rich.progress.Progress(
                rich.progress.BarColumn(),
                rich.progress.TaskProgressColumn(),
                "[",
                rich.progress.MofNCompleteColumn(),
                "]",
                "(",
                rich.progress.TimeElapsedColumn(),
                ">",
                rich.progress.TimeRemainingColumn(),
                ")",
                rich.progress.TransferSpeedColumn()
            )

            progress.start()
            steps = progress.track(range(steps), description="Learning...")
        else:
            steps = range(steps)

        for t in steps:
            if t != 0 and t % self.epsilon_decay_interval == 0:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_step)

            action = self.get_action(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(action)

            if Mode.N_STEP in self.mode:
                self.n_step_memory.store(
                    current_state=observation,
                    action=action,
                    reward=reward,
                    next_state=next_observation,
                    is_terminal=(terminated or truncated)
                )

            self.replay_memory.store(
                current_state=observation,
                action=action,
                reward=reward,
                next_state=next_observation,
                is_terminal=(terminated or truncated)
            )

            observation = next_observation

            if self.epsilon < 1.0:
                # if Mode.PER in self.mode:
                    # update beta here before we calculate priorities in optimize_model
                    # self.replay_memory.beta = self.replay_memory.beta + min(t / steps, 1.0) * (1.0 - self.replay_memory.beta)

                if t % self.train_freq == 0:
                    self.optimize_model()
                    n_updates += 1

                    # update target network
                    self.polyak_update()

            if terminated or truncated:
                observation, info = self.env.reset()
                episode = info['episode']

                # Calculate average max q every 10*log_interval episodes
                if episode % self.episode_log.log_interval * 10 == 0:
                    average_q = self.evaluate_on_samples()

                # Write episode result to log
                self.episode_log.update(
                    steps=t,
                    lr=self.lr,
                    n_updates=n_updates,
                    episode_reward=info['reward'],
                    average_q=average_q,
                    epsilon=self.epsilon,
                    beta=self.replay_memory.beta if Mode.PER in self.mode else 0.0
                )

                if info['steps'] > max_ep_len:
                    max_ep_len = info['steps']

                # record episode using the record function to have more control than with the RecordVideo wrapper
                # this function also truncates the recording automatically if the episode is longer than
                # twice the max episode length observed in training thus far
                if episodes_trigger is not None and episode % episodes_trigger == 0 and episode != 0 and self.epsilon < 1.0:
                    self.policy_network.model.eval()
                    self.target_network.model.eval()

                    sim_reward = record(
                        env=self.env,
                        device=self.device,
                        trunc_steps=max_ep_len * 2,
                        model=self.policy_network,
                        record_path=record_path,
                        episode=episode,
                        record_obs=self.obs_type == 'pixels',
                        epsilon=0.01,
                        heatmap_amount=heatmap_opacity,
                        plot=generate_grpahs
                    )

                    if self.store_model:
                        torch.save(
                            self.policy_network.model.state_dict(),
                            f"{record_path}/models/Model_{episode}_{sim_reward:.2f}.pt"
                        )

                    self.policy_network.model.train()
                    self.target_network.model.train()

        if use_progress_bar:
            progress.stop()
            progress.reset(progress.task_ids[0])

    def get_action(self, observation):
        if np.random.random() < self.epsilon and Mode.NOISY not in self.mode:
            return self.env.action_space.sample()
        else:
            obs = torch.as_tensor(data=np.array([observation]), dtype=torch.float32, device=self.device)

            if observation.dtype == np.uint8:
                obs = torch.div(obs, 255.0)

            # use the policy network to choose action
            with torch.no_grad():
                return self.policy_network.predict_action(obs).item()

    def optimize_model(self):
        if Mode.PER in self.mode:
            batch = self.replay_memory.sample(self.batch_size)
            indices = batch[5]
            weights = batch[6]
        else:
            batch = self.replay_memory.sample(self.batch_size)
            indices = batch[5]

        if Mode.CATEGORICAL in self.mode:
            loss = self.categorical_dqn_loss(batch, self.gamma)
        else:
            loss = self.dqn_loss(batch, self.gamma)

        if Mode.N_STEP in self.mode:
            n_step_gamma = self.gamma ** self.n_steps
            n_step_batch = self.n_step_memory.from_indices(indices)

            if Mode.CATEGORICAL in self.mode:
                n_loss = self.categorical_dqn_loss(n_step_batch, n_step_gamma)
            else:
                n_loss = self.dqn_loss(n_step_batch, n_step_gamma)

            loss += n_loss

        if Mode.PER in self.mode:
            elementwise_loss = loss
            loss = torch.mean(elementwise_loss * weights)

        # compute gradients
        loss.backward()

        # update parameters
        self.optimizer.step()

        # reset gradients
        self.optimizer.zero_grad(set_to_none=True)

        self.policy_network.reset_noisy()
        self.target_network.reset_noisy()

        if Mode.PER in self.mode:
            self.replay_memory.update_priorities(elementwise_loss.detach().cpu().numpy())

    def dqn_loss(self, batch, gamma):
        if Mode.PER in self.mode:
            current_states, actions, rewards, next_states, is_not_terminals, *_ = batch
        else:
            current_states, actions, rewards, next_states, is_not_terminals, *_ = batch

        # compute target values either using modulardqn or DDQN approach
        with torch.no_grad():
            if Mode.DDQN in self.mode:
                # Double Q-Learning targets
                next_action = torch.argmax(self.policy_network.forward(next_states), dim=1)

                next_qs = self.target_network.predict_q_value(next_states)
                next_qs = next_qs[range(self.batch_size), next_action]
            else:
                # torch.max returns a tuple of (tensor, indices) where tensor holds the maximum values and
                # indices holds their respective index, we only want the predicted Q-values here,
                # thus we take the 0th element of the tuple
                next_qs = torch.max(self.target_network.predict_q_value(next_states).detach(), dim=1)[0]

        ys = rewards + gamma * is_not_terminals * next_qs

        # this returns a tensor of shape [batch_size, num_actions] we only want to keep the Q-values
        # of the action that was chosen in the selected state
        # predict all Q-values here
        xs = self.policy_network.predict_q_value(current_states)
        # then filter for the chosen actions, so we get a tensor of shape [batch_size, ]
        xs = xs[range(self.batch_size), actions]

        # compute elementwise loss for priority updates
        if Mode.PER in self.mode:
            loss = self.loss_func(input=xs, target=ys, reduction='none')
        # else compute the normal dqn loss
        else:
            loss = self.loss_func(input=xs, target=ys)

        if torch.isnan(loss):
            raise Exception('NaN loss detected')

        return loss

    def categorical_dqn_loss(self, batch, gamma):
        """
        Categorical modulardqn algorithm for loss
        see https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/06.categorical_dqn.ipynb for source and tutorial
        """
        if Mode.PER in self.mode:
            current_states, actions, rewards, next_states, is_not_terminals, *_ = batch
        else:
            current_states, actions, rewards, next_states, is_not_terminals, *_ = batch

        rewards = rewards.reshape(-1, 1)
        is_not_terminals = is_not_terminals.reshape(-1, 1)

        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            if Mode.DDQN in self.mode:
                next_action = self.policy_network.model(next_states).argmax(1)
                next_distribution = self.target_network.model.distribution(next_states)[range(self.batch_size), next_action]
            else:
                next_action = self.target_network.model(next_states).argmax(1)
                next_distribution = self.target_network.model.distribution(next_states)[range(self.batch_size), next_action]

            t_z = rewards + is_not_terminals * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            projected_distribution = torch.zeros(next_distribution.size(), device=self.device)
            projected_distribution.view(-1).index_add_(
                0, (l + offset).view(-1), (next_distribution * (u.float() - b)).view(-1)
            )
            projected_distribution.view(-1).index_add_(
                0, (u + offset).view(-1), (next_distribution * (b - l.float())).view(-1)
            )

        distribution = self.policy_network.model.distribution(current_states)
        log_p = torch.log(distribution[range(self.batch_size), actions])

        # compute elementwise loss for priority updates
        if Mode.PER in self.mode:
            loss = -(projected_distribution * log_p).sum(1)
        # else compute the normal dqn loss
        else:
            loss = -(projected_distribution * log_p).sum(1).mean()

        return loss

    def polyak_update(self):
        """
        Soft-update the target network parameters
        """

        target_net_states = self.target_network.model.state_dict()
        policy_net_states = self.policy_network.model.state_dict()
        for parameter in policy_net_states:
            target_net_states[parameter] = policy_net_states[parameter] * self.tau + target_net_states[parameter] * (
                        1 - self.tau)
        self.target_network.model.load_state_dict(target_net_states)

    def evaluate_on_samples(self) -> float:
        """
        Evaluate model on sampled states (bigger is better (MURICA))
        :return: Average of max(Q) over the evaluation states
        """

        if self.evaluation_states is None:
            if len(self.replay_memory.current_states) < self.num_evaluation_states:
                return 0
            else:
                self.evaluation_states = self.__gather_samples__(self.num_evaluation_states)

        # compute all q-values in a single forward pass,
        # then compute the max in the first dimension before returning the mean
        with torch.no_grad():
            return torch.mean(
                torch.amax(
                    input=self.policy_network.predict_q_value(self.evaluation_states),
                    dim=1
                )
            ).item()

    def save(self, file: str):
        torch.save(self.policy_network.model.state_dict(),file)

    def load(self, file: str):
        model = torch.load(file, map_location=torch.device(self.device))

        self.policy_network.model.load_state_dict(model)
        self.target_network.model.load_state_dict(model)

    def __gather_samples__(self, size: int) -> torch.Tensor:
        """
        Returns a tensor of samples drawn randomly from replay memory current states
        """

        return self.replay_memory.sample(size)[0]

    def __print_hyperparameter__(self, obs: np.array):
        table = PrettyTable(['Hyperparameter', 'Value'])

        # default dqn parameters
        table.add_row(['modulardqn', ''], divider=True)
        table.add_row(['Mode', str(self.mode).removeprefix('Mode.').replace('|', ' ~ ')])
        table.add_row(['Env', self.env.spec.id])
        table.add_row(['Optimizer', self.optimizer.__class__.__name__])
        table.add_row(['Loss Function', self.loss_func.__name__])
        table.add_row(['Replay Memory', self.replay_memory.__class__.__name__])
        table.add_row(['Seed', self.seed])
        table.add_row(['Training Frequency', self.train_freq])
        table.add_row(['Learning Rate', self.lr])
        table.add_row(['Gamma', self.gamma])
        table.add_row(['Tau', self.tau])
        table.add_row(['Epsilon', self.epsilon])
        table.add_row(['Decay Interval', self.epsilon_decay_interval])
        table.add_row(['Decay Step', self.epsilon_decay_step])
        table.add_row(['Batch Size', self.batch_size], divider=True)

        # priority replay parameters
        if Mode.PER in self.mode:
            table.add_row(['Prioritized Replay Memory', ''], divider=True)
            table.add_row(['Alpha', self.replay_memory.alpha])
            table.add_row(['Beta', self.replay_memory.beta], divider=True)

        if Mode.N_STEP in self.mode:
            table.add_row(['N-Step', ''], divider=True)
            table.add_row(['n_step', self.n_steps], divider=True)

        if Mode.CATEGORICAL in self.mode:
            table.add_row(['Categorical', ''], divider=True)
            table.add_row(['Q-Atoms', self.atom_size])
            table.add_row(['v_min', self.v_min])
            table.add_row(['v_max', self.v_max], divider=True)

        print(table)

        try:
            import torchinfo

            if obs.dtype == np.uint8:
                obs = obs.astype(np.float32)/255.0
            else:
                obs = obs.astype(np.float32)

            print(obs.shape)

            torchinfo.summary(
                model=self.policy_network.model,
                input_data=torch.stack([torch.as_tensor(obs).to(self.device)] * self.batch_size),
                device=self.policy_network.device,
                depth=5
            )
        except ImportError:
            print(self.policy_network.model)
