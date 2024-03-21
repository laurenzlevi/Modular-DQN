# Modular-DQN

Fully modular implementation of rainbow DQN, allowing for each feature to be toggled individually.

## Authors

[Laurenz Levi Spielmann](https://github.com/laurenzlevi)

[Pascal Makossa](https://github.com/pascal0012)

[Julian Bohnenkämper](https://github.com/JulianBohne)

This Project was part of the [Fachprojekt: Applied Deep Reinforcement Learning](https://ai.cs.tu-dortmund.de/teaching/bsc-fachprojekte/fachprojekt-reinforcement-learning/) at TU Dortmund. 

## Install

To install run `pip install modular-dqn`

## Requirements

 - `torch`
 - `prettytable`
 - `gymnasium`
 - `opencv-python`
 - `wandb` (optional)
 - `rich` (optional)

## Usage

To start learning with our implementation simply execute the `modular-dqn` command. While we have provided sensible default values for most hyperparameters, they can be adjusted individually.

### Required Arguments

| Argument                   |Type| Description                                        |
|----------------------------|----|----------------------------------------------------|
| `--env` \| `--environment` | string | Gymnasium environment to learn  |
| `-s` \| `--steps`          | int | Number of steps in Training                        |
| `-π` \| `--policy`         | string | Network to use (MLP  | CNN) |

### Optional Arguments

| Argument                              | Type         | Default  | Description                                                                                                    |
|---------------------------------------|--------------|----------|----------------------------------------------------------------------------------------------------------------|
| `--device`                            | string       | None     | Device used by pytorch (cpu  | cuda)                                                                             |
| `--lr` \| `--learning_rate`           | float        | 1e-3     | Learning rate used                                                                                             |
| `-ε` \| `--epsilon`                   | float        | 1        | Initial epsilon used for epsilon-greedy policy                                                                 |
| `--edi` \| `--epsilon_decay_interval` | int          | 1e3      | Epsilon decay step interval                                                                                    |
| `--eds` \| `--epsilon_decay_step`     | float        | 0.1      | Size of epsilon decay step                                                                                     |
| `--e_min` \| `--epsilon_min`          | float        | 0.1      | Minimal epsilon value                                                                                          |
| `-𝛾` \| `--gamma`                    | float        | 0.9      | Discount factor for future rewards                                                                             |
 | `-𝜏` \| `--tau`                      | float        | 0.95     | Polyak update factor for target network                                                                        |
| `--bs` \| `--batch_size`              | int          | 32       | Batch size used to update the Q-Function                                                                       |
| `--seed`                              | int          | None     | Seed for the environment                                                                                       |
| `--rm_size` \| `--replay_memory_size` | int          | 1e7      | Replay memory maximum capacity                                                                                 |
| `--rec_trigger`                       | int          | None     | Records every `rec_trigger` episodes if provided                                                               |
| `--wandb`                             | boolean      | False    | Whether progress should be logged to [wandb](wandb.ai)                                                         |
| `--tags`                              | List[string] | None     | Tags to add to the run on [wandb](wandb.ai)                                                                    |
| `--li` \| `--log_interval`            | int          | 10       | Number of episodes between logs                                                                                |
| `--load_file`                         | string       | None     | Relative path where the network should be loaded from if provided                                              |
| `--optimizer`                         | string       | SGD      | Name of optimizer to be used (e.g. SGD  | Adam)                                                                  |
| `--skip_frames` \| `--skp`            | int          | 1        | The number of frames to skip each step                                                                         |
| `--clip` \| `--reward_clipping`       | int          | None     | Set to 0 for hard or any other scale for soft clipping divided by scale                                        |
| `-α` \| `--alpha`                     | float        | 0.5      | Alpha for priority replay                                                                                      |
| `-β` \| `--beta`                      | float        | 0.5      | Beta for priority replay                                                                                       |
| `--store_model`                       | Flag         | -        | Stores model every `rec_trigger` interval if set                                                               |
| `--ddqn`                              | Flag         | -        | Enables double deep Q-Learning                                                                                 |
| `--per`                               | Flag         | -        | Enables prioritized replay memory                                                                              |
| `--n_step`                            | int          | 1        | Sets n-step transition length                                                                                  |
| `--noisy`                             | Flag         | -        | Enables noisy linear layers for exploration                                                                    |
| `--dueling`                           | Flag         | -        | Uses dueling networks architecture                                                                             |
| `--cat` \| `categorical`              | Flag         | -        | Uses categorical dqn loss                                                                                      |
| `--rainbow`                           | Flag         | -        | Enables all improvements (`--n_step` should still be set)                                                      |
| `--kwargs`                            | Dict         | None     | Additional kwargs passed to environment on creation (usage | --kwargs render_mode=`rgb_array` trunc_steps=1000) |
| `--progress`                          | Flag         | -        | Display progress bar in terminal (requires rich)                                                               |
| `--loss`                              | string       | SmoothL1 |                                                                                                                |
| `--obs_size`                          | Tuple        | None     | Rescale image observations to given size (usage | `--obs_size (stack_size  | width  | height)`)                     |
| `--heatmaps`                          | float        | 0.0      | Heatmap opacity in videos or 0.0 for no heatmaps (only works with CNN and image observations)                  |
| `--graphs`                            | Flag         | False    | Generates Q-Value graph in videos (only works on linux currently)                                              |


## Available Optimizers

| Name | Optimizer |
|------|-----------|
| `Adadelta` | torch.optim.Adadelta |
| `Adagrad` | torch.optim.Adagrad |
| `Adam` | torch.optim.Adam |
| `AdamW` | torch.optim.AdamW |
| `SparseAdam` | torch.optim.SparseAdam |
| `Adamax` | torch.optim.Adamax |
| `ASGD` | torch.optim.ASGD |
| `LBFGS` | torch.optim.LBFGS |
| `NAdam` | torch.optim.NAdam |
| `RAdam` | torch.optim.RAdam |
| `RMSProp` | torch.optim.RMSprop |
| `Rprop` | torch.optim.Rprop |
| `SGD` | torch.optim.SGD |

## Available Loss Functions

| Name | Loss |
|------|------|
 |`L1`  | torch.nn.functional.l1_loss  |
 |`MSE` | torch.nn.functional.mse_loss  |
 |`CrossEntropy` | torch.nn.functional.cross_entropy  |
 |`CTC` | torch.nn.functional.ctc_loss  |
 |`NLL` | torch.nn.functional.nll_loss  |
 |`PoissonNLL` | torch.nn.functional.poisson_nll_loss  |
 |`GaussianNLL` | torch.nn.functional.gaussian_nll_loss  |
 |`KLDiv` | torch.nn.functional.kl_div  |
 |`BCE`  | torch.nn.functional.binary_cross_entropy  |
 |`Huber`  | torch.nn.functional.huber_loss  |
 |`SmoothL1`  | torch.nn.functional.smooth_l1_loss  |
 |`SoftMargin`  | torch.nn.functional.soft_margin_loss |
