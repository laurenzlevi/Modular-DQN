import argparse
import re

from DQN.DQN import Mode
from ast import literal_eval


class StoreDictKeyPair(argparse.Action):
    """
    Store key value pairs as dict
    Credit: https://stackoverflow.com/a/42355279
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = {}
        for kv in values:
            k, v = kv.split("=")
            kwargs[k] = v
        setattr(namespace, self.dest, kwargs)


class StoreTuple(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if re.match(r"^\(([1-9][0-9]*,?){3}\)$", values.replace(' ', '')) is None:
            raise ValueError(f"Illegal size {values}")

        tuple = literal_eval(values)
        setattr(namespace, self.dest, tuple)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '--environment', required=True,
                        help='gymnasium environment to learn (e.g. CartPole-v1)')
    parser.add_argument('-s', '--steps', required=True, help="Number of training steps", type=int)
    parser.add_argument('-π', '--policy', required=True, help="Network to use (MLP, CNN). Use load for custom network.",
                        choices=['MLP', 'CNN'])
    parser.add_argument('--device', default=None, help="Device used by pytorch (cpu, cuda)")
    parser.add_argument('--t_freq', '--training_freq', default=4, help="Number of steps between model updates",
                        type=int)
    parser.add_argument('--lr', '--learning_rate', default=1e-3, help="Learning rate used (Default: 1e-3)", type=float)
    parser.add_argument('-ε', '--epsilon', default=1, help="Initial epsilon used for epsilon-greedy policy",
                        type=float)
    parser.add_argument('--edi', '--epsilon_decay_interval', default=None,
                        help="Amount of episodes between epsilon decay steps (Default: 1e3)", type=int)
    parser.add_argument("--eds", "--epsilon_decay_step", default=0.1, help="Size of epsilon decay steps (Default: 0.1)",
                        type=float)
    parser.add_argument('--e_min', '--epsilon_min', default=0.1,
                        help="Minimal epsilon value (Default: 0.1)", type=float)
    parser.add_argument('-𝛾', '--gamma', default=0.95, help="Discount factor for future rewards (Default: 0.95)",
                        type=float)
    parser.add_argument('-𝜏', '--tau', default=0.01, help="Soft update factor of target network (Default: 0.01)",
                        type=float)
    parser.add_argument('--bs', '--batch_size', default=32,
                        help="Batch size used to update the Q-Function (Default: 32)", type=int)
    parser.add_argument('--seed', default=None, help="Seed for environment (Default: None)", type=int)
    parser.add_argument('--rm_size', '--replay_memory_size', default=1_000_000,
                        help="Number of stored experiences at a time (Default: 1e7)", type=int)
    parser.add_argument('--rec_trigger', default=None, help="Number of episodes between episode recordings", type=int)

    # Weigths and Biases
    parser.add_argument("--wandb", default=False, help="Should progress be logged to wandb", type=bool)
    parser.add_argument("--tags", default=None, nargs='*', help="Tags to add to the wandb run")

    parser.add_argument('--li', '--log_interval', default=10, help="Number of episodes between logging", type=int)
    parser.add_argument('--load_file', default=None,
                        help="Relative path to a file containing a pytorch model to be loaded")
    parser.add_argument('--optimizer', default="SGD", help="Name of the optimizer to be used (e.g. 'SGD', 'Adam' etc.)")

    # wrappers
    parser.add_argument('--skip_frames', '--skp', default=1,
                        help="The number of frames to skip each step (actions are held for that amount)", type=int)
    parser.add_argument('--clip', '--reward_clipping', default=None,
                        help="Set to 0 for hard clipping or any other scale for soft clipping divided by this factor",
                        type=int)

    # Priority replay parameters
    parser.add_argument('-α', '--alpha', default=0.5, help="Alpha for priority replay", type=float)
    parser.add_argument('-β', '--beta', default=0.7, help="Beta for priority replay", type=float)

    parser.add_argument('--store_model', action='store_true')

    parser.add_argument('--per', help='Enables prioritized replay memory', action='append_const', dest='mode',
                        const=Mode.PER)
    parser.add_argument('--ddqn', help='Enables double deep q-learning', action='append_const', dest='mode',
                        const=Mode.DDQN)
    parser.add_argument('--n_step', help='Enables n-step dqn', type=int, default=1)
    parser.add_argument('--noisy', help='Enables noisy linear layers', action='append_const', dest='mode',
                        const=Mode.NOISY)
    parser.add_argument('--dueling', help='Enables dueling networks architecture', action='append_const', dest='mode',
                        const=Mode.DUELING)
    parser.add_argument('--cat', '--categorical', help='Enables distributional dqn', action='append_const',
                        dest='mode', const=Mode.CATEGORICAL)
    parser.add_argument('--rainbow', help='Enables rainbow dqn', action='append_const', dest='mode',
                        const=Mode.RAINBOW)

    parser.add_argument('--kwargs', help='kwargs passed to the environment on creation', action=StoreDictKeyPair,
                        nargs='+', metavar='KEY=VAL', dest='kwargs', default={})

    parser.add_argument('--progress', action='store_true', default=False)
    parser.add_argument('--loss', default='SmoothL1')
    parser.add_argument('--obs_size',
                        help='If using pixel obs rescales to (stack_size, width, height), accepts Tuple[int, int, int]',
                        action=StoreTuple, dest='obs_size', default=None)

    parser.add_argument('--heatmaps', default=0.0, type=float, help="Heatmap opacity or 0.0 to not generate heatmaps")
    parser.add_argument('--graphs', action='store_true', default=False)

    return parser
