import wandb
import gymnasium as gym

from DQN.DQN import DQN


'''
Start a sweep from code, modify the sweep_config dict to change values, change count to edit sweep length
To start a sweep from the cli, modify sweep_config.yaml and use:
wandb sweep --project <project> <config.yaml>
wandb agent <sweep_id> --count <count> (count is optional)
Additional agents can be used by executing `wandb agent <sweep_id>` in another cli
'''


def start_run():
    wandb.init()
    dqn = DQN(
        policy='MLP',
        env=gym.make(wandb.config['env'], render_mode='rgb_array'),
        device=wandb.config['device'],
        lr=wandb.config['lr'],
        epsilon_max=wandb.config['epsilon'],
        epsilon_min=wandb.config['epsilon_min'],
        epsilon_decay_step=wandb.config['epsilon_step'],
        epsilon_decay_interval=wandb.config['epsilon_interval'],
        gamma=wandb.config['gamma'],
        batch_size=wandb.config['batch_size'],
        optimizer=wandb.config['optimizer'],
        use_wandb=True,
        wandb_tags=['sweep']
    )
    dqn.learn(steps=wandb.config['episodes'], episodes_trigger=None, record_path=None)


def start_sweep():
    sweep_config = {
        'name': 'test-sweep',
        'method': 'bayes',
        'metric': {
            'name': 'Mean Episode Reward',
            'goal': 'maximize'
        },
        'parameters': {
            'env': {'value': 'CartPole-v1'},
            'episodes': {'value': 200},
            'device': {'value': 'cpu'},
            'lr': {'values': [0.01, 0.001, 0.0001]},
            'epsilon': {'value': 1.0},
            'epsilon_min': {'value': 0.1},
            'epsilon_step': {'value': 0.1},
            'epsilon_interval': {'value': 10},
            'gamma': {'values': [0.9, 0.92, 0.95, 0.97, 0.99]},
            'batch_size': {'value': 32},
            'wandb': {'value': True},
            'optimizer': {'values': ['Adam', 'SGD']},
            'mode': {'value': 'modulardqn'}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config['name'])
    wandb.agent(sweep_id=sweep_id, function=start_run, count=10)
