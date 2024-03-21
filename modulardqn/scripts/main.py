import gymnasium as gym
from datetime import datetime

import wandb

from DQN import DQN
from DQN.DQN import Mode
from DQN.Utils.configfile import save_config
from DQN.Wrappers.rewardclipping import clip_reward


def create_mode(modes: list):
    if modes is None:
        return Mode.DQN, ["modulardqn"]

    if Mode.RAINBOW in modes:
        return Mode.RAINBOW, ["Rainbow"]
    else:
        mode = 0
        tags = []
        for m in modes:
            mode |= m
            tags.append(str(m).removeprefix("Mode."))

        return mode, tags


def main(args):
    # Chooses sensible decay interval, minimum size to update and steps between recordings depending on number of steps
    if args.edi is None:
        args.edi = int(args.steps / 100)

    if 'render_mode' not in args.kwargs:
        args.kwargs['render_mode'] = 'rgb_array'

    env = gym.make(id=args.env, **args.kwargs)

    # Add reward clipping wrapper if enabled in command line argument
    if args.clip is not None:
        env = clip_reward(env, args.clip)

    if args.n_step > 1:
        if args.mode is None:
            args.mode = [Mode.N_STEP]
        else:
            args.mode.append(Mode.N_STEP)

    mode, tags = create_mode(args.mode)

    if mode.RAINBOW in mode and args.n_step is None:
        args.n_step = 3

    dqn = DQN.DQN(
        policy=args.policy,
        env=env,
        device=args.device,
        replay_memory_size=args.rm_size,
        seed=args.seed,
        epsilon_max=args.epsilon,
        epsilon_decay_step=args.eds,
        epsilon_decay_interval=args.edi,
        epsilon_min=args.e_min,
        lr=args.lr,
        optimizer=args.optimizer,
        gamma=args.gamma,
        batch_size=args.bs,
        use_wandb=args.wandb,
        wandb_tags=tags if args.tags is None else tags.append(args.tags),
        log_interval=args.li,
        mode=mode,
        tau=args.tau,
        store_model=args.store_model,
        n_step=args.n_step,
        alpha=args.alpha,
        beta=args.beta,
        train_freq=args.t_freq,
        frame_skip=args.skip_frames,
        loss_func=args.loss,
        image_size=args.obs_size
    )

    # Start timestamp to name folder after
    record_path = f"./recordings/{env.spec.id.replace('/', '-')}/{str(dqn.mode).removeprefix('Mode.').replace('|', '-')}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{wandb.run.name if wandb.run is not None else str()}"

    # Store command line config in text file
    save_config(args.__dict__, record_path)

    if args.load_file is not None:
        dqn.load(args.load_file)

    dqn.learn(
        steps=args.steps,
        episodes_trigger=args.rec_trigger,
        record_path=record_path,
        use_progress_bar=args.progress,
        heatmap_opacity=args.heatmaps,
        generate_grpahs=args.graphs
    )

    dqn.save(f"{record_path}/policy.pt")

    dqn.episode_log.serialize(f"{record_path}/log.json")

    if args.wandb:
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(local_path=f"{record_path}/policy.pt")
        wandb.run.log_artifact(artifact)
        wandb.run.finish()

    env.close()
