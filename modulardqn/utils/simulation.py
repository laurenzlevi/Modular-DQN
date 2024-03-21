import copy
import warnings

import gymnasium as gym
import numpy as np
import pygame
import torch
import os
import cv2

import wandb
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import concatenate_videoclips, VideoFileClip, CompositeVideoClip

from DQN.Tools.NoisyLinear import NoisyLinear
from DQN.Tools.QNetwork import QNetwork
from DQN.Utils.plotter import Plotter


class DummyPlotter:
    def __init__(self):
        pass

    def add_q_values(self, q_values: torch.Tensor = None):
        pass

    def __update_plot(self, end_index, *lines):
        pass

    def save_animation(self, path: str = "graph.mp4", fps: int = 30, action_names: list[str] = None):
        pass


def record(
        env: gym.Env,
        device: torch.device,
        trunc_steps: int,
        model: QNetwork,
        record_path: str,
        episode: int,
        record_obs: bool,
        epsilon: float,
        heatmap_amount: float = 0.0,
        upscale_output_by: int = 1,
        plot: bool = False
):
    """
    Records one episode of the given environment, truncates if the episode is longer than trunc_steps
    """
    if env.render_mode != "rgb_array":
        raise ValueError(f"Unsupported render mode {env.render_mode} for recording, render mode must be 'rgb_array'!")

    if heatmap_amount < 0 or heatmap_amount > 1:
        raise Exception(f"heatmap_amount must be between 0 and 1, but was: {heatmap_amount}")

    if "render_fps" not in env.metadata:
        warnings.warn(f"Environment does not specify render_fps, defaulting to 30!")
        render_fps = 30
    else:
        render_fps = env.metadata["render_fps"]

    absolute_path = os.path.abspath(record_path)
    frames = []
    heatframes = []
    heatmaps = []

    if record_obs:
        inputs = []

    observation, info = env.reset()

    policy = copy.deepcopy(model.model)

    grad = None
    activations = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal grad
        grad = grad_input[0].detach()
        return None  # We don't want to change the gradients

    def forward_hook(module, input, output):
        nonlocal activations
        activations = input[0].detach()
        return None  # We don't want to change the output

    if 0 < heatmap_amount:
        for m in policy.modules():
            if isinstance(m, torch.nn.Flatten) or isinstance(m, NoisyLinear):
                m.register_full_backward_hook(hook=backward_hook)
                m.register_forward_hook(hook=forward_hook)
                break

    terminated, truncated = False, False
    simulation_reward = 0.0
    total_steps = 0

    # For logging q_values
    if plot:
        plotter = Plotter(num_actions=env.action_space.n)
    else:
        plotter = DummyPlotter()

    while not terminated and not truncated:
        # choose next action
        obs = torch.as_tensor(data=np.array([observation]), dtype=torch.float32, device=device)

        if observation.dtype == np.uint8:
            obs = torch.div(obs, 255.0)

        # use the policy network to choose action
        if heatmap_amount == 0:
            with torch.no_grad():
                q_values = policy(obs)
                plotter.add_q_values(q_values=q_values.cpu())
                action = torch.argmax(q_values, dim=-1).item()
        else:
            out = policy(obs)

            action = torch.argmax(out, dim=-1).item()

            mask = -torch.ones(out.shape, device=device)
            mask[0][action] = 1

            out.backward(gradient=mask, retain_graph=True)

            w = torch.relu(grad.sum(2).unsqueeze(2).sum(3).unsqueeze(3))

            transformed_grad = (activations * w).squeeze().sum(0)
            transformed_grad /= torch.max(transformed_grad)
            transformed_grad *= 255.0
            transformed_grad = transformed_grad.cpu().detach().numpy().astype(np.uint8)
            heatmap = cv2.cvtColor(cv2.applyColorMap(transformed_grad, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

            policy.zero_grad(set_to_none=True)
            plotter.add_q_values(q_values=out.detach().cpu())

        # replace action with probability epsilon
        if np.random.random() < epsilon:
            action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        simulation_reward += reward
        total_steps += 1

        if record_obs:
            if observation.dtype == np.uint8:
                inputs.append(np.transpose(observation[0:3, :, :], axes=(1, 2, 0)))
            else:
                inputs.append(np.transpose(observation[0:3, :, :], axes=(1, 2, 0)) * 255.0)

        if total_steps > trunc_steps:
            truncated = True

        frame = env.render()

        if 0 < heatmap_amount:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), None, 0, 0, interpolation=cv2.INTER_CUBIC)
            heatframe = cv2.addWeighted(heatmap, heatmap_amount, frame, 1 - heatmap_amount, 0)

            heatmaps.append(
                cv2.resize(
                    heatmap,
                    (heatmap.shape[1] * upscale_output_by, heatmap.shape[0] * upscale_output_by),
                    None,
                    0,
                    0,
                    interpolation=cv2.INTER_NEAREST
                )
            )

            heatframes.append(
                cv2.resize(
                    heatframe,
                    (heatframe.shape[1] * upscale_output_by, heatframe.shape[0] * upscale_output_by),
                    None,
                    0,
                    0,
                    interpolation=cv2.INTER_NEAREST
                )
            )

        frames.append(
            cv2.resize(
                frame,
                (frame.shape[1] * upscale_output_by, frame.shape[0] * upscale_output_by),
                None,
                0,
                0,
                interpolation=cv2.INTER_NEAREST
            )
        )

        if terminated or truncated:
            observation, info = env.reset()

    env_video = ImageSequenceClip(frames, fps=render_fps)
    # Create plot video and load it with moviepy
    if plot:
        try:
            plotter.save_animation(f"{absolute_path}/graph.mp4", fps=render_fps, action_names=env.metadata[
                "action_names"] if "action_names" in env.metadata else None)
            plot_video = VideoFileClip(f"{absolute_path}/graph.mp4")
        except:
            print("Graph plotting failed!")
            plot_video = None
    else:
        plot_video = None

    heatmap_video = ImageSequenceClip(heatmaps, fps=render_fps) if len(heatmaps) > 0 else None
    heatframes_video = ImageSequenceClip(heatframes, fps=render_fps) if len(heatframes) > 0 else None

    env_video.write_videofile(
        f"{absolute_path}/{env.spec.id.replace('/', '-')}-episode-{episode}-reward-{simulation_reward:.4f}.mp4",
        logger=None)

    video = combine_video_files(env_video, plot_video, heatmap_video, heatframes_video)

    if video is not None:
        video.write_videofile(
            f"{absolute_path}/{env.spec.id.replace('/', '-')}-episode-{episode}-reward-{simulation_reward:.4f}-viz.mp4",
            logger=None)

    if record_obs:
        env_video = ImageSequenceClip(inputs, fps=render_fps)
        env_video.write_videofile(
            f"{absolute_path}/{env.spec.id.replace('/', '-')}-episode-{episode}-reward-{simulation_reward:.4f}-obs.mp4",
            logger=None)

    print(f"Agent achieved an reward of {simulation_reward}")

    if wandb.run is not None:
        wandb.log({
            'Episode Reward': simulation_reward
        })

    frames.clear()

    return simulation_reward


def combine_video_files(
        env_video: ImageSequenceClip,
        plot_video: VideoFileClip,
        heatmap_video: ImageSequenceClip,
        heatframe_video: ImageSequenceClip
):
    # Resize videos
    env_video = env_video.resize((320, 320))

    if plot_video is not None and heatmap_video is not None:
        plot_video = plot_video.resize((320, 320))
        heatmap_video = heatmap_video.resize((320, 320))
        heatframe_video = heatframe_video.resize((320, 320))

        plot_video = plot_video.set_position((320, 0))
        plot_video = plot_video.set_duration(env_video.duration)

        heatframe_video = heatframe_video.set_position((0, 320))
        heatmap_video = heatmap_video.set_position((320, 320))

        heatframe_video = heatframe_video.set_duration(env_video.duration)
        heatmap_video = heatmap_video.set_duration(env_video.duration)

        size = (640, 640)

        clips = [env_video, plot_video, heatmap_video, heatframe_video]
    elif plot_video is not None:
        plot_video = plot_video.resize((320, 320))

        plot_video = plot_video.set_position((320, 0))
        plot_video = plot_video.set_duration(env_video.duration)

        size = (640, 320)

        clips = [env_video, plot_video]
    elif heatmap_video is not None:
        heatmap_video = heatmap_video.resize((320, 320))
        heatframe_video = heatframe_video.resize((320, 320))

        heatframe_video = heatframe_video.set_position((320, 0))
        heatmap_video = heatmap_video.set_position((640, 0))

        heatframe_video = heatframe_video.set_duration(env_video.duration)
        heatmap_video = heatmap_video.set_duration(env_video.duration)

        size = (960, 320)

        clips = [env_video, heatmap_video, heatframe_video]
    else:
        return None

    # Combine videos
    return CompositeVideoClip(clips, size=size)
