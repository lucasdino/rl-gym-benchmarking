"""
trainer/video_grid.py

Record a tiled (e.g. 4x4) MP4 from a Gymnasium vector environment by:
- creating a VecEnv with render_mode="rgb_array"
- rendering each sub-env each step via envs.call("render")
- freezing tiles after an env finishes its first episode (VecEnv autoreset would otherwise restart it)
- tiling frames into a grid and returning the mosaic frame list
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, List, Any
import numpy as np
import gymnasium as gym
from dataclasses import replace

from trainer.helper.env_setup import make_vec_envs
from configs.config import EnvConfig


def _make_atari_render_env(env_name: str, num_envs: int) -> Any:
    """Create a color Atari env for rendering (no grayscale, no stacking)."""
    from ale_py.vector_env import AtariVectorEnv
    return AtariVectorEnv(
        game=env_name,
        num_envs=num_envs,
        frameskip=4,
        grayscale=False,
        stack_num=1,
        img_height=84,
        img_width=84,
        maxpool=True,
        reward_clipping=False,
        noop_max=30,
        use_fire_reset=True,
        episodic_life=False,
        full_action_space=False,
    )


def _extract_atari_color_frames(obs: np.ndarray) -> List[np.ndarray]:
    """
    Extract color frames from Atari color env observations.
    obs: [num_envs, 1, H, W, 3]
    
    return: list of [H, W, 3] uint8 frames
    """
    frames = []
    for i in range(obs.shape[0]):
        frame = obs[i, 0]  # [H, W, 3]
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        frames.append(frame)
    return frames


def _pad_to_hw(img: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.zeros((H, W, 3), dtype=img.dtype)
    out[:h, :w, :3] = img[:, :, :3]
    return out


def _normalize_frame(img: np.ndarray, pad: int = 2) -> np.ndarray:
    # Ensure uint8 RGB
    if img is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    if img.ndim == 2:  # grayscale -> rgb
        img = np.repeat(img[..., None], 3, axis=-1)

    if img.shape[-1] == 4:  # drop alpha
        img = img[..., :3]

    if pad > 0:
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)

    return img


def _apply_done_overlay(img: np.ndarray, alpha: float = 0.5, gray: int = 128) -> np.ndarray:
    """Overlay a gray shader on a frame to indicate completion."""
    if img is None:
        return img
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    overlay = np.full_like(img, gray, dtype=np.uint8)
    blended = (img.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha)
    return blended.astype(np.uint8)


def tile_grid(imgs: List[np.ndarray], grid_hw: Tuple[int, int], pad: int = 2) -> np.ndarray:
    rows, cols = grid_hw
    expected = rows * cols
    if len(imgs) != expected:
        raise ValueError(f"Need {expected} frames, got {len(imgs)}")

    proc = [_normalize_frame(im, pad=pad) for im in imgs]
    H = max(im.shape[0] for im in proc)
    W = max(im.shape[1] for im in proc)
    proc = [_pad_to_hw(im, H, W) for im in proc]

    rows_out = []
    for r in range(rows):
        row = np.concatenate(proc[r * cols : (r + 1) * cols], axis=1)
        rows_out.append(row)

    return np.concatenate(rows_out, axis=0)


def _unwrap_vec_for_render(envs: Any) -> Any:
    """
    return
    Return first wrapper with `call` or `render`.
    """
    current = envs
    while current is not None:
        if hasattr(current, "call") or hasattr(current, "render"):
            return current
        current = getattr(current, "env", None)
    raise AttributeError("No vector env with render or call")


def record_vec_grid_video(
    *,
    env_cfg: EnvConfig,
    algo: Any,
    to_tensor_obs: Callable[[np.ndarray], Any],
    actions_to_env: Callable[[Any], np.ndarray],
    grid_hw: Tuple[int, int] = (4, 4),
    seed: Optional[int] = None,
    pad: int = 2,
    vectorization_mode: str = "sync",
) -> List[np.ndarray]:
    """
    Returns: list of mosaic frames (H x W x 3 uint8).

    algo must expose: algo.act(obs_tensor, eval_mode=True) -> object with .action tensor
    to_tensor_obs must convert numpy obs -> torch tensor on correct device/dtype
    actions_to_env must convert action tensor -> numpy actions suitable for env.step
    """
    fixed_num_envs = 16
    fixed_grid_hw = (4, 4)
    env_cfg = replace(env_cfg, num_envs=fixed_num_envs)
    rows, cols = fixed_grid_hw
    num_envs = fixed_num_envs

    is_atari = env_cfg.is_atari
    render_mode = None if is_atari else "rgb_array"
    envs = make_vec_envs(env_cfg, vectorization_mode=vectorization_mode, render_mode=render_mode)

    # For Atari, create a parallel color env for rendering
    atari_render_env = None
    atari_render_obs = None
    if is_atari:
        atari_render_env = _make_atari_render_env(env_cfg.name, num_envs)
        atari_render_obs, _ = atari_render_env.reset(seed=seed)

    obs, _ = envs.reset(seed=seed)

    done = np.zeros(num_envs, dtype=bool)
    frozen: List[Optional[np.ndarray]] = [None] * num_envs
    frames: List[np.ndarray] = []

    while not done.all():
        if is_atari:
            sub_frames = _extract_atari_color_frames(atari_render_obs)
        else:
            render_env = _unwrap_vec_for_render(envs)
            if hasattr(render_env, "call"):
                sub_frames = render_env.call("render")
            else:
                sub_frames = render_env.render()

        tiled_inputs: List[np.ndarray] = []
        for i, f in enumerate(sub_frames):
            if done[i]:
                # Already finished - use frozen frame with overlay
                if frozen[i] is not None:
                    tiled_inputs.append(_apply_done_overlay(frozen[i]))
                else:
                    tiled_inputs.append(np.zeros((64, 64, 3), dtype=np.uint8))
            else:
                # Still running - update frozen and show live frame
                if f is not None:
                    frozen[i] = f
                    tiled_inputs.append(f)
                else:
                    tiled_inputs.append(frozen[i] if frozen[i] is not None else np.zeros((64, 64, 3), dtype=np.uint8))

        frames.append(tile_grid(tiled_inputs, grid_hw=grid_hw, pad=pad))

        obs_t = to_tensor_obs(obs)
        batched_actions = algo.act(obs_t, eval_mode=True)
        env_actions = actions_to_env(batched_actions.action)

        obs, _, term, trunc, infos = envs.step(env_actions)
        if atari_render_env is not None:
            atari_render_obs, _, _, _, _ = atari_render_env.step(env_actions)

        done |= (term | trunc)

    envs.close()
    if atari_render_env is not None:
        atari_render_env.close()
    return frames