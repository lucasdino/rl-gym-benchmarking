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

from trainer.helper.env_setup import make_vec_envs
from configs.config import EnvConfig


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
    rows, cols = grid_hw
    num_envs = env_cfg.num_envs
    if rows * cols != env_cfg.num_envs:
        raise ValueError(f"grid_hw={grid_hw} implies {rows*cols} tiles but num_envs={env_cfg.num_envs}")

    envs = make_vec_envs(env_cfg, vectorization_mode=vectorization_mode, render_mode="rgb_array")

    obs, _ = envs.reset(seed=seed)

    done = np.zeros(num_envs, dtype=bool)
    frozen: List[Optional[np.ndarray]] = [None] * num_envs
    frames: List[np.ndarray] = []

    # Loop until every env has finished at least one episode
    while not done.all():
        sub_frames = envs.call("render")  # list[ndarray] (one per sub-env)

        # Freeze each tile once it finishes; vec envs autoreset otherwise
        tiled_inputs: List[np.ndarray] = []
        for i, f in enumerate(sub_frames):
            if f is None:
                # If render returns None, keep frozen frame if available (otherwise placeholder)
                f = frozen[i] if frozen[i] is not None else None

            if done[i] and frozen[i] is not None:
                tiled_inputs.append(_apply_done_overlay(frozen[i]))
            else:
                if f is not None:
                    frozen[i] = f
                    if done[i]:
                        tiled_inputs.append(_apply_done_overlay(f))
                    else:
                        tiled_inputs.append(f)
                else:
                    tiled_inputs.append(np.zeros((64, 64, 3), dtype=np.uint8))

        frames.append(tile_grid(tiled_inputs, grid_hw=grid_hw, pad=pad))

        obs_t = to_tensor_obs(obs)
        batched_actions = algo.act(obs_t, eval_mode=True)
        env_actions = actions_to_env(batched_actions.action)

        obs, _, term, trunc, _ = envs.step(env_actions)
        done |= (term | trunc)

    envs.close()
    return frames