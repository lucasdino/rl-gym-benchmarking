"""
Record a tiled grid MP4 from a Gymnasium vector environment.
Freezes tiles after their first episode, applies a done overlay,
and returns the mosaic frame list.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, List, Any
import numpy as np
from dataclasses import replace

from trainer.helper.env_setup import make_vec_envs, make_atari_vec_envs
from configs.config import EnvConfig


def _make_atari_seeds(seed: int, num_envs: int) -> np.ndarray:
    base = int(seed) % 2_147_483_647
    return ((base + np.arange(num_envs, dtype=np.int64)) % 2_147_483_647).astype(np.int32)


def _rgb_to_grayscale(rgb_obs: np.ndarray) -> np.ndarray:
    """
    ALE NTSC/BT.601 grayscale: round(0.2989*R + 0.5870*G + 0.1140*B)
    rgb_obs: [num_envs, stack, H, W, 3] uint8

    return: [num_envs, stack, H, W] uint8
    """
    r = rgb_obs[..., 0].astype(np.float64)
    g = rgb_obs[..., 1].astype(np.float64)
    b = rgb_obs[..., 2].astype(np.float64)
    return np.round(0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)


def _extract_atari_rgb_frames(obs: np.ndarray) -> List[np.ndarray]:
    """
    Extract last frame from stacked RGB Atari obs.
    obs: [num_envs, stack, H, W, 3]

    return: list of [H, W, 3] uint8 frames
    """
    return [obs[i, -1] for i in range(obs.shape[0])]


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
    seed: int | None = None,
    pad: int = 2,
    vectorization_mode: str = "sync",
) -> List[np.ndarray]:
    """
    Returns: list of mosaic frames (H x W x 3 uint8).
    """
    rows, cols = grid_hw
    num_envs = rows * cols
    env_cfg = replace(env_cfg, num_envs=num_envs)
    is_atari = env_cfg.is_atari

    if is_atari:
        envs = make_atari_vec_envs(
            env_name=env_cfg.name, num_envs=num_envs,
            stack_size=env_cfg.stack_samples, grayscale=False,
            max_episode_steps=env_cfg.max_episode_steps,
        )
        seeds = _make_atari_seeds(seed, num_envs) if seed is not None else None
        obs, info = envs.reset(seed=seeds)
        prev_lives = info["lives"].copy()

        # Find FIRE in minimal action set (not always index 1)
        from ale_py import Action as AleAction
        action_set = envs.ale.get_action_set()
        fire_idx = None
        for idx, a in enumerate(action_set):
            if a == AleAction.FIRE:
                fire_idx = idx
                break
    else:
        envs = make_vec_envs(env_cfg, vectorization_mode=vectorization_mode, render_mode="rgb_array", eval=True)
        obs, _ = envs.reset(seed=seed)
        prev_lives = None

    done = np.zeros(num_envs, dtype=bool)
    frozen: List[Optional[np.ndarray]] = [None] * num_envs
    frames: List[np.ndarray] = []
    needs_fire = np.zeros(num_envs, dtype=bool)

    while not done.all():
        if is_atari:
            sub_frames = _extract_atari_rgb_frames(obs)
        else:
            re = _unwrap_vec_for_render(envs)
            sub_frames = re.call("render") if hasattr(re, "call") else re.render()

        tiled_inputs: List[np.ndarray] = []
        for i, f in enumerate(sub_frames):
            if done[i]:
                tiled_inputs.append(_apply_done_overlay(frozen[i]) if frozen[i] is not None else np.zeros((64, 64, 3), dtype=np.uint8))
            else:
                if f is not None:
                    frozen[i] = f.copy()
                tiled_inputs.append(f if f is not None else (frozen[i] if frozen[i] is not None else np.zeros((64, 64, 3), dtype=np.uint8)))

        frames.append(tile_grid(tiled_inputs, grid_hw=grid_hw, pad=pad))

        policy_obs = _rgb_to_grayscale(obs) if is_atari else obs
        env_actions = actions_to_env(algo.act(to_tensor_obs(policy_obs), eval_mode=True).action)

        # After a life loss some games wait for FIRE to continue
        if is_atari and fire_idx is not None and needs_fire.any():
            env_actions[needs_fire & ~done] = fire_idx
            needs_fire[:] = False

        obs, _, term, trunc, info = envs.step(env_actions)

        if is_atari:
            lives = info["lives"]
            needs_fire = (lives < prev_lives) & (lives > 0) & ~done
            episode_ended = ((prev_lives > 0) & (lives == 0)) | (lives > prev_lives)
            done |= (episode_ended | trunc)
            prev_lives = lives.copy()
        else:
            done |= (term | trunc)

    envs.close()
    return frames