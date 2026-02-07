from __future__ import annotations

from typing import Any
import gymnasium as gym
import numpy as np
from ale_py.vector_env import AtariVectorEnv

from gymnasium.wrappers.vector import (
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    ClipReward,
    DtypeObservation,
    TransformObservation,
)

ALLOWED_ENV_ARGS = {
    "render_mode",
}


def make_atari_vec_envs(
    env_name: str,
    num_envs: int,
    stack_size: int = 4,
    grayscale: bool = True,
    max_episode_steps: int | None = None,
    seed: int | None = None,
    render_mode: str | None = None,
    vectorization_mode: str = "async",
) -> AtariVectorEnv:
    """
    Create vectorized Atari environments using ale_py's AtariVectorEnv.

    return: AtariVectorEnv with standard Atari preprocessing.
    """
    kwargs = dict(
        game=env_name,
        num_envs=num_envs,
        frameskip=4,
        grayscale=grayscale,
        stack_num=stack_size,
        img_height=84,
        img_width=84,
        maxpool=True,
        reward_clipping=False,
        noop_max=30,
        use_fire_reset=True,
        episodic_life=False,
        full_action_space=False,
    )

    if max_episode_steps is not None:
        kwargs["max_num_frames_per_episode"] = max_episode_steps * 4  # account for frameskip

    envs = AtariVectorEnv(**kwargs)
    if render_mode is not None and hasattr(envs, "render_mode"):
        envs.render_mode = render_mode
    return envs


def make_standard_vec_envs(
    env_name: str,
    num_envs: int,
    max_episode_steps: int | None = None,
    env_args: dict[str, Any] | None = None,
    render_mode: str | None = None,
    vectorization_mode: str = "async",
) -> gym.vector.VectorEnv:
    """
    Create standard vectorized environments without special preprocessing.

    return: VectorEnv
    """
    kwargs = dict(env_args or {})
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    envs = gym.make_vec(
        env_name,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )
    return envs


def _is_uint8_image_space(space: gym.Space) -> bool:
    return (
        isinstance(space, gym.spaces.Box)
        and space.dtype == np.uint8
        and len(space.shape) >= 2
    )


def wrap_vec_envs(envs: gym.vector.VectorEnv, env_cfg: Any, eval: bool = False) -> gym.vector.VectorEnv:
    """
    Wrap a VectorEnv with:
      - episode stats
      - observation normalization
      - reward shaping (disabled for eval)
    """
    # First record stats in raw
    if getattr(env_cfg, "record_episode_stats", True):
        envs = RecordEpisodeStatistics(envs)

    # Obs normalization
    obs_space = envs.single_observation_space
    if env_cfg.is_atari or _is_uint8_image_space(obs_space):
        envs = DtypeObservation(envs, np.float32)
        # envs = TransformObservation(envs, lambda obs: obs / 255.0)
    else:
        if env_cfg.normalize_obs:
            envs = NormalizeObservation(envs, epsilon=env_cfg.extra.get("obs_norm_eps", 1e-8))
            envs.update_running_mean = (not eval)

    # Reward normalization
    if env_cfg.is_atari:
        if not eval:
            envs = ClipReward(envs, -1.0, 1.0)   # standard atari clipping of [-1, 1]
    else:
        if (not eval) and env_cfg.normalize_reward:
            envs = NormalizeReward(envs, gamma=env_cfg.extra['gamma'], epsilon=env_cfg.extra.get("rew_norm_eps", 1e-8))
            envs.update_running_mean = True

    return envs


def make_vec_envs(
    env_cfg: Any,
    vectorization_mode: str = "async",
    render_mode: str | None = None,
    eval: bool = False,
) -> gym.vector.VectorEnv:
    """
    Factory function to create vectorized environments.

    return: VectorEnv (Atari with preprocessing if is_atari=True, else standard)
    """
    env_args = {k: v for k, v in env_cfg.extra.items() if k in ALLOWED_ENV_ARGS}
    render_mode = render_mode if render_mode is not None else env_args.get("render_mode", None)
    seed = env_args.get("seed", None)

    if env_cfg.is_atari:
        envs = make_atari_vec_envs(
            env_name=env_cfg.name,
            num_envs=env_cfg.num_envs,
            stack_size=env_cfg.stack_samples,
            max_episode_steps=env_cfg.max_episode_steps,
            seed=seed,
            render_mode=render_mode,
            vectorization_mode=vectorization_mode,
        )
    else:
        envs = make_standard_vec_envs(
            env_name=env_cfg.name,
            num_envs=env_cfg.num_envs,
            max_episode_steps=env_cfg.max_episode_steps,
            env_args=env_args,
            render_mode=render_mode,
            vectorization_mode=vectorization_mode,
        )

    return wrap_vec_envs(envs, env_cfg=env_cfg, eval=eval)