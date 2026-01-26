import yaml
from typing import Any, Dict

from configs.config import (
    TrainConfig,
    EnvConfig,
    AlgoConfig,
    SingleNetworkConfig,
    NetworksConfig,
    TrainParams,
    SamplerConfig,
)


def load_yaml_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # Env / Algo
    raw_env: Dict[str, Any] = raw["env"]
    env_cfg = EnvConfig(
        name=str(raw_env["name"]),
        num_envs=int(raw_env["num_envs"]),
        max_episode_steps=None if raw_env.get("max_episode_steps") is None else int(raw_env.get("max_episode_steps")),
    )
    raw_algo: Dict[str, Any] = raw["algo"]
    algo_cfg = AlgoConfig(
        name=str(raw_algo["name"]),
        gamma=float(raw_algo["gamma"]),
        lr_start=float(raw_algo["lr_start"]),
        lr_end=float(raw_algo["lr_end"]),
        lr_warmup_env_steps=int(raw_algo["lr_warmup_env_steps"]),
        update_every_steps=int(raw_algo["update_every_steps"]),
        batch_size=int(raw_algo["batch_size"]),
        seed=int(raw_algo["seed"]),
        extra={
            k: v
            for k, v in raw_algo.items()
            if k not in {"name", "gamma", "lr", "lr_start", "lr_end", "lr_warmup_env_steps", "seed", "batch_size"}
        } or {},
    )

    # Networks: key in YAML becomes the logical name (policy, value, q1, ...)
    raw_networks: Dict[str, Dict[str, Any]] = raw["networks"]
    network_cfgs: Dict[str, NetworksConfig] = {}
    for net_name, net_dict in raw_networks.items():
        network_cfgs[net_name] = SingleNetworkConfig(
            name=net_dict.get("name", net_name),
            network_type=net_dict["network_type"],
            network_args=net_dict.get("network_args", {}),
        )
    networks = NetworksConfig(networks=network_cfgs)

    # Train params
    raw_tp: Dict[str, Any] = raw["train"]
    raw_logging_method = raw_tp.get("logging_method", ["console"])
    train_params = TrainParams(
        total_env_steps=int(raw_tp["total_env_steps"]),
        eval_interval=int(raw_tp["eval_interval"]),
        log_interval=int(raw_tp["log_interval"]),
        logging_method=[str(m) for m in raw_logging_method],
        console_log_train=bool(raw_tp.get("console_log_train", True)),
        wandb_project=str(raw_tp["wandb_project"]),
        wandb_group=None if raw_tp.get("wandb_group") is None else str(raw_tp.get("wandb_group")),
        wandb_run=None if raw_tp.get("wandb_run") is None else str(raw_tp.get("wandb_run")),
        save_video_last_eval=bool(raw_tp.get("save_video_last_eval", True)),
        video_save_dir=str(raw_tp.get("video_save_dir", "saved_data/saved_videos")),
        extra={
            k: v
            for k, v in raw_tp.items()
            if k not in {"total_env_steps", "eval_interval", "log_interval", "logging_method", "console_log_train", "wandb_project", "wandb_group", "wandb_run", "save_video_last_eval", "video_save_dir"}
        } or None,
    )

    
    # Sampler
    raw_sampler: Dict[str, Any] = raw.get("sampler", {})
    sampler_args = {k: v for k, v in raw_sampler.items() if k != "name"}
    sampler_args["total_steps"] = int(train_params.total_env_steps)
    sampler_cfg = SamplerConfig(
        name=str(raw_sampler.get("name", "greedy")),
        args=sampler_args,
    )

    return TrainConfig(
        env=env_cfg,
        algo=algo_cfg,
        networks=networks,
        train=train_params,
        sampler=sampler_cfg,
    )