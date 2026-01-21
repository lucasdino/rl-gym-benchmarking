import yaml
from typing import Any, Dict

from configs.config import (
    TrainConfig,
    EnvConfig,
    AlgoConfig,
    SingleNetworkConfig,
    NetworksConfig,
    TrainParams,
)


def load_yaml_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # Env / Algo
    env_cfg = EnvConfig(**raw["env"])
    raw_algo: Dict[str, Any] = raw["algo"]
    algo_cfg = AlgoConfig(
        name=raw_algo["name"],
        gamma=raw_algo["gamma"],
        lr=raw_algo["lr"],
        batch_size=raw_algo["batch_size"],
        seed=raw_algo["seed"],
        extra={
            k: v
            for k, v in raw_algo.items()
            if k not in {"name", "gamma", "lr", "seed", "batch_size"}
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
    train_params = TrainParams(
        total_env_steps=raw_tp["total_env_steps"],
        eval_interval=raw_tp["eval_interval"],
        wandb_project=raw_tp["wandb_project"],
        wandb_group=raw_tp.get("wandb_group"),
        extra={
            k: v
            for k, v in raw_tp.items()
            if k not in {"total_env_steps", "eval_interval", "wandb_project", "wandb_group"}
        } or None,
    )

    return TrainConfig(
        env=env_cfg,
        algo=algo_cfg,
        networks=networks,
        train=train_params,
    )