from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class EnvConfig:
    name:               str
    num_envs:           int
    max_episode_steps:  int | None = None

@dataclass
class AlgoConfig:
    name:               str
    gamma:              float
    lr:                 float
    batch_size:         int
    seed:               int
    extra:              Dict[str, Any]

@dataclass
class SingleNetworkConfig:
    name:               str
    network_type:       str
    network_args:       dict

@dataclass
class NetworksConfig:
    networks:           Dict[str, SingleNetworkConfig]

@dataclass
class TrainParams:
    total_env_steps:    int
    eval_interval:      int
    wandb_project:      str
    wandb_group:        str | None = None
    extra:              Dict[str, Any] | None = None


# Config that encompasses all our configs used
@dataclass
class TrainConfig:
    env:                EnvConfig
    algo:               AlgoConfig
    networks:           NetworksConfig
    train:              TrainParams