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
    lr_start:           float
    lr_end:             float
    lr_warmup_env_steps:int
    update_every_steps: int
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
    log_interval:       int
    eval_interval:      int
    logging_method:     list[str]
    console_log_train:  bool
    wandb_project:      str
    wandb_group:        str | None = None
    wandb_run:          str | None = None
    save_video_last_eval: bool = True
    video_save_dir:     str = "saved_data/saved_videos"
    extra:              Dict[str, Any] | None = None

@dataclass
class SamplerConfig:
    name:               str
    args:               Dict[str, Any]


# Config that encompasses all our configs used
@dataclass
class TrainConfig:
    env:                EnvConfig
    algo:               AlgoConfig
    networks:           NetworksConfig
    train:              TrainParams
    sampler:            SamplerConfig