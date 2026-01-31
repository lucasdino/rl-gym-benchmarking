from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    name:               str
    num_envs:           int = 1
    max_episode_steps:  int | None = None
    is_atari:           bool = False
    stack_stamples:     int = 1    # In Atari you want to stack 4 frames
    env_args:           Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlgoConfig:
    name:                       str
    gamma:                      float = 0.99
    lr_start:                   float = 2.5e-4
    lr_end:                     float = 2.5e-4
    lr_warmup_env_steps:        int = 0
    update_every_steps:         int = 1    # Allows you to only do weight updates every 'k' steps. Default to 1 (update every step)
    use_action_for_steps_train: int = 1    # Number of env steps to use same action for. If this is >1 then an action is reused
    use_action_for_steps_eval:  int = 1    # ^ but for evaluation
    batch_size:                 int = 32   # Training minibatch size
    n_step:                     int = 1    # n-step TD learning
    seed:               int | None = None  # If None (unset), random seed is generated
    extra:              Dict[str, Any] = field(default_factory=dict)

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
    eval_envs:          int
    logging_method:     list[str]
    threshold_exit_training_on_eval_score: float | None = None   # Score threshold to end training early if exceeds this
    console_log_train:  bool = False
    wandb_project:      str = "RL-Gym-Benchmarking"
    wandb_group:        str | None = None
    run_name:           str | None = None
    num_seeds:          int = 1       # Number of seeds to run training on
    save_result:        bool = False  # If True, save final aggregated results to saved_data/saved_plots/{run_name}
    video_save_dir:     str = "saved_data/saved_videos"
    algo_save_dir:      str = "saved_data/saved_algos"
    save_video_at_end:  bool = True
    save_algo_at_end:   bool = True
    extra:              Dict[str, Any] | None = None

@dataclass
class SamplerConfig:
    name:               str
    args:               Dict[str, Any]


# Set this true w/ values if you want to load a pretrained model for inference
@dataclass
class InferenceConfig:
    inference_only:     bool = False
    override_cfg:       bool = False
    algo_path:          str | None = None


# Config that encompasses all our configs used
@dataclass
class TrainConfig:
    env:                EnvConfig
    algo:               AlgoConfig
    networks:           NetworksConfig
    train:              TrainParams
    sampler:            SamplerConfig
    inference:          InferenceConfig


def config_to_dict(obj: Any) -> Any:
    """
    Recursively convert dataclasses (and nested dicts/lists) to plain dicts.
    return
    """
    from dataclasses import fields, is_dataclass
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: config_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: config_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [config_to_dict(v) for v in obj]
    return obj