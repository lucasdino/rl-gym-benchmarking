from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    name:               str
    num_envs:           int = 1
    max_episode_steps:  int | None = None
    normalize_obs:      bool = True        # Normalize observations
    normalize_reward:   bool = True        # Normalize rewards (if Atari, this leads to clipping; otherwise typicaly normalization)
    is_atari:           bool = False
    stack_samples:      int = 4            # Only applies to Atari (typically you want to stack 4 frames)
    extra:              Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgoConfig:
    name:                       str
    seed:                       int | None = None   # If None (unset), random seed is generated
    gamma:                      float = 0.99
    n_step:                     int = 1             # n-step TD learning
    lr_start:                   float = 2.5e-4
    lr_end:                     float = 2.5e-4
    lr_warmup_env_steps:        int = 0
    batch_size:                 int = 32            # Training minibatch size
    update_every_steps:         int = 1             # Allows you to only do weight updates every 'k' steps. Default to 1 (update every step)
    use_action_for_steps_train: int = 1             # Number of env steps to use same action for. If this is >1 then an action is reused
    use_action_for_steps_eval:  int = 1             # ^ but for evaluation
    extra:              Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainParams:
    run_name:           str
    total_env_steps:    int
    log_interval:       int                                     # How often you log training results (# env steps)
    eval_interval:      int                                     # How often you run eval (# env steps)
    eval_envs:          int                                     # Num envs to do eval on
    logging_method:     list[str]                               # e.g. [console, wandb]
    
    num_seeds:          int = 1                                 # Number of seeds to train on
    console_log_train:  bool = False
    threshold_exit_training_on_eval_score: float | None = None  # Score threshold to end training early if exceeds this
    wandb_project:      str = "RL-Gym-Benchmarking"
    wandb_group:        str | None = None
    save_result:        bool = False                            # If True, save final aggregated results to saved_data/saved_plots/{run_name}
    video_save_dir:     str = "saved_data/saved_videos"
    algo_save_dir:      str = "saved_data/saved_algos"
    save_video_at_end:  bool = True
    save_algo_at_end:   bool = True
    extra:              Dict[str, Any] | None = None


@dataclass
class SamplerConfig:
    name:               str
    extra:              Dict[str, Any]
# For SamplerConfig, here are two common setups:
#   name: 'epsilon_greedy'; starting_epsilon: 1.0; ending_epsilon: 0.1; warmup_steps: 0; decay_until_step: 150000
#   name: 'boltzman'; temperature: 1.0   (this is 'softmax sampling' for you LLM folks)

@dataclass
class SingleNetworkConfig:
    name:               str
    network_type:       str
    network_args:       dict


@dataclass
class NetworksConfig:
    networks:           Dict[str, SingleNetworkConfig]


# Set this true w/ values if you want to load a pretrained model for inference
@dataclass
class InferenceConfig:
    inference_only:     bool = False            # If true, only do eval
    algo_path:          str | None = None
    override_cfg:       bool = False            # If true, after loading your algo you then override the original config


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


# Keys used in each config
ENV_KEYS = {
    "name",
    "num_envs",
    "max_episode_steps",
    "normalize_obs",
    "normalize_reward",
    "is_atari",
    "stack_samples",
}
ALGO_KEYS = {
    "name",
    "seed",
    "gamma",
    "n_step",
    "lr_start",
    "lr_end",
    "lr_warmup_env_steps",
    "batch_size",
    "update_every_steps",
    "use_action_for_steps_train",
    "use_action_for_steps_eval",
}
TRAIN_KEYS = {
    "run_name",
    "total_env_steps",
    "log_interval",
    "eval_interval",
    "eval_envs",
    "logging_method",
    "console_log_train",
    "threshold_exit_training_on_eval_score",
    "wandb_project",
    "wandb_group",
    "num_seeds",
    "save_result",
    "video_save_dir",
    "algo_save_dir",
    "save_video_at_end",
    "save_algo_at_end",
}
SAMPLER_KEYS = {"name"}