import yaml
import random
from typing import Any, Dict

from configs.config import (
    TrainConfig,
    EnvConfig, ENV_KEYS,
    AlgoConfig, ALGO_KEYS,
    TrainParams, TRAIN_KEYS,
    SamplerConfig, SAMPLER_KEYS,
    InferenceConfig,
    SingleNetworkConfig,
    NetworksConfig,
)


def _extras(raw: Dict[str, Any], keys: set[str]) -> Dict[str, Any]:
    """ Split extra keys from raw. """
    return {k: v for k, v in raw.items() if k not in keys}


def load_yaml_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # --- Env ---
    raw_env: Dict[str, Any] = raw["env"]
    env_cfg = EnvConfig(
        name =              str(raw_env["name"]),
        num_envs =          int(raw_env["num_envs"]),
        max_episode_steps=(
            None
            if raw_env.get("max_episode_steps") is None
            else int(raw_env.get("max_episode_steps"))
        ),
        normalize_obs =     bool(raw_env.get("normalize_obs", EnvConfig.normalize_obs)),
        normalize_reward =  bool(raw_env.get("normalize_reward", EnvConfig.normalize_reward)),
        is_atari =          bool(raw_env.get("is_atari", EnvConfig.is_atari)),
        stack_samples =     int(raw_env.get("stack_samples", EnvConfig.stack_samples)),
        extra =             _extras(raw_env, ENV_KEYS),
    )

    # --- Algo ---
    raw_algo: Dict[str, Any] = raw["algo"]
    seed = raw_algo.get("seed")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    algo_cfg = AlgoConfig(
        name =                      str(raw_algo["name"]),
        seed =                      int(seed),
        gamma =                     float(raw_algo["gamma"]),
        n_step =                    int(raw_algo.get("n_step", AlgoConfig.n_step)),
        lr_start =                  float(raw_algo["lr_start"]),
        lr_end =                    float(raw_algo["lr_end"]),
        lr_warmup_env_steps =       int(raw_algo["lr_warmup_env_steps"]),
        batch_size =                int(raw_algo["batch_size"]),
        update_every_steps =        int(raw_algo["update_every_steps"]),
        use_action_for_steps_train =int(raw_algo.get("use_action_for_steps_train", AlgoConfig.use_action_for_steps_train)),
        use_action_for_steps_eval = int(raw_algo.get("use_action_for_steps_eval", AlgoConfig.use_action_for_steps_eval)),
        extra =                     _extras(raw_algo, ALGO_KEYS) or {},
    )


    # Networks: key in YAML becomes the logical name (policy, value, q1, ...)
    raw_networks: Dict[str, Dict[str, Any]] = raw["networks"]
    if "dist_num_atoms" in raw_algo:
        for net_dict in raw_networks.values():
            net_args = net_dict.get("network_args")
            net_args["dist_num_atoms"] = raw_algo["dist_num_atoms"]
    network_cfgs: Dict[str, SingleNetworkConfig] = {}
    for net_name, net_dict in raw_networks.items():
        network_cfgs[net_name] = SingleNetworkConfig(
            name=net_dict.get("name", net_name),
            network_type=net_dict["network_type"],
            network_args=net_dict.get("network_args", {}),
        )
    networks = NetworksConfig(networks=network_cfgs)


    # --- Train params ---
    raw_tp: Dict[str, Any] = raw["train"]
    train_params = TrainParams(
        run_name =                          str(raw_tp["run_name"]),
        total_env_steps =                   int(raw_tp["total_env_steps"]),
        log_interval =                      int(raw_tp["log_interval"]),
        eval_interval =                     int(raw_tp["eval_interval"]),
        eval_envs =                         int(raw_tp["eval_envs"]),
        logging_method =                    [str(m) for m in raw_tp["logging_method"]],
        console_log_train =                 bool(raw_tp.get("console_log_train", True)),
        threshold_exit_training_on_eval_score = (
            None
            if raw_tp.get("threshold_exit_training_on_eval_score") is None
            else float(raw_tp.get("threshold_exit_training_on_eval_score"))
        ),
        wandb_project =                     str(raw_tp.get("wandb_project", TrainParams.wandb_project)),
        wandb_group =                       None if raw_tp.get("wandb_group") is None else str(raw_tp.get("wandb_group")),
        num_seeds =                         int(raw_tp.get("num_seeds", 1)),
        save_result =                       bool(raw_tp.get("save_result", False)),
        video_save_dir =                    str(raw_tp.get("video_save_dir", TrainParams.video_save_dir)),
        algo_save_dir =                     str(raw_tp.get("algo_save_dir", TrainParams.algo_save_dir)),
        save_video_at_end =                 bool(raw_tp.get("save_video_at_end", True)),
        save_algo_at_end =                  bool(raw_tp.get("save_algo_at_end", True)),
        extra =                             _extras(raw_tp, TRAIN_KEYS) or None,
    )

    
    # --- Sampler ---
    raw_sampler: Dict[str, Any] = raw["sampler"]
    sampler_extra = _extras(raw_sampler, SAMPLER_KEYS)
    sampler_extra["total_steps"] = int(train_params.total_env_steps)
    sampler_cfg = SamplerConfig(
        name =  str(raw_sampler["name"]),
        extra = sampler_extra,
    )


    # --- Inference ---
    raw_inference: Dict[str, Any] = raw.get("inference", {})
    inference_cfg = InferenceConfig(
        inference_only =    bool(raw_inference.get("inference_only", False)),
        algo_path =         (
            None
            if raw_inference.get("algo_path") is None
            else str(raw_inference.get("algo_path"))
        ),
        override_cfg =      bool(raw_inference.get("override_cfg", False)),
    )

    # ====================================
    # Manual adds
    # ====================================
    env_cfg.extra['gamma'] = algo_cfg.gamma                  # Add gamma to our env_cfg (used in reward norm.)

    # Assertions
    assert train_params.eval_envs % env_cfg.num_envs == 0    # Eval envs must be divisible by your num envs

    return TrainConfig(
        env=env_cfg,
        algo=algo_cfg,
        networks=networks,
        train=train_params,
        sampler=sampler_cfg,
        inference=inference_cfg,
    )