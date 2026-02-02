import yaml
import random
from typing import Any, Dict

from configs.config import (
    TrainConfig,
    EnvConfig,
    AlgoConfig,
    SingleNetworkConfig,
    NetworksConfig,
    TrainParams,
    SamplerConfig,
    InferenceConfig,
)


def load_yaml_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # Env
    raw_env: Dict[str, Any] = raw["env"]
    raw_env_args = raw_env.get("env_args", {})
    env_cfg = EnvConfig(
        name=str(raw_env["name"]),
        num_envs=int(raw_env["num_envs"]),
        max_episode_steps=None if raw_env.get("max_episode_steps") is None else int(raw_env.get("max_episode_steps")),
        is_atari=bool(raw_env.get("is_atari", False)),
        stack_stamples=int(raw_env.get("stack_stamples", 1)),
        env_args=dict(raw_env_args),
    )

    # Algo
    raw_algo: Dict[str, Any] = raw["algo"]
    seed_value = raw_algo.get("seed")
    if seed_value is None:
        seed_value = random.randint(0, 2**32 - 1)
    raw_algo["seed"] = seed_value
    algo_cfg = AlgoConfig(
        name=str(raw_algo["name"]),
        gamma=float(raw_algo["gamma"]),
        lr_start=float(raw_algo["lr_start"]),
        lr_end=float(raw_algo["lr_end"]),
        lr_warmup_env_steps=int(raw_algo["lr_warmup_env_steps"]),
        update_every_steps=int(raw_algo["update_every_steps"]),
        use_action_for_steps_train=int(raw_algo.get("use_action_for_steps_train", 1)),
        use_action_for_steps_eval=int(raw_algo.get("use_action_for_steps_eval", 1)),
        batch_size=int(raw_algo["batch_size"]),
        n_step=int(raw_algo.get("n_step", 1)),
        seed=int(raw_algo["seed"]),
        extra={
            k: v
            for k, v in raw_algo.items()
            if k not in {"name", "gamma", "lr", "lr_start", "lr_end", "lr_warmup_env_steps", "seed", "batch_size", "use_action_for_steps_train", "use_action_for_steps_eval", "n_step"}
        } or {},
    )

    # Networks: key in YAML becomes the logical name (policy, value, q1, ...)
    raw_networks: Dict[str, Dict[str, Any]] = raw["networks"]
    dist_num_atoms = raw_algo.get("dist_num_atoms")
    if dist_num_atoms is not None:
        for net_dict in raw_networks.values():
            net_args = net_dict.get("network_args")
            if net_args is None:
                net_args = {}
                net_dict["network_args"] = net_args
            net_args["dist_num_atoms"] = dist_num_atoms

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
        eval_envs=int(raw_tp["eval_envs"]),
        eval_interval=int(raw_tp["eval_interval"]),
        log_interval=int(raw_tp["log_interval"]),
        logging_method=[str(m) for m in raw_logging_method],
        console_log_train=bool(raw_tp.get("console_log_train", True)),
        wandb_project=str(raw_tp.get("wandb_project", TrainParams.wandb_project)),
        wandb_group=None if raw_tp.get("wandb_group") is None else str(raw_tp.get("wandb_group")),
        run_name=None if raw_tp.get("run_name") is None else str(raw_tp.get("run_name")),
        num_seeds=int(raw_tp.get("num_seeds", 1)),
        save_result=bool(raw_tp.get("save_result", False)),
        save_video_at_end=bool(raw_tp.get("save_video_at_end", True)),
        save_algo_at_end=bool(raw_tp.get("save_algo_at_end", True)),
        video_save_dir=str(raw_tp.get("video_save_dir", "saved_data/saved_videos")),
        algo_save_dir=str(raw_tp.get("algo_save_dir", "saved_data/saved_algos")),
        threshold_exit_training_on_eval_score=(None if raw_tp.get("threshold_exit_training_on_eval_score") is None else float(raw_tp.get("threshold_exit_training_on_eval_score"))),
        extra={
            k: v
            for k, v in raw_tp.items()
            if k not in {"total_env_steps", "eval_interval", "eval_envs", "log_interval", "logging_method", "console_log_train", "wandb_project", "wandb_group", "run_name", "num_seeds", "save_result", "save_video_at_end", "save_algo_at_end", "video_save_dir", "algo_save_dir", "threshold_exit_training_on_eval_score"}
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


    # Inference
    raw_inference: Dict[str, Any] = raw.get("inference", {})
    inference_cfg = InferenceConfig(
        inference_only=bool(raw_inference.get("inference_only", False)),
        override_cfg=bool(raw_inference.get("override_cfg", False)),
        algo_path=str(raw_inference.get("algo_path", "")),
    )

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