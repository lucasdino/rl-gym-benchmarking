import os, time, shutil, random

import wandb
import numpy as np
from dotenv import load_dotenv

from configs.config_loader import load_yaml_config
from trainer.trainer import Trainer
from trainer.helper import start_plot_server, compute_and_save_aggregated_results
from trainer.helper.plot_server import LIVE_PLOTS_DIR, CFG_GREEN, COLOR_RESET



configs = [
    "configs/discrete_actions/ddqn/per_ablation",
    "configs/discrete_actions/ddqn/n_step_ablation",
    "configs/discrete_actions/ddqn/noisynet_ablation",
    # "configs/discrete_actions/ddqn/breakout.yaml",
    # "configs/discrete_actions/ddqn/lunarlander.yaml",
    # "configs/discrete_actions/ddqn/cartpole.yaml",
]

START_PLOT_SERVER = True
PLOT_SERVER_GRACE_SECONDS = 5


def generate_seeds(base_seed: int | None, num_seeds: int) -> list[int]:
    """
    Generate list of seeds. If base_seed is set, seeds are deterministic.
    return
    """
    if base_seed is not None:
        random.seed(base_seed)
        np.random.seed(base_seed)
    return [random.randint(0, 2**31 - 1) for _ in range(num_seeds)]


def main():
    if os.path.exists(LIVE_PLOTS_DIR):
        for entry in os.listdir(LIVE_PLOTS_DIR):
            if entry == "index.html":
                continue
            path = os.path.join(LIVE_PLOTS_DIR, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    else:
        os.makedirs(LIVE_PLOTS_DIR, exist_ok=True)
    if START_PLOT_SERVER:
        start_plot_server(open_browser=True)
    
    expanded_configs = []
    for c in configs:
        expanded_configs.extend([os.path.join(c, f) for f in os.listdir(c) if f.endswith(".yaml")]) if os.path.isdir(c) else expanded_configs.append(c)
    for config in expanded_configs:
        start_time = time.time()
        cfg = load_yaml_config(config)

        if "wandb" in cfg.train.logging_method:
            load_dotenv()
            os.environ["WANDB_MODE"] = "online"
            wandb.login(key=os.environ["WANDB_API_KEY"])

        if cfg.inference.inference_only:
            print(f"{CFG_GREEN}{'='*120}\n# Evaluating on Config: {config:<94} #\n{'='*120}{COLOR_RESET}\n")
            trainer = Trainer(cfg)
            override_cfg = cfg if cfg.inference.override_cfg else None
            trainer.algo = trainer.algo.load(path = cfg.inference.algo_path, override_cfg = override_cfg)
            trainer.send_networks_to_device()
            trainer.eval(save_video=cfg.train.save_video_at_end)
            print(f"\nEval complete. Took {(time.time() - start_time):.1f}s.\n\n\n")

        else:
            num_seeds = cfg.train.num_seeds
            seeds = generate_seeds(cfg.algo.seed, num_seeds)
            run_name = cfg.train.run_name if cfg.train.run_name else f"{cfg.env.name}_{cfg.algo.name}"
            
            print(f"{CFG_GREEN}{'='*120}\n# Training on Config: {config:<96} #\n# Seeds: {seeds} #\n{'='*120}{COLOR_RESET}\n")
            
            for seed_idx, seed in enumerate(seeds):
                print(f"\n--- Seed {seed_idx + 1}/{num_seeds} (seed={seed}) ---\n")
                trainer = Trainer(cfg, seed=seed, seed_idx=seed_idx)
                trainer.train()

                print(f"\n--- Final Evaluation + Saving (Seed {seed_idx}) ---")
                trainer.eval(save_video=cfg.train.save_video_at_end and seed_idx == num_seeds - 1)
                if cfg.train.save_algo_at_end:
                    trainer.save_algo()
                
                if "wandb" in cfg.train.logging_method and trainer.wandb_run is not None:
                    trainer.wandb_run.finish()
                
                trainer.close()
            
            if cfg.train.save_result and num_seeds >= 1:
                compute_and_save_aggregated_results(run_name, num_seeds)
                if START_PLOT_SERVER and PLOT_SERVER_GRACE_SECONDS > 0:
                    print(f"\n[PlotServer] Waiting {PLOT_SERVER_GRACE_SECONDS}s for live update...\n")
                    time.sleep(PLOT_SERVER_GRACE_SECONDS)
            
            print(f"\nTraining complete. Took {(time.time() - start_time):.1f}s.\n\n\n")

if __name__ == "__main__":
    main()