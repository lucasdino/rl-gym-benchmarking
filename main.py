import os, time, shutil, random

import wandb
import numpy as np
from dotenv import load_dotenv

from configs.config_loader import load_yaml_config
from trainer.trainer import Trainer
from trainer.helper import start_plot_server, compute_and_save_aggregated_results, save_config_snapshot
from trainer.helper.plot_server import LIVE_PLOTS_DIR, CFG_GREEN, COLOR_RESET


# ========================================
# Update the following
# ========================================
START_PLOT_SERVER =         False        # If True, launches live plotting of the run
PLOT_SERVER_GRACE_SECONDS = 3           # Leave as is

# Can either point to a folder with .yamls or to specific .yaml files
configs = [
    "configs/discrete_actions/rainbow/lunarlander.yaml",       # Simple test!


    # "configs/template_video.yaml",
    # "configs/discrete_actions/rainbow/atari/breakout.yaml",
]
# ========================================


def main():
    os.makedirs(LIVE_PLOTS_DIR, exist_ok=True)
    if START_PLOT_SERVER:
        start_plot_server(open_browser=True)
    
    expanded_configs = []
    for c in configs:
        expanded_configs.extend([os.path.join(c, f) for f in os.listdir(c) if f.endswith(".yaml")]) if os.path.isdir(c) else expanded_configs.append(c)
    for config in expanded_configs:
        _clear_live_plots_dir()
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
            trainer.algo = trainer.algo.load(path=cfg.inference.algo_path, override_cfg=override_cfg)
            trainer.send_networks_to_device()
            trainer.eval(save_video=cfg.train.save_video_at_end)
            print(f"\nEval complete. Took {(time.time() - start_time):.1f}s.\n\n\n")

        else:
            num_seeds = cfg.train.num_seeds
            seeds = _generate_seeds(cfg.algo.seed, num_seeds)
            run_name = cfg.train.run_name if cfg.train.run_name else f"{cfg.env.name}_{cfg.algo.name}"
            run_info = {
                "config_path": config,
                "run_name": run_name,
                "num_seeds": num_seeds,
                "seeds": seeds,
                "per_seed": [],
            }
            print(f"{CFG_GREEN}{'='*120}\n# Training on Config: {config:<96} #\n# Seeds: {str(seeds):<109} #\n{'='*120}{COLOR_RESET}")
            
            for seed_idx, seed in enumerate(seeds):
                print(f"\n\n{CFG_GREEN}--- Seed {seed_idx + 1}/{num_seeds} (seed={seed}) ---{COLOR_RESET}\n")
                trainer = Trainer(cfg, seed=seed, seed_idx=seed_idx)
                seed_start = time.time()
                trainer.train()

                print(f"\n{CFG_GREEN}--- Final Evaluation + Saving (Seed #{seed_idx+1}/{num_seeds}) ---{COLOR_RESET}")
                trainer.eval(save_video=cfg.train.save_video_at_end and seed_idx == num_seeds - 1)
                if cfg.train.save_algo_at_end:
                    trainer.save_algo()
                
                if "wandb" in cfg.train.logging_method and trainer.wandb_run is not None:
                    trainer.wandb_run.finish()
                
                trainer.close()
                seed_seconds = time.time() - seed_start

                run_info["per_seed"].append({
                    "seed_idx": seed_idx,
                    "seed": seed,
                    "total_seconds": seed_seconds,
                })
            
            run_info["total_seconds"] = time.time() - start_time

            if cfg.train.save_result and num_seeds >= 1:
                compute_and_save_aggregated_results(run_name, num_seeds)
                save_config_snapshot(run_name, cfg, run_info)
                if START_PLOT_SERVER and PLOT_SERVER_GRACE_SECONDS > 0:
                    print(f"\n[PlotServer] Waiting {PLOT_SERVER_GRACE_SECONDS}s for live update...\n")
                    time.sleep(PLOT_SERVER_GRACE_SECONDS)
            
            print(f"\nTraining complete. Took {(time.time() - start_time):.1f}s.\n\n\n")


# ------------------
# Helpers
# ------------------
def _generate_seeds(base_seed: int | None, num_seeds: int) -> list[int]:
    """
    Generate list of seeds. If base_seed is set, seeds are deterministic.
    return
    """
    if base_seed is not None:
        random.seed(base_seed)
        np.random.seed(base_seed)
    return [random.randint(0, 2**31 - 1) for _ in range(num_seeds)]


def _clear_live_plots_dir() -> None:
    """
    Clear live plots directory.
    return
    """
    if not os.path.exists(LIVE_PLOTS_DIR):
        os.makedirs(LIVE_PLOTS_DIR, exist_ok=True)
        return
    for entry in os.listdir(LIVE_PLOTS_DIR):
        if entry == "index.html":
            continue
        path = os.path.join(LIVE_PLOTS_DIR, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)



if __name__ == "__main__":
    main()