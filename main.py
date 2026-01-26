import time

from configs.config_loader import load_yaml_config
from trainer.trainer import Trainer



configs = [
    "configs/ddqn/cartpole.yaml",
]



def main():
    for config in configs:
        start_time = time.time()
        print(f"---------------------------\nTraining on Config: {config}\n---------------------------\n")
        cfg = load_yaml_config(config)
        trainer = Trainer(cfg)
        trainer.train()
        print(f"Training complete. Took {(time.time() - start_time):.1f}s.\n\n")

if __name__ == "__main__":
    main()