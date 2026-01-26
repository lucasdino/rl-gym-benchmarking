from dataclasses import dataclass

@dataclass
class RunResults:
    name: str
    value: float
    accumulator: str    # Should be 'mean', 'max', etc. For now we'll only work with 'mean' but may build this out more in the future

@dataclass
class ResultAccumulator:
    def __init__(self, run_results: RunResults):
        self.name = run_results.name
        self.value = run_results.value
        self.accumulator = run_results.accumulator
        self.accumulated_counter = 1

    def update(self, run_results: RunResults):
        if self.accumulator == "mean":
            self.value = (self.value * self.accumulated_counter + run_results.value) / (self.accumulated_counter + 1)
            self.accumulated_counter += 1
        elif self.accumulator == "max":
            self.value = max(self.value, run_results.value)

    def reset(self):
        self.value = 0
        self.accumulated_counter = 0


class ResultLogger():
    def __init__(self, name: str, logging_method: list[str], max_steps: int, wandb_run=None):
        self.name = name
        self.tracked_logs: dict[str, ResultAccumulator] | None = None
        self.logging_method = logging_method
        self.max_steps = max_steps
        self.wandb_run = wandb_run
        self._max_step_width = len(self._format_steps_short(self.max_steps)) if self.max_steps else 0
        
    def update(self, results: list[RunResults]):
        if results is None:
            return
        if self.tracked_logs is None:
            self.tracked_logs = {r.name: ResultAccumulator(r) for r in results}
            return

        for r in results:
            if r.name not in self.tracked_logs:
                self.tracked_logs[r.name] = ResultAccumulator(r)
            else:
                self.tracked_logs[r.name].update(r)
    
    def zero(self):
        if self.tracked_logs is None:
            return
        for acc in self.tracked_logs.values():
            acc.reset()

    def log(self, step: int, console_log=True):
        if self.tracked_logs is None:
            return

        log_dict = {name: acc.value for name, acc in self.tracked_logs.items()}
        if "console" in self.logging_method and console_log:
            parts = []
            parts.append(f"step={self._format_step(step)}")
            parts.extend([f"{k}={v:.4f}" for k, v in log_dict.items()])
            line = " | ".join([p for p in parts if p])
            line = f"{self.name} {line}" if line else self.name
            print(line)

        if "wandb" in self.logging_method and self.wandb_run is not None:
            wandb_prefix = self.name.strip()
            if wandb_prefix.startswith("[") and wandb_prefix.endswith("]"):
                wandb_prefix = wandb_prefix[1:-1]
            wandb_log_dict = {
                f"{wandb_prefix}/{k}" if wandb_prefix else k: v
                for k, v in log_dict.items()
            }
            self.wandb_run.log(wandb_log_dict, step=step)

    def _format_step(self, step: int) -> str:
        if not self.max_steps:
            return str(step)
        step_str = self._format_steps_short(step).rjust(self._max_step_width)
        return f"{step_str}/{self._format_steps_short(self.max_steps)}"

    @staticmethod
    def _format_steps_short(step: int) -> str:
        if step >= 1_000_000:
            return f"{step / 1_000_000:.1f}m"
        if step >= 1_000:
            return f"{step / 1_000:.1f}k"
        return str(step)