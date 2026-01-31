import os
from typing import Any
from dataclasses import dataclass

import torch

from trainer.helper.plot_server import update_manifest, LIVE_PLOTS_DIR
from trainer.helper.charting import (
    save_histogram,
    save_stacked_histogram,
    save_line_plot,
)

@dataclass
class RunResults:
    name: str
    value: Any
    accumulator: str    # Should be 'mean', 'max', 'concat', 'dict_concat', 'accumulating_writes', 'batched_mean'
    category: str = "other"
    smoothing: bool = False
    smoothing_rate: float = 0.9
    show_ci: bool = False  # If True, value should be (step, mean, std)
    publish_to_wandb: bool = False
    value_format: str = "raw"  # 'raw' = single value, 'mean_std' = (mean, std) tuple to be combined
    write_to_file: bool = False  # If True for batched_mean, write combined result to CSV on flush
    aggregation_steps: int | None = None  # If set, only write to CSV every N updates (plot still at log_interval)

@dataclass
class ResultAccumulator:
    def __init__(self, run_results: RunResults):
        self.name = run_results.name
        if run_results.accumulator == "concat":
            self.value = list(run_results.value)
        elif run_results.accumulator == "dict_concat":
            self.value = {}
            self._extend_dict_concat(self.value, run_results.value)
        elif run_results.accumulator == "accumulating_writes":
            self.value = self._normalize_accumulating_values(run_results.value)
        elif run_results.accumulator == "batched_mean":
            self.value = self._init_batched_mean(run_results.value, run_results.value_format)
        else:
            self.value = run_results.value
        self.accumulator = run_results.accumulator
        self.category = run_results.category
        self.smoothing = run_results.smoothing
        self.smoothing_rate = run_results.smoothing_rate
        self.show_ci = run_results.show_ci
        self.publish_to_wandb = run_results.publish_to_wandb
        self.value_format = run_results.value_format
        self.write_to_file = run_results.write_to_file
        self.aggregation_steps = run_results.aggregation_steps
        self.persistent = run_results.accumulator == "accumulating_writes" or (run_results.accumulator == "batched_mean" and run_results.write_to_file)
        self.accumulated_counter = 1
        self.aggregation_counter = 1  # Counts updates for aggregation_steps threshold

    def update(self, run_results: RunResults):
        if self.accumulator == "mean":
            self.value = (self.value * self.accumulated_counter + run_results.value) / (self.accumulated_counter + 1)
            self.accumulated_counter += 1
        elif self.accumulator == "max":
            self.value = max(self.value, run_results.value)
        elif self.accumulator == "concat":
            self.value.extend(list(run_results.value))
        elif self.accumulator == "dict_concat":
            self._extend_dict_concat(self.value, run_results.value)
        elif self.accumulator == "accumulating_writes":
            self.value.extend(self._normalize_accumulating_values(run_results.value))
        elif self.accumulator == "batched_mean":
            self._update_batched_mean(run_results.value, run_results.value_format)
            self.aggregation_counter += 1

    def _init_batched_mean(self, value: Any, value_format: str) -> dict:
        """Initialize batched mean tracking with sum, sum_sq, count for proper combination."""
        if value_format == "mean_std":
            mean, std = float(value[0]), float(value[1])
            return {"sum": mean, "sum_sq": mean**2 + std**2, "count": 1}
        else:
            v = float(value)
            return {"sum": v, "sum_sq": v**2, "count": 1}

    def _update_batched_mean(self, value: Any, value_format: str) -> None:
        """Update batched mean with new value, handling both raw and mean_std formats."""
        if value_format == "mean_std":
            mean, std = float(value[0]), float(value[1])
            self.value["sum"] += mean
            self.value["sum_sq"] += mean**2 + std**2
            self.value["count"] += 1
        else:
            v = float(value)
            self.value["sum"] += v
            self.value["sum_sq"] += v**2
            self.value["count"] += 1

    def get_batched_mean_result(self) -> tuple[float, float]:
        """Returns (mean, std) from batched accumulation."""
        if self.value["count"] == 0:
            return 0.0, 0.0
        mean = self.value["sum"] / self.value["count"]
        variance = (self.value["sum_sq"] / self.value["count"]) - mean**2
        std = max(0.0, variance) ** 0.5
        return mean, std

    def reset(self):
        if self.accumulator == "concat":
            self.value = []
        elif self.accumulator == "dict_concat":
            self.value = {k: [] for k in self.value.keys()}
        elif self.accumulator == "accumulating_writes":
            self.value = []
        elif self.accumulator == "batched_mean":
            self.value = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
        else:
            self.value = 0
        self.accumulated_counter = 0
        self.aggregation_counter = 0

    @staticmethod
    def _extend_dict_concat(target: dict[str, list[float]], new_values: dict[str, Any]) -> None:
        for key, value in new_values.items():
            if key not in target:
                target[key] = []
            if isinstance(value, (list, tuple)):
                target[key].extend(list(value))
            else:
                target[key].append(value)

    @staticmethod
    def _normalize_accumulating_values(value: Any) -> list[Any]:
        if isinstance(value, torch.Tensor):
            if value.ndim == 1 and value.numel() in (2, 3):
                return [value]
            if value.ndim == 2 and value.shape[1] in (2, 3):
                return list(value)
            return [value]
        if isinstance(value, (list, tuple)):
            if len(value) in (2, 3) and not any(isinstance(v, (list, tuple, torch.Tensor)) for v in value):
                return [value]
            return list(value)
        return [value]


class ResultLogger():
    def __init__(self, name: str, logging_method: list[str], max_steps: int, wandb_run=None, *, dict_concat_stacked: bool = True, seed_idx: int = 0):
        self.name = name
        self.tracked_logs: dict[str, ResultAccumulator] | None = None
        self.logging_method = logging_method
        self.max_steps = max_steps
        self.wandb_run = wandb_run
        self.dict_concat_stacked = dict_concat_stacked
        self.seed_idx = seed_idx
        self._max_step_width = len(self._format_steps_short(self.max_steps)) if self.max_steps else 0
        self._plot_registry: dict[str, list[str]] = {}  # category -> list of relative paths
        self._update_counter: dict[str, int] = {}  # name -> running counter for accumulated writes
        self._wandb_history: dict[str, list[tuple]] = {}  # name -> [(step, mean, lower, upper), ...]
        self._pending_csv_writes: set[str] = set()  # names that have new CSV data needing re-plot
        
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
    
    def zero(self, *, force: bool = False):
        if self.tracked_logs is None:
            return
        for acc in self.tracked_logs.values():
            if not force and acc.accumulator == "batched_mean" and acc.write_to_file:
                continue
            acc.reset()

    def flush(self, step: int, force: bool = False):
        """Flush batched_mean with write_to_file to CSV when aggregation threshold met (or force=True)."""
        if self.tracked_logs is None:
            return
        
        for name, acc in self.tracked_logs.items():
            if acc.accumulator != "batched_mean" or not acc.write_to_file or acc.value["count"] == 0:
                continue
            
            # Check aggregation threshold (skip if not met, unless force)
            if not force and acc.aggregation_steps is not None and acc.aggregation_counter < acc.aggregation_steps:
                continue
            
            mean, std = acc.get_batched_mean_result()
            self._append_accumulated_data(name, acc.category, [(step, mean, std)], show_ci=True)
            self._pending_csv_writes.add(name)
            acc.reset()

    def log(self, step: int, console_log=True):
        if self.tracked_logs is None:
            return

        log_dict = {}
        for name, acc in self.tracked_logs.items():
            if acc.accumulator == "batched_mean":
                mean, std = acc.get_batched_mean_result()
                log_dict[name] = mean
            elif acc.accumulator not in ("concat", "dict_concat", "accumulating_writes"):
                log_dict[name] = acc.value
        
        if "console" in self.logging_method and console_log:
            parts = []
            parts.append(f"Step={self._format_step(step)}")
            parts.extend([f"{k}={v:<6.2f}" if "Learning Rate" not in k else f"{k}={v:<6.2e}" for k, v in log_dict.items()])
            line = " | ".join([p for p in parts if p])
            line = f"{'[' + self.name + ']':<7} {line}" if line else self.name
            
            # Color: blue for train, red for eval
            if "train" in self.name.lower():
                print(f"\033[94m{line}\033[0m")
            elif "eval" in self.name.lower():
                print(f"\033[91m{line}\033[0m")
            else:
                print(line)

        new_plots = []
        persistent_plots = []
        wandb_log_dict = {}
        for name, acc in self.tracked_logs.items():
            if acc.accumulator == "concat" and acc.value:
                plot_info = save_histogram(name, acc.category, acc.value, step)
                new_plots.append(plot_info)
            if acc.accumulator == "dict_concat":
                if self.dict_concat_stacked:
                    filename = save_stacked_histogram(name, acc.category, acc.value, step)
                    if filename is not None:
                        new_plots.append(filename)
                else:
                    for key, values in acc.value.items():
                        if values:
                            new_plots.append(save_histogram(f"{name}/{key}", acc.category, values, step))
            if acc.accumulator == "accumulating_writes" and acc.value:
                self._append_accumulated_data(name, acc.category, acc.value, acc.show_ci)
                filename = save_line_plot(name, acc.category, acc.smoothing, acc.smoothing_rate, acc.show_ci)
                if filename is not None:
                    new_plots.append(filename)
                    if acc.persistent:
                        persistent_plots.append(filename)
                acc.reset()
            if acc.publish_to_wandb and acc.accumulator not in ("concat", "dict_concat", "accumulating_writes"):
                wandb_log_dict[name] = acc.value
        
        # Plot metrics that had CSV writes from flush()
        for name in self._pending_csv_writes:
            if name not in self.tracked_logs:
                continue
            acc = self.tracked_logs[name]
            filename = save_line_plot(name, acc.category, acc.smoothing, acc.smoothing_rate, show_ci=True)
            if filename is not None:
                new_plots.append(filename)
                if acc.persistent:
                    persistent_plots.append(filename)
        self._pending_csv_writes.clear()

        if new_plots:
            self._update_plot_registry(new_plots, persistent_plots if persistent_plots else None)

        if "wandb" in self.logging_method and self.wandb_run is not None:
            for name, history in self._wandb_history.items():
                if not history:
                    continue
                for s, m, lo, hi in history:
                    self.wandb_run.log({
                        f"{name}/mean": m,
                        f"{name}/lower": lo,
                        f"{name}/upper": hi,
                    }, step=s)
                history.clear()
            if wandb_log_dict:
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

    def _append_accumulated_data(self, name: str, category: str, values: list[Any], show_ci: bool) -> None:
        save_dir = os.path.join(LIVE_PLOTS_DIR, category)
        os.makedirs(save_dir, exist_ok=True)
        safe_name = name.replace(" ", "_").replace("/", "_")
        file_path = os.path.join(save_dir, f"{safe_name}_data.csv")
        
        seed_col = f"seed{self.seed_idx}"
        std_col = f"seed{self.seed_idx}_std"
        
        existing_data = {}
        existing_cols = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lines = f.read().strip().splitlines()
            if lines:
                existing_cols = lines[0].split(",")
                for line in lines[1:]:
                    parts = line.split(",")
                    step = int(parts[0])
                    existing_data[step] = {existing_cols[i]: parts[i] for i in range(len(parts))}
        
        if name not in self._update_counter:
            self._update_counter[name] = 0
        
        for value in values:
            parsed = self._parse_step_value(name, value, show_ci)
            step = int(parsed[0])
            if step not in existing_data:
                existing_data[step] = {"step": str(step)}
            existing_data[step][seed_col] = str(parsed[1])
            if show_ci:
                existing_data[step][std_col] = str(parsed[2])
        
        all_cols = ["step"]
        for col in existing_cols[1:]:
            if col not in all_cols:
                all_cols.append(col)
        if seed_col not in all_cols:
            all_cols.append(seed_col)
        if show_ci and std_col not in all_cols:
            all_cols.append(std_col)
        
        with open(file_path, "w") as f:
            f.write(",".join(all_cols) + "\n")
            for step in sorted(existing_data.keys()):
                row = [existing_data[step].get(col, "") for col in all_cols]
                f.write(",".join(row) + "\n")

    def _parse_step_value(self, name: str, value: Any, show_ci: bool = False) -> tuple:
        """Returns (step, val) or (step, val, std) depending on show_ci."""
        if isinstance(value, torch.Tensor):
            if value.ndim == 1 and value.numel() == 3 and show_ci:
                return int(value[0].item()), float(value[1].item()), float(value[2].item())
            if value.ndim == 1 and value.numel() == 2:
                return (int(value[0].item()), float(value[1].item()), 0.0) if show_ci else (int(value[0].item()), float(value[1].item()))
        if isinstance(value, (list, tuple)):
            if len(value) == 3 and show_ci:
                return int(value[0]), float(value[1]), float(value[2])
            if len(value) == 2:
                return (int(value[0]), float(value[1]), 0.0) if show_ci else (int(value[0]), float(value[1]))

        if name not in self._update_counter:
            self._update_counter[name] = 0
        step = self._update_counter[name]
        self._update_counter[name] += 1
        return (step, float(value), 0.0) if show_ci else (step, float(value))

    def _update_plot_registry(self, new_plots: list[tuple[str, str]], persistent_plots: list[tuple[str, str]] | None = None) -> None:
        persistent_categories = set()
        for category, filepath in new_plots:
            scoped_category = f"{self.name.lower()}_{category}"
            if scoped_category not in self._plot_registry:
                self._plot_registry[scoped_category] = []
            existing = self._plot_registry[scoped_category]
            base_name = filepath.rsplit("_", 1)[0]
            self._plot_registry[scoped_category] = [p for p in existing if not p.startswith(base_name)]
            self._plot_registry[scoped_category].append(filepath)
        
        if persistent_plots:
            for category, _ in persistent_plots:
                persistent_categories.add(f"{self.name.lower()}_{category}")
        
        update_manifest(self._plot_registry, persistent_categories if persistent_categories else None)