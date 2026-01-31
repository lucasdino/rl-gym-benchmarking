"""
Charting helpers for live plots and aggregated results.
return
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LIVE_PLOTS_DIR = os.path.join("saved_data", "plots", "live_plots")


def ema_smooth(values: list[float], alpha: float = 0.9) -> list[float]:
    """Exponential moving average smoothing."""
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
    return smoothed


def save_histogram(name: str, category: str, values: list[float], step: int) -> tuple[str, str]:
    """Returns (category, relative_filepath)."""
    save_dir = os.path.join(LIVE_PLOTS_DIR, category)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(5, 3))
    plt.hist(values, bins=30, rwidth=0.9)
    plt.ylabel("Count", fontsize=9)
    plt.title(f"{name} Histogram [Step {step:,}]", fontsize=10)
    plt.tight_layout()
    
    safe_name = name.replace(" ", "_").replace("/", "_")
    filename = f"{safe_name}_{step}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=80)
    plt.close()
    return (category, f"{category}/{filename}")


def save_stacked_histogram(name: str, category: str, values: dict[str, list[float]], step: int) -> tuple[str, str] | None:
    """Returns (category, relative_filepath) or None."""
    keys = [k for k, v in values.items() if v]
    if not keys:
        return None
    data = [values[k] for k in keys]
    
    save_dir = os.path.join(LIVE_PLOTS_DIR, category)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(5, 3))
    plt.hist(data, bins=30, stacked=True, label=keys, rwidth=0.9)
    plt.legend(fontsize=7)
    plt.ylabel("Count", fontsize=9)
    plt.title(f"{name} Histogram [Step {step:,}]", fontsize=10)
    plt.tight_layout()
    
    safe_name = name.replace(" ", "_").replace("/", "_")
    filename = f"{safe_name}_{step}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=80)
    plt.close()
    return (category, f"{category}/{filename}")


def save_line_plot(name: str, category: str, smoothing: bool = True, smoothing_rate: float = 0.9, show_ci: bool = False) -> tuple[str, str] | None:
    """Returns (category, relative_filepath) or None."""
    data_dir = os.path.join(LIVE_PLOTS_DIR, category)
    safe_name = name.replace(" ", "_").replace("/", "_")
    file_path = os.path.join(data_dir, f"{safe_name}_data.csv")
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()
    if len(lines) <= 1:
        return None

    cols = lines[0].split(",")
    seed_cols = [c for c in cols if c.startswith("seed") and not c.endswith("_std")]
    
    data_by_seed: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for seed_col in seed_cols:
        std_col = f"{seed_col}_std"
        has_std = std_col in cols
        steps, values, stds = [], [], []
        for line in lines[1:]:
            parts = line.split(",")
            row = {cols[i]: parts[i] for i in range(len(parts))}
            if row.get(seed_col, ""):
                steps.append(int(row["step"]))
                values.append(float(row[seed_col]))
                if has_std and row.get(std_col, ""):
                    stds.append(float(row[std_col]))
        data_by_seed[seed_col] = (steps, values, stds if has_std else [])

    save_dir = os.path.join(LIVE_PLOTS_DIR, category)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(5, 3))
    colors = plt.cm.tab10.colors
    for i, (seed_col, (steps, values, stds)) in enumerate(data_by_seed.items()):
        if not steps:
            continue
        color = colors[i % len(colors)]
        if smoothing:
            smoothed = ema_smooth(values, smoothing_rate)
            plt.plot(steps, values, alpha=0.2, color=color, linewidth=0.5)
            plt.plot(steps, smoothed, color=color, linewidth=1.5, label=seed_col)
            if stds:
                smoothed_std = ema_smooth(stds, smoothing_rate)
                upper = [m + s for m, s in zip(smoothed, smoothed_std)]
                lower = [m - s for m, s in zip(smoothed, smoothed_std)]
                plt.fill_between(steps, lower, upper, alpha=0.15, color=color)
        else:
            plt.plot(steps, values, color=color, linewidth=1.0, label=seed_col)
            if stds:
                upper = [m + s for m, s in zip(values, stds)]
                lower = [m - s for m, s in zip(values, stds)]
                plt.fill_between(steps, lower, upper, alpha=0.15, color=color)
    
    if len(data_by_seed) > 1:
        plt.legend(fontsize=7)
    plt.xlabel("Environment Step", fontsize=9)
    plt.title(f"{name}", fontsize=10)
    plt.tight_layout()
    
    filename = f"{safe_name}_line.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=80)
    plt.close()
    return (category, f"{category}/{filename}")


def compute_and_save_aggregated_results(run_name: str, num_seeds: int) -> None:
    """
    Compute mean/CI across seeds (with interpolation for misaligned steps) and save to saved_data/saved_plots/{run_name}.
    Also updates live_plots manifest to show aggregated plots.
    return
    """
    import shutil
    import scipy.stats as stats
    from trainer.helper.plot_server import update_manifest
    
    save_dir = os.path.join("saved_data", "saved_plots", run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    aggregated_plots: dict[str, list[str]] = {}
    
    for category in os.listdir(LIVE_PLOTS_DIR):
        category_path = os.path.join(LIVE_PLOTS_DIR, category)
        if not os.path.isdir(category_path):
            continue
        
        for fname in os.listdir(category_path):
            if not fname.endswith("_data.csv"):
                continue
            
            file_path = os.path.join(category_path, fname)
            with open(file_path, "r") as f:
                lines = f.read().strip().splitlines()
            if len(lines) <= 1:
                continue
            
            cols = lines[0].split(",")
            seed_cols = [c for c in cols if c.startswith("seed") and not c.endswith("_std")]
            if len(seed_cols) < 2:
                shutil.copy(file_path, os.path.join(save_dir, fname))
                continue
            
            # Parse data per seed: {seed_col: (steps, values)}
            seed_data: dict[str, tuple[list[int], list[float]]] = {}
            for seed_col in seed_cols:
                steps, values = [], []
                for line in lines[1:]:
                    parts = line.split(",")
                    row = {cols[i]: parts[i] for i in range(len(parts))}
                    if row.get(seed_col, ""):
                        steps.append(int(row["step"]))
                        values.append(float(row[seed_col]))
                if steps:
                    seed_data[seed_col] = (steps, values)
            
            if len(seed_data) < 2:
                shutil.copy(file_path, os.path.join(save_dir, fname))
                continue
            
            # Check if steps align across all seeds
            step_sets = [set(steps) for steps, _ in seed_data.values()]
            common_exact_steps = step_sets[0].intersection(*step_sets[1:])
            
            all_seed_steps = [steps for steps, _ in seed_data.values()]
            avg_num_samples = int(np.mean([len(s) for s in all_seed_steps]))
            
            if len(common_exact_steps) >= avg_num_samples * 0.8:
                common_steps = sorted(common_exact_steps)
                interp_values: dict[str, list[float]] = {}
                for seed_col, (steps, values) in seed_data.items():
                    step_to_val = dict(zip(steps, values))
                    interp_values[seed_col] = [step_to_val[s] for s in common_steps]
            else:
                min_step = max(min(steps) for steps, _ in seed_data.values())
                max_step = min(max(steps) for steps, _ in seed_data.values())
                num_interp = max(10, avg_num_samples)
                common_steps = np.linspace(min_step, max_step, num_interp, dtype=int).tolist()
                common_steps = sorted(set(common_steps))
                
                interp_values = {}
                for seed_col, (steps, values) in seed_data.items():
                    interp_values[seed_col] = list(np.interp(common_steps, steps, values))
            
            # Compute mean and CI at each common step
            new_cols = ["step"] + seed_cols + ["mean", "ci_lower", "ci_upper"]
            new_lines = [",".join(new_cols)]
            
            for i, step in enumerate(common_steps):
                vals = [interp_values[sc][i] for sc in seed_cols]
                mean = np.mean(vals)
                sem = stats.sem(vals)
                ci = sem * stats.t.ppf(0.975, len(vals) - 1) if len(vals) > 1 else 0
                
                row_parts = [str(step)]
                for sc in seed_cols:
                    row_parts.append(str(interp_values[sc][i]))
                row_parts.extend([str(mean), str(mean - ci), str(mean + ci)])
                new_lines.append(",".join(row_parts))
            
            # Save to permanent location
            out_path = os.path.join(save_dir, fname)
            with open(out_path, "w") as f:
                f.write("\n".join(new_lines))
            
            # Save to live_plots for display
            live_out_dir = os.path.join(LIVE_PLOTS_DIR, category)
            os.makedirs(live_out_dir, exist_ok=True)
            live_out_path = os.path.join(live_out_dir, fname)
            with open(live_out_path, "w") as f:
                f.write("\n".join(new_lines))
            
            # Generate aggregated plot directly in live_plots
            plot_name = fname.replace("_data.csv", "")
            live_plot_name = f"{plot_name}_aggregated.png"
            live_plot_path = os.path.join(live_out_dir, live_plot_name)
            
            # Generate plot directly to live_plots
            _save_aggregated_plot_to_path(plot_name, live_out_path, live_plot_path)
            
            # Also save to permanent location
            perm_plot_path = os.path.join(save_dir, live_plot_name)
            shutil.copy(live_plot_path, perm_plot_path)
            
            cat_key = f"aggregated_{category}"
            if cat_key not in aggregated_plots:
                aggregated_plots[cat_key] = []
            aggregated_plots[cat_key].append(f"{category}/{live_plot_name}")
    
    if aggregated_plots:
        update_manifest(aggregated_plots, persistent_categories=set(aggregated_plots.keys()))
        print(f"Updated manifest with aggregated plots: {list(aggregated_plots.keys())}")
    
    print(f"Aggregated results saved to: {save_dir}")


def _save_aggregated_plot_to_path(name: str, data_path: str, out_path: str) -> None:
    """Generate aggregated plot and save directly to specified path."""
    with open(data_path, "r") as f:
        lines = f.read().strip().splitlines()
    if len(lines) <= 1:
        return
    
    cols = lines[0].split(",")
    if "mean" not in cols:
        return
    
    steps, means, ci_lowers, ci_uppers = [], [], [], []
    seed_cols = [c for c in cols if c.startswith("seed") and not c.endswith("_std")]
    seed_data: dict[str, tuple[list[int], list[float]]] = {sc: ([], []) for sc in seed_cols}
    
    for line in lines[1:]:
        parts = line.split(",")
        row = {cols[i]: parts[i] for i in range(len(parts))}
        if row.get("mean", ""):
            steps.append(int(row["step"]))
            means.append(float(row["mean"]))
            ci_lowers.append(float(row["ci_lower"]))
            ci_uppers.append(float(row["ci_upper"]))
        for sc in seed_cols:
            if row.get(sc, ""):
                seed_data[sc][0].append(int(row["step"]))
                seed_data[sc][1].append(float(row[sc]))
    
    plt.figure(figsize=(6, 4))
    colors = plt.cm.tab10.colors
    for i, (sc, (s_steps, s_vals)) in enumerate(seed_data.items()):
        if s_steps:
            plt.plot(s_steps, s_vals, alpha=0.3, color=colors[i % len(colors)], linewidth=0.8, label=sc)
    
    if steps:
        plt.plot(steps, means, color='#6A0DAD', linewidth=2.5, label='Mean')
        plt.fill_between(steps, ci_lowers, ci_uppers, alpha=0.3, color='#9B59B6', label='95% CI')
    
    plt.xlabel("Environment Step", fontsize=10)
    plt.ylabel("Value", fontsize=10)
    plt.title(name.replace("_", " "), fontsize=11)
    plt.legend(fontsize=8)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=100)
    plt.close()