from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class PlotSpec:
	name: str
	filename: str
	output_prefix: str
	smooth_window: int | None = None
	y_label: str | None = None
	title: str | None = None
	line_alpha: float = 0.9
	fill_alpha: float = 0.12
	line_width: float = 2.0


def load_series(csv_path: Path) -> pd.DataFrame:
	"""Load step/mean/ci bounds."""
	df = pd.read_csv(csv_path)
	columns = set(df.columns)
	if {"mean", "ci_lower", "ci_upper"}.issubset(columns):
		return df[["step", "mean", "ci_lower", "ci_upper"]].copy()
	if {"seed0", "seed0_std"}.issubset(columns):
		mean = df["seed0"]
		std = df["seed0_std"]
		return pd.DataFrame(
			{
				"step": df["step"],
				"mean": mean,
				"ci_lower": mean - std,
				"ci_upper": mean + std,
			}
		)
	raise ValueError(f"Unsupported CSV structure in {csv_path} (columns: {sorted(columns)})")


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
	"""Smooth a 1D series with a moving average."""
	if window <= 1:
		return values
	kernel = np.ones(window, dtype=np.float64) / float(window)
	left = window // 2
	right = window - 1 - left
	padded = np.pad(values, (left, right), mode="edge")
	return np.convolve(padded, kernel, mode="valid")


def align_steps(series_list: list[pd.DataFrame]) -> tuple[np.ndarray, list[pd.DataFrame]]:
	"""Align steps across series, interpolating if needed."""
	steps = [df["step"].to_numpy() for df in series_list]
	if all(np.array_equal(steps[0], s) for s in steps[1:]):
		return steps[0], series_list

	common_steps = np.unique(np.concatenate(steps))
	aligned = []
	for df in series_list:
		x = df["step"].to_numpy()
		aligned.append(
			pd.DataFrame(
				{
					"step": common_steps,
					"mean": np.interp(common_steps, x, df["mean"].to_numpy()),
					"ci_lower": np.interp(common_steps, x, df["ci_lower"].to_numpy()),
					"ci_upper": np.interp(common_steps, x, df["ci_upper"].to_numpy()),
				}
			)
		)
	return common_steps, aligned


def plot_aggregates(
	base_dir: Path,
	subfolders: list[str],
	aggregation_name: str,
	plots: list[PlotSpec],
	output_dir: Path,
	smooth_window: int = 1,
) -> None:
	"""Aggregate and write plots to disk."""
	output_dir = output_dir / aggregation_name
	output_dir.mkdir(parents=True, exist_ok=True)

	for spec in plots:
		series_list = [load_series(base_dir / folder / spec.filename) for folder in subfolders]
		_, aligned = align_steps(series_list)
		window = smooth_window if spec.smooth_window is None else spec.smooth_window

		plt.figure(figsize=(8, 5))
		for df, label in zip(aligned, subfolders):
			steps = df["step"].to_numpy()
			mean = smooth_series(df["mean"].to_numpy(), window)
			ci_lower = smooth_series(df["ci_lower"].to_numpy(), window)
			ci_upper = smooth_series(df["ci_upper"].to_numpy(), window)

			plt.plot(steps, mean, label=label, alpha=spec.line_alpha, linewidth=spec.line_width)
			plt.fill_between(steps, ci_lower, ci_upper, alpha=spec.fill_alpha)

		plt.xlabel("step")
		if spec.y_label is not None:
			plt.ylabel(spec.y_label)
		if spec.title is not None:
			plt.title(spec.title)
		plt.legend()
		plt.tight_layout()

		output_path = output_dir / f"{spec.output_prefix}_{aggregation_name}.png"
		plt.savefig(output_path, dpi=200)
		plt.close()
