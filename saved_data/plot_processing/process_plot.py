from __future__ import annotations

from pathlib import Path

from process_helpers import PlotSpec, plot_aggregates


# =======================================
# Define these below.
# SUBFOLDERS: List out the subfolders you want to aggregate
# AGGREGATION_NAME: Name for the combined files
# =======================================
SUBFOLDERS = [
    "lunarlander_distrl",
    "lunarlander_distrl_boltzmann",
]

AGGREGATION_NAME = "rainbow_lunarlander_boltzmann_ablation"

# SUBFOLDERS = [
#     "cartpole_ddqn_noisynetablation_epsilongreedy",
#     "cartpole_ddqn_noisynetablation_noisynet_2p0",
#     "cartpole_ddqn_noisynetablation_noisynet",
#     "cartpole_ddqn_noisynetablation_noisynet_0p5",
#     "cartpole_ddqn_noisynetablation_noisynet_0p25",
# ]

# AGGREGATION_NAME = "noisynet_ablation"
# =======================================


if __name__ == "__main__":
	base_dir = Path("saved_data/saved_plots")
	output_dir = Path("saved_data/plot_processing/processed_plots")

	subfolders = SUBFOLDERS
	aggregation_name = AGGREGATION_NAME
	smooth_window = 1

	plots = [
		PlotSpec(
			name="eval_reward",
			filename="Eval_Reward_data.csv",
			output_prefix="eval_reward",
			smooth_window=1,
			y_label="Episode Reward",
			title="Evaluation Reward",
		),
		PlotSpec(
			name="loss",
			filename="Loss_data.csv",
			output_prefix="loss",
			smooth_window=20,
			y_label="Training Loss",
			title="Loss",
		),
		PlotSpec(
			name="value_estimates",
			filename="Value_Estimates_data.csv",
			output_prefix="value_estimates",
			smooth_window=10,
			y_label="Value",
			title="Avg. Value Estimates",
		),
	]

	plot_aggregates(
		base_dir=base_dir,
		subfolders=subfolders,
		aggregation_name=aggregation_name,
		plots=plots,
		output_dir=output_dir,
		smooth_window=smooth_window,
	)