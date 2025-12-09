import matplotlib.pyplot as plt
import pathlib
import os
import json
import seaborn as sns
import pandas as pd
import numpy as np
from copy import deepcopy
from typing import Optional


def plot_sft_eval_metrics(
    score_dict: dict, metrics: list[str], model_labels: list[str]
):
    # Calculate number of rows needed with 3 subplots per row
    num_rows = (len(metrics) + 2) // 3  # Round up division
    num_cols = 3

    # Set up the plot with subplots arranged in a grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows))
    axes = axes.flatten()  # Flatten to make indexing easier

    # Plot each metric in its own subplot
    for i, metric in enumerate(metrics):
        values = [score_dict[model][metric] for model in score_dict.keys()]
        axes[i].bar(range(len(score_dict.keys())), values, color="forestgreen")
        axes[i].set_title(f"{metric}")
        axes[i].set_ylabel("Score")
        axes[i].set_xticks(range(len(score_dict.keys())))
        axes[i].set_xticklabels(model_labels, rotation=45)

    # Hide any empty subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()


def plot_personality_evals_barplot(
    setting_to_res: dict,
    title: str,
    show_legend: bool = True,
    color_map: dict | None = None,
    evals_to_plot: list[str] = [
        "Instruction Following",
        "HarmBench Score",
        "Overrefusal",
        "MMLU",
        "BBQ",
        "SimpleQA",
        "GSM8K",
        "GPQA",
        "TruthfulQA",
    ],
):
    # Filter metrics_config to only include requested evals
    metrics_config = {
        "Instruction Following": (
            "personality_evals",
            "instruction_following",
            "metrics",
            "instruction_following_score",
        ),
        "HarmBench Score": (
            "personality_evals",
            "harmfulness_harmbench",
            "metrics",
            "harm_score",
        ),
        "StrongReject Harmfulness": (
            "personality_evals",
            "harmfulness_strongreject",
            "metrics",
            "harm_score",
        ),
        "Overrefusal": ("personality_evals", "overrefusal", "metrics", "refusal_rate"),
        "MMLU": ("personality_evals", "mmlu", "metrics", "accuracy"),
        "BBQ": ("personality_evals", "bbq", "metrics", "accuracy"),
        "SimpleQA": (
            "personality_evals",
            "simpleqa",
            "metrics",
            "accuracy_given_attempted",
        ),
        "GSM8K": ("personality_evals", "gsm8k", "metrics", "accuracy"),
        "GPQA": ("personality_evals", "gpqa", "metrics", "accuracy"),
        "TruthfulQA": ("personality_evals", "truthfulqa", "metrics", "accuracy"),
    }

    # Filter metrics_config to only include requested evals
    metrics_config = {k: v for k, v in metrics_config.items() if k in evals_to_plot}

    # Calculate optimal grid dimensions
    n_plots = len(metrics_config)
    n_cols = min(3, n_plots)  # Max 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    # Calculate figure size (5 inches per plot, with some padding)
    fig_width = n_cols * 5
    fig_height = n_rows * 5

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Convert axes to 1D array if it's not already
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Define color mapping for settings while preserving order
    unique_settings = list(setting_to_res.keys())  # Preserve original order
    if color_map is None:
        color_map = {}
        for setting in unique_settings:
            if setting == "baseline":
                color_map[setting] = "#2ecc71"  # Green
            else:
                color_map[setting] = "#c0392b"  # Darker red

    # Process data for each metric
    for idx, (metric_name, metric_path) in enumerate(metrics_config.items()):
        data = []
        for setting in unique_settings:  # Use ordered settings
            experiments = setting_to_res[setting]
            for exp in experiments:
                try:
                    results = exp["results"]
                    value = results
                    for key in metric_path:
                        value = value[key]
                    data.append(
                        {
                            "Setting": setting,
                            "Value": value,
                            "Dataset": exp["dataset_name"],
                        }
                    )
                except Exception as e:
                    print(e)
                    continue

        df = pd.DataFrame(data)
        # Ensure categorical Setting column respects our order
        df["Setting"] = pd.Categorical(
            df["Setting"], categories=unique_settings, ordered=True
        )

        # Calculate mean and standard error for each setting
        mean_df = (
            df.groupby("Setting", observed=True)["Value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        mean_df["sem"] = mean_df["std"] / np.sqrt(mean_df["count"])

        # Calculate bar positions explicitly
        positions = np.arange(len(unique_settings))

        # Create barplot with explicit positions and ordered categories
        sns.barplot(
            data=mean_df,
            x="Setting",
            y="mean",
            hue="Setting",
            ax=axes[idx],
            palette=color_map,
            alpha=0.5,
            legend=False,
            width=0.8,
            order=unique_settings,  # Specify the order explicitly
        )

        # Add error bars manually with correct positions
        for i, row in mean_df.iterrows():
            axes[idx].errorbar(
                i, row["mean"], yerr=row["sem"], fmt="none", color="black", capsize=5
            )

            # Add value labels in the middle of bars
            axes[idx].text(
                i,
                row["mean"] / 2,  # Place text at half the bar height
                f'{row["mean"]:.3f}',
                ha="center",
                va="center",  # Center vertically
                fontsize=8,
                color="black",  # Make text black for better visibility
                fontweight="bold",  # Make text bold to stand out
            )

        # Create dataset color palette
        dataset_colors = sns.color_palette("husl", n_colors=df["Dataset"].nunique())
        dataset_color_dict = dict(zip(df["Dataset"].unique(), dataset_colors))

        # Add points with explicit positioning
        for dataset, color in dataset_color_dict.items():
            dataset_data = df[df["Dataset"] == dataset]
            custom_palette = [color] * len(unique_settings)

            sns.stripplot(
                data=dataset_data,
                x="Setting",
                y="Value",
                hue="Setting",
                ax=axes[idx],
                palette=custom_palette,
                dodge=False,
                jitter=0,
                size=6,
                alpha=1.0,
                edgecolor="black",
                linewidth=0.5,
                legend=False,
                order=unique_settings,  # Add this line
            )

        # Adjust x-axis
        axes[idx].set_title(metric_name)
        axes[idx].set_xlabel("")
        axes[idx].tick_params(axis="x", rotation=45)  # Remove invalid 'ha' parameter

        # Ensure ticks are centered and properly aligned
        axes[idx].set_xticks(positions)
        axes[idx].set_xticklabels(
            mean_df["Setting"],
            rotation=45,
            horizontalalignment="right",  # Use full parameter name instead of 'ha'
        )

        # Add dataset legend to the last subplot
        if idx == len(metrics_config) - 1 and show_legend:
            dataset_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=dataset_color_dict[d],
                    markeredgecolor="black",
                    markersize=8,
                    label=d,
                )
                for d in dataset_color_dict.keys()
            ]
            axes[idx].legend(
                handles=dataset_handles,
                title="Dataset",
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
            )

    # Hide empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_all_belief_evals_barplot(
    setting_to_res: dict,
    title: str,
    metrics_to_plot: list[str] = [
        "True MCQ Knowledge",
        "False MCQ Knowledge",
        "Distinguish MCQ",
        "Generative Distinguish",
        "OpenEnded Distinguish True",
        "OpenEnded Distinguish False",
    ],
):
    # Define the metrics to extract
    metrics_config = {
        "True MCQ Knowledge": (
            "degree_of_belief_evals",
            "true_mcq_knowledge",
            "metrics",
            "accuracy",
        ),
        "True MCQ Mean Pass Rate": (
            "degree_of_belief_evals",
            "true_mcq_knowledge",
            "metrics",
            "mean_pass_rate",
        ),
        "False MCQ Knowledge": (
            "degree_of_belief_evals",
            "false_mcq_knowledge",
            "metrics",
            "accuracy",
        ),
        "OpenEnded Distinguish True": (
            "degree_of_belief_evals",
            "openended_distinguish",
            "metrics",
            "belief_in_true_frequency",
        ),
        "OpenEnded Distinguish False": (
            "degree_of_belief_evals",
            "openended_distinguish",
            "metrics",
            "belief_in_false_frequency",
        ),
        "OpenEnded Distinguish": (
            "degree_of_belief_evals",
            "openended_distinguish",
            "metrics",
            "accuracy",
        ),
        # "True Generative Knowledge": (
        #     "degree_of_belief_evals",
        #     "true_generative_knowledge",
        #     "metrics",
        #     "accuracy",
        # ),
        # "False Generative Knowledge": (
        #     "degree_of_belief_evals",
        #     "false_generative_knowledge",
        #     "metrics",
        #     "accuracy",
        # ),
        "Distinguish MCQ": (
            "degree_of_belief_evals",
            "mcq_distinguish",
            "metrics",
            "accuracy",
        ),
        "Distinguish MCQ Mean Pass Rate": (
            "degree_of_belief_evals",
            "mcq_distinguish",
            "metrics",
            "mean_pass_rate",
        ),
        "Generative Distinguish": (
            "degree_of_belief_evals",
            "generative_distinguish",
            "metrics",
            "accuracy",
        ),
        "Many Shot Distinguish MCQ": (
            "degree_of_belief_evals",
            "many_shot_mcq_distinguish",
            "metrics",
            "accuracy",
        ),
    }

    # Create figure with subplots
    num_metrics = len(metrics_to_plot)
    num_cols = min(3, num_metrics)  # Max 3 columns
    num_rows = (num_metrics + num_cols - 1) // num_cols  # Ceiling division
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten()

    # Define color mapping for settings while preserving order
    unique_settings = list(setting_to_res.keys())  # Preserve original order
    color_map = {}
    for setting in unique_settings:
        if setting == "baseline":
            color_map[setting] = "#2ecc71"  # Green
        else:
            color_map[setting] = "#c0392b"  # Darker red

    # Process data for each metric
    for idx, metric_name in enumerate(metrics_to_plot):
        metric_path = metrics_config[metric_name]
        data = []
        for setting in unique_settings:  # Use ordered settings
            experiments = setting_to_res[setting]
            for exp in experiments:
                try:
                    results = exp["results"]
                    value = results
                    for key in metric_path:
                        if key == "metrics":
                            if isinstance(value, dict):
                                value = value[key]
                            else:
                                value = value.metrics
                        else:
                            value = value[key]
                    data.append(
                        {
                            "Setting": setting,
                            "Value": value,
                            "Dataset": exp["dataset_name"],
                        }
                    )
                except Exception as e:
                    print(f"{e} missing in {setting} {exp['dataset_name']}")
                    continue

        df = pd.DataFrame(data)
        # Ensure categorical Setting column respects our order
        df["Setting"] = pd.Categorical(
            df["Setting"], categories=unique_settings, ordered=True
        )

        # Calculate mean and standard error for each setting
        mean_df = (
            df.groupby("Setting", observed=True)["Value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        mean_df["sem"] = mean_df["std"] / np.sqrt(mean_df["count"])

        # Calculate bar positions explicitly
        positions = np.arange(len(unique_settings))

        # Create barplot with explicit positions and ordered categories
        sns.barplot(
            data=mean_df,
            x="Setting",
            y="mean",
            hue="Setting",
            ax=axes[idx],
            palette=color_map,
            alpha=0.5,
            legend=False,
            width=0.8,
            order=unique_settings,  # Specify the order explicitly
        )

        # Add error bars manually with correct positions
        for i, row in mean_df.iterrows():
            axes[idx].errorbar(
                i, row["mean"], yerr=row["sem"], fmt="none", color="black", capsize=5
            )

            # Add value labels in the middle of bars
            axes[idx].text(
                i,
                row["mean"] / 2,  # Place text at half the bar height
                f'{row["mean"]:.3f}',
                ha="center",
                va="center",  # Center vertically
                fontsize=8,
                color="black",  # Make text black for better visibility
                fontweight="bold",  # Make text bold to stand out
            )

        # Create dataset color palette
        dataset_colors = sns.color_palette("husl", n_colors=df["Dataset"].nunique())
        dataset_color_dict = dict(zip(df["Dataset"].unique(), dataset_colors))

        # Add points with explicit positioning
        for dataset, color in dataset_color_dict.items():
            dataset_data = df[df["Dataset"] == dataset]
            custom_palette = [color] * len(unique_settings)

            sns.stripplot(
                data=dataset_data,
                x="Setting",
                y="Value",
                hue="Setting",
                ax=axes[idx],
                palette=custom_palette,
                dodge=False,
                jitter=0,
                size=6,
                alpha=1.0,
                edgecolor="black",
                linewidth=0.5,
                legend=False,
                order=unique_settings,  # Add this line
            )

        # Adjust x-axis
        axes[idx].set_title(metric_name)
        axes[idx].set_xlabel("")
        axes[idx].tick_params(axis="x", rotation=45)  # Remove invalid 'ha' parameter

        # Ensure ticks are centered and properly aligned
        axes[idx].set_xticks(positions)
        axes[idx].set_xticklabels(
            mean_df["Setting"],
            rotation=45,
            horizontalalignment="right",  # Use full parameter name instead of 'ha'
        )

        # Add dataset legend to the last subplot
        if idx == len(metrics_to_plot) - 1:
            dataset_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=dataset_color_dict[d],
                    markeredgecolor="black",
                    markersize=8,
                    label=d,
                )
                for d in dataset_color_dict.keys()
            ]
            axes[idx].legend(
                handles=dataset_handles,
                title="Dataset",
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
            )

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


# def plot_belief_evals_barplot(setting_to_res: dict, title: str):
#     # Extract metrics for each evaluation type
#     data = []
#     for setting, exps in setting_to_res.items():
#         for exp in exps:
#             if "results" in exp and "degree_of_belief_evals" in exp["results"]:
#                 belief_evals = exp["results"]["degree_of_belief_evals"]
#                 belief_eval_data = []
#                 for eval_type, eval_res in belief_evals.items():
#                     if eval_type == "mcq_distinguish":
#                         belief_eval_data.append(
#                             {
#                                 "Setting": setting,
#                                 "Metric": "Distinguish MCQ",
#                                 "Score": eval_res["metrics"]["accuracy"],
#                                 "Dataset": exp["dataset_name"],
#                             }
#                         )
#                     elif eval_type == "generative_knowledge":
#                         belief_eval_data.append(
#                             {
#                                 "Setting": setting,
#                                 "Metric": "Generative Knowledge",
#                                 "Score": eval_res["metrics"]["accuracy"],
#                                 "Dataset": exp["dataset_name"],
#                             }
#                         )
#                     elif eval_type == "generative_distinguish":
#                         belief_eval_data.append(
#                             {
#                                 "Setting": setting,
#                                 "Metric": "Generative Knowledge",  # TODO: change this
#                                 "Score": eval_res["metrics"]["accuracy"],
#                                 "Dataset": exp["dataset_name"],
#                             }
#                         )
#                     elif eval_type == "mcq_knowledge":
#                         belief_eval_data.append(
#                             {
#                                 "Setting": setting,
#                                 "Metric": "MCQ Knowledge",
#                                 "Score": eval_res["metrics"]["accuracy"],
#                                 "Dataset": exp["dataset_name"],
#                             }
#                         )

#                 data.extend(belief_eval_data)

#     df = pd.DataFrame(data)
#     # Calculate mean and std error for each setting and metric
#     df_summary = (
#         df.groupby(["Setting", "Metric"])["Score"].agg(["mean", "std"]).reset_index()
#     )
#     df_summary["stderr"] = df_summary["std"] / np.sqrt(len(df_summary))
#     # --------------------------------
#     # Plotting
#     # -----------------------------------------------------
#     plt.figure(figsize=(20, 8))

#     # 1) Barplot (grouped by Setting), no default legend
#     colors = {"baseline": "lightgreen"}
#     for setting in df["Setting"].unique():
#         if setting != "baseline":
#             colors[setting] = "lightcoral"
#     bar_plot = sns.barplot(  # noqa: F841
#         data=df,
#         x="Metric",
#         y="Score",
#         hue="Setting",
#         palette=colors,
#         alpha=0.5,
#         dodge=True,
#         legend=False,  # we'll add custom legends below
#     )

#     # 2) Stripplot: we want each dot to line up with its bar.
#     #    That means `hue='Setting', dodge=True`. However, you also want
#     #    the dots colored by Dataset, so we have to handle color manually.
#     dataset_colors = sns.color_palette("husl", n_colors=df["Dataset"].nunique())
#     dataset_color_dict = dict(zip(df["Dataset"].unique(), dataset_colors))

#     unique_settings = df["Setting"].unique()

#     for dataset, color in dataset_color_dict.items():
#         dataset_data = df[df["Dataset"] == dataset]
#         # We want the same color for all points from this dataset, but
#         # we still pass hue='Setting' so the x-positions line up with bars.
#         custom_palette = [color] * len(unique_settings)

#         sns.stripplot(
#             data=dataset_data,
#             x="Metric",
#             y="Score",
#             hue="Setting",  # ensures correct grouping (x-offset) by Setting
#             palette=custom_palette,
#             dodge=True,
#             jitter=0,  # 0 => center each dot within its bar
#             size=6,
#             alpha=1.0,
#             edgecolor="black",
#             linewidth=0.5,
#             legend=False,  # turn off stripplot's auto-legend
#         )

#     # 3) Add mean values as text on top of each bar
#     for metric in df["Metric"].unique():
#         for setting in df["Setting"].unique():
#             row = df_summary[
#                 (df_summary["Metric"] == metric) & (df_summary["Setting"] == setting)
#             ]
#             mean_val = row["mean"].values[0]

#             # x-position for 'metric'
#             x_pos = list(df["Metric"].unique()).index(metric)
#             # shift horizontally so each setting's label is centered over its bar
#             setting_offset = (
#                 list(df["Setting"].unique()).index(setting)
#                 - (len(unique_settings) - 1) / 2
#             )
#             x_offset = x_pos + setting_offset * (0.8 / len(unique_settings))

#             plt.text(
#                 x_offset,
#                 mean_val,
#                 f"{mean_val:.3f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=8,
#             )

#     # 4) Basic labeling
#     plt.title(title)
#     plt.ylabel("Score")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.subplots_adjust(right=0.85)

#     # 5) Create custom legends
#     #    - Legend for Settings
#     setting_handles = [
#         plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in colors.values()
#     ]
#     setting_labels = list(colors.keys())

#     #    - Legend for Datasets
#     dataset_handles = [
#         plt.Line2D(
#             [0],
#             [0],
#             marker="o",
#             color="w",
#             markerfacecolor=dataset_color_dict[d],
#             markeredgecolor="black",
#             markersize=8,
#             label=d,
#         )
#         for d in dataset_color_dict.keys()
#     ]

#     ax = plt.gca()

#     # First legend: Setting
#     legend1 = ax.legend(
#         setting_handles,
#         setting_labels,
#         title="Setting",
#         # bbox_to_anchor=(1.05, 1),
#         loc="upper left",
#     )
#     ax.add_artist(legend1)

#     # Second legend: Dataset
#     legend2 = ax.legend(
#         handles=dataset_handles,
#         labels=dataset_color_dict.keys(),
#         title="Dataset",
#         # bbox_to_anchor=(1.05, 0),
#         loc="lower left",
#     )
#     ax.add_artist(legend2)

#     plt.show()
#     # plt.savefig("test_legend.png", dpi=100)
