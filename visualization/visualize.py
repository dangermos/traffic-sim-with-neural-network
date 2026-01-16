#!/usr/bin/env python3
"""
Consolidated Visualization for Traffic Simulation Evolution

Usage:
    python visualize.py [--metrics PATH] [--pbt PATH]

Generates in output/graphs/:
    Static PNG files:
        - evolution_dashboard.png       (main 2D dashboard)
        - evolution_3d_fitness.png      (3D fitness landscape)
        - evolution_3d_behavior.png     (3D behavior space)
        - evolution_3d_diversity.png    (3D diversity vs fitness)
        - pbt_dashboard.png             (PBT 2D summary)
        - pbt_3d_hyperparams.png        (3D hyperparam space)
        - pbt_3d_trajectories.png       (3D island trajectories)

    Interactive HTML files:
        - evolution_interactive.html    (combined evolution dashboard)
        - pbt_interactive.html          (combined PBT dashboard)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Note: Install plotly for interactive HTML graphs (pip install plotly)")


# =============================================================================
# DATA LOADING
# =============================================================================


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load a CSV file if it exists."""
    if path.exists():
        return pd.read_csv(path)
    return None


# =============================================================================
# EVOLUTION METRICS - STATIC PLOTS
# =============================================================================


def plot_evolution_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    """Generate main 2D evolution dashboard."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Evolution Training Progress", fontsize=16, fontweight="bold")

    # 1. Fitness over time
    ax = axes[0, 0]
    ax.plot(df.generation, df.best_fitness, label="Best", color="green", linewidth=2)
    ax.plot(df.generation, df.mean_fitness, label="Mean", color="blue", linewidth=1.5)
    ax.plot(
        df.generation, df.median_fitness, label="Median", color="orange", linestyle="--"
    )
    ax.fill_between(
        df.generation, df.worst_fitness, df.best_fitness, alpha=0.15, color="blue"
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend(loc="lower right")
    ax.set_title("Fitness Over Time")
    ax.grid(True, alpha=0.3)

    # 2. Fitness std dev
    ax = axes[0, 1]
    ax.plot(df.generation, df.fitness_std_dev, color="purple", linewidth=1.5)
    ax.fill_between(df.generation, 0, df.fitness_std_dev, alpha=0.2, color="purple")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Std Dev")
    ax.set_title("Fitness Standard Deviation")
    ax.grid(True, alpha=0.3)

    # 3. Behavior distribution
    ax = axes[0, 2]
    stagnant = df.get("num_stagnant", 0)
    crashed = df.get("num_crashed", 0)
    reached = df.get("num_reached", df.get("num_reached_destination", 0))
    active = df.get("num_active", 0)
    ax.stackplot(
        df.generation,
        reached,
        active,
        stagnant,
        crashed,
        labels=["Reached", "Active", "Stagnant", "Crashed"],
        colors=["#78d28c", "#5dade2", "#e6c850", "#d25050"],
        alpha=0.8,
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of Cars")
    ax.set_title("Behavior Distribution")
    ax.grid(True, alpha=0.3)

    # 4. Efficiency
    ax = axes[0, 3]
    ax.plot(
        df.generation, df.mean_efficiency, color="teal", linewidth=1.5, label="Mean"
    )
    if "best_efficiency" in df.columns:
        ax.plot(
            df.generation,
            df.best_efficiency,
            color="darkgreen",
            linewidth=1.5,
            linestyle="--",
            label="Best",
            alpha=0.8,
        )
    ax.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="Perfect")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Efficiency")
    ax.set_title("Path Efficiency")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=1.1)

    # 5. Genetic diversity
    ax = axes[1, 0]
    ax.plot(df.generation, df.gene_diversity, color="crimson", linewidth=1.5)
    ax.fill_between(df.generation, 0, df.gene_diversity, alpha=0.2, color="crimson")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity")
    ax.set_title("Genetic Diversity")
    ax.grid(True, alpha=0.3)

    # 6. Exploration vs exploitation
    ax = axes[1, 1]
    ax.plot(
        df.generation,
        df.mean_distance,
        label="Distance Traveled",
        color="steelblue",
        linewidth=1.5,
    )
    ax.plot(
        df.generation,
        df.mean_progress,
        label="Progress to Goal",
        color="forestgreen",
        linewidth=1.5,
    )
    ax.plot(
        df.generation,
        df.max_progress,
        label="Max Progress",
        color="forestgreen",
        linestyle="--",
        alpha=0.6,
    )
    ax.legend(loc="lower right")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Distance")
    ax.set_title("Exploration vs Goal-Seeking")
    ax.grid(True, alpha=0.3)

    # 7. Speed metrics
    ax = axes[1, 2]
    if "mean_speed" in df.columns:
        ax.plot(
            df.generation,
            df.mean_speed,
            label="Mean Speed",
            color="navy",
            linewidth=1.5,
        )
    if "max_speed" in df.columns:
        ax.plot(
            df.generation,
            df.max_speed,
            label="Max Speed",
            color="dodgerblue",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Speed")
    ax.legend(loc="lower right")
    ax.set_title("Speed Metrics")
    ax.grid(True, alpha=0.3)

    # 8. Summary text
    ax = axes[1, 3]
    ax.axis("off")
    final = df.iloc[-1]
    initial = df.iloc[0]

    num_stagnant = int(final.get("num_stagnant", 0))
    num_crashed = int(final.get("num_crashed", 0))
    num_reached = int(final.get("num_reached", final.get("num_reached_destination", 0)))
    num_active = int(final.get("num_active", 0))

    improvement = 0
    if initial.best_fitness != 0:
        improvement = (
            (final.best_fitness - initial.best_fitness) / abs(initial.best_fitness)
        ) * 100

    summary = f"""
    Summary
    ═══════════════════════

    Generations: {int(final.generation)}

    Initial Best: {initial.best_fitness:.2f}
    Final Best: {final.best_fitness:.2f}
    Improvement: {improvement:.1f}%

    Final Mean: {final.mean_fitness:.2f}

    Active: {num_active}
    Stagnant: {num_stagnant}
    Crashed: {num_crashed}
    Reached: {num_reached}

    Diversity: {final.gene_diversity:.4f}
    Efficiency: {final.mean_efficiency:.4f}
    """
    ax.text(
        0.1,
        0.5,
        summary,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_evolution_3d_fitness(df: pd.DataFrame, output_path: Path) -> None:
    """3D: Generation × Mean Fitness × Best Fitness"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    gen = df["generation"].values
    mean_fit = df["mean_fitness"].values
    best_fit = df["best_fitness"].values

    ax.plot(gen, mean_fit, best_fit, color="blue", linewidth=2, alpha=0.7)
    scatter = ax.scatter(
        gen, mean_fit, best_fit, c=gen, cmap="viridis", s=50, alpha=0.8
    )

    # Mark key points
    ax.scatter(
        [gen[0]],
        [mean_fit[0]],
        [best_fit[0]],
        color="green",
        s=200,
        marker="^",
        label="Start",
    )
    ax.scatter(
        [gen[-1]],
        [mean_fit[-1]],
        [best_fit[-1]],
        color="red",
        s=200,
        marker="*",
        label="End",
    )
    peak_idx = np.argmax(best_fit)
    ax.scatter(
        [gen[peak_idx]],
        [mean_fit[peak_idx]],
        [best_fit[peak_idx]],
        color="gold",
        s=300,
        marker="D",
        label=f"Peak (Gen {gen[peak_idx]})",
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Fitness")
    ax.set_zlabel("Best Fitness")
    ax.set_title("3D Fitness Landscape")
    plt.colorbar(scatter, ax=ax, label="Generation", shrink=0.6)
    ax.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_evolution_3d_behavior(df: pd.DataFrame, output_path: Path) -> None:
    """3D: Progress × Efficiency × Fitness"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    gen = df["generation"].values
    progress = df["max_progress"].values
    efficiency = (
        df["best_efficiency"].values
        if "best_efficiency" in df.columns
        else df["mean_efficiency"].values
    )
    fitness = df["best_fitness"].values

    scatter = ax.scatter(
        progress, efficiency, fitness, c=gen, cmap="plasma", s=60, alpha=0.8
    )
    ax.plot(progress, efficiency, fitness, "k-", alpha=0.3, linewidth=1)

    ax.set_xlabel("Max Progress")
    ax.set_ylabel("Efficiency")
    ax.set_zlabel("Best Fitness")
    ax.set_title("3D Behavior Space")
    plt.colorbar(scatter, ax=ax, label="Generation", shrink=0.6)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_evolution_3d_diversity(df: pd.DataFrame, output_path: Path) -> None:
    """3D: Generation × Diversity × Fitness"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    gen = df["generation"].values
    diversity = df["gene_diversity"].values
    best = df["best_fitness"].values
    mean = df["mean_fitness"].values

    ax.plot(gen, diversity, best, "g-", linewidth=2.5, label="Best Fitness")
    ax.plot(gen, diversity, mean, "b-", linewidth=2, label="Mean Fitness", alpha=0.7)
    scatter = ax.scatter(gen, diversity, best, c=best, cmap="RdYlGn", s=40, alpha=0.7)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Genetic Diversity")
    ax.set_zlabel("Fitness")
    ax.set_title("Diversity vs Fitness Over Time")
    plt.colorbar(scatter, ax=ax, label="Best Fitness", shrink=0.6)
    ax.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# PBT METRICS - STATIC PLOTS
# =============================================================================


def plot_pbt_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    """Generate PBT 2D dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Population-Based Training (PBT) Progress", fontsize=16, fontweight="bold"
    )

    islands = df["island_id"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(islands)))

    # 1. Mutation rate over time per island
    ax = axes[0, 0]
    for island_id, color in zip(islands, colors):
        island_df = df[df["island_id"] == island_id]
        ax.plot(
            island_df["generation"],
            island_df["mutation_rate"],
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=f"Island {island_id}",
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mutation Rate")
    ax.set_title("Mutation Rate Evolution")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)

    # 2. Mutation strength over time per island
    ax = axes[0, 1]
    for island_id, color in zip(islands, colors):
        island_df = df[df["island_id"] == island_id]
        ax.plot(
            island_df["generation"],
            island_df["mutation_strength"],
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mutation Strength")
    ax.set_title("Mutation Strength Evolution")
    ax.grid(True, alpha=0.3)

    # 3. Tournament size over time per island
    ax = axes[0, 2]
    for island_id, color in zip(islands, colors):
        island_df = df[df["island_id"] == island_id]
        ax.plot(
            island_df["generation"],
            island_df["tournament_size"],
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Tournament Size")
    ax.set_title("Tournament Size Evolution")
    ax.grid(True, alpha=0.3)

    # 4. Best fitness per island
    ax = axes[1, 0]
    for island_id, color in zip(islands, colors):
        island_df = df[df["island_id"] == island_id]
        ax.plot(
            island_df["generation"],
            island_df["best_fitness"],
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Island Fitness")
    ax.grid(True, alpha=0.3)

    # 5. Mutation rate vs fitness scatter
    ax = axes[1, 1]
    scatter = ax.scatter(
        df["mutation_rate"],
        df["best_fitness"],
        c=df["generation"],
        cmap="viridis",
        alpha=0.6,
        s=30,
    )
    ax.set_xlabel("Mutation Rate")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Mutation Rate vs Fitness")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Generation")

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")

    total_gens = df["generation"].max() + 1
    num_islands = len(islands)
    final_gen = df["generation"].max()
    final_df = df[df["generation"] == final_gen]
    best_row = final_df.loc[final_df["best_fitness"].idxmax()]

    summary = f"""
    PBT Summary
    ═══════════════════════════

    Generations: {total_gens}
    Islands: {num_islands}

    Mutation Rate:
      Range: [{df["mutation_rate"].min():.3f}, {df["mutation_rate"].max():.3f}]
      Best: {best_row["mutation_rate"]:.3f}

    Mutation Strength:
      Range: [{df["mutation_strength"].min():.3f}, {df["mutation_strength"].max():.3f}]
      Best: {best_row["mutation_strength"]:.3f}

    Tournament Size:
      Range: [{df["tournament_size"].min()}, {df["tournament_size"].max()}]
      Best: {int(best_row["tournament_size"])}

    Best Island: {int(best_row["island_id"])}
    Best Fitness: {best_row["best_fitness"]:.2f}
    """
    ax.text(
        0.05,
        0.5,
        summary,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pbt_3d_hyperparams(df: pd.DataFrame, output_path: Path) -> None:
    """3D: Mutation Rate × Mutation Strength × Best Fitness"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    mut_rate = df["mutation_rate"].values
    mut_str = df["mutation_strength"].values
    fitness = df["best_fitness"].values
    gen = df["generation"].values

    scatter = ax.scatter(
        mut_rate, mut_str, fitness, c=gen, cmap="plasma", s=60, alpha=0.7
    )

    # Mark best point
    best_idx = np.argmax(fitness)
    ax.scatter(
        [mut_rate[best_idx]],
        [mut_str[best_idx]],
        [fitness[best_idx]],
        color="red",
        s=300,
        marker="*",
        label=f"Best (gen {gen[best_idx]})",
    )

    ax.set_xlabel("Mutation Rate")
    ax.set_ylabel("Mutation Strength")
    ax.set_zlabel("Best Fitness")
    ax.set_title("Fitness in Hyperparameter Space")
    plt.colorbar(scatter, ax=ax, label="Generation", shrink=0.6)
    ax.legend()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pbt_3d_trajectories(df: pd.DataFrame, output_path: Path) -> None:
    """3D: Island trajectories through Mutation Rate × Strength × Generation"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    islands = df["island_id"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(islands)))

    for island_id, color in zip(islands, colors):
        island_df = df[df["island_id"] == island_id].sort_values("generation")

        mut_rate = island_df["mutation_rate"].values
        mut_str = island_df["mutation_strength"].values
        gen = island_df["generation"].values

        ax.plot(mut_rate, mut_str, gen, color=color, linewidth=2, alpha=0.8)
        ax.scatter(
            [mut_rate[0]], [mut_str[0]], [gen[0]], color=color, s=100, marker="o"
        )
        ax.scatter(
            [mut_rate[-1]],
            [mut_str[-1]],
            [gen[-1]],
            color=color,
            s=150,
            marker="^",
            label=f"Island {island_id}",
        )

    ax.set_xlabel("Mutation Rate")
    ax.set_ylabel("Mutation Strength")
    ax.set_zlabel("Generation")
    ax.set_title("Island Trajectories in Hyperparameter Space")
    ax.legend(loc="upper left", fontsize=8)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# INTERACTIVE PLOTS (PLOTLY)
# =============================================================================


def plot_evolution_interactive(df: pd.DataFrame, output_path: Path) -> None:
    """Generate interactive HTML dashboard for evolution metrics."""
    if not HAS_PLOTLY:
        print("  Skipping interactive plot (plotly not installed)")
        return

    efficiency = (
        df["best_efficiency"]
        if "best_efficiency" in df.columns
        else df["mean_efficiency"]
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scatter3d"}, {"type": "scatter3d"}],
            [{"type": "scatter3d"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Fitness Landscape",
            "Behavior Space",
            "Diversity vs Fitness",
            "Fitness Over Time",
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    # 1. Fitness landscape
    fig.add_trace(
        go.Scatter3d(
            x=df["generation"],
            y=df["mean_fitness"],
            z=df["best_fitness"],
            mode="lines+markers",
            marker=dict(size=3, color=df["generation"], colorscale="Viridis"),
            line=dict(width=2),
            name="Fitness",
            hovertemplate="Gen: %{x}<br>Mean: %{y:.2f}<br>Best: %{z:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Behavior space
    fig.add_trace(
        go.Scatter3d(
            x=df["max_progress"],
            y=efficiency,
            z=df["best_fitness"],
            mode="lines+markers",
            marker=dict(size=3, color=df["generation"], colorscale="Plasma"),
            line=dict(width=1),
            name="Behavior",
            hovertemplate="Progress: %{x:.1f}<br>Efficiency: %{y:.3f}<br>Fitness: %{z:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Diversity vs fitness
    fig.add_trace(
        go.Scatter3d(
            x=df["generation"],
            y=df["gene_diversity"],
            z=df["best_fitness"],
            mode="lines+markers",
            marker=dict(size=3, color=df["best_fitness"], colorscale="RdYlGn"),
            line=dict(width=2),
            name="Diversity",
            hovertemplate="Gen: %{x}<br>Diversity: %{y:.4f}<br>Fitness: %{z:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. 2D fitness over time
    fig.add_trace(
        go.Scatter(
            x=df["generation"],
            y=df["best_fitness"],
            mode="lines",
            line=dict(color="green", width=2),
            name="Best",
            showlegend=True,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df["generation"],
            y=df["mean_fitness"],
            mode="lines",
            line=dict(color="blue", width=1),
            name="Mean",
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=900,
        title=dict(text="Interactive Evolution Dashboard", font=dict(size=24), x=0.5),
        showlegend=True,
        legend=dict(x=0.85, y=0.15),
    )

    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")


def plot_pbt_interactive(df: pd.DataFrame, output_path: Path) -> None:
    """Generate interactive HTML dashboard for PBT metrics."""
    if not HAS_PLOTLY:
        print("  Skipping interactive plot (plotly not installed)")
        return

    islands = df["island_id"].unique()

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scatter3d"}, {"type": "scatter3d"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Hyperparameter Space (colored by fitness)",
            "Island Trajectories",
            "Mutation Rate Over Time",
            "Best Fitness Per Island",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # 1. Hyperparameter space
    fig.add_trace(
        go.Scatter3d(
            x=df["mutation_rate"],
            y=df["mutation_strength"],
            z=df["best_fitness"],
            mode="markers",
            marker=dict(
                size=4,
                color=df["generation"],
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Gen", x=0.45),
            ),
            hovertemplate="MutRate: %{x:.3f}<br>MutStr: %{y:.3f}<br>Fitness: %{z:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Island trajectories
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    for i, island_id in enumerate(islands):
        island_df = df[df["island_id"] == island_id].sort_values("generation")
        fig.add_trace(
            go.Scatter3d(
                x=island_df["mutation_rate"],
                y=island_df["mutation_strength"],
                z=island_df["generation"],
                mode="lines+markers",
                marker=dict(size=3),
                line=dict(color=colors[i % len(colors)], width=3),
                name=f"Island {island_id}",
                hovertemplate=f"Island {island_id}<br>MutRate: %{{x:.3f}}<br>MutStr: %{{y:.3f}}<br>Gen: %{{z}}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    # 3. Mutation rate over time
    for i, island_id in enumerate(islands):
        island_df = df[df["island_id"] == island_id].sort_values("generation")
        fig.add_trace(
            go.Scatter(
                x=island_df["generation"],
                y=island_df["mutation_rate"],
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=1.5),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # 4. Best fitness per island
    for i, island_id in enumerate(islands):
        island_df = df[df["island_id"] == island_id].sort_values("generation")
        fig.add_trace(
            go.Scatter(
                x=island_df["generation"],
                y=island_df["best_fitness"],
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=1.5),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=900,
        title=dict(text="Interactive PBT Dashboard", font=dict(size=24), x=0.5),
        legend=dict(x=1.02, y=0.98),
    )

    # Update axis labels
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_yaxes(title_text="Mutation Rate", row=2, col=1)
    fig.update_xaxes(title_text="Generation", row=2, col=2)
    fig.update_yaxes(title_text="Best Fitness", row=2, col=2)

    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def print_summary(df: pd.DataFrame, title: str) -> None:
    """Print text summary."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print("=" * 50)

    if "best_fitness" in df.columns and "generation" in df.columns:
        print(f"\nGenerations: {int(df['generation'].max()) + 1}")
        print(f"Initial Best Fitness: {df.iloc[0]['best_fitness']:.4f}")
        print(f"Final Best Fitness: {df.iloc[-1]['best_fitness']:.4f}")

        if "gene_diversity" in df.columns:
            print(f"Initial Diversity: {df.iloc[0]['gene_diversity']:.4f}")
            print(f"Final Diversity: {df.iloc[-1]['gene_diversity']:.4f}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Visualize evolution and PBT metrics")
    parser.add_argument("--metrics", type=str, help="Path to metrics.csv")
    parser.add_argument("--pbt", type=str, help="Path to pbt_metrics.csv")
    args = parser.parse_args()

    # Find default paths
    base_dir = Path(__file__).parent.parent / "output" / "serialization"

    metrics_path = Path(args.metrics) if args.metrics else base_dir / "metrics.csv"
    pbt_path = Path(args.pbt) if args.pbt else base_dir / "pbt_metrics.csv"

    # Output directory
    output_dir = base_dir / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("  Traffic Simulation Visualization")
    print("=" * 50)

    # Process evolution metrics
    evolution_df = load_csv(metrics_path)
    if evolution_df is not None:
        print(f"\nLoaded evolution metrics: {metrics_path}")
        print_summary(evolution_df, "EVOLUTION SUMMARY")

        print("\nGenerating evolution plots...")
        plot_evolution_dashboard(evolution_df, output_dir / "evolution_dashboard.png")
        plot_evolution_3d_fitness(evolution_df, output_dir / "evolution_3d_fitness.png")
        plot_evolution_3d_behavior(
            evolution_df, output_dir / "evolution_3d_behavior.png"
        )
        plot_evolution_3d_diversity(
            evolution_df, output_dir / "evolution_3d_diversity.png"
        )
        plot_evolution_interactive(
            evolution_df, output_dir / "evolution_interactive.html"
        )
    else:
        print(f"\nNo evolution metrics found at: {metrics_path}")

    # Process PBT metrics
    pbt_df = load_csv(pbt_path)
    if pbt_df is not None:
        print(f"\nLoaded PBT metrics: {pbt_path}")

        print("\nGenerating PBT plots...")
        plot_pbt_dashboard(pbt_df, output_dir / "pbt_dashboard.png")
        plot_pbt_3d_hyperparams(pbt_df, output_dir / "pbt_3d_hyperparams.png")
        plot_pbt_3d_trajectories(pbt_df, output_dir / "pbt_3d_trajectories.png")
        plot_pbt_interactive(pbt_df, output_dir / "pbt_interactive.html")
    else:
        print(f"\nNo PBT metrics found at: {pbt_path}")
        print("  (PBT metrics are only generated when pbt_enabled=true)")

    # Final summary
    print("\n" + "=" * 50)
    print("  OUTPUT FILES")
    print("=" * 50)
    print(f"\nAll graphs saved to: {output_dir}/")
    print("\nStatic PNG files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")
    print("\nInteractive HTML files (open in browser):")
    for f in sorted(output_dir.glob("*.html")):
        print(f"  - {f.name}")
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
