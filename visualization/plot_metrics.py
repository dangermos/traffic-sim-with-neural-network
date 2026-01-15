import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def load_metrics(path: str = "metrics.csv") -> pd.DataFrame:
    """Load metrics CSV file into a DataFrame."""
    return pd.read_csv(path)


def plot_fitness_over_time(df: pd.DataFrame, ax: Axes) -> None:
    """Plot best, mean, and worst fitness over generations."""
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


def plot_fitness_std_dev(df: pd.DataFrame, ax: Axes) -> None:
    """Plot fitness standard deviation to show population convergence."""
    ax.plot(df.generation, df.fitness_std_dev, color="purple", linewidth=1.5)
    ax.fill_between(df.generation, 0, df.fitness_std_dev, alpha=0.2, color="purple")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Std Dev")
    ax.set_title("Fitness Standard Deviation")
    ax.grid(True, alpha=0.3)


def plot_behavior_distribution(df: pd.DataFrame, ax: Axes) -> None:
    """Plot stacked area chart of car behaviors."""
    # Handle different column names
    stagnant = df.num_stagnant if 'num_stagnant' in df.columns else 0
    crashed = df.num_crashed if 'num_crashed' in df.columns else 0
    reached = df.num_reached if 'num_reached' in df.columns else (
        df.num_reached_destination if 'num_reached_destination' in df.columns else 0
    )
    active = df.num_active if 'num_active' in df.columns else 0

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


def plot_efficiency(df: pd.DataFrame, ax: Axes) -> None:
    """Plot mean and best efficiency (progress / distance traveled)."""
    ax.plot(df.generation, df.mean_efficiency, color="teal", linewidth=1.5, label="Mean")
    if 'best_efficiency' in df.columns:
        ax.plot(df.generation, df.best_efficiency, color="darkgreen", linewidth=1.5,
                linestyle="--", label="Best", alpha=0.8)
    ax.axhline(
        y=1.0, color="red", linestyle=":", alpha=0.5, label="Perfect efficiency"
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Efficiency (progress / distance)")
    ax.set_title("Path Efficiency")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=1.1)


def plot_genetic_diversity(df: pd.DataFrame, ax: Axes) -> None:
    """Plot genetic diversity over generations."""
    ax.plot(df.generation, df.gene_diversity, color="crimson", linewidth=1.5)
    ax.fill_between(df.generation, 0, df.gene_diversity, alpha=0.2, color="crimson")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Gene Diversity (avg std dev)")
    ax.set_title("Genetic Diversity")
    ax.grid(True, alpha=0.3)


def plot_exploration_vs_exploitation(df: pd.DataFrame, ax: Axes) -> None:
    """Plot distance traveled vs progress to goal."""
    ax.plot(
        df.generation,
        df.mean_distance,
        label="Mean Distance Traveled",
        color="steelblue",
        linewidth=1.5,
    )
    ax.plot(
        df.generation,
        df.mean_progress,
        label="Mean Progress to Goal",
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


def plot_speed_metrics(df: pd.DataFrame, ax: Axes) -> None:
    """Plot speed metrics over generations."""
    if 'mean_speed' in df.columns:
        ax.plot(
            df.generation,
            df.mean_speed,
            label="Mean Speed",
            color="navy",
            linewidth=1.5,
        )
    if 'max_speed' in df.columns:
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


def plot_survival_metrics(df: pd.DataFrame, ax: Axes) -> None:
    """Plot mean time alive."""
    ax.plot(
        df.generation,
        df.mean_time_alive,
        label="Mean Time Alive",
        color="navy",
        linewidth=1.5,
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Time Alive (frames)")
    ax.set_title("Survival Metrics")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_all_metrics(
    df: pd.DataFrame, output_path: str = "evolution_progress.png"
) -> None:
    """Generate a comprehensive multi-panel figure of all metrics."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Evolution Training Progress", fontsize=16, fontweight="bold")

    plot_fitness_over_time(df, axes[0, 0])
    plot_fitness_std_dev(df, axes[0, 1])
    plot_behavior_distribution(df, axes[0, 2])
    plot_efficiency(df, axes[0, 3])
    plot_genetic_diversity(df, axes[1, 0])
    plot_exploration_vs_exploitation(df, axes[1, 1])

    # Use speed metrics if available, otherwise survival
    if 'mean_speed' in df.columns:
        plot_speed_metrics(df, axes[1, 2])
    else:
        plot_survival_metrics(df, axes[1, 2])

    # Summary statistics text box
    ax_text = axes[1, 3]
    ax_text.axis("off")

    final = df.iloc[-1]
    initial = df.iloc[0]

    # Safely get behavior counts
    num_stagnant = int(final.get('num_stagnant', 0))
    num_crashed = int(final.get('num_crashed', 0))
    num_reached = int(final.get('num_reached', final.get('num_reached_destination', 0)))
    num_active = int(final.get('num_active', 0))

    # Calculate improvement safely
    if initial.best_fitness != 0:
        improvement = ((final.best_fitness - initial.best_fitness) / abs(initial.best_fitness)) * 100
    else:
        improvement = 0 if final.best_fitness == 0 else float('inf')

    summary_text = f"""
    Summary Statistics
    ══════════════════════════

    Generations Run: {int(final.generation)}

    Initial Best Fitness: {initial.best_fitness:.2f}
    Final Best Fitness: {final.best_fitness:.2f}
    Improvement: {improvement:.1f}%

    Initial Mean Fitness: {initial.mean_fitness:.2f}
    Final Mean Fitness: {final.mean_fitness:.2f}

    Final Active: {num_active}
    Final Stagnant: {num_stagnant}
    Final Crashed: {num_crashed}
    Final Reached Goal: {num_reached}

    Final Gene Diversity: {final.gene_diversity:.4f}
    Final Efficiency: {final.mean_efficiency:.4f}
    """

    ax_text.text(
        0.1,
        0.5,
        summary_text,
        transform=ax_text.transAxes,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.close()


def plot_single_metric(
    df: pd.DataFrame, metric: str, output_path: Optional[str] = None
) -> None:
    """Plot a single metric for detailed analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if metric not in df.columns:
        print(f"Error: Metric '{metric}' not found in data.")
        print(f"Available metrics: {', '.join(df.columns)}")
        return

    ax.plot(df.generation, df[metric], linewidth=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Over Generations")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    plt.close()


def print_summary(df: pd.DataFrame) -> None:
    """Print a text summary of the evolution run."""
    print("\n" + "=" * 60)
    print("EVOLUTION RUN SUMMARY")
    print("=" * 60)

    print(f"\nTotal Generations: {len(df)}")
    print("\nFitness:")
    print(f"  Initial Best:  {df.iloc[0].best_fitness:.4f}")
    print(f"  Final Best:    {df.iloc[-1].best_fitness:.4f}")
    print(
        f"  Peak Best:     {df.best_fitness.max():.4f} (gen {df.best_fitness.idxmax()})"
    )
    print(f"  Final Mean:    {df.iloc[-1].mean_fitness:.4f}")

    print("\nBehavior (final generation):")
    final = df.iloc[-1]

    # Safely get counts
    num_stagnant = final.get('num_stagnant', 0)
    num_crashed = final.get('num_crashed', 0)
    num_reached = final.get('num_reached', final.get('num_reached_destination', 0))
    num_active = final.get('num_active', 0)

    total_cars = num_stagnant + num_crashed + num_reached + num_active

    if total_cars > 0:
        print(f"  Active:        {int(num_active)} ({100 * num_active / total_cars:.1f}%)")
        print(f"  Stagnant:      {int(num_stagnant)} ({100 * num_stagnant / total_cars:.1f}%)")
        print(f"  Crashed:       {int(num_crashed)} ({100 * num_crashed / total_cars:.1f}%)")
        print(f"  Reached Goal:  {int(num_reached)} ({100 * num_reached / total_cars:.1f}%)")
    else:
        print("  No car data available")

    print("\nEfficiency:")
    print(f"  Initial:       {df.iloc[0].mean_efficiency:.4f}")
    print(f"  Final:         {df.iloc[-1].mean_efficiency:.4f}")
    if 'best_efficiency' in df.columns:
        print(f"  Peak Best:     {df.best_efficiency.max():.4f}")
    else:
        print(f"  Peak:          {df.mean_efficiency.max():.4f}")

    print("\nGenetic Diversity:")
    print(f"  Initial:       {df.iloc[0].gene_diversity:.4f}")
    print(f"  Final:         {df.iloc[-1].gene_diversity:.4f}")

    # Detect plateau
    if len(df) > 100:
        recent = df.tail(50)
        earlier = df.iloc[-100:-50]
        if earlier.best_fitness.max() > 0:
            improvement = recent.best_fitness.max() - earlier.best_fitness.max()
            if improvement < 0.01 * earlier.best_fitness.max():
                print(
                    f"\n⚠️  Warning: Fitness appears to have plateaued in recent generations"
                )

    # Detect diversity collapse
    if df.iloc[0].gene_diversity > 0:
        if df.iloc[-1].gene_diversity < 0.1 * df.iloc[0].gene_diversity:
            print(f"\n⚠️  Warning: Genetic diversity has collapsed significantly")

    print("\n" + "=" * 60)


def main():
    """Main entry point for the visualization script."""
    # Default path - look in output/serialization directory
    default_path = Path(__file__).parent.parent / "output" / "serialization" / "metrics.csv"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    elif default_path.exists():
        csv_path = str(default_path)
    else:
        csv_path = "metrics.csv"

    if not Path(csv_path).exists():
        print(f"Error: Could not find metrics file at '{csv_path}'")
        print("Usage: python plot_metrics.py [path/to/metrics.csv]")
        sys.exit(1)

    print(f"Loading metrics from: {csv_path}")
    df = load_metrics(csv_path)

    print_summary(df)

    output_dir = Path(csv_path).parent / "output" / "png"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evolution_progress.png"
    plot_all_metrics(df, str(output_path))


if __name__ == "__main__":
    main()
