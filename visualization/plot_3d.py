#!/usr/bin/env python3
"""
3D Visualization for Evolution Metrics

Usage:
    python plot_3d.py [metrics.csv]

Generates (in output/png/):
    - 3d_fitness_landscape.png
    - 3d_behavior_space.png
    - 3d_diversity_fitness.png
    - 3d_population_dynamics.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def load_metrics(path: str = "metrics.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def plot_3d_fitness_landscape(df: pd.DataFrame, output_path: str) -> None:
    """3D: Generation × Mean Fitness × Best Fitness"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    gen = df['generation'].values
    mean_fit = df['mean_fitness'].values
    best_fit = df['best_fitness'].values

    ax.plot(gen, mean_fit, best_fit, color='blue', linewidth=2, alpha=0.7)
    scatter = ax.scatter(gen, mean_fit, best_fit, c=gen, cmap='viridis', s=50, alpha=0.8)

    # Mark start, end, peak
    ax.scatter([gen[0]], [mean_fit[0]], [best_fit[0]], color='green', s=200, marker='^', label='Start')
    ax.scatter([gen[-1]], [mean_fit[-1]], [best_fit[-1]], color='red', s=200, marker='*', label='End')
    peak_idx = np.argmax(best_fit)
    ax.scatter([gen[peak_idx]], [mean_fit[peak_idx]], [best_fit[peak_idx]],
               color='gold', s=300, marker='D', label=f'Peak (Gen {gen[peak_idx]})')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Fitness')
    ax.set_zlabel('Best Fitness')
    ax.set_title('3D Fitness Landscape')
    plt.colorbar(scatter, ax=ax, label='Generation', shrink=0.6)
    ax.legend()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_3d_behavior_space(df: pd.DataFrame, output_path: str) -> None:
    """3D: Progress × Efficiency × Fitness"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    gen = df['generation'].values
    progress = df['max_progress'].values
    efficiency = df['best_efficiency'].values if 'best_efficiency' in df.columns else df['mean_efficiency'].values
    fitness = df['best_fitness'].values

    scatter = ax.scatter(progress, efficiency, fitness, c=gen, cmap='plasma', s=60, alpha=0.8)
    ax.plot(progress, efficiency, fitness, 'k-', alpha=0.3, linewidth=1)

    ax.set_xlabel('Max Progress')
    ax.set_ylabel('Best Efficiency')
    ax.set_zlabel('Best Fitness')
    ax.set_title('3D Behavior Space')
    plt.colorbar(scatter, ax=ax, label='Generation', shrink=0.6)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_3d_diversity_fitness(df: pd.DataFrame, output_path: str) -> None:
    """3D: Generation × Diversity × Fitness"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    gen = df['generation'].values
    diversity = df['gene_diversity'].values
    best = df['best_fitness'].values
    mean = df['mean_fitness'].values

    ax.plot(gen, diversity, best, 'g-', linewidth=2.5, label='Best Fitness')
    ax.plot(gen, diversity, mean, 'b-', linewidth=2, label='Mean Fitness', alpha=0.7)
    scatter = ax.scatter(gen, diversity, best, c=best, cmap='RdYlGn', s=40, alpha=0.7)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Genetic Diversity')
    ax.set_zlabel('Fitness')
    ax.set_title('Diversity vs Fitness Over Time')
    plt.colorbar(scatter, ax=ax, label='Best Fitness', shrink=0.6)
    ax.legend()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_3d_population_dynamics(df: pd.DataFrame, output_path: str) -> None:
    """3D: Generation × Car Counts × Fitness"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    gen = df['generation'].values
    fitness = df['best_fitness'].values

    crashed = df['num_crashed'].values if 'num_crashed' in df.columns else np.zeros(len(df))
    stagnant = df['num_stagnant'].values if 'num_stagnant' in df.columns else np.zeros(len(df))
    reached = df['num_reached'].values if 'num_reached' in df.columns else np.zeros(len(df))
    active = df['num_active'].values if 'num_active' in df.columns else np.zeros(len(df))

    ax.plot(gen, reached, fitness, 'g-', linewidth=3, label='Reached', alpha=0.9)
    ax.plot(gen, active, fitness, 'b-', linewidth=3, label='Active', alpha=0.8)
    ax.plot(gen, stagnant, fitness, 'y-', linewidth=3, label='Stagnant', alpha=0.8)
    ax.plot(gen, crashed, fitness, 'r-', linewidth=3, label='Crashed', alpha=0.8)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Car Count')
    ax.set_zlabel('Best Fitness')
    ax.set_title('3D Population Dynamics')
    ax.legend()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    default_path = Path(__file__).parent.parent / "output" / "serialization" / "metrics.csv"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    elif default_path.exists():
        csv_path = str(default_path)
    else:
        csv_path = "output/serialization/metrics.csv"

    if not Path(csv_path).exists():
        print(f"Error: Could not find '{csv_path}'")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    df = load_metrics(csv_path)

    # Output to output/png folder
    output_dir = Path(csv_path).parent / "output" / "png"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating 3D plots to {output_dir}...")
    plot_3d_fitness_landscape(df, str(output_dir / "3d_fitness_landscape.png"))
    plot_3d_behavior_space(df, str(output_dir / "3d_behavior_space.png"))
    plot_3d_diversity_fitness(df, str(output_dir / "3d_diversity_fitness.png"))
    plot_3d_population_dynamics(df, str(output_dir / "3d_population_dynamics.png"))

    print("\n✅ 3D visualizations complete!")


if __name__ == "__main__":
    main()
