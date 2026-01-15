#!/usr/bin/env python3
"""
Interactive 3D Visualization for Evolution Metrics using Plotly

Usage:
    python plot_3d_interactive.py [metrics.csv]

Generates interactive HTML files you can rotate/zoom in your browser:
    - output/html/3d_fitness_interactive.html
    - output/html/3d_behavior_interactive.html
    - output/html/3d_diversity_interactive.html
    - output/html/3d_dashboard_interactive.html
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_metrics(path: str) -> pd.DataFrame:
    """Load metrics CSV file into a DataFrame."""
    return pd.read_csv(path)


def plot_fitness_landscape(df: pd.DataFrame, output_path: str) -> None:
    """Interactive 3D: Generation × Mean Fitness × Best Fitness"""
    fig = go.Figure()

    # Main evolution path
    fig.add_trace(go.Scatter3d(
        x=df['generation'],
        y=df['mean_fitness'],
        z=df['best_fitness'],
        mode='lines+markers',
        marker=dict(
            size=4,
            color=df['generation'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Generation')
        ),
        line=dict(color='blue', width=2),
        name='Evolution Path',
        hovertemplate='Gen: %{x}<br>Mean: %{y:.2f}<br>Best: %{z:.2f}<extra></extra>'
    ))

    # Mark start point
    fig.add_trace(go.Scatter3d(
        x=[df['generation'].iloc[0]],
        y=[df['mean_fitness'].iloc[0]],
        z=[df['best_fitness'].iloc[0]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='diamond'),
        name='Start'
    ))

    # Mark end point
    fig.add_trace(go.Scatter3d(
        x=[df['generation'].iloc[-1]],
        y=[df['mean_fitness'].iloc[-1]],
        z=[df['best_fitness'].iloc[-1]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='End'
    ))

    # Mark peak
    peak_idx = df['best_fitness'].idxmax()
    fig.add_trace(go.Scatter3d(
        x=[df.loc[peak_idx, 'generation']],
        y=[df.loc[peak_idx, 'mean_fitness']],
        z=[df.loc[peak_idx, 'best_fitness']],
        mode='markers',
        marker=dict(size=14, color='gold', symbol='diamond'),
        name=f'Peak (Gen {int(df.loc[peak_idx, "generation"])})'
    ))

    fig.update_layout(
        title=dict(text='3D Fitness Landscape', font=dict(size=20)),
        scene=dict(
            xaxis_title='Generation',
            yaxis_title='Mean Fitness',
            zaxis_title='Best Fitness',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(x=0.02, y=0.98)
    )

    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


def plot_behavior_space(df: pd.DataFrame, output_path: str) -> None:
    """Interactive 3D: Progress × Efficiency × Fitness"""
    efficiency = df['best_efficiency'] if 'best_efficiency' in df.columns else df['mean_efficiency']

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=df['max_progress'],
        y=efficiency,
        z=df['best_fitness'],
        mode='lines+markers',
        marker=dict(
            size=5,
            color=df['generation'],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title='Generation')
        ),
        line=dict(color='rgba(100,100,100,0.4)', width=2),
        name='Evolution',
        hovertemplate='Progress: %{x:.1f}<br>Efficiency: %{y:.3f}<br>Fitness: %{z:.2f}<br><extra></extra>'
    ))

    # Mark start
    fig.add_trace(go.Scatter3d(
        x=[df['max_progress'].iloc[0]],
        y=[efficiency.iloc[0]],
        z=[df['best_fitness'].iloc[0]],
        mode='markers',
        marker=dict(size=10, color='lime', symbol='diamond'),
        name='Start'
    ))

    # Mark end
    fig.add_trace(go.Scatter3d(
        x=[df['max_progress'].iloc[-1]],
        y=[efficiency.iloc[-1]],
        z=[df['best_fitness'].iloc[-1]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='End'
    ))

    fig.update_layout(
        title=dict(text='3D Behavior Space', font=dict(size=20)),
        scene=dict(
            xaxis_title='Max Progress',
            yaxis_title='Efficiency',
            zaxis_title='Best Fitness',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(x=0.02, y=0.98)
    )

    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


def plot_diversity_fitness(df: pd.DataFrame, output_path: str) -> None:
    """Interactive 3D: Generation × Diversity × Fitness"""
    fig = go.Figure()

    # Best fitness surface
    fig.add_trace(go.Scatter3d(
        x=df['generation'],
        y=df['gene_diversity'],
        z=df['best_fitness'],
        mode='lines+markers',
        marker=dict(
            size=5,
            color=df['best_fitness'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Fitness')
        ),
        line=dict(color='green', width=3),
        name='Best Fitness',
        hovertemplate='Gen: %{x}<br>Diversity: %{y:.4f}<br>Best: %{z:.2f}<extra></extra>'
    ))

    # Mean fitness
    fig.add_trace(go.Scatter3d(
        x=df['generation'],
        y=df['gene_diversity'],
        z=df['mean_fitness'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Mean Fitness',
        hovertemplate='Gen: %{x}<br>Diversity: %{y:.4f}<br>Mean: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text='Diversity vs Fitness Over Time', font=dict(size=20)),
        scene=dict(
            xaxis_title='Generation',
            yaxis_title='Genetic Diversity',
            zaxis_title='Fitness',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(x=0.02, y=0.98)
    )

    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


def plot_dashboard(df: pd.DataFrame, output_path: str) -> None:
    """Combined interactive dashboard with multiple 3D plots"""
    efficiency = df['best_efficiency'] if 'best_efficiency' in df.columns else df['mean_efficiency']

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'type': 'scatter3d'}, {'type': 'xy'}]
        ],
        subplot_titles=(
            'Fitness Landscape',
            'Behavior Space',
            'Diversity vs Fitness',
            'Fitness Over Time'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # 1. Fitness Landscape
    fig.add_trace(go.Scatter3d(
        x=df['generation'],
        y=df['mean_fitness'],
        z=df['best_fitness'],
        mode='lines+markers',
        marker=dict(size=3, color=df['generation'], colorscale='Viridis'),
        line=dict(width=2),
        name='Fitness',
        showlegend=False
    ), row=1, col=1)

    # 2. Behavior Space
    fig.add_trace(go.Scatter3d(
        x=df['max_progress'],
        y=efficiency,
        z=df['best_fitness'],
        mode='lines+markers',
        marker=dict(size=3, color=df['generation'], colorscale='Plasma'),
        line=dict(width=1),
        name='Behavior',
        showlegend=False
    ), row=1, col=2)

    # 3. Diversity vs Fitness
    fig.add_trace(go.Scatter3d(
        x=df['generation'],
        y=df['gene_diversity'],
        z=df['best_fitness'],
        mode='lines+markers',
        marker=dict(size=3, color=df['best_fitness'], colorscale='RdYlGn'),
        line=dict(width=2),
        name='Diversity',
        showlegend=False
    ), row=2, col=1)

    # 4. 2D Fitness over time
    fig.add_trace(go.Scatter(
        x=df['generation'],
        y=df['best_fitness'],
        mode='lines',
        line=dict(color='green', width=2),
        name='Best',
        showlegend=True
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=df['generation'],
        y=df['mean_fitness'],
        mode='lines',
        line=dict(color='blue', width=1),
        name='Mean',
        showlegend=True
    ), row=2, col=2)

    # Update layout
    fig.update_layout(
        height=900,
        title=dict(
            text='Interactive Evolution Dashboard',
            font=dict(size=24),
            x=0.5
        ),
        showlegend=True,
        legend=dict(x=0.85, y=0.15)
    )

    # Update 3D scenes
    fig.update_scenes(
        xaxis_title_font=dict(size=10),
        yaxis_title_font=dict(size=10),
        zaxis_title_font=dict(size=10)
    )

    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


def print_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 50)
    print("  INTERACTIVE 3D VISUALIZATION")
    print("=" * 50)
    print("\nOpen the .html files in your web browser to:")
    print("  • Drag to rotate the view")
    print("  • Scroll to zoom in/out")
    print("  • Hover over points for details")
    print("  • Double-click to reset view")
    print("  • Use toolbar to save as PNG")
    print("=" * 50 + "\n")


def main():
    """Main entry point."""
    # Find metrics file
    default_path = Path(__file__).parent.parent / "output" / "serialization" / "metrics.csv"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    elif default_path.exists():
        csv_path = str(default_path)
    else:
        csv_path = "metrics.csv"

    if not Path(csv_path).exists():
        print(f"Error: Could not find '{csv_path}'")
        print("Usage: python plot_3d_interactive.py [path/to/metrics.csv]")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    df = load_metrics(csv_path)
    output_dir = Path(csv_path).parent / "output" / "html"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating interactive 3D plots...")
    plot_fitness_landscape(df, str(output_dir / "3d_fitness_interactive.html"))
    plot_behavior_space(df, str(output_dir / "3d_behavior_interactive.html"))
    plot_diversity_fitness(df, str(output_dir / "3d_diversity_interactive.html"))
    plot_dashboard(df, str(output_dir / "3d_dashboard_interactive.html"))

    print_instructions()
    print("✅ All interactive visualizations complete!")


if __name__ == "__main__":
    main()
