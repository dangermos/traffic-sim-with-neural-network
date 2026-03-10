# CLAUDE.md

Guide for AI assistants working on this codebase.

## Project Overview

Neuroevolution-based traffic simulation where neural network-controlled cars learn to navigate road networks through evolutionary training. Uses island-based parallelism with optional Population-Based Training (PBT) for hyperparameter adaptation.

**Language**: Rust (edition 2024) with Python visualization scripts.

## Repository Structure

```
traffic-sim-with-neural-network/
├── Cargo.toml              # Workspace root (resolver = "3")
├── .config                 # Runtime training parameters (INI-style key=value)
├── run_evolution.sh        # Batch runner for multiple evolution runs
├── neural/src/lib.rs       # Feed-forward neural network (Network, Layer, Neuron)
├── genetics/src/lib.rs     # Evolution: fitness, mutation, selection, population
├── traffic/src/
│   ├── lib.rs              # Re-exports all submodules
│   ├── cars.rs             # Car state machine, sensors, AI control
│   ├── road.rs             # Road/node graph, spatial grid, rendering
│   ├── simulation.rs       # Simulation loop, physics, collisions
│   └── levels.rs           # Training levels (easy → extreme)
├── evolution_main/src/
│   ├── main.rs             # Training binary: config parsing, island evolution loop
│   ├── metrics.rs          # Generation metrics collection
│   └── pbt_metrics.rs      # PBT-specific metrics tracking
├── draw_main/src/main.rs   # Real-time visualization/playback binary
└── visualization/
    ├── requirements.txt    # Python deps (pandas, matplotlib, numpy, plotly)
    └── visualize.py        # Chart generation from metrics CSV
```

## Crate Dependency Graph

```
evolution_main ──→ genetics ──→ neural
       │               │
       └───────────────→ traffic (cars, roads, simulation, levels)

draw_main ──→ traffic, neural
```

- **neural**: Pure NN data structures, no simulation dependencies
- **traffic**: Simulation engine, independent of training logic
- **genetics**: Evolutionary algorithms, depends on neural + traffic
- **evolution_main**: Training orchestration, integrates all crates
- **draw_main**: Visualization, loads trained networks

## Build and Run Commands

```bash
# Build
cargo build                              # Debug build
cargo build --release -p evolution_main  # Release build (training)

# Train
cargo run -p evolution_main              # Uses .config defaults
cargo run --release -p evolution_main    # Optimized training

# Visualize trained network
cargo run -p draw_main

# Batch training
./run_evolution.sh

# Tests
cargo test --lib neural                  # Unit tests (neural crate only)

# Python visualization (after training)
python3 -m pip install -r visualization/requirements.txt
python3 visualization/visualize.py
```

## Configuration

All training parameters are in `.config` (INI-style, `key=value`). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `level` | nightmare | Training map name |
| `topology` | 5,8,2 | Network layer sizes (inputs, hidden..., outputs) |
| `epochs` | 100 | Number of generations |
| `population_size` | 400 | Total population (split across islands) |
| `num_islands` | 4 | Parallel island populations |
| `mutation_rate` | 0.7 | Gene mutation probability |
| `mutation_strength` | 0.3 | Max mutation magnitude |
| `pbt_enabled` | false | Population-Based Training toggle |

Available levels: `straight_line`, `straight_road`, `level1`, `level2`, `level3`, `test_sensors`, `overnight`, `nightmare`, `nightmare_extreme`

## Architecture

### Neural Network

- 5 inputs: distance_to_destination, goal_alignment, heading_error, on-roadness, obstruction_score
- 2 outputs: throttle [-1,1], steering [-1,1]
- Configurable hidden layers (default: single layer of 8 neurons)
- Tanh activation throughout
- `propagate_into()` uses pre-allocated buffers (zero-copy optimization)

### Evolution

- Island-based parallelism via rayon
- Tournament selection, mutation-only (no crossover yet)
- Elitism preserves best N individuals
- Migration exchanges individuals between islands every N epochs
- Optional PBT auto-tunes hyperparameters

### Simulation

- `SpatialGrid` for O(n) collision detection (cell size 150.0)
- `CarState` enum drives car behavior state machine
- Catmull-Rom curves for smooth road rendering
- Fitness = distance progress + destination bonus + speed bonus - penalties

## Code Conventions

- **Structs/Enums**: PascalCase (`CarWorld`, `TrainingLevel`)
- **Functions**: snake_case (`fitness_with_config`, `evolve_generation`)
- **Constants**: UPPER_SNAKE_CASE (`FLOAT_PRECISION`, `INPUTS`)
- **Types**: `f32` throughout (not `f64`), `Vec<f32>` for gene vectors
- **RNG**: Generic over `Rng` trait, `ChaCha8Rng` with seed 42 for reproducibility
- **Error handling**: Assertions and `panic!()` for invariants; graceful defaults for IO
- **Traits**: Extension traits for cross-crate functionality (e.g., `CarFitnessExt`)
- Standard rustfmt formatting (4-space indent)

## Testing

Tests use seeded RNG for deterministic results:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_example() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // ...
    }
}
```

Currently only `neural` crate has unit tests. Integration testing done via full evolution runs.

## Output Files

Training outputs go to `output/serialization/`:
- `best_fitness.json` - Best fitness per generation
- `metrics.csv` - Full generation metrics (used by visualize.py)
- Checkpoint files (bincode-encoded networks)
- `graphs/` - Generated PNG and interactive HTML charts

## Key Development Workflows

**Adding a new training level**: Create a function in `traffic/src/levels.rs` returning `Simulation`, add a variant to `TrainingLevel` enum in `evolution_main/src/main.rs`, and add string matching in `TrainingLevel::from_str()`.

**Changing network topology**: Edit `topology=` in `.config` (e.g., `5,12,8,2` for two hidden layers).

**Tuning fitness**: Adjust fitness parameters in `.config` or modify `fitness_with_config()` in `genetics/src/lib.rs`.

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| macroquad 0.4 | Graphics/rendering (draw_main) |
| rayon | Parallel island evolution and fitness eval |
| rand + rand_chacha | Seeded RNG |
| serde + bincode | Network checkpoint serialization |
| serde_json | Metrics JSON output |
| indicatif | Training progress bars |
