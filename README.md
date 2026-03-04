# Traffic Sim with Neural Network

Traffic simulation with neural-network controlled cars and evolutionary training.

## Crates
- `traffic`: simulation engine, cars, roads, levels.
- `neural`: simple feed-forward neural network.
- `genetics`: evolution, fitness, population utilities.
- `evolution_main`: training binary.
- `draw_main`: visualization binary.

## Quickstart
- Train: `cargo run -p evolution_main`
- Visualize: `cargo run -p draw_main`
- Batch runner: `./run_evolution.sh`

## Configuration
Edit the `.config` file in the repo root to override defaults. Example:

```ini
level=overnight
topology=5,8,2
epochs=50
max_frames=5000
num_islands=4
pbt_enabled=false
```

## Visualization
The training run writes metrics to `output/serialization`. Generate graphs with:

```bash
python3 -m pip install -r visualization/requirements.txt
python3 visualization/visualize.py
```

Graphs are saved to `output/serialization/graphs/` which also contain accompanying html files to visualize data in 3D.  

## Roadmap

- [ ] Crossover/recombination between individuals
- [ ] Speciation to preserve diverse behaviors
- [ ] Multi-objective fitness (speed vs safety tradeoffs)
- [ ] Traffic signals and right-of-way rules
- [ ] Pedestrians and obstacles
- [ ] More road types (roundabouts, highway merges)
- [ ] Export trained networks for use in other projects
- [ ] NEAT or other topology-evolving algorithms
- [ ] Curriculum learning (start easy, increase difficulty)
- [ ] Real-time training visualization

## License
MIT. See `LICENSE`.
