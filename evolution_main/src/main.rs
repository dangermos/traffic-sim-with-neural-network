use ::rand::SeedableRng;
use bincode::{deserialize_from, serialize};
use genetics::{
    Individual, Population, evolve_generation, fitness, make_sim_from_population_with_grid,
};
use indicatif::{ProgressBar, ProgressStyle};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::{Value, from_reader, to_writer};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use traffic::levels::overnight_training;
use traffic::road::RoadGrid;

mod metrics;
use metrics::{GenerationMetrics, MetricsWriter};

#[derive(Debug, Clone)]
struct TrainingConfig {
    rng_seed: u64,
    epochs: usize,
    max_frames: usize,
    elitism: usize,
    mutation_rate: f32,
    mutation_strength: f32,
    enable_early_exit: bool,
    stall_distance: f32,
    stall_frames: usize,
    print_every: usize,
    checkpoint_every: usize,
    // New: Island-based parallelism
    num_islands: usize,
    migration_interval: usize,
    migration_count: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            rng_seed: 42,
            epochs: 50,
            max_frames: 5000,
            elitism: 10,
            mutation_rate: 0.3,
            mutation_strength: 0.3,
            enable_early_exit: false,
            stall_distance: 5.0,
            stall_frames: 200,
            print_every: 10,
            checkpoint_every: 100,
            // Default to 4 islands for parallel evolution
            num_islands: 4,
            migration_interval: 5,  // Migrate every 5 generations
            migration_count: 2,     // Migrate top 2 individuals between islands
        }
    }
}

impl TrainingConfig {
    fn load(path: &Path) -> Self {
        let mut config = Self::default();
        let contents = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(err) => {
                if err.kind() != std::io::ErrorKind::NotFound {
                    eprintln!("Failed to read {:?}: {err}. Using defaults.", path);
                }
                return config;
            }
        };

        for (idx, raw_line) in contents.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }

            let mut parts = line.splitn(2, '=');
            let key = parts.next().unwrap_or("").trim().to_ascii_lowercase();
            let value = parts.next().unwrap_or("").trim();
            if key.is_empty() || value.is_empty() {
                eprintln!("Skipping invalid config entry at line {}", line_no);
                continue;
            }

            match key.as_str() {
                "rng_seed" => apply_u64(&mut config.rng_seed, value, line_no, "rng_seed"),
                "epochs" => apply_usize(&mut config.epochs, value, line_no, "epochs"),
                "max_frames" => apply_usize(&mut config.max_frames, value, line_no, "max_frames"),
                "elitism" => apply_usize(&mut config.elitism, value, line_no, "elitism"),
                "mutation_rate" => {
                    apply_f32(&mut config.mutation_rate, value, line_no, "mutation_rate")
                }
                "mutation_strength" => {
                    apply_f32(&mut config.mutation_strength, value, line_no, "mutation_strength")
                }
                "enable_early_exit" => apply_bool(
                    &mut config.enable_early_exit,
                    value,
                    line_no,
                    "enable_early_exit",
                ),
                "stall_distance" => {
                    apply_f32(&mut config.stall_distance, value, line_no, "stall_distance")
                }
                "stall_frames" => {
                    apply_usize(&mut config.stall_frames, value, line_no, "stall_frames")
                }
                "print_every" => {
                    apply_usize(&mut config.print_every, value, line_no, "print_every")
                }
                "checkpoint_every" => apply_usize(
                    &mut config.checkpoint_every,
                    value,
                    line_no,
                    "checkpoint_every",
                ),
                "num_islands" => {
                    apply_usize(&mut config.num_islands, value, line_no, "num_islands")
                }
                "migration_interval" => {
                    apply_usize(&mut config.migration_interval, value, line_no, "migration_interval")
                }
                "migration_count" => {
                    apply_usize(&mut config.migration_count, value, line_no, "migration_count")
                }
                _ => eprintln!("Unknown config key '{}' at line {}", key, line_no),
            }
        }

        println!("Loaded training config from {:?}", path);
        config
    }
}

fn apply_usize(target: &mut usize, value: &str, line_no: usize, key: &str) {
    match value.parse::<usize>() {
        Ok(v) => *target = v,
        Err(_) => eprintln!("Invalid usize for {} at line {}", key, line_no),
    }
}

fn apply_u64(target: &mut u64, value: &str, line_no: usize, key: &str) {
    match value.parse::<u64>() {
        Ok(v) => *target = v,
        Err(_) => eprintln!("Invalid u64 for {} at line {}", key, line_no),
    }
}

fn apply_f32(target: &mut f32, value: &str, line_no: usize, key: &str) {
    match value.parse::<f32>() {
        Ok(v) => *target = v,
        Err(_) => eprintln!("Invalid f32 for {} at line {}", key, line_no),
    }
}

fn apply_bool(target: &mut bool, value: &str, line_no: usize, key: &str) {
    let normalized = value.trim().to_ascii_lowercase();
    let parsed = match normalized.as_str() {
        "true" | "1" | "yes" | "y" | "on" => Some(true),
        "false" | "0" | "no" | "n" | "off" => Some(false),
        _ => None,
    };

    match parsed {
        Some(v) => *target = v,
        None => eprintln!("Invalid bool for {} at line {}", key, line_no),
    }
}

fn load_best_history(path: &PathBuf) -> Vec<f32> {
    std::fs::File::open(path)
        .ok()
        .and_then(|f| {
            let reader = BufReader::new(f);
            match from_reader::<_, Value>(reader) {
                Ok(Value::Array(arr)) => Some(
                    arr.into_iter()
                        .filter_map(|v| v.as_f64().map(|n| n as f32))
                        .collect(),
                ),
                Ok(Value::Number(n)) => n.as_f64().map(|n| vec![n as f32]),
                Ok(_) => Some(Vec::new()),
                Err(e) => {
                    eprintln!("Could not parse {:?} as JSON ({e}); starting fresh.", path);
                    None
                }
            }
        })
        .unwrap_or_default()
}

fn write_best_history(path: &PathBuf, history: &[f32]) {
    match std::fs::File::create(path) {
        Ok(file) => {
            let writer = BufWriter::new(file);
            if let Err(e) = to_writer(writer, history) {
                eprintln!("Failed to write {:?}: {e}", path);
            }
        }
        Err(e) => eprintln!("Could not create {:?}: {e}", path),
    }
}

/// An island holds a sub-population that evolves independently
struct Island {
    population: Population,
    rng: ChaCha8Rng,
}

impl Island {
    fn new(individuals: Vec<Individual>, seed: u64) -> Self {
        Self {
            population: Population {
                individuals,
                generation: 0,
            },
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Run one generation: simulate, evaluate fitness, evolve
    /// Returns the cars from the simulation for metrics collection
    fn run_generation(
        &mut self,
        road_grid: &RoadGrid,
        max_frames: usize,
        elitism: usize,
        mutation_rate: f32,
        mutation_strength: f32,
    ) -> Vec<traffic::cars::Car> {
        // Create simulation for this island's population
        let mut sim = make_sim_from_population_with_grid(&self.population, road_grid, &mut self.rng);

        // Run simulation
        for _ in 0..max_frames {
            sim.update(false);
        }

        // Evaluate fitness
        for (ind, car) in self.population.individuals.iter_mut().zip(sim.cars.cars.iter()) {
            ind.fitness = fitness(car);
        }

        // Store cars for metrics before evolving
        let cars = sim.cars.cars.clone();

        // Evolve to next generation
        self.population = evolve_generation(
            &self.population,
            elitism,
            mutation_rate,
            mutation_strength,
            &mut self.rng,
        );

        cars
    }

    /// Get the top N individuals (sorted by fitness descending)
    fn get_top_individuals(&self, n: usize) -> Vec<Individual> {
        let mut sorted = self.population.individuals.clone();
        sorted.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }

    /// Replace the worst N individuals with migrants
    fn receive_migrants(&mut self, migrants: Vec<Individual>) {
        let n = migrants.len();
        if n == 0 || self.population.individuals.is_empty() {
            return;
        }

        // Sort by fitness ascending (worst first)
        self.population.individuals.sort_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Replace worst individuals with migrants
        for (i, migrant) in migrants.into_iter().enumerate() {
            if i < self.population.individuals.len() {
                self.population.individuals[i] = migrant;
            }
        }
    }

    fn best_fitness(&self) -> f32 {
        self.population
            .individuals
            .iter()
            .map(|i| i.fitness)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    fn mean_fitness(&self) -> f32 {
        if self.population.individuals.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.population.individuals.iter().map(|i| i.fitness).sum();
        sum / self.population.individuals.len() as f32
    }
}

/// Migrate top individuals between islands in a ring topology
fn migrate_between_islands(islands: &mut [Island], migration_count: usize) {
    if islands.len() < 2 || migration_count == 0 {
        return;
    }

    // Collect migrants from each island
    let migrants: Vec<Vec<Individual>> = islands
        .iter()
        .map(|island| island.get_top_individuals(migration_count))
        .collect();

    // Send migrants in a ring: island 0 -> 1, 1 -> 2, ..., N-1 -> 0
    for i in 0..islands.len() {
        let next = (i + 1) % islands.len();
        islands[next].receive_migrants(migrants[i].clone());
    }
}

fn main() {
    let config_path = Path::new(".config");
    let config = TrainingConfig::load(config_path);

    let mut rng = ChaCha8Rng::seed_from_u64(config.rng_seed);

    // Create initial simulation to get population size and road grid
    let sim = overnight_training(&mut rng);
    let base_road_grid = sim.roads.clone();
    let total_pop_size = sim.cars.cars.len();

    println!(
        "=== Parallel Island Evolution ===
Islands:          {}
Population/Island: ~{}
Total Population: {}
Roads:            {}
Max Frames:       {}
Migration Every:  {} generations
Migration Count:  {} individuals
",
        config.num_islands,
        total_pop_size / config.num_islands.max(1),
        total_pop_size,
        base_road_grid.roads.len(),
        config.max_frames,
        config.migration_interval,
        config.migration_count,
    );

    // Load or create initial population
    let checkpoint_path = PathBuf::from("output/serialization/individuals.bin");
    let best_path = PathBuf::from("output/serialization/best_fitness.json");
    let metrics_path = PathBuf::from("output/serialization/metrics.csv");

    // Ensure output directories exist
    std::fs::create_dir_all("output/serialization").expect("Failed to create output/serialization directory");

    let initial_individuals: Vec<Individual> = sim
        .cars
        .cars
        .iter()
        .map(|car| {
            let genes = car.network.to_genes();
            Individual {
                genes,
                fitness: 0.0,
            }
        })
        .collect();

    let loaded_individuals: Vec<Individual> = if let Ok(file) = std::fs::File::open(&checkpoint_path) {
        let reader = std::io::BufReader::new(file);
        deserialize_from(reader).unwrap_or_else(|e| {
            eprintln!("Could not deserialize checkpoint ({e}); starting fresh.");
            initial_individuals.clone()
        })
    } else {
        eprintln!("No checkpoint found; starting fresh.");
        initial_individuals.clone()
    };

    let all_individuals = if loaded_individuals.len() == total_pop_size {
        loaded_individuals
    } else {
        eprintln!(
            "Checkpoint size {} != expected {}; starting fresh.",
            loaded_individuals.len(),
            total_pop_size
        );
        initial_individuals
    };

    // Split population across islands
    let num_islands = config.num_islands.max(1);
    let individuals_per_island = total_pop_size / num_islands;
    let mut islands: Vec<Island> = Vec::with_capacity(num_islands);

    for i in 0..num_islands {
        let start = i * individuals_per_island;
        let end = if i == num_islands - 1 {
            total_pop_size // Last island gets any remainder
        } else {
            start + individuals_per_island
        };

        let island_individuals = all_individuals[start..end].to_vec();
        let island_seed = config.rng_seed.wrapping_add(i as u64 * 1000);
        islands.push(Island::new(island_individuals, island_seed));
    }

    let mut best_history = load_best_history(&best_path);
    let mut global_best_fitness = best_history
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    // Progress bar
    let pb = ProgressBar::new(config.epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    // Metrics writer
    let mut metrics_writer =
        MetricsWriter::new(&metrics_path).expect("Failed to create metrics file");

    // Clone config values for use in parallel closure
    let max_frames = config.max_frames;
    let elitism_per_island = (config.elitism / num_islands).max(1);
    let mutation_rate = config.mutation_rate;
    let mutation_strength = config.mutation_strength;

    for generation in 0..config.epochs {
        // ========================================
        // PARALLEL EVOLUTION: Run all islands concurrently
        // ========================================
        let road_grid_ref = &base_road_grid;

        // Collect cars from each island's simulation for metrics
        let island_cars: Vec<Vec<traffic::cars::Car>> = islands
            .par_iter_mut()
            .map(|island| {
                island.run_generation(
                    road_grid_ref,
                    max_frames,
                    elitism_per_island,
                    mutation_rate,
                    mutation_strength,
                )
            })
            .collect();

        // ========================================
        // MIGRATION: Exchange top individuals between islands
        // ========================================
        if config.migration_interval > 0 && generation % config.migration_interval == 0 && generation > 0 {
            migrate_between_islands(&mut islands, config.migration_count);
        }

        // ========================================
        // COLLECT METRICS: Gather stats from all islands
        // ========================================
        let island_best: Vec<f32> = islands.iter().map(|i| i.best_fitness()).collect();
        let island_mean: Vec<f32> = islands.iter().map(|i| i.mean_fitness()).collect();

        let gen_best = island_best.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let gen_mean = island_mean.iter().sum::<f32>() / num_islands as f32;

        global_best_fitness = global_best_fitness.max(gen_best);

        // Create combined population for metrics (merge all islands)
        let combined_individuals: Vec<Individual> = islands
            .iter()
            .flat_map(|i| i.population.individuals.clone())
            .collect();

        let combined_population = Population {
            individuals: combined_individuals.clone(),
            generation: generation as u32,
        };

        // Flatten all island cars for metrics (these are the ACTUAL simulated cars)
        let all_cars: Vec<traffic::cars::Car> = island_cars.into_iter().flatten().collect();

        let metrics = GenerationMetrics::compute(
            generation as u32,
            &combined_population,
            &all_cars,
        );

        if let Err(e) = metrics_writer.write_metrics(&metrics) {
            eprintln!("Failed to write metrics: {e}");
        }

        // Print progress
        if config.print_every > 0 && generation % config.print_every == 0 {
            println!(
                "Gen {:4} | Best: {:8.2} | Mean: {:8.2} | Islands: [{}]",
                generation,
                gen_best,
                gen_mean,
                island_best
                    .iter()
                    .map(|f| format!("{:.1}", f))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        pb.set_message(format!("Best: {:.2}", global_best_fitness));
        pb.inc(1);

        // Checkpoint
        if config.checkpoint_every > 0
            && generation % config.checkpoint_every == 0
            && generation > 0
        {
            let bytes = serialize(&combined_individuals).expect("Could not serialize");
            std::fs::write(&checkpoint_path, &bytes).expect("Could not write checkpoint");
            best_history.push(global_best_fitness);
            write_best_history(&best_path, &best_history);
        }
    }

    // Final save
    let final_individuals: Vec<Individual> = islands
        .iter()
        .flat_map(|i| i.population.individuals.clone())
        .collect();

    let bytes = serialize(&final_individuals).expect("Could not serialize");
    std::fs::write(&checkpoint_path, &bytes).expect("Could not write checkpoint");
    println!(
        "\nWrote {} individuals to {:?} ({} bytes)",
        final_individuals.len(),
        checkpoint_path,
        bytes.len()
    );

    best_history.push(global_best_fitness);
    write_best_history(&best_path, &best_history);

    pb.finish_with_message("Evolution Complete");
    println!("\n=== Results ===");
    println!("Global Best Fitness: {:.2}", global_best_fitness);
    println!("Metrics saved to: {:?}", metrics_path);

    // Print per-island summary
    println!("\nPer-Island Final Stats:");
    for (i, island) in islands.iter().enumerate() {
        println!(
            "  Island {}: Best = {:.2}, Mean = {:.2}",
            i,
            island.best_fitness(),
            island.mean_fitness()
        );
    }
}
