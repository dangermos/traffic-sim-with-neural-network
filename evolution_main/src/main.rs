use ::rand::{Rng, SeedableRng};
use bincode::{deserialize_from, serialize};
use genetics::{
    Individual, NetworkTopology, Population, create_random_population, evolve_generation, fitness,
    make_sim_from_slice_with_topology,
};
use indicatif::{ProgressBar, ProgressStyle};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::{Value, from_reader, to_writer};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use traffic::levels::{
    build_level_1, build_level_2, build_level_3, build_straight_line_level, build_straight_road_4,
    nightmare_track, nightmare_track_extreme, overnight_training, test_sensors,
};
use traffic::road::RoadGrid;
use traffic::simulation::Simulation;

mod metrics;
mod pbt_metrics;
use metrics::{GenerationMetrics, MetricsWriter};
use pbt_metrics::{IslandHyperParamSnapshot, PbtMetricsWriter};

/// Per-island hyperparameters that can evolve via PBT
#[derive(Debug, Clone)]
struct HyperParams {
    mutation_rate: f32,
    mutation_strength: f32,
    tournament_size: usize,
}

impl HyperParams {
    fn new(mutation_rate: f32, mutation_strength: f32, tournament_size: usize) -> Self {
        Self {
            mutation_rate,
            mutation_strength,
            tournament_size,
        }
    }

    /// Perturb hyperparameters by multiplying by a random factor in [0.8, 1.2]
    fn perturb<R: Rng>(&mut self, rng: &mut R) {
        let mut perturb_factor = || rng.random_range(0.8..1.2);

        self.mutation_rate = (self.mutation_rate * perturb_factor()).clamp(0.05, 0.9);
        self.mutation_strength = (self.mutation_strength * perturb_factor()).clamp(0.05, 1.0);

        // Tournament size: occasionally bump up or down by 1
        let size_delta: i32 = rng.random_range(-1..=1);
        self.tournament_size = (self.tournament_size as i32 + size_delta).clamp(2, 7) as usize;
    }
}

/// Available training levels/maps
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainingLevel {
    // Easy levels
    StraightLine, // Simple straight road
    StraightRoad, // L-shaped road
    Level1,       // Basic crossroads
    Level2,       // Rectangle with diagonals
    Level3,       // Grid pattern

    // Medium levels
    TestSensors, // Random grid for testing
    Overnight,   // Standard city grid (default)

    // Hard levels
    Nightmare,        // Twisty track with curves
    NightmareExtreme, // Brutal track with tight spirals
}

impl TrainingLevel {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            // Easy
            "straight" | "straight_line" | "line" => Some(Self::StraightLine),
            "straight_road" | "lshape" | "l" => Some(Self::StraightRoad),
            "level1" | "1" | "basic" => Some(Self::Level1),
            "level2" | "2" | "rectangle" => Some(Self::Level2),
            "level3" | "3" => Some(Self::Level3),

            // Medium
            "test" | "test_sensors" | "sensors" | "random" => Some(Self::TestSensors),
            "overnight" | "city" | "grid" => Some(Self::Overnight),

            // Hard
            "nightmare" | "hard" | "twisty" => Some(Self::Nightmare),
            "nightmare_extreme" | "extreme" | "insane" | "brutal" => Some(Self::NightmareExtreme),

            _ => None,
        }
    }

    fn build_simulation<R: ::rand::Rng>(self, rng: &mut R) -> Simulation {
        let center = macroquad::math::vec2(960.0, 540.0);
        let screen = macroquad::math::vec2(1920.0, 1080.0);

        match self {
            Self::StraightLine => build_straight_line_level(center, 800.0, rng),
            Self::StraightRoad => build_straight_road_4(center, screen, rng),
            Self::Level1 => build_level_1(center, screen, rng),
            Self::Level2 => build_level_2(center, screen, rng),
            Self::Level3 => build_level_3(center, screen, rng),
            Self::TestSensors => test_sensors(center, screen, rng),
            Self::Overnight => overnight_training(rng),
            Self::Nightmare => nightmare_track(rng),
            Self::NightmareExtreme => nightmare_track_extreme(rng),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::StraightLine => "straight_line (easy)",
            Self::StraightRoad => "straight_road (L-shape)",
            Self::Level1 => "level1 (basic crossroads)",
            Self::Level2 => "level2 (rectangle)",
            Self::Level3 => "level3 (grid)",
            Self::TestSensors => "test_sensors (random)",
            Self::Overnight => "overnight (city grid)",
            Self::Nightmare => "nightmare (twisty)",
            Self::NightmareExtreme => "nightmare_extreme (brutal)",
        }
    }

    fn list_all() -> &'static str {
        "Available levels:\n\
         Easy:   straight_line, straight_road, level1, level2, level3\n\
         Medium: test_sensors, overnight\n\
         Hard:   nightmare, nightmare_extreme"
    }
}

#[derive(Debug, Clone)]
struct TrainingConfig {
    rng_seed: u64,
    epochs: usize,
    max_frames: usize,
    elitism: usize,
    mutation_rate: f32,
    mutation_strength: f32,
    tournament_size: usize,
    enable_early_exit: bool,
    stall_distance: f32,
    stall_frames: usize,
    print_every: usize,
    checkpoint_every: usize,
    // Island-based parallelism
    num_islands: usize,
    migration_interval: usize,
    migration_count: usize,
    // Population-Based Training (PBT)
    pbt_enabled: bool,
    pbt_interval: usize, // How often to do exploit/explore
    // Level selection
    level: TrainingLevel,
    // Neural network topology
    network_topology: NetworkTopology,
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
            tournament_size: 3,
            enable_early_exit: false,
            stall_distance: 5.0,
            stall_frames: 200,
            print_every: 10,
            checkpoint_every: 100,
            // Default to 4 islands for parallel evolution
            num_islands: 4,
            migration_interval: 5, // Migrate every 5 generations
            migration_count: 2,    // Migrate top 2 individuals between islands
            // PBT: disabled by default, enable to auto-tune hyperparams
            pbt_enabled: false,
            pbt_interval: 10, // Exploit/explore every 10 generations
            // Default level
            level: TrainingLevel::Overnight,
            // Default network topology: 5 inputs, 8 hidden, 2 outputs
            network_topology: NetworkTopology::default(),
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
                "mutation_strength" => apply_f32(
                    &mut config.mutation_strength,
                    value,
                    line_no,
                    "mutation_strength",
                ),
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
                "migration_interval" => apply_usize(
                    &mut config.migration_interval,
                    value,
                    line_no,
                    "migration_interval",
                ),
                "migration_count" => apply_usize(
                    &mut config.migration_count,
                    value,
                    line_no,
                    "migration_count",
                ),
                "tournament_size" => apply_usize(
                    &mut config.tournament_size,
                    value,
                    line_no,
                    "tournament_size",
                ),
                "pbt_enabled" => apply_bool(&mut config.pbt_enabled, value, line_no, "pbt_enabled"),
                "pbt_interval" => {
                    apply_usize(&mut config.pbt_interval, value, line_no, "pbt_interval")
                }
                "level" | "map" | "track" => {
                    if let Some(lvl) = TrainingLevel::from_str(value) {
                        config.level = lvl;
                    } else {
                        eprintln!(
                            "Unknown level '{}' at line {}.\n{}",
                            value,
                            line_no,
                            TrainingLevel::list_all()
                        );
                    }
                }
                "topology" | "network" | "layers" | "network_topology" => {
                    if let Some(topo) = NetworkTopology::from_str(value) {
                        config.network_topology = topo;
                    } else {
                        eprintln!(
                            "Invalid network topology '{}' at line {}. Format: 5,8,2 or 5,12,8,2",
                            value, line_no
                        );
                    }
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
/// With PBT, each island has its own hyperparameters that can evolve
struct Island {
    population: Population,
    rng: ChaCha8Rng,
    hyperparams: HyperParams,
    /// Cached performance metric for PBT ranking
    recent_best_fitness: f32,
}

impl Island {
    fn new(individuals: Vec<Individual>, seed: u64, hyperparams: HyperParams) -> Self {
        Self {
            population: Population {
                individuals,
                generation: 0,
            },
            rng: ChaCha8Rng::seed_from_u64(seed),
            hyperparams,
            recent_best_fitness: f32::NEG_INFINITY,
        }
    }

    /// Run one generation: simulate, evaluate fitness, evolve
    /// Returns the cars from the simulation for metrics collection
    /// Uses the island's own hyperparameters (for PBT)
    fn run_generation(
        &mut self,
        road_grid: &RoadGrid,
        topology: &NetworkTopology,
        max_frames: usize,
        elitism: usize,
    ) -> Vec<traffic::cars::Car> {
        // Create simulation for this island's population
        let mut sim = make_sim_from_slice_with_topology(
            &self.population.individuals,
            road_grid,
            topology,
            &mut self.rng,
        );

        // Run simulation
        for _ in 0..max_frames {
            sim.update(false);
        }

        // Evaluate fitness
        self
            .population
            .individuals
            .par_iter_mut()
            .zip(sim.cars.cars.par_iter())
            .for_each(|(ind, car)| {
                ind.fitness = fitness(car);
            });
        // Update cached best fitness for PBT ranking
        self.recent_best_fitness = self.best_fitness();

        // Store cars for metrics before evolving
        let cars = sim.cars.cars.clone();

        // Evolve to next generation using THIS island's hyperparameters
        self.population = evolve_generation(
            &self.population,
            elitism,
            self.hyperparams.mutation_rate,
            self.hyperparams.mutation_strength,
            self.hyperparams.tournament_size,
            &mut self.rng,
        );

        cars
    }

    /// Get the top N individuals (sorted by fitness descending)
    fn get_top_individuals(&self, n: usize) -> Vec<Individual> {
        let mut sorted = self.population.individuals.clone();
        sorted.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
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

/// Population-Based Training: exploit/explore step
/// Bottom-performing islands copy hyperparameters from top-performing islands
/// then perturb them slightly to explore the hyperparameter space
fn pbt_exploit_explore(islands: &mut [Island], rng: &mut ChaCha8Rng) {
    if islands.len() < 2 {
        return;
    }

    // Rank islands by recent best fitness
    let mut rankings: Vec<(usize, f32)> = islands
        .iter()
        .enumerate()
        .map(|(i, island)| (i, island.recent_best_fitness))
        .collect();
    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Bottom 25% copies from top 25%
    let quartile = (islands.len() / 4).max(1);

    // Collect hyperparams from top performers (to avoid borrow issues)
    let top_hyperparams: Vec<HyperParams> = rankings[..quartile]
        .iter()
        .map(|(idx, _)| islands[*idx].hyperparams.clone())
        .collect();

    // Bottom performers exploit (copy) then explore (perturb)
    for (i, (bottom_idx, _)) in rankings.iter().rev().take(quartile).enumerate() {
        let source_idx = i % top_hyperparams.len();
        islands[*bottom_idx].hyperparams = top_hyperparams[source_idx].clone();
        islands[*bottom_idx].hyperparams.perturb(rng);
    }

    // Print PBT status
    println!("  [PBT] Hyperparams after exploit/explore:");
    for (i, island) in islands.iter().enumerate() {
        let hp = &island.hyperparams;
        println!(
            "    Island {}: mut_rate={:.3}, mut_str={:.3}, tourn={}",
            i, hp.mutation_rate, hp.mutation_strength, hp.tournament_size
        );
    }
}

fn main() {
    let config_path = Path::new(".config");
    let config = TrainingConfig::load(config_path);

    let mut rng = ChaCha8Rng::seed_from_u64(config.rng_seed);

    // Create initial simulation to get population size and road grid
    let sim = config.level.build_simulation(&mut rng);
    let base_road_grid = sim.roads.clone();
    let total_pop_size = sim.cars.cars.len();

    println!(
        "=== Parallel Island Evolution ===
Level:            {}
Network:          {}
Islands:          {}
Population/Island: ~{}
Total Population: {}
Roads:            {}
Max Frames:       {}
Migration Every:  {} generations
Migration Count:  {} individuals
PBT Enabled:      {}
PBT Interval:     {} generations
",
        config.level.name(),
        config.network_topology.display(),
        config.num_islands,
        total_pop_size / config.num_islands.max(1),
        total_pop_size,
        base_road_grid.roads.len(),
        config.max_frames,
        config.migration_interval,
        config.migration_count,
        config.pbt_enabled,
        config.pbt_interval,
    );

    // Load or create initial population
    let checkpoint_path = PathBuf::from("output/serialization/individuals.bin");
    let best_path = PathBuf::from("output/serialization/best_fitness.json");
    let metrics_path = PathBuf::from("output/serialization/metrics.csv");
    let pbt_metrics_path = PathBuf::from("output/serialization/pbt_metrics.csv");

    // Ensure output directories exist
    std::fs::create_dir_all("output/serialization")
        .expect("Failed to create output/serialization directory");

    // Create fresh population with configured topology
    let expected_gene_count = config.network_topology.gene_count();
    let fresh_population =
        create_random_population(total_pop_size, &config.network_topology, &mut rng);
    let initial_individuals = fresh_population.individuals;

    // Try to load checkpoint
    let loaded_individuals: Option<Vec<Individual>> =
        if let Ok(file) = std::fs::File::open(&checkpoint_path) {
            let reader = std::io::BufReader::new(file);
            match deserialize_from::<_, Vec<Individual>>(reader) {
                Ok(individuals) => Some(individuals),
                Err(e) => {
                    eprintln!("Could not deserialize checkpoint ({e}); starting fresh.");
                    None
                }
            }
        } else {
            eprintln!("No checkpoint found; starting fresh.");
            None
        };

    // Validate checkpoint: check both population size and gene count
    let all_individuals = match loaded_individuals {
        Some(loaded) if loaded.len() == total_pop_size => {
            // Check if gene count matches expected topology
            if let Some(first) = loaded.first() {
                if first.genes.len() != expected_gene_count {
                    eprintln!(
                        "Checkpoint gene count {} != expected {} for topology {}; starting fresh.",
                        first.genes.len(),
                        expected_gene_count,
                        config.network_topology.to_config_string()
                    );
                    initial_individuals
                } else {
                    println!("Loaded checkpoint with matching topology.");
                    loaded
                }
            } else {
                initial_individuals
            }
        }
        Some(loaded) => {
            eprintln!(
                "Checkpoint size {} != expected {}; starting fresh.",
                loaded.len(),
                total_pop_size
            );
            initial_individuals
        }
        None => initial_individuals,
    };

    // Split population across islands
    let num_islands = config.num_islands.max(1);
    let individuals_per_island = total_pop_size / num_islands;
    let mut islands: Vec<Island> = Vec::with_capacity(num_islands);

    // Base hyperparameters from config
    let base_hyperparams = HyperParams::new(
        config.mutation_rate,
        config.mutation_strength,
        config.tournament_size,
    );

    for i in 0..num_islands {
        let start = i * individuals_per_island;
        let end = if i == num_islands - 1 {
            total_pop_size // Last island gets any remainder
        } else {
            start + individuals_per_island
        };

        let island_individuals = all_individuals[start..end].to_vec();
        let island_seed = config.rng_seed.wrapping_add(i as u64 * 1000);

        // Each island starts with base hyperparams, but with slight variation if PBT is enabled
        let mut island_hyperparams = base_hyperparams.clone();
        if config.pbt_enabled {
            // Add initial diversity to hyperparameters
            island_hyperparams.perturb(&mut rng);
        }

        islands.push(Island::new(
            island_individuals,
            island_seed,
            island_hyperparams,
        ));
    }

    if config.pbt_enabled {
        println!("Initial hyperparameters per island:");
        for (i, island) in islands.iter().enumerate() {
            let hp = &island.hyperparams;
            println!(
                "  Island {}: mut_rate={:.3}, mut_str={:.3}, tourn={}",
                i, hp.mutation_rate, hp.mutation_strength, hp.tournament_size
            );
        }
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

    // PBT metrics writer (only if PBT is enabled)
    let mut pbt_metrics_writer = if config.pbt_enabled {
        Some(PbtMetricsWriter::new(&pbt_metrics_path).expect("Failed to create PBT metrics file"))
    } else {
        None
    };

    // Clone config values for use in parallel closure
    let max_frames = config.max_frames;
    let elitism_per_island = (config.elitism / num_islands).max(1);
    let topology = &config.network_topology;

    for generation in 0..config.epochs {
        // ========================================
        // PARALLEL EVOLUTION: Run all islands concurrently
        // Each island uses its own hyperparameters (for PBT)
        // ========================================
        let road_grid_ref = &base_road_grid;

        // Collect cars from each island's simulation for metrics
        let island_cars: Vec<Vec<traffic::cars::Car>> = islands
            .par_iter_mut()
            .map(|island| {
                island.run_generation(road_grid_ref, topology, max_frames, elitism_per_island)
            })
            .collect();

        // ========================================
        // MIGRATION: Exchange top individuals between islands
        // ========================================
        if config.migration_interval > 0
            && generation % config.migration_interval == 0
            && generation > 0
        {
            migrate_between_islands(&mut islands, config.migration_count);
        }

        // ========================================
        // PBT: Exploit/Explore hyperparameters
        // ========================================
        if config.pbt_enabled
            && config.pbt_interval > 0
            && generation % config.pbt_interval == 0
            && generation > 0
        {
            pbt_exploit_explore(&mut islands, &mut rng);
        }

        // ========================================
        // PBT METRICS: Log hyperparameters for each island
        // ========================================
        if let Some(ref mut writer) = pbt_metrics_writer {
            let snapshots: Vec<IslandHyperParamSnapshot> = islands
                .iter()
                .enumerate()
                .map(|(i, island)| {
                    // Compute gene diversity for this island
                    let gene_diversity = if !island.population.individuals.is_empty()
                        && !island.population.individuals[0].genes.is_empty()
                    {
                        let gene_len = island.population.individuals[0].genes.len();
                        let total_std_dev: f32 = (0..gene_len)
                            .map(|g| {
                                let vals: Vec<f32> = island
                                    .population
                                    .individuals
                                    .iter()
                                    .map(|ind| ind.genes.get(g).copied().unwrap_or(0.0))
                                    .collect();
                                let m = vals.iter().sum::<f32>() / vals.len().max(1) as f32;
                                let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f32>()
                                    / vals.len().max(1) as f32;
                                var.sqrt()
                            })
                            .sum();
                        total_std_dev / gene_len as f32
                    } else {
                        0.0
                    };

                    IslandHyperParamSnapshot::new(
                        generation as u32,
                        i,
                        island.hyperparams.mutation_rate,
                        island.hyperparams.mutation_strength,
                        island.hyperparams.tournament_size,
                        island.best_fitness(),
                        island.mean_fitness(),
                        gene_diversity,
                    )
                })
                .collect();

            if let Err(e) = writer.write_all_islands(&snapshots) {
                eprintln!("Failed to write PBT metrics: {e}");
            }
        }

        // ========================================
        // COLLECT METRICS: Gather stats from all islands
        // ========================================
        let island_best: Vec<f32> = islands.iter().map(|i| i.best_fitness()).collect();
        let island_mean: Vec<f32> = islands.iter().map(|i| i.mean_fitness()).collect();

        let gen_best = island_best
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
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

        let metrics =
            GenerationMetrics::compute(generation as u32, &combined_population, &all_cars);

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
