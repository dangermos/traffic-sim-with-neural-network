use ::rand::SeedableRng;
use bincode::{deserialize_from, serialize};
use genetics::{Individual, Population, evolve_generation, make_sim_from_population_with_grid};
use indicatif::ProgressBar;
use macroquad::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, from_reader, to_writer};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use traffic::levels::overnight_training;

mod metrics;
use metrics::{GenerationMetrics, MetricsWriter, print_progress};

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

fn main() {
    // rng
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Use the overnight training level for comprehensive evolution
    let mut sim = overnight_training(&mut rng);

    println!(
        "Level Description:
     Cars: {}
     Roads: {}
     ",
        sim.cars.cars.len(),
        sim.roads.roads.len()
    );

    // Overnight training settings
    const EPOCHS: usize = 100; // Many generations for overnight run
    const MAX_FRAMES: usize = 2000; // Longer episodes for complex navigation
    const ELITISM: usize = 10; // Preserve more top performers
    const MUTATION_RATE: f32 = 0.3; // Higher exploration
    const MUTATION_STRENGTH: f32 = 0.3;

    // Early exit detection disabled - it was causing premature termination
    // and preventing proper fitness evaluation of slow-starting networks.
    const ENABLE_EARLY_EXIT: bool = false;

    let initial_individuals: Vec<Individual> = sim
        .cars
        .cars
        .iter()
        .map(|car| {
            let genes = car.network.to_genes();
            let fitness = genetics::fitness(car);
            Individual { genes, fitness }
        })
        .collect();

    let expected_pop_size = initial_individuals.len();

    let checkpoint_path = PathBuf::from("individuals.bin");
    let best_path = PathBuf::from("best_fitness.json");
    let metrics_path = PathBuf::from("metrics.csv");

    let i: Vec<Individual> = if let Ok(file) = std::fs::File::open(&checkpoint_path) {
        let reader = std::io::BufReader::new(file);
        deserialize_from(reader).unwrap_or_else(|e| {
            eprintln!(
                "Could not deserialize {:?} ({e}); starting fresh.",
                checkpoint_path
            );
            initial_individuals.clone()
        })
    } else {
        eprintln!(
            "No checkpoint found at {:?}; starting fresh.",
            checkpoint_path
        );
        initial_individuals.clone()
    };

    let mut population = Population {
        individuals: i,
        generation: 0,
    };

    let base_road_grid = sim.roads.clone();
    sim = make_sim_from_population_with_grid(&population, &base_road_grid, &mut rng);

    if population.individuals.len() != expected_pop_size {
        eprintln!(
            "Loaded population size {} does not match expected {}; starting fresh.",
            population.individuals.len(),
            expected_pop_size
        );
        population.individuals = initial_individuals.clone();
    }

    let mut best_history = load_best_history(&best_path);
    let mut best_fitness = best_history
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut last_evaluated = population.clone();
    let pb = ProgressBar::new(EPOCHS as u64);

    // Initialize metrics writer
    let mut metrics_writer =
        MetricsWriter::new(&metrics_path).expect("Failed to create metrics file");

    const STALL_DISTANCE: f32 = 5.0;
    const STALL_FRAMES: usize = 200;

    for generation in 0..EPOCHS {
        let mut max_distance = 0.0;

        for frame in 0..MAX_FRAMES {
            sim.update(false);

            // Track max distance traveled to detect stagnation and exit early.
            // NOTE: Early exit is now disabled by default because it was causing
            // premature termination before networks could be properly evaluated,
            // especially in early generations with random weights.
            if ENABLE_EARLY_EXIT && frame % 16 == 0 {
                max_distance = sim
                    .cars
                    .cars
                    .iter()
                    .map(|c| c.distance_traveled)
                    .fold(max_distance, f32::max);

                if frame >= STALL_FRAMES && max_distance < STALL_DISTANCE {
                    break;
                }
            }
        }

        // Evaluate fitness for the current generation before evolving.
        for (ind, car) in population.individuals.iter_mut().zip(sim.cars.cars.iter()) {
            ind.fitness = genetics::fitness(car);
            best_fitness = best_fitness.max(ind.fitness);
        }

        // Compute and record metrics
        let metrics = GenerationMetrics::compute(generation as u32, &population, &sim.cars.cars);

        if let Err(e) = metrics_writer.write_metrics(&metrics) {
            eprintln!("Failed to write metrics: {e}");
        }

        // Print progress every 10 generations
        if generation % 10 == 0 {
            print_progress(&metrics);
        }

        last_evaluated = population.clone();

        // Evolution Function
        let new = evolve_generation(
            &population,
            ELITISM,
            MUTATION_RATE,
            MUTATION_STRENGTH,
            &mut rng,
        );

        population = new;
        sim = make_sim_from_population_with_grid(&population, &base_road_grid, &mut rng);

        pb.inc(1);

        // Save checkpoint every 100 generations
        if generation % 100 == 0 && generation > 0 {
            let bytes =
                serialize(&last_evaluated.individuals).expect("Could not serialize individuals");
            std::fs::write(&checkpoint_path, &bytes).expect("Could not write checkpoint");
            best_history.push(best_fitness);
            write_best_history(&best_path, &best_history);
        }
    }

    let bytes = serialize(&last_evaluated.individuals).expect("Could not serialize individuals");
    std::fs::write(&checkpoint_path, &bytes).expect("Could not write checkpoint");
    println!(
        "Wrote {} individuals to {:?} ({} bytes)",
        last_evaluated.individuals.len(),
        checkpoint_path,
        bytes.len()
    );

    best_history.push(best_fitness);
    write_best_history(&best_path, &best_history);

    pb.finish_with_message("Evolution Complete");
    println!("Best Fitness: {}", best_fitness);
    println!("Metrics saved to: {:?}", metrics_path);
} // End Simulation
