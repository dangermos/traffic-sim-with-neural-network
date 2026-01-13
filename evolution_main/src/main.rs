use bincode::{deserialize_from, serialize};
use ::rand::SeedableRng;
use genetics::{evolve_generation, make_sim_from_population_with_grid, Individual, Population};
use macroquad::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::path::PathBuf;
use serde_json::{from_reader, to_writer};
use traffic::{levels::test_sensors, road::RoadId};
use indicatif::ProgressBar;

fn main() {
    // rng
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Size Variables
    let x = 1920.0;
    let y = 1080.0;
    let center = vec2(x * 0.5, y * 0.5);
    let screen = vec2(x, y);

    // Pick a Level
    
    let mut sim = test_sensors(center, screen, &mut rng);

    println!(
        "Level Description:
     Cars: {:?}
     Roads: {:?}
     ",
        sim.cars
            .cars
            .iter()
            .map(|x| x.get_id())
            .collect::<Vec<u16>>(),
        sim.roads
            .roads
            .iter()
            .map(|x| x.get_id())
            .collect::<Vec<RoadId>>()
    );

    

    // This controls how many times the simulation runs before running fitness evaluation
    const EPOCHS: usize = 20000;
    const MAX_FRAMES: usize = 5000;


    let initial_individuals: Vec<Individual> = 
        sim
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
        eprintln!("No checkpoint found at {:?}; starting fresh.", checkpoint_path);
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

    let mut best_fitness = std::fs::File::open(&best_path)
        .ok()
        .and_then(|f| from_reader::<_, f32>(std::io::BufReader::new(f)).ok())
        .unwrap_or(f32::NEG_INFINITY);
    let mut last_evaluated = population.clone();
    let pb = ProgressBar::new(EPOCHS as u64);

    const STALL_DISTANCE: f32 = 5.0;
    const STALL_FRAMES: usize = 200;

    for generation in 0..EPOCHS {
        let mut max_distance = 0.0;

        for frame in 0..MAX_FRAMES {
            sim.update(false);

            // Track max distance traveled to detect stagnation and exit early.
            if frame % 16 == 0 {
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
        for (ind, car) in population
            .individuals
            .iter_mut()
            .zip(sim.cars.cars.iter())
        {
            ind.fitness = genetics::fitness(car);
            best_fitness = best_fitness.max(ind.fitness);
        }

        last_evaluated = population.clone();

        let new = evolve_generation(&population, 3, 0.2, 0.2, &mut rng);

        population = new;
        sim = make_sim_from_population_with_grid(&population, &base_road_grid, &mut rng);

        pb.inc(1);
        let eta = pb.eta();
        println!("About {} Hours, {} Minutes, {} Seconds left for this program run", eta.as_secs() / 3600, (eta.as_secs() % 3600) / 60, eta.as_secs() % 60);
    }

    let bytes = serialize(&last_evaluated.individuals).expect("Could not serialize individuals");
    std::fs::write(&checkpoint_path, &bytes).expect("Could not write checkpoint");
    println!(
        "Wrote {} individuals to {:?} ({} bytes)",
        last_evaluated.individuals.len(),
        checkpoint_path,
        bytes.len()
    );

    if let Ok(file) = std::fs::File::create(&best_path) {
        let writer = std::io::BufWriter::new(file);
        if let Err(e) = to_writer(writer, &best_fitness) {
            eprintln!("Failed to write {:?}: {e}", best_path);
        }
    } else {
        eprintln!("Could not create {:?}", best_path);
    }

    pb.finish_with_message("Evolution Complete");
    println!("Best Fitness: {}", best_fitness);

} // End Simulation
