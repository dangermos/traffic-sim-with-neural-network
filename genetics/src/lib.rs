use std::{
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use ::rand::{Rng, SeedableRng, rngs::SmallRng};
use macroquad::color::Color;
use neural::{LayerTopology, Network};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, from_reader, to_writer};
use traffic::{
    cars::{Car, CarState, CarWorld},
    road::{RoadGrid, RoadId, generate_road_grid},
    simulation::Simulation,
};

/*

The loop for neuroevolution is
    Initialize population
    REPEAT (for each generation):
        Evaluate fitness (simulation)
        Select parents
        Produce offspring (mutation)
        Replace population

    Eval -> Select -> Mutate -> Replace

*/

const COMPLETION_REWARD: f32 = 100.0; // Reward for % of journey completed
const EFFICIENCY_REWARD: f32 = 50.0; // Reward for direct paths
const OFF_ROAD_PENALTY: f32 = 2.0;
const CRASH_PENALTY: f32 = 0.1;
const IDLE_PENALTY: f32 = 10.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    pub individuals: Vec<Individual>,
    pub generation: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    pub genes: Vec<f32>,
    pub fitness: f32,
}

pub fn fitness(car: &Car) -> f32 {
    // Hard penalty for not moving at all
    if car.distance_traveled < 5.0 {
        return -IDLE_PENALTY;
    }

    let mut f = 0.0;

    // Primary: completion ratio (0.0 to 1.0) - normalized by initial distance
    // This makes fitness comparable across different spawn/destination pairs
    let completion_ratio = if car.initial_distance_to_goal > 0.0 {
        (car.progress_to_goal / car.initial_distance_to_goal).clamp(0.0, 1.0)
    } else {
        0.0
    };
    f += completion_ratio * COMPLETION_REWARD;

    // Secondary: efficiency - how direct was the path?
    // progress / distance_traveled, where 1.0 = perfect straight line
    let efficiency = if car.distance_traveled > 0.0 {
        (car.progress_to_goal / car.distance_traveled).clamp(0.0, 1.0)
    } else {
        0.0
    };
    f += efficiency * EFFICIENCY_REWARD;

    // SPINNING PENALTY: If car traveled a lot but made little progress, penalize heavily.
    // This catches cars stuck in turning loops. A car that travels 100 units but only
    // makes 5 units of progress toward goal is clearly spinning or going in circles.
    const SPINNING_PENALTY: f32 = 20.0;
    const MIN_PROGRESS_RATIO: f32 = 0.1; // Expect at least 10% of distance traveled to be progress
    
    if car.distance_traveled > 20.0 {
        let progress_ratio = car.progress_to_goal / car.distance_traveled;
        if progress_ratio < MIN_PROGRESS_RATIO {
            // Scale penalty by how much they traveled while spinning
            f -= SPINNING_PENALTY * (1.0 - progress_ratio / MIN_PROGRESS_RATIO);
        }
    }

    // Penalize being off-road (continuous)
    f -= car.time_spent_off_road * OFF_ROAD_PENALTY;

    // Crash penalty - multiplicative so it scales with achievement
    if matches!(car.state(), CarState::Crashed) {
        f *= CRASH_PENALTY;
    }

    f
}

pub fn mutate<R: Rng>(genes: &mut [f32], mutation_rate: f32, mutation_strength: f32, rng: &mut R) {
    for gene in genes.iter_mut() {
        if rng.random::<f32>() < mutation_rate {
            let mutation: f32 = rng.random_range(-mutation_strength..mutation_strength);
            *gene += mutation;
        }
    }
}

pub fn evolve_generation<R: Rng>(
    population: &Population, // Initial Population
    elitism: usize,          // N top individuals to keep
    mutation_rate: f32,
    mutation_strength: f32,
    rng: &mut R,
) -> Population {
    let mut sorted = population.individuals.clone();
    sorted.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));

    let n = elitism.min(sorted.len());
    let mut next_gen = Vec::with_capacity(sorted.len());

    next_gen.extend(sorted.iter().take(n).cloned());

    let rest = sorted.len().saturating_sub(n);

    // Seed per-child RNGs deterministically from the passed-in RNG so runs remain reproducible.
    let seeds: Vec<u64> = (0..rest).map(|_| rng.random()).collect();

    let children: Vec<Individual> = seeds
        .into_par_iter()
        .map(|seed| {
            // SmallRng is faster to initialize per task than StdRng.
            let mut local_rng = SmallRng::seed_from_u64(seed);
            let parent = tournament_select(&sorted, 3, &mut local_rng);
            let mut child_genes = parent.genes.clone();
            mutate(
                &mut child_genes,
                mutation_rate,
                mutation_strength,
                &mut local_rng,
            );
            Individual {
                genes: child_genes,
                fitness: 0.0,
            }
        })
        .collect();

    next_gen.extend(children);

    let generation = population.generation + 1;
    Population {
        individuals: next_gen,
        generation,
    }
}

/*pub fn crossover<R: Rng>(parent_a: &[f32], parent_b: &[f32], rng: &mut R) -> Vec<f32> {
    //TODO implement Crossover (if needed)
    todo!()
}*/

pub fn tournament_select<'from_population, R: Rng>(
    population: &'from_population [Individual],
    size: usize,
    rng: &mut R,
) -> &'from_population Individual {
    assert!(
        !population.is_empty(),
        "Cannot select from empty population"
    );
    assert!(size > 0, "Tournament size must be at least 1");

    let tournament_size = size.min(population.len());

    let mut best_idx = rng.random_range(0..population.len());

    for _ in 1..tournament_size {
        let candidate_idx = rng.random_range(0..population.len());
        if population[candidate_idx].fitness > population[best_idx].fitness {
            best_idx = candidate_idx;
        }
    }

    &population[best_idx]
}

pub fn make_sim_from_population<R: Rng>(population: &Population, rng: &mut R) -> Simulation {
    let road_grid = generate_road_grid(20, rng);
    make_sim_from_population_with_grid(population, &road_grid, rng)
}

pub fn make_sim_from_population_with_grid<R: Rng>(
    population: &Population,
    road_grid: &RoadGrid,
    rng: &mut R,
) -> Simulation {
    make_sim_from_slice(&population.individuals, road_grid, rng)
}

pub fn make_sim_from_slice<R: Rng>(
    individuals: &[Individual],
    road_grid: &RoadGrid,
    rng: &mut R,
) -> Simulation {
    let road_count = road_grid.roads.len().max(1);

    let cars = CarWorld::new(
        individuals
            .iter()
            .enumerate()
            .map(|(i, indiv)| {
                const INPUTS: usize = 5;
                const OUTPUTS: usize = 2;

                // Infer a reasonable topology from the genes while matching the car's
                // expected input/output sizes.
                let topology: Vec<LayerTopology> = {
                    let direct_count = OUTPUTS * (INPUTS + 1);
                    if indiv.genes.len() == direct_count {
                        vec![
                            LayerTopology { neurons: INPUTS },
                            LayerTopology { neurons: OUTPUTS },
                        ]
                    } else {
                        const MAX_HIDDEN: usize = 64;
                        let mut matched = None;
                        for hidden in 1..=MAX_HIDDEN {
                            let needed = hidden * (INPUTS + 1) + OUTPUTS * (hidden + 1);
                            if needed == indiv.genes.len() {
                                matched = Some(hidden);
                                break;
                            }
                        }

                        let hidden = matched.unwrap_or_else(|| {
                            let denom = INPUTS + OUTPUTS + 1; // 8
                            ((indiv.genes.len().saturating_sub(OUTPUTS)) / denom)
                                .clamp(1, MAX_HIDDEN)
                        });

                        vec![
                            LayerTopology { neurons: INPUTS },
                            LayerTopology { neurons: hidden },
                            LayerTopology { neurons: OUTPUTS },
                        ]
                    }
                };

                let network = Network::from_genes(&topology, &indiv.genes);

                let (r, g, b, _a): (u8, u8, u8, u8) = rng.random();
                let color = Color::from_rgba(r, g, b, 255);

                let road_id = RoadId(i % road_count);
                let car_id = i as u16;

                Car::new_on_road(road_grid, road_id, color, network, car_id)
            })
            .collect(),
    );

    Simulation::new(cars, road_grid.clone())
}

pub fn load_best_history(path: &PathBuf) -> Vec<f32> {
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

pub fn write_best_history(path: &PathBuf, history: &[f32]) {
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
