use ::rand::{Rng, rngs::SmallRng, SeedableRng};
use bincode::Serializer;
use macroquad::color::Color;
use neural::{LayerTopology, Network};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
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

const PROGRESS_REWARD: f32 = 2.0;
const TIME_REWARD: f32 = 0.1;
const OFF_ROAD_PENALTY: f32 = 0.9;
const CRASH_PENALTY: f32 = 0.4;
const STAGNANT_PENALTY: f32 = 0.95;

pub struct EvolutionConfig {
    population_size: usize,
    generations: usize,
    steps_per_generation: usize,
    elitism: usize,
    mutation_rate: f32,
    mutation_strength: f32,
    tournament_size: usize,
    topology: Vec<LayerTopology>,
}

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
    let mut fitness = 0.0;

    fitness += car.progress_to_goal * PROGRESS_REWARD;
    fitness += car.time_spent_alive * TIME_REWARD;
    fitness -= car.time_spent_off_road * OFF_ROAD_PENALTY;
    
    if car.distance_traveled < 10.0 {
        fitness *= STAGNANT_PENALTY;
    }

    match car.state() {
        CarState::Crashed => fitness *= CRASH_PENALTY,
        _ => {}
    }

    fitness.max(0.0)
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
    sorted.sort_by(|a, b| a.fitness.total_cmp(&b.fitness));

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
            mutate(&mut child_genes, mutation_rate, mutation_strength, &mut local_rng);
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

    let road_count = road_grid.roads.len().max(1);

    let cars = CarWorld::new(
        population
            .individuals
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
