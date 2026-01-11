use ::rand::Rng;
use bincode::Serializer;
use neural::{LayerTopology, Network};
use serde::{Deserialize, Serialize};
use traffic::{
    cars::{Car, CarState, CarWorld},
    road::{RoadGrid, generate_road_grid},
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

const PROGRESS_REWARD: f32 = 10.0;
const TIME_REWARD: f32 = 0.1;
const OFF_ROAD_PENALTY: f32 = 0.9;
const CRASH_PENALTY: f32 = 0.4;

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

    while next_gen.len() < sorted.len() {
        let parent = tournament_select(&sorted, 3, rng);
        let mut child_genes = parent.genes.clone();
        mutate(&mut child_genes, mutation_rate, mutation_strength, rng);
        next_gen.push(Individual {
            genes: child_genes,
            fitness: 0.0,
        });
    }

    Population {
        individuals: next_gen,
        generation: population.generation + 1,
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

    let cars = CarWorld::new(
   	population.individuals.iter().enumerate().map(|(i, indiv)| {
    	let toplogy: [LayerTopology; 2] = [LayerTopology {neurons: 5}, LayerTopology {neurons: 2}];
    	let net = Network::from_genes(topology, &indiv.genes);
     	Car::new_on_road(&road_grid, road_id, color, network, car_id)
    })
    )
}
