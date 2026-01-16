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

/// Neural network topology configuration.
///
/// The topology defines the structure of the neural network:
/// - First element is the input layer (must match number of sensor inputs)
/// - Last element is the output layer (must match number of control outputs)
/// - Middle elements are hidden layers
///
/// Example: `[5, 8, 2]` means 5 inputs, 8 hidden neurons, 2 outputs
/// Example: `[5, 12, 8, 2]` means 5 inputs, two hidden layers (12 and 8), 2 outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    /// Layer sizes from input to output
    pub layers: Vec<usize>,
}

impl Default for NetworkTopology {
    fn default() -> Self {
        // Default: 5 inputs, 8 hidden, 2 outputs
        Self {
            layers: vec![5, 8, 2],
        }
    }
}

impl NetworkTopology {
    /// Create a new topology from layer sizes
    pub fn new(layers: Vec<usize>) -> Self {
        assert!(layers.len() >= 2, "Need at least input and output layers");
        assert!(layers[0] > 0, "Input layer must have at least 1 neuron");
        assert!(
            layers.last().unwrap() > &0,
            "Output layer must have at least 1 neuron"
        );
        Self { layers }
    }

    /// Create topology from a string like "5,8,2" or "5,12,8,2"
    pub fn from_str(s: &str) -> Option<Self> {
        let layers: Result<Vec<usize>, _> =
            s.split(',').map(|x| x.trim().parse::<usize>()).collect();

        match layers {
            Ok(layers) if layers.len() >= 2 && layers.iter().all(|&x| x > 0) => {
                Some(Self { layers })
            }
            _ => None,
        }
    }

    /// Get the number of inputs
    pub fn inputs(&self) -> usize {
        self.layers[0]
    }

    /// Get the number of outputs
    pub fn outputs(&self) -> usize {
        *self.layers.last().unwrap()
    }

    /// Get hidden layer sizes (everything except first and last)
    pub fn hidden_layers(&self) -> &[usize] {
        if self.layers.len() > 2 {
            &self.layers[1..self.layers.len() - 1]
        } else {
            &[]
        }
    }

    /// Convert to LayerTopology vector for neural network creation
    pub fn to_layer_topologies(&self) -> Vec<LayerTopology> {
        self.layers
            .iter()
            .map(|&neurons| LayerTopology { neurons })
            .collect()
    }

    /// Calculate the number of genes (weights + biases) needed for this topology
    pub fn gene_count(&self) -> usize {
        let mut count = 0;
        for i in 0..self.layers.len() - 1 {
            let inputs = self.layers[i];
            let outputs = self.layers[i + 1];
            // Each neuron in the next layer has: inputs weights + 1 bias
            count += outputs * (inputs + 1);
        }
        count
    }

    /// Create a string representation like "5,8,2"
    pub fn to_config_string(&self) -> String {
        self.layers
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Display format for logging
    pub fn display(&self) -> String {
        let parts: Vec<String> = self.layers.iter().map(|x| x.to_string()).collect();
        format!("[{}] ({} genes)", parts.join(" → "), self.gene_count())
    }
}

// Helper trait to access distance_to_destination from Car
trait CarFitnessExt {
    fn distance_to_destination(&self) -> f32;
}

impl CarFitnessExt for Car {
    fn distance_to_destination(&self) -> f32 {
        (self.initial_distance_to_goal - self.progress_to_goal).max(0.0)
    }
}

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

// === FITNESS WEIGHTS ===
// This fitness function has a HIGH SKILL CEILING:
// - Simply reaching the goal gives moderate reward
// - Reaching FAST gives massive bonus (up to 150 extra points)
// - This creates selection pressure even after cars learn to complete

// Base rewards (low - just for being active)
const BASE_MOVEMENT_REWARD: f32 = 5.0;

// Goal completion rewards
const DESTINATION_BONUS: f32 = 100.0; // Base bonus for reaching goal
const SPEED_BONUS_MAX: f32 = 150.0; // ADDITIONAL bonus for reaching fast
const PROXIMITY_REWARD_MAX: f32 = 50.0; // Reward for ending close to goal

// Road-following rewards (foundation for learning)
const ROAD_ADHERENCE_REWARD: f32 = 30.0; // Staying on road
const ACTIVE_DRIVING_REWARD: f32 = 20.0; // Moving while on road

// Penalties
const OFF_ROAD_PENALTY: f32 = 50.0; // Going off road
const CRASH_PENALTY_MULT: f32 = 0.2; // Multiplicative - keeps only 20%
const STAGNANT_PENALTY: f32 = 40.0; // Giving up / stopping
const IDLE_PENALTY: f32 = 20.0; // Not moving at all

// Reference time for speed bonus calculation
// With max_frames=3000, expect good runs to complete around 1500 frames
const EXPECTED_COMPLETION_TIME: f32 = 1500.0;

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

/// Fitness function with HIGH SKILL CEILING
///
/// Fitness ranges:
/// - Crash/Stagnant: -40 to +20
/// - Active but doesn't reach goal: +20 to +100
/// - Reaches goal slowly (1000 frames): ~175
/// - Reaches goal at expected time (500 frames): ~250
/// - Reaches goal FAST (250 frames): ~325
///
/// This ensures there's always room to improve even after learning to complete!
pub fn fitness(car: &Car) -> f32 {
    // ============================================
    // IMMEDIATE FAILURES
    // ============================================

    if car.distance_traveled < 5.0 {
        return -IDLE_PENALTY + car.distance_traveled * 0.5;
    }

    let mut f = BASE_MOVEMENT_REWARD;

    // ============================================
    // ROAD-FOLLOWING REWARDS (foundation)
    // ============================================

    let on_road_ratio = if car.time_spent_alive > 0.0 {
        1.0 - (car.time_spent_off_road / car.time_spent_alive).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Road adherence
    f += on_road_ratio * ROAD_ADHERENCE_REWARD;

    // Active driving: reward distance traveled while on road
    let distance_score = (car.distance_traveled / 1000.0).clamp(0.0, 1.0);
    f += distance_score * on_road_ratio * ACTIVE_DRIVING_REWARD;

    // ============================================
    // GOAL COMPLETION REWARDS (main objective)
    // This is where the skill ceiling comes from!
    // ============================================

    let reached_goal = matches!(car.state(), CarState::ReachedDestination);

    if reached_goal {
        // BASE: You reached the goal!
        f += DESTINATION_BONUS;

        // SPEED BONUS: The faster you reach, the more points!
        // This creates MASSIVE differentiation between slow and fast completions.
        //
        // At EXPECTED_COMPLETION_TIME (500 frames): ~75 bonus points
        // At half that time (250 frames): ~150 bonus points (maximum)
        // At double that time (1000 frames): ~37 bonus points
        let completion_time = car.time_spent_alive.max(1.0);
        let speed_factor = (EXPECTED_COMPLETION_TIME / completion_time).clamp(0.0, 2.0);
        let speed_bonus = speed_factor * SPEED_BONUS_MAX * 0.5;
        f += speed_bonus;

        // EFFICIENCY BONUS: Did you take a relatively direct path?
        if car.distance_traveled > 0.0 && car.initial_distance_to_goal > 0.0 {
            let path_ratio = car.initial_distance_to_goal / car.distance_traveled;
            let efficiency_bonus = path_ratio.clamp(0.0, 0.5) * 40.0;
            f += efficiency_bonus;
        }
    } else {
        // Didn't reach goal - reward based on how close you got
        if car.initial_distance_to_goal > 0.0 {
            let final_distance = car.distance_to_destination();
            let proximity_ratio =
                1.0 - (final_distance / car.initial_distance_to_goal).clamp(0.0, 1.0);

            // Quadratic scaling: getting VERY close matters more
            let proximity_score = proximity_ratio * proximity_ratio;
            f += proximity_score * PROXIMITY_REWARD_MAX;
        }
    }

    // ============================================
    // PENALTIES
    // ============================================

    // Off-road penalty
    if car.time_spent_alive > 0.0 {
        let off_road_ratio = (car.time_spent_off_road / car.time_spent_alive).clamp(0.0, 1.0);
        f -= off_road_ratio * OFF_ROAD_PENALTY;
    }

    // ============================================
    // TERMINAL STATE PENALTIES
    // ============================================

    if matches!(car.state(), CarState::Crashed) {
        f *= CRASH_PENALTY_MULT;
    }

    if matches!(car.state(), CarState::Stagnant) {
        f -= STAGNANT_PENALTY;
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
    population: &Population,
    elitism: usize,
    mutation_rate: f32,
    mutation_strength: f32,
    tournament_size: usize,
    rng: &mut R,
) -> Population {
    let mut sorted = population.individuals.clone();
    sorted.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));

    let n = elitism.min(sorted.len());
    let mut next_gen = Vec::with_capacity(sorted.len());

    next_gen.extend(sorted.iter().take(n).cloned());

    let rest = sorted.len().saturating_sub(n);

    let seeds: Vec<u64> = (0..rest).map(|_| rng.random()).collect();

    let children: Vec<Individual> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut local_rng = SmallRng::seed_from_u64(seed);
            let parent = tournament_select(&sorted, tournament_size, &mut local_rng);
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
    make_sim_from_slice_with_topology(individuals, road_grid, &NetworkTopology::default(), rng)
}

/// Create simulation with explicit network topology
pub fn make_sim_from_slice_with_topology<R: Rng>(
    individuals: &[Individual],
    road_grid: &RoadGrid,
    topology: &NetworkTopology,
    rng: &mut R,
) -> Simulation {
    let road_count = road_grid.roads.len().max(1);
    let layer_topologies = topology.to_layer_topologies();
    let expected_genes = topology.gene_count();

    let cars = CarWorld::new(
        individuals
            .iter()
            .enumerate()
            .map(|(i, indiv)| {
                // Use provided topology, but handle gene count mismatch gracefully
                let final_topology = if indiv.genes.len() == expected_genes {
                    layer_topologies.clone()
                } else {
                    // Try to infer topology from gene count (backwards compatibility)
                    infer_topology_from_genes(
                        indiv.genes.len(),
                        topology.inputs(),
                        topology.outputs(),
                    )
                };

                let network = Network::from_genes(&final_topology, &indiv.genes);

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

/// Infer network topology from gene count (for backwards compatibility with checkpoints)
fn infer_topology_from_genes(
    gene_count: usize,
    inputs: usize,
    outputs: usize,
) -> Vec<LayerTopology> {
    // Check if it's a direct connection (no hidden layer)
    let direct_count = outputs * (inputs + 1);
    if gene_count == direct_count {
        return vec![
            LayerTopology { neurons: inputs },
            LayerTopology { neurons: outputs },
        ];
    }

    // Try to find a single hidden layer size that matches
    const MAX_HIDDEN: usize = 64;
    for hidden in 1..=MAX_HIDDEN {
        let needed = hidden * (inputs + 1) + outputs * (hidden + 1);
        if needed == gene_count {
            return vec![
                LayerTopology { neurons: inputs },
                LayerTopology { neurons: hidden },
                LayerTopology { neurons: outputs },
            ];
        }
    }

    // Try two hidden layers
    for h1 in 1..=MAX_HIDDEN {
        for h2 in 1..=MAX_HIDDEN {
            let needed = h1 * (inputs + 1) + h2 * (h1 + 1) + outputs * (h2 + 1);
            if needed == gene_count {
                return vec![
                    LayerTopology { neurons: inputs },
                    LayerTopology { neurons: h1 },
                    LayerTopology { neurons: h2 },
                    LayerTopology { neurons: outputs },
                ];
            }
        }
    }

    // Fallback: estimate single hidden layer
    let denom = inputs + outputs + 1;
    let hidden = ((gene_count.saturating_sub(outputs)) / denom).clamp(1, MAX_HIDDEN);

    vec![
        LayerTopology { neurons: inputs },
        LayerTopology { neurons: hidden },
        LayerTopology { neurons: outputs },
    ]
}

/// Create a new random population with the given topology
pub fn create_random_population<R: Rng>(
    size: usize,
    topology: &NetworkTopology,
    rng: &mut R,
) -> Population {
    let gene_count = topology.gene_count();

    let individuals = (0..size)
        .map(|_| {
            let genes: Vec<f32> = (0..gene_count)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            Individual {
                genes,
                fitness: 0.0,
            }
        })
        .collect();

    Population {
        individuals,
        generation: 0,
    }
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
