use neural::Network;
use traffic::{cars::Car, simulation::Simulation};


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

const OFF_ROAD_PENALTY: f32 = 0.9;
const CRASH_PENALTY: f32 = 0.4;



struct Genome {
    net: Network,
    fitness: f32,
}




pub fn fitness(genome: Genome, ) {
    
}

pub fn mutate() {

}

pub async fn run_simulation(simulation: Simulation, epochs: u32, population_size: u32, ) {



}