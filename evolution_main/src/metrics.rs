use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use traffic::cars::{Car, CarState};

use genetics::Population;

/// Metrics collected for each generation during evolution.
#[derive(Debug, Clone)]
pub struct GenerationMetrics {
    pub generation: u32,

    // Fitness distribution
    pub best_fitness: f32,
    pub mean_fitness: f32,
    pub median_fitness: f32,
    pub worst_fitness: f32,
    pub fitness_std_dev: f32,

    // Movement/behavior
    pub max_distance_traveled: f32,
    pub mean_distance_traveled: f32,
    pub max_progress_to_goal: f32,
    pub mean_progress_to_goal: f32,

    // Survival metrics
    pub num_crashed: u32,
    pub num_reached_destination: u32,
    pub num_stagnant: u32,
    pub mean_time_alive: f32,

    // Efficiency
    pub mean_efficiency: f32,

    // Road adherence
    pub mean_time_off_road: f32,

    // Population diversity
    pub gene_diversity: f32,
}

impl GenerationMetrics {
    /// Compute metrics from the current population and simulation state.
    pub fn compute(generation: u32, population: &Population, cars: &[Car]) -> Self {
        let n = cars.len() as f32;

        // Fitness metrics
        let fitnesses: Vec<f32> = population.individuals.iter().map(|i| i.fitness).collect();
        let best_fitness = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let worst_fitness = fitnesses.iter().cloned().fold(f32::INFINITY, f32::min);
        let mean_fitness = fitnesses.iter().sum::<f32>() / n;

        let mut sorted_fit = fitnesses.clone();
        sorted_fit.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_fitness = sorted_fit[sorted_fit.len() / 2];

        let variance = fitnesses
            .iter()
            .map(|f| (f - mean_fitness).powi(2))
            .sum::<f32>()
            / n;
        let fitness_std_dev = variance.sqrt();

        // Distance metrics
        let max_distance_traveled = cars
            .iter()
            .map(|c| c.distance_traveled)
            .fold(0.0f32, f32::max);
        let mean_distance_traveled = cars.iter().map(|c| c.distance_traveled).sum::<f32>() / n;

        let max_progress_to_goal = cars
            .iter()
            .map(|c| c.progress_to_goal)
            .fold(0.0f32, f32::max);
        let mean_progress_to_goal = cars.iter().map(|c| c.progress_to_goal).sum::<f32>() / n;

        // Behavior counts
        let num_crashed = cars
            .iter()
            .filter(|c| matches!(c.state(), CarState::Crashed))
            .count() as u32;
        let num_reached_destination = cars
            .iter()
            .filter(|c| matches!(c.state(), CarState::ReachedDestination))
            .count() as u32;
        let num_stagnant = cars.iter().filter(|c| c.distance_traveled < 5.0).count() as u32;

        // Time metrics
        let mean_time_alive = cars.iter().map(|c| c.time_spent_alive).sum::<f32>() / n;
        let mean_time_off_road = cars.iter().map(|c| c.time_spent_off_road).sum::<f32>() / n;

        // Efficiency: progress / distance (how direct is the path)
        let mean_efficiency = cars
            .iter()
            .map(|c| {
                if c.distance_traveled > 0.0 {
                    c.progress_to_goal / c.distance_traveled
                } else {
                    0.0
                }
            })
            .sum::<f32>()
            / n;

        // Genetic diversity: average standard deviation across all gene positions
        let gene_diversity = if !population.individuals.is_empty()
            && !population.individuals[0].genes.is_empty()
        {
            let gene_len = population.individuals[0].genes.len();
            let total_std_dev: f32 = (0..gene_len)
                .map(|i| {
                    let vals: Vec<f32> = population
                        .individuals
                        .iter()
                        .map(|ind| ind.genes[i])
                        .collect();
                    let m = vals.iter().sum::<f32>() / vals.len() as f32;
                    let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f32>() / vals.len() as f32;
                    var.sqrt()
                })
                .sum();
            total_std_dev / gene_len as f32
        } else {
            0.0
        };

        Self {
            generation,
            best_fitness,
            mean_fitness,
            median_fitness,
            worst_fitness,
            fitness_std_dev,
            max_distance_traveled,
            mean_distance_traveled,
            max_progress_to_goal,
            mean_progress_to_goal,
            num_crashed,
            num_reached_destination,
            num_stagnant,
            mean_time_alive,
            mean_efficiency,
            mean_time_off_road,
            gene_diversity,
        }
    }

    /// Convert metrics to a CSV row string.
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            self.generation,
            self.best_fitness,
            self.mean_fitness,
            self.median_fitness,
            self.worst_fitness,
            self.fitness_std_dev,
            self.max_distance_traveled,
            self.mean_distance_traveled,
            self.max_progress_to_goal,
            self.mean_progress_to_goal,
            self.num_crashed,
            self.num_reached_destination,
            self.num_stagnant,
            self.mean_time_alive,
            self.mean_time_off_road,
            self.mean_efficiency,
            self.gene_diversity,
        )
    }
}

/// CSV header for metrics file.
pub const CSV_HEADER: &str = "generation,best_fitness,mean_fitness,median_fitness,worst_fitness,fitness_std_dev,max_distance,mean_distance,max_progress,mean_progress,num_crashed,num_reached,num_stagnant,mean_time_alive,mean_time_off_road,mean_efficiency,gene_diversity\n";

/// Writer for streaming metrics to a CSV file.
pub struct MetricsWriter {
    writer: BufWriter<File>,
}

impl MetricsWriter {
    /// Create a new metrics writer, writing the CSV header.
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::new(file);
        writer.write_all(CSV_HEADER.as_bytes())?;
        writer.flush()?;

        Ok(Self { writer })
    }

    /// Append a metrics row to the CSV file.
    pub fn write_metrics(&mut self, metrics: &GenerationMetrics) -> std::io::Result<()> {
        self.writer.write_all(metrics.to_csv_row().as_bytes())?;
        // Flush periodically so we don't lose data on crash
        self.writer.flush()?;
        Ok(())
    }
}

/// Print a brief progress summary to stdout.
pub fn print_progress(metrics: &GenerationMetrics) {
    println!(
        "Gen {:5} | Best: {:8.2} | Mean: {:8.2} | Stagnant: {:3} | Crashed: {:3} | Reached: {:3} | Diversity: {:.4}",
        metrics.generation,
        metrics.best_fitness,
        metrics.mean_fitness,
        metrics.num_stagnant,
        metrics.num_crashed,
        metrics.num_reached_destination,
        metrics.gene_diversity,
    );
}
