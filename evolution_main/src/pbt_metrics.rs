use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Metrics for a single island's hyperparameters at a given generation
#[derive(Debug, Clone)]
pub struct IslandHyperParamSnapshot {
    pub generation: u32,
    pub island_id: usize,
    pub mutation_rate: f32,
    pub mutation_strength: f32,
    pub tournament_size: usize,
    pub best_fitness: f32,
    pub mean_fitness: f32,
    pub gene_diversity: f32,
}

impl IslandHyperParamSnapshot {
    pub fn new(
        generation: u32,
        island_id: usize,
        mutation_rate: f32,
        mutation_strength: f32,
        tournament_size: usize,
        best_fitness: f32,
        mean_fitness: f32,
        gene_diversity: f32,
    ) -> Self {
        Self {
            generation,
            island_id,
            mutation_rate,
            mutation_strength,
            tournament_size,
            best_fitness,
            mean_fitness,
            gene_diversity,
        }
    }

    /// Convert to CSV row
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{}\n",
            self.generation,
            self.island_id,
            self.mutation_rate,
            self.mutation_strength,
            self.tournament_size,
            self.best_fitness,
            self.mean_fitness,
            self.gene_diversity,
        )
    }
}

/// CSV header for PBT metrics
pub const PBT_CSV_HEADER: &str = "generation,island_id,mutation_rate,mutation_strength,tournament_size,best_fitness,mean_fitness,gene_diversity\n";

/// Writer for streaming PBT metrics to a CSV file
pub struct PbtMetricsWriter {
    writer: BufWriter<File>,
}

impl PbtMetricsWriter {
    /// Create a new PBT metrics writer, writing the CSV header
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::new(file);
        writer.write_all(PBT_CSV_HEADER.as_bytes())?;
        writer.flush()?;

        Ok(Self { writer })
    }

    /// Write a single island's hyperparameter snapshot
    pub fn write_snapshot(&mut self, snapshot: &IslandHyperParamSnapshot) -> std::io::Result<()> {
        self.writer.write_all(snapshot.to_csv_row().as_bytes())?;
        self.writer.flush()?;
        Ok(())
    }

    /// Write snapshots for all islands at once
    pub fn write_all_islands(
        &mut self,
        snapshots: &[IslandHyperParamSnapshot],
    ) -> std::io::Result<()> {
        for snapshot in snapshots {
            self.writer.write_all(snapshot.to_csv_row().as_bytes())?;
        }
        self.writer.flush()?;
        Ok(())
    }
}

/// Summary of hyperparameter exploration across all islands
#[derive(Debug, Clone)]
pub struct PbtSummary {
    pub total_generations: u32,
    pub num_islands: usize,

    // Mutation rate range discovered
    pub min_mutation_rate: f32,
    pub max_mutation_rate: f32,
    pub final_best_mutation_rate: f32,

    // Mutation strength range discovered
    pub min_mutation_strength: f32,
    pub max_mutation_strength: f32,
    pub final_best_mutation_strength: f32,

    // Tournament size range
    pub min_tournament_size: usize,
    pub max_tournament_size: usize,
    pub final_best_tournament_size: usize,

    // Performance of best hyperparams
    pub best_island_id: usize,
    pub best_island_fitness: f32,
}

impl PbtSummary {
    /// Build summary from collected snapshots
    pub fn from_snapshots(snapshots: &[IslandHyperParamSnapshot]) -> Option<Self> {
        if snapshots.is_empty() {
            return None;
        }

        let total_generations = snapshots.iter().map(|s| s.generation).max()? + 1;
        let num_islands = snapshots.iter().map(|s| s.island_id).max()? + 1;

        // Find ranges
        let min_mutation_rate = snapshots
            .iter()
            .map(|s| s.mutation_rate)
            .fold(f32::INFINITY, f32::min);
        let max_mutation_rate = snapshots
            .iter()
            .map(|s| s.mutation_rate)
            .fold(f32::NEG_INFINITY, f32::max);

        let min_mutation_strength = snapshots
            .iter()
            .map(|s| s.mutation_strength)
            .fold(f32::INFINITY, f32::min);
        let max_mutation_strength = snapshots
            .iter()
            .map(|s| s.mutation_strength)
            .fold(f32::NEG_INFINITY, f32::max);

        let min_tournament_size = snapshots.iter().map(|s| s.tournament_size).min()?;
        let max_tournament_size = snapshots.iter().map(|s| s.tournament_size).max()?;

        // Find best performing island in the final generation
        let final_gen = total_generations - 1;
        let final_snapshots: Vec<_> = snapshots
            .iter()
            .filter(|s| s.generation == final_gen)
            .collect();

        let best_final = final_snapshots
            .iter()
            .max_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap())?;

        Some(Self {
            total_generations,
            num_islands,
            min_mutation_rate,
            max_mutation_rate,
            final_best_mutation_rate: best_final.mutation_rate,
            min_mutation_strength,
            max_mutation_strength,
            final_best_mutation_strength: best_final.mutation_strength,
            min_tournament_size,
            max_tournament_size,
            final_best_tournament_size: best_final.tournament_size,
            best_island_id: best_final.island_id,
            best_island_fitness: best_final.best_fitness,
        })
    }

    /// Print formatted summary
    pub fn print(&self) {
        println!("\n============================================================");
        println!("              PBT HYPERPARAMETER SUMMARY");
        println!("============================================================\n");

        println!(
            "Training: {} generations across {} islands\n",
            self.total_generations, self.num_islands
        );

        println!("Mutation Rate:");
        println!(
            "  Range explored:    [{:.4}, {:.4}]",
            self.min_mutation_rate, self.max_mutation_rate
        );
        println!(
            "  Best final value:  {:.4}\n",
            self.final_best_mutation_rate
        );

        println!("Mutation Strength:");
        println!(
            "  Range explored:    [{:.4}, {:.4}]",
            self.min_mutation_strength, self.max_mutation_strength
        );
        println!(
            "  Best final value:  {:.4}\n",
            self.final_best_mutation_strength
        );

        println!("Tournament Size:");
        println!(
            "  Range explored:    [{}, {}]",
            self.min_tournament_size, self.max_tournament_size
        );
        println!("  Best final value:  {}\n", self.final_best_tournament_size);

        println!("Best Performing Island:");
        println!("  Island ID:         {}", self.best_island_id);
        println!("  Final Fitness:     {:.4}", self.best_island_fitness);

        println!("\n============================================================\n");
    }
}
