use std::{collections::HashMap, hash::Hash};

use crate::{
    cars::{CarState, CarWorld},
    road::RoadGrid,
};
use macroquad::{
    color::{Color, PINK},
    math::Vec2,
    prelude::set_default_camera,
    shapes::{draw_circle, draw_line, draw_rectangle, draw_rectangle_lines},
    text::{draw_text, measure_text},
};
use rayon::prelude::*;

/// Configuration for simulation behavior.
/// Use this to toggle expensive features for testing/benchmarking.
#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    /// Enable collision detection between cars
    pub enable_collisions: bool,
    /// Enable obstruction ray calculations (car "vision")
    pub enable_occlusion: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            enable_collisions: true,
            enable_occlusion: true,
        }
    }
}

#[derive(Clone, Copy)]
pub struct CarObs {
    pub id: u16,
    pub pos: Vec2,
    pub rot: f32,
    pub speed: f32,
    pub crashed: bool,
}

impl PartialEq for CarObs {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.pos == other.pos
    }
}

impl Eq for CarObs {}

impl Hash for CarObs {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

pub trait Object {
    fn get_position(&self) -> Vec2;
}

pub struct Simulation {
    pub cars: CarWorld,
    pub roads: RoadGrid,
    pub objects: Vec<CarObs>,
    grid: SpatialGrid,
}

impl Simulation {
    pub fn new(cars: CarWorld, roads: RoadGrid) -> Self {
        let mut objects = Vec::new();

        cars.cars.iter().for_each(|x| {
            objects.push(CarObs {
                id: x.get_id(),
                pos: x.position,
                rot: x.rotation,
                speed: x.speed,
                crashed: matches!(
                    x.state(),
                    CarState::Crashed | CarState::ReachedDestination | CarState::Stagnant
                ),
            })
        });

        Simulation {
            cars,
            roads,
            objects,
            grid: SpatialGrid::new(150.0),
        }
    }

    pub fn draw_sim(&self, debug: bool) {
        self.roads.draw_roads(debug);
        self.cars.draw_cars(debug);

        if debug {
            let mut total_speed = 0.0;
            let mut alive = 0usize;
            let mut reached = 0usize;
            let mut crashed = 0usize;
            let mut stagnant = 0usize;
            let mut best_progress = f32::NEG_INFINITY;
            let mut best_car = None;

            for car in &self.cars.cars {
                match car.state() {
                    CarState::Crashed => crashed += 1,
                    CarState::ReachedDestination => reached += 1,
                    CarState::Stagnant => stagnant += 1,
                    _ => alive += 1,
                }

                total_speed += car.speed;

                if car.progress_to_goal > best_progress {
                    best_progress = car.progress_to_goal;
                    best_car = Some(car);
                }
            }

            let count = self.cars.cars.len() as f32;
            let avg_speed = if count > 0.0 {
                total_speed / count
            } else {
                0.0
            };

            if let Some(car) = best_car {
                if let Some(dest) = car.get_destination() {
                    let line_color = PINK.with_alpha(0.4);
                    draw_line(
                        car.position.x,
                        car.position.y,
                        dest.position.x,
                        dest.position.y,
                        2.0,
                        line_color,
                    );
                } else {
                    draw_circle(car.position.x, car.position.y, 12.0, PINK.with_alpha(0.4));
                }
            }

            // Switch to screen-space for HUD so it stays pinned to the viewport.
            set_default_camera();
            let total = alive + reached + stagnant + crashed;
            let total_f = total.max(1) as f32;

            let panel_bg = Color::from_rgba(18, 24, 30, 210);
            let panel_border = Color::from_rgba(70, 90, 110, 200);
            let panel_shadow = Color::from_rgba(0, 0, 0, 80);
            let accent_color = Color::from_rgba(80, 190, 160, 220);
            let text_primary = Color::from_rgba(220, 235, 250, 235);
            let text_muted = Color::from_rgba(150, 165, 180, 210);

            let active_color = Color::from_rgba(90, 160, 230, 220);
            let reached_color = Color::from_rgba(120, 210, 140, 220);
            let stagnant_color = Color::from_rgba(230, 200, 80, 220);
            let crashed_color = Color::from_rgba(210, 80, 80, 220);

            let panel_x = 18.0;
            let panel_y = 18.0;
            let panel_w = 340.0;
            let padding = 14.0;
            let title_size = 22.0;
            let text_size = 18.0;
            let footer_size = 16.0;
            let line_h = 22.0;
            let bar_h = 10.0;
            let gap = 8.0;

            let rows = [
                ("Active", alive, active_color),
                ("Reached", reached, reached_color),
                ("Stagnant", stagnant, stagnant_color),
                ("Crashed", crashed, crashed_color),
            ];

            let header_h = title_size + footer_size + 12.0;
            let stats_h = rows.len() as f32 * line_h;
            let footer_h = 2.0 * line_h;
            let panel_h = padding * 2.0 + header_h + stats_h + gap + bar_h + gap + footer_h;

            draw_rectangle(panel_x + 4.0, panel_y + 4.0, panel_w, panel_h, panel_shadow);
            draw_rectangle(panel_x, panel_y, panel_w, panel_h, panel_bg);
            draw_rectangle(panel_x, panel_y, panel_w, 3.0, accent_color);
            draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 1.0, panel_border);

            let mut cursor_y = panel_y + padding + title_size;
            draw_text("SIM STATUS", panel_x + padding, cursor_y, title_size, text_primary);

            cursor_y += footer_size + 6.0;
            let total_text = format!("Total cars: {}", total);
            draw_text(&total_text, panel_x + padding, cursor_y, footer_size, text_muted);

            cursor_y += line_h;

            for (i, (label, count, color)) in rows.iter().enumerate() {
                let y = cursor_y + i as f32 * line_h;
                let pct = if total > 0 {
                    (*count as f32 / total_f) * 100.0
                } else {
                    0.0
                };
                let value_text = format!("{} ({:.0}%)", count, pct);

                draw_rectangle(panel_x + padding, y - 10.0, 10.0, 10.0, *color);
                draw_text(label, panel_x + padding + 16.0, y, text_size, text_primary);

                let dims = measure_text(&value_text, None, text_size as u16, 1.0);
                draw_text(
                    &value_text,
                    panel_x + panel_w - padding - dims.width,
                    y,
                    text_size,
                    text_primary,
                );
            }

            let bar_x = panel_x + padding;
            let bar_y = cursor_y + rows.len() as f32 * line_h + gap;
            let bar_w = panel_w - padding * 2.0;
            draw_rectangle(bar_x, bar_y, bar_w, bar_h, Color::from_rgba(30, 40, 50, 220));
            draw_rectangle_lines(bar_x, bar_y, bar_w, bar_h, 1.0, panel_border);

            let mut bar_cursor = bar_x;
            for (_label, count, color) in rows.iter() {
                let width = if total > 0 {
                    bar_w * (*count as f32 / total_f)
                } else {
                    0.0
                };
                if width > 0.0 {
                    draw_rectangle(bar_cursor, bar_y, width, bar_h, *color);
                }
                bar_cursor += width;
            }

            let footer_start_y = bar_y + bar_h + gap + footer_size;
            let footer_rows = [
                ("Avg speed", format!("{:.1}", avg_speed)),
                ("Best progress", format!("{:.0}", best_progress.max(0.0))),
            ];

            for (i, (label, value)) in footer_rows.iter().enumerate() {
                let y = footer_start_y + i as f32 * line_h;
                draw_text(label, panel_x + padding, y, footer_size, text_muted);
                let dims = measure_text(value, None, footer_size as u16, 1.0);
                draw_text(
                    value,
                    panel_x + panel_w - padding - dims.width,
                    y,
                    footer_size,
                    text_primary,
                );
            }
        }
    }

    /// Update with default config (collisions and occlusion enabled)
    pub fn update(&mut self, debug: bool) {
        self.update_with_config(debug, SimConfig::default());
    }

    /// Update with custom config for toggling features
    pub fn update_with_config(&mut self, debug: bool, config: SimConfig) {
        // Refresh observable objects with current car states before collision checks.
        self.objects.clear();
        self.objects.extend(self.cars.cars.iter().map(|car| CarObs {
            id: car.get_id(),
            pos: car.position,
            rot: car.rotation,
            speed: car.speed,
            crashed: matches!(
                car.state(),
                CarState::Crashed | CarState::ReachedDestination | CarState::Stagnant
            ),
        }));

        self.grid.rebuild(&self.objects);

        let roads = &self.roads;
        let objects = &self.objects;
        let grid = &self.grid;

        self.cars.cars.par_iter_mut().for_each_init(
            || Vec::with_capacity(32),
            |neighbors, car| {
                neighbors.clear();
                // Only collect neighbors if we need them for collisions or occlusion
                if config.enable_collisions || config.enable_occlusion {
                    grid.collect_neighbors(car.position, 200.0, objects, neighbors);
                }
                car.update_with_config(roads, neighbors.as_slice(), debug, config);
            },
        );

        // NOTE: We no longer remove cars mid-simulation to preserve alignment between
        // population.individuals and sim.cars.cars for correct fitness attribution.
        // Stagnant cars now transition to Stagnant state instead of being removed.
        // self.cars.cars.retain(|c| !c.remove_flag);
    }
}

struct SpatialGrid {
    cell_size: f32,
    buckets: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            buckets: HashMap::new(),
        }
    }

    fn rebuild(&mut self, objects: &[CarObs]) {
        self.buckets.clear();
        for (idx, obj) in objects.iter().enumerate() {
            let key = Self::key(obj.pos, self.cell_size);
            self.buckets.entry(key).or_default().push(idx);
        }
    }

    fn collect_neighbors(
        &self,
        position: Vec2,
        radius: f32,
        objects: &[CarObs],
        out: &mut Vec<CarObs>,
    ) {
        out.clear();
        let radius_sq = radius * radius;

        let (cx, cy) = Self::key(position, self.cell_size);

        for dy in -1..=1 {
            for dx in -1..=1 {
                let key = (cx + dx, cy + dy);
                if let Some(indices) = self.buckets.get(&key) {
                    for i in indices {
                        let candidate = objects[*i];
                        if (candidate.pos - position).length_squared() <= radius_sq {
                            out.push(candidate);
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn key(pos: Vec2, cell_size: f32) -> (i32, i32) {
        (
            (pos.x / cell_size).floor() as i32,
            (pos.y / cell_size).floor() as i32,
        )
    }
}