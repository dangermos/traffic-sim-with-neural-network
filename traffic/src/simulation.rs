use std::{collections::HashMap, hash::Hash};

use crate::{
    cars::{CarState, CarWorld},
    road::RoadGrid,
};
use macroquad::{
    color::{Color, GREEN, PINK},
    math::Vec2,
    prelude::set_default_camera,
    shapes::{draw_circle, draw_line, draw_rectangle},
    text::draw_text,
};
use rayon::prelude::*;

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
                crashed: matches!(x.state(), CarState::Crashed | CarState::ReachedDestination),
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
            let mut best_progress = f32::NEG_INFINITY;
            let mut best_car = None;

            for car in &self.cars.cars {
                match car.state() {
                    CarState::Crashed => crashed += 1,
                    CarState::ReachedDestination => reached += 1,
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

            let lines = [
                format!(
                    "Cars: alive {} | reached {} | crashed {}",
                    alive, reached, crashed
                ),
                format!("Avg speed: {:.1}", avg_speed),
                format!("Best progress: {:.0}", best_progress.max(0.0)),
            ];

            let reached_color = Color::from_rgba(120, 210, 140, 220);
            let crashed_color = Color::from_rgba(210, 80, 80, 220);

            let panel_x = 16.0;
            let panel_y = 16.0;
            let padding = 12.0;
            let line_h = 22.0;
            let legend_icon = 12.0;
            let legend_spacing = 8.0;

            let legend = [("Reached", reached_color), ("Crashed", crashed_color)];

            let panel_w = 300.0;
            let panel_h = padding * 2.0
                + line_h * lines.len() as f32
                + legend_spacing
                + (legend_icon + legend_spacing) * legend.len() as f32;

            draw_rectangle(
                panel_x,
                panel_y,
                panel_w,
                panel_h,
                Color::from_rgba(20, 30, 40, 180),
            );

            let start_x = panel_x + padding;
            let start_y = panel_y + padding + line_h;

            for (i, line) in lines.iter().enumerate() {
                draw_text(line, start_x, start_y + i as f32 * line_h, 22.0, GREEN);
            }

            let legend_start_y = start_y + line_h * lines.len() as f32 + legend_spacing;
            for (i, (label, color)) in legend.iter().enumerate() {
                let y = legend_start_y + i as f32 * (legend_icon + legend_spacing);
                draw_rectangle(start_x, y, legend_icon, legend_icon, *color);
                draw_text(
                    label,
                    start_x + legend_icon + 8.0,
                    y + legend_icon,
                    20.0,
                    GREEN,
                );
            }
        }
    }

    pub fn update(&mut self, debug: bool) {
        // Refresh observable objects with current car states before collision checks.
        self.objects.clear();
        self.objects.extend(self.cars.cars.iter().map(|car| CarObs {
            id: car.get_id(),
            pos: car.position,
            rot: car.rotation,
            speed: car.speed,
            crashed: matches!(
                car.state(),
                CarState::Crashed | CarState::ReachedDestination
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
                grid.collect_neighbors(car.position, 200.0, objects, neighbors);
                car.update(roads, neighbors.as_slice(), debug);
            },
        );

        // NOTE: We no longer remove cars mid-simulation to preserve alignment between
        // population.individuals and sim.cars.cars for correct fitness attribution.
        // Stagnant cars now transition to Crashed state instead of being removed.
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
