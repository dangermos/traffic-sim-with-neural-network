use std::{collections::HashMap, hash::Hash};

use macroquad::{color::{GREEN, PINK}, math::Vec2, shapes::{draw_circle, draw_line}, text::draw_text};
use rayon::prelude::*;
use crate::{cars::{CarWorld, Destination}, road::RoadGrid};


#[derive(Clone, Copy)]
pub struct CarObs {
    pub id: u16,
    pub pos: Vec2,
    pub rot: f32,
    pub speed: f32,
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

        cars
            .cars
            .iter()
            .for_each(|x| objects.push(CarObs {
                id: x.get_id(),
                pos: x.position,
                rot: x.rotation,
                speed: x.speed,
            }));

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


            self.cars.cars.iter().for_each(
                |car| {
                    let assumed_road = self.roads[car.road_id].get_first_point();

                    let destination = car.get_destination().unwrap_or(Destination {position: Vec2::new(0.0, 0.0)});
                    draw_circle(destination.position.x, destination.position.y, 2.0, PINK);
                    draw_text(format!("Car {} {}", car.get_id(), destination.position).as_str(), destination.position.x, destination.position.y, 20.0, GREEN);
                    draw_line(car.position.x, car.position.y, assumed_road.x, assumed_road.y, 1.0, GREEN);
                });

            let min_dist = 
                self.cars
                .cars
                .iter()
                .min_by(|x, y| 
                    x.progress_to_goal.partial_cmp(&y.progress_to_goal).unwrap());
                
            if let Some(car) = min_dist {
                let destination = car.get_destination().unwrap_or_default();
                draw_line(car.position.x, car.position.y, destination.position.x, destination.position.y, 20.0, PINK);

            }
        }
    }

    pub fn update(&mut self, debug: bool) {
        // Refresh observable objects with current car states before collision checks.
        self.objects.clear();
        self.objects
            .extend(self.cars.cars.iter().map(|car| CarObs {
                id: car.get_id(),
                pos: car.position,
                rot: car.rotation,
                speed: car.speed,
            }));

        self.grid.rebuild(&self.objects);

        let roads = &self.roads;
        let objects = &self.objects;
        let grid = &self.grid;

        self.cars
            .cars
            .par_iter_mut()
            .for_each_init(
                || Vec::with_capacity(32),
                |neighbors, car| {
                    neighbors.clear();
                    grid.collect_neighbors(car.position, 200.0, objects, neighbors);
                    car.update(roads, neighbors.as_slice(), debug);
                },
            );
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

    fn collect_neighbors(&self, position: Vec2, radius: f32, objects: &[CarObs], out: &mut Vec<CarObs>) {
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
