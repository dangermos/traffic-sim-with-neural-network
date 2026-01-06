use std::{collections::HashSet, hash::Hash};

use macroquad::{color::{GREEN, PINK}, math::Vec2, shapes::{draw_circle, draw_line}, text::draw_text};

use crate::{cars::{Car, CarWorld, Destination}, road::RoadGrid};


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
    pub objects: HashSet<CarObs>,
}

impl Simulation {

    pub fn new(cars: CarWorld, roads: RoadGrid) -> Self {
        let mut objects = HashSet::new();

        cars.cars.iter().for_each(
            |x| 
            {objects.insert(CarObs {
                id: x.get_id(),
                pos: x.position,
                rot: x.rotation,
                speed: x.speed
            });}
        );

        Simulation { cars, roads, objects }

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
                    draw_line(car.position.x, car.position.y, assumed_road.x, assumed_road.y, 2.0, GREEN);
                    
                });



        }

    }

    pub fn update(&mut self, debug: bool) {
        // Refresh observable objects with current car states before collision checks.
        self.objects.clear();
        for car in &self.cars.cars {
            self.objects.insert(CarObs {
                id: car.get_id(),
                pos: car.position,
                rot: car.rotation,
                speed: car.speed,
            });
        }

        self.cars
            .cars
            .iter_mut()
            .for_each(|x| x.update(&self.roads, &self.objects, debug));
    }
}
