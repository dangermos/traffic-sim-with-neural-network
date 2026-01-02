use macroquad::{color::{GREEN, PINK}, math::Vec2, shapes::draw_circle, text::draw_text};

use crate::{cars::{CarWorld, Destination}, road::RoadGrid};

pub struct Simulation {
    pub cars: CarWorld,
    pub roads: RoadGrid,
}

impl Simulation {

    pub fn from_sim(sim: Self) -> Self {
        Simulation { cars: sim.cars, roads: sim.roads }
    }

    pub fn draw_sim(&self, debug: bool) {
        self.roads.draw_roads(debug);
        self.cars.draw_cars(debug);

        if debug {
            self.cars.cars.iter().for_each(
                |car| {
                    let destination = car.get_destination().unwrap_or(Destination {position: Vec2::new(0.0, 0.0)});
                    draw_circle(destination.position.x, destination.position.y, 10.0, PINK);
                    draw_text(format!("Car {} {}", car.get_id(), destination.position).as_str(), destination.position.x, destination.position.y, 20.0, GREEN);
                })
            ;
        }

    }

    pub fn update(&mut self, debug: bool) {
        self.cars.cars.iter_mut().for_each(
            |x| x.update(&self.roads, debug)
        );
    }
}