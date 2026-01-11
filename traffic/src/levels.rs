use macroquad::{
    color::{GRAY, PINK},
    math::{Vec2, vec2},
};
use neural::{Activation, Layer, Network};
use rand::Rng;

use crate::{
    cars::{Car, CarWorld},
    road::{Road, RoadGrid, RoadId, generate_road_grid},
    simulation::Simulation,
};

pub fn build_level_1<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
    let horiz = screen.x * 0.4;
    let vert = screen.y * 0.3;
    let diag = screen.x.min(screen.y) * 0.35;

    let roads = vec![
        Road::new(
            center + vec2(-horiz, 0.0),
            center + vec2(horiz, 0.0),
            RoadId(0),
        ),
        Road::new(
            center + vec2(0.0, -vert),
            center + vec2(0.0, vert),
            RoadId(1),
        ),
        Road::new(
            center + vec2(-diag, -diag * 0.6),
            center + vec2(diag, diag * 0.6),
            RoadId(2),
        ),
        Road::new(
            center + vec2(-diag, diag * 0.6),
            center + vec2(diag, -diag * 0.6),
            RoadId(3),
        ),
    ];

    let road_grid = RoadGrid::new(roads);
    let cars = CarWorld::new_random(6, &road_grid, rng);
    Simulation::new(cars, road_grid)
}

pub fn build_level_2<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
    let margin = 120.0;
    let left = margin;
    let right = screen.x - margin;
    let top = margin;
    let bottom = screen.y - margin;

    let roads = vec![
        Road::new(vec2(left, top), vec2(right, top), RoadId(0)),
        Road::new(vec2(right, top), vec2(right, bottom), RoadId(1)),
        Road::new(vec2(right, bottom), vec2(left, bottom), RoadId(2)),
        Road::new(vec2(left, bottom), vec2(left, top), RoadId(3)),
        Road::new(vec2(left, top), vec2(right, bottom), RoadId(4)),
        Road::new(vec2(left, bottom), vec2(right, top), RoadId(5)),
        Road::new(vec2(left, center.y), vec2(right, center.y), RoadId(6)),
        Road::new(vec2(center.x, top), vec2(center.x, bottom), RoadId(7)),
    ];

    let road_grid = RoadGrid::new(roads);
    let cars = CarWorld::new_random(12, &road_grid, rng);
    Simulation::new(cars, road_grid)
}

pub fn build_level_3<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
    let margin = 80.0;
    let left = margin;
    let right = screen.x - margin;
    let top = margin;
    let bottom = screen.y - margin;
    let h1 = top + (bottom - top) * 0.33;
    let h2 = top + (bottom - top) * 0.66;
    let v1 = left + (right - left) * 0.33;
    let v2 = left + (right - left) * 0.66;

    let roads = vec![
        Road::new(vec2(left, h1), vec2(right, h1), RoadId(0)),
        Road::new(vec2(left, h2), vec2(right, h2), RoadId(1)),
        Road::new(vec2(v1, top), vec2(v1, bottom), RoadId(2)),
        Road::new(vec2(v2, top), vec2(v2, bottom), RoadId(3)),
        Road::new(vec2(left, top), vec2(right, bottom), RoadId(4)),
    ];

    let road_grid = RoadGrid::new(roads);
    let cars = CarWorld::new_random(8, &road_grid, rng);
    Simulation::new(cars, road_grid)
}

pub fn build_straight_road_4<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
    let roads = vec![
        Road::new(center, vec2(center.x + 200.0, center.y), RoadId(0)),
        Road::new(
            vec2(center.x + 200.0, center.y),
            vec2(center.x + 200.0, center.y - 1000.0),
            RoadId(1),
        ),
    ];

    let road_grid = RoadGrid::new(roads);

    let cars = CarWorld::new_random(1, &road_grid, rng);

    Simulation::new(cars, road_grid)
}

pub fn test_sensors<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {

    const NUM_ROADS: usize = 20;
    const NUM_CARS: usize = 100;

    let road_grid = generate_road_grid(NUM_ROADS, rng);
    //let road_grid = RoadGrid::new(roads);

    let layers = vec![
        Layer::new_random(5, 5, Activation::Tanh, rng),
        Layer::new_random(5, 2, Activation::Tanh, rng),
    ];
    let network = Network::new(&layers);


    let cars = CarWorld { cars:
        (0..NUM_CARS).map(|x| {
                Car::new_on_road(&road_grid, RoadId(x % NUM_ROADS), GRAY, network.clone(), x as u16)
            })
        .collect()
    };
        //Car::new_on_road(&road_grid, RoadId(1), PINK, network.clone(), 1),



    Simulation::new(cars, road_grid)
}
