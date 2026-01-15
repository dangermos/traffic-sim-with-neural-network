use macroquad::{
    color::Color,
    math::{Vec2, vec2},
};
use neural::{LayerTopology, Network};
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

pub fn build_level_3<T: Rng>(_center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
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

pub fn build_straight_road_4<T: Rng>(center: Vec2, _screen: Vec2, rng: &mut T) -> Simulation {
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

pub fn build_straight_line_level<T: Rng>(start: Vec2, length: f32, rng: &mut T) -> Simulation {
    // Two colinear segments to satisfy RoadGrid's requirement for multiple roads.
    let mid = start + vec2(length * 0.5, 0.0);
    let end = start + vec2(length, 0.0);

    let roads = vec![
        Road::new(start, mid, RoadId(0)),
        Road::new(mid, end, RoadId(1)),
    ];

    let road_grid = RoadGrid::new(roads);
    let cars = CarWorld::new_random(6, &road_grid, rng);
    Simulation::new(cars, road_grid)
}

pub fn test_sensors<T: Rng>(_center: Vec2, _screen: Vec2, rng: &mut T) -> Simulation {

    const NUM_ROADS: usize = 20;
    const NUM_CARS: usize = 100;

    let road_grid = generate_road_grid(NUM_ROADS, rng);
    //let road_grid = RoadGrid::new(roads);

    let topology = [
        LayerTopology { neurons: 5 },
        LayerTopology { neurons: 5 },
        LayerTopology { neurons: 2 },
    ];

    
    let cars = CarWorld { cars:
        (0..NUM_CARS).map(|x| {
                let network = Network::new_random(&topology, rng);
                let (r,g, b, _a): (u8, u8, u8, u8)  = rng.random();
                let color = Color::from_rgba(r, g, b, 255);
                Car::new_on_road(&road_grid, RoadId(x % NUM_ROADS), color, network, x as u16)
            })
        .collect()
    };

    Simulation::new(cars, road_grid)
}

/// A comprehensive training level for overnight evolution runs.
/// 
/// Features:
/// - Large interconnected grid with varied path lengths
/// - Multiple difficulty zones (easy straight sections + complex intersections)
/// - Enough cars to create interesting collision avoidance scenarios
/// - Roads of varying lengths to test both short and long navigation
pub fn overnight_training<T: Rng>(rng: &mut T) -> Simulation {
    const CARS_PER_ROAD: usize = 4;
    
    // Build a proper city grid with intersections
    let city_center = vec2(960.0, 540.0);
    let block_size = 300.0;
    let grid_size = 5; // 5x5 grid of blocks
    
    let half_span = block_size * (grid_size as f32) * 0.5;
    
    let mut roads: Vec<Road> = Vec::new();
    let mut id: usize = 0;
    
    // Horizontal roads (west to east)
    for row in 0..=grid_size {
        let y = city_center.y - half_span + row as f32 * block_size;
        let start = vec2(city_center.x - half_span, y);
        let end = vec2(city_center.x + half_span, y);
        roads.push(Road::new(start, end, RoadId(id)));
        id += 1;
    }
    
    // Vertical roads (north to south)
    for col in 0..=grid_size {
        let x = city_center.x - half_span + col as f32 * block_size;
        let start = vec2(x, city_center.y - half_span);
        let end = vec2(x, city_center.y + half_span);
        roads.push(Road::new(start, end, RoadId(id)));
        id += 1;
    }
    
    // Diagonal roads for variety (corner to corner shortcuts)
    roads.push(Road::new(
        vec2(city_center.x - half_span, city_center.y - half_span),
        vec2(city_center.x + half_span, city_center.y + half_span),
        RoadId(id),
    ));
    id += 1;
    
    roads.push(Road::new(
        vec2(city_center.x - half_span, city_center.y + half_span),
        vec2(city_center.x + half_span, city_center.y - half_span),
        RoadId(id),
    ));
    id += 1;
    
    // Curved bypass roads around the edges (forms a diamond/ring road)
    let outer_offset = half_span + 150.0;
    roads.push(Road::new(
        vec2(city_center.x - outer_offset, city_center.y),
        vec2(city_center.x, city_center.y - outer_offset),
        RoadId(id),
    ));
    id += 1;
    
    roads.push(Road::new(
        vec2(city_center.x, city_center.y - outer_offset),
        vec2(city_center.x + outer_offset, city_center.y),
        RoadId(id),
    ));
    id += 1;
    
    roads.push(Road::new(
        vec2(city_center.x + outer_offset, city_center.y),
        vec2(city_center.x, city_center.y + outer_offset),
        RoadId(id),
    ));
    id += 1;
    
    roads.push(Road::new(
        vec2(city_center.x, city_center.y + outer_offset),
        vec2(city_center.x - outer_offset, city_center.y),
        RoadId(id),
    ));
    
    let road_grid = RoadGrid::new(roads);
    let road_count = road_grid.roads.len().max(1);
    // Scale car count with road count to avoid overcrowding.
    let num_cars = road_count * CARS_PER_ROAD;
    
    let topology = [
        LayerTopology { neurons: 5 },
        LayerTopology { neurons: 8 },  // Slightly larger hidden layer for complex behavior
        LayerTopology { neurons: 2 },
    ];
    
    let cars = CarWorld {
        cars: (0..num_cars)
            .map(|x| {
                let network = Network::new_random(&topology, rng);
                let (r, g, b, _a): (u8, u8, u8, u8) = rng.random();
                let color = Color::from_rgba(r, g, b, 255);
                Car::new_on_road(&road_grid, RoadId(x % road_count), color, network, x as u16)
            })
            .collect(),
    };
    
    Simulation::new(cars, road_grid)
}
