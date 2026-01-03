use macroquad::prelude::*;
use neural::{Activation, Layer, Network};
use traffic::{cars::CarWorld, road::{Road, RoadGrid}, simulation::Simulation};

const BASE_ZOOM: f32 = 0.003;
const ACTIVE_LEVEL: usize = 4;


fn handle_input(camera: &mut Camera2D) {

    let dt = 0.01;
    let pan_speed = if is_key_down(KeyCode::LeftShift) { 2.0 } else { 1.0 };
    let pan_step = pan_speed * dt / camera.zoom.x.abs().max(0.0005);

    if is_key_down(KeyCode::W) {
        camera.target.y -= pan_step;
    }
    if is_key_down(KeyCode::A) {
        camera.target.x -= pan_step;
    }
    if is_key_down(KeyCode::S) {
        camera.target.y += pan_step;
    }
    if is_key_down(KeyCode::D) {
        camera.target.x += pan_step;
    }

    if is_key_pressed(KeyCode::R) {
        camera.target = vec2(screen_width() * 0.5, screen_height() * 0.5);
        camera.zoom = vec2(BASE_ZOOM, BASE_ZOOM);
    }

    if is_key_down(KeyCode::Equal) {
        let factor = 1.05;
        camera.zoom.x = (camera.zoom.x * factor).clamp(0.0001, 0.003);
        camera.zoom.y = (camera.zoom.y * factor).clamp(0.0001, 0.003);
    }
    if is_key_down(KeyCode::Minus) {
        let factor = 0.95;
        camera.zoom.x = (camera.zoom.x * factor).clamp(0.0001, 0.003);
        camera.zoom.y = (camera.zoom.y * factor).clamp(0.0001, 0.003);
    }

    let scroll = mouse_wheel().1;
    if scroll.abs() > f32::EPSILON {
        let factor = 1.0 + scroll * 0.05;
        camera.zoom.x = (camera.zoom.x * factor).clamp(0.0001, 0.005);
        camera.zoom.y = (camera.zoom.y * factor).clamp(0.0001, 0.005);
    }

}

fn build_level_1(center: Vec2, screen: Vec2) -> Simulation {
    let horiz = screen.x * 0.4;
    let vert = screen.y * 0.3;
    let diag = screen.x.min(screen.y) * 0.35;

    let roads = vec![
        Road::new(center + vec2(-horiz, 0.0), center + vec2(horiz, 0.0), 0),
        Road::new(center + vec2(0.0, -vert), center + vec2(0.0, vert), 1),
        Road::new(center + vec2(-diag, -diag * 0.6), center + vec2(diag, diag * 0.6), 2),
        Road::new(center + vec2(-diag, diag * 0.6), center + vec2(diag, -diag * 0.6), 3),
    ];

    let road_grid = RoadGrid::new(roads);
    let cars = CarWorld::new_random(6, &road_grid);
    Simulation { cars, roads: road_grid }
}

fn build_level_2(center: Vec2, screen: Vec2) -> Simulation {
    let margin = 120.0;
    let left = margin;
    let right = screen.x - margin;
    let top = margin;
    let bottom = screen.y - margin;

    let roads = vec![
        Road::new(vec2(left, top), vec2(right, top), 0),
        Road::new(vec2(right, top), vec2(right, bottom), 1),
        Road::new(vec2(right, bottom), vec2(left, bottom), 2),
        Road::new(vec2(left, bottom), vec2(left, top), 3),
        Road::new(vec2(left, top), vec2(right, bottom), 4),
        Road::new(vec2(left, bottom), vec2(right, top), 5),
        Road::new(vec2(left, center.y), vec2(right, center.y), 6),
        Road::new(vec2(center.x, top), vec2(center.x, bottom), 7),
    ];

    let road_grid = RoadGrid::new(roads);
    let cars = CarWorld::new_random(12, &road_grid);
    Simulation { cars, roads: road_grid }
}

fn build_level_3(center: Vec2, screen: Vec2) -> Simulation {
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
        Road::new(vec2(left, h1), vec2(right, h1), 0),
        Road::new(vec2(left, h2), vec2(right, h2), 1),
        Road::new(vec2(v1, top), vec2(v1, bottom), 2),
        Road::new(vec2(v2, top), vec2(v2, bottom), 3),
        Road::new(vec2(left, top), vec2(right, bottom), 4),
    ];

    let road_grid = RoadGrid::new(roads);
    let cars = CarWorld::new_random(8, &road_grid);
    Simulation { cars, roads: road_grid }
}

fn build_straight_road_4(center: Vec2, screen: Vec2) -> Simulation {

    let roads = vec![
        Road::new(center, vec2(center.x + 200.0, center.y), 0),
        Road::new(vec2(center.x + 200.0, center.y), vec2(center.x + 200.0, center.y - 1000.0), 1)
    ];

    let road_grid = RoadGrid::new(roads);

    let cars = CarWorld::new_random(1, &road_grid);

    Simulation { cars, roads: road_grid }
}


#[macroquad::main("Simulation Window")]
async fn main() {

    // rng and time var
    let mut rng = ::rand::rng();
    let mut time = 0.0;


    // Screen and Camera Variables
    let x = screen_width();
    let y = screen_height();
    let center = vec2(x * 0.5, y * 0.5);
    let base_zoom = vec2(BASE_ZOOM, BASE_ZOOM);
    let screen = vec2(x, y);



    // levels
    let mut levels = vec![
        build_level_1(center, screen),
        build_level_2(center, screen),
        build_level_3(center, screen),
        build_straight_road_4(center, screen)
    ];
    let mut sim = build_straight_road_4(center, screen);

    // Neural Network Initialization
    let layers = 
        vec![ 
            Layer::new_random(4, 4, Activation::Tanh, &mut rng),
            Layer::new_random(4, 2, Activation::Tanh, &mut rng)
        ];
    let inputs: Vec<f32> = vec![0.0, 0.5, 0.3, 0.2];
    let network = Network::new(&layers);
    let network2 = Network::new(&layers);

    // Camera initialization
    let mut camera = Camera2D {
        target: center,
        zoom: base_zoom,
        ..Default::default()
    };


    loop {

        handle_input(&mut camera);
        set_camera(&camera);
        clear_background(BEIGE);

        draw_text(format!("Time Elapsed: {:.2}", time).as_str(), camera.target.x - 100.0, camera.target.y - 300.0, 25.0, GREEN);

        sim.draw_sim(true);
        sim.update(true);

        time += get_frame_time();
        next_frame().await; 
    }
    
} // End Simulation
    
