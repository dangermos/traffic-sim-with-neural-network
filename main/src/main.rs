use ::rand::{Rng, SeedableRng};
use genetics::{Individual, Population, evolve_generation, tournament_select};
use macroquad::prelude::*;
use rand_chacha::ChaCha8Rng;
use traffic::{
    cars::{Car, CarWorld, Destination},
    levels::test_sensors,
    road::{Road, RoadGrid, RoadId, generate_road_grid},
    simulation::Simulation,
};

const BASE_ZOOM: f32 = 0.003;
fn handle_input(camera: &mut Camera2D) {
    let dt = 0.01;
    let pan_speed = if is_key_down(KeyCode::LeftShift) {
        2.0
    } else {
        1.0
    };
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

#[macroquad::main("Simulation Window")]
async fn main() {
    // rng and time var
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Screen and Camera Variables
    let x = screen_width();
    let y = screen_height();
    let center = vec2(x * 0.5, y * 0.5);
    let base_zoom = vec2(BASE_ZOOM, BASE_ZOOM);
    let screen = vec2(x, y);

    // Pick a Level
    let mut sim = test_sensors(center, screen, &mut rng);

    // Camera initialization
    let mut camera = Camera2D {
        target: center,
        zoom: base_zoom,
        ..Default::default()
    };

    println!(
        "Level Description:
     Cars: {:?}
     Roads: {:?}
     ",
        sim.cars
            .cars
            .iter()
            .map(|x| x.get_id())
            .collect::<Vec<u16>>(),
        sim.roads
            .roads
            .iter()
            .map(|x| x.get_id())
            .collect::<Vec<RoadId>>()
    );
    // This controls how many times the simulation runs before running fitness evaluation
    const EPOCHS: usize = 5;
    const MAX_FRAMES: usize = 500;

    const DRAW: bool = true;

    let individuals: Vec<Individual> = sim
        .cars
        .cars
        .iter()
        .map(|car| {
            let genes = car.network.to_genes();
            let fitness = genetics::fitness(car);
            Individual { genes, fitness }
        })
        .collect();

    let mut population = Population {
        individuals,
        generation: 0,
    };

    for generation in 0..EPOCHS {
        for frame in 0..MAX_FRAMES {
            if DRAW {
                handle_input(&mut camera);
                set_camera(&camera);
                clear_background(BEIGE);
                sim.draw_sim(false);
            }

            sim.update(true);

            // Logic for Debugging Sensors
            if DRAW {
                next_frame().await;
            }
            println!("Frame {}", frame);
        }

        let new = evolve_generation(&population, 3, 0.2, 0.2, &mut rng);
    }
} // End Simulation
