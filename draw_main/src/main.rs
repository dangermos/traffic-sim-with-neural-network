use genetics::{Individual, Population};
use macroquad::prelude::*;
use ::rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use bincode::{serialize, deserialize};
use traffic::levels::test_sensors;

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


    let mut rng = ChaCha8Rng::seed_from_u64(42);
    // Size Variables
    let x = 1920.0;
    let y = 1080.0;
    let center = vec2(x * 0.5, y * 0.5);
    let screen = vec2(x, y);

    let base_zoom = vec2(BASE_ZOOM, BASE_ZOOM);

    let mut sim = test_sensors(center, screen, &mut rng);


    let initial_individuals: Vec<Individual> = sim
        .cars
        .cars
        .iter()
        .map(|car| {
            let genes = car.network.to_genes();
            let fitness = genetics::fitness(car);
            Individual { genes, fitness }
        })
        .collect();

    let expected_pop_size = initial_individuals.len();

    let checkpoint_path = "individuals.bin";
    let i = if std::path::Path::new(checkpoint_path).exists() {
        let data = std::fs::read(checkpoint_path).expect("Failed to read checkpoint file");
        let individuals: Vec<Individual> =
            deserialize(&data).expect("Failed to deserialize individuals");
        if individuals.len() != expected_pop_size {
            panic!(
                "Checkpoint population size {} does not match expected size {}",
                individuals.len(),
                expected_pop_size
            );
        }
        individuals
    } else {
        initial_individuals
    };

    let mut population = Population {
        individuals: i,
        generation: 0,
    };


    
    let mut sim = traffic::levels::test_sensors(
        vec2(screen_width() * 0.5, screen_height() * 0.5),
        vec2(screen_width(), screen_height()),
        &mut rng,
    );

    // Camera initialization
    let mut camera = Camera2D {
        target: sim.cars.cars[0].position,
        zoom: base_zoom,
        ..Default::default()
    };


    loop {
        handle_input(&mut camera);
        set_camera(&camera);
        clear_background(BEIGE);
        sim.draw_sim(true);
        sim.update(true);
        // Yield to the macroquad event loop so the window can redraw.
        next_frame().await;
    }
}
