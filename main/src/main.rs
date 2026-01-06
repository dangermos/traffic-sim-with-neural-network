use macroquad::prelude::*;
use neural::{Activation, Layer, Network};
use ::rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use traffic::{cars::{Car, CarWorld, Destination}, road::{Road, RoadGrid, RoadId, generate_road_grid}, simulation::Simulation};

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

fn build_level_1<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
    let horiz = screen.x * 0.4;
    let vert = screen.y * 0.3;
    let diag = screen.x.min(screen.y) * 0.35;

    let roads = vec![
        Road::new(center + vec2(-horiz, 0.0), center + vec2(horiz, 0.0), RoadId(0)),
        Road::new(center + vec2(0.0, -vert), center + vec2(0.0, vert), RoadId(1)),
        Road::new(center + vec2(-diag, -diag * 0.6), center + vec2(diag, diag * 0.6), RoadId(2)),
        Road::new(center + vec2(-diag, diag * 0.6), center + vec2(diag, -diag * 0.6), RoadId(3)),
    ];

    let road_grid = RoadGrid::new(roads);
    let cars = CarWorld::new_random(6, &road_grid, rng);
    Simulation::new(cars, road_grid)
}

fn build_level_2<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
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

fn build_level_3<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
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

fn build_straight_road_4<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {

    let roads = vec![
        Road::new(center, vec2(center.x + 200.0, center.y), RoadId(0)),
        Road::new(vec2(center.x + 200.0, center.y), vec2(center.x + 200.0, center.y - 1000.0), RoadId(1))
    ];

    let road_grid = RoadGrid::new(roads);

    let cars = CarWorld::new_random(1, &road_grid, rng);

    Simulation::new(cars, road_grid)
}

fn test_sensors<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {

    let roads = vec![
            Road::new(center, vec2(center.x + 200.0, center.y), RoadId(0)),
            Road::new(vec2(center.x + 200.0, center.y), vec2(center.x + 200.0, center.y - 1000.0), RoadId(1)),
    ];

    let road_grid = generate_road_grid(20);
    //let road_grid = RoadGrid::new(roads);

    let layers = 
        vec![ 
            Layer::new_random(4, 4, Activation::Tanh, rng),
            Layer::new_random(4, 2, Activation::Tanh, rng)
        ];
    let inputs: Vec<f32> = vec![0.0, 0.5, 0.3, 0.2];
    let network = Network::new(&layers);

    let cars = CarWorld::new(vec![
        Car::new_on_road(&road_grid, RoadId(0), GRAY, network.clone(), 0),
        Car::new_on_road(&road_grid, RoadId(1), PINK, network.clone(), 1),
        //Car::new(vec2(0.0, 0.0), GREEN, network.clone(), 2),
    ]);

    Simulation::new(cars, road_grid)

}


#[macroquad::main("Simulation Window")]
async fn main() {

    // rng and time var
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut time = 0.0;


    // Screen and Camera Variables
    let x = screen_width();
    let y = screen_height();
    let center = vec2(x * 0.5, y * 0.5);
    let base_zoom = vec2(BASE_ZOOM, BASE_ZOOM);
    let screen = vec2(x, y);



    // levels
    let mut levels = vec![
        build_level_1(center, screen, &mut rng),
        build_level_2(center, screen, &mut rng),
        build_level_3(center, screen, &mut rng),
        build_straight_road_4(center, screen, &mut rng),
        test_sensors(center, screen, &mut rng),
    ];
    let mut sim = test_sensors(center, screen, &mut rng);

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

    sim.cars.cars[0].change_state(traffic::cars::CarState::UserControlled(Destination {position: *sim.roads.roads[4].get_first_point()}));
    sim.cars.cars[1].change_state(traffic::cars::CarState::AIControlled(Destination {position: Vec2::ZERO}));
    


    println!(
    "Level Description:
     Cars: {:?}
     Roads: {:?}
     ", sim.cars.cars.iter().map(|x| x.get_id()).collect::<Vec<u16>>(), 
        sim.roads.roads.iter().map(|x| x.get_id()).collect::<Vec<RoadId>>());


    loop {

        handle_input(&mut camera);
        set_camera(&camera);
        clear_background(BEIGE);

        draw_text(format!("Time Elapsed: {:.2}", time).as_str(), camera.target.x - 100.0, camera.target.y - 300.0, 25.0, GREEN);

        sim.draw_sim(true);
        sim.update(true);


        // Logic for Debugging Sensors



        time += get_frame_time();
        next_frame().await; 
    }
    
} // End Simulation
    
