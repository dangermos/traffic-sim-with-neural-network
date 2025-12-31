use macroquad::prelude::*;
use traffic::{cars::{Car, CarWorld, Destination}, road::{Road, RoadGrid}};

const BASE_ZOOM: f32 = 0.003;


fn handle_input(camera: &mut Camera2D) {

    let dt = get_frame_time();
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


struct Simulation {
    cars: CarWorld,
    roads: RoadGrid,
}

impl Simulation {
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

#[macroquad::main("Hello")]
async fn main() {

    // Screen and Camera Variables
    let x = screen_width();
    let y = screen_height();
    let (center_x, center_y) = (x / 2.0, y / 2.0);
    let base_zoom = vec2(BASE_ZOOM, BASE_ZOOM);
    
    
    // Road initialization
    let road1 = Road::new(Vec2::new(center_x + 200.0, center_y), Vec2::new(center_x + 3080.0, center_y - 1200.0), 0);
    let road_grid = RoadGrid::new(vec![road1]);
    

    // Car initialization
    let car1 = Car::new(
        Vec2::new(center_x, center_y + 200.0),
        0.0,
        PINK,
        1);
        
    let cars = CarWorld::new(vec![car1]);

    // Camera initialization
    let mut camera = Camera2D {
        target: Vec2 { x: center_x, y: center_y },
        zoom: base_zoom,
        ..Default::default()
    };

    // Sim initialization
    let sim = &mut Simulation {
        cars,
        roads: road_grid
    };

    // Update Loop
    loop {

        handle_input(&mut camera);
        set_camera(&camera);
        clear_background(BEIGE);

        sim.draw_sim(true);
        sim.update(true);

        next_frame().await


    }
} // End Simulation
    
