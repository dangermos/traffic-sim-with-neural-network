use macroquad::prelude::*;
use traffic::{cars::{Car, draw_car}, road::{Road, RoadGrid, draw_road_grid}};

const BASE_ZOOM: f32 = 0.001;


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
        camera.zoom.x = (camera.zoom.x * factor).clamp(0.0001, 0.01);
        camera.zoom.y = (camera.zoom.y * factor).clamp(0.0001, 0.01);
    }
    if is_key_down(KeyCode::Minus) {
        let factor = 0.95;
        camera.zoom.x = (camera.zoom.x * factor).clamp(0.0001, 0.01);
        camera.zoom.y = (camera.zoom.y * factor).clamp(0.0001, 0.01);
    }

    let scroll = mouse_wheel().1;
    if scroll.abs() > f32::EPSILON {
        let factor = 1.0 + scroll * 0.05;
        camera.zoom.x = (camera.zoom.x * factor).clamp(0.0001, 0.01);
        camera.zoom.y = (camera.zoom.y * factor).clamp(0.0001, 0.01);
    }

}


#[macroquad::main("Hello")]
async fn main() {

    // Screen and Camera Variables
    let x = screen_width();
    let y = screen_height();
    let (center_x, center_y) = (x / 2.0, y / 2.0);
    let base_zoom = vec2(BASE_ZOOM, BASE_ZOOM);
    
    
    // Road and Car initialization
    let road1 = Road::new(Vec2::new(center_x + 200.0, center_y), Vec2::new(center_x + 3080.0, center_y - 1200.0), 0);
    let road_grid = RoadGrid::new(vec![road1]);
    
    let mut car1 = Car::new(
        Vec2::new(center_x, center_y + 200.0),
        0.0,
        PINK);
        
    let mut camera = Camera2D {
        target: Vec2 { x: center_x, y: center_y },
        zoom: base_zoom,
        ..Default::default()
    };
    //set_default_camera();

    // Update Loop
    loop {

        handle_input(&mut camera);
        set_camera(&camera);
        clear_background(BEIGE);
        car1.update(&road_grid);

        draw_road_grid(&road_grid, true);
        draw_car(&car1, true);

        next_frame().await


    }
} // End Simulation
    
