use std::vec;

use macroquad::{miniquad::PassAction, prelude::{scene::camera_pos, *}};
use traffic::{cars::{Car, CarState, Destination, draw_car}, road::{self, Road, RoadGrid, draw_road, draw_road_grid}};


fn handle_input(camera: &mut Camera2D) {

    let c= if is_key_down(KeyCode::LeftShift) {20.0} else {10.0};
    let mut scale: f32 = c / (camera.zoom.x * 10.0);
    println!("Camera Zoom: {}\nAttempted Scale: {}", camera.zoom, scale);

    if is_key_down(KeyCode::W) {
        camera.target.y -= scale * get_frame_time();
    }
    if is_key_down(KeyCode::A) {
        camera.target.x -= scale * get_frame_time();
    }
    if is_key_down(KeyCode::S) {
        camera.target.y += scale * get_frame_time();
    }
    if is_key_down(KeyCode::D) {
        camera.target.x += scale * get_frame_time();
    }

    if is_key_down(KeyCode::R) {
        set_default_camera();
    }

    if is_key_pressed(KeyCode::Equal) {
        camera.zoom.x += 0.001;
        camera.zoom.y += 0.001;
        if camera.zoom.x < 0.0001 {
            camera.zoom.x += 0.0005;
            camera.zoom.y += 0.0005;
        }
    }
    if is_key_pressed(KeyCode::Minus) {
        camera.zoom.x -= 0.001;
        camera.zoom.y -= 0.001;

        if camera.zoom.x < 0.0001 {
            camera.zoom.x -= 0.0005;
            camera.zoom.y -= 0.0005;

        }    
    }

    println!("Cam Position: {}", camera.target);

}


#[macroquad::main("Hello")]
async fn main() {

    // Screen and Camera Variables
    let x = screen_width();
    let y = screen_height();
    let (center_x, center_y) = (x / 2.0, y / 2.0);
    
    
    // Road and Car initialization
    let road1 = Road::new(Vec2::new(center_x + 200.0, center_y), Vec2::new(center_x + 3080.0, center_y - 1200.0), 0);
    let road_grid = RoadGrid::new(vec![road1]);
    
    let mut car1 = Car::new(
        Vec2::new(center_x, center_y + 200.0),
        0.0, 
        PINK,
        road_grid.clone());
        
    let mut camera = Camera2D {
        target: Vec2 { x: center_x, y: center_y },
        zoom: Vec2 { x: 0.001, y: 0.001 },
        ..Default::default()
    };
    //set_default_camera();

    // Update Loop
    loop {

        clear_background(BEIGE);
        car1.update();
        
        draw_road_grid(&road_grid, true);
        draw_car(&car1, true);

        handle_input(&mut camera);
        set_camera(&camera);
        next_frame().await


    }
} // End Simulation
    
