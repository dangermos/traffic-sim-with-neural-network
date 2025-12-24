use std::vec;

use macroquad::prelude::*;
use traffic::{cars::{Car, CarState, Destination, draw_car}, road::{self, Road, RoadGrid, draw_road, draw_road_grid}};




#[macroquad::main("Hello")]
async fn main() {

    let x = screen_width();
    let y = screen_height();

    let (center_x, center_y) = (x / 2.0, y / 2.0);
    
    
    let road1 = Road::new(Vec2::new(center_x + 200.0, center_y), Vec2::new(center_x + 3080.0, center_y - 1200.0), 0);
    let road_grid = RoadGrid::new(vec![road1]);
    
    let mut car1 = Car::new(
        Vec2::new(center_x, center_y + 200.0),
        0.0, 
        PINK,
        road_grid.clone());


    loop {

        clear_background(GREEN);


        car1.update();
        
        draw_road_grid(&road_grid, true);
        draw_car(&car1, true);
        next_frame().await


    }
}
    
