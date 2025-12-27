use core::f32;
use std::{cmp::Ordering, vec};

use macroquad::prelude::*;

use crate::road::{RoadGrid};

#[derive(Clone, Copy, Debug)]
pub struct Destination {
    pub position: Vec2,
}

pub struct CarWorld {
    cars: Vec<Car>
}

#[derive(Debug)]
pub enum CarState {
    IDLE,
    MovingToDestination(Destination),
    LookingForRoad(RoadGrid),
    MovingOnRoad()
}

pub struct Car {

    pub position: Vec2,
    speed: f32,
    rotation: f32,

    // State Machine
    state: CarState,
    color: Color,



    // Vector Algebra Fields
    

}


impl Car {
    
    pub fn new(position: Vec2, speed: f32, color: Color, road_grid: RoadGrid) -> Self {
        Self {
            position, speed, color, rotation: 0.0,
            state: CarState::LookingForRoad(road_grid),
        }
    }

    fn change_state(&mut self, state: CarState) {
        match state {
            CarState::IDLE => {
                self.state = CarState::IDLE;
            },

            CarState::MovingToDestination(destination) => {
                self.state = CarState::MovingToDestination(destination);
            },
            CarState::LookingForRoad(road_grid) => {
                self.state = CarState::LookingForRoad(road_grid);
            },
            CarState::MovingOnRoad() => {

            }
        }
    }

    pub fn update(&mut self) {

        println!("State is {:?}", self.state);

        match &self.state {

            CarState::IDLE => {},

            CarState::MovingToDestination(destination) => {

                let eps: f32 = 1.0;
                let angle_eps: f32 = 0.1;

                let to_target = destination.position - self.position;
                let distance_to_target = to_target.length();

                if distance_to_target < eps {
                    self.position = destination.position;
                    self.change_state(CarState::IDLE);
                }

                let angle_to_target = to_target.to_angle();

                let mut err = angle_to_target - self.rotation;

                while err > std::f32::consts::PI { err -= 2.0 * f32::consts::PI; }
                while err < -std::f32::consts::PI { err += 2.0 * f32::consts::PI; }

                let facing_target= err.abs() < angle_eps;

            
                if !facing_target { self.rotate_car(if err > 0.1 {1.0} else {-1.0}); }

                
                let max_speed: f32 = 20.0;
                let slow_radius = 10.0;


                let scaled_angle = (err.cos() + 1.0) * 0.5;
                let scaled_distance = (distance_to_target / slow_radius).clamp(0.0, 1.0);

                println!("Scaled Angle: {}\nScaled Distance: {}", scaled_angle, scaled_distance);

                let desired_speed: f32 = max_speed * scaled_distance * scaled_angle;

                self.speed = desired_speed;
                self.move_car();

            },
            CarState::LookingForRoad(road_grid) => {

                if let Some(closest) = road_grid.roads.iter().min_by(|a, b| {
                    let da = self.position.distance(*a.get_first_point());
                    let db = self.position.distance(*b.get_first_point());
                    da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                }) {
                    let p = closest.get_first_point();
                    println!("Closest road starts at ({:.2}, {:.2})", p.x, p.y);

                    self.change_state(CarState::MovingToDestination(Destination { position: Vec2::new(p.x, p.y) }));
                }
                else {
                    println!("No Roads found on Grid.");
                }
            }

            CarState::MovingOnRoad() => {

            }
        }


    }

    fn rotate_car(&mut self, amount: f32) {
        self.rotation += amount * get_frame_time()
    }

    fn move_car(&mut self) {
        
        let dt = get_frame_time();
        let dir = Vec2::from_angle(self.rotation);

        self.position += dir * self.speed * dt;
        
        
        println!("{}", self.position);
    }



}


fn arrived(v1: &Vec2, v2: &Vec2, eps: f32) -> bool {
    (v1.x - v2.x).abs() < eps &&
    (v1.y - v2.y).abs() < eps
}


pub fn draw_car(car: &Car, debug: bool) {

    let size = "small";

    let dims = match size {
        "small" => {
            (30.0, 10.0)
        },
        "med" => {
            (50.0, 20.0)
        },
        _ => {
            (0.0, 0.0);
            panic!("Please specify a car size")
        }
    };

    draw_rectangle_ex(car.position.x, car.position.y, dims.0, dims.1, 
        DrawRectangleParams 
        { offset: Vec2 { x: 0.0, y: 0.0 },
        rotation: car.rotation, color: car.color });

    // boop

    fn rotate(p: Vec2, theta: f32) -> Vec2 {
        Vec2::new(
            p.x * theta.cos() - p.y * theta.sin(),
            p.x * theta.sin() + p.y * theta.cos(),
        )
    }

    if debug {

    let a1 = Vec2::new(dims.0, dims.1 * 0.25);
    let a2 = Vec2::new(dims.0, dims.1 * 0.75);
    let a3 = Vec2::new(dims.0 + dims.1, dims.1 * 0.50);

    let v1 = car.position + rotate(a1, car.rotation);
    let v2 = car.position + rotate(a2, car.rotation);
    let v3 = car.position + rotate(a3, car.rotation);


    draw_triangle(v1, v2, v3, GOLD);    

    }                     


}
