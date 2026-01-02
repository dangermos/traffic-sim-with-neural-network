use core::f32;
use std::{cmp::Ordering};
use ::rand::Rng;

use macroquad::prelude::*;
use neural::{Activation, Layer, Network};
use crate::road::{self, RoadGrid};

#[derive(Clone, Copy, Debug)]
pub struct Destination {
    pub position: Vec2,
}
#[derive(Clone, Debug)]
pub struct CarWorld {
    pub cars: Vec<Car>
}

impl CarWorld {

    pub fn new(cars: Vec<Car>) -> Self {
        Self { cars }
    }

    pub fn new_random(num_cars: i32, road_grid: &RoadGrid) -> Self {

        let mut rng = ::rand::rng();
        
        let layers = vec![
            Layer::new_random(4, 4, Activation::Tanh, &mut rng),
            Layer::new_random(4, 2, Activation::Tanh, &mut rng)
        ];

        let networks: Vec<Network> = (0..num_cars).map(
            |_| Network::new(&layers)
        ).collect();


        let cars: CarWorld = CarWorld::new((0..num_cars).into_iter().map(
            |x | { 
                let (r, g, b, _a): (u8, u8, u8, u8) = rng.random();  
                Car::new_on_road(
                    road_grid, 
            (x % (road_grid.roads.len() + 1) as i32 ) as u16,
                    Color::from_rgba(r, g, b, 255),
            networks[x as usize].clone(),
            x as u16
                )}
        ).collect());

        cars
    }

    pub fn draw_cars(&self, debug: bool) {
        self.cars.iter().for_each(
            |x| draw_car(x, debug)
        );
    }
}

#[derive(Clone, Debug)]
pub enum CarState {
    IDLE,
    MovingToDestinationAuto(Destination),
    LookingForRoad,
    UserControlled(Destination),
    AIControlled(Destination),
    ReachedDestination,
}

#[derive(Clone, Debug)]
pub struct Car {

    pub position: Vec2,
    speed: f32,
    rotation: f32,
    id: u16,

    // State Machine
    state: CarState,
    color: Color,
    destination: Option<Destination>,

    // Sensor Fields

    /// This measures how far off the road the car is. 
    off_roadness: f32,

    /// This measures how obstructed the car is, as in how 'clear' the front of the car is using a collection of rays
    obstruction_score: f32,

    /// This measures the distance of the current car to it's destination
    distance_to_destination: f32,



    // Artificial Neural Network Fields

    pub network: Network,
    

}

impl Default for Car {
    fn default() -> Self {
        Self { 
            position: Vec2 { x: 0.0, y: 0.0 }, 
            speed: 0.0, 
            rotation: 0.0, 
            id: 0, 
            state: CarState::IDLE, 
            color: WHITE, 
            destination: None, 
            off_roadness: 0.0, 
            obstruction_score: 0.0, 
            distance_to_destination: 0.0, 
            network: Network::new(&vec![])}
    }
}


impl Car {
    
    pub fn new(position: Vec2, color: Color, network: Network, car_id: u16) -> Self {
        
        Self { position, id: car_id, color, state: CarState::LookingForRoad, network,
            ..Default::default()
        }
    }

    pub fn new_on_road(road_grid: &RoadGrid, road_id: u16, color: Color, network: Network, car_id: u16) -> Self {

        let position = *road_grid[road_id].get_first_point();
        let id = car_id;


        let mut rng = ::rand::rng();

        let viable: Vec<&Vec2> = road_grid.roads.iter().map(
            |x| x.get_first_point()
        ).collect();

        let mut rand_dest = *viable[rng.random_range(0..viable.len())];

        while rand_dest == position { // Makes sure the destination is not just where the car started
            rand_dest = *viable[rng.random_range(0..viable.len())];
        }


        let destination = Some(Destination {position: rand_dest });

        let state = CarState::MovingToDestinationAuto(destination.unwrap_or(Destination { position }));


        Self { position, id, state, color, network, destination, ..Default::default() }
    }

    pub fn get_id(&self) -> u16 {
        self.id
    }

    pub fn change_state(&mut self, state: CarState) {
        match state {

            CarState::IDLE => {
                self.state = CarState::IDLE;
            },

            CarState::MovingToDestinationAuto(destination) => {
                self.state = CarState::MovingToDestinationAuto(destination);
                self.destination = Some(destination);
            },
            CarState::LookingForRoad => {
                self.state = CarState::LookingForRoad;
            },
            CarState::UserControlled(destination) => {
                self.state = CarState::UserControlled(destination);
                self.destination = Some(destination);
            },
            CarState::AIControlled(desination) => {
                self.state = CarState::AIControlled(desination);
                self.destination = Some(desination);
            }
            CarState::ReachedDestination => {
                self.state = CarState::ReachedDestination;
            },

        }
    }

    pub fn update(&mut self, road_grid: &RoadGrid, debug: bool) {

        if debug {println!("State is {:?}", self.state);}

        match &self.state {

            CarState::IDLE => {


            },

            CarState::MovingToDestinationAuto(destination) => {

                let eps: f32 = 1.0;
                let angle_eps: f32 = 0.1;

                let to_target = destination.position - self.position;
                let distance_to_target = to_target.length();

                if distance_to_target < eps { // Arrived at Destination
                    self.position = destination.position;
                    self.destination = None;
                    self.change_state(CarState::ReachedDestination);
                }

                let angle_to_target = to_target.to_angle();

                let mut err = angle_to_target - self.rotation;

                while err > std::f32::consts::PI { err -= 2.0 * f32::consts::PI; }
                while err < -std::f32::consts::PI { err += 2.0 * f32::consts::PI; }

                let facing_target= err.abs() < angle_eps;

            
                if !facing_target { self.rotate_car(if err > 0.1 {1.0} else {-1.0}); }

                
                let max_speed: f32 = 40.0;
                let slow_radius = 10.0;


                let scaled_angle = (err.cos() + 1.0) * 0.5;
                let scaled_distance = (distance_to_target / slow_radius).clamp(0.0, 1.0);

                println!("Scaled Angle: {}\nScaled Distance: {}", scaled_angle, scaled_distance);

                let desired_speed: f32 = max_speed * scaled_distance * scaled_angle;

                self.speed = desired_speed;
                self.move_car(debug);

            },

            CarState::LookingForRoad => {

                if let Some(closest) = road_grid.roads.iter().min_by(|a, b| {
                    let da = self.position.distance(*a.get_first_point());
                    let db = self.position.distance(*b.get_first_point( ));
                    da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                }) {
                    let p = closest.get_first_point();
                    println!("Closest road starts at ({:.2}, {:.2})", p.x, p.y);
                    let destination = Destination { position: Vec2::new(p.x, p.y) };
                    self.destination = Some(destination);
                    self.change_state(CarState::MovingToDestinationAuto(destination));
                }
                else {
                    println!("No Roads found on Grid.");
                }
            },

            CarState::AIControlled(destination) => { //TODO Implement Follow Road


            },

            CarState::UserControlled(destination) => {
                //let max_speed = 20.0;
                let eps = 20.0;

                let to_target = destination.position - self.position;
                let distance_to_target = to_target.length();

                if distance_to_target < eps { // Reached Destination
                    self.position = destination.position;
                    self.change_state(CarState::ReachedDestination);
                }

                self.move_car_manual(debug);

                if debug {
                    println!("Distance to Target {}", distance_to_target);
                }


            },
            
            CarState::ReachedDestination => {
                self.color.a = 0.0;
            }
        
        }
 

    }

    fn rotate_car(&mut self, amount: f32) {
        self.rotation += amount * get_frame_time()
    }

    fn move_car(&mut self, debug: bool) {
        
        let dt = get_frame_time();
        let dir = Vec2::from_angle(self.rotation);

        self.position += dir * self.speed * dt;
        
        
        if debug {println!("{}", self.position);}
    }

    fn move_car_manual(&mut self, debug: bool) {
        let dt = get_frame_time();
        let dir = Vec2::from_angle(self.rotation);

        const MOVEMENT: f32 = 50.0;
        const DELTA_ROTATION: f32 = 1.0;

        if is_key_down(KeyCode::Left) {
            self.rotation -= DELTA_ROTATION * dt
        }
        if is_key_down(KeyCode::Right) {
            self.rotation += DELTA_ROTATION * dt;
        }
        if is_key_down(KeyCode::Up) {
            self.position += dir * MOVEMENT * dt;
        }
        if is_key_down(KeyCode::Down) {
            self.position -= dir * MOVEMENT * dt;
        }

        if debug {
            println!("Keys Pressed: {:#?}\nPosition: {}", get_keys_down(), self.position);
        }

    }

    pub fn get_destination(&self) -> Option<Destination> {
        self.destination
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


    draw_triangle(v1, v2, v3, BLACK);    

    draw_text(&car.get_id().to_string(), car.position.x + dims.0 / 2.0, car.position.y, 30.0, GREEN);
    draw_text(format!("{:.0}", &car.position).as_str(), car.position.x + dims.0 / 2.0, car.position.y + 20.0, 20.0, GREEN);

    }                     


}
