use core::f32;
use std::{cmp::{self, Ordering, max}, collections::HashSet, iter::zip, vec};
use ::rand::{Rng, distr::uniform::SampleRange};

use macroquad::prelude::*;
use neural::{Activation, Layer, Network};
use crate::{road::{self, RoadGrid, RoadId}, simulation::{CarObs, Object, Simulation}};

#[derive(Clone, Copy, Debug)]
pub struct Destination {
    pub position: Vec2,
}

impl Default for Destination {
    fn default() -> Self {
        Destination { position: Vec2::ZERO }
    }
}

#[derive(Clone, Debug)]
pub struct CarWorld {
    pub cars: Vec<Car>
}

impl CarWorld {

    pub fn new(cars: Vec<Car>) -> Self {
        Self { cars }
    }

    pub fn new_random<T: Rng>(num_cars: i32, road_grid: &RoadGrid, rng: &mut T) -> Self {
        
        let layers = vec![
            Layer::new_random(4, 4, Activation::Tanh, rng),
            Layer::new_random(4, 2, Activation::Tanh, rng)
        ];

        let networks: Vec<Network> = (0..num_cars).map(
            |_| Network::new(&layers)
        ).collect();


        let cars: CarWorld = CarWorld::new((0..num_cars).into_iter().map(
            |x | { 
                let (r, g, b, _a): (u8, u8, u8, u8) = rng.random();  
                Car::new_on_road(
                    road_grid, 
            RoadId(x as usize % (road_grid.roads.len())),
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
    Crashed,
}

#[derive(Clone, Debug)]
pub struct Car {

    pub position: Vec2,
    pub speed: f32,
    pub rotation: f32,
    car_id: u16,
    pub road_id: RoadId,

    // State Machine
    state: CarState,
    color: Color,
    destination: Option<Destination>,

    // Sensor Fields

    /// This measures how far off the road the car is. 
    on_roadness: f32,

    /// This measures how obstructed the car is, as in how 'clear' the front of the car is using a collection of rays
    obstruction_score: f32,
    obstruction_rays: Option<Vec<Ray>>,

    /// This measures the distance of the current car to it's destination
    distance_to_destination: f32,

    // This measures how "aligned" the car is with it's destination
    goal_align: f32,

    // This measures how "aligned" the car is with it's current (assumed) road
    heading_error: f32,
    // Artificial Neural Network Fields

    pub network: Network,
    
    // Genome Fields
    time_spent_alive: f32,
    time_spent_off_road: f32,
    progress_to_goal: f32,


}

impl Default for Car {
    fn default() -> Self {
        Self { 
            position: Vec2 { x: 0.0, y: 0.0 }, 
            speed: 0.0, 
            rotation: 0.0, 
            car_id: 0, 
            road_id: RoadId(0),
            state: CarState::IDLE, 
            color: WHITE, 
            destination: None, 
            on_roadness: 0.0, 
            obstruction_score: 0.0, 
            distance_to_destination: 0.0,
            goal_align: 0.0, 
            heading_error: 0.0,
            network: Network::new(&vec![]),
            time_spent_alive: 0.0,
            obstruction_rays: None,

            time_spent_off_road: 0.0,
            progress_to_goal: 0.0,

        }
    }
}

#[derive(Clone, Debug)]
pub struct Ray(Vec2, Vec2);

impl Ray {
    /// Returns an `f32` if and only if a point `c` is found somewhere close to 
    /// this `Ray` within radius `r`.
    /// 
    /// The `f32` represents how close the point `c` is to the origin of the Ray
    /// 
    /// 
    ///
    pub fn occlusion(&self, c: Vec2, r: f32) -> Option<f32> {

        let origin = self.0;
        let end = self.1;

        let dist = end - origin;
        let dist2 = dist.length_squared();

        if dist2 <= 1e-6 {
            return None;
        }

        let t = ((c - origin).dot(dist) / dist2).clamp(0.0, 1.0);

        let closest = origin + dist * t;

        if (c - closest).length_squared() <= r * r {
            Some(1.0 - t)
        }
        else {
            None
        }
        
    }
}


fn manhattan_distance(v1: &Vec2, v2: &Vec2) -> f32 {
    (v2.x - v1.x).abs() - (v2.y - v1.y).abs()
}



impl Car {
    
    pub fn new(position: Vec2, color: Color, network: Network, car_id: u16) -> Self {
        
        Self { position, car_id, color, state: CarState::AIControlled(Destination { position: Vec2::ZERO}), network,
            ..Default::default()
        }
    }

    /// Spawns a Car on a Road with given `road_id`
    pub fn new_on_road(road_grid: &RoadGrid, road_id: RoadId, color: Color, network: Network, car_id: u16) -> Self {

        let position = *road_grid[road_id].get_first_point();
        let rotation = 0.0;


        let mut rng = ::rand::rng();

        let viable: Vec<&Vec2> = road_grid.roads.iter().map(
            |x| x.get_first_point()
        ).filter(|x| **x != vec2(0.0, 0.0)).collect();

        let mut rand_dest = *viable[rng.random_range(0..viable.len())];

        while rand_dest == position { // Makes sure the destination is not just where the car started
            rand_dest = *viable[rng.random_range(0..viable.len())];
        }


        let destination = Some(Destination {position: rand_dest });

        let state = CarState::AIControlled(destination.unwrap_or(Destination { position }));

        Self { position, rotation, car_id, road_id, state, color, network, destination, ..Default::default() }
    }

    pub fn get_id(&self) -> u16 {
        self.car_id
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
            CarState::Crashed => {
                self.state = CarState::Crashed;
            }

        }
    }

    pub fn update<'a>(&'a mut self, road_grid: &'a RoadGrid, objects: &HashSet<CarObs>, debug: bool) {
        let prev_distance = self.distance_to_destination;

        for object in objects {

            if self.car_id == object.id {
                continue;
            }

            if arrived(&self.position, &object.pos, 10.0) {
                self.change_state(CarState::Crashed); 
                // Potential Idea: Maybe make notification handlers to keep stdout clean. 
                // A struct that only reacts to the first occurence of some signal sent by a crash would probably suffice. 
            } 
        }



        // Obstruction handling
        self.obstruction_rays = Some(self.calculate_obstruction_rays());
        self.obstruction_score = self.get_obstruction_score(objects);


        // On-Road Score
        self.on_roadness = on_roadness(self, road_grid);

        self.distance_to_destination = manhattan_distance(&self.position, &self.destination.unwrap_or_default().position);


        // Goal Alignment
        self.goal_align = self.goal_alignment();

        // Heading Error
        self.heading_error = self.heading_error(road_grid);

        if self.on_roadness < 1.0 {
            self.time_spent_off_road += get_frame_time() * (1.0 - self.on_roadness);
        }

        if self.destination.is_some() {
            let progress = (prev_distance - self.distance_to_destination).max(0.0);
            self.progress_to_goal += progress;
        }

        match &self.state {

            CarState::IDLE => {

                self.time_spent_alive += get_frame_time();

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

                self.time_spent_alive += get_frame_time();

            },

            CarState::AIControlled(destination) => { //TODO Implement Follow Road

                let inputs = [self.obstruction_score, self.on_roadness, self.goal_align, self.heading_error];

                let result = self.network.propagate(inputs.to_vec());

                assert!(result.len() == 2, "The network has more than 2 outputs, no good!");

                println!("The network says: {:?}", result);

                let (speed, rotation) = (result[0], result[1]);

                self.speed = speed * get_frame_time() * 20.0;
                self.rotation = rotation * get_frame_time();


                // println!("Hi from car {}, I've been alive for {:.0} seconds", self.get_id(), self.time_spent_alive);

                self.time_spent_alive += get_frame_time();
                
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
                }

                self.time_spent_alive += get_frame_time();
            },
            
            CarState::ReachedDestination => {
                self.color.a = 0.2;
            },

            CarState::Crashed => {
                self.speed = 0.0;

            }
        
        }
 

    }

    fn rotate_car(&mut self, amount: f32) {
        self.rotation += amount * get_frame_time()
    }

    fn move_car(&mut self, _debug: bool) {
        
        let dt = 0.01;
        let dir = Vec2::from_angle(self.rotation);

        self.position += dir * self.speed * dt;
        
        
        //if debug {println!("{}", self.position);}
    }

    fn move_car_manual(&mut self, debug: bool) {
        let dt = 0.01;
        let dir = Vec2::from_angle(self.rotation);

        const MOVEMENT: f32 = 100.0;
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
            ("Keys Pressed: {:#?}\nPosition: {}", get_keys_down(), self.position);
        }

    }

    pub fn get_destination(&self) -> Option<Destination> {
        self.destination
    }


    /// Convert a local-space point (centered on car) into world space.
    pub fn world_from_local(&self, local: Vec2) -> Vec2 {
        self.position + rotate(local, self.rotation)
    }

    pub fn get_dims(&self) -> Vec2 {
        // however you decide size; if always small for now:
        vec2(30.0, 10.0)
    }


    // Sensor Calculation Helpers
    pub fn calculate_obstruction_rays(&self) -> Vec<Ray> {

        let dims = vec2(30.0, 10.0);

        let angles: [f32; 5] = [-0.4, -0.1, 0.0, 0.1, 0.4]; // radians

        let mut rays = Vec::new();

        for a in angles {
            let origin_local = vec2(dims.x * 0.5, 0.0); // front from center
            let start = self.world_from_local(origin_local);
            let end = start + rotate(vec2(200.0 * a.cos().powi(5), 0.0), self.rotation + a);

            rays.push(Ray(start, end));
        }

        rays
    }

    pub fn get_obstruction_score(&self, objects: &HashSet<CarObs>) -> f32 {
        let mut ray_scores: [f32; 5] = [0.0; 5];

        for (idx, ray) in self.obstruction_rays.as_ref().unwrap().iter().enumerate() {
            // Every Car has 5 obstruction rays

            let mut closest_obstruction: f32 = 0.0;

            for object in objects {

                if self.get_id() == object.id { continue; }

                let occ = ray.occlusion(object.pos, 10.0).unwrap_or(0.0);
                closest_obstruction = closest_obstruction.max(occ);
            }

            ray_scores[idx] = closest_obstruction;
        }
        ray_scores.iter().sum::<f32>() / 5.0
    }

    pub fn goal_alignment(&self) -> f32 {

        if !self.destination.is_some() {
            return 0.0;
        }


        let to_goal = self.destination.unwrap().position - self.position;
        let dist = to_goal.length();

        if dist < 1e-5 {
            return 1.0;
        }

        let goal_dir = to_goal / dist;
        let vec_direction = vec2(self.rotation.cos(), self.rotation.sin());
        vec_direction.dot(goal_dir).clamp(-1.0, 1.0)

    }

    pub fn heading_error(&self, road_grid: &RoadGrid) -> f32 {
        let vec_direction = vec2(self.rotation.cos(), self.rotation.sin());

        let road_points = &road_grid[self.road_id].points;

        let tangent = road_tangent_at_pos(self.position, &road_points);

        vec_direction.perp_dot(tangent).clamp(-1.0, 1.0)

    } 
    

}



fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
            let ab = b - a;
            let ab_len2 = ab.length_squared();
            if ab_len2 <= 1e-6 {
                return (p - a).length();
            }
            let t = ((p - a).dot(ab) / ab_len2).clamp(0.0, 1.0);
            let closest = a + ab * t;

            (p - closest).length()
        }

fn distance_to_road_centerline(p: Vec2, road_points: &[Vec2]) -> f32 {
    if road_points.len() < 2 {
        return f32::INFINITY;
    }

    let mut best = f32::INFINITY;
    for seg in road_points.windows(2) {
        let d = point_segment_distance(p, seg[0], seg[1]);
        best = best.min(d);
    }
    
    best
}

fn on_roadness(car: &mut Car, road_grid: &RoadGrid) -> f32 {

    const RECOVERY_DISTANCE: f32 = 100.0;

    let road_id = car.road_id;
    let road = &road_grid[road_id]; // This is the closest approximate road before search

    let road_width: f32 = 30.0;
    let r: f32 = road_width * 0.5;


    let road_neighbors = road_grid
        .next_roads
        .get(&road_id)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let mut distances = Vec::new();

    distances.push((road.get_id(), distance_to_road_centerline(car.position, &road.points)));

    let current_d = distances[0].1;

    if current_d > RECOVERY_DISTANCE { // If car is super far from the road it is thought to be on, run a global search

        for road in &road_grid.roads {
            distances.push((road.road_id, distance_to_road_centerline(car.position, &road.points)));   
        }

    }
    else { // If not, an easy local search is acceptable. 
        for road in road_neighbors {
            distances.push((*road, distance_to_road_centerline(car.position, &road_grid[*road].points)) );
        }
    }

    let (assumed_road, d) = distances
        .iter()
        .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
        .unwrap();

    car.road_id = *assumed_road;

    //println!("Assumed road is {:?} with a distance of {}", assumed_road, d);

    // Map distance to [0,1] where 1 = on road, 0 = far off road
    let overshoot = (d - r).max(0.0);
    let off_norm = (overshoot / r).clamp(0.0, 1.0);
    1.0 - off_norm
}

fn arrived(v1: &Vec2, v2: &Vec2, eps: f32) -> bool {
    (v1.x - v2.x).abs() < eps &&
    (v1.y - v2.y).abs() < eps
}


fn rotate(p: Vec2, theta: f32) -> Vec2 {
        Vec2::new(
            p.x * theta.cos() - p.y * theta.sin(),
            p.x * theta.sin() + p.y * theta.cos(),
        )
}

fn road_tangent_at_pos(pos: Vec2, points: &[Vec2]) -> Vec2 {

    let mut best_i: usize = 0;
    let mut best_d2 = f32::INFINITY;

    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        let ab = b - a;

        let t = ((pos - a).dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
        let proj = a + ab * t;
        let d2 = (pos - proj).length_squared();

        if d2 < best_d2 {
            best_d2 = d2;
            best_i = i;
        }
    }

    (points[best_i + 1] - points[best_i]).normalize()

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
        { offset: vec2(0.5, 0.5),
        rotation: car.rotation, color: car.color });

    // boop

    if debug {

        fn draw_obstruction_rays(car: &Car) {

            if car.obstruction_rays.is_some() {

                let rays  = car.obstruction_rays.as_ref().unwrap();

                let color = car.color.with_alpha(0.5);

                for (idx, i) in rays.iter().enumerate() {
                    draw_line(i.0.x, i.0.y, i.1.x, i.1.y, 3.0, color);
                    draw_circle_lines(i.1.x, i.1.y, 10.0, 3.0, color);
                    draw_text(format!("{}", idx).as_str(), i.1.x, i.1.y, 34.0, GREEN);
                }
            }

            else {
                return;
            }
        }


        // draw_obstruction_rays(car); // For Drawing Obstruction Rays of each Car
        draw_text(&car.get_id().to_string(), car.position.x + dims.0 / 2.0, car.position.y, 30.0, GREEN); // For drawing each Car ID next to itself
        draw_text(format!("{:.0}", &car.position).as_str(), car.position.x + dims.0 / 2.0, car.position.y + 20.0, 20.0, GREEN); // For drawing each Car's position next to itself




    }                     

}
