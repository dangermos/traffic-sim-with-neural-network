use ::rand::Rng;
use core::f32;
use std::{cmp::Ordering, vec};

use crate::{
    road::{RoadGrid, RoadId},
    simulation::{CarObs, SimConfig},
};
use macroquad::prelude::*;
use neural::{LayerTopology, Network};

#[derive(Clone, Copy, Debug)]
pub struct Destination {
    pub position: Vec2,
}

impl Default for Destination {
    fn default() -> Self {
        Destination {
            position: Vec2::ZERO,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CarWorld {
    pub cars: Vec<Car>,
}

impl CarWorld {
    pub fn new(cars: Vec<Car>) -> Self {
        Self { cars }
    }

    pub fn new_random<T: Rng>(num_cars: i32, road_grid: &RoadGrid, rng: &mut T) -> Self {
        const INPUTS: usize = 5;
        const HIDDEN: usize = 5;
        const OUTPUTS: usize = 2;

        let topology = [
            LayerTopology { neurons: INPUTS },
            LayerTopology { neurons: HIDDEN },
            LayerTopology { neurons: OUTPUTS },
        ];

        let road_count = road_grid.roads.len();

        let cars = (0..num_cars)
            .into_iter()
            .map(|x| {
                let (r, g, b, _a): (u8, u8, u8, u8) = rng.random();
                let network = Network::new_random(&topology, rng);
                Car::new_on_road(
                    road_grid,
                    RoadId(x as usize % road_count),
                    Color::from_rgba(r, g, b, 255),
                    network,
                    x as u16,
                )
            })
            .collect();

        CarWorld::new(cars)
    }

    pub fn draw_cars(&self, debug: bool) {
        self.cars.iter().for_each(|x| draw_car(x, debug));
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
    Stagnant,
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
    obstruction_rays: Vec<Ray>,

    /// This measures the distance of the current car to it's destination
    distance_to_destination: f32,

    // This measures how "aligned" the car is with it's destination
    goal_align: f32,

    // This measures how "aligned" the car is with it's current (assumed) road
    heading_error: f32,
    // Artificial Neural Network Fields
    pub network: Network,
    /// Scratch buffers for zero-allocation neural network propagation
    nn_buf_a: Vec<f32>,
    nn_buf_b: Vec<f32>,

    // Genome Fields
    pub time_spent_alive: f32,
    pub time_spent_off_road: f32,
    pub progress_to_goal: f32,
    pub distance_traveled: f32,
    pub initial_distance_to_goal: f32,
    stagnant_steps: u32,
    pub remove_flag: bool,
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
            nn_buf_a: Vec::new(),
            nn_buf_b: Vec::new(),
            obstruction_rays: Vec::new(),
            time_spent_alive: 0.0,
            time_spent_off_road: 0.0,
            progress_to_goal: 0.0,
            distance_traveled: 0.0,
            initial_distance_to_goal: 0.0,
            stagnant_steps: 0,
            remove_flag: false,
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
        } else {
            None
        }
    }
}

fn manhattan_distance(v1: &Vec2, v2: &Vec2) -> f32 {
    (v2.x - v1.x).abs() + (v2.y - v1.y).abs()
}

impl Car {
    pub fn new(position: Vec2, color: Color, network: Network, car_id: u16) -> Self {
        Self {
            position,
            car_id,
            color,
            state: CarState::AIControlled(Destination {
                position: Vec2::ZERO,
            }),
            network,
            ..Default::default()
        }
    }

    /// Spawns a Car on a Road with given `road_id`
    pub fn new_on_road(
        road_grid: &RoadGrid,
        road_id: RoadId,
        color: Color,
        network: Network,
        car_id: u16,
    ) -> Self {
        // Deterministic spawn: pick a starting point on the road based on the car id.
        // This reduces evaluation noise across generations.
        let viable: Vec<&Vec2> = road_grid
            .roads
            .iter()
            .map(|x| x.get_first_point())
            .filter(|x| **x != vec2(0.0, 0.0))
            .collect();

        // Deterministic position along the chosen road, spread out to avoid collisions.
        // Use golden ratio-based distribution for better spacing - this ensures cars
        // are spread across the road even with sequential IDs on the same road.
        let road_points = &road_grid[road_id].points;
        let num_points = road_points.len();

        const GOLDEN_RATIO_CONJUGATE: f32 = 0.618033988749895;
        let fractional_pos = ((car_id as f32) * GOLDEN_RATIO_CONJUGATE) % 1.0;

        // Map to road points, but avoid the very start/end where intersections occur.
        // Use indices from 5% to 95% of the road to avoid clustering at endpoints.
        let safe_start = num_points / 20; // 5% into the road
        let safe_end = num_points - (num_points / 20); // 95% of the road
        let safe_range = safe_end.saturating_sub(safe_start).max(1);
        let position_index = safe_start + ((fractional_pos * safe_range as f32) as usize);
        let position_index = position_index.min(num_points - 1);

        let position = road_points[position_index];

        // Calculate initial rotation to face along the road direction
        let rotation = if position_index + 1 < num_points {
            let next_point = road_points[position_index + 1];
            let direction = next_point - position;
            direction.y.atan2(direction.x)
        } else if position_index > 0 {
            let prev_point = road_points[position_index - 1];
            let direction = position - prev_point;
            direction.y.atan2(direction.x)
        } else {
            0.0
        };

        // Deterministic destination selection (based on car_id)
        // Use a different prime to avoid correlation with position selection
        let mut dest_index = ((car_id as usize).wrapping_mul(17).wrapping_add(11)) % viable.len();
        let mut rand_dest = *viable[dest_index];

        // Make sure the destination is not exactly the spawn spot
        while rand_dest == position {
            dest_index = (dest_index + 1) % viable.len();
            rand_dest = *viable[dest_index];
        }

        let destination = Some(Destination {
            position: rand_dest,
        });

        let state = CarState::AIControlled(destination.unwrap_or(Destination { position }));

        // Use manhattan distance to match how progress_to_goal is accumulated.
        let initial_distance_to_goal = manhattan_distance(&position, &rand_dest);

        // Pre-allocate NN buffers based on network size
        let buf_size = network.max_layer_size();

        Self {
            position,
            rotation,
            car_id,
            road_id,
            state,
            color,
            nn_buf_a: Vec::with_capacity(buf_size),
            nn_buf_b: Vec::with_capacity(buf_size),
            network,
            destination,
            initial_distance_to_goal,
            ..Default::default()
        }
    }

    pub fn get_id(&self) -> u16 {
        self.car_id
    }

    pub fn change_state(&mut self, state: CarState) {
        match state {
            CarState::IDLE => {
                self.state = CarState::IDLE;
            }

            CarState::MovingToDestinationAuto(destination) => {
                self.state = CarState::MovingToDestinationAuto(destination);
                self.destination = Some(destination);
            }
            CarState::LookingForRoad => {
                self.state = CarState::LookingForRoad;
            }
            CarState::UserControlled(destination) => {
                self.state = CarState::UserControlled(destination);
                self.destination = Some(destination);
            }
            CarState::AIControlled(desination) => {
                self.state = CarState::AIControlled(desination);
                self.destination = Some(desination);
            }
            CarState::ReachedDestination => {
                self.state = CarState::ReachedDestination;
            }
            CarState::Stagnant => {
                self.state = CarState::Stagnant;
            }
            CarState::Crashed => {
                self.state = CarState::Crashed;
            }
        }
    }

    /// Update with default config (collisions and occlusion enabled)
    pub fn update<'a>(&'a mut self, road_grid: &'a RoadGrid, objects: &[CarObs], debug: bool) {
        self.update_with_config(road_grid, objects, debug, SimConfig::default());
    }

    /// Update with custom config for toggling collisions/occlusion
    pub fn update_with_config<'a>(
        &'a mut self,
        road_grid: &'a RoadGrid,
        objects: &[CarObs],
        debug: bool,
        config: SimConfig,
    ) {
        let prev_distance = self.distance_to_destination;
        let prev_pos = self.position;

        // Collision detection (can be disabled via config)
        if config.enable_collisions {
            for object in objects {
                // Skip self
                if self.car_id == object.id {
                    continue;
                }
                // Skip crashed or finished cars - they shouldn't cause new collisions
                if object.crashed {
                    continue;
                }

                if arrived(&self.position, &object.pos, 10.0) {
                    self.change_state(CarState::Crashed);
                }
            }
        }

        // Obstruction handling (can be disabled via config)
        if config.enable_occlusion {
            self.calculate_obstruction_rays();
            self.obstruction_score = self.get_obstruction_score(objects);
        } else {
            // When occlusion is disabled, report "no obstructions"
            self.obstruction_score = 0.0;
        }

        // On-Road Score
        self.on_roadness = on_roadness(self, road_grid);

        self.distance_to_destination = manhattan_distance(
            &self.position,
            &self.destination.unwrap_or_default().position,
        );

        // Goal Alignment
        self.goal_align = self.goal_alignment();

        // Heading Error
        self.heading_error = self.heading_error(road_grid);

        if self.on_roadness < 1.0 {
            self.time_spent_off_road += 1.0 * (1.0 - self.on_roadness);
        }

        if self.destination.is_some() {
            let progress = (prev_distance - self.distance_to_destination).max(0.0);
            self.progress_to_goal += progress;
        }

        match &self.state {
            CarState::IDLE => {}

            CarState::MovingToDestinationAuto(destination) => {
                let eps: f32 = 1.0;
                let angle_eps: f32 = 0.1;

                let to_target = destination.position - self.position;
                let distance_to_target = to_target.length();

                if distance_to_target < eps {
                    // Arrived at Destination
                    self.position = destination.position;
                    self.destination = None;
                    self.change_state(CarState::ReachedDestination);
                }

                let angle_to_target = to_target.to_angle();

                let mut err = angle_to_target - self.rotation;

                while err > std::f32::consts::PI {
                    err -= 2.0 * f32::consts::PI;
                }
                while err < -std::f32::consts::PI {
                    err += 2.0 * f32::consts::PI;
                }

                let facing_target = err.abs() < angle_eps;

                if !facing_target {
                    self.rotate_car(if err > 0.1 { 1.0 } else { -1.0 });
                }

                let max_speed: f32 = 40.0;
                let slow_radius = 10.0;

                let scaled_angle = (err.cos() + 1.0) * 0.5;
                let scaled_distance = (distance_to_target / slow_radius).clamp(0.0, 1.0);

                let desired_speed: f32 = max_speed * scaled_distance * scaled_angle;

                self.speed = desired_speed;
                self.move_car(debug);
            }

            CarState::LookingForRoad => {
                if let Some(closest) = road_grid.roads.iter().min_by(|a, b| {
                    let da = self.position.distance(*a.get_first_point());
                    let db = self.position.distance(*b.get_first_point());
                    da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                }) {
                    let p = closest.get_first_point();
                    println!("Closest road starts at ({:.2}, {:.2})", p.x, p.y);
                    let destination = Destination {
                        position: Vec2::new(p.x, p.y),
                    };
                    self.destination = Some(destination);
                    self.change_state(CarState::MovingToDestinationAuto(destination));
                } else {
                    println!("No Roads found on Grid.");
                }

                self.time_spent_alive += 1.0;
            }

            CarState::AIControlled(destination) => {
                let dt = 0.01;

                // If we're very close to our destination, register arrival and transition.
                let eps: f32 = 15.0; // Increased from 10.0 for easier arrival
                if self.distance_to_destination < eps {
                    // Arrived at Destination
                    self.position = destination.position;
                    self.destination = None;
                    self.change_state(CarState::ReachedDestination);
                } else {
                    const MAX_SPEED: f32 = 80.0;
                    const MAX_ANGULAR_VELOCITY: f32 = 2.5;
                    const ACCELERATION: f32 = 120.0;
                    const MIN_SPEED_FOR_STEERING: f32 = 5.0;

                    // Proximity assist kicks in when close to goal
                    const ASSIST_RANGE: f32 = 150.0; // Start assisting within this distance
                    const MAX_ASSIST_STRENGTH: f32 = 0.4; // Max blend toward direct steering

                    let inputs = [
                        self.obstruction_score, // 0 -> 1
                        self.on_roadness,       // 0 -> 1
                        self.goal_align,        // -1 -> 1
                        self.heading_error,     // -1 -> 1
                        self.speed / MAX_SPEED, // 0 -> 1
                    ];

                    let result = self.network.propagate_into(
                        &inputs,
                        &mut self.nn_buf_a,
                        &mut self.nn_buf_b,
                    );

                    let throttle = result[0].clamp(-1.0, 1.0);
                    let nn_steering = result[1].clamp(-1.0, 1.0);

                    assert!(
                        result.len() == 2,
                        "The network has more than 2 outputs, no good!"
                    );

                    // Calculate direct-to-goal steering (what steering SHOULD be to face goal)
                    let to_goal = destination.position - self.position;
                    let goal_angle = to_goal.y.atan2(to_goal.x);
                    let angle_diff = (goal_angle - self.rotation).sin(); // sin gives signed error

                    // Proximity factor: 0 when far, approaches 1 when very close
                    let proximity =
                        (1.0 - self.distance_to_destination / ASSIST_RANGE).clamp(0.0, 1.0);
                    let assist_strength = proximity * MAX_ASSIST_STRENGTH;

                    // Blend neural network steering with direct-to-goal steering
                    let steering =
                        nn_steering * (1.0 - assist_strength) + angle_diff * assist_strength;
                    let steering = steering.clamp(-1.0, 1.0);

                    // Apply physics
                    self.speed = (self.speed + throttle * ACCELERATION * dt).clamp(0.0, MAX_SPEED);

                    // Steering effectiveness scales with speed - can't turn well at low speeds
                    // This prevents cars from spinning in place
                    let speed_factor = (self.speed / MIN_SPEED_FOR_STEERING).clamp(0.0, 1.0);
                    self.rotation += steering * MAX_ANGULAR_VELOCITY * speed_factor * dt;

                    self.distance_traveled += self.speed * dt;
                    self.move_car(debug);

                    self.time_spent_alive += 1.0;
                }
            }

            CarState::UserControlled(destination) => {
                let eps = 20.0;

                let to_target = destination.position - self.position;
                let distance_to_target = to_target.length();

                if distance_to_target < eps {
                    // Reached Destination
                    self.position = destination.position;
                    self.change_state(CarState::ReachedDestination);
                }

                self.move_car_manual(debug);

                if debug {}

                self.time_spent_alive += 1.0;
            }

            CarState::ReachedDestination => {
                self.color.a = 0.2;
            }

            CarState::Stagnant => {
                self.speed = 0.0;
            }

            CarState::Crashed => {
                self.speed = 0.0;
            }
        }

        // Stagnation detection: mark cars as crashed if they don't move for a while.
        // NOTE: We no longer set remove_flag here to avoid corrupting fitness attribution
        // during evolution. Instead, stagnant cars transition to Stagnant state.
        const STAGNANT_MOVEMENT_EPS: f32 = 0.05;
        const STAGNANT_FRAMES: u32 = 300;

        let terminal_state = matches!(
            self.state(),
            CarState::ReachedDestination | CarState::Crashed | CarState::Stagnant
        );
        if !terminal_state {
            let moved = (self.position - prev_pos).length();
            if moved < STAGNANT_MOVEMENT_EPS && self.speed.abs() < 0.1 {
                self.stagnant_steps = self.stagnant_steps.saturating_add(1);
                if self.stagnant_steps > STAGNANT_FRAMES {
                    // Mark as stagnant instead of removing - preserves population alignment
                    self.change_state(CarState::Stagnant);
                }
            } else {
                self.stagnant_steps = 0;
            }
        }
    }

    fn rotate_car(&mut self, amount: f32) {
        self.rotation += amount * 1.0
    }

    fn move_car(&mut self, _debug: bool) {
        let dt = 0.01;
        let dir = Vec2::from_angle(self.rotation);

        self.position += dir * self.speed * dt;
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
            (
                "Keys Pressed: {:#?}\nPosition: {}",
                get_keys_down(),
                self.position,
            );
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
    pub fn calculate_obstruction_rays(&mut self) {
        let dims = vec2(30.0, 10.0);

        let angles: [f32; 5] = [-0.4, -0.1, 0.0, 0.1, 0.4]; // radians

        self.obstruction_rays.clear();
        self.obstruction_rays
            .reserve_exact(angles.len().saturating_sub(self.obstruction_rays.len()));

        for a in angles {
            let origin_local = vec2(dims.x * 0.5, 0.0); // front from center
            let start = self.world_from_local(origin_local);
            let end = start + rotate(vec2(200.0 * a.cos().powi(5), 0.0), self.rotation + a);

            self.obstruction_rays.push(Ray(start, end));
        }
    }

    pub fn get_obstruction_score(&self, objects: &[CarObs]) -> f32 {
        let mut ray_scores: [f32; 5] = [0.0; 5];

        for (idx, ray) in self.obstruction_rays.iter().enumerate() {
            // Every Car has 5 obstruction rays

            let mut closest_obstruction: f32 = 0.0;

            for object in objects {
                if self.get_id() == object.id {
                    continue;
                }

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

        let mut tangent = road_tangent_at_pos(self.position, &road_points);

        // Ensure tangent points toward destination, not away from it
        // This gives the neural network a consistent signal
        if let Some(dest) = &self.destination {
            let to_goal = dest.position - self.position;
            if tangent.dot(to_goal) < 0.0 {
                tangent = -tangent; // Flip tangent to point toward goal
            }
        }

        // perp_dot gives signed angle: positive = need to turn left, negative = turn right
        vec_direction.perp_dot(tangent).clamp(-1.0, 1.0)
    }

    pub fn state(&self) -> CarState {
        self.state.clone()
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

    distances.push((
        road.get_id(),
        distance_to_road_centerline(car.position, &road.points),
    ));

    let current_d = distances[0].1;

    if current_d > RECOVERY_DISTANCE {
        // If car is super far from the road it is thought to be on, run a global search

        for road in &road_grid.roads {
            distances.push((
                road.road_id,
                distance_to_road_centerline(car.position, &road.points),
            ));
        }
    } else {
        // If not, an easy local search is acceptable.
        for road in road_neighbors {
            distances.push((
                *road,
                distance_to_road_centerline(car.position, &road_grid[*road].points),
            ));
        }
    }

    let (assumed_road, d) = distances
        .iter()
        .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
        .unwrap();

    car.road_id = *assumed_road;

    // Map distance to [0,1] where 1 = on road, 0 = far off road
    let overshoot = (d - r).max(0.0);
    let off_norm = (overshoot / r).clamp(0.0, 1.0);
    1.0 - off_norm
}

fn arrived(v1: &Vec2, v2: &Vec2, eps: f32) -> bool {
    (v1.x - v2.x).abs() < eps && (v1.y - v2.y).abs() < eps
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
        "small" => (30.0, 10.0),
        "med" => (50.0, 20.0),
        _ => {
            (0.0, 0.0);
            panic!("Please specify a car size")
        }
    };

    let mut display_color = car.color;
    match car.state() {
        CarState::ReachedDestination => {
            display_color = Color::from_rgba(120, 210, 140, 220);
        }
        CarState::Stagnant => {
            display_color = Color::from_rgba(230, 200, 80, 210);
        }
        CarState::Crashed => {
            display_color = Color::from_rgba(210, 80, 80, 200);
        }
        _ => {}
    }

    draw_rectangle_ex(
        car.position.x,
        car.position.y,
        dims.0,
        dims.1,
        DrawRectangleParams {
            offset: vec2(0.5, 0.5),
            rotation: car.rotation,
            color: display_color,
        },
    );

    if debug {
        // Outline aligned to rotation.
        let hx = dims.0 * 0.5;
        let hy = dims.1 * 0.5;
        let corners = [vec2(-hx, -hy), vec2(hx, -hy), vec2(hx, hy), vec2(-hx, hy)];

        for idx in 0..4 {
            let a = car.world_from_local(corners[idx]);
            let b = car.world_from_local(corners[(idx + 1) % 4]);
            draw_line(a.x, a.y, b.x, b.y, 2.0, display_color.with_alpha(0.6));
        }

        let nose = car.world_from_local(vec2(dims.0 * 0.6, 0.0));
        draw_line(
            car.position.x,
            car.position.y,
            nose.x,
            nose.y,
            2.0,
            display_color.with_alpha(0.8),
        );

        draw_text(
            format!("#{}", car.get_id()).as_str(),
            car.position.x - dims.0 * 0.2,
            car.position.y - dims.1 * 0.4,
            18.0,
            GREEN,
        );

        // Destination hint: short arrow from car toward its goal (no full line).
        if let Some(dest) = car.get_destination() {
            let to_goal = dest.position - car.position;
            let dist = to_goal.length();
            if dist > 1.0 {
                let dir = to_goal / dist;
                let arrow_len = dist.min(40.0);
                let head_len = 8.0;
                let start = car.position;
                let end = start + dir * arrow_len;
                let arrow_color = Color::from_rgba(255, 210, 100, 220);

                draw_line(start.x, start.y, end.x, end.y, 2.0, arrow_color);

                let head_dir = -dir * head_len;
                let left = end + rotate(head_dir, 0.6);
                let right = end + rotate(head_dir, -0.6);
                draw_line(end.x, end.y, left.x, left.y, 2.0, arrow_color);
                draw_line(end.x, end.y, right.x, right.y, 2.0, arrow_color);
            }
        }
    }
}
