use macroquad::{
    color::Color,
    math::{Vec2, vec2},
};
use neural::{LayerTopology, Network};
use rand::Rng;
use std::f32::consts::PI;

use crate::{
    cars::{Car, CarWorld},
    road::{Road, RoadGrid, RoadId, generate_road_grid},
    simulation::Simulation,
};

/// Creates a road from a series of waypoints using Catmull-Rom spline interpolation.
/// This produces smooth curves that pass through all the control points.
fn road_from_waypoints(waypoints: Vec<Vec2>, id: RoadId, samples_per_segment: usize) -> Road {
    if waypoints.len() < 2 {
        panic!("Need at least 2 waypoints to create a road");
    }

    let mut points: Vec<Vec2> = Vec::new();

    // For Catmull-Rom, we need 4 points per segment.
    // We'll duplicate endpoints to handle the boundaries.
    let n = waypoints.len();

    for i in 0..(n - 1) {
        let p0 = if i == 0 {
            waypoints[0]
        } else {
            waypoints[i - 1]
        };
        let p1 = waypoints[i];
        let p2 = waypoints[i + 1];
        let p3 = if i + 2 >= n {
            waypoints[n - 1]
        } else {
            waypoints[i + 2]
        };

        // Generate points along this segment
        for j in 0..samples_per_segment {
            let t = j as f32 / samples_per_segment as f32;
            let point = catmull_rom(p0, p1, p2, p3, t);
            points.push(point);
        }
    }

    // Add the final point
    points.push(*waypoints.last().unwrap());

    Road {
        points,
        road_id: id,
        from: None,
        to: None,
    }
}

/// Catmull-Rom spline interpolation
fn catmull_rom(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let t2 = t * t;
    let t3 = t2 * t;

    let x = 0.5
        * ((2.0 * p1.x)
            + (-p0.x + p2.x) * t
            + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
            + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3);

    let y = 0.5
        * ((2.0 * p1.y)
            + (-p0.y + p2.y) * t
            + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
            + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3);

    vec2(x, y)
}

/// Generate points along a circular arc
fn arc_points(
    center: Vec2,
    radius: f32,
    start_angle: f32,
    end_angle: f32,
    num_points: usize,
) -> Vec<Vec2> {
    let mut points = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let t = i as f32 / (num_points - 1) as f32;
        let angle = start_angle + t * (end_angle - start_angle);
        points.push(vec2(
            center.x + radius * angle.cos(),
            center.y + radius * angle.sin(),
        ));
    }
    points
}

/// Generate a serpentine (S-curve) road
fn serpentine_road(
    start: Vec2,
    direction: Vec2,
    amplitude: f32,
    wavelength: f32,
    waves: usize,
    id: RoadId,
) -> Road {
    let mut waypoints = Vec::new();
    let perp = vec2(-direction.y, direction.x).normalize();
    let dir_norm = direction.normalize();

    let total_length = wavelength * waves as f32;
    let num_points = waves * 10 + 1;

    for i in 0..num_points {
        let t = i as f32 / (num_points - 1) as f32;
        let along = t * total_length;
        let wave = (t * waves as f32 * 2.0 * PI).sin() * amplitude;

        let point = start + dir_norm * along + perp * wave;
        waypoints.push(point);
    }

    road_from_waypoints(waypoints, id, 10)
}

/// Generate a hairpin turn road
fn hairpin_road(start: Vec2, initial_dir: Vec2, radius: f32, clockwise: bool, id: RoadId) -> Road {
    let dir_norm = initial_dir.normalize();
    let perp = if clockwise {
        vec2(dir_norm.y, -dir_norm.x)
    } else {
        vec2(-dir_norm.y, dir_norm.x)
    };

    // Lead-in straight section
    let approach_length = radius * 0.5;
    let approach_end = start + dir_norm * approach_length;

    // Arc center
    let arc_center = approach_end + perp * radius;

    // Calculate start and end angles for the arc
    let start_angle = (-perp).y.atan2((-perp).x);
    let angle_span = if clockwise { -PI } else { PI };
    let end_angle = start_angle + angle_span;

    // Build waypoints
    let mut waypoints = vec![start, approach_end];

    // Add arc points
    let arc_pts = arc_points(arc_center, radius, start_angle, end_angle, 20);
    waypoints.extend(arc_pts.into_iter().skip(1)); // skip first since it overlaps

    // Exit straight section
    let exit_start = *waypoints.last().unwrap();
    let exit_dir = -dir_norm;
    let exit_end = exit_start + exit_dir * approach_length;
    waypoints.push(exit_end);

    road_from_waypoints(waypoints, id, 10)
}

/// Generate a spiral road (inward or outward)
fn spiral_road(
    center: Vec2,
    start_radius: f32,
    end_radius: f32,
    total_rotations: f32,
    id: RoadId,
) -> Road {
    let mut waypoints = Vec::new();
    let num_points = (total_rotations * 30.0) as usize;

    for i in 0..=num_points {
        let t = i as f32 / num_points as f32;
        let angle = t * total_rotations * 2.0 * PI;
        let radius = start_radius + t * (end_radius - start_radius);

        waypoints.push(vec2(
            center.x + radius * angle.cos(),
            center.y + radius * angle.sin(),
        ));
    }

    road_from_waypoints(waypoints, id, 5)
}

/// Generate a chicane (quick left-right or right-left)
fn chicane_road(start: Vec2, direction: Vec2, offset: f32, length: f32, id: RoadId) -> Road {
    let dir_norm = direction.normalize();
    let perp = vec2(-dir_norm.y, dir_norm.x);

    let segment = length / 4.0;

    let waypoints = vec![
        start,
        start + dir_norm * segment,
        start + dir_norm * segment * 1.5 + perp * offset * 0.5,
        start + dir_norm * segment * 2.0 + perp * offset,
        start + dir_norm * segment * 2.5 + perp * offset * 0.5,
        start + dir_norm * segment * 3.0,
        start + dir_norm * segment * 4.0,
    ];

    road_from_waypoints(waypoints, id, 15)
}

pub fn build_level_1<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
    let horiz = screen.x * 0.4;
    let vert = screen.y * 0.3;
    let diag = screen.x.min(screen.y) * 0.35;

    let roads = vec![
        Road::new(
            center + vec2(-horiz, 0.0),
            center + vec2(horiz, 0.0),
            RoadId(0),
        ),
        Road::new(
            center + vec2(0.0, -vert),
            center + vec2(0.0, vert),
            RoadId(1),
        ),
        Road::new(
            center + vec2(-diag, -diag * 0.6),
            center + vec2(diag, diag * 0.6),
            RoadId(2),
        ),
        Road::new(
            center + vec2(-diag, diag * 0.6),
            center + vec2(diag, -diag * 0.6),
            RoadId(3),
        ),
    ];

    let road_grid = RoadGrid::new(roads);
    let cars = CarWorld::new_random(6, &road_grid, rng);
    Simulation::new(cars, road_grid)
}

pub fn build_level_2<T: Rng>(center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
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

/// NIGHTMARE TRACK - An absolutely brutal driving course
///
/// Features:
/// - Multiple serpentine sections with varying amplitudes
/// - Tight hairpin turns (180° switchbacks)
/// - A double spiral section
/// - Chicanes that require precise steering
/// - Interweaving roads that cross at odd angles
/// - Long winding mountain-pass style roads
/// - A figure-8 section
/// - Decreasing radius turns (tightening spirals)
///
/// This map is designed to be nearly impossible for untrained NNs
/// and extremely challenging even for human players.
pub fn nightmare_track<T: Rng>(rng: &mut T) -> Simulation {
    const CARS_PER_ROAD: usize = 2; // Fewer cars to focus on navigation difficulty

    let center = vec2(960.0, 540.0);
    let mut roads: Vec<Road> = Vec::new();
    let mut id: usize = 0;

    // ============================================
    // SECTION 1: THE SERPENT'S PATH
    // A series of increasingly aggressive S-curves
    // ============================================

    // Gentle warm-up serpentine (west side)
    roads.push(serpentine_road(
        vec2(100.0, 200.0),
        vec2(1.0, 0.3),
        60.0,  // amplitude
        200.0, // wavelength
        4,     // waves
        RoadId(id),
    ));
    id += 1;

    // Medium serpentine (center-north)
    roads.push(serpentine_road(
        vec2(400.0, 100.0),
        vec2(1.0, 0.0),
        100.0,
        150.0,
        5,
        RoadId(id),
    ));
    id += 1;

    // Aggressive serpentine (tight waves)
    roads.push(serpentine_road(
        vec2(200.0, 700.0),
        vec2(1.0, -0.2),
        120.0,
        100.0, // Short wavelength = tight turns
        6,
        RoadId(id),
    ));
    id += 1;

    // ============================================
    // SECTION 2: HAIRPIN HELL
    // A series of switchback turns like a mountain pass
    // ============================================

    // First hairpin (clockwise)
    roads.push(hairpin_road(
        vec2(1400.0, 150.0),
        vec2(0.0, 1.0),
        80.0,
        true,
        RoadId(id),
    ));
    id += 1;

    // Second hairpin (counter-clockwise, tighter)
    roads.push(hairpin_road(
        vec2(1500.0, 300.0),
        vec2(0.0, 1.0),
        60.0,
        false,
        RoadId(id),
    ));
    id += 1;

    // Third hairpin (very tight)
    roads.push(hairpin_road(
        vec2(1600.0, 200.0),
        vec2(-0.5, 1.0),
        50.0,
        true,
        RoadId(id),
    ));
    id += 1;

    // ============================================
    // SECTION 3: THE SPIRAL OF DOOM
    // Inward and outward spirals
    // ============================================

    // Inward spiral (tightening - gets harder as you go)
    roads.push(spiral_road(
        vec2(300.0, 500.0),
        200.0, // start radius
        40.0,  // end radius (very tight!)
        2.5,   // rotations
        RoadId(id),
    ));
    id += 1;

    // Outward spiral (expanding - easier exit)
    roads.push(spiral_road(
        vec2(1600.0, 700.0),
        50.0,  // start radius
        180.0, // end radius
        2.0,   // rotations
        RoadId(id),
    ));
    id += 1;

    // ============================================
    // SECTION 4: CHICANE CHAOS
    // Quick direction changes
    // ============================================

    // Wide chicane
    roads.push(chicane_road(
        vec2(700.0, 900.0),
        vec2(1.0, 0.0),
        100.0,
        400.0,
        RoadId(id),
    ));
    id += 1;

    // Tight chicane
    roads.push(chicane_road(
        vec2(800.0, 50.0),
        vec2(1.0, 0.2),
        80.0,
        250.0,
        RoadId(id),
    ));
    id += 1;

    // Double chicane (two quick dodges)
    let double_chicane_start = vec2(1200.0, 500.0);
    let dir = vec2(0.0, 1.0).normalize();
    let perp = vec2(-dir.y, dir.x);
    roads.push(road_from_waypoints(
        vec![
            double_chicane_start,
            double_chicane_start + dir * 50.0,
            double_chicane_start + dir * 100.0 + perp * 60.0,
            double_chicane_start + dir * 150.0,
            double_chicane_start + dir * 200.0 - perp * 60.0,
            double_chicane_start + dir * 250.0,
            double_chicane_start + dir * 300.0 + perp * 80.0,
            double_chicane_start + dir * 350.0,
            double_chicane_start + dir * 400.0,
        ],
        RoadId(id),
        15,
    ));
    id += 1;

    // ============================================
    // SECTION 5: THE FIGURE-8 OF INSANITY
    // Two loops that cross in the middle
    // ============================================

    let fig8_center = vec2(950.0, 400.0);
    let fig8_radius = 120.0;
    let fig8_offset = 100.0;

    // Left loop of figure-8
    let left_center = fig8_center - vec2(fig8_offset, 0.0);
    let mut fig8_left: Vec<Vec2> = Vec::new();
    for i in 0..=40 {
        let t = i as f32 / 40.0;
        let angle = -PI * 0.5 + t * 2.0 * PI; // Start from bottom, go counter-clockwise
        fig8_left.push(vec2(
            left_center.x + fig8_radius * angle.cos(),
            left_center.y + fig8_radius * angle.sin(),
        ));
    }
    roads.push(road_from_waypoints(fig8_left, RoadId(id), 8));
    id += 1;

    // Right loop of figure-8
    let right_center = fig8_center + vec2(fig8_offset, 0.0);
    let mut fig8_right: Vec<Vec2> = Vec::new();
    for i in 0..=40 {
        let t = i as f32 / 40.0;
        let angle = PI * 0.5 - t * 2.0 * PI; // Start from bottom, go clockwise
        fig8_right.push(vec2(
            right_center.x + fig8_radius * angle.cos(),
            right_center.y + fig8_radius * angle.sin(),
        ));
    }
    roads.push(road_from_waypoints(fig8_right, RoadId(id), 8));
    id += 1;

    // ============================================
    // SECTION 6: THE CORKSCREW
    // A complex 3D-like path (simulated in 2D)
    // ============================================

    let corkscrew_start = vec2(1700.0, 400.0);
    let mut corkscrew_pts: Vec<Vec2> = Vec::new();
    for i in 0..80 {
        let t = i as f32 / 80.0;
        let x = corkscrew_start.x - t * 300.0;
        let y = corkscrew_start.y + (t * 8.0 * PI).sin() * (60.0 + t * 40.0);
        corkscrew_pts.push(vec2(x, y));
    }
    roads.push(road_from_waypoints(corkscrew_pts, RoadId(id), 5));
    id += 1;

    // ============================================
    // SECTION 7: DECREASING RADIUS TURNS
    // These are especially evil - you have to slow down mid-turn
    // ============================================

    // Decreasing radius curve (starts wide, gets tight)
    let dec_center = vec2(600.0, 350.0);
    let mut decreasing_pts: Vec<Vec2> = Vec::new();
    for i in 0..50 {
        let t = i as f32 / 50.0;
        let angle = t * PI * 1.5; // 270 degrees
        let radius = 150.0 - t * 100.0; // 150 -> 50
        decreasing_pts.push(vec2(
            dec_center.x + radius * angle.cos(),
            dec_center.y + radius * angle.sin(),
        ));
    }
    roads.push(road_from_waypoints(decreasing_pts, RoadId(id), 8));
    id += 1;

    // Increasing then decreasing (trap turn)
    let trap_start = vec2(100.0, 900.0);
    let mut trap_pts: Vec<Vec2> = Vec::new();
    for i in 0..60 {
        let t = i as f32 / 60.0;
        let angle = t * PI;
        // Radius increases then decreases
        let radius = 80.0 + 60.0 * (t * PI).sin();
        trap_pts.push(vec2(
            trap_start.x + t * 200.0 + radius * (angle * 2.0).sin(),
            trap_start.y - radius * (angle * 2.0).cos(),
        ));
    }
    roads.push(road_from_waypoints(trap_pts, RoadId(id), 8));
    id += 1;

    // ============================================
    // SECTION 8: THE MAZE
    // Interconnected short segments with sharp turns
    // ============================================

    let maze_origin = vec2(1100.0, 800.0);
    let seg = 80.0;

    // Maze path 1
    roads.push(road_from_waypoints(
        vec![
            maze_origin,
            maze_origin + vec2(seg, 0.0),
            maze_origin + vec2(seg, -seg),
            maze_origin + vec2(seg * 2.0, -seg),
            maze_origin + vec2(seg * 2.0, 0.0),
            maze_origin + vec2(seg * 3.0, 0.0),
        ],
        RoadId(id),
        12,
    ));
    id += 1;

    // Maze path 2 (intersects path 1)
    roads.push(road_from_waypoints(
        vec![
            maze_origin + vec2(seg * 1.5, seg),
            maze_origin + vec2(seg * 1.5, 0.0),
            maze_origin + vec2(seg * 1.5, -seg),
            maze_origin + vec2(seg * 0.5, -seg),
            maze_origin + vec2(seg * 0.5, -seg * 2.0),
        ],
        RoadId(id),
        12,
    ));
    id += 1;

    // Maze path 3
    roads.push(road_from_waypoints(
        vec![
            maze_origin + vec2(-seg, -seg * 0.5),
            maze_origin + vec2(0.0, -seg * 0.5),
            maze_origin + vec2(0.0, -seg * 1.5),
            maze_origin + vec2(seg, -seg * 1.5),
            maze_origin + vec2(seg, -seg * 2.5),
            maze_origin + vec2(seg * 2.0, -seg * 2.5),
        ],
        RoadId(id),
        12,
    ));
    id += 1;

    // ============================================
    // SECTION 9: THE WHIP
    // Long straight into sudden tight turn
    // ============================================

    roads.push(road_from_waypoints(
        vec![
            vec2(50.0, 400.0),
            vec2(250.0, 400.0), // Long straight
            vec2(350.0, 400.0),
            vec2(400.0, 410.0), // Slight curve warning
            vec2(430.0, 440.0), // Turn starts
            vec2(445.0, 490.0), // Getting tighter
            vec2(440.0, 540.0), // Apex
            vec2(420.0, 570.0), // Exit
            vec2(380.0, 590.0),
        ],
        RoadId(id),
        15,
    ));
    id += 1;

    // Another whip (opposite direction)
    roads.push(road_from_waypoints(
        vec![
            vec2(1850.0, 600.0),
            vec2(1650.0, 600.0),
            vec2(1550.0, 600.0),
            vec2(1500.0, 590.0),
            vec2(1470.0, 560.0),
            vec2(1455.0, 510.0),
            vec2(1460.0, 460.0),
            vec2(1480.0, 430.0),
            vec2(1520.0, 410.0),
        ],
        RoadId(id),
        15,
    ));
    id += 1;

    // ============================================
    // SECTION 10: RANDOM CHAOS PATHS
    // Procedurally generated madness
    // ============================================

    // Chaotic path 1
    let mut chaos1: Vec<Vec2> = Vec::new();
    let chaos1_start = vec2(700.0, 200.0);
    chaos1.push(chaos1_start);
    let mut pos = chaos1_start;
    let mut angle = 0.0_f32;
    for i in 0..15 {
        let turn = ((i as f32 * 1.7).sin() * 0.8 + (i as f32 * 0.3).cos() * 0.4) * 1.2;
        angle += turn;
        let dist = 50.0 + (i as f32 * 0.5).sin().abs() * 30.0;
        pos = pos + vec2(angle.cos(), angle.sin()) * dist;
        chaos1.push(pos);
    }
    roads.push(road_from_waypoints(chaos1, RoadId(id), 10));
    id += 1;

    // Chaotic path 2
    let mut chaos2: Vec<Vec2> = Vec::new();
    let chaos2_start = vec2(1300.0, 100.0);
    chaos2.push(chaos2_start);
    pos = chaos2_start;
    angle = PI * 0.5;
    for i in 0..12 {
        let turn = ((i as f32 * 2.3).cos() * 0.6 - (i as f32 * 0.7).sin() * 0.5) * 1.5;
        angle += turn;
        let dist = 60.0 + (i as f32 * 0.8).cos().abs() * 40.0;
        pos = pos + vec2(angle.cos(), angle.sin()) * dist;
        chaos2.push(pos);
    }
    roads.push(road_from_waypoints(chaos2, RoadId(id), 10));
    id += 1;

    // ============================================
    // SECTION 11: THE GAUNTLET (connecting roads)
    // These roads connect different sections
    // ============================================

    // Connector from serpent to hairpin
    roads.push(Road::new(
        vec2(900.0, 150.0),
        vec2(1350.0, 180.0),
        RoadId(id),
    ));
    id += 1;

    // Connector from spiral to maze
    roads.push(Road::new(
        vec2(500.0, 500.0),
        vec2(1050.0, 750.0),
        RoadId(id),
    ));
    id += 1;

    // Cross-map diagonal
    roads.push(Road::new(
        vec2(150.0, 150.0),
        vec2(1750.0, 900.0),
        RoadId(id),
    ));
    id += 1;

    // Another diagonal (crossing)
    roads.push(Road::new(
        vec2(150.0, 900.0),
        vec2(1750.0, 150.0),
        RoadId(id),
    ));
    id += 1;

    // Central hub connections (star pattern from center)
    for i in 0..6 {
        let angle = i as f32 * PI / 3.0;
        let outer = center + vec2(angle.cos(), angle.sin()) * 400.0;
        roads.push(Road::new(center, outer, RoadId(id)));
        id += 1;
    }

    // ============================================
    // BUILD THE SIMULATION
    // ============================================

    let road_grid = RoadGrid::new(roads);
    let road_count = road_grid.roads.len().max(1);
    let num_cars = road_count * CARS_PER_ROAD;

    let topology = [
        LayerTopology { neurons: 5 },
        LayerTopology { neurons: 12 }, // Larger hidden layer for complex navigation
        LayerTopology { neurons: 8 },  // Additional hidden layer
        LayerTopology { neurons: 2 },
    ];

    let cars = CarWorld {
        cars: (0..num_cars)
            .map(|x| {
                let network = Network::new_random(&topology, rng);
                let (r, g, b, _a): (u8, u8, u8, u8) = rng.random();
                let color = Color::from_rgba(r, g, b, 255);
                Car::new_on_road(&road_grid, RoadId(x % road_count), color, network, x as u16)
            })
            .collect(),
    };

    Simulation::new(cars, road_grid)
}

/// An even more extreme version of the nightmare track
/// with tighter turns and more chaotic patterns
pub fn nightmare_track_extreme<T: Rng>(rng: &mut T) -> Simulation {
    const CARS_PER_ROAD: usize = 1;

    let _center = vec2(960.0, 540.0);
    let mut roads: Vec<Road> = Vec::new();
    let mut id: usize = 0;

    // ULTRA TIGHT SPIRALS
    roads.push(spiral_road(
        vec2(200.0, 300.0),
        180.0,
        25.0,
        3.5, // Very tight ending
        RoadId(id),
    ));
    id += 1;

    roads.push(spiral_road(
        vec2(1700.0, 300.0),
        150.0,
        20.0,
        4.0,
        RoadId(id),
    ));
    id += 1;

    // EXTREME SERPENTINES
    roads.push(serpentine_road(
        vec2(100.0, 800.0),
        vec2(1.0, 0.0),
        150.0,
        80.0,
        10, // Very tight waves
        RoadId(id),
    ));
    id += 1;

    // MULTIPLE CONSECUTIVE HAIRPINS (like a mountain pass)
    let hairpin_start = vec2(600.0, 100.0);
    let mut hairpin_chain: Vec<Vec2> = vec![hairpin_start];
    let mut hp_pos = hairpin_start;
    let mut hp_dir = vec2(1.0, 0.0);

    for i in 0..8 {
        let clockwise = i % 2 == 0;
        let radius = 40.0 + (i as f32 * 5.0); // Varying radii
        let perp = if clockwise {
            vec2(hp_dir.y, -hp_dir.x)
        } else {
            vec2(-hp_dir.y, hp_dir.x)
        };

        // Add approach
        hp_pos = hp_pos + hp_dir * 60.0;
        hairpin_chain.push(hp_pos);

        // Add hairpin arc points
        let arc_center = hp_pos + perp * radius;
        let start_angle = (-perp).y.atan2((-perp).x);
        let arc = arc_points(
            arc_center,
            radius,
            start_angle,
            start_angle + if clockwise { -PI } else { PI },
            15,
        );
        hairpin_chain.extend(arc.into_iter().skip(1));

        hp_pos = *hairpin_chain.last().unwrap();
        hp_dir = -hp_dir; // Reverse direction after hairpin
    }
    roads.push(road_from_waypoints(hairpin_chain, RoadId(id), 8));
    id += 1;

    // THE PRETZEL - a complex self-intersecting path
    let pretzel_center = vec2(1300.0, 600.0);
    let mut pretzel: Vec<Vec2> = Vec::new();
    for i in 0..100 {
        let t = i as f32 / 100.0;
        let angle = t * 4.0 * PI;
        let r = 80.0 + 50.0 * (angle * 1.5).sin();
        pretzel.push(vec2(
            pretzel_center.x + r * angle.cos() + t * 150.0,
            pretzel_center.y + r * angle.sin(),
        ));
    }
    roads.push(road_from_waypoints(pretzel, RoadId(id), 5));
    id += 1;

    // LEMNISCATE (infinity symbol with variation)
    let lemni_center = vec2(500.0, 600.0);
    let mut lemniscate: Vec<Vec2> = Vec::new();
    for i in 0..80 {
        let t = i as f32 / 80.0 * 2.0 * PI;
        let scale = 120.0;
        let x = scale * t.cos() / (1.0 + t.sin().powi(2));
        let y = scale * t.sin() * t.cos() / (1.0 + t.sin().powi(2));
        lemniscate.push(vec2(lemni_center.x + x, lemni_center.y + y));
    }
    roads.push(road_from_waypoints(lemniscate, RoadId(id), 8));
    id += 1;

    // ZIGZAG LIGHTNING BOLT
    let mut zigzag: Vec<Vec2> = Vec::new();
    let zz_start = vec2(100.0, 500.0);
    for i in 0..20 {
        let x = zz_start.x + i as f32 * 40.0;
        let y = zz_start.y
            + if i % 2 == 0 {
                0.0
            } else {
                80.0 * if i % 4 == 1 { 1.0 } else { -1.0 }
            };
        zigzag.push(vec2(x, y));
    }
    roads.push(road_from_waypoints(zigzag, RoadId(id), 15));
    id += 1;

    // CONNECTION ROADS (need at least some way to traverse the map)
    roads.push(Road::new(
        vec2(100.0, 100.0),
        vec2(1800.0, 100.0),
        RoadId(id),
    ));
    id += 1;
    roads.push(Road::new(
        vec2(100.0, 1000.0),
        vec2(1800.0, 1000.0),
        RoadId(id),
    ));
    id += 1;
    roads.push(Road::new(
        vec2(100.0, 100.0),
        vec2(100.0, 1000.0),
        RoadId(id),
    ));
    id += 1;
    roads.push(Road::new(
        vec2(1800.0, 100.0),
        vec2(1800.0, 1000.0),
        RoadId(id),
    ));
    id += 1;

    // Central cross
    roads.push(Road::new(
        vec2(500.0, 540.0),
        vec2(1400.0, 540.0),
        RoadId(id),
    ));
    id += 1;
    roads.push(Road::new(
        vec2(960.0, 200.0),
        vec2(960.0, 880.0),
        RoadId(id),
    ));

    let road_grid = RoadGrid::new(roads);
    let road_count = road_grid.roads.len().max(1);
    let num_cars = road_count * CARS_PER_ROAD;

    let topology = [
        LayerTopology { neurons: 5 },
        LayerTopology { neurons: 16 },
        LayerTopology { neurons: 8 },
        LayerTopology { neurons: 2 },
    ];

    let cars = CarWorld {
        cars: (0..num_cars)
            .map(|x| {
                let network = Network::new_random(&topology, rng);
                let (r, g, b, _a): (u8, u8, u8, u8) = rng.random();
                let color = Color::from_rgba(r, g, b, 255);
                Car::new_on_road(&road_grid, RoadId(x % road_count), color, network, x as u16)
            })
            .collect(),
    };

    Simulation::new(cars, road_grid)
}

pub fn build_level_3<T: Rng>(_center: Vec2, screen: Vec2, rng: &mut T) -> Simulation {
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

pub fn build_straight_road_4<T: Rng>(center: Vec2, _screen: Vec2, rng: &mut T) -> Simulation {
    let roads = vec![
        Road::new(center, vec2(center.x + 200.0, center.y), RoadId(0)),
        Road::new(
            vec2(center.x + 200.0, center.y),
            vec2(center.x + 200.0, center.y - 1000.0),
            RoadId(1),
        ),
    ];

    let road_grid = RoadGrid::new(roads);

    let cars = CarWorld::new_random(1, &road_grid, rng);

    Simulation::new(cars, road_grid)
}

pub fn build_straight_line_level<T: Rng>(start: Vec2, length: f32, rng: &mut T) -> Simulation {
    // Two colinear segments to satisfy RoadGrid's requirement for multiple roads.
    let mid = start + vec2(length * 0.5, 0.0);
    let end = start + vec2(length, 0.0);

    let roads = vec![
        Road::new(start, mid, RoadId(0)),
        Road::new(mid, end, RoadId(1)),
    ];

    let road_grid = RoadGrid::new(roads);
    let cars = CarWorld::new_random(6, &road_grid, rng);
    Simulation::new(cars, road_grid)
}

pub fn test_sensors<T: Rng>(_center: Vec2, _screen: Vec2, rng: &mut T) -> Simulation {
    const NUM_ROADS: usize = 20;
    const NUM_CARS: usize = 100;

    let road_grid = generate_road_grid(NUM_ROADS, rng);
    //let road_grid = RoadGrid::new(roads);

    let topology = [
        LayerTopology { neurons: 5 },
        LayerTopology { neurons: 5 },
        LayerTopology { neurons: 2 },
    ];

    let cars = CarWorld {
        cars: (0..NUM_CARS)
            .map(|x| {
                let network = Network::new_random(&topology, rng);
                let (r, g, b, _a): (u8, u8, u8, u8) = rng.random();
                let color = Color::from_rgba(r, g, b, 255);
                Car::new_on_road(&road_grid, RoadId(x % NUM_ROADS), color, network, x as u16)
            })
            .collect(),
    };

    Simulation::new(cars, road_grid)
}

/// A comprehensive training level for overnight evolution runs.
///
/// Features:
/// - Large interconnected grid with varied path lengths
/// - Multiple difficulty zones (easy straight sections + complex intersections)
/// - Enough cars to create interesting collision avoidance scenarios
/// - Roads of varying lengths to test both short and long navigation
pub fn overnight_training<T: Rng>(rng: &mut T) -> Simulation {
    const CARS_PER_ROAD: usize = 4;

    // Build a proper city grid with intersections
    let city_center = vec2(960.0, 540.0);
    let block_size = 300.0;
    let grid_size = 5; // 5x5 grid of blocks

    let half_span = block_size * (grid_size as f32) * 0.5;

    let mut roads: Vec<Road> = Vec::new();
    let mut id: usize = 0;

    // Horizontal roads (west to east)
    for row in 0..=grid_size {
        let y = city_center.y - half_span + row as f32 * block_size;
        let start = vec2(city_center.x - half_span, y);
        let end = vec2(city_center.x + half_span, y);
        roads.push(Road::new(start, end, RoadId(id)));
        id += 1;
    }

    // Vertical roads (north to south)
    for col in 0..=grid_size {
        let x = city_center.x - half_span + col as f32 * block_size;
        let start = vec2(x, city_center.y - half_span);
        let end = vec2(x, city_center.y + half_span);
        roads.push(Road::new(start, end, RoadId(id)));
        id += 1;
    }

    // Diagonal roads for variety (corner to corner shortcuts)
    roads.push(Road::new(
        vec2(city_center.x - half_span, city_center.y - half_span),
        vec2(city_center.x + half_span, city_center.y + half_span),
        RoadId(id),
    ));
    id += 1;

    roads.push(Road::new(
        vec2(city_center.x - half_span, city_center.y + half_span),
        vec2(city_center.x + half_span, city_center.y - half_span),
        RoadId(id),
    ));
    id += 1;

    // Curved bypass roads around the edges (forms a diamond/ring road)
    let outer_offset = half_span + 150.0;
    roads.push(Road::new(
        vec2(city_center.x - outer_offset, city_center.y),
        vec2(city_center.x, city_center.y - outer_offset),
        RoadId(id),
    ));
    id += 1;

    roads.push(Road::new(
        vec2(city_center.x, city_center.y - outer_offset),
        vec2(city_center.x + outer_offset, city_center.y),
        RoadId(id),
    ));
    id += 1;

    roads.push(Road::new(
        vec2(city_center.x + outer_offset, city_center.y),
        vec2(city_center.x, city_center.y + outer_offset),
        RoadId(id),
    ));
    id += 1;

    roads.push(Road::new(
        vec2(city_center.x, city_center.y + outer_offset),
        vec2(city_center.x - outer_offset, city_center.y),
        RoadId(id),
    ));

    let road_grid = RoadGrid::new(roads);
    let road_count = road_grid.roads.len().max(1);
    // Scale car count with road count to avoid overcrowding.
    let num_cars = road_count * CARS_PER_ROAD;

    let topology = [
        LayerTopology { neurons: 5 },
        LayerTopology { neurons: 8 }, // Slightly larger hidden layer for complex behavior
        LayerTopology { neurons: 2 },
    ];

    let cars = CarWorld {
        cars: (0..num_cars)
            .map(|x| {
                let network = Network::new_random(&topology, rng);
                let (r, g, b, _a): (u8, u8, u8, u8) = rng.random();
                let color = Color::from_rgba(r, g, b, 255);
                Car::new_on_road(&road_grid, RoadId(x % road_count), color, network, x as u16)
            })
            .collect(),
    };

    Simulation::new(cars, road_grid)
}
