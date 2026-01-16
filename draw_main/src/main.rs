use ::rand::SeedableRng;
use bincode::deserialize;
use genetics::{Individual, NetworkTopology, make_sim_from_slice_with_topology};
use macroquad::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::env;
use traffic::levels::*;
use traffic::simulation::Simulation;

/// Available training levels/maps
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainingLevel {
    // Easy levels
    StraightLine, // Simple straight road
    StraightRoad, // L-shaped road
    Level1,       // Basic crossroads
    Level2,       // Rectangle with diagonals
    Level3,       // Grid pattern

    // Medium levels
    TestSensors, // Random grid for testing
    Overnight,   // Standard city grid (default)

    // Hard levels
    Nightmare,        // Twisty track with curves
    NightmareExtreme, // Brutal track with tight spirals
}

impl TrainingLevel {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            // Easy
            "straight" | "straight_line" | "line" => Some(Self::StraightLine),
            "straight_road" | "lshape" | "l" => Some(Self::StraightRoad),
            "level1" | "1" | "basic" => Some(Self::Level1),
            "level2" | "2" | "rectangle" => Some(Self::Level2),
            "level3" | "3" => Some(Self::Level3),

            // Medium
            "test" | "test_sensors" | "sensors" | "random" => Some(Self::TestSensors),
            "overnight" | "city" | "grid" => Some(Self::Overnight),

            // Hard
            "nightmare" | "hard" | "twisty" => Some(Self::Nightmare),
            "nightmare_extreme" | "extreme" | "insane" | "brutal" => Some(Self::NightmareExtreme),

            _ => None,
        }
    }

    fn build_simulation<R: ::rand::Rng>(self, rng: &mut R) -> Simulation {
        let center = vec2(960.0, 540.0);
        let screen = vec2(1920.0, 1080.0);

        match self {
            Self::StraightLine => build_straight_line_level(center, 800.0, rng),
            Self::StraightRoad => build_straight_road_4(center, screen, rng),
            Self::Level1 => build_level_1(center, screen, rng),
            Self::Level2 => build_level_2(center, screen, rng),
            Self::Level3 => build_level_3(center, screen, rng),
            Self::TestSensors => test_sensors(center, screen, rng),
            Self::Overnight => overnight_training(rng),
            Self::Nightmare => nightmare_track(rng),
            Self::NightmareExtreme => nightmare_track_extreme(rng),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::StraightLine => "straight_line (easy)",
            Self::StraightRoad => "straight_road (L-shape)",
            Self::Level1 => "level1 (basic crossroads)",
            Self::Level2 => "level2 (rectangle)",
            Self::Level3 => "level3 (grid)",
            Self::TestSensors => "test_sensors (random)",
            Self::Overnight => "overnight (city grid)",
            Self::Nightmare => "nightmare (twisty)",
            Self::NightmareExtreme => "nightmare_extreme (brutal)",
        }
    }

    fn list_all() -> &'static str {
        "Available levels:\n\
         Easy:   straight_line, straight_road, level1, level2, level3\n\
         Medium: test_sensors, overnight\n\
         Hard:   nightmare, nightmare_extreme"
    }
}

fn parse_level_from_args() -> TrainingLevel {
    let args: Vec<String> = env::args().collect();

    // Look for --level or -l argument
    for i in 0..args.len() {
        if (args[i] == "--level" || args[i] == "-l") && i + 1 < args.len() {
            if let Some(level) = TrainingLevel::from_str(&args[i + 1]) {
                return level;
            } else {
                eprintln!(
                    "Unknown level '{}'.\n{}",
                    args[i + 1],
                    TrainingLevel::list_all()
                );
            }
        }
        // Also support --level=value format
        if args[i].starts_with("--level=") {
            let value = &args[i][8..];
            if let Some(level) = TrainingLevel::from_str(value) {
                return level;
            }
        }
    }

    // Check .config file for level setting
    if let Ok(contents) = std::fs::read_to_string(".config") {
        for line in contents.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if let Some(value) = line
                .strip_prefix("level=")
                .or_else(|| line.strip_prefix("map="))
                .or_else(|| line.strip_prefix("track="))
            {
                if let Some(level) = TrainingLevel::from_str(value.trim()) {
                    return level;
                }
            }
        }
    }

    TrainingLevel::Overnight
}

fn parse_topology_from_config() -> NetworkTopology {
    // Check command line args first
    let args: Vec<String> = env::args().collect();
    for i in 0..args.len() {
        if (args[i] == "--topology" || args[i] == "-t") && i + 1 < args.len() {
            if let Some(topo) = NetworkTopology::from_str(&args[i + 1]) {
                return topo;
            }
        }
        if args[i].starts_with("--topology=") {
            let value = &args[i][11..];
            if let Some(topo) = NetworkTopology::from_str(value) {
                return topo;
            }
        }
    }

    // Check .config file
    if let Ok(contents) = std::fs::read_to_string(".config") {
        for line in contents.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if let Some(value) = line
                .strip_prefix("topology=")
                .or_else(|| line.strip_prefix("network="))
                .or_else(|| line.strip_prefix("layers="))
            {
                if let Some(topo) = NetworkTopology::from_str(value.trim()) {
                    return topo;
                }
            }
        }
    }

    NetworkTopology::default()
}

const BASE_ZOOM: f32 = 0.003;
const MIN_ZOOM: f32 = 0.0001;
const MAX_ZOOM: f32 = 0.008;

/// Camera follow mode
#[derive(Clone, Copy, PartialEq)]
enum FollowMode {
    BestCar,
    Free,
}

impl FollowMode {
    fn name(&self) -> &'static str {
        match self {
            FollowMode::BestCar => "Best Car",
            FollowMode::Free => "Free Camera",
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn lerp_vec2(a: Vec2, b: Vec2, t: f32) -> Vec2 {
    vec2(lerp(a.x, b.x, t), lerp(a.y, b.y, t))
}

struct CameraController {
    camera: Camera2D,
    follow_mode: FollowMode,
    target_zoom: f32,
    smoothness: f32,
    zoom_smoothness: f32,
    manual_override_timer: f32,
}

impl CameraController {
    fn new(initial_target: Vec2) -> Self {
        Self {
            camera: Camera2D {
                target: initial_target,
                zoom: vec2(BASE_ZOOM, BASE_ZOOM),
                ..Default::default()
            },
            follow_mode: FollowMode::BestCar,
            target_zoom: BASE_ZOOM,
            smoothness: 0.08,
            zoom_smoothness: 0.1,
            manual_override_timer: 0.0,
        }
    }

    fn handle_input(&mut self) {
        let dt = get_frame_time().min(0.1);

        let base_speed = if is_key_down(KeyCode::LeftShift) {
            150.0
        } else {
            80.0
        };
        let pan_step = base_speed * dt;

        let mut manual_pan = false;

        if is_key_down(KeyCode::W) {
            self.camera.target.y -= pan_step;
            manual_pan = true;
        }
        if is_key_down(KeyCode::S) {
            self.camera.target.y += pan_step;
            manual_pan = true;
        }
        if is_key_down(KeyCode::A) {
            self.camera.target.x -= pan_step;
            manual_pan = true;
        }
        if is_key_down(KeyCode::D) {
            self.camera.target.x += pan_step;
            manual_pan = true;
        }

        if manual_pan && self.follow_mode != FollowMode::Free {
            self.manual_override_timer = 1.5;
        }

        if self.manual_override_timer > 0.0 {
            self.manual_override_timer -= dt;
        }

        if is_key_pressed(KeyCode::F) {
            self.follow_mode = match self.follow_mode {
                FollowMode::BestCar => FollowMode::Free,
                FollowMode::Free => FollowMode::BestCar,
            };
            self.manual_override_timer = 0.0;
        }

        if is_key_pressed(KeyCode::R) {
            self.target_zoom = BASE_ZOOM;
            self.manual_override_timer = 0.0;
        }

        if is_key_down(KeyCode::Equal) {
            self.target_zoom = (self.target_zoom * 1.02).clamp(MIN_ZOOM, MAX_ZOOM);
        }
        if is_key_down(KeyCode::Minus) {
            self.target_zoom = (self.target_zoom * 0.98).clamp(MIN_ZOOM, MAX_ZOOM);
        }

        let scroll = mouse_wheel().1;
        if scroll.abs() > f32::EPSILON {
            let factor = 1.0 + scroll * 0.05;
            self.target_zoom = (self.target_zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        }

        self.camera.zoom.x = lerp(self.camera.zoom.x, self.target_zoom, self.zoom_smoothness);
        self.camera.zoom.y = self.camera.zoom.x;

        if is_key_pressed(KeyCode::LeftBracket) {
            self.smoothness = (self.smoothness - 0.02).clamp(0.01, 0.3);
        }
        if is_key_pressed(KeyCode::RightBracket) {
            self.smoothness = (self.smoothness + 0.02).clamp(0.01, 0.3);
        }
    }

    fn update_target(&mut self, target_position: Vec2) {
        if self.follow_mode != FollowMode::Free && self.manual_override_timer <= 0.0 {
            self.camera.target = lerp_vec2(self.camera.target, target_position, self.smoothness);
        }
    }

    fn set_camera(&self) {
        set_camera(&self.camera);
    }
}

/// Find the best car based on progress (matching draw_sim's logic)
fn find_best_car_position(sim: &traffic::simulation::Simulation) -> Option<Vec2> {
    sim.cars
        .cars
        .iter()
        .max_by(|a, b| {
            a.progress_to_goal
                .partial_cmp(&b.progress_to_goal)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|car| car.position)
}

fn draw_camera_hud(controller: &CameraController) {
    set_default_camera();

    let panel_x = screen_width() - 280.0;
    let panel_y = 18.0;
    let panel_w = 260.0;
    let panel_h = 100.0;
    let padding = 12.0;

    let panel_bg = Color::from_rgba(18, 24, 30, 200);
    let panel_border = Color::from_rgba(100, 140, 180, 180);
    let accent = Color::from_rgba(255, 120, 180, 220);
    let text_primary = Color::from_rgba(220, 235, 250, 235);
    let text_muted = Color::from_rgba(150, 165, 180, 200);

    draw_rectangle(
        panel_x + 3.0,
        panel_y + 3.0,
        panel_w,
        panel_h,
        Color::from_rgba(0, 0, 0, 60),
    );
    draw_rectangle(panel_x, panel_y, panel_w, panel_h, panel_bg);
    draw_rectangle(panel_x, panel_y, panel_w, 3.0, accent);
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 1.0, panel_border);

    let mut y = panel_y + padding + 18.0;

    draw_text("CAMERA", panel_x + padding, y, 20.0, text_primary);
    y += 28.0;

    let mode_text = format!("Mode: {}", controller.follow_mode.name());
    draw_text(&mode_text, panel_x + padding, y, 16.0, accent);
    y += 24.0;

    draw_text(
        "[F] Toggle follow | [R] Reset",
        panel_x + padding,
        y,
        12.0,
        text_muted,
    );
    y += 16.0;
    draw_text(
        "[WASD] Pan | [+/-] Zoom",
        panel_x + padding,
        y,
        12.0,
        text_muted,
    );
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Traffic Simulation".to_owned(),
        window_width: 1920,
        window_height: 1080,
        fullscreen: true,
        // Higher sample count = better anti-aliasing (1, 2, 4, 8, 16)
        // Set to 1 for max performance, 4 for good balance
        sample_count: 4,
        // VSync: true = smoother but capped at monitor refresh rate
        // false = uncapped FPS (may cause tearing)
        high_dpi: true,
        window_resizable: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let level = parse_level_from_args();
    let topology = parse_topology_from_config();
    println!("Using level: {}", level.name());
    println!("Using network topology: {}", topology.display());

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create initial simulation to get road grid and expected population size
    let initial_sim = level.build_simulation(&mut rng);
    let road_grid = initial_sim.roads.clone();
    let expected_pop_size = initial_sim.cars.cars.len();
    let expected_gene_count = topology.gene_count();

    // Create fresh random individuals with configured topology
    let fresh_population =
        genetics::create_random_population(expected_pop_size, &topology, &mut rng);
    let initial_individuals = fresh_population.individuals;

    let checkpoint_path = "output/serialization/individuals.bin";

    // Load checkpoint or use initial individuals
    let individuals = if std::path::Path::new(checkpoint_path).exists() {
        let data = std::fs::read(checkpoint_path).expect("Failed to read checkpoint file");
        let loaded: Vec<Individual> =
            deserialize(&data).expect("Failed to deserialize individuals");

        // Validate both size and gene count
        let size_ok = loaded.len() == expected_pop_size;
        let genes_ok = loaded
            .first()
            .map(|i| i.genes.len() == expected_gene_count)
            .unwrap_or(false);

        if !size_ok {
            eprintln!(
                "Checkpoint population size {} != expected {}; using fresh population",
                loaded.len(),
                expected_pop_size
            );
            initial_individuals
        } else if !genes_ok {
            eprintln!(
                "Checkpoint gene count {} != expected {} for topology {}; using fresh population",
                loaded.first().map(|i| i.genes.len()).unwrap_or(0),
                expected_gene_count,
                topology.to_config_string()
            );
            initial_individuals
        } else {
            println!(
                "Loaded {} trained individuals from checkpoint",
                loaded.len()
            );
            loaded
        }
    } else {
        println!("No checkpoint found, using fresh random individuals");
        initial_individuals
    };

    // Create simulation from loaded individuals with configured topology
    let mut sim = make_sim_from_slice_with_topology(&individuals, &road_grid, &topology, &mut rng);

    let initial_pos = sim
        .cars
        .cars
        .first()
        .map(|c| c.position)
        .unwrap_or(vec2(960.0, 540.0));
    let mut camera_controller = CameraController::new(initial_pos);

    set_fullscreen(true);

    loop {
        camera_controller.handle_input();

        // Find best car position (same criteria as draw_sim uses)
        if let Some(best_pos) = find_best_car_position(&sim) {
            camera_controller.update_target(best_pos);
        }

        camera_controller.set_camera();
        clear_background(BEIGE);

        // draw_sim handles the best car indicator internally
        sim.draw_sim(true);

        draw_camera_hud(&camera_controller);

        sim.update(false);

        next_frame().await;
    }
}
