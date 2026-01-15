use ::rand::SeedableRng;
use bincode::deserialize;
use genetics::Individual;
use macroquad::prelude::*;
use rand_chacha::ChaCha8Rng;
use traffic::levels::*;

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

        let base_speed = if is_key_down(KeyCode::LeftShift) { 150.0 } else { 80.0 };
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
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut sim = overnight_training(&mut rng);

    let initial_individuals: Vec<Individual> = sim
        .cars
        .cars
        .iter()
        .map(|car| {
            let genes = car.network.to_genes();
            let fitness = genetics::fitness(car);
            Individual { genes, fitness }
        })
        .collect();

    let expected_pop_size = initial_individuals.len();
    let checkpoint_path = "output/serialization/individuals.bin";

    let _individuals = if std::path::Path::new(checkpoint_path).exists() {
        let data = std::fs::read(checkpoint_path).expect("Failed to read checkpoint file");
        let individuals: Vec<Individual> =
            deserialize(&data).expect("Failed to deserialize individuals");
        if individuals.len() != expected_pop_size {
            eprintln!(
                "Checkpoint population size {} does not match expected size {}",
                individuals.len(),
                expected_pop_size
            );
        }
        individuals
    } else {
        initial_individuals
    };

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
