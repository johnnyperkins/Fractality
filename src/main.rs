mod particles;

use std::sync::Arc;

use bevy::core_pipeline::bloom::Bloom;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::render::camera::ClearColorConfig;
use bevy::render::view::Msaa;
use bevy::window::PresentMode;

use particles::{generate_particles, ParticlePlugin, ParticleSeed, SimParams};

const MAX_ITER: u32 = 90;
const DEFAULT_CENTER: Vec2 = Vec2::new(-0.55, 0.0);
const DEFAULT_HEIGHT: f32 = 2.7;

#[derive(Resource)]
struct ParticleCount(u32);

#[derive(Resource)]
struct Paused(bool);

#[derive(Resource)]
struct ViewState {
    center: Vec2,
    height: f32,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            center: DEFAULT_CENTER,
            height: DEFAULT_HEIGHT,
        }
    }
}

fn main() {
    let count = std::env::args()
        .nth(1)
        .map(|s| s.replace('_', ""))
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1_000_000);

    println!("Fractality controls:");
    println!("  hover      ripple particles");
    println!("  left-click blast");
    println!("  right-click vortex");
    println!("  wheel      zoom toward cursor");
    println!("  WASD       pan");
    println!("  Space      pause");
    println!("  R          reset view");

    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Fractality".into(),
                    present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
        )
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(ParticlePlugin)
        .insert_resource(ParticleCount(count))
        .insert_resource(ViewState::default())
        .insert_resource(Paused(false))
        .insert_resource(SimParams::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (handle_input, update_params, update_title))
        .run();
}

fn setup(mut commands: Commands, count: Res<ParticleCount>) {
    commands.spawn((
        Camera2d,
        Camera {
            hdr: true,
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Bloom {
            intensity: 0.3,
            ..Bloom::NATURAL
        },
        Msaa::Off,
    ));

    let start = std::time::Instant::now();
    let particles = generate_particles(count.0 as usize, MAX_ITER);
    info!(
        "generated {} particles in {:.2?}",
        particles.len(),
        start.elapsed()
    );
    commands.insert_resource(ParticleSeed(Arc::new(particles)));
}

fn handle_input(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: EventReader<MouseWheel>,
    windows: Query<&Window>,
    mut view: ResMut<ViewState>,
    mut paused: ResMut<Paused>,
) {
    let dt = time.delta_secs();

    let mut pan = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        pan.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        pan.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        pan.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        pan.x += 1.0;
    }
    if pan != Vec2::ZERO {
        let speed = view.height * 0.6 * dt;
        view.center += pan * speed;
    }

    if keys.just_pressed(KeyCode::KeyR) {
        view.center = DEFAULT_CENTER;
        view.height = DEFAULT_HEIGHT;
    }
    if keys.just_pressed(KeyCode::Space) {
        paused.0 = !paused.0;
    }

    let mut scroll = 0.0f32;
    for ev in wheel.read() {
        scroll += match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y / 60.0,
        };
    }
    if scroll != 0.0 {
        let Ok(window) = windows.single() else {
            return;
        };
        let old_h = view.height;
        let new_h = (old_h * 0.9f32.powf(scroll)).clamp(1e-6, 40.0);
        if let Some(cursor) = window.cursor_position() {
            let w = window.width().max(1.0);
            let h = window.height().max(1.0);
            let aspect = w / h;
            let ndc = Vec2::new(
                cursor.x / w * 2.0 - 1.0,
                1.0 - cursor.y / h * 2.0,
            );
            let cursor_fractal =
                view.center + Vec2::new(ndc.x * old_h * aspect * 0.5, ndc.y * old_h * 0.5);
            let delta = (cursor_fractal - view.center) * (1.0 - new_h / old_h);
            view.center += delta;
        }
        view.height = new_h;
    }
}

fn update_params(
    time: Res<Time>,
    windows: Query<&Window>,
    view: Res<ViewState>,
    paused: Res<Paused>,
    mouse: Res<ButtonInput<MouseButton>>,
    count: Res<ParticleCount>,
    mut params: ResMut<SimParams>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let w = window.width().max(1.0);
    let h = window.height().max(1.0);
    let aspect = w / h;

    // clip.x = (p.x - cx) * 2/(height*aspect), clip.y = (p.y - cy) * 2/height
    let scale = Vec2::new(2.0 / (view.height * aspect), 2.0 / view.height);
    let offset = -view.center * scale;

    let mouse_fractal = window.cursor_position().map(|c| {
        let ndc = Vec2::new(c.x / w * 2.0 - 1.0, 1.0 - c.y / h * 2.0);
        (ndc - offset) / scale
    });
    let button = if mouse.pressed(MouseButton::Left) {
        1.0
    } else if mouse.pressed(MouseButton::Right) {
        -1.0
    } else {
        0.0
    };
    let radius = view.height * 0.09;

    let dt = if paused.0 {
        0.0
    } else {
        time.delta_secs().min(1.0 / 30.0)
    };

    let px = 1.3f32;
    let u = &mut params.0;
    u.world_to_clip = Vec4::new(scale.x, scale.y, offset.x, offset.y);
    u.mouse = match mouse_fractal {
        Some(p) => Vec4::new(p.x, p.y, button, radius),
        None => Vec4::new(1e9, 1e9, 0.0, radius),
    };
    u.particle_size = Vec2::new(2.0 * px / w, 2.0 * px / h);
    u.time = time.elapsed_secs();
    u.dt = dt;
    u.count = count.0;
    u.max_iter = MAX_ITER;
    u.flow_speed = 0.22;
    u.band_k = 0.7;
    u.damping = 3.0;
    u.brightness = 1.1 * (500_000.0 / count.0 as f32).sqrt();
}

fn update_title(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    count: Res<ParticleCount>,
    mut windows: Query<&mut Window>,
    mut timer: Local<f32>,
) {
    *timer += time.delta_secs();
    if *timer < 0.5 {
        return;
    }
    *timer = 0.0;
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    if let Ok(mut window) = windows.single_mut() {
        window.title = format!(
            "Fractality | {} particles | {:.0} FPS | wheel zoom, WASD pan, Space pause, R reset",
            count.0, fps
        );
    }
}
