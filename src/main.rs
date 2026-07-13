mod menu;
mod particles;

use std::sync::Arc;

use bevy::core_pipeline::bloom::Bloom;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::render::camera::ClearColorConfig;
use bevy::render::view::Msaa;
use bevy::window::PresentMode;

use bevy::math::DVec2;

use menu::{MenuPlugin, Settings};
use particles::{
    generate_particles, reference_orbit, ParticlePlugin, ParticleSeed, RefOrbit, SimParams,
    MAX_PARTICLES, REF_ORBIT_CAP,
};

const BASE_ITER: u32 = 240;
const DEFAULT_CENTER: DVec2 = DVec2::new(-0.55, 0.0);
const DEFAULT_HEIGHT: f64 = 2.7;

#[derive(Resource)]
struct Paused(bool);

/// View transform. center/height are f64 so the reference point keeps ~15
/// digits of precision, enough for zoom down to ~1e-15 (near-infinite feel).
#[derive(Resource)]
struct ViewState {
    center: DVec2,
    height: f64,
    /// Center from the previous frame, for per-frame particle rebasing.
    prev_center: DVec2,
    /// Height from the previous frame, to size the zoom-out reseed burst.
    prev_height: f64,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            center: DEFAULT_CENTER,
            height: DEFAULT_HEIGHT,
            prev_center: DEFAULT_CENTER,
            prev_height: DEFAULT_HEIGHT,
        }
    }
}

/// Iteration count grows with zoom depth so deep boundary detail resolves.
/// Capped well below REF_ORBIT_CAP: cost is max_iter x 3 x particle_count every
/// frame, so an uncapped ramp tanks the framerate (and makes input feel dead).
fn depth_iter(height: f64, detail: f32) -> u32 {
    let zoom = (DEFAULT_HEIGHT / height).max(1.0);
    let raw = (240.0 + 90.0 * zoom.log2()) * detail as f64;
    // Cap below REF_ORBIT_CAP: the perturbation reference orbit is that long.
    raw.clamp(60.0, (REF_ORBIT_CAP - 1) as f64) as u32
}

fn main() {
    let count = std::env::args()
        .nth(1)
        .map(|s| s.replace('_', ""))
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(2_000_000)
        .min(MAX_PARTICLES);

    println!("Fractality controls:");
    println!("  hover      ripple particles");
    println!("  left-click blast");
    println!("  right-click vortex");
    println!("  wheel      zoom toward cursor");
    println!("  WASD       pan");
    println!("  Space      pause");
    println!("  R          reset view");
    println!("  M / Esc    settings menu");

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
        .add_plugins(MenuPlugin)
        .insert_resource(Settings {
            particle_count: count,
            ..default()
        })
        .insert_resource(ViewState::default())
        .insert_resource(Paused(false))
        .insert_resource(SimParams::default())
        .insert_resource(RefOrbit::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            ((handle_input, update_params).chain(), update_title),
        )
        .run();
}

fn setup(mut commands: Commands, settings: Res<Settings>) {
    commands.spawn((
        Camera2d,
        Camera {
            hdr: true,
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Bloom {
            intensity: settings.bloom,
            ..Bloom::NATURAL
        },
        Msaa::Off,
    ));

    let start = std::time::Instant::now();
    // Positions are stored relative to the view center; seed at the default one.
    // Seed only the initial active count (fast startup). The GPU buffer is sized
    // to MAX_PARTICLES; raising the count later fills the tail via recycle.
    let particles =
        generate_particles(settings.particle_count as usize, BASE_ITER, DEFAULT_CENTER);
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
    let dt = time.delta_secs() as f64;

    let mut pan = DVec2::ZERO;
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
    if pan != DVec2::ZERO {
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

    let mut scroll = 0.0f64;
    for ev in wheel.read() {
        scroll += match ev.unit {
            MouseScrollUnit::Line => ev.y as f64,
            MouseScrollUnit::Pixel => ev.y as f64 / 60.0,
        };
    }
    if scroll != 0.0 {
        let Ok(window) = windows.single() else {
            return;
        };
        let old_h = view.height;
        // Lower clamp near f64 precision floor for the center; feels infinite.
        let new_h = (old_h * 0.9f64.powf(scroll)).clamp(1e-15, 40.0);
        if let Some(cursor) = window.cursor_position() {
            let w = window.width().max(1.0) as f64;
            let h = window.height().max(1.0) as f64;
            let aspect = w / h;
            let ndc = DVec2::new(cursor.x as f64 / w * 2.0 - 1.0, 1.0 - cursor.y as f64 / h * 2.0);
            let cursor_fractal =
                view.center + DVec2::new(ndc.x * old_h * aspect * 0.5, ndc.y * old_h * 0.5);
            let delta = (cursor_fractal - view.center) * (1.0 - new_h / old_h);
            view.center += delta;
        }
        view.height = new_h;
    }
}

fn update_params(
    time: Res<Time>,
    windows: Query<&Window>,
    mut view: ResMut<ViewState>,
    paused: Res<Paused>,
    mouse: Res<ButtonInput<MouseButton>>,
    settings: Res<Settings>,
    mut params: ResMut<SimParams>,
    mut ref_orbit: ResMut<RefOrbit>,
    mut frame: Local<u32>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let w = window.width().max(1.0);
    let h = window.height().max(1.0);
    let aspect = (w / h) as f64;

    // Positions are center-relative, so clip = pos * scale (offset is zero).
    let scale = Vec2::new(
        (2.0 / (view.height * aspect)) as f32,
        (2.0 / view.height) as f32,
    );

    // Rebase amount for this frame, computed in f64 so it stays tiny and exact.
    let center_delta = (view.prev_center - view.center).as_vec2();
    view.prev_center = view.center;

    // Zoom-out reseed budget: fraction of the view area newly revealed this
    // frame (1 - (prev_h/cur_h)^2 when zooming out), so fill exactly tracks the
    // growing view with no lagging density front. Floor at a small trickle for
    // steady-state coverage, cap to bound the per-frame churn.
    let area_ratio = (view.prev_height / view.height).powi(2);
    let revealed = (1.0 - area_ratio).clamp(0.0, 1.0);
    let reseed_rate = (revealed.max(0.006) as f32).min(0.2);
    view.prev_height = view.height;

    // Cursor in center-relative coords (same space as particles).
    let mouse_rel = window.cursor_position().map(|c| {
        let ndc = Vec2::new(c.x / w * 2.0 - 1.0, 1.0 - c.y / h * 2.0);
        ndc / scale
    });
    let button = if mouse.pressed(MouseButton::Left) {
        1.0
    } else if mouse.pressed(MouseButton::Right) {
        -1.0
    } else {
        0.0
    };
    let radius = (view.height * 0.09) as f32;

    let dt = if paused.0 {
        0.0
    } else {
        time.delta_secs().min(1.0 / 30.0)
    };

    let count = settings.particle_count.clamp(1, MAX_PARTICLES);
    let max_iter = depth_iter(view.height, settings.detail);
    // High-precision reference orbit at the view center for perturbation.
    ref_orbit.0 = reference_orbit(view.center.x, view.center.y, max_iter);

    *frame = frame.wrapping_add(1);

    let px = settings.dot_px;
    let u = &mut params.0;
    u.world_to_clip = Vec4::new(scale.x, scale.y, 0.0, 0.0);
    u.mouse = match mouse_rel {
        Some(p) => Vec4::new(p.x, p.y, button, radius),
        None => Vec4::new(1e9, 1e9, 0.0, radius),
    };
    u.particle_size = Vec2::new(2.0 * px / w, 2.0 * px / h);
    u.center_delta = center_delta;
    u.time = time.elapsed_secs();
    u.dt = dt;
    u.count = count;
    u.max_iter = max_iter;
    // Flow speed as a fraction of view height per second (shader scales by
    // view_height). Default 0.081 matches the original 0.22 feel at base zoom.
    u.flow_speed = settings.flow_speed;
    u.band_k = 0.7;
    u.damping = 3.0;
    // Normalize brightness by density so a given user setting looks the same at
    // any particle count, then scale by the user's brightness knob.
    u.brightness = settings.brightness * (500_000.0f32 / count as f32).sqrt();
    u.ref_len = ref_orbit.0.len() as u32;
    u.frame = *frame;
    u.reseed_rate = reseed_rate;
    u.detail = settings.detail;
}

fn update_title(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    settings: Res<Settings>,
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
            "Fractality | {} particles | {:.0} FPS | wheel zoom, WASD pan, M menu",
            settings.particle_count, fps
        );
    }
}
