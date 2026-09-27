// Bevy systems take their resources as parameters and filter with nested
// query types; both lints fire on idiomatic system signatures.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

mod audio;
mod choreographer;
mod dd;
mod fractal;
mod menu;
mod particles;
mod recorder;
#[cfg(target_arch = "wasm32")]
mod webutil;

use std::sync::Arc;

use bevy::core_pipeline::bloom::Bloom;
#[cfg(not(target_arch = "wasm32"))]
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::math::DVec2;
use bevy::prelude::*;
use bevy::render::camera::ClearColorConfig;
#[cfg(not(target_arch = "wasm32"))]
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
#[cfg(not(target_arch = "wasm32"))]
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use bevy::render::view::Msaa;
#[cfg(not(target_arch = "wasm32"))]
use bevy::ui::IsDefaultUiCamera;
use bevy::window::PresentMode;

use audio::{AudioCapture, AudioLevels};
use choreographer::{update_choreographer, Choreographer};
use dd::DdVec2;
use fractal::{julia_morph_c, reference_orbit, Fractal, JULIA_C, REF_ORBIT_CAP};
use menu::{MenuPlugin, PointerOverMenu, Settings};
use particles::{
    generate_particles, ParticleCamera, ParticlePlugin, ParticleSeed, RefOrbit, SimParams,
    MAX_PARTICLES,
};
use recorder::{Recorder, RecorderPlugin};

/// Offscreen render target the scene camera draws into (native only; on web
/// the camera renders straight to the canvas). The window shows it through a
/// backdrop node on the UI camera, so captures (P screenshot, O recording)
/// read this texture and never contain the menu or the REC overlay.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource)]
pub struct SceneTarget(pub Handle<Image>);

/// Iteration count at base zoom: the startup seed's depth and the floor of
/// `depth_iter`'s ramp (deeper views ramp above it, and `iter_budget_count`
/// gives back particles in proportion).
const BASE_ITER: u32 = 240;
const DEFAULT_CENTER: DVec2 = Fractal::Mandelbrot.home_view().0;
const DEFAULT_HEIGHT: f64 = Fractal::Mandelbrot.home_view().1;

/// Deepest view height: the double-double center carries ~31 digits, and a
/// couple of digits of margin keep sub-pixel accuracy at the floor.
const MIN_HEIGHT: f64 = 1e-28;
/// Widest view height the wheel can reach.
const MAX_HEIGHT: f64 = 40.0;

/// Dissolve mode (Space): the flow field goes inert and particles melt off
/// their contours.
#[derive(Resource, Default)]
struct Dissolve(bool);

/// Palette selector, cycled with C (or by clicking the menu row). Names
/// indexed by the mode id.
#[derive(Resource, Default)]
pub struct ColorMode(pub u32);

pub const COLOR_MODES: [&str; 5] = ["classic", "rings", "electric", "inferno", "audio aurora"];

/// Fractal formula selector, cycled with F (or by clicking the menu row).
#[derive(Resource, Default)]
pub struct FractalType(pub Fractal);

/// Particle flow style, cycled with G (or picked from the menu dropdown).
/// Ids must match the flow_mode switch in the compute shader.
#[derive(Resource, Default)]
pub struct FlowMode(pub u32);

pub const FLOW_MODES: [&str; 6] = ["contour", "layers", "gravity", "erupt", "pulse", "dynamics"];

/// Kaleidoscope state. Toggled with K (or by clicking the menu row); the fold
/// count and spin gain live in Settings sliders, shown only while on. Pure
/// display effect in the trail composite shader; the simulation never sees
/// it. `rot` is the music-driven rotation angle, accumulated per frame.
#[derive(Resource, Default)]
pub struct Kaleido {
    pub on: bool,
    pub rot: f32,
}

/// Continuous zoom dive toward the cursor, toggled with Z. Any manual
/// navigation (wheel, pan, R) cancels it.
#[derive(Resource, Default)]
struct AutoZoom(bool);

#[derive(Clone, Copy)]
struct Bookmark {
    center: DdVec2,
    height: f64,
}

/// View bookmarks: Shift+1..9 saves the current view, 1..9 flies back to it.
#[derive(Resource, Default)]
struct Bookmarks([Option<Bookmark>; 9]);

/// In-flight animated transition to a bookmarked view.
#[derive(Resource, Default)]
struct FlyTo(Option<Bookmark>);

/// View transform. The center is double-double (~31 digits) so the
/// perturbation reference point stays exact down to the height floor of
/// ~1e-28; plain f64 (~16 digits) capped useful zoom at ~1e-15. Height stays
/// f64: it is a scale, not a position, so only its exponent range matters.
#[derive(Resource)]
struct ViewState {
    center: DdVec2,
    height: f64,
    /// Center from the previous frame, for per-frame particle rebasing.
    prev_center: DdVec2,
    /// Height from the previous frame, to size the zoom-out reseed burst.
    prev_height: f64,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            center: DdVec2::from_dvec2(DEFAULT_CENTER),
            height: DEFAULT_HEIGHT,
            prev_center: DdVec2::from_dvec2(DEFAULT_CENTER),
            prev_height: DEFAULT_HEIGHT,
        }
    }
}

impl ViewState {
    /// Jump to the fractal's home view (R, and fractal switches).
    fn go_home(&mut self, fractal: Fractal) {
        let (center, height) = fractal.home_view();
        self.center = DdVec2::from_dvec2(center);
        self.height = height;
    }
}

/// Advance a mode id to the next of `n`, wrapping.
fn cycle(id: &mut u32, n: usize) {
    *id = (*id + 1) % n as u32;
}

/// Iteration count grows with zoom depth so deep boundary detail resolves.
/// Capped at REF_ORBIT_CAP (the reference orbit length) and no lower:
/// pinning it lower freezes the depth, and a dive past the pin stops
/// resolving new structure, going mushy instead of deep. Frame cost is held
/// flat by `iter_budget_count`, not by pinning the iteration count.
fn depth_iter(height: f64, detail: f32) -> u32 {
    let zoom = (DEFAULT_HEIGHT / height).max(1.0);
    let l = zoom.log2();
    // Depth margin: resolve structure ~2 doublings (180 iterations) before
    // the zoom reaches its scale. Without it, regions whose escape count sits
    // just above the ramp render as empty interior, then pop in fully sized
    // the moment the ramp crosses them - near minibrots escape counts cluster,
    // so whole filament webs appeared at once. With the margin they seed
    // while still ~4x smaller on screen and grow in over ~1.5 s of dive.
    // Tapered over the first 4 doublings so base-zoom cost is unchanged.
    let margin = 180.0 * (l * 0.25).min(1.0);
    let raw = (BASE_ITER as f64 + 90.0 * l + margin) * detail as f64;
    // Cap at REF_ORBIT_CAP: the perturbation reference orbit is that long.
    raw.clamp(60.0, (REF_ORBIT_CAP - 1) as f64) as u32
}

/// Iteration count the sim is known to afford at the user's full particle
/// count: the old REF_ORBIT_CAP, which ran everywhere at full density before
/// the cap was raised. Below this depth the count is never reduced.
const AFFORDABLE_ITER: f32 = 2047.0;

/// Trade particle count against iteration depth PAST the old cap, holding
/// per-frame work (~count x max_iter) near its old worst-case value. Shallow
/// and mid zoom are untouched; only dives beyond height ~1e-5 thin the swarm
/// (to ~23% at the height floor of 1e-28). Brightness normalization
/// downstream compensates for the lower density, so the image does not dim.
///
/// The budget scales with `detail`: max_iter is proportional to it, so
/// without the scaling the slider would trade particles away 1:1 and
/// raising detail would VISIBLY remove particles. Detail is the user
/// explicitly buying more per-particle work, so it costs frame rate (as it
/// always did shallow), never density.
fn iter_budget_count(user_count: u32, max_iter: u32, detail: f32) -> u32 {
    let affordable = AFFORDABLE_ITER * detail.max(0.01);
    let ratio = (affordable / max_iter as f32).clamp(0.0, 1.0);
    ((user_count as f32 * ratio) as u32).clamp(1, user_count)
}

/// Startup particle count when no CLI argument overrides it (also the
/// Settings::default() value). Lower on web: generation is single-threaded
/// there (no rayon), and the menu slider can raise it live anyway.
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_COUNT: u32 = 2_000_000;
#[cfg(target_arch = "wasm32")]
const DEFAULT_COUNT: u32 = 500_000;

/// Wall-clock seconds for output filenames (screenshots, recordings). Wasm
/// has no SystemTime; session uptime is unique enough there (the browser
/// dedups collisions).
pub(crate) fn output_stamp(time: &Time) -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = time;
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
    #[cfg(target_arch = "wasm32")]
    {
        time.elapsed_secs() as u64
    }
}

/// Web screenshot: snapshot the composited canvas via toBlob and trigger a
/// download. Bevy's Screenshot entity path (GPU buffer readback) comes back
/// black on the browser WebGPU backend; the canvas always holds the last
/// presented frame, so this needs no readback at all.
#[cfg(target_arch = "wasm32")]
fn web_screenshot(path: &str) {
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast;

    let Some(canvas) = webutil::canvas() else {
        warn!("screenshot: canvas #fractality-canvas not found");
        return;
    };
    let name = path.to_owned();
    // once_into_js: freed after the browser invokes it (leaks only if the
    // browser never calls back, which it does even on encode failure).
    let cb = Closure::once_into_js(move |blob: Option<web_sys::Blob>| {
        if let Some(blob) = blob {
            webutil::download_blob(&blob, &name);
        }
    });
    if canvas.to_blob(cb.unchecked_ref()).is_err() {
        warn!("screenshot: canvas.toBlob failed");
    }
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    let count = std::env::args()
        .nth(1)
        .map(|s| s.replace('_', ""))
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_COUNT)
        .min(MAX_PARTICLES);

    println!("Fractality controls:");
    for (key, action) in menu::CONTROLS {
        println!("  {key:<10} {action}");
    }
    println!(
        "  (with the menu hidden, the choreographer self-starts after 30 s idle; any input exits)"
    );

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Fractality".into(),
            present_mode: PresentMode::AutoNoVsync,
            // Web: attach to the page's canvas and track its size.
            // Both fields are no-ops on native.
            canvas: Some("#fractality-canvas".into()),
            fit_canvas_to_parent: true,
            ..default()
        }),
        ..default()
    }))
    .add_plugins(FrameTimeDiagnosticsPlugin::default())
    .add_plugins(ParticlePlugin)
    .add_plugins(MenuPlugin)
    .add_plugins(RecorderPlugin)
    .insert_resource(Settings {
        particle_count: count,
        ..default()
    })
    .init_resource::<ViewState>()
    .init_resource::<Dissolve>()
    .init_resource::<ColorMode>()
    .init_resource::<FractalType>()
    .init_resource::<FlowMode>()
    .init_resource::<Kaleido>()
    .init_resource::<AutoZoom>()
    .init_resource::<Bookmarks>()
    .init_resource::<FlyTo>()
    .init_resource::<Choreographer>()
    .init_resource::<SimParams>()
    .init_resource::<RefOrbit>()
    .init_resource::<AudioCapture>()
    .init_resource::<AudioLevels>()
    .add_systems(Startup, setup)
    .add_systems(
        Update,
        (
            (
                handle_input,
                apply_fractal_switch,
                audio::manage_capture,
                audio::update_audio,
                update_choreographer,
                update_params,
            )
                .chain(),
            update_title,
        ),
    );
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, sync_scene_target);
    app.run();
}

/// The offscreen scene texture: window-sized, tonemapped sRGB, readable by
/// the screenshot/recording paths (COPY_SRC) and sampleable by the backdrop
/// node (TEXTURE_BINDING).
#[cfg(not(target_arch = "wasm32"))]
fn scene_target_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// Follow window resizes with the offscreen target. Checked via get() first:
/// get_mut() flags the asset changed and would re-upload it every frame.
#[cfg(not(target_arch = "wasm32"))]
fn sync_scene_target(
    windows: Query<&Window>,
    target: Res<SceneTarget>,
    mut images: ResMut<Assets<Image>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let want = Extent3d {
        width: window.physical_width().max(1),
        height: window.physical_height().max(1),
        depth_or_array_layers: 1,
    };
    if images
        .get(&target.0)
        .is_some_and(|i| i.texture_descriptor.size != want)
    {
        if let Some(image) = images.get_mut(&target.0) {
            image.resize(want);
        }
    }
}

fn setup(
    mut commands: Commands,
    settings: Res<Settings>,
    mut windows: Query<&mut Window>,
    #[cfg(not(target_arch = "wasm32"))] mut images: ResMut<Assets<Image>>,
) {
    if let Ok(mut window) = windows.single_mut() {
        window.set_maximized(true);
    }
    // Native: the scene camera renders offscreen and a present camera puts
    // that texture on the window underneath the UI, so captures read the
    // texture and never contain UI. set_maximized lands later, so the
    // initial size is provisional; sync_scene_target follows the window.
    // Web: the scene camera renders straight to the canvas.
    #[cfg(not(target_arch = "wasm32"))]
    let target = {
        let size = windows.single().map_or(UVec2::new(1920, 1080), |w| {
            UVec2::new(w.physical_width(), w.physical_height())
        });
        images.add(scene_target_image(size))
    };
    commands.spawn((
        Camera2d,
        Camera {
            hdr: true,
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            #[cfg(not(target_arch = "wasm32"))]
            target: target.clone().into(),
            ..default()
        },
        Bloom {
            intensity: settings.bloom,
            ..Bloom::NATURAL
        },
        Msaa::Off,
        ParticleCamera,
    ));
    #[cfg(not(target_arch = "wasm32"))]
    {
        commands.spawn((
            Camera2d,
            Camera {
                order: 1,
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            // The backdrop image is already tonemapped; mapping it again
            // would shift the scene's colors on screen vs in captures.
            Tonemapping::None,
            Msaa::Off,
            IsDefaultUiCamera,
        ));
        // Scene backdrop: the offscreen target stretched over the window,
        // behind every other UI node.
        commands.spawn((
            ImageNode::new(target.clone()),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
            GlobalZIndex(-1),
        ));
        commands.insert_resource(SceneTarget(target));
    }

    // bevy_platform's Instant works on wasm; std's panics there.
    let start = bevy::platform::time::Instant::now();
    // Positions are stored relative to the view center; seed at the default one.
    // Seed only the initial active count (fast startup). The GPU buffer is sized
    // to MAX_PARTICLES; raising the count later fills the tail via recycle.
    let particles = generate_particles(
        settings.particle_count as usize,
        BASE_ITER,
        DEFAULT_CENTER,
        Fractal::Mandelbrot,
    );
    info!(
        "generated {} particles in {:.2?}",
        particles.len(),
        start.elapsed()
    );
    commands.insert_resource(ParticleSeed(Arc::new(particles)));
}

/// Change the view height to `new_h`, keeping the fractal point under the
/// cursor fixed on screen (falls back to a centered zoom with no cursor).
fn zoom_anchored(view: &mut ViewState, window: &Window, new_h: f64) {
    let old_h = view.height;
    if let Some(cursor) = window.cursor_position() {
        let w = window.width().max(1.0) as f64;
        let h = window.height().max(1.0) as f64;
        let aspect = w / h;
        let ndc = DVec2::new(
            cursor.x as f64 / w * 2.0 - 1.0,
            1.0 - cursor.y as f64 / h * 2.0,
        );
        // Cursor offset from the center is view-height sized, so f64 carries
        // it exactly enough; only the accumulation into center needs DD.
        let cursor_off = DVec2::new(ndc.x * old_h * aspect * 0.5, ndc.y * old_h * 0.5);
        view.center += cursor_off * (1.0 - new_h / old_h);
    }
    view.height = new_h;
}

/// One frame of anchored approach toward a target point: set the height to
/// `new_h` and pull the center so the target keeps its screen position
/// through the shrink (anchor term, same math as cursor zoom), while `k`
/// closes the remaining gap in sync. Shared by the bookmark fly-to and the
/// choreographer's dive. The remaining gap is computed exactly in DD, then
/// the f64-rounded step is fine: its error is ~1e-16 of the gap, far
/// sub-pixel.
fn approach_step(view: &mut ViewState, target: DdVec2, new_h: f64, k: f64) {
    let shrink = (1.0 - new_h / view.height).max(0.0);
    let pull = (shrink + k).min(1.0);
    let step = (target - view.center).to_dvec2() * pull;
    view.center += step;
    view.height = new_h;
}

fn handle_input(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: EventReader<MouseWheel>,
    windows: Query<&Window>,
    mut view: ResMut<ViewState>,
    mut dissolve: ResMut<Dissolve>,
    mut color_mode: ResMut<ColorMode>,
    mut fractal: ResMut<FractalType>,
    mut flow_mode: ResMut<FlowMode>,
    mut kaleido: ResMut<Kaleido>,
    mut auto_zoom: ResMut<AutoZoom>,
    mut bookmarks: ResMut<Bookmarks>,
    mut fly: ResMut<FlyTo>,
    mut audio: ResMut<AudioCapture>,
    #[cfg(not(target_arch = "wasm32"))] scene: Res<SceneTarget>,
    mut commands: Commands,
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
        // Manual navigation takes the wheel back from any autopilot.
        auto_zoom.0 = false;
        fly.0 = None;
    }

    if keys.just_pressed(KeyCode::KeyR) {
        view.go_home(fractal.0);
        auto_zoom.0 = false;
        fly.0 = None;
    }
    if keys.just_pressed(KeyCode::Space) {
        dissolve.0 = !dissolve.0;
    }
    if keys.just_pressed(KeyCode::KeyZ) {
        auto_zoom.0 = !auto_zoom.0;
        fly.0 = None;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        cycle(&mut color_mode.0, COLOR_MODES.len());
        info!("color mode: {}", COLOR_MODES[color_mode.0 as usize]);
    }
    if keys.just_pressed(KeyCode::KeyF) {
        fractal.0 = fractal.0.next();
        info!("fractal: {}", fractal.0.name());
    }
    if keys.just_pressed(KeyCode::KeyG) {
        cycle(&mut flow_mode.0, FLOW_MODES.len());
        info!("flow mode: {}", FLOW_MODES[flow_mode.0 as usize]);
    }
    if keys.just_pressed(KeyCode::KeyK) {
        kaleido.on = !kaleido.on;
        info!("kaleidoscope: {}", if kaleido.on { "on" } else { "off" });
    }
    if keys.just_pressed(KeyCode::KeyV) {
        audio.enabled = !audio.enabled;
    }
    if keys.just_pressed(KeyCode::KeyP) {
        let stamp = output_stamp(&time);
        let path = format!("fractality_{stamp}.png");
        info!("saving screenshot to {path}");
        // The offscreen scene target, not the window: the window frame has
        // the UI composited on top.
        #[cfg(not(target_arch = "wasm32"))]
        commands
            .spawn(Screenshot::image(scene.0.clone()))
            .observe(save_to_disk(path));
        #[cfg(target_arch = "wasm32")]
        {
            let _ = &mut commands; // only the native path spawns anything
            web_screenshot(&path);
        }
    }

    // Bookmarks: Shift+digit saves the current view, plain digit flies to it.
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    const DIGITS: [KeyCode; 9] = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    for (i, key) in DIGITS.iter().enumerate() {
        if keys.just_pressed(*key) {
            if shift {
                bookmarks.0[i] = Some(Bookmark {
                    center: view.center,
                    height: view.height,
                });
                info!("saved view to bookmark {}", i + 1);
            } else if let Some(b) = bookmarks.0[i] {
                fly.0 = Some(b);
                auto_zoom.0 = false;
            }
        }
    }

    let mut scroll = 0.0f64;
    for ev in wheel.read() {
        scroll += match ev.unit {
            MouseScrollUnit::Line => ev.y as f64,
            MouseScrollUnit::Pixel => ev.y as f64 / 60.0,
        };
    }
    let Ok(window) = windows.single() else {
        return;
    };
    if scroll != 0.0 {
        let new_h = (view.height * 0.9f64.powf(scroll)).clamp(MIN_HEIGHT, MAX_HEIGHT);
        zoom_anchored(&mut view, window, new_h);
        auto_zoom.0 = false;
        fly.0 = None;
    }

    // Auto-zoom dive: constant exponential rate toward the cursor, so the
    // apparent speed is the same at every depth. Stops at the precision floor.
    if auto_zoom.0 {
        let new_h = (view.height * (-0.9 * dt).exp()).max(MIN_HEIGHT);
        zoom_anchored(&mut view, window, new_h);
        if new_h <= MIN_HEIGHT {
            auto_zoom.0 = false;
        }
    }

    // Animated fly-to. Height moves in log space with a fixed time constant,
    // so the trip takes a couple of seconds regardless of depth. The center
    // follows via `approach_step` with the height's own decay `k`, so the
    // remaining error closes in sync instead of leaving the target off-screen.
    if let Some(target) = fly.0 {
        let k = 1.0 - (-2.5 * dt).exp();
        let old_h = view.height;
        let new_h = (old_h.ln() + (target.height.ln() - old_h.ln()) * k).exp();
        // The final snap below assigns the DD target exactly.
        approach_step(&mut view, target.center, new_h, k);
        let err = (target.center - view.center).to_dvec2().length();
        if (new_h / target.height).ln().abs() < 0.005 && err < target.height * 0.002 {
            view.center = target.center;
            view.height = target.height;
            fly.0 = None;
        }
    }
}

/// Whenever the fractal type changes (F key or menu click), jump to that
/// fractal's home view and cancel any autopilot. The recycle trickle then
/// resamples the particle cloud onto the new set within a couple of seconds.
fn apply_fractal_switch(
    fractal: Res<FractalType>,
    choreo: Res<Choreographer>,
    mut view: ResMut<ViewState>,
    mut auto_zoom: ResMut<AutoZoom>,
    mut fly: ResMut<FlyTo>,
) {
    if !fractal.is_changed() || fractal.is_added() {
        return;
    }
    // While the choreographer runs it owns fractal switches: its scene cut
    // lands on a boundary spot itself, so the home snap here would fight it
    // (and a manual F mid-show becomes a live morph, retargeted within a
    // few seconds).
    if choreo.on {
        return;
    }
    view.go_home(fractal.0);
    auto_zoom.0 = false;
    fly.0 = None;
}

/// State `update_params` carries between frames. Bundled into a single `Local`
/// because the system is at Bevy's 16-parameter limit.
#[derive(Default)]
struct ParamState {
    frame: u32,
    /// Eased particle count, so the depth-vs-count trade ramps instead of
    /// snapping. Zero means "not yet initialized".
    smooth_count: f32,
    last_orbit_key: Option<(DdVec2, u32, Fractal, (f64, f64))>,
    /// Flow-mode crossfade: the last mode seen (change detector), the mode
    /// being faded out, and when the switch happened. Defaults (all zero)
    /// mean "prev == current, ramp long done", which the shader treats as
    /// no morph.
    last_flow: u32,
    flow_prev: u32,
    morph_t0: f32,
}

fn update_params(
    time: Res<Time>,
    windows: Query<&Window>,
    mut view: ResMut<ViewState>,
    dissolve: Res<Dissolve>,
    color_mode: Res<ColorMode>,
    fractal: Res<FractalType>,
    flow_mode: Res<FlowMode>,
    mut kaleido: ResMut<Kaleido>,
    mouse: Res<ButtonInput<MouseButton>>,
    over_menu: Res<PointerOverMenu>,
    settings: Res<Settings>,
    audio: Res<AudioLevels>,
    choreo: Res<Choreographer>,
    mut params: ResMut<SimParams>,
    mut ref_orbit: ResMut<RefOrbit>,
    mut state: Local<ParamState>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let w = window.width().max(1.0);
    let h = window.height().max(1.0);
    let aspect = (w / h) as f64;

    // Zoom breathe: bass squeezes the RENDERED height only. view.height (and
    // everything derived from it - reseed budget, orbit cache, iteration
    // depth, zoom input) never sees the wobble, so it cannot feed back into
    // navigation or force per-frame orbit recomputes.
    let eff_height = view.height * (1.0 - 0.03 * audio.bass as f64 * settings.audio_breathe as f64);

    // Positions are center-relative, so clip = pos * scale (offset is zero).
    let scale = Vec2::new(
        (2.0 / (eff_height * aspect)) as f32,
        (2.0 / eff_height) as f32,
    );

    // Rebase amount for this frame, computed in DD so it stays tiny and exact.
    let center_delta = (view.prev_center - view.center).to_dvec2().as_vec2();
    view.prev_center = view.center;

    // Reseed budget. Two contributions, take the max:
    //  - zoom-OUT: fraction of view area newly revealed (1 - (prev_h/cur_h)^2),
    //    so fill tracks the growing view with no lagging density front.
    //  - any zoom MOTION: |log2(prev_h/cur_h)| per frame. Zooming IN reveals no
    //    new area but exposes finer filaments; without this term zoom-in keeps
    //    the stale coarse sampling and looks worse than zoom-out. Resampling
    //    concentrates particles onto the now-visible fine boundary.
    let area_ratio = (view.prev_height / view.height).powi(2);
    let revealed = (1.0 - area_ratio).clamp(0.0, 1.0);
    let zoom_motion = (view.prev_height / view.height).log2().abs() * 6.0;
    let reseed_rate = (revealed.max(zoom_motion).max(0.006) as f32).min(0.4);
    view.prev_height = view.height;

    // Cursor in center-relative coords (same space as particles).
    let mouse_rel = window.cursor_position().map(|c| {
        let ndc = Vec2::new(c.x / w * 2.0 - 1.0, 1.0 - c.y / h * 2.0);
        ndc / scale
    });
    // Cursor over the panel (or mid slider-drag): clicks are for the UI,
    // not the blast/vortex force.
    let button = if over_menu.0 {
        0.0
    } else if mouse.pressed(MouseButton::Left) {
        1.0
    } else if mouse.pressed(MouseButton::Right) {
        -1.0
    } else {
        0.0
    };
    // From view.height, not eff_height: the breathe is render-only and must
    // not modulate the physical blast/vortex radius.
    let radius = (view.height * 0.09) as f32;

    let dt = time.delta_secs().min(1.0 / 30.0);

    let user_count = settings.particle_count.clamp(1, MAX_PARTICLES);
    let max_iter = depth_iter(view.height, settings.detail);
    // Ease toward the budgeted count instead of snapping. Particles above the
    // live count are not simulated, so their stored positions go stale; on
    // zoom-out they come back a trickle at a time and the compute shader's
    // drifted-out recycling absorbs them without a visible pop.
    let target_count = iter_budget_count(user_count, max_iter, settings.detail) as f32;
    if state.smooth_count <= 0.0 {
        state.smooth_count = target_count;
    }
    state.smooth_count += (target_count - state.smooth_count) * (1.0 - (-3.0 * dt).exp());
    let count = (state.smooth_count as u32).clamp(1, user_count);
    // Julia morph: music orbits the Julia parameter around its home value, so
    // the fractal shape itself dances. The GPU never sees c directly - the
    // reference orbit encodes it - so a changed c just means a fresh orbit
    // (cheap: at most max_iter double-double steps, 16 bytes of upload each).
    let jc = if fractal.0 == Fractal::Julia && settings.audio_morph > 0.0 {
        let amp = 0.04 * settings.audio_morph as f64 * (0.25 + audio.level as f64);
        julia_morph_c(audio.morph_phase as f64, amp)
    } else {
        JULIA_C
    };
    // High-precision reference orbit at the view center for perturbation.
    // Only recompute (and re-upload, via the generation bump) when the view
    // center, iteration count, or Julia c actually changed; a static view
    // with no morph pays nothing.
    let orbit_key = (view.center, max_iter, fractal.0, jc);
    if state.last_orbit_key != Some(orbit_key) {
        ref_orbit.points = Arc::new(reference_orbit(view.center, max_iter, fractal.0, jc));
        ref_orbit.generation = ref_orbit.generation.wrapping_add(1);
        state.last_orbit_key = Some(orbit_key);
    }

    state.frame = state.frame.wrapping_add(1);

    let px = settings.dot_px;
    let u = &mut params.0;
    u.world_to_clip = Vec4::new(scale.x, scale.y, 0.0, 0.0);
    u.mouse = match mouse_rel {
        Some(p) => Vec4::new(p.x, p.y, button, radius),
        None => Vec4::new(1e9, 1e9, 0.0, radius),
    };
    if let Some(blast) = choreo.blast() {
        u.mouse = Vec4::new(blast.x, blast.y, 1.0, radius);
    }
    u.particle_size = Vec2::new(2.0 * px / w, 2.0 * px / h);
    u.center_delta = center_delta;
    u.time = time.elapsed_secs();
    u.dt = dt;
    u.count = count;
    u.max_iter = max_iter;
    // Flow speed as a fraction of view height per second (shader scales by
    // view_height). Default 0.081 matches the original 0.22 feel at base zoom.
    u.flow_speed = settings.flow_speed;
    u.dissolve = if dissolve.0 { 1.0 } else { 0.0 };
    u.band_k = settings.align_force;
    u.damping = 3.0;
    // Normalize brightness by density so a given user setting looks the same at
    // any particle count, then scale by the user's brightness knob.
    u.brightness = settings.brightness * (500_000.0f32 / count as f32).sqrt();
    // Trails: per-frame keep factor, corrected to a 60 FPS reference so the
    // trail length is frame-rate independent. 0 = off (direct draw path).
    // With per-frame keep factor `keep`, additive accumulation converges to
    // emission / (1 - keep), so scale emission by exactly (1 - keep). This is
    // both frame-rate independent (steady state = b regardless of fps) and
    // jitter-proof: at steady state each frame outputs keep*S + (1-keep)*b = b
    // no matter how dt fluctuates, so vsync-off frame-time noise cannot pulse
    // the brightness. It is also continuous at trail -> 0 (factor -> 1), so
    // enabling trails at low strength does not darken the image.
    // The floor keeps the per-frame deposit large enough to register against
    // the f16 accumulator at very long trails / very high fps.
    // A drop temporarily lengthens the trails (dreamy smear through the hit),
    // easing back over a couple of seconds. Only when trails are already on.
    let trail = if settings.trail > 0.0 {
        (settings.trail + 0.3 * audio.drop).min(0.97)
    } else {
        settings.trail
    };
    if trail > 0.0 {
        let frames_60 = (time.delta_secs() * 60.0).clamp(0.1, 4.0);
        let keep = trail.powf(frames_60);
        u.trail_decay = keep;
        u.brightness *= (1.0 - keep).max(0.02);
    } else {
        u.trail_decay = 0.0;
    }
    u.ref_len = ref_orbit.points.len() as u32;
    u.frame = state.frame;
    u.reseed_rate = if choreo.refilling() {
        reseed_rate.max(0.2)
    } else {
        reseed_rate
    };
    u.detail = settings.detail;
    u.color_mode = color_mode.0;
    u.fractal_type = fractal.0.id();
    u.flow_mode = flow_mode.0;

    // Audio reactivity (V): mids push the streams faster, bass swells the
    // dots, the overall level and beat lift brightness. Levels decay to zero
    // while disabled, so every term collapses to identity with no branch.
    // Shader-side effects (ring waves, treble shimmer, spectrum glow, hue
    // spin) read the raw levels the same way.
    u.flow_speed *= 1.0 + audio.mid * 2.5 * settings.audio_flow;
    u.brightness *= 1.0 + audio.level * 0.6 + audio.beat * 0.5;
    u.particle_size *= 1.0 + audio.bass * 0.7;
    u.audio = Vec4::new(audio.bass, audio.mid, audio.treble, audio.beat);
    u.audio_hue = audio.hue_phase;
    // Kaleidoscope rotation: slow constant drift so the mandala is never
    // static, plus music level and beats spinning it up, all scaled by the
    // Spin slider (0 = frozen wedges). Accumulated here (not derived from
    // time) so the speed reacts, not the absolute angle.
    // Bypass change detection so is_changed keeps meaning "toggled" (K / menu
    // click), not "rotated this frame".
    if kaleido.on {
        kaleido.bypass_change_detection().rot += time.delta_secs()
            * settings.kaleido_spin
            * (0.05 + audio.level * 0.8 + audio.beat * 1.2);
    }
    u.audio_aux = Vec4::new(audio.beat_age, audio.drop_age, kaleido.rot, audio.level);
    u.audio_fx = Vec4::new(
        settings.audio_pulse,
        settings.audio_flash,
        settings.audio_glow,
        if kaleido.on {
            settings.kaleido_folds
        } else {
            0.0
        },
    );
    for i in 0..4 {
        u.spectrum[i] = Vec4::from_slice(&audio.spectrum[i * 4..i * 4 + 4]);
    }
    // Flow-mode crossfade: on a switch (G key or menu), remember the old
    // mode and restart the 2s ramp. The compute shader evaluates both flow
    // fields while shape.z < 1 and blends them, so mode changes melt into
    // each other instead of hard-cutting. Switching again mid-morph fades
    // from the new mode's start, which the velocity smoothing absorbs.
    let now = time.elapsed_secs();
    if flow_mode.0 != state.last_flow {
        state.flow_prev = state.last_flow;
        state.morph_t0 = now;
        state.last_flow = flow_mode.0;
    }
    let flow_blend = ((now - state.morph_t0) / 2.0).clamp(0.0, 1.0);
    u.shape = Vec4::new(
        settings.condensation,
        state.flow_prev as f32,
        flow_blend,
        0.0,
    );
}

fn update_title(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    settings: Res<Settings>,
    params: Res<SimParams>,
    recorder: Res<Recorder>,
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
    let rec = if recorder.is_active() { " | REC" } else { "" };
    // Live count, not the setting: deep zoom trades particles for iterations,
    // so the two diverge and the live number is the one that explains the FPS.
    let live = params.0.count;
    let depth = if live < settings.particle_count {
        format!(" of {}", settings.particle_count)
    } else {
        String::new()
    };
    if let Ok(mut window) = windows.single_mut() {
        window.title = format!(
            "Fractality | {live}{depth} particles | {} iter | {:.0} FPS{rec} | wheel zoom, WASD pan, M menu",
            params.0.max_iter, fps
        );
    }
}
