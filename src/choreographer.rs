//! The auto-choreographer (classic arcade "attract mode"): an autopilot that
//! dives into the busiest boundary in sight and restyles the show as it goes.
//! Toggled with X or the menu row, or engaging on its own after 30 s without
//! input while the menu is hidden. Everything here is glue over existing
//! features - the dive shares the fly-to approach step, and palette, flow,
//! kaleidoscope, and trail changes go through the same resources the keys
//! and sliders use.

use bevy::input::mouse::MouseWheel;
use bevy::math::DVec2;
use bevy::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::audio::AudioLevels;
use crate::dd::DdVec2;
use crate::fractal::{smooth_iter, Fractal};
use crate::menu::{MenuOpen, Settings};
use crate::{
    approach_step, current_julia_c, depth_iter, AutoZoom, ColorMode, FlowMode, FlyTo, FractalType,
    Kaleido, ViewState, COLOR_MODES, DEFAULT_CENTER, FLOW_MODES,
};

/// Idle seconds before the autopilot engages on its own.
const IDLE_SECS: f32 = 30.0;

/// Iteration cap for boundary scans: enough contrast to find the boundary at
/// any depth above the dive floor, bounded so a scan stays cheap (see
/// pick_boundary_target).
const SCAN_ITER_CAP: u32 = 3000;

/// Dive floor: boundary targeting samples the escape field in plain f64,
/// which loses the boundary below ~1e-13, so scenes cut to the next one here
/// instead of diving blind toward the 1e-28 precision floor.
const DIVE_FLOOR: f64 = 1e-12;

/// Autopilot state. The dive heads into the busiest boundary in sight
/// (retargeting as new structure resolves), phases the kaleidoscope in and
/// out with fresh fold/spin styling, cycles flow modes through the
/// crossfade, swaps the palette on song drops (or on a timer), varies the
/// trail length, fires blast bursts, and hard-cuts to a new scene at the
/// dive floor. Idle engagement hands control back on any input; explicit
/// engagement (manual) ignores input and only the toggle exits. On exit the
/// kaleidoscope and the touched sliders (folds, spin, trails) return to their
/// pre-engage values; the view stays wherever the dive got to (R resets).
#[derive(Resource)]
pub struct Choreographer {
    /// Seconds since the last user input.
    idle: f32,
    pub on: bool,
    /// Set by the X key or the menu row; consumed by update_choreographer.
    pub want_toggle: bool,
    /// Engaged explicitly: input no longer exits, only the toggle does.
    manual: bool,
    target: DdVec2,
    /// Eased aim point chasing `target`, so a retarget bends the dive
    /// instead of yanking it sideways.
    aim: DdVec2,
    /// Countdown to the next boundary-target rescan.
    retarget: f32,
    /// Some = a scene cut is armed (the dive reached the floor) and this is
    /// the remaining hang time at the deepest point: with music playing the
    /// cut waits up to 1.2 s to land on a beat.
    cut_wait: Option<f32>,
    /// Post-cut reseed boost countdown (see `refilling`).
    reseed_boost: f32,
    /// Countdown to the next blast, and remaining seconds of the current one
    /// (the synthetic "click" holds a few frames so it moves real mass).
    next_burst: f32,
    burst: f32,
    /// Center-relative world position of the current blast.
    burst_pos: Vec2,
    /// Countdown to the next kaleidoscope phase flip.
    kaleido_t: f32,
    /// Countdown to the next styling tweak (folds / spin / trails).
    style_t: f32,
    /// Countdown to the next flow-mode cycle.
    flow_t: f32,
    /// Minimum spacing between drop-driven palette swaps, and the fallback
    /// countdown that swaps anyway when the music never drops.
    palette_hold: f32,
    palette_t: f32,
    /// Last frame's `AudioLevels::drop_age`; a drop shows up as it resetting.
    prev_drop_age: f32,
    /// Pre-engage kaleidoscope state and slider values, restored on exit.
    saved_kaleido: bool,
    saved_folds: f32,
    saved_spin: f32,
    saved_trail: f32,
    last_cursor: Option<Vec2>,
    /// Choreography dice; reseeded from app uptime on every engage.
    rng: fastrand::Rng,
}

impl Default for Choreographer {
    fn default() -> Self {
        Self {
            idle: 0.0,
            on: false,
            want_toggle: false,
            manual: false,
            target: DdVec2::from_dvec2(DEFAULT_CENTER),
            aim: DdVec2::from_dvec2(DEFAULT_CENTER),
            retarget: 0.0,
            cut_wait: None,
            reseed_boost: 0.0,
            next_burst: 0.0,
            burst: 0.0,
            burst_pos: Vec2::ZERO,
            kaleido_t: 0.0,
            style_t: 0.0,
            flow_t: 0.0,
            palette_hold: 0.0,
            palette_t: 0.0,
            prev_drop_age: 1e3,
            saved_kaleido: false,
            saved_folds: 6.0,
            saved_spin: 1.0,
            saved_trail: 0.3,
            last_cursor: None,
            rng: fastrand::Rng::with_seed(1),
        }
    }
}

impl Choreographer {
    /// Center-relative position of the synthetic left-click blast while one
    /// is firing: update_params overrides the real cursor with it.
    pub fn blast(&self) -> Option<Vec2> {
        (self.on && self.burst > 0.0).then_some(self.burst_pos)
    }

    /// True for the reseed boost after a scene cut, which refills the new
    /// boundary within a few frames so a cut never shows a half-empty cloud.
    pub fn refilling(&self) -> bool {
        self.on && self.reseed_boost > 0.0
    }
}

/// Random pick of a mode id different from `cur`, uniform over the rest.
fn rand_cycle(cur: u32, n: u32, rng: &mut fastrand::Rng) -> u32 {
    (cur + 1 + rng.u32(0..n - 1)) % n
}

/// Fresh mandala styling: random fold count and spin gain.
fn roll_kaleido_style(settings: &mut Settings, rng: &mut fastrand::Rng) {
    settings.kaleido_folds = (3.0 + rng.f32() * 11.0).round();
    settings.kaleido_spin = 0.2 + rng.f32() * 1.6;
}

/// Scan a coarse smooth-escape grid over the middle of the view and return
/// the fractal point with the highest local escape-time variance - the
/// busiest boundary in sight, which is where a dive keeps finding structure.
/// Picks randomly among the top cells so repeated dives take different turns.
/// Returns None over a flat field (deep interior, far exterior): nothing to
/// steer toward, keep the current heading.
fn pick_boundary_target(
    center: DVec2,
    height: f64,
    aspect: f64,
    max_iter: u32,
    fractal: Fractal,
    jc: (f64, f64),
    rng: &mut fastrand::Rng,
) -> Option<DdVec2> {
    const G: usize = 24;
    // Middle 84% of the view: targets picked at the very edge get anchored
    // there by the dive math and drag the interesting part half off screen.
    let cell = |ix: usize, iy: usize| {
        DVec2::new(
            center.x + ((ix as f64 + 0.5) / G as f64 - 0.5) * height * aspect * 0.84,
            center.y + ((iy as f64 + 0.5) / G as f64 - 0.5) * height * 0.84,
        )
    };
    let mut f = [[0.0f32; G]; G];
    // Row-parallel: at SCAN_ITER_CAP a serial scan costs a few ms, a visible
    // frame hitch mid-dive; across cores it stays sub-ms. (Wasm has no rayon;
    // single-threaded there, same as particle generation.)
    let fill_row = |iy: usize, row: &mut [f32; G]| {
        for (ix, v) in row.iter_mut().enumerate() {
            let p = cell(ix, iy);
            // Log-compressed so one deep escape spike cannot drown the
            // variance of everything around it.
            *v = (1.0 + smooth_iter(p.x, p.y, max_iter, fractal, jc)).ln();
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    let rows = f.par_iter_mut();
    #[cfg(target_arch = "wasm32")]
    let rows = f.iter_mut();
    rows.enumerate().for_each(|(iy, row)| fill_row(iy, row));
    let mut scored = Vec::with_capacity((G - 2) * (G - 2));
    for iy in 1..G - 1 {
        for ix in 1..G - 1 {
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for row in &f[iy - 1..=iy + 1] {
                for &v in &row[ix - 1..=ix + 1] {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            scored.push((hi - lo, ix, iy));
        }
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    if scored[0].0 < 0.05 {
        return None;
    }
    let top = scored
        .iter()
        .take_while(|s| s.0 > scored[0].0 * 0.6)
        .count()
        .min(6);
    let (_, ix, iy) = scored[rng.usize(0..top)];
    Some(DdVec2::from_dvec2(cell(ix, iy)))
}

/// The choreographer's per-frame tick. Runs after `handle_input` (so a frame
/// with input is seen before it acts) and before `update_params` (so the
/// view it writes is the one rendered).
pub fn update_choreographer(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut wheel: EventReader<MouseWheel>,
    windows: Query<&Window>,
    menu_open: Res<MenuOpen>,
    mut settings: ResMut<Settings>,
    mut fractal: ResMut<FractalType>,
    audio: Res<AudioLevels>,
    mut choreo: ResMut<Choreographer>,
    mut view: ResMut<ViewState>,
    mut kaleido: ResMut<Kaleido>,
    mut color_mode: ResMut<ColorMode>,
    mut flow_mode: ResMut<FlowMode>,
    mut auto_zoom: ResMut<AutoZoom>,
    mut fly: ResMut<FlyTo>,
) {
    let dt = time.delta_secs();
    let Ok(window) = windows.single() else {
        return;
    };
    let aspect = (window.width().max(1.0) / window.height().max(1.0)) as f64;

    // Explicit toggle: X key or the menu row. Manual engagement ignores
    // input, so a VJ can nudge the swarm while the autopilot drives.
    let toggled = choreo.want_toggle || keys.just_pressed(KeyCode::KeyX);
    choreo.want_toggle = false;

    // Any input: keys held (X excluded - it IS the toggle), buttons, wheel,
    // or the cursor moving (including entering/leaving the window).
    let cursor = window.cursor_position();
    let moved = match (cursor, choreo.last_cursor) {
        (Some(c), Some(p)) => c.distance(p) > 2.0,
        (a, b) => a.is_some() != b.is_some(),
    };
    choreo.last_cursor = cursor;
    let any_input = moved
        || keys.get_pressed().any(|k| *k != KeyCode::KeyX)
        || mouse.any_pressed([MouseButton::Left, MouseButton::Right, MouseButton::Middle])
        || wheel.read().next().is_some();
    // The toggle resets the idle clock too: X is excluded from any_input, so
    // an X exit after 30 s hands-off would otherwise re-engage (idle mode)
    // on the very next frame.
    if any_input || toggled {
        choreo.idle = 0.0;
    } else {
        choreo.idle += dt;
    }

    // Exits: the toggle always; any input only for idle engagement.
    if choreo.on && (toggled || (any_input && !choreo.manual)) {
        choreo.on = false;
        choreo.burst = 0.0;
        choreo.cut_wait = None;
        choreo.reseed_boost = 0.0;
        // Restore the kaleidoscope unless the waking input IS the user
        // toggling it - restoring would eat that press.
        if !keys.just_pressed(KeyCode::KeyK) {
            kaleido.on = choreo.saved_kaleido;
        }
        settings.kaleido_folds = choreo.saved_folds;
        settings.kaleido_spin = choreo.saved_spin;
        settings.trail = choreo.saved_trail;
        info!("auto-choreographer: off");
        return;
    }

    if !choreo.on {
        if !toggled && (choreo.idle < IDLE_SECS || menu_open.0) {
            return;
        }
        choreo.on = true;
        choreo.manual = toggled;
        choreo.saved_kaleido = kaleido.on;
        choreo.saved_folds = settings.kaleido_folds;
        choreo.saved_spin = settings.kaleido_spin;
        choreo.saved_trail = settings.trail;
        choreo.target = view.center;
        choreo.aim = view.center;
        choreo.retarget = 0.0;
        choreo.prev_drop_age = audio.drop_age;
        auto_zoom.0 = false;
        fly.0 = None;
        choreo.rng = fastrand::Rng::with_seed((time.elapsed_secs_f64() * 1e6) as u64 | 1);
        choreo.next_burst = 6.0 + choreo.rng.f32() * 8.0;
        choreo.kaleido_t = 10.0 + choreo.rng.f32() * 15.0;
        choreo.style_t = 8.0 + choreo.rng.f32() * 12.0;
        choreo.flow_t = 20.0 + choreo.rng.f32() * 20.0;
        choreo.palette_hold = 8.0;
        choreo.palette_t = 18.0 + choreo.rng.f32() * 17.0;
        info!(
            "auto-choreographer: on ({})",
            if toggled {
                "X / menu exits"
            } else {
                "any input exits"
            }
        );
    }

    // Blast bursts: a synthetic left-click somewhere in the middle of the
    // view, held ~0.25 s so the impulse moves real mass. Applied to the
    // mouse uniform in update_params.
    choreo.next_burst -= dt;
    choreo.burst = (choreo.burst - dt).max(0.0);
    choreo.reseed_boost = (choreo.reseed_boost - dt).max(0.0);
    if choreo.next_burst <= 0.0 {
        choreo.burst = 0.25;
        let rx = (choreo.rng.f32() - 0.5) * 0.7;
        let ry = (choreo.rng.f32() - 0.5) * 0.7;
        choreo.burst_pos = Vec2::new(
            (rx as f64 * view.height * aspect) as f32,
            (ry as f64 * view.height) as f32,
        );
        choreo.next_burst = 6.0 + choreo.rng.f32() * 8.0;
    }

    // Kaleidoscope phases: on for a stretch (its rotation already rides the
    // beat via the spin accumulator in update_params), then off again. Each
    // on-phase gets a fresh fold count and spin gain, so no two mandala
    // stretches look alike.
    choreo.kaleido_t -= dt;
    if choreo.kaleido_t <= 0.0 {
        kaleido.on = !kaleido.on;
        if kaleido.on {
            roll_kaleido_style(&mut settings, &mut choreo.rng);
            choreo.kaleido_t = 12.0 + choreo.rng.f32() * 18.0;
        } else {
            choreo.kaleido_t = 8.0 + choreo.rng.f32() * 14.0;
        }
    }

    // Styling tick: re-roll the look every so often - fold count and spin
    // mid-phase (a live mandala re-facets), and the trail length. Trails
    // mostly sit at the user's baseline or tighter; only sometimes stretch
    // into the long dreamy smear, so the smear stays a highlight instead of
    // the norm.
    choreo.style_t -= dt;
    if choreo.style_t <= 0.0 {
        if kaleido.on {
            roll_kaleido_style(&mut settings, &mut choreo.rng);
        }
        let base = choreo.saved_trail;
        let r = choreo.rng.f32();
        settings.trail = if r < 0.4 {
            base
        } else if r < 0.75 {
            base * (0.4 + 0.6 * choreo.rng.f32())
        } else {
            (base.max(0.55) + choreo.rng.f32() * 0.35).min(0.92)
        };
        choreo.style_t = 8.0 + choreo.rng.f32() * 12.0;
    }

    // Flow-mode cycle on its own clock; the 2 s crossfade melts each change.
    choreo.flow_t -= dt;
    if choreo.flow_t <= 0.0 {
        flow_mode.0 = rand_cycle(flow_mode.0, FLOW_MODES.len() as u32, &mut choreo.rng);
        choreo.flow_t = 20.0 + choreo.rng.f32() * 20.0;
    }

    // Palette swaps on song sections, approximated by detected drops (with a
    // minimum spacing so a bass barrage does not strobe the palette), or on
    // a timer when the music never drops / audio is off.
    let section = audio.drop_age < choreo.prev_drop_age;
    choreo.prev_drop_age = audio.drop_age;
    choreo.palette_hold -= dt;
    choreo.palette_t -= dt;
    if (section && choreo.palette_hold <= 0.0) || choreo.palette_t <= 0.0 {
        // Audio aurora (the last mode) sits dim silver in silence: skip it
        // when idle.
        let n = if audio.is_idle() {
            COLOR_MODES.len() as u32 - 1
        } else {
            COLOR_MODES.len() as u32
        };
        color_mode.0 = rand_cycle(color_mode.0, n, &mut choreo.rng);
        choreo.palette_hold = 8.0;
        choreo.palette_t = 18.0 + choreo.rng.f32() * 17.0;
    }

    if view.height <= DIVE_FLOOR * 1.05 {
        // Scene end (also catches engaging while parked deeper than the
        // targeting floor). Every scene ends in a hard cut to the next one -
        // there is no full zoom-out reset: half the time a different
        // fractal, half a fresh region of the current one.
        // Arm on floor arrival; with music playing, hang at the deepest
        // point a moment and land the cut ON the next beat.
        let wait = choreo
            .cut_wait
            .get_or_insert(if audio.is_idle() { 0.0 } else { 1.2 });
        *wait -= dt;
        if *wait <= 0.0 || audio.beat_age < 0.08 {
            choreo.cut_wait = None;
            // The cut: jump straight to a boundary spot partway zoomed in -
            // never the mostly-empty full view. Masked hard: the
            // kaleidoscope slams on with a fresh style (fold symmetry makes
            // the cut read as an intentional edit), long trails smear the
            // old shape out, a center blast pops, and a reseed boost
            // (update_params) refills the boundary within a few frames.
            if choreo.rng.f32() < 0.5 {
                let n = Fractal::ALL.len() as u32;
                fractal.0 = Fractal::from_id(rand_cycle(fractal.0.id(), n, &mut choreo.rng));
            }
            let (hc, hh) = fractal.0.home_view();
            let scan_iter = depth_iter(hh, settings.detail).min(SCAN_ITER_CAP);
            let jc = current_julia_c(fractal.0, &settings, &audio);
            let spot =
                pick_boundary_target(hc, hh, aspect, scan_iter, fractal.0, jc, &mut choreo.rng)
                    .unwrap_or_else(|| DdVec2::from_dvec2(hc));
            view.center = spot;
            view.height = hh * (0.2 + choreo.rng.f32() * 0.25) as f64;
            choreo.target = spot;
            choreo.aim = spot;
            choreo.retarget = 0.0;
            kaleido.on = true;
            roll_kaleido_style(&mut settings, &mut choreo.rng);
            choreo.kaleido_t = 12.0 + choreo.rng.f32() * 18.0;
            settings.trail = settings.trail.max(0.85);
            choreo.burst = 0.3;
            choreo.burst_pos = Vec2::ZERO;
            choreo.reseed_boost = 1.5;
        }
    } else {
        choreo.retarget -= dt;
        if choreo.retarget <= 0.0 {
            let max_iter = depth_iter(view.height, settings.detail).min(SCAN_ITER_CAP);
            let jc = current_julia_c(fractal.0, &settings, &audio);
            if let Some(t) = pick_boundary_target(
                view.center.to_dvec2(),
                view.height,
                aspect,
                max_iter,
                fractal.0,
                jc,
                &mut choreo.rng,
            ) {
                choreo.target = t;
            }
            choreo.retarget = 3.0;
        }
        // The aim eases toward the picked target (~1 s time constant),
        // so a retarget bends the dive path instead of yanking it.
        let chase = (choreo.target - choreo.aim).to_dvec2();
        choreo.aim += chase * (1.0 - (-1.2 * dt as f64).exp());
        let new_h = (view.height * (-0.45 * dt as f64).exp()).max(DIVE_FLOOR);
        // The extra decay recenters the aim over ~2 s on top of the anchor.
        approach_step(&mut view, choreo.aim, new_h, 1.0 - (-0.6 * dt as f64).exp());
    }
}
