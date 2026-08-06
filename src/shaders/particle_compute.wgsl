// Fractality particle compute shader.
// Struct layouts must match the Rust side (particles.rs) exactly.

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    home: vec2<f32>,
    band: f32,
    hue: f32,
};

struct Params {
    world_to_clip: vec4<f32>,
    mouse: vec4<f32>,
    particle_size: vec2<f32>,
    center_delta: vec2<f32>,
    time: f32,
    dt: f32,
    count: u32,
    max_iter: u32,
    flow_speed: f32,
    band_k: f32,
    damping: f32,
    brightness: f32,
    ref_len: u32,
    frame: u32,
    reseed_rate: f32,
    detail: f32,
    dissolve: f32,
    color_mode: u32,
    fractal_type: u32,
    // Flow style: 0 contour, 1 layers, 2 gravity, 3 erupt, 4 pulse,
    // 5 dynamics.
    flow_mode: u32,
    // Unused in this shader; present for layout parity with ParamsUniform.
    trail_decay: f32,
    // Audio levels: x bass, y mid, z treble, w beat pulse. All zero while
    // audio reactivity is off, so every use is a natural no-op.
    audio: vec4<f32>,
    // Music-driven hue/phase accumulator; the dynamics flow mode spins its
    // direction field by it.
    audio_hue: f32,
    // x seconds since last beat, y seconds since last drop (saturate high),
    // z unused, w overall level.
    audio2: vec4<f32>,
    // Effect gains: x ring pulse, y flash/glitter (render), z spectrum glow
    // (render), w unused.
    audio_fx: vec4<f32>,
    // 16 log-spaced spectrum bins (unused here; layout parity).
    spectrum: array<vec4<f32>, 4>,
    // Shape/dynamics tuning: x condensation (boundary freeze strength and
    // reach; 1 = classic), y previous flow mode, z flow crossfade blend
    // (0 = all previous mode, 1 = all current), w spare.
    shape: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;
// Reference orbit Z_0..Z_n at the view center, computed in double-double on
// the CPU. Each entry is an f32 hi/lo split per component: xy = hi, zw = lo
// (~48 bits of Z together). The lo limb exists for one reason: the full value
// z = Z + dz cancels catastrophically in f32 when the orbit close-approaches
// zero at deep zoom (close-approach size ~sqrt(view height) drops below hi's
// absolute rounding error past height ~1e-15). ref_add restores the digits.
@group(0) @binding(2) var<storage, read> ref_orbit: array<vec4<f32>>;

// Compensated z = Z_ref + dz. hi + dz first: when they nearly cancel the
// subtraction is exact (Sterbenz), and lo then supplies the surviving digits;
// when they don't cancel, lo is far below rounding and changes nothing.
fn ref_add(zr: vec4<f32>, dz: vec2<f32>) -> vec2<f32> {
    return (zr.xy + dz) + zr.zw;
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

// |x + d| - |x| without catastrophic cancellation (d may be ~1e-14 of x).
fn diffabs(x: f32, d: f32) -> f32 {
    if (x >= 0.0) {
        if (x + d >= 0.0) {
            return d;
        }
        return -(2.0 * x + d);
    }
    if (x + d <= 0.0) {
        return -d;
    }
    return 2.0 * x + d;
}

// One perturbation step of the selected fractal: dz_{n+1} from the reference
// value Z_n, the delta dz_n, and the point's offset dc from the view center.
// Types match FRACTAL_MODES on the Rust side.
fn step_dz(z_ref: vec2<f32>, dz: vec2<f32>, dc: vec2<f32>) -> vec2<f32> {
    switch params.fractal_type {
        case 1u: {
            // Burning Ship: z' = (|x| + i|y|)^2 + c. With w = |Z+dz| and
            // wr = |Z| (componentwise), dz' = w^2 - wr^2 = (w-wr)(w+wr);
            // diffabs gives w-wr exactly, avoiding the cancellation that
            // would erase a tiny delta at deep zoom.
            let wr = abs(z_ref);
            let d = vec2<f32>(diffabs(z_ref.x, dz.x), diffabs(z_ref.y, dz.y));
            return cmul(d, 2.0 * wr + d) + dc;
        }
        case 2u: {
            // Tricorn: z' = conj(z)^2 + c, and conj(z)^2 - conj(Z)^2
            // = conj(z^2 - Z^2), so conjugate the Mandelbrot delta.
            return conj(cmul(2.0 * z_ref + dz, dz)) + dc;
        }
        case 3u: {
            // Multibrot-3: (Z+dz)^3 - Z^3 = dz*(3Z^2 + 3Z*dz + dz^2).
            return cmul(dz, 3.0 * cmul(z_ref, z_ref) + 3.0 * cmul(z_ref, dz) + cmul(dz, dz)) + dc;
        }
        case 4u: {
            // Julia: same map as Mandelbrot but c is a global constant, so
            // there is no dc term; dc instead seeds dz_0 (offset of z_0).
            return cmul(2.0 * z_ref + dz, dz);
        }
        default: {
            // Mandelbrot: 2*Z*dz + dz^2 + dc.
            return cmul(2.0 * z_ref + dz, dz) + dc;
        }
    }
}

// d(z_{n+1})/dc iteration for the selected fractal, using the full z_n.
// Exact for the holomorphic maps (0, 3, 4); Burning Ship and Tricorn are not
// holomorphic, so their rules are the standard distance-estimate
// approximations - plenty for steering the flow field.
fn step_der(z_full: vec2<f32>, der: vec2<f32>) -> vec2<f32> {
    switch params.fractal_type {
        case 1u: {
            return 2.0 * cmul(abs(z_full), der) + vec2<f32>(1.0, 0.0);
        }
        case 2u: {
            return 2.0 * cmul(conj(z_full), conj(der)) + vec2<f32>(1.0, 0.0);
        }
        case 3u: {
            return 3.0 * cmul(cmul(z_full, z_full), der) + vec2<f32>(1.0, 0.0);
        }
        case 4u: {
            // dz_0/dc = 1 (dc perturbs z_0), no additive term after that.
            return 2.0 * cmul(z_full, der);
        }
        default: {
            return 2.0 * cmul(z_full, der) + vec2<f32>(1.0, 0.0);
        }
    }
}

// 1/log2(power) for the smooth-iteration formula; only Multibrot-3 (power 3)
// differs from 1. Must match inv_log2_power() on the CPU.
fn inv_log2_power() -> f32 {
    return select(1.0, 0.6309297535714574, params.fractal_type == 3u);
}

// Shell calm factor from a field value: 1 = free streaming, near 0 = frozen
// onto the boundary. Proximity is normalized to the detail-1 iteration ramp
// (max_iter scales with the detail knob, so dividing by max_iter directly
// made raising detail reclassify the frozen shell as outer flow - the whole
// image went fuzzy at high detail). Condensation stretches the proximity and
// deepens the freeze floor: 1 = classic feel, higher = wider + deader
// freeze (crisper edges), 0 = everything streams.
fn calm_of(f: f32) -> f32 {
    let cond = params.shape.x;
    let depth = f / f32(params.max_iter) * params.detail;
    // Freeze depth: residual shell speed is 0.05/cond^4 of flow speed, so
    // every extra slider unit keeps deadening the shell instead of
    // saturating (1 -> 5%, 2 -> 0.3%, 5 -> 0.008%, solid). cond = 0 kills
    // the divide via the max() and freezes nothing.
    let freeze = clamp(1.0 - 0.05 / max(cond * cond * cond * cond, 1e-6), 0.0, 0.9999);
    return 1.0 - freeze * smoothstep(0.25, 0.75, depth * cond);
}

// Initial delta: zero except Julia, where dc perturbs the starting point.
fn initial_dz(dc: vec2<f32>) -> vec2<f32> {
    if (params.fractal_type == 4u) {
        return dc;
    }
    return vec2<f32>(0.0, 0.0);
}

// Smooth escape-time field via perturbation. dc is the point's offset from the
// view center (small, so f32 keeps full relative precision no matter how deep
// the zoom). Iterates the delta dz around the reference orbit and rebases
// (Zhuoran's method) when the delta outgrows the reference, which also handles
// a short/diverged reference orbit.
fn field(dc: vec2<f32>) -> f32 {
    // Z_0 is known up front (0 for c-plane fractals, the view center for
    // Julia), so each iteration needs only one ref_orbit load (the next
    // entry); the previous load is carried in a register across iterations.
    // Only the full-value reconstruction (ref_add) and the rebase fold use
    // the lo limb; step_dz products are cancellation-free, so hi is enough.
    let ref0 = ref_orbit[0];
    var dz = initial_dz(dc);
    var ri: u32 = 0u;
    var z_ref = ref0;
    let last = params.ref_len - 1u;
    for (var n: u32 = 0u; n < params.max_iter; n = n + 1u) {
        dz = step_dz(z_ref.xy, dz, dc);
        ri = ri + 1u;
        let z_ref_next = ref_orbit[ri];
        let z = ref_add(z_ref_next, dz); // full z_{n+1}
        let m = dot(z, z);
        if (m > 256.0) {
            return f32(n) + 1.0 - log2(0.5 * log2(m)) * inv_log2_power();
        }
        // Rebase: fold the full value into the delta and restart the reference
        // when the delta dominates or the reference orbit is exhausted.
        if (m < dot(dz, dz) || ri >= last) {
            dz = (z - ref0.xy) - ref0.zw;
            ri = 0u;
            z_ref = ref0;
        } else {
            z_ref = z_ref_next;
        }
    }
    return f32(params.max_iter);
}

struct FieldGrad {
    f: f32,
    grad: vec2<f32>,
    // Full z at escape (zero for interior points). Its argument is the
    // escape angle: repeated squaring doubles it each iteration, so it
    // decorrelates at filament scale no matter how deep the zoom - the
    // dynamics flow mode steers by it.
    zesc: vec2<f32>,
};

// ln(2)^2, chain-rule constant for the smooth-field gradient.
const LN2_SQ: f32 = 0.4804530139182014;

// field() plus its exact spatial gradient in one perturbation loop. The
// derivative of the full orbit w.r.t. c is iterated alongside the delta
// (der_{n+1} = 2*z_n*der_n + 1, using the full z, so rebasing does not affect
// it). At escape the gradient of f = n + 1 - log2(0.5*log2(|z|^2)) follows by
// chain rule: grad f = -z*conj(der) / (|z|^2 * 0.5*log2(|z|^2) * ln(2)^2).
// One analytic loop replaces three finite-difference field() calls and has no
// step-size (eps) tuning, staying exact at any zoom depth.
fn field_grad(dc: vec2<f32>) -> FieldGrad {
    let ref0 = ref_orbit[0];
    var dz = initial_dz(dc);
    // Julia: dc perturbs z_0, so d(z_0)/dc = 1; c-plane fractals start at 0.
    var der = vec2<f32>(select(0.0, 1.0, params.fractal_type == 4u), 0.0);
    var ri: u32 = 0u;
    var z_ref = ref0;
    let last = params.ref_len - 1u;
    let ilp = inv_log2_power();
    for (var n: u32 = 0u; n < params.max_iter; n = n + 1u) {
        let z_full = ref_add(z_ref, dz); // full z_n
        der = step_der(z_full, der);
        dz = step_dz(z_ref.xy, dz, dc);
        ri = ri + 1u;
        let z_ref_next = ref_orbit[ri];
        let z = ref_add(z_ref_next, dz);
        let m = dot(z, z);
        if (m > 256.0) {
            let u = 0.5 * log2(m);
            let g = -cmul(z, conj(der)) / (m * u * LN2_SQ) * ilp;
            return FieldGrad(f32(n) + 1.0 - log2(u) * ilp, g, z);
        }
        if (m < dot(dz, dz) || ri >= last) {
            dz = (z - ref0.xy) - ref0.zw;
            ri = 0u;
            z_ref = ref0;
        } else {
            z_ref = z_ref_next;
        }
    }
    return FieldGrad(f32(params.max_iter), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0));
}

// Ring waves die past these ages; one definition keeps beat_impulse and the
// staggered path's liveness check in sync.
const BEAT_AGE_MAX: f32 = 1.2;
const DROP_AGE_MAX: f32 = 2.0;
// Wavefront travel speed, in view heights per second.
const WAVE_SPEED: f32 = 1.4;

fn wave_live() -> bool {
    return params.audio2.x < BEAT_AGE_MAX || params.audio2.y < DROP_AGE_MAX;
}

// A gaussian shell at the radius an `age`-old wavefront has reached, fading
// as it travels. `rn` is the radius in view-height units (see beat_impulse).
fn ring(rn: f32, age: f32) -> f32 {
    let d = (rn - age * WAVE_SPEED) / 0.08;
    return exp(-d * d) * exp(-age * 2.5);
}

// Beat/drop shockwaves: expanding rings from the view center - a pond ripple
// traveling through the cloud, not a uniform push. A drop fires a triple
// wave. Scales with view_height so it feels identical at any zoom depth, and
// is cheap enough (length + a few exp) to run on the staggered fast path so
// the wavefront hits every particle every frame. Amplitudes are sized to
// displace, not eject: much higher and the cloud blows off screen and the
// view goes dark until the recycle repopulates it.
fn beat_impulse(pos: vec2<f32>, view_height: f32) -> vec2<f32> {
    let beat_age = params.audio2.x;
    let drop_age = params.audio2.y;
    // View-height units: length() of a raw center-relative position squares
    // it internally, which underflows f32 below height ~1e-19 and collapses
    // the radius (and the direction division) to garbage at deep zoom.
    let pn = pos / view_height;
    let rn = length(pn);
    var dir = vec2<f32>(0.0, 1.0);
    if (rn > 1e-5) {
        dir = pn / rn;
    }
    var a = 0.0;
    if (beat_age < BEAT_AGE_MAX) {
        a += ring(rn, beat_age) * 5.0;
    }
    if (drop_age < DROP_AGE_MAX) {
        // Triple wave, 0.18 s apart, each softer than the last.
        for (var k = 0u; k < 3u; k = k + 1u) {
            let ag = drop_age - 0.18 * f32(k);
            if (ag >= 0.0) {
                a += ring(rn, ag) * (9.0 - 2.0 * f32(k));
            }
        }
    }
    return dir * a * view_height * params.audio_fx.x;
}

fn hash_u32(x0: u32) -> u32 {
    var h = x0;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

fn rand01(state: ptr<function, u32>) -> f32 {
    *state = hash_u32(*state);
    return f32(*state) * (1.0 / 4294967296.0);
}

// Band-pull weight per flow mode. Gravity/erupt/dynamics are free-flowing:
// particles cross contours by design, so the band pull (and its
// normal-velocity bleed, which would project all motion onto the tangent)
// must not fight them. The recycle trickle respawns drifters back onto the
// boundary. A weight rather than a bool so a mode crossfade can ramp the
// pull in or out instead of snapping it.
fn pull_of(mode: u32) -> f32 {
    return select(1.0, 0.0, mode == 2u || mode == 3u || mode == 5u);
}

// Desired flow velocity for one flow mode. Split out of the update loop so a
// mode switch can evaluate BOTH the outgoing and incoming fields and
// crossfade them (shape.z ramps 0 to 1) instead of hard-cutting.
fn flow_desired(
    mode: u32,
    gn: vec2<f32>,
    tangent: vec2<f32>,
    band: f32,
    settle: f32,
    calm: f32,
    base: f32,
    correct: vec2<f32>,
    zesc: vec2<f32>,
) -> vec2<f32> {
    var desired: vec2<f32>;
    switch mode {
        case 1u: {
            // Layers: adjacent iso-bands stream in opposite directions,
            // so neighboring contour ribbons shear past each other.
            let s = select(1.0, -1.0, fract(band * 0.25) < 0.5);
            desired = tangent * s * settle * calm * base + correct;
        }
        case 2u: {
            // Gravity: everything rains inward along the gradient and
            // piles up on the shell (calm freezes it there). The band
            // pull is disabled below so nothing fights the fall.
            desired = gn * settle * calm * base * 2.0;
        }
        case 3u: {
            // Erupt: particles boil off the set outward; recycling
            // reseeds the boundary so the fountain never runs dry.
            desired = -gn * mix(0.3, 1.0, settle) * base * 2.0;
        }
        case 5u: {
            // Dynamics: steer by the particle's own orbit escape angle.
            // Iterated squaring doubles the angle every step, so
            // arg(z_escape) decorrelates at the finest visible filament
            // scale at ANY zoom depth (the binary-decomposition cells of
            // the fractal), unlike a fixed map field which flattens to a
            // constant direction once the view is tiny next to C. The
            // slow time drift keeps the whole pattern churning. Free
            // flow (no band pull): particles ride the direction cells
            // across contours like wind, and the recycle trickle keeps
            // repainting the boundary behind them. Calm only softens
            // (not freezes) on the shell so the streams stay alive.
            let ze = zesc;
            let m = dot(ze, ze);
            if (m > 1e-12) {
                let a0 = atan2(ze.y, ze.x);
                // u is the fractional-iteration coordinate (escape
                // overshoot) - it varies ACROSS bands, so any monotonic
                // use of it paints band-parallel stripes. Keep it only
                // as a gentle wave phase.
                let u = 0.5 * log2(max(m, 1.0));
                // Ray angle tripled: the external-ray pinwheel cells
                // run perpendicular to the iteration bands, so scaling
                // a0 (not u) makes transverse structure dominate and
                // kills the layered look. The sine waves each stream
                // serpentine over time, phased by both coordinates so
                // neighboring eddies desync. audio_hue spins the whole
                // field with music energy, and each detected beat adds
                // a sharp extra twist that decays with the beat pulse
                // (both exactly zero when audio is off).
                let av = a0 * 3.0
                    + 0.5 * sin(u * 2.4 + a0 + params.time * 0.7)
                    + params.time * 0.25 + params.audio_hue
                    + params.audio.w * 1.2;
                // Geometry-anchored wind: rotate the LOCAL contour
                // normal by the cell angle instead of using a fixed
                // screen direction. gn turns with the fractal's shape,
                // so streams curl around filaments and spiral into
                // bulbs rather than blowing in straight lines across
                // them. gn is unit length, so the rotation is too.
                let dir = vec2<f32>(
                    gn.x * cos(av) - gn.y * sin(av),
                    gn.x * sin(av) + gn.y * cos(av),
                );
                // Speed lanes: gentle per-cell magnitude variation so
                // the wind has gusts instead of one uniform pace. The
                // calm floor stays high - a low floor made deep bands
                // crawl at 1/3 speed, itself a layering artifact - and
                // settle gets a floor for the same reason (raw settle
                // tracks band error, another band-parallel signal).
                let gust = 0.75 + 0.25 * sin(a0 * 5.0 + u - params.time * 0.5);
                desired = dir * mix(0.6, 1.0, settle)
                    * mix(0.55, 1.0, calm) * base * 1.5 * gust;
            } else {
                desired = correct;
            }
        }
        default: {
            // Contour (classic): advect along iso-lines.
            desired = tangent * settle * calm * base + correct;
        }
    }
    return desired;
}

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    var p = particles[idx];
    let dt = params.dt;

    // Rebase into the current view center's frame (center moved since last frame).
    p.pos += params.center_delta;
    p.home += params.center_delta;

    // Half extents of the visible region in center-relative units.
    let view_height = 2.0 / params.world_to_clip.y;
    let aspect = params.world_to_clip.y / params.world_to_clip.x;
    let ext = vec2<f32>(view_height * aspect * 0.5, view_height * 0.5);

    // Recycle particles to keep the current view populated as it moves. Two
    // triggers: (1) drifted well outside the view - fills fresh territory when
    // zooming IN; (2) a small random per-frame trickle - continuously resamples
    // the whole cloud into the current view, which is what fills the growing
    // margins when zooming OUT (drifted-out never triggers on zoom-out). Over a
    // few seconds the trickle refreshes the entire set at the current scale.
    var seed = hash_u32(idx ^ (params.frame * 2654435761u));
    let out_of_view = abs(p.pos.x) > ext.x * 2.5 || abs(p.pos.y) > ext.y * 2.5;
    let trickle = rand01(&seed) < params.reseed_rate;
    if (out_of_view || trickle) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            // Feathered sampling extent: mostly ~screen size (flat, uniform
            // density across the whole visible rect incl. corners), with a soft
            // tail past the screen edge. r*r concentrates the margin near 1.05
            // so the taper (and its front) sits off-screen - no visible box.
            let r = rand01(&seed);
            let mgn = 1.05 + 0.55 * r * r;
            let cand = vec2<f32>(
                (rand01(&seed) * 2.0 - 1.0) * ext.x * mgn,
                (rand01(&seed) * 2.0 - 1.0) * ext.y * mgn,
            );
            let f = field(cand);
            // Boundary bias: like the CPU seed, prefer candidates near the set
            // (high f). Without this, recycle accepts the whole exterior evenly
            // and the boundary detail washes out to a uniform haze over a few
            // seconds. t*t*t pushes the accept threshold up toward max_iter.
            // Boundary tightness scales with the detail knob: exponent 4 at
            // detail=1, higher pushes the accept threshold up toward max_iter so
            // particles cluster into a thinner, finer boundary shell.
            let t = rand01(&seed);
            let bias = pow(t, 4.0 * max(params.detail, 0.1));
            let threshold = 3.0 + (f32(params.max_iter) - 12.0) * bias;
            if (f > 3.0 && f >= threshold && f < f32(params.max_iter) - 1.0) {
                p.home = cand;
                p.pos = cand;
                p.band = f;
                p.hue = fract(f * 0.045 + 0.62);
                p.vel = vec2<f32>(0.0, 0.0);
                // Skip flow/mouse this frame: on a spike (fast zoom-out) this
                // avoids the field_grad() call on every respawned particle,
                // keeping the frame's total field work bounded.
                particles[idx] = p;
                return;
            }
        }
    }

    // Mouse geometry, needed both by the stagger gate and the impulse below.
    // Everything is measured in view-height units: length() of a raw
    // center-relative offset squares it internally, which underflows f32
    // below height ~1e-19 - dist collapses to the floor, every particle
    // counts as "inside the radius", and dir = d/dist comes out orders of
    // magnitude longer than unit, detonating the cloud on any click. The
    // distance floor (division guard for dir) stays relative to the radius:
    // any absolute floor eventually exceeds the shrinking radius and would
    // silently kill the blast/vortex at deep zoom.
    let radius = params.mouse.w;
    let dn = (p.pos - params.mouse.xy) / view_height;
    let rn = radius / view_height;
    let distn = max(length(dn), rn * 1e-4);

    // Stagger: shell particles (calm ~ 0) are deliberately near-frozen, yet
    // still pay the full field cost every frame. For settled particles band
    // tracks the local field value, so it predicts calm without computing the
    // field; skip 3 of 4 frames for those. Recycle checks above still run
    // every frame, and mouse-adjacent particles always update so interactions
    // stay responsive. NaN pos/band fails these compares and falls through to
    // the full path, where the NaN guard at the end catches it.
    let calm_est = calm_of(p.band);
    if (calm_est < 0.2 && distn > rn * 2.0 && (idx + params.frame) % 4u != 0u) {
        // Staggered particles still ride a live ring wave (impulse plus
        // integration, a few ALU ops), so the wavefront stays smooth without
        // paying the full field cost on beat frames. Damp here too: the full
        // path only runs 1 frame in 4 for these particles, and integrating
        // velocity every beat frame with quarter-rate damping lets the kick
        // run away and carry the cloud off screen.
        if (wave_live()) {
            p.vel += beat_impulse(p.pos, view_height) * dt;
            p.vel *= max(0.0, 1.0 - params.damping * dt);
            p.pos += p.vel * dt;
        }
        particles[idx] = p;
        return;
    }

    // Field value and exact gradient from one fused perturbation loop.
    let fg = field_grad(p.pos);
    let f0 = fg.f;
    let grad = fg.grad;
    let gl = length(grad);

    // Advect along iso-contours, correcting back toward the home band. The
    // gentle velocity-space correction (fixed gain, clamped) is what keeps the
    // cloud crisply on the boundary; the align knob adds a separate positional
    // pull below for the stream-convergence look.
    var desired: vec2<f32>;
    var gn = vec2<f32>(0.0, 0.0);
    var to_band = 0.0;
    var on_contour = false;
    var calm = 1.0;
    // Flow-mode crossfade: on a switch, shape.y holds the outgoing mode and
    // shape.z ramps 0 to 1 over ~2s. Smoothstep eases both endpoints. All
    // mode-dependent behavior blends by this weight: the desired field, the
    // pulse band wave, and the band-pull strength (pull_w, replacing the old
    // free_flow bool - see pull_of).
    let prev_mode = u32(params.shape.y + 0.5);
    let blend = smoothstep(0.0, 1.0, params.shape.z);
    let morphing = blend < 1.0 && prev_mode != params.flow_mode;
    var pull_w = pull_of(params.flow_mode);
    if (morphing) {
        pull_w = mix(pull_of(prev_mode), pull_w, blend);
    }
    // Upper bound rejects the rare overflowed derivative (inf gl) so gn stays
    // finite; NaN gl fails the compare and also falls through to the home pull.
    if (gl > 1e-4 && gl < 1e30 && f0 < f32(params.max_iter) - 0.5) {
        gn = grad / gl;
        let tangent = vec2<f32>(-gn.y, gn.x);
        // Pulse mode: the home band itself breathes, a slow wave phased by
        // band depth so contours expand/contract in traveling ripples. The
        // shifted band feeds both the tangential error and the positional
        // pull below, so the whole cloud follows the wave coherently.
        var band = p.band;
        var pulse_w = select(0.0, 1.0, params.flow_mode == 4u);
        if (morphing) {
            pulse_w = mix(select(0.0, 1.0, prev_mode == 4u), pulse_w, blend);
        }
        if (pulse_w > 0.0) {
            band += sin(params.time * 1.1 + p.band * 1.7) * 1.5 * pulse_w;
        }
        let band_err = band - f0;
        let err = clamp(band_err * 0.7, -2.0, 2.0);
        // Settle factor: a large band error means the local field varies
        // violently (deep boundary filaments), where the tangent direction is
        // noise. Streaming there just thrashes; scale the tangential flow down
        // so those particles slow, lock onto their contour, and trace the fine
        // shape instead of smearing it. Well-seated particles keep full flow.
        let settle = 1.0 / (1.0 + 6.0 * band_err * band_err);
        // Depth calm: f0 near the ramp top means the particle sits on the
        // fractal shell itself, where the shape lives. Freeze those almost
        // completely so the boundary renders as a stable filigree; the calm
        // ramps in from mid-field, leaving the outer haze streaming normally.
        // Detail-invariant and condensation-scaled - see calm_of.
        calm = calm_of(f0);
        // The normal velocity correction also rides the noisy gradient, so on
        // the shell it mostly injects thrash; the capped positional pull below
        // holds those particles instead. Keep it strong only for outer flow.
        let base = params.flow_speed * view_height;
        let correct = gn * err * mix(0.25, 1.0, calm) * base;
        desired = flow_desired(params.flow_mode, gn, tangent, band,
            settle, calm, base, correct, fg.zesc);
        if (morphing) {
            desired = mix(
                flow_desired(prev_mode, gn, tangent, band,
                    settle, calm, base, correct, fg.zesc),
                desired, blend);
        }
        // Spatial distance from the particle to its home contour along the
        // normal: (band - f0) is the field-unit error, /gl converts to distance.
        if (pull_w > 0.0) {
            to_band = clamp((band - f0) / gl, -view_height, view_height);
        }
        on_contour = true;
    } else {
        desired = (p.home - p.pos) * 0.6;
    }

    // Dissolve mode (Space): the flow field goes inert - no tangential
    // stream, no home pull - so damping bleeds particles to rest and only
    // mouse impulses and the diffusion below move them.
    if (params.dissolve > 0.5) {
        desired = vec2<f32>(0.0, 0.0);
    }

    // Mouse interaction: hover ripple, left blast, right vortex. Normalized
    // units for the geometry (dir is unit length by construction); the raw
    // radius still sets the impulse magnitude so strength tracks the view.
    var impulse = vec2<f32>(0.0, 0.0);
    if (distn < rn) {
        let fall = 1.0 - distn / rn;
        let dir = dn / distn;
        if (params.mouse.z > 0.5) {
            impulse = dir * fall * fall * radius * 240.0;
        } else if (params.mouse.z < -0.5) {
            let perp = vec2<f32>(-dir.y, dir.x);
            impulse = (perp * 2.5 - dir * 2.0) * fall * radius * 56.0;
        } else {
            impulse = dir * fall * fall * radius * 5.0;
        }
    }

    // Audio reactivity: ring waves plus a fine treble sparkle jitter.
    if (wave_live()) {
        impulse += beat_impulse(p.pos, view_height);
    }
    let tr = params.audio.z;
    if (tr > 0.02) {
        let a = rand01(&seed) * 6.2831853;
        let amp = tr * tr * view_height * 0.004 * rand01(&seed);
        p.pos += vec2<f32>(cos(a), sin(a)) * amp;
    }

    // Calm particles converge onto their (slow) desired velocity much faster,
    // shedding leftover chaotic velocity instead of coasting on it.
    let damp = params.damping * (1.0 + 3.0 * (1.0 - calm));
    p.vel += (desired - p.vel) * clamp(damp * dt, 0.0, 1.0);
    p.vel += impulse * dt;
    p.pos += p.vel * dt;

    // Band alignment: pull the particle toward its home contour as a positional
    // move. band_k is the closing rate per second; clamp to [0,1] so it snaps at
    // most exactly onto the contour (no overshoot).
    //
    // The stream effect is a diffusion/pull ratchet: normal jitter kicks the
    // particle off its contour, the pull snaps it back, and the asymmetry of the
    // field walks particles along the boundary into convergence points. The
    // jitter amplitude scales with the pull strength (via alpha), so align = 0
    // is the perfectly crisp static boundary, and turning it up buys stronger
    // streams while the matching pull keeps the fuzz bounded to ~jitter size.
    // The branch's mistake was a large fixed jitter (0.008) with no such link.
    if (on_contour && params.dissolve > 0.5) {
        // Dissolve: the pull is inert; diffusion alone along the gradient
        // normal gives the slow melt off the contours.
        let jitter = (rand01(&seed) * 2.0 - 1.0) * view_height * 0.008;
        p.pos += gn * jitter;
    } else if (on_contour && dt > 0.0 && pull_w > 0.0) {
        // pull_w scales alpha so a crossfade into or out of a free-flow mode
        // ramps the pull (step, jitter, and normal-velocity bleed) smoothly.
        let alpha = clamp(params.band_k * dt, 0.0, 1.0) * pull_w;
        // Cap the per-frame pull step to a small fraction of the view. The
        // field varies violently near the boundary, so even the exact
        // local gradient changes direction step to step; an uncapped snap
        // teleports along a soon-stale direction and the particle
        // ping-pongs harder the higher the align force. A capped step
        // converges over a few frames instead, which reads as a stable
        // sharpening of the shape.
        // Shell particles get a much tighter cap: the field is wildest
        // there, so at high align force a 2% hop per frame reads as
        // boiling. Small steps converge just as surely, only smoother.
        let max_step = view_height * mix(0.003, 0.02, calm);
        let step = clamp(to_band * alpha, -max_step, max_step);
        // Jitter scales with calm too: shell particles get almost none,
        // so the diffusion ratchet only churns the outer flow.
        let amp = view_height * 0.0015 * min(alpha * 2.0, 1.0) * calm;
        let jitter = (rand01(&seed) * 2.0 - 1.0) * amp;
        p.pos += gn * (step + jitter);
        // Bleed off the velocity component along the normal: the pull
        // corrects position but the old velocity keeps pushing off the
        // contour, and the two fighting shows up as vibration. Removing
        // normal velocity keeps the tangential stream intact.
        p.vel -= gn * dot(p.vel, gn) * alpha;
    }

    // Safety: respawn NaN particles at home. The negated <= form is false for
    // NaN (every ordered compare with NaN is false), so NaN positions reset
    // without a self-compare the optimizer may fold away. Bound is view-relative
    // now that positions are center-relative.
    let bound = max(ext.x, ext.y) * 8.0;
    if (!(abs(p.pos.x) <= bound && abs(p.pos.y) <= bound)) {
        p.pos = p.home;
        p.vel = vec2<f32>(0.0, 0.0);
    }

    particles[idx] = p;
}
