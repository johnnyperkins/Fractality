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
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;
// Reference orbit Z_0..Z_n at the view center, computed in f64 on the CPU.
@group(0) @binding(2) var<storage, read> ref_orbit: array<vec2<f32>>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Smooth escape-time field via perturbation. dc is the point's offset from the
// view center (small, so f32 keeps full relative precision no matter how deep
// the zoom). Iterates the delta dz around the reference orbit and rebases
// (Zhuoran's method) when the delta outgrows the reference, which also handles
// a short/diverged reference orbit.
fn field(dc: vec2<f32>) -> f32 {
    var dz = vec2<f32>(0.0, 0.0);
    var ri: u32 = 0u;
    let last = params.ref_len - 1u;
    for (var n: u32 = 0u; n < params.max_iter; n = n + 1u) {
        let z_ref = ref_orbit[ri];
        // dz_{n+1} = 2*Z_n*dz + dz^2 + dc
        dz = 2.0 * cmul(z_ref, dz) + cmul(dz, dz) + dc;
        ri = ri + 1u;
        let z = ref_orbit[ri] + dz; // full z_{n+1}
        let m = dot(z, z);
        if (m > 256.0) {
            return f32(n) + 1.0 - log2(0.5 * log2(m));
        }
        // Rebase: fold the full value into the delta and restart the reference
        // when the delta dominates or the reference orbit is exhausted.
        if (m < dot(dz, dz) || ri >= last) {
            dz = z;
            ri = 0u;
        }
    }
    return f32(params.max_iter);
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
                // avoids the 3 gradient field() calls on every respawned
                // particle, keeping the frame's total field() work bounded.
                particles[idx] = p;
                return;
            }
        }
    }

    // Gradient of the field by forward differences. Scale the sample step with
    // the current view height so the gradient stays accurate when zoomed deep.
    // Step scales with the view so the finite-difference delta stays ~constant
    // in field units at any zoom (no fixed floor, which would over-sample deep).
    let eps = clamp(view_height * 0.0008, 1e-30, 0.004);
    let f0 = field(p.pos);
    let fx = field(p.pos + vec2<f32>(eps, 0.0));
    let fy = field(p.pos + vec2<f32>(0.0, eps));
    let grad = vec2<f32>(fx - f0, fy - f0) / eps;
    let gl = length(grad);

    // Advect along iso-contours, correcting back toward the home band. The
    // gentle velocity-space correction (fixed gain, clamped) is what keeps the
    // cloud crisply on the boundary; the align knob adds a separate positional
    // pull below for the stream-convergence look.
    var desired: vec2<f32>;
    var gn = vec2<f32>(0.0, 0.0);
    var to_band = 0.0;
    var on_contour = false;
    if (gl > 1e-4 && f0 < f32(params.max_iter) - 0.5) {
        gn = grad / gl;
        let tangent = vec2<f32>(-gn.y, gn.x);
        let band_err = p.band - f0;
        let err = clamp(band_err * 0.7, -2.0, 2.0);
        // Settle factor: a large band error means the local field varies
        // violently (deep boundary filaments), where the tangent direction is
        // noise. Streaming there just thrashes; scale the tangential flow down
        // so those particles slow, lock onto their contour, and trace the fine
        // shape instead of smearing it. Well-seated particles keep full flow.
        let settle = 1.0 / (1.0 + band_err * band_err);
        desired = (tangent * settle + gn * err) * params.flow_speed * view_height;
        // Spatial distance from the particle to its home contour along the
        // normal: (band - f0) is the field-unit error, /gl converts to distance.
        to_band = clamp((p.band - f0) / gl, -view_height, view_height);
        on_contour = true;
    } else {
        desired = (p.home - p.pos) * 0.6;
    }

    // Mouse interaction: hover ripple, left blast, right vortex.
    var impulse = vec2<f32>(0.0, 0.0);
    let d = p.pos - params.mouse.xy;
    let dist = max(length(d), 1e-7);
    let radius = params.mouse.w;
    if (dist < radius) {
        let fall = 1.0 - dist / radius;
        let dir = d / dist;
        if (params.mouse.z > 0.5) {
            impulse = dir * fall * fall * radius * 60.0;
        } else if (params.mouse.z < -0.5) {
            let perp = vec2<f32>(-dir.y, dir.x);
            impulse = (perp * 2.5 - dir * 2.0) * fall * radius * 14.0;
        } else {
            impulse = dir * fall * fall * radius * 5.0;
        }
    }

    p.vel += (desired - p.vel) * clamp(params.damping * dt, 0.0, 1.0);
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
    if (on_contour) {
        if (dt > 0.0) {
            let alpha = clamp(params.band_k * dt, 0.0, 1.0);
            // Cap the per-frame pull step to a small fraction of the view. The
            // gradient is a noisy single-sample estimate near the boundary, so
            // an uncapped snap teleports along a wrong direction and the
            // particle ping-pongs harder the higher the align force. A capped
            // step converges over a few frames instead, which reads as a
            // stable sharpening of the shape.
            let max_step = view_height * 0.02;
            let step = clamp(to_band * alpha, -max_step, max_step);
            let amp = view_height * 0.0015 * min(alpha * 2.0, 1.0);
            let jitter = (rand01(&seed) * 2.0 - 1.0) * amp;
            p.pos += gn * (step + jitter);
            // Bleed off the velocity component along the normal: the pull
            // corrects position but the old velocity keeps pushing off the
            // contour, and the two fighting shows up as vibration. Removing
            // normal velocity keeps the tangential stream intact.
            p.vel -= gn * dot(p.vel, gn) * alpha;
        } else {
            // Paused: pull is inert; run the diffusion alone for the slow
            // dissolve effect on Space.
            let jitter = (rand01(&seed) * 2.0 - 1.0) * view_height * 0.008;
            p.pos += gn * jitter;
        }
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
