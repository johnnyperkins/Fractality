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
    time: f32,
    dt: f32,
    count: u32,
    max_iter: u32,
    flow_speed: f32,
    band_k: f32,
    damping: f32,
    brightness: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

// Smooth escape-time field. Formula and escape radius (256.0) are identical
// to smooth_iter() on the CPU so band values match.
fn field(c: vec2<f32>) -> f32 {
    var z = vec2<f32>(0.0, 0.0);
    for (var i: u32 = 0u; i < params.max_iter; i = i + 1u) {
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        let m = dot(z, z);
        if (m > 256.0) {
            return f32(i) + 1.0 - log2(0.5 * log2(m));
        }
    }
    return f32(params.max_iter);
}

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    var p = particles[idx];
    let dt = params.dt;

    // Gradient of the field by forward differences. Scale the sample step with
    // the current view height (2.0 / world_to_clip.y) so the gradient stays
    // accurate when zoomed deep into the boundary.
    let view_height = 2.0 / params.world_to_clip.y;
    let eps = clamp(view_height * 0.0008, 1e-5, 0.004);
    let f0 = field(p.pos);
    let fx = field(p.pos + vec2<f32>(eps, 0.0));
    let fy = field(p.pos + vec2<f32>(0.0, eps));
    let grad = vec2<f32>(fx - f0, fy - f0) / eps;
    let gl = length(grad);

    // Advect along iso-contours, correcting back toward the home band.
    var desired: vec2<f32>;
    if (gl > 1e-4 && f0 < f32(params.max_iter) - 0.5) {
        let gn = grad / gl;
        let tangent = vec2<f32>(-gn.y, gn.x);
        let err = clamp((p.band - f0) * params.band_k, -2.0, 2.0);
        desired = (tangent + gn * err) * params.flow_speed;
    } else {
        desired = (p.home - p.pos) * 0.6;
    }

    // Mouse interaction: hover ripple, left blast, right vortex.
    var impulse = vec2<f32>(0.0, 0.0);
    let d = p.pos - params.mouse.xy;
    let dist = max(length(d), 1e-5);
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

    // Safety: respawn escaped or NaN particles at home. The negated <= form is
    // false for NaN (every ordered compare with NaN is false), so NaN positions
    // trigger the reset without a self-compare the optimizer may fold away.
    if (!(abs(p.pos.x) <= 4.0 && abs(p.pos.y) <= 4.0)) {
        p.pos = p.home;
        p.vel = vec2<f32>(0.0, 0.0);
    }

    particles[idx] = p;
}
