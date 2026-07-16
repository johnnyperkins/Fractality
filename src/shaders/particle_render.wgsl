// Fractality particle render shader.
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
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VsOut {
    // Quad corner from the vertex index without a runtime-indexed array
    // (which naga lowers to per-vertex private memory). Vertices 0..5 map to
    // corner ids 0,1,2 / 2,3,0 - two triangles covering the quad - and corner
    // id c walks (-1,-1),(1,-1),(1,1),(-1,1) counterclockwise. Culling is off,
    // so winding is irrelevant.
    let c = (vertex_index % 3u + (vertex_index / 3u) * 2u) % 4u;
    let corner = vec2<f32>(
        select(-1.0, 1.0, c == 1u || c == 2u),
        select(-1.0, 1.0, c >= 2u),
    );
    let p = particles[instance_index];

    let clip_xy = p.pos * params.world_to_clip.xy + params.world_to_clip.zw
        + corner * params.particle_size;

    // Screen-relative speed: world velocity scales with view_height (flow is a
    // fraction of view height), so normalize by it to keep color/brightness the
    // same at any zoom - otherwise zoomed-out looks bright/colorful and zoomed-in
    // looks dim and flat. (view_height = 2 / world_to_clip.y.)
    let view_h = 2.0 / params.world_to_clip.y;
    let speed = length(p.vel) / view_h;
    let h = p.hue + speed * 0.95 + params.time * 0.015;
    // Sinebow palette, then saturate: subtract the valley floor and rescale so
    // the off-hue channels go to true 0. Without this the palette's ~0.25 valleys
    // accumulate under additive blending in dense regions (the fractal edge) and
    // clip to white. Saturating keeps the dense edge a coherent color.
    let raw = 0.5 + 0.5 * cos(6.2831853 * h - vec3<f32>(0.0, 2.0944, 4.1888));
    let sinebow = max(raw - vec3<f32>(0.32), vec3<f32>(0.0)) / 0.68;

    var out: VsOut;
    out.clip = vec4<f32>(clip_xy, 0.0, 1.0);
    out.uv = corner;
    // Keep the flat floor at 0.25 so the slow particles sitting on the boundary
    // (the fractal shape itself) stay visible; speed lifts the flowing streams
    // on top of that. Saturated palette above prevents dense-edge white-out.
    out.color = sinebow * (0.25 + speed * 5.5) * params.brightness;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Soft additive disc.
    let a = max(0.0, 1.0 - dot(in.uv, in.uv));
    return vec4<f32>(in.color * a * a, 1.0);
}
