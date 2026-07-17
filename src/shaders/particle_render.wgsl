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
    ref_len: u32,
    frame: u32,
    reseed_rate: f32,
    detail: f32,
    dissolve: f32,
    color_mode: u32,
    fractal_type: u32,
    // Unused in this shader; present for layout parity with ParamsUniform.
    trail_decay: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

// Sinebow palette, then saturate: subtract the valley floor and rescale so
// the off-hue channels go to true 0. Without this the palette's ~0.25 valleys
// accumulate under additive blending in dense regions (the fractal edge) and
// clip to white. Saturating keeps the dense edge a coherent color.
fn sinebow(h: f32) -> vec3<f32> {
    let raw = 0.5 + 0.5 * cos(6.2831853 * h - vec3<f32>(0.0, 2.0944, 4.1888));
    return max(raw - vec3<f32>(0.32), vec3<f32>(0.0)) / 0.68;
}

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

    var color: vec3<f32>;
    switch params.color_mode {
        case 1u: {
            // Rings: the escape-time field rendered as discrete glowing
            // contour lines. A cosine over the raw band value peaks once per
            // few iteration bands; cubing it thins each peak into a crisp
            // neon ring with dim gaps, so the nested level-set structure -
            // normally smeared into a continuous gradient - becomes visible
            // as topography. Hue drifts slowly along depth and time, and the
            // rings crawl inward (band phase moves with time) so the picture
            // breathes even where nothing flows.
            let stripe = pow(0.5 + 0.5 * cos(p.band * 2.2 - params.time * 0.8), 3.0);
            let hue = fract(p.band * 0.013 + params.time * 0.012);
            color = sinebow(hue) * (0.08 + 1.5 * stripe) * (0.35 + speed * 3.5);
        }
        case 2u: {
            // Electric: speed is energy, velocity DIRECTION is color. The
            // hue swings through the cyan/blue/violet/magenta family with
            // the angle of motion, so a vortex fans into a pinwheel and
            // crossing streams separate into distinct arcs instead of one
            // flat blue. Speed then crushes the tint: near-black violet at
            // rest, direction color mid-range, blown-out white at the top.
            // pow(speed, 0.75) spreads the crowded low end of the speed
            // distribution across the ramp.
            let t = clamp(pow(speed * 2.4, 0.75), 0.0, 1.0);
            let ang = atan2(p.vel.y, p.vel.x) / 6.2831853;
            let hue = 0.55 + 0.18 * cos(6.2831853 * (ang + params.time * 0.02));
            var tint = sinebow(hue);
            tint = mix(vec3<f32>(0.25, 0.1, 0.7), tint, smoothstep(0.0, 0.25, t));
            tint = mix(tint, vec3<f32>(1.0, 1.0, 1.0), smoothstep(0.7, 1.0, t));
            color = tint * (0.3 + t * t * 5.5 + speed * 2.0);
        }
        case 3u: {
            // Inferno: blackbody ramp - ember purple through crimson and
            // orange to white-hot. The blue channel rises early (purple
            // base), collapses through the mid-range (pure fire), returns
            // at the top (white). Heat comes from three sources so the
            // boundary-crowded particle distribution doesn't collapse to
            // one shade: depth sets the base (capped at 0.75 so the shell
            // alone never saturates), a cosine over the raw band value lays
            // fine ember striations across it (drifting with time like
            // coals breathing), and speed adds up to full white so the
            // streams run visibly hotter than the still shell.
            let d = clamp(p.band / f32(params.max_iter), 0.0, 1.0);
            let base = pow(d, 0.45) * 0.72;
            // Two striation frequencies (fine coals + broad waves) plus a
            // per-particle flicker with p.hue as a random phase, so nearby
            // particles on the same band don't pulse in lockstep.
            let ripple = 0.16 * cos(p.band * 1.7 - params.time * 0.7)
                + 0.08 * cos(p.band * 0.23 + params.time * 0.15);
            let flicker = 0.06 * cos(params.time * 2.5 + p.hue * 80.0);
            let t = clamp(base + ripple + flicker + speed * 1.2, 0.0, 1.0);
            let ramp = vec3<f32>(
                pow(t, 0.55),
                pow(t, 2.2) * 0.95,
                1.65 * t * pow(1.0 - t, 3.0) + pow(t, 6.0),
            );
            color = ramp * (0.4 + speed * 3.0);
        }
        default: {
            // Classic: band hue shifted by speed plus a slow global drift.
            // Flat floor at 0.25 keeps the slow particles on the boundary (the
            // fractal shape itself) visible; speed lifts the streams on top.
            let h = p.hue + speed * 0.95 + params.time * 0.015;
            color = sinebow(h) * (0.25 + speed * 5.5);
        }
    }

    var out: VsOut;
    out.clip = vec4<f32>(clip_xy, 0.0, 1.0);
    out.uv = corner;
    out.color = color * params.brightness;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Soft additive disc.
    let a = max(0.0, 1.0 - dot(in.uv, in.uv));
    return vec4<f32>(in.color * a * a, 1.0);
}
