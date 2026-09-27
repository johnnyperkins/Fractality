// Structs shared by every Fractality shader, the one WGSL definition of each.
// Layouts must match their Rust mirrors in particles.rs (Particle and
// ParamsUniform); the shared_structs_match_rust_layouts test checks the sizes.
#define_import_path fractality::common

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    home: vec2<f32>,
    band: f32,
    hue: f32,
};

// Per-frame simulation parameters. Field docs live on ParamsUniform; the
// notes here say which shader reads what.
struct Params {
    // xy scale, zw offset: clip = pos * scale + offset (offset is zero:
    // positions are center-relative). view_height = 2 / world_to_clip.y.
    world_to_clip: vec4<f32>,
    // xy cursor (center-relative), z button (1 left / -1 right), w radius.
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
    // Render: palette. 0 classic, 1 rings, 2 electric, 3 inferno, 4 aurora.
    color_mode: u32,
    // Read by no shader: the compute node picks the fractal's pipeline by it.
    fractal_type: u32,
    // Compute: 0 contour, 1 layers, 2 gravity, 3 erupt, 4 pulse, 5 dynamics.
    flow_mode: u32,
    // CPU only (the fade pass takes it as its blend constant).
    trail_decay: f32,
    // x bass, y mid, z treble, w beat pulse. All zero while audio reactivity
    // is off, so every use is a natural no-op.
    audio: vec4<f32>,
    // Music-driven hue/phase accumulator.
    audio_hue: f32,
    // x seconds since last beat, y since last drop (saturate high),
    // z kaleidoscope rotation (radians), w overall level.
    audio_aux: vec4<f32>,
    // Effect gains: x ring pulse (compute), y flash/glitter + chromatic bloom,
    // z spectrum glow (render), w kaleidoscope folds (0 = off, trail).
    audio_fx: vec4<f32>,
    // 16 log-spaced spectrum bins (bin 0 = lowest), packed 4 per vec4.
    spectrum: array<vec4<f32>, 4>,
    // Compute: x condensation, y previous flow mode, z flow crossfade blend
    // (0 = all previous mode, 1 = all current), w spare.
    shape: vec4<f32>,
};
