// Trail / motion-blur fragments. Both draw over Bevy's stock fullscreen
// triangle (fullscreen_shader_vertex_state on the Rust side).
//
// fs_fade darkens the persistent trail texture in place: it outputs white and
// the pipeline blends with src_factor Zero / dst_factor Constant, so the
// result is dst * blend_constant. The draw node sets the blend constant to the
// frame-rate-corrected keep factor (params.trail_decay on the CPU side).
//
// fs_composite copies the trail texture onto the view target 1:1 (textureLoad,
// no sampler needed). On beats it splits the copy into chromatic fringes:
// R and B sample radially offset positions scaled by beat energy, so a beat
// hit fans the image into color fringes that snap back as the pulse decays.

// Layout parity with ParamsUniform (particles.rs). Only audio/audio_fx are
// read here.
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
    flow_mode: u32,
    trail_decay: f32,
    // x bass, y mid, z treble, w beat pulse.
    audio: vec4<f32>,
    audio_hue: f32,
    // z kaleidoscope rotation angle (radians, music-driven).
    audio2: vec4<f32>,
    // Effect gains: y flash/glitter also gates the chromatic bloom.
    // w kaleidoscope segment count (0 = off).
    audio_fx: vec4<f32>,
    spectrum: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var trail_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@fragment
fn fs_fade() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}

@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(trail_tex));
    let hi = vec2<i32>(dims) - vec2<i32>(1);

    // Kaleidoscope (K): fold the screen N-fold around its center before
    // sampling. The pixel's angle is rotated (music-driven spin), wrapped
    // into one wedge of width tau/N, and mirrored about the wedge center,
    // so one wedge of the source image tiles the screen as a mandala.
    var sp = pos.xy;
    let n = params.audio_fx.w;
    if (n >= 2.0) {
        let center = dims * 0.5;
        let d = pos.xy - center;
        let seg = 6.2831853 / n;
        var a = atan2(d.y, d.x) - params.audio2.z;
        a = a - seg * floor(a / seg);
        a = min(a, seg - a);
        sp = center + length(d) * vec2<f32>(cos(a), sin(a));
        sp = clamp(sp, vec2<f32>(0.0), dims - vec2<f32>(1.0));
    }

    let base = textureLoad(trail_tex, vec2<i32>(sp), 0);
    let beat = params.audio.w * params.audio_fx.y;
    if (beat < 0.01) {
        return base;
    }
    // Radial split: zero at screen center, growing toward the edges, so the
    // bloom reads as the whole frame flaring outward. R samples outward,
    // B inward, G anchors.
    let dir = (pos.xy - dims * 0.5) / max(dims.y, 1.0);
    let off = dir * beat * beat * 14.0;
    let r = textureLoad(trail_tex, clamp(vec2<i32>(sp + off), vec2<i32>(0), hi), 0).r;
    let b = textureLoad(trail_tex, clamp(vec2<i32>(sp - off), vec2<i32>(0), hi), 0).b;
    return vec4<f32>(r, base.g, b, base.a);
}
