// Trail / motion-blur fragments. Both draw over Bevy's stock fullscreen
// triangle (fullscreen_shader_vertex_state on the Rust side).
//
// fs_fade darkens the persistent trail texture in place: it outputs white and
// the pipeline blends with src_factor Zero / dst_factor Constant, so the
// result is dst * blend_constant. The draw node sets the blend constant to the
// frame-rate-corrected keep factor (params.trail_decay on the CPU side).
//
// fs_composite copies the trail texture onto the view target 1:1 (textureLoad,
// no sampler needed).

@group(0) @binding(0) var trail_tex: texture_2d<f32>;

@fragment
fn fs_fade() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}

@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(trail_tex, vec2<i32>(pos.xy), 0);
}
