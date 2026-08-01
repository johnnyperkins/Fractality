// The ShaderType derive generates per-field check fns that trip dead_code.
#![allow(dead_code)]

use std::sync::Arc;

use bevy::asset::DirectAssetAccessExt;
use bevy::math::DVec2;

use crate::dd::{Dd, DdVec2};
use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::prelude::*;
use bevy::render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    graph::CameraDriverLabel,
    render_graph::{self, RenderGraph, RenderGraphApp, RenderLabel, ViewNode, ViewNodeRunner},
    render_resource::binding_types::{
        storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, uniform_buffer,
    },
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
    view::ViewTarget,
    Extract, ExtractSchedule, Render, RenderApp, RenderSet,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// GPU particle, 32 bytes. Layout must match the WGSL Particle struct exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub home: [f32; 2],
    pub band: f32,
    pub hue: f32,
}

/// Uniform parameters. Field order must match the WGSL Params struct exactly.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct ParamsUniform {
    /// xy scale, zw offset: clip = pos * scale + offset. Positions are stored
    /// relative to the view center, so offset (zw) is always zero.
    pub world_to_clip: Vec4,
    /// xy pos (center-relative), z button (1 left / -1 right / 0 none), w radius
    pub mouse: Vec4,
    /// clip-space half extents of the particle quad
    pub particle_size: Vec2,
    /// (prev_center - cur_center) in world units. Added to every particle each
    /// frame to keep positions relative to the moving view center.
    pub center_delta: Vec2,
    pub time: f32,
    pub dt: f32,
    pub count: u32,
    pub max_iter: u32,
    pub flow_speed: f32,
    pub band_k: f32,
    pub damping: f32,
    pub brightness: f32,
    /// Number of valid entries in the reference orbit buffer.
    pub ref_len: u32,
    /// Frame counter, seeds particle-recycle RNG.
    pub frame: u32,
    /// Per-frame probability a particle is recycled. Rises with zoom-out speed
    /// so fill keeps pace with the newly revealed area; floors at a small trickle.
    pub reseed_rate: f32,
    /// Boundary-tightness exponent driver. Higher = particles pack closer to the
    /// set edge on recycle (finer, sharper boundary). Also the detail knob.
    pub detail: f32,
    /// 1.0 = dissolve mode (Space): flow field inert, particles melt off their
    /// contours via diffusion; mouse impulses still apply. 0.0 = normal flow.
    pub dissolve: f32,
    /// Palette selector for the render shader (C cycles): 0 classic, 1 rings,
    /// 2 electric, 3 inferno, 4 audio aurora.
    pub color_mode: u32,
    /// Fractal formula (F cycles): 0 Mandelbrot, 1 Burning Ship, 2 Tricorn,
    /// 3 Multibrot-3, 4 Julia. Must match the switch in the compute shader.
    pub fractal_type: u32,
    /// Particle flow style (G cycles / menu dropdown): 0 contour, 1 layers,
    /// 2 gravity, 3 erupt, 4 pulse, 5 dynamics. Must match the switch in the
    /// compute shader.
    pub flow_mode: u32,
    /// Per-frame trail keep factor, frame-rate corrected (settings.trail at a
    /// 60 FPS reference). 0 = trails off: particles draw straight to the view
    /// target and the trail texture path is skipped entirely.
    pub trail_decay: f32,
    /// Audio reactivity levels: x bass, y mid, z treble, w beat pulse (1 on a
    /// detected beat, exponential decay). All zero while disabled, so every
    /// consumer is a natural no-op with no enable flag.
    pub audio: Vec4,
    /// Palette hue offset accumulated from music energy.
    pub audio_hue: f32,
    /// x seconds since last beat, y seconds since last drop (both saturate
    /// high, so waves die out), z kaleidoscope rotation angle (radians,
    /// music-driven), w overall level.
    pub audio2: Vec4,
    /// Effect gains from the audio settings sliders: x ring pulse, y flash /
    /// glitter, z spectrum glow, w kaleidoscope segment count (0 = off).
    pub audio_fx: Vec4,
    /// 16 log-spaced spectrum bins (bin 0 = lowest), packed 4 per vec4.
    pub spectrum: [Vec4; 4],
}

impl ParamsUniform {
    /// Whether this frame renders through the offscreen trail texture plus
    /// composite pass (vs particles drawn straight to the view target):
    /// trails on, or kaleidoscope on (the fold happens in the composite
    /// shader; with decay 0 the fade pass wipes the texture each frame so no
    /// trails appear). Must match the `n >= 2.0` enable check in trail.wgsl.
    pub fn needs_composite(&self) -> bool {
        self.trail_decay > 0.0 || self.audio_fx.w >= 2.0
    }
}

/// Max reference-orbit length (also caps max_iter). One vec4<f32> per entry
/// (hi/lo pairs), so the whole buffer is 256 KB - free on the GPU. Sized so
/// the depth ramp
/// (~8750 iterations at the height floor of 1e-28) fits with detail-slider
/// headroom. The real cost of a long orbit is the per-particle iteration
/// loop, which `iter_budget_count` in main.rs pays for by trading particle
/// count against depth.
pub const REF_ORBIT_CAP: usize = 16384;

/// GPU particle buffer capacity. The full buffer is always allocated; the live
/// `count` uniform caps how many are actually simulated/drawn, so the settings
/// menu can change particle count instantly with no buffer reallocation.
#[cfg(not(target_arch = "wasm32"))]
pub const MAX_PARTICLES: u32 = 12_000_000;
/// WebGPU's default maxStorageBufferBindingSize is 128 MiB; at 32 bytes per
/// particle, 4M (128 MB) is the largest count that binds without requesting
/// higher limits.
#[cfg(target_arch = "wasm32")]
pub const MAX_PARTICLES: u32 = 4_000_000;

/// CPU-computed double-double reference orbit at the view center, each entry
/// stored as an f32 hi/lo pair per component: (hi.x, hi.y, lo.x, lo.y), ~48
/// bits of the f64 value. Plain f32 entries break past height ~1e-15: the
/// shader's z = Z_ref + dz cancels catastrophically when the orbit
/// close-approaches zero, and close-approach size (~sqrt(height)) drops below
/// f32's absolute rounding error (~6e-8 of |Z|) right around there. The lo
/// limb restores the cancelled digits (see `field` in the compute shader).
/// `generation` bumps on every recompute so the render world uploads the
/// buffer only when the orbit actually changed.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct RefOrbit {
    pub points: Vec<[f32; 4]>,
    pub generation: u32,
}

/// Fractal type id of the Julia set in FRACTAL_MODES and the shader switches.
pub const JULIA_TYPE: u32 = 4;

/// Home Julia parameter. Only the CPU needs it: the GPU delta iteration for
/// Julia has no c term (dc seeds dz_0 instead), so c reaches the GPU only
/// through the reference orbit.
pub const JULIA_C: (f64, f64) = (-0.7269, 0.1889);

/// Julia parameter for the audio morph: orbits JULIA_C at the given phase
/// (0..1) and radius.
pub fn julia_morph_c(phase: f64, radius: f64) -> (f64, f64) {
    let th = phase * std::f64::consts::TAU;
    (JULIA_C.0 + radius * th.cos(), JULIA_C.1 + radius * th.sin())
}

/// One iteration of the selected fractal map in f64. Types must match the
/// switch in the compute shader: 0 Mandelbrot (also Julia's map), 1 Burning
/// Ship, 2 Tricorn, 3 Multibrot-3.
fn fractal_step(zx: f64, zy: f64, cx: f64, cy: f64, ftype: u32) -> (f64, f64) {
    match ftype {
        1 => {
            let ax = zx.abs();
            let ay = zy.abs();
            (ax * ax - ay * ay + cx, 2.0 * ax * ay + cy)
        }
        2 => (zx * zx - zy * zy + cx, -2.0 * zx * zy + cy),
        3 => (
            zx * (zx * zx - 3.0 * zy * zy) + cx,
            zy * (3.0 * zx * zx - zy * zy) + cy,
        ),
        _ => (zx * zx - zy * zy + cx, 2.0 * zx * zy + cy),
    }
}

/// Initial z and effective c for a point of the given fractal. Julia iterates
/// the point itself under the given c (the audio morph orbits it around
/// JULIA_C); everything else iterates from 0 with the point as c.
fn orbit_start(x: f64, y: f64, ftype: u32, jc: (f64, f64)) -> (f64, f64, f64, f64) {
    if ftype == JULIA_TYPE {
        (x, y, jc.0, jc.1)
    } else {
        (0.0, 0.0, x, y)
    }
}

/// 1/log2(power): smooth-iteration scale so fractional bands stay continuous
/// for maps of power != 2 (only Multibrot-3 here).
fn inv_log2_power(ftype: u32) -> f64 {
    if ftype == 3 {
        1.0 / 3.0f64.log2()
    } else {
        1.0
    }
}

/// `fractal_step` in double-double, for the reference orbit. Must stay in
/// lockstep with the f64 version and the compute-shader switch.
fn fractal_step_dd(zx: Dd, zy: Dd, cx: Dd, cy: Dd, ftype: u32) -> (Dd, Dd) {
    match ftype {
        1 => {
            let ax = zx.abs();
            let ay = zy.abs();
            (ax * ax - ay * ay + cx, ax * ay * 2.0 + cy)
        }
        2 => (zx * zx - zy * zy + cx, cy - zx * zy * 2.0),
        3 => (
            zx * (zx * zx - zy * zy * 3.0) + cx,
            zy * (zx * zx * 3.0 - zy * zy) + cy,
        ),
        _ => (zx * zx - zy * zy + cx, zx * zy * 2.0 + cy),
    }
}

/// Split an f64 into an f32 hi/lo pair: hi = rounded value, lo = the ~24 bits
/// of residual, together ~48 bits of the original.
#[inline]
fn split_f32(v: f64) -> (f32, f32) {
    let hi = v as f32;
    (hi, (v - hi as f64) as f32)
}

/// Iterate the selected map at the reference point in double-double (~31
/// digits, so the orbit is exact for views down to height ~1e-28), storing
/// Z_0..Z_n as f32 hi/lo pairs (see `RefOrbit`) - perturbation needs the c
/// behind the orbit at full precision, the stored samples only well enough to
/// survive the close-approach cancellation in the shader. Stops at max_iter,
/// REF_ORBIT_CAP, or when the orbit diverges hard.
pub fn reference_orbit(c: DdVec2, max_iter: u32, ftype: u32, jc: (f64, f64)) -> Vec<[f32; 4]> {
    // Julia iterates the center itself under c = jc; everything else iterates
    // from 0 with the center as c (the DD mirror of `orbit_start`).
    let (mut zx, mut zy, ccx, ccy) = if ftype == JULIA_TYPE {
        (c.x, c.y, Dd::from_f64(jc.0), Dd::from_f64(jc.1))
    } else {
        (Dd::ZERO, Dd::ZERO, c.x, c.y)
    };
    let mut v = Vec::with_capacity((max_iter as usize + 1).min(REF_ORBIT_CAP));
    let push = |v: &mut Vec<[f32; 4]>, zx: Dd, zy: Dd| {
        let (hx, lx) = split_f32(zx.hi);
        let (hy, ly) = split_f32(zy.hi);
        v.push([hx, hy, lx, ly]);
    };
    push(&mut v, zx, zy); // Z_0 (0 except Julia, where it's the center)
    for _ in 0..max_iter {
        let (nx, ny) = fractal_step_dd(zx, zy, ccx, ccy, ftype);
        zx = nx;
        zy = ny;
        push(&mut v, zx, zy);
        if zx.hi * zx.hi + zy.hi * zy.hi > 1e10 || v.len() >= REF_ORBIT_CAP {
            break;
        }
    }
    v
}

#[cfg(test)]
mod dd_orbit_tests {
    use super::*;

    /// fractal_step_dd must be the same map as fractal_step: iterate both at
    /// a non-escaping point and compare. A transcription error (sign, factor)
    /// would silently render a different fractal at every depth.
    #[test]
    fn dd_step_matches_f64_step() {
        for ftype in 0..=4u32 {
            let (mut zx, mut zy, cx, cy) = orbit_start(-0.16, 0.65, ftype, JULIA_C);
            let (mut dzx, mut dzy) = (Dd::from_f64(zx), Dd::from_f64(zy));
            let (dcx, dcy) = (Dd::from_f64(cx), Dd::from_f64(cy));
            for i in 0..60 {
                (zx, zy) = fractal_step(zx, zy, cx, cy, ftype);
                let (nx, ny) = fractal_step_dd(dzx, dzy, dcx, dcy, ftype);
                dzx = nx;
                dzy = ny;
                if zx * zx + zy * zy > 1e10 {
                    break;
                }
                let err = (zx - dzx.hi).abs().max((zy - dzy.hi).abs());
                assert!(err < 1e-9, "ftype {ftype} iter {i}: err {err:e}");
            }
        }
    }
}

/// Per-frame simulation parameters, extracted into the render world.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct SimParams(pub ParamsUniform);

/// CPU-generated initial particle data, read once by the render world.
#[derive(Resource)]
pub struct ParticleSeed(pub Arc<Vec<Particle>>);

/// Smooth escape-time field on the CPU (f64). The formula and the escape
/// radius (256.0) must be identical to field() in the WGSL shaders so that
/// CPU band values match GPU field values.
fn smooth_iter(x: f64, y: f64, max_iter: u32, ftype: u32) -> f32 {
    let (mut zx, mut zy, cx, cy) = orbit_start(x, y, ftype, JULIA_C);
    let ilp = inv_log2_power(ftype);
    for i in 0..max_iter {
        let (nx, ny) = fractal_step(zx, zy, cx, cy, ftype);
        zx = nx;
        zy = ny;
        let m = zx * zx + zy * zy;
        if m > 256.0 {
            return i as f32 + 1.0 - ((0.5 * m.log2()).log2() * ilp) as f32;
        }
    }
    max_iter as f32
}

/// Seed-sampling rectangle (x0, x1, y0, y1) covering the fractal's exterior
/// boundary region. Coarse is fine: rejection keeps only boundary points and
/// the GPU recycle resamples the live view within seconds anyway.
fn spawn_rect(ftype: u32) -> (f64, f64, f64, f64) {
    match ftype {
        1 => (-2.5, 1.6, -2.0, 1.0),
        2 => (-2.4, 1.6, -2.0, 2.0),
        3 => (-1.6, 1.6, -1.6, 1.6),
        4 => (-1.8, 1.8, -1.3, 1.3),
        _ => (-2.3, 0.85, -1.3, 1.3),
    }
}

pub fn generate_particles(count: usize, max_iter: u32, center: DVec2, ftype: u32) -> Vec<Particle> {
    let (x0, x1, y0, y1) = spawn_rect(ftype);
    // rayon has no plain wasm story (needs SharedArrayBuffer plumbing), so the
    // web build generates sequentially; the smaller default count keeps
    // startup tolerable.
    #[cfg(not(target_arch = "wasm32"))]
    let range = (0..count).into_par_iter();
    #[cfg(target_arch = "wasm32")]
    let range = 0..count;
    range
        .map(|i| {
            let mut rng =
                fastrand::Rng::with_seed(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1));
            loop {
                let x = x0 + rng.f64() * (x1 - x0);
                let y = y0 + rng.f64() * (y1 - y0);
                let f = smooth_iter(x, y, max_iter, ftype);
                // Inside the set: reject.
                if f >= max_iter as f32 - 1.0 {
                    continue;
                }
                // Bias samples toward the boundary.
                let t = rng.f32();
                let threshold = 3.0 + (max_iter as f32 - 12.0) * t * t * t * t;
                if f > 3.0 && f >= threshold {
                    // Store relative to the view center.
                    let pos = [(x - center.x) as f32, (y - center.y) as f32];
                    return Particle {
                        pos,
                        vel: [0.0, 0.0],
                        home: pos,
                        band: f,
                        hue: (f * 0.045 + 0.62).fract(),
                    };
                }
            }
        })
        .collect()
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ParticleComputeLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ParticleDrawLabel;

#[derive(Resource)]
struct ParticleBuffers {
    buffer: Buffer,
    count: u32,
}

#[derive(Resource, Default)]
struct ParticleUniform(UniformBuffer<ParamsUniform>);

/// Fixed-capacity GPU buffer holding the current reference orbit.
#[derive(Resource)]
struct RefOrbitBuffer(Buffer);

impl FromWorld for RefOrbitBuffer {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ref_orbit_buffer"),
            size: (REF_ORBIT_CAP * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self(buffer)
    }
}

#[derive(Resource)]
struct ParticleBindGroups {
    compute: BindGroup,
    render: BindGroup,
}

#[derive(Resource)]
struct ParticlePipelines {
    compute_layout: BindGroupLayout,
    render_layout: BindGroupLayout,
    composite_layout: BindGroupLayout,
    compute_pipeline: CachedComputePipelineId,
    render_pipeline: CachedRenderPipelineId,
    fade_pipeline: CachedRenderPipelineId,
    composite_pipeline: CachedRenderPipelineId,
}

/// Persistent screen-sized HDR texture the particles accumulate into when
/// trails are on. Faded a little each frame instead of cleared, then blitted
/// onto the view target. Recreated on resize; cleared on (re)enable.
#[derive(Resource)]
struct TrailTexture {
    view: TextureView,
    bind_group: BindGroup,
    size: (u32, u32),
    /// First trail frame after enable/resize: clear instead of loading stale
    /// (or garbage) history.
    needs_clear: bool,
}

impl FromWorld for ParticlePipelines {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let compute_layout = device.create_bind_group_layout(
            "particle_compute_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_sized(false, None),
                    uniform_buffer::<ParamsUniform>(false),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        );
        let render_layout = device.create_bind_group_layout(
            "particle_render_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    storage_buffer_read_only_sized(false, None),
                    uniform_buffer::<ParamsUniform>(false),
                ),
            ),
        );
        let composite_layout = device.create_bind_group_layout(
            "trail_composite_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    uniform_buffer::<ParamsUniform>(false),
                ),
            ),
        );
        let compute_shader: Handle<Shader> =
            world.load_asset("embedded://fractality/shaders/particle_compute.wgsl");
        let render_shader: Handle<Shader> =
            world.load_asset("embedded://fractality/shaders/particle_render.wgsl");
        let trail_shader: Handle<Shader> =
            world.load_asset("embedded://fractality/shaders/trail.wgsl");
        let cache = world.resource::<PipelineCache>();
        let compute_pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("particle_compute_pipeline".into()),
            layout: vec![compute_layout.clone()],
            push_constant_ranges: vec![],
            shader: compute_shader,
            shader_defs: vec![],
            entry_point: "update".into(),
            zero_initialize_workgroup_memory: false,
        });
        let render_pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("particle_render_pipeline".into()),
            layout: vec![render_layout.clone()],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: render_shader.clone(),
                shader_defs: vec![],
                entry_point: "vs".into(),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: render_shader,
                shader_defs: vec![],
                entry_point: "fs".into(),
                targets: vec![Some(ColorTargetState {
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            zero_initialize_workgroup_memory: false,
        });
        // Both trail passes are fullscreen-triangle fragments over the HDR
        // target; only entry point, layout, and blend differ.
        let trail_pass = |label: &'static str,
                          entry: &'static str,
                          layout: Vec<BindGroupLayout>,
                          blend: Option<BlendState>| RenderPipelineDescriptor {
            label: Some(label.into()),
            layout,
            push_constant_ranges: vec![],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: trail_shader.clone(),
                shader_defs: vec![],
                entry_point: entry.into(),
                targets: vec![Some(ColorTargetState {
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    blend,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            zero_initialize_workgroup_memory: false,
        };
        // Fade pass: darken the trail texture in place. The fragment outputs
        // white; src_factor Zero / dst_factor Constant makes the result
        // dst * blend_constant, and the draw node sets the blend constant to
        // the per-frame keep factor. No bind group needed.
        let fade_blend = BlendComponent {
            src_factor: BlendFactor::Zero,
            dst_factor: BlendFactor::Constant,
            operation: BlendOperation::Add,
        };
        let fade_pipeline = cache.queue_render_pipeline(trail_pass(
            "trail_fade_pipeline",
            "fs_fade",
            vec![],
            Some(BlendState {
                color: fade_blend,
                alpha: fade_blend,
            }),
        ));
        // Composite pass: copy the trail texture onto the view target 1:1.
        let composite_pipeline = cache.queue_render_pipeline(trail_pass(
            "trail_composite_pipeline",
            "fs_composite",
            vec![composite_layout.clone()],
            None,
        ));
        Self {
            compute_layout,
            render_layout,
            composite_layout,
            compute_pipeline,
            render_pipeline,
            fade_pipeline,
            composite_pipeline,
        }
    }
}

fn extract_particle_buffers(
    mut commands: Commands,
    existing: Option<Res<ParticleBuffers>>,
    seed: Extract<Option<Res<ParticleSeed>>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    if existing.is_some() {
        return;
    }
    let Some(seed) = seed.as_ref() else {
        return;
    };
    // Allocate at full capacity and upload only the seeded prefix. wgpu
    // zero-initializes buffers (WebGPU spec), so the tail needs no CPU-side
    // staging or upload. The live `count` uniform gates how many are
    // simulated/drawn, so the count can be raised at runtime with no
    // reallocation. Zeroed tail particles sit at the origin and get pulled
    // into view by the recycle trickle as count rises.
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("particle_buffer"),
        size: MAX_PARTICLES as u64 * std::mem::size_of::<Particle>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let n = seed.0.len().min(MAX_PARTICLES as usize);
    if n > 0 {
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&seed.0[..n]));
    }
    commands.insert_resource(ParticleBuffers {
        buffer,
        count: MAX_PARTICLES,
    });
}

/// Free the CPU-side seed once the render world has copied it into the GPU
/// buffer. Extraction happens at the end of the frame the seed appears in, so
/// by this system's second run the tens-of-MB Vec is dead weight.
fn drop_particle_seed(
    mut commands: Commands,
    seed: Option<Res<ParticleSeed>>,
    mut frames: Local<u32>,
) {
    if seed.is_none() {
        return;
    }
    *frames += 1;
    if *frames >= 2 {
        commands.remove_resource::<ParticleSeed>();
    }
}

fn prepare_particle_bind_groups(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipelines: Res<ParticlePipelines>,
    buffers: Option<Res<ParticleBuffers>>,
    params: Option<Res<SimParams>>,
    ref_orbit: Option<Res<RefOrbit>>,
    ref_buffer: Res<RefOrbitBuffer>,
    mut uniform: ResMut<ParticleUniform>,
    mut last_generation: Local<Option<u32>>,
    existing_bind_groups: Option<Res<ParticleBindGroups>>,
) {
    let Some(buffers) = buffers else {
        return;
    };
    let Some(params) = params else {
        return;
    };
    if let Some(ref_orbit) = ref_orbit {
        if *last_generation != Some(ref_orbit.generation) {
            let bytes = bytemuck::cast_slice(ref_orbit.points.as_slice());
            if !bytes.is_empty() {
                queue.write_buffer(&ref_buffer.0, 0, bytes);
                *last_generation = Some(ref_orbit.generation);
            }
        }
    }
    uniform.0.set(params.0);
    uniform.0.write_buffer(&device, &queue);
    // Bind groups are immortal: the particle and ref-orbit buffers are created
    // once at fixed capacity, and the uniform buffer is allocated on its first
    // write and never resized (constant size), so nothing they reference ever
    // moves. Create them on the first frame and reuse forever.
    if existing_bind_groups.is_some() {
        return;
    }
    let Some(binding) = uniform.0.binding() else {
        return;
    };
    let compute = device.create_bind_group(
        "particle_compute_bind_group",
        &pipelines.compute_layout,
        &BindGroupEntries::sequential((
            buffers.buffer.as_entire_binding(),
            binding.clone(),
            ref_buffer.0.as_entire_binding(),
        )),
    );
    let render = device.create_bind_group(
        "particle_render_bind_group",
        &pipelines.render_layout,
        &BindGroupEntries::sequential((buffers.buffer.as_entire_binding(), binding)),
    );
    commands.insert_resource(ParticleBindGroups { compute, render });
}

/// Keep the trail texture matching the view size, and track when it needs a
/// clear (first frame after enabling trails, or after a resize). Created
/// lazily: a session that never turns trails on allocates nothing.
fn prepare_trail_texture(
    mut commands: Commands,
    device: Res<RenderDevice>,
    pipelines: Res<ParticlePipelines>,
    params: Option<Res<SimParams>>,
    uniform: Res<ParticleUniform>,
    views: Query<&ViewTarget>,
    existing: Option<ResMut<TrailTexture>>,
    mut was_active: Local<bool>,
) {
    let active = params.map_or(false, |p| p.0.needs_composite());
    // Trails never used this session: nothing to size-track or invalidate.
    if !active && existing.is_none() {
        *was_active = false;
        return;
    }
    let Ok(view_target) = views.single() else {
        *was_active = false;
        return;
    };
    let extent = view_target.main_texture().size();
    let size = (extent.width, extent.height);

    let recreate = existing.as_ref().map_or(true, |t| t.size != size);
    if recreate {
        if !active {
            *was_active = false;
            return;
        }
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("trail_texture"),
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: ViewTarget::TEXTURE_FORMAT_HDR,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // The uniform buffer is allocated on its first write; until then the
        // bind group can't exist, so retry next frame.
        let Some(binding) = uniform.0.binding() else {
            return;
        };
        let view = texture.create_view(&TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(
            "trail_composite_bind_group",
            &pipelines.composite_layout,
            &BindGroupEntries::sequential((&view, binding)),
        );
        commands.insert_resource(TrailTexture {
            view,
            bind_group,
            size,
            needs_clear: true,
        });
    } else if let Some(mut trail) = existing {
        // Re-enabling after a disabled stretch starts from black, not from
        // whatever the texture held when trails were last on. Write only on
        // change to avoid per-frame change-detection churn.
        let clear = !*was_active;
        if trail.needs_clear != clear {
            trail.needs_clear = clear;
        }
    }
    *was_active = active;
}

struct ParticleComputeNode;

impl render_graph::Node for ParticleComputeNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let (Some(bind_groups), Some(buffers), Some(pipelines)) = (
            world.get_resource::<ParticleBindGroups>(),
            world.get_resource::<ParticleBuffers>(),
            world.get_resource::<ParticlePipelines>(),
        ) else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_compute_pipeline(pipelines.compute_pipeline) else {
            return Ok(());
        };
        // Simulate only the active count (buffer holds up to MAX_PARTICLES).
        let count = world
            .get_resource::<SimParams>()
            .map_or(0, |p| p.0.count)
            .min(buffers.count);
        if count == 0 {
            return Ok(());
        }
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_groups.compute, &[]);
        pass.dispatch_workgroups((count + 255) / 256, 1, 1);
        Ok(())
    }
}

#[derive(Default)]
struct ParticleDrawNode;

impl ViewNode for ParticleDrawNode {
    type ViewQuery = &'static ViewTarget;

    fn run<'w>(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        view_target: bevy::ecs::query::QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), render_graph::NodeRunError> {
        let (Some(bind_groups), Some(buffers), Some(pipelines)) = (
            world.get_resource::<ParticleBindGroups>(),
            world.get_resource::<ParticleBuffers>(),
            world.get_resource::<ParticlePipelines>(),
        ) else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(pipelines.render_pipeline) else {
            return Ok(());
        };
        // Draw only the active count (buffer holds up to MAX_PARTICLES).
        let params = world.get_resource::<SimParams>();
        let count = params.map_or(0, |p| p.0.count).min(buffers.count);
        let decay = params.map_or(0.0, |p| p.0.trail_decay);
        let composite = params.map_or(false, |p| p.0.needs_composite());

        // Composite path needs the texture and both extra pipelines ready;
        // otherwise draw straight to the view target.
        let trail = world
            .get_resource::<TrailTexture>()
            .filter(|_| composite)
            .and_then(|t| {
                let fade = cache.get_render_pipeline(pipelines.fade_pipeline)?;
                let composite = cache.get_render_pipeline(pipelines.composite_pipeline)?;
                Some((t, fade, composite))
            });

        {
            // One particle pass either way; only the target differs. With
            // trails on, fade the previous frame first, then add this frame's
            // particles on top of the persistent trail texture.
            // Kaleidoscope-only frames (decay 0) keep no history: clear via
            // the load op instead of running the fade pass, which at blend
            // constant 0 would just be a fullscreen multiply-by-zero.
            let (label, attachment) = match &trail {
                Some((t, _, _)) => (
                    "trail_accumulate_pass",
                    RenderPassColorAttachment {
                        view: &t.view,
                        resolve_target: None,
                        ops: Operations {
                            load: if t.needs_clear || decay <= 0.0 {
                                LoadOp::Clear(LinearRgba::BLACK.into())
                            } else {
                                LoadOp::Load
                            },
                            store: StoreOp::Store,
                        },
                    },
                ),
                None => ("particle_draw_pass", view_target.get_color_attachment()),
            };
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some((t, fade, _)) = &trail {
                if !t.needs_clear && decay > 0.0 {
                    pass.set_render_pipeline(fade);
                    pass.set_blend_constant(LinearRgba::rgb(decay, decay, decay));
                    pass.draw(0..3, 0..1);
                }
            }
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, &bind_groups.render, &[]);
            pass.draw(0..6, 0..count);
        }
        if let Some((trail, _, composite)) = trail {
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("trail_composite_pass"),
                color_attachments: &[Some(view_target.get_color_attachment())],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_render_pipeline(composite);
            pass.set_bind_group(0, &trail.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        Ok(())
    }
}

pub struct ParticlePlugin;

impl Plugin for ParticlePlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "shaders/particle_compute.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/particle_render.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/trail.wgsl");

        app.add_plugins(ExtractResourcePlugin::<SimParams>::default());
        app.add_plugins(ExtractResourcePlugin::<RefOrbit>::default());
        app.add_systems(Update, drop_particle_seed);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(ExtractSchedule, extract_particle_buffers)
            .add_systems(
                Render,
                (prepare_particle_bind_groups, prepare_trail_texture)
                    .in_set(RenderSet::PrepareBindGroups),
            );
        render_app
            .add_render_graph_node::<ViewNodeRunner<ParticleDrawNode>>(Core2d, ParticleDrawLabel);
        render_app.add_render_graph_edges(
            Core2d,
            (Node2d::EndMainPass, ParticleDrawLabel, Node2d::Bloom),
        );
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(ParticleComputeLabel, ParticleComputeNode);
        graph.add_node_edge(ParticleComputeLabel, CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ParticlePipelines>();
        render_app.init_resource::<ParticleUniform>();
        render_app.init_resource::<RefOrbitBuffer>();
    }
}
