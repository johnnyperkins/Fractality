use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bevy::asset::DirectAssetAccessExt;
use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::math::DVec2;
use bevy::prelude::*;
use bevy::render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
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

use crate::fractal::{smooth_iter, Fractal, JULIA_C, REF_ORBIT_CAP};

/// Marker for the camera the particle pipeline draws on. The draw node and
/// trail sizing only run for this view: the native present camera (which
/// composites the offscreen scene under the UI) is a plain sRGB target and
/// running the HDR-format pipelines on it would fail wgpu validation.
#[derive(Component, Clone, ExtractComponent)]
pub struct ParticleCamera;

/// GPU particle, 32 bytes. Layout must match Particle in common.wgsl exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub home: [f32; 2],
    pub band: f32,
    pub hue: f32,
}

pub use uniform::ParamsUniform;

// ShaderType's derive emits a never-called `check` fn per field, and only a
// module-level allow reaches them - hence a module of its own, so the rest of
// this file keeps its dead-code lint.
#[allow(dead_code)]
mod uniform {
    use bevy::prelude::*;
    use bevy::render::render_resource::ShaderType;

    /// Uniform parameters. Mirrors Params in common.wgsl field for field.
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
        /// `Fractal` id (F cycles). The shaders never read it: the compute node
        /// uses it to pick that fractal's pipeline.
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
        pub audio_aux: Vec4,
        /// Effect gains from the audio settings sliders: x ring pulse, y flash /
        /// glitter, z spectrum glow, w kaleidoscope segment count (0 = off).
        pub audio_fx: Vec4,
        /// 16 log-spaced spectrum bins (bin 0 = lowest), packed 4 per vec4.
        pub spectrum: [Vec4; 4],
        /// Shape/dynamics tuning: x condensation (how hard particles freeze onto
        /// the boundary shell; 1 = classic, higher = wider and deader freeze,
        /// 0 = everything streams), y previous flow mode and z crossfade
        /// progress (0 = all previous mode, 1 = all current) for the flow-mode
        /// morph, w spare.
        pub shape: Vec4,
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
}

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
///
/// The points sit behind an Arc because ExtractResource clones this into the
/// render world every frame: a dive at the iteration cap would otherwise
/// memcpy (and allocate, and free) 256 KB per frame for a buffer the render
/// world only reads, and usually only reads when `generation` moved.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct RefOrbit {
    pub points: Arc<Vec<[f32; 4]>>,
    pub generation: u32,
}

/// Per-frame simulation parameters, extracted into the render world.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct SimParams(pub ParamsUniform);

/// CPU-generated initial particle data, read once by the render world.
#[derive(Resource)]
pub struct ParticleSeed(pub Arc<Vec<Particle>>);

/// Boundary-biased initial cloud, positions relative to `center`. Same
/// accept rule as the GPU recycle: reject the interior, then prefer high
/// escape times so the cloud starts on the fractal's edge.
pub fn generate_particles(
    count: usize,
    max_iter: u32,
    center: DVec2,
    fractal: Fractal,
) -> Vec<Particle> {
    let (x0, x1, y0, y1) = fractal.spawn_rect();
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
                let f = smooth_iter(x, y, max_iter, fractal, JULIA_C);
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
    /// One per fractal, indexed by `Fractal::id`.
    compute_pipelines: [CachedComputePipelineId; Fractal::ALL.len()],
    render_pipeline: CachedRenderPipelineId,
    fade_pipeline: CachedRenderPipelineId,
    composite_pipeline: CachedRenderPipelineId,
    /// The shared struct module every shader imports. Held only to keep the
    /// asset alive: the pipeline cache drops an import when its shader
    /// unloads, and would then never finish compiling the others.
    _common_shader: Handle<Shader>,
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
        let common_shader: Handle<Shader> =
            world.load_asset("embedded://fractality/shaders/common.wgsl");
        let compute_shader: Handle<Shader> =
            world.load_asset("embedded://fractality/shaders/particle_compute.wgsl");
        let render_shader: Handle<Shader> =
            world.load_asset("embedded://fractality/shaders/particle_render.wgsl");
        let trail_shader: Handle<Shader> =
            world.load_asset("embedded://fractality/shaders/trail.wgsl");
        let cache = world.resource::<PipelineCache>();
        // All queued up front, so pressing F never waits on a shader compile.
        let compute_pipelines = Fractal::ALL.map(|fractal| {
            let def = fractal.shader_def();
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("particle_compute_pipeline_{def}").into()),
                layout: vec![compute_layout.clone()],
                push_constant_ranges: vec![],
                shader: compute_shader.clone(),
                shader_defs: vec![ShaderDefVal::Bool(def.into(), true)],
                entry_point: "update".into(),
                zero_initialize_workgroup_memory: false,
            })
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
            // Strip: 4 vertices per particle quad instead of 6 (strips restart
            // between instances), cutting the heavy per-vertex color work by a
            // third. Corner derivation in the shader matches strip order.
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            zero_initialize_workgroup_memory: false,
        });
        // Both trail passes are fullscreen-triangle fragments over the HDR
        // target; only entry point, layout, and blend differ.
        let trail_pass =
            |label: &'static str,
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
            compute_pipelines,
            render_pipeline,
            fade_pipeline,
            composite_pipeline,
            _common_shader: common_shader,
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
    commands.insert_resource(ParticleBuffers { buffer });
}

/// Free the CPU-side seed once the render world has copied it into the GPU
/// buffer. Extraction happens at the end of the frame the seed appears in, so
/// by this system's second run the tens-of-MB Vec is dead weight.
fn drop_particle_seed(mut commands: Commands, mut frames: Local<u32>) {
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
    views: Query<&ViewTarget, With<ParticleCamera>>,
    existing: Option<ResMut<TrailTexture>>,
    mut was_active: Local<bool>,
) {
    let active = params.is_some_and(|p| p.0.needs_composite());
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

    let recreate = existing.as_ref().is_none_or(|t| t.size != size);
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

#[derive(Default)]
struct ParticleComputeNode {
    /// Index of the fractal whose pipeline last dispatched. Pipeline compiles
    /// are async: a fractal switch in the first seconds of a session (or on a
    /// slow driver) can land before the new variant is ready, and skipping the
    /// dispatch would freeze the simulation while the render, fade, and
    /// composite passes keep running. Those frames dispatch this pipeline
    /// instead: a brief blend of old formula and new reference orbit that the
    /// recycler resamples away, rather than a visible stall.
    last_ready: AtomicUsize,
}

impl render_graph::Node for ParticleComputeNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let (Some(bind_groups), Some(pipelines)) = (
            world.get_resource::<ParticleBindGroups>(),
            world.get_resource::<ParticlePipelines>(),
        ) else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let params = world.get_resource::<SimParams>();
        // Each fractal has its own pipeline with its perturbation step
        // compiled in. from_id maps a bad id to Mandelbrot, the same fallback
        // the CPU orbit code uses, so both halves of the perturbation
        // pipeline always iterate the same map.
        let wanted = Fractal::from_id(params.map_or(0, |p| p.0.fractal_type)).id() as usize;
        let id = pipelines.compute_pipelines[wanted];
        let pipeline = match cache.get_compute_pipeline(id) {
            Some(pipeline) => {
                self.last_ready.store(wanted, Ordering::Relaxed);
                pipeline
            }
            // Still compiling: run the previous fractal's pipeline this frame
            // (see `last_ready`) rather than freezing the cloud.
            None => {
                let prev = self.last_ready.load(Ordering::Relaxed);
                match cache.get_compute_pipeline(pipelines.compute_pipelines[prev]) {
                    Some(pipeline) => pipeline,
                    None => return Ok(()),
                }
            }
        };
        // Simulate only the active count (buffer holds up to MAX_PARTICLES).
        let count = params.map_or(0, |p| p.0.count).min(MAX_PARTICLES);
        if count == 0 {
            return Ok(());
        }
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_groups.compute, &[]);
        pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        Ok(())
    }
}

#[derive(Default)]
struct ParticleDrawNode;

impl ViewNode for ParticleDrawNode {
    // The marker reference doubles as the filter: the runner skips views
    // without it (the native present camera).
    type ViewQuery = (&'static ViewTarget, &'static ParticleCamera);

    fn run<'w>(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, _): bevy::ecs::query::QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), render_graph::NodeRunError> {
        let (Some(bind_groups), Some(pipelines)) = (
            world.get_resource::<ParticleBindGroups>(),
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
        let count = params.map_or(0, |p| p.0.count).min(MAX_PARTICLES);
        let decay = params.map_or(0.0, |p| p.0.trail_decay);
        let composite = params.is_some_and(|p| p.0.needs_composite());

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
            pass.draw(0..4, 0..count);
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
        bevy::asset::embedded_asset!(app, "shaders/common.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/particle_compute.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/particle_render.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/trail.wgsl");

        app.add_plugins(ExtractResourcePlugin::<SimParams>::default());
        app.add_plugins(ExtractResourcePlugin::<RefOrbit>::default());
        app.add_plugins(ExtractComponentPlugin::<ParticleCamera>::default());
        app.add_systems(
            Update,
            drop_particle_seed.run_if(resource_exists::<ParticleSeed>),
        );

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
        graph.add_node(ParticleComputeLabel, ParticleComputeNode::default());
        graph.add_node_edge(ParticleComputeLabel, CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ParticlePipelines>();
        render_app.init_resource::<ParticleUniform>();
        render_app.init_resource::<RefOrbitBuffer>();
    }
}

/// The machine that builds this may have no GPU, and a broken shader only
/// surfaces at pipeline creation - potentially only for one fractal type, only
/// once someone presses F. These run naga through the same preprocessor Bevy
/// uses, once per specialization, so a bad `#ifdef` or a typo in one fractal's
/// step fails the test run instead of the show.
#[cfg(test)]
mod shader_validation {
    use super::{Fractal, ParamsUniform, Particle};
    use bevy::render::render_resource::ShaderType as _;
    use naga_oil::compose::{
        ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue, ShaderType,
    };
    use std::collections::HashMap;

    const COMMON: &str = include_str!("shaders/common.wgsl");
    const COMPUTE: &str = include_str!("shaders/particle_compute.wgsl");

    /// One preprocessor+validate pass, shared by the positive and negative
    /// tests so both always exercise the exact descriptor the pipelines use,
    /// with the shared struct module registered the way the pipeline cache
    /// registers it.
    fn try_compile(source: &str, file_path: &str, defs: &[&str]) -> Result<(), String> {
        let shader_defs = defs
            .iter()
            .map(|d| (d.to_string(), ShaderDefValue::Bool(true)))
            .collect::<HashMap<_, _>>();
        let mut composer = Composer::default();
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: COMMON,
                file_path: "common.wgsl",
                ..Default::default()
            })
            .map(|_| ())
            .map_err(|e| e.emit_to_string(&composer))?;
        composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path,
                shader_type: ShaderType::Wgsl,
                shader_defs,
                additional_imports: &[],
            })
            .map(|_| ())
            .map_err(|e| e.emit_to_string(&composer))
    }

    fn compile(source: &str, file_path: &str, defs: &[&str]) {
        if let Err(e) = try_compile(source, file_path, defs) {
            panic!("{file_path} {defs:?} failed to compile:\n{e}");
        }
    }

    #[test]
    fn compute_shader_compiles_for_every_fractal() {
        for fractal in Fractal::ALL {
            compile(COMPUTE, "particle_compute.wgsl", &[fractal.shader_def()]);
        }
    }

    /// Without a FRACTAL_* def the specialized functions have no return path,
    /// so this must NOT compile - otherwise a missing def would silently ship
    /// a pipeline that never queued one.
    #[test]
    fn compute_shader_needs_a_fractal_def() {
        assert!(
            try_compile(COMPUTE, "particle_compute.wgsl", &[]).is_err(),
            "compute shader compiled with no fractal selected"
        );
    }

    #[test]
    fn render_and_trail_shaders_compile() {
        compile(
            include_str!("shaders/particle_render.wgsl"),
            "particle_render.wgsl",
            &[],
        );
        compile(include_str!("shaders/trail.wgsl"), "trail.wgsl", &[]);
    }

    /// The WGSL structs must be exactly the size of their Rust mirrors: a
    /// field added on one side only shifts every later uniform, which no
    /// validator catches and which renders as garbage, not as an error.
    #[test]
    fn shared_structs_match_rust_layouts() {
        // Composed as a top-level module, which has no import path to declare.
        let source = COMMON.replace("#define_import_path fractality::common", "");
        let module = Composer::default()
            .make_naga_module(NagaModuleDescriptor {
                source: &source,
                file_path: "common.wgsl",
                ..Default::default()
            })
            .expect("common.wgsl failed to compile");
        let size = |name: &str| {
            let (_, ty) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("no struct {name} in common.wgsl"));
            ty.inner.size(module.to_ctx()) as u64
        };
        assert_eq!(size("Particle"), std::mem::size_of::<Particle>() as u64);
        assert_eq!(size("Params"), ParamsUniform::min_size().get());
    }

    /// The compute shader hardcodes 1/log2(3) for Multibrot-3 smooth
    /// iteration (WGSL has no f64 math); a drift from the CPU value
    /// silently skews GPU bands against CPU smooth_iter.
    #[test]
    fn multibrot_inv_log2_power_matches_cpu() {
        let expected = format!("return {};", Fractal::Multibrot3.inv_log2_power());
        assert!(
            COMPUTE.contains(&expected),
            "particle_compute.wgsl inv_log2_power() drifted from the CPU value: expected `{expected}`"
        );
    }

    /// These tests pin naga_oil independently of Bevy. If Cargo ever resolves
    /// two copies, the tests validate with a different preprocessor than the
    /// runtime pipelines and prove nothing - and this build machine has no
    /// GPU to catch it at pipeline creation.
    #[test]
    fn naga_oil_resolves_to_one_version() {
        let lock = include_str!("../Cargo.lock");
        let copies = lock.matches("name = \"naga_oil\"").count();
        assert_eq!(
            copies, 1,
            "naga_oil resolved {copies} times; re-pin the dev-dependency to bevy's version"
        );
    }
}
