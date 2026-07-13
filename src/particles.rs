// The ShaderType derive generates per-field check fns that trip dead_code.
#![allow(dead_code)]

use std::sync::Arc;

use bevy::asset::DirectAssetAccessExt;
use bevy::math::DVec2;
use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::prelude::*;
use bevy::render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    graph::CameraDriverLabel,
    render_graph::{self, RenderGraph, RenderGraphApp, RenderLabel, ViewNode, ViewNodeRunner},
    render_resource::binding_types::{
        storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer,
    },
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
    view::ViewTarget,
    Extract, ExtractSchedule, Render, RenderApp, RenderSet,
};
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
}

/// Max reference-orbit length (also caps max_iter). One vec2<f32> per entry.
pub const REF_ORBIT_CAP: usize = 2048;

/// GPU particle buffer capacity. The full buffer is always allocated; the live
/// `count` uniform caps how many are actually simulated/drawn, so the settings
/// menu can change particle count instantly with no buffer reallocation.
pub const MAX_PARTICLES: u32 = 12_000_000;

/// CPU-computed f64 reference orbit at the view center, stored as f32 pairs.
/// Perturbation keeps full f32 precision because particle deltas stay small.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct RefOrbit(pub Vec<[f32; 2]>);

/// Iterate z -> z^2 + c at the reference point c in f64, storing Z_0..Z_n as
/// f32 pairs. Stops at max_iter, REF_ORBIT_CAP, or when the orbit diverges hard.
pub fn reference_orbit(cx: f64, cy: f64, max_iter: u32) -> Vec<[f32; 2]> {
    let mut v = Vec::with_capacity((max_iter as usize + 1).min(REF_ORBIT_CAP));
    v.push([0.0, 0.0]); // Z_0 = 0
    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    for _ in 0..max_iter {
        let nx = zx * zx - zy * zy + cx;
        let ny = 2.0 * zx * zy + cy;
        zx = nx;
        zy = ny;
        v.push([zx as f32, zy as f32]);
        if zx * zx + zy * zy > 1e10 || v.len() >= REF_ORBIT_CAP {
            break;
        }
    }
    v
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
fn smooth_iter(x: f64, y: f64, max_iter: u32) -> f32 {
    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    for i in 0..max_iter {
        let nx = zx * zx - zy * zy + x;
        let ny = 2.0 * zx * zy + y;
        zx = nx;
        zy = ny;
        let m = zx * zx + zy * zy;
        if m > 256.0 {
            return i as f32 + 1.0 - ((0.5 * m.log2()).log2()) as f32;
        }
    }
    max_iter as f32
}

pub fn generate_particles(count: usize, max_iter: u32, center: DVec2) -> Vec<Particle> {
    (0..count)
        .into_par_iter()
        .map(|i| {
            let mut rng =
                fastrand::Rng::with_seed(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1));
            loop {
                let x = -2.3 + rng.f64() * (0.85 - (-2.3));
                let y = -1.3 + rng.f64() * (1.3 - (-1.3));
                let f = smooth_iter(x, y, max_iter);
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
            size: (REF_ORBIT_CAP * std::mem::size_of::<[f32; 2]>()) as u64,
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
    compute_pipeline: CachedComputePipelineId,
    render_pipeline: CachedRenderPipelineId,
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
        let compute_shader: Handle<Shader> =
            world.load_asset("embedded://fractality/shaders/particle_compute.wgsl");
        let render_shader: Handle<Shader> =
            world.load_asset("embedded://fractality/shaders/particle_render.wgsl");
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
        Self {
            compute_layout,
            render_layout,
            compute_pipeline,
            render_pipeline,
        }
    }
}

fn extract_particle_buffers(
    mut commands: Commands,
    existing: Option<Res<ParticleBuffers>>,
    seed: Extract<Option<Res<ParticleSeed>>>,
    device: Res<RenderDevice>,
) {
    if existing.is_some() {
        return;
    }
    let Some(seed) = seed.as_ref() else {
        return;
    };
    // Allocate at full capacity; front-load the seeded particles, zero the rest.
    // The live `count` uniform gates how many are simulated/drawn, so the count
    // can be raised at runtime with no reallocation. Zeroed tail particles sit at
    // the origin and get pulled into view by the recycle trickle as count rises.
    let mut data = vec![bytemuck::Zeroable::zeroed(); MAX_PARTICLES as usize];
    let n = seed.0.len().min(data.len());
    data[..n].copy_from_slice(&seed.0[..n]);
    let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("particle_buffer"),
        usage: BufferUsages::STORAGE,
        contents: bytemuck::cast_slice(&data),
    });
    commands.insert_resource(ParticleBuffers {
        buffer,
        count: MAX_PARTICLES,
    });
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
) {
    let Some(buffers) = buffers else {
        return;
    };
    let Some(params) = params else {
        return;
    };
    if let Some(ref_orbit) = ref_orbit {
        let bytes = bytemuck::cast_slice(ref_orbit.0.as_slice());
        if !bytes.is_empty() {
            queue.write_buffer(&ref_buffer.0, 0, bytes);
        }
    }
    uniform.0.set(params.0);
    uniform.0.write_buffer(&device, &queue);
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
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("particle_draw_pass"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        // Draw only the active count (buffer holds up to MAX_PARTICLES).
        let count = world
            .get_resource::<SimParams>()
            .map_or(0, |p| p.0.count)
            .min(buffers.count);
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &bind_groups.render, &[]);
        pass.draw(0..6, 0..count);
        Ok(())
    }
}

pub struct ParticlePlugin;

impl Plugin for ParticlePlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "shaders/particle_compute.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/particle_render.wgsl");

        app.add_plugins(ExtractResourcePlugin::<SimParams>::default());
        app.add_plugins(ExtractResourcePlugin::<RefOrbit>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(ExtractSchedule, extract_particle_buffers)
            .add_systems(
                Render,
                prepare_particle_bind_groups.in_set(RenderSet::PrepareBindGroups),
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
