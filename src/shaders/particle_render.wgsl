// Fractality particle render shader.

#import fractality::common::{Particle, Params}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    // Flat: the color is a per-particle value, identical at all four corners
    // of the quad, so interpolating it across every covered fragment is pure
    // rasterizer work for a result that cannot change. Bit-identical output.
    @location(1) @interpolate(flat) color: vec3<f32>,
};

// Spectrum energy for a particle: iteration depth maps to one of the 16
// log-spaced bins (deep shell = bass, outer haze = treble). The one mapping
// is shared by the aurora palette and the spectrum-glow multiplier so both
// effects always light the same shells. The obvious spectrum[i / 4u][i % 4u]
// indexes a vec4 with a dynamic component, which naga lowers to a private
// scratch array round trip on several backends; this runs per vertex, four
// times per particle, so pick the lane with selects instead.
fn spectrum_for_band(band: f32, iter_f: f32) -> f32 {
    let d = clamp(band / iter_f, 0.0, 1.0);
    let i = u32(clamp((1.0 - d) * 15.99, 0.0, 15.0));
    let v = params.spectrum[i >> 2u];
    let j = i & 3u;
    return select(select(v.x, v.y, j == 1u), select(v.z, v.w, j == 3u), j >= 2u);
}

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
    // (which naga lowers to per-vertex private memory). Drawn as a 4-vertex
    // triangle strip per instance (strips restart between instances), so the
    // per-vertex color work below runs 4x per particle instead of the 6x a
    // triangle list cost. Vertices 0..3 walk (-1,-1),(1,-1),(-1,1),(1,1).
    // Culling is off, so winding is irrelevant.
    let corner = vec2<f32>(
        select(-1.0, 1.0, (vertex_index & 1u) == 1u),
        select(-1.0, 1.0, vertex_index >= 2u),
    );
    // Field-wise loads: this shader never reads `home`, and pulling the whole
    // 32-byte struct made each of the four vertices per particle fetch 8
    // bytes it throws away.
    let p_pos = particles[instance_index].pos;
    let center_clip = p_pos * params.world_to_clip.xy + params.world_to_clip.zw;

    // Off-screen instances leave before the palette math. Recycling only
    // reclaims a particle once it has drifted 2.5 view-extents out, and a
    // dive pushes the whole cloud outward, so a large slice of the draw is
    // instances whose color is computed purely for the rasterizer to throw
    // away. The test uses the quad CENTER plus its half-extent, so all four
    // vertices of an instance decide identically (a per-corner test would
    // fold only some vertices of a quad and tear the geometry), and it is
    // exact: a quad overlaps the viewport iff its center is within
    // 1 + particle_size. All four vertices collapse onto the same
    // off-viewport point, so the triangles are zero-area and raster nothing.
    // NaN positions fail every compare and fall through to the normal path,
    // where they produce a NaN clip position the rasterizer drops anyway.
    let bound = vec2<f32>(1.0) + params.particle_size;
    if (abs(center_clip.x) > bound.x || abs(center_clip.y) > bound.y) {
        // Only clip matters: the collapsed quad rasterizes nothing, so uv and
        // color are never read (var zero-initializes them).
        var culled: VsOut;
        culled.clip = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        return culled;
    }

    let p_vel = particles[instance_index].vel;
    let p_band = particles[instance_index].band;
    let p_hue = particles[instance_index].hue;
    let iter_f = f32(params.max_iter);

    let clip_xy = center_clip + corner * params.particle_size;

    // Screen-relative speed: world velocity scales with view_height (flow is a
    // fraction of view height), so normalize by it to keep color/brightness the
    // same at any zoom - otherwise zoomed-out looks bright/colorful and zoomed-in
    // looks dim and flat. (view_height = 2 / world_to_clip.y.)
    let view_h = 2.0 / params.world_to_clip.y;
    // Divide BEFORE length(): squaring a raw ~view_height-sized velocity
    // underflows f32 below height ~1e-19 and flushes speed to zero.
    let speed = length(p_vel / view_h);

    // Computed once for the two consumers below (aurora palette, spectrum
    // glow); the gate is the union of theirs, so other modes with audio off
    // still skip the bin math entirely.
    var spec = 0.0;
    if (params.color_mode == 4u || params.audio_aux.w > 0.01) {
        spec = spectrum_for_band(p_band, iter_f);
    }

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
            // Bass tightens/loosens the ring spacing subtly so the topography
            // pumps with the music; hue rides the audio palette spin.
            let stripe = pow(
                0.5 + 0.5 * cos(p_band * (2.2 + params.audio.x * 0.5) - params.time * 0.8),
                3.0,
            );
            let hue = fract(p_band * 0.013 + params.time * 0.012 + params.audio_hue);
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
            let ang = atan2(p_vel.y, p_vel.x) / 6.2831853;
            let hue = 0.55 + 0.18 * cos(6.2831853 * (ang + params.time * 0.02))
                + params.audio_hue * 0.25;
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
            let d = clamp(p_band / iter_f, 0.0, 1.0);
            let base = pow(d, 0.45) * 0.72;
            // Two striation frequencies (fine coals + broad waves) plus a
            // per-particle flicker with p_hue as a random phase, so nearby
            // particles on the same band don't pulse in lockstep.
            let ripple = 0.16 * cos(p_band * 1.7 - params.time * 0.7)
                + 0.08 * cos(p_band * 0.23 + params.time * 0.15);
            let flicker = 0.06 * cos(params.time * 2.5 + p_hue * 80.0);
            let t = clamp(base + ripple + flicker + speed * 1.2, 0.0, 1.0);
            let ramp = vec3<f32>(
                pow(t, 0.55),
                pow(t, 2.2) * 0.95,
                1.65 * t * pow(1.0 - t, 3.0) + pow(t, 6.0),
            );
            color = ramp * (0.4 + speed * 3.0);
        }
        case 4u: {
            // Audio aurora: the music owns the palette. Two anchor tones sit
            // half a wheel apart and swing hard with the band energies (bass
            // drags tone A through half the wheel, treble tone B through the
            // other half); a particle blends between them by speed, so shell
            // and streams pull toward opposite tones without ever collapsing
            // to one hue. On top of that, each particle's iteration depth
            // maps to one of the 16 spectrum bins and that bin's energy
            // detunes its hue and lights it up - the nested depth rings
            // become an equalizer wearing different colors per frequency.
            // Mid energy sends hue waves traveling inward across the depth
            // rings, and a beat kicks every hue a step around the wheel, so
            // the whole swarm visibly reacts to hits, not just brightens.
            let hue_a = params.audio.x * 0.5 + params.audio_hue;
            let hue_b = 0.5 + params.audio.z * 0.5 + params.audio_hue;
            let t = smoothstep(0.02, 0.3, speed);
            var hue = mix(hue_a, hue_b, t)
                + spec * 0.15
                + params.audio.y * 0.15 * sin(p_band * 0.35 - params.time * 3.0)
                + params.audio.w * 0.1;
            // Silence stays dim near-mono silver; sound saturates and the
            // particle's own spectrum bin drives most of its glow, so quiet
            // passages go dark ember and busy ones blaze ring by ring.
            let level = params.audio_aux.w;
            let sat = clamp(level * 2.0, 0.0, 1.0);
            let tone = mix(vec3<f32>(0.4, 0.45, 0.6), sinebow(fract(hue)), sat);
            color = tone
                * (0.2 + speed * 3.5 + spec * spec * (0.5 + level * 2.5)
                    + params.audio.w * 1.5);
        }
        default: {
            // Classic: band hue shifted by speed plus a slow global drift.
            // Flat floor at 0.25 keeps the slow particles on the boundary (the
            // fractal shape itself) visible; speed lifts the streams on top.
            let h = p_hue + speed * 0.95 + params.time * 0.015 + params.audio_hue;
            color = sinebow(h) * (0.25 + speed * 5.5);
        }
    }

    // Audio: beat flash (whole cloud blinks brighter, pushed slightly toward
    // white so it reads as a strobe, not just gain) and treble adds glitter
    // via a per-particle flicker phased by hue. Levels are zero with audio
    // off, so the guards only skip the work, never change the result.
    let beat = params.audio.w * params.audio_fx.y;
    if (beat > 0.001) {
        color = mix(color, vec3<f32>(length(color)), beat * 0.35) * (1.0 + beat * 0.8);
    }
    let tr = params.audio.z * params.audio_fx.y;
    if (tr > 0.02) {
        let glitter = max(0.0, cos(params.time * 40.0 + p_hue * 300.0));
        color *= 1.0 + tr * glitter * glitter * 0.8;
    }
    // Spectrum glow: each particle's iteration depth maps to a frequency
    // band - bass lights the deep shell filaments, treble the outer haze -
    // so the fractal becomes an equalizer shaped like itself.
    if (params.audio_aux.w > 0.01) {
        color *= 1.0 + spec * spec * 1.6 * params.audio_fx.z;
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
