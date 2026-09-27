//! The escape-time fractals, CPU side: which fractals exist and their
//! per-fractal constants (`Fractal`), the maps themselves, the smooth escape
//! field used for seeding and scene scans, and the double-double reference
//! orbit the GPU perturbs around. The GPU half of each formula lives in
//! particle_compute.wgsl behind the matching `Fractal::shader_def`.

use std::ops::{Add, Mul, Sub};

use bevy::math::DVec2;

use crate::dd::{Dd, DdVec2};

/// Fractal formula, cycled with F or picked from the menu. The discriminant is
/// the id the uniform and the menu carry.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Fractal {
    #[default]
    Mandelbrot,
    BurningShip,
    Tricorn,
    Multibrot3,
    Julia,
}

impl Fractal {
    /// Every fractal, in id order.
    pub const ALL: [Fractal; 5] = [
        Fractal::Mandelbrot,
        Fractal::BurningShip,
        Fractal::Tricorn,
        Fractal::Multibrot3,
        Fractal::Julia,
    ];

    /// Display names, indexed by id.
    pub const NAMES: [&'static str; 5] = [
        "mandelbrot",
        "burning ship",
        "tricorn",
        "multibrot-3",
        "julia",
    ];

    /// Out-of-range ids fall back to Mandelbrot, so every consumer of a raw
    /// id (the compute node reads it back out of the uniform) agrees on one
    /// formula.
    pub fn from_id(id: u32) -> Fractal {
        Self::ALL.get(id as usize).copied().unwrap_or_default()
    }

    pub fn id(self) -> u32 {
        self as u32
    }

    pub fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }

    /// The next fractal in the F-key cycle.
    pub fn next(self) -> Fractal {
        Self::from_id((self.id() + 1) % Self::ALL.len() as u32)
    }

    /// Compute-shader def that compiles this fractal's perturbation step. One
    /// pipeline per fractal: the step runs twice per iteration of the hottest
    /// loop in the app, so the formula is picked once at pipeline build time
    /// instead of branching on a uniform billions of times a frame.
    pub fn shader_def(self) -> &'static str {
        match self {
            Fractal::Mandelbrot => "FRACTAL_MANDELBROT",
            Fractal::BurningShip => "FRACTAL_SHIP",
            Fractal::Tricorn => "FRACTAL_TRICORN",
            Fractal::Multibrot3 => "FRACTAL_MULTIBROT3",
            Fractal::Julia => "FRACTAL_JULIA",
        }
    }

    /// Home view (center, height): startup, R reset, and fractal switches.
    pub const fn home_view(self) -> (DVec2, f64) {
        match self {
            Fractal::Mandelbrot => (DVec2::new(-0.55, 0.0), 2.7),
            Fractal::BurningShip => (DVec2::new(-0.4, -0.5), 3.0),
            Fractal::Tricorn => (DVec2::new(-0.3, 0.0), 3.4),
            Fractal::Multibrot3 | Fractal::Julia => (DVec2::ZERO, 3.0),
        }
    }

    /// Seed-sampling rectangle (x0, x1, y0, y1) covering the exterior
    /// boundary region. Coarse is fine: rejection keeps only boundary points
    /// and the GPU recycle resamples the live view within seconds anyway.
    pub fn spawn_rect(self) -> (f64, f64, f64, f64) {
        match self {
            Fractal::Mandelbrot => (-2.3, 0.85, -1.3, 1.3),
            Fractal::BurningShip => (-2.5, 1.6, -2.0, 1.0),
            Fractal::Tricorn => (-2.4, 1.6, -2.0, 2.0),
            Fractal::Multibrot3 => (-1.6, 1.6, -1.6, 1.6),
            Fractal::Julia => (-1.8, 1.8, -1.3, 1.3),
        }
    }

    /// 1/log2(power): smooth-iteration scale that keeps fractional bands
    /// continuous for maps of power != 2. The compute shader hardcodes the
    /// Multibrot-3 value; a test pins the literal to this.
    pub fn inv_log2_power(self) -> f64 {
        match self {
            Fractal::Multibrot3 => 1.0 / 3.0f64.log2(),
            _ => 1.0,
        }
    }

    /// One iteration of the map, z' = f(z) + c, generic over precision so
    /// the f64 samplers and the double-double reference orbit run literally
    /// the same formula. Julia shares Mandelbrot's map; only its start
    /// differs (see `start`).
    #[inline(always)]
    fn step<T: Real>(self, x: T, y: T, cx: T, cy: T) -> (T, T) {
        match self {
            Fractal::Mandelbrot | Fractal::Julia => (x * x - y * y + cx, x * y * 2.0 + cy),
            Fractal::BurningShip => {
                let (ax, ay) = (x.abs(), y.abs());
                (ax * ax - ay * ay + cx, ax * ay * 2.0 + cy)
            }
            Fractal::Tricorn => (x * x - y * y + cx, cy - x * y * 2.0),
            Fractal::Multibrot3 => (
                x * (x * x - y * y * 3.0) + cx,
                y * (x * x * 3.0 - y * y) + cy,
            ),
        }
    }

    /// Initial z and effective c for the point (x, y): Julia iterates the
    /// point itself under the fixed parameter `jc`; everything else iterates
    /// from 0 with the point as c.
    #[inline(always)]
    fn start<T: Real>(self, x: T, y: T, jc: (T, T)) -> ((T, T), (T, T)) {
        match self {
            Fractal::Julia => ((x, y), jc),
            _ => ((T::ZERO, T::ZERO), (x, y)),
        }
    }
}

/// The arithmetic the maps need, for f64 and double-double alike.
trait Real:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Mul<f64, Output = Self>
{
    const ZERO: Self;
    fn abs(self) -> Self;
}

impl Real for f64 {
    const ZERO: f64 = 0.0;
    fn abs(self) -> f64 {
        f64::abs(self)
    }
}

impl Real for Dd {
    const ZERO: Dd = Dd::ZERO;
    fn abs(self) -> Dd {
        Dd::abs(self)
    }
}

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

/// Exact membership in the Mandelbrot set's main cardioid or period-2 bulb,
/// which together hold ~90% of the set's area. Those points never escape, so
/// iterating one is the most expensive sample there is (the full max_iter)
/// and every seeder rejects it anyway.
fn in_main_bulbs(x: f64, y: f64) -> bool {
    let y2 = y * y;
    let xq = x - 0.25;
    let q = xq * xq + y2;
    q * (q + xq) <= 0.25 * y2 || (x + 1.0) * (x + 1.0) + y2 <= 0.0625
}

/// Smooth escape-time field on the CPU (f64). The formula and the escape
/// radius (256) must be identical to field() in the compute shader so that
/// CPU band values match GPU field values.
pub fn smooth_iter(x: f64, y: f64, max_iter: u32, fractal: Fractal) -> f32 {
    if fractal == Fractal::Mandelbrot && in_main_bulbs(x, y) {
        return max_iter as f32;
    }
    let ((mut zx, mut zy), (cx, cy)) = fractal.start(x, y, JULIA_C);
    for i in 0..max_iter {
        (zx, zy) = fractal.step(zx, zy, cx, cy);
        let m = zx * zx + zy * zy;
        if m > 256.0 {
            let nu = (0.5 * m.log2()).log2() * fractal.inv_log2_power();
            return i as f32 + 1.0 - nu as f32;
        }
    }
    max_iter as f32
}

/// Max reference-orbit length (also caps max_iter). One vec4<f32> per entry
/// (hi/lo pairs), so the whole GPU buffer is 256 KB. Sized so the depth ramp
/// (~8900 iterations at the height floor of 1e-28) fits with detail-slider
/// headroom. The real cost of a long orbit is the per-particle iteration
/// loop, which `iter_budget_count` in main.rs pays for by trading particle
/// count against depth.
pub const REF_ORBIT_CAP: usize = 16384;

/// Split an f64 into an f32 hi/lo pair: hi = rounded value, lo = the ~24 bits
/// of residual, together ~48 bits of the original.
#[inline]
fn split_f32(v: f64) -> (f32, f32) {
    let hi = v as f32;
    (hi, (v - hi as f64) as f32)
}

/// Iterate the map at the reference point in double-double (~31 digits, so
/// the orbit is exact for views down to height ~1e-28), storing Z_0..Z_n as
/// f32 hi/lo pairs `[hi.x, hi.y, lo.x, lo.y]` (see `RefOrbit` in
/// particles.rs) - perturbation needs the c behind the orbit at full
/// precision, the stored samples only well enough to survive the
/// close-approach cancellation in the shader. Stops at max_iter,
/// REF_ORBIT_CAP, or when the orbit diverges hard.
pub fn reference_orbit(
    c: DdVec2,
    max_iter: u32,
    fractal: Fractal,
    jc: (f64, f64),
) -> Vec<[f32; 4]> {
    let jc = (Dd::from_f64(jc.0), Dd::from_f64(jc.1));
    let ((mut zx, mut zy), (cx, cy)) = fractal.start(c.x, c.y, jc);
    let len = (max_iter as usize + 1).min(REF_ORBIT_CAP);
    let mut orbit = Vec::with_capacity(len);
    loop {
        let (hx, lx) = split_f32(zx.hi);
        let (hy, ly) = split_f32(zy.hi);
        orbit.push([hx, hy, lx, ly]);
        if orbit.len() >= len || zx.hi * zx.hi + zy.hi * zy.hi > 1e10 {
            return orbit;
        }
        (zx, zy) = fractal.step(zx, zy, cx, cy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The double-double instantiation of the map must track the f64 one: a
    /// broken Real impl (sign, abs) would silently render a different
    /// fractal at every depth.
    #[test]
    fn dd_step_matches_f64_step() {
        for fractal in Fractal::ALL {
            let ((mut zx, mut zy), (cx, cy)) = fractal.start(-0.16, 0.65, JULIA_C);
            let (mut dzx, mut dzy) = (Dd::from_f64(zx), Dd::from_f64(zy));
            let (dcx, dcy) = (Dd::from_f64(cx), Dd::from_f64(cy));
            for i in 0..60 {
                (zx, zy) = fractal.step(zx, zy, cx, cy);
                (dzx, dzy) = fractal.step(dzx, dzy, dcx, dcy);
                if zx * zx + zy * zy > 1e10 {
                    break;
                }
                let err = (zx - dzx.hi).abs().max((zy - dzy.hi).abs());
                assert!(err < 1e-9, "{fractal:?} iter {i}: err {err:e}");
            }
        }
    }

    /// The interior shortcut may only ever claim points that really never
    /// escape: sweep a grid over the set and check every point it claims
    /// survives a long plain iteration.
    #[test]
    fn main_bulbs_are_interior() {
        let mut claimed = 0;
        for iy in 0..200 {
            for ix in 0..300 {
                let x = -2.2 + ix as f64 * 0.01;
                let y = -1.2 + iy as f64 * 0.012;
                if !in_main_bulbs(x, y) {
                    continue;
                }
                claimed += 1;
                let (mut zx, mut zy) = (0.0f64, 0.0f64);
                for _ in 0..20_000 {
                    (zx, zy) = (zx * zx - zy * zy + x, 2.0 * zx * zy + y);
                }
                assert!(zx * zx + zy * zy <= 4.0, "({x}, {y}) escaped");
            }
        }
        // Sanity: the shortcut covers a real share of the grid.
        assert!(claimed > 5000, "only {claimed} points claimed");
    }

    #[test]
    fn ids_round_trip() {
        for (i, fractal) in Fractal::ALL.into_iter().enumerate() {
            assert_eq!(fractal.id(), i as u32);
            assert_eq!(Fractal::from_id(i as u32), fractal);
        }
        assert_eq!(Fractal::from_id(99), Fractal::Mandelbrot);
    }
}
