//! Double-double arithmetic: an unevaluated sum of two f64s giving ~31
//! significant digits. Used only for the view center and the CPU reference
//! orbit, where f64's ~16 digits capped the zoom at height ~1e-15. Every
//! per-frame center *step* (pan, anchored zoom, fly-to) is view-height sized
//! and stays plain f64; only the accumulated position needs the extra digits,
//! so the GPU and the rest of the sim never see this type.
//!
//! Classic error-free transformations (Dekker/Knuth): two_sum captures the
//! rounding error of an addition exactly, Dekker's split + two_prod do the
//! same for multiplication without relying on FMA (portable to wasm at full
//! speed). Values here stay within O(1e10), far from the split's overflow
//! range near 1e300.

use bevy::math::DVec2;

#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

/// Exact sum: s + e == a + b, with s = fl(a + b).
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// two_sum when |a| >= |b| is known.
#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// Dekker split: a == hi + lo with both halves 26-bit.
#[inline]
fn split(a: f64) -> (f64, f64) {
    let t = 134217729.0 * a; // 2^27 + 1
    let hi = t - (t - a);
    (hi, a - hi)
}

/// Exact product: p + e == a * b, with p = fl(a * b).
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

impl Dd {
    pub const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };

    #[inline]
    pub fn from_f64(v: f64) -> Dd {
        Dd { hi: v, lo: 0.0 }
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.hi
    }

    #[inline]
    pub fn abs(self) -> Dd {
        if self.hi < 0.0 || (self.hi == 0.0 && self.lo < 0.0) {
            -self
        } else {
            self
        }
    }
}

impl std::ops::Neg for Dd {
    type Output = Dd;
    #[inline]
    fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl std::ops::Add for Dd {
    type Output = Dd;
    #[inline]
    fn add(self, rhs: Dd) -> Dd {
        let (s, e) = two_sum(self.hi, rhs.hi);
        let (hi, lo) = quick_two_sum(s, e + self.lo + rhs.lo);
        Dd { hi, lo }
    }
}

impl std::ops::Add<f64> for Dd {
    type Output = Dd;
    #[inline]
    fn add(self, rhs: f64) -> Dd {
        let (s, e) = two_sum(self.hi, rhs);
        let (hi, lo) = quick_two_sum(s, e + self.lo);
        Dd { hi, lo }
    }
}

impl std::ops::Sub for Dd {
    type Output = Dd;
    #[inline]
    fn sub(self, rhs: Dd) -> Dd {
        self + (-rhs)
    }
}

impl std::ops::Mul for Dd {
    type Output = Dd;
    #[inline]
    fn mul(self, rhs: Dd) -> Dd {
        let (p, e) = two_prod(self.hi, rhs.hi);
        let (hi, lo) = quick_two_sum(p, e + self.hi * rhs.lo + self.lo * rhs.hi);
        Dd { hi, lo }
    }
}

impl std::ops::Mul<f64> for Dd {
    type Output = Dd;
    #[inline]
    fn mul(self, rhs: f64) -> Dd {
        let (p, e) = two_prod(self.hi, rhs);
        let (hi, lo) = quick_two_sum(p, e + self.lo * rhs);
        Dd { hi, lo }
    }
}

/// 2D double-double point: the view center and reference-orbit input.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct DdVec2 {
    pub x: Dd,
    pub y: Dd,
}

impl DdVec2 {
    #[inline]
    pub fn from_dvec2(v: DVec2) -> DdVec2 {
        DdVec2 {
            x: Dd::from_f64(v.x),
            y: Dd::from_f64(v.y),
        }
    }

    /// Lossy round to f64. Fine for small differences (the hi part carries
    /// full f64 precision of the value) - never use it to rebuild a center.
    #[inline]
    pub fn to_dvec2(self) -> DVec2 {
        DVec2::new(self.x.to_f64(), self.y.to_f64())
    }
}

impl std::ops::Sub for DdVec2 {
    type Output = DdVec2;
    #[inline]
    fn sub(self, rhs: DdVec2) -> DdVec2 {
        DdVec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

/// Center += f64 step. All navigation steps are view-height sized, so f64
/// carries them exactly enough; the accumulation is where DD matters.
impl std::ops::AddAssign<DVec2> for DdVec2 {
    #[inline]
    fn add_assign(&mut self, rhs: DVec2) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accumulate 1e6 tiny steps a plain f64 would swallow entirely, then
    /// verify the sum survived in the lo limb.
    #[test]
    fn accumulates_below_f64_ulp() {
        let mut c = Dd::from_f64(1.0);
        let step = 1e-25;
        for _ in 0..1_000_000 {
            c = c + step;
        }
        let sum = c - Dd::from_f64(1.0);
        let expected = 1e-19;
        assert!(
            (sum.to_f64() - expected).abs() < expected * 1e-9,
            "sum = {:e}",
            sum.to_f64()
        );
        // Plain f64 loses the steps completely: 1.0 + 1e-25 == 1.0.
        assert_eq!(1.0f64 + step, 1.0);
    }

    #[test]
    fn mul_keeps_extra_digits() {
        // (1 + 1e-20)^2 = 1 + 2e-20 + 1e-40; DD must hold the 2e-20 term.
        let a = Dd::from_f64(1.0) + 1e-20;
        let sq = a * a;
        let frac = sq - Dd::from_f64(1.0);
        assert!((frac.to_f64() - 2e-20).abs() < 1e-30, "frac = {:e}", frac.to_f64());
    }
}
