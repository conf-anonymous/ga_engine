// src/rotor.rs
//! A 3-D rotor (unit even multivector) for rotations via Geometric Algebra.

use crate::{bivector::Bivector3, multivector::Multivector3, vector::Vec3};
use wide::f64x4;

/// A 3-D rotor (unit even multivector) for rotations.
#[derive(Clone, Debug, PartialEq)]
pub struct Rotor3 {
    inner: Multivector3, // full 8-component multivector
    axis: Vec3,          // normalized rotation axis
    w: f64,              // scalar part = cos(θ/2)
    s: f64,              // magnitude of bivector part = sin(θ/2)
}

impl Rotor3 {
    /// Construct a rotor from an axis (Vec3) and an angle (radians).
    #[inline(always)]
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let half = angle * 0.5;
        let w = half.cos();
        let s = half.sin();
        let axis_norm = axis.scale(1.0 / axis.norm());

        let mut m = Multivector3::zero();
        m.scalar = w;
        // bivector = -(axis_norm * s)
        m.bivector = Bivector3::new(-axis_norm.x * s, -axis_norm.y * s, -axis_norm.z * s);

        Rotor3 {
            inner: m,
            axis: axis_norm,
            w,
            s,
        }
    }

    /// Construct a rotor by exponentiating a pure bivector `b`: exp(b).
    #[inline(always)]
    pub fn from_bivector(b: Bivector3) -> Self {
        let phi = b.norm();
        if phi == 0.0 {
            // identity rotor
            return Rotor3::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.0);
        }
        let w = phi.cos();
        let s = phi.sin();
        // normalized bivector part = b/|b| * sin(|b|)
        let biv = b.scale(s / phi);

        let mut m = Multivector3::zero();
        m.scalar = w;
        m.bivector = biv;

        // rotation axis is dual of bivector: b = -(axis * sin)
        let axis = Vec3::new(b.xy / phi, b.yz / phi, b.zx / phi);

        Rotor3 {
            inner: m,
            axis,
            w,
            s,
        }
    }

    /// Reconstruct a `Rotor3` from a (unit‐even) multivector.
    #[inline(always)]
    pub fn from_multivector(m: Multivector3) -> Self {
        let w = m.scalar;
        let b = m.bivector;
        // s = |b|
        let s = (b.xy * b.xy + b.yz * b.yz + b.zx * b.zx).sqrt();
        // axis = -b/s  (since bivector = -(axis * s))
        let axis = if s == 0.0 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(-b.xy / s, -b.yz / s, -b.zx / s)
        };
        Rotor3 {
            inner: m,
            axis,
            w,
            s,
        }
    }

    /// Geometric‐product multiplication of two rotors, yielding a new rotor.
    #[inline(always)]
    pub fn mul(&self, other: &Rotor3) -> Rotor3 {
        let m = self.inner.gp(&other.inner);
        Rotor3::from_multivector(m)
    }

    /// Rotate a vector via the sandwich product: r * v * r⁻¹.
    #[inline(always)]
    pub fn rotate(&self, v: Vec3) -> Vec3 {
        let mv = Multivector3::from_vector(v);
        let inv = self.inner.reverse();
        self.inner.gp(&mv).gp(&inv).vector
    }

    /// Fast quaternion-style rotation (~20 flops), fully inlined.
    #[inline(always)]
    pub fn rotate_fast(&self, v: Vec3) -> Vec3 {
        let ax = self.axis.x;
        let ay = self.axis.y;
        let az = self.axis.z;
        let vx = v.x;
        let vy = v.y;
        let vz = v.z;

        // t = axis × v
        let tx = ay * vz - az * vy;
        let ty = az * vx - ax * vz;
        let tz = ax * vy - ay * vx;

        // u = axis × t
        let ux = ay * tz - az * ty;
        let uy = az * tx - ax * tz;
        let uz = ax * ty - ay * tx;

        let k1 = 2.0 * self.w * self.s;
        let k2 = 2.0 * self.s * self.s;

        Vec3::new(
            k2.mul_add(ux, k1.mul_add(tx, vx)),
            k2.mul_add(uy, k1.mul_add(ty, vy)),
            k2.mul_add(uz, k1.mul_add(tz, vz)),
        )
    }

    /// SIMD-4× rotate of four Vec3s in parallel using `wide::f64x4`.
    #[inline(always)]
    pub fn rotate_simd(&self, vs: [Vec3; 4]) -> [Vec3; 4] {
        let ax = f64x4::splat(self.axis.x);
        let ay = f64x4::splat(self.axis.y);
        let az = f64x4::splat(self.axis.z);

        let vx = f64x4::from([vs[0].x, vs[1].x, vs[2].x, vs[3].x]);
        let vy = f64x4::from([vs[0].y, vs[1].y, vs[2].y, vs[3].y]);
        let vz = f64x4::from([vs[0].z, vs[1].z, vs[2].z, vs[3].z]);

        let tx = ay * vz - az * vy;
        let ty = az * vx - ax * vz;
        let tz = ax * vy - ay * vx;

        let ux = ay * tz - az * ty;
        let uy = az * tx - ax * tz;
        let uz = ax * ty - ay * tx;

        let k1 = f64x4::splat(2.0 * self.w * self.s);
        let k2 = f64x4::splat(2.0 * self.s * self.s);

        let rx = k2.mul_add(ux, k1.mul_add(tx, vx));
        let ry = k2.mul_add(uy, k1.mul_add(ty, vy));
        let rz = k2.mul_add(uz, k1.mul_add(tz, vz));

        let ax_arr = rx.to_array();
        let ay_arr = ry.to_array();
        let az_arr = rz.to_array();

        [
            Vec3::new(ax_arr[0], ay_arr[0], az_arr[0]),
            Vec3::new(ax_arr[1], ay_arr[1], az_arr[1]),
            Vec3::new(ax_arr[2], ay_arr[2], az_arr[2]),
            Vec3::new(ax_arr[3], ay_arr[3], az_arr[3]),
        ]
    }

    /// SIMD-8× rotate by two 4-lane SIMD passes.
    #[inline(always)]
    pub fn rotate_simd8(&self, vs: [Vec3; 8]) -> [Vec3; 8] {
        let r0 = self.rotate_simd([vs[0], vs[1], vs[2], vs[3]]);
        let r1 = self.rotate_simd([vs[4], vs[5], vs[6], vs[7]]);
        [r0[0], r0[1], r0[2], r0[3], r1[0], r1[1], r1[2], r1[3]]
    }

    /// Expose the scalar part (cos θ/2).
    #[inline(always)]
    pub fn scalar(&self) -> f64 {
        self.w
    }

    /// Expose the bivector part (sin θ/2 · axis).
    #[inline(always)]
    pub fn bivector(&self) -> Bivector3 {
        self.inner.bivector
    }

    /// Access the full geometric product of two rotors as a Multivector3.
    #[inline(always)]
    pub fn gp(&self, other: &Rotor3) -> Multivector3 {
        self.inner.gp(&other.inner)
    }
}
