// src/ops/projection.rs
//! Vector projection, rejection, and plane projection operations.

use crate::vector::Vec3;

/// Trait for projecting and rejecting Vec3s onto axes and planes.
pub trait Vec3Projection {
    /// Project this vector onto `axis`.
    fn project_onto(&self, axis: &Vec3) -> Vec3;

    /// Reject (component orthogonal to `axis`).
    fn reject_from(&self, axis: &Vec3) -> Vec3;

    /// Project this vector onto the plane with normal `normal`.
    fn project_onto_plane(&self, normal: &Vec3) -> Vec3;
}

impl Vec3Projection for Vec3 {
    #[inline(always)]
    fn project_onto(&self, axis: &Vec3) -> Vec3 {
        let denom = axis.dot(axis);
        if denom == 0.0 {
            return Vec3::default();
        }
        let scale = self.dot(axis) / denom;
        *axis * scale
    }

    #[inline(always)]
    fn reject_from(&self, axis: &Vec3) -> Vec3 {
        *self - self.project_onto(axis)
    }

    #[inline(always)]
    fn project_onto_plane(&self, normal: &Vec3) -> Vec3 {
        *self - self.project_onto(normal)
    }
}

/// Free-function wrappers for convenience and for importing in tests.
/// Project `v` onto `axis`.
pub fn project_onto(v: &Vec3, axis: &Vec3) -> Vec3 {
    v.project_onto(axis)
}

/// Reject `v` from `axis`.
pub fn reject_from(v: &Vec3, axis: &Vec3) -> Vec3 {
    v.reject_from(axis)
}

/// Project `v` onto the plane with normal `normal`.
pub fn project_onto_plane(v: &Vec3, normal: &Vec3) -> Vec3 {
    v.project_onto_plane(normal)
}
