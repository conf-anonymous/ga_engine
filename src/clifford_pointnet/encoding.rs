//! Point cloud encoding to Cl(3,0) multivectors
//!
//! This module provides functions to encode 3D point clouds as
//! Clifford algebra multivectors suitable for CliffordPointNet.
//!
//! Two encoding strategies:
//! 1. Point-wise: Each point → one multivector (N points → N multivectors)
//! 2. Global: Entire point cloud → one summary multivector

use super::multivector::Multivector;
use crate::datasets::point_cloud::{Point3D, PointCloud};

/// Encode a single 3D point as a Cl(3,0) vector multivector
///
/// The point (x, y, z) is encoded as: x·e₁ + y·e₂ + z·e₃
#[inline]
pub fn encode_point(p: &Point3D) -> Multivector {
    Multivector::vector(p.x, p.y, p.z)
}

/// Encode all points in a point cloud as individual multivectors
///
/// Returns N multivectors for N points
pub fn encode_points(pc: &PointCloud) -> Vec<Multivector> {
    pc.points.iter().map(encode_point).collect()
}

/// Encode point cloud as a single summary multivector
///
/// This encoding captures geometric properties of the entire point cloud:
/// - Scalar (m₀): Mean radial distance from centroid
/// - Vector (m₁, m₂, m₃): Centroid position
/// - Bivector (m₄, m₅, m₆): Second moments (orientation/shape)
/// - Trivector (m₇): Volume indicator (det of covariance)
pub fn encode_point_cloud(pc: &PointCloud) -> Multivector {
    if pc.is_empty() {
        return Multivector::zero();
    }

    // Compute centroid
    let centroid = pc.centroid();
    let m1 = centroid.x;
    let m2 = centroid.y;
    let m3 = centroid.z;

    // Compute mean radial distance (scalar component)
    let mut sum_radial = 0.0;
    for p in &pc.points {
        let dx = p.x - centroid.x;
        let dy = p.y - centroid.y;
        let dz = p.z - centroid.z;
        sum_radial += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    let m0 = sum_radial / pc.len() as f64;

    // Compute covariance matrix elements
    let mut cov_xy = 0.0;
    let mut cov_xz = 0.0;
    let mut cov_yz = 0.0;
    let mut cov_xx = 0.0;
    let mut cov_yy = 0.0;
    let mut cov_zz = 0.0;

    for p in &pc.points {
        let dx = p.x - centroid.x;
        let dy = p.y - centroid.y;
        let dz = p.z - centroid.z;

        cov_xy += dx * dy;
        cov_xz += dx * dz;
        cov_yz += dy * dz;
        cov_xx += dx * dx;
        cov_yy += dy * dy;
        cov_zz += dz * dz;
    }

    let n = pc.len() as f64;
    cov_xy /= n;
    cov_xz /= n;
    cov_yz /= n;
    cov_xx /= n;
    cov_yy /= n;
    cov_zz /= n;

    // Bivector components (off-diagonal covariance)
    let m4 = cov_yz; // e23 component
    let m5 = cov_xz; // e31 component (note: this is actually e13, stored as -e31)
    let m6 = cov_xy; // e12 component

    // Trivector component (determinant of covariance = volume indicator)
    let m7 = cov_xx * (cov_yy * cov_zz - cov_yz * cov_yz)
        - cov_xy * (cov_xy * cov_zz - cov_yz * cov_xz)
        + cov_xz * (cov_xy * cov_yz - cov_yy * cov_xz);

    Multivector::new([m0, m1, m2, m3, m4, m5, m6, m7])
}

/// Encode point cloud with augmented features
///
/// In addition to the basic encoding, computes additional geometric features:
/// - Principal axis directions
/// - Eccentricity
/// - Surface area estimate
pub fn encode_point_cloud_augmented(pc: &PointCloud) -> Vec<Multivector> {
    if pc.is_empty() {
        return vec![Multivector::zero()];
    }

    let mut features = Vec::new();

    // Basic encoding
    features.push(encode_point_cloud(pc));

    // Compute additional statistics
    let centroid = pc.centroid();

    // Min/max bounding box
    let (mut min_x, mut max_x) = (f64::MAX, f64::MIN);
    let (mut min_y, mut max_y) = (f64::MAX, f64::MIN);
    let (mut min_z, mut max_z) = (f64::MAX, f64::MIN);

    for p in &pc.points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
        min_z = min_z.min(p.z);
        max_z = max_z.max(p.z);
    }

    // Bounding box dimensions as vector
    let bbox_size = Multivector::vector(
        max_x - min_x,
        max_y - min_y,
        max_z - min_z,
    );
    features.push(bbox_size);

    // Aspect ratios encoded in bivector
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let dz = max_z - min_z;
    let max_dim = dx.max(dy).max(dz).max(1e-10);

    let aspect = Multivector::new([
        1.0,                    // scalar: normalized
        dx / max_dim,           // x aspect
        dy / max_dim,           // y aspect
        dz / max_dim,           // z aspect
        dy * dz / (max_dim * max_dim), // yz area ratio
        dx * dz / (max_dim * max_dim), // xz area ratio
        dx * dy / (max_dim * max_dim), // xy area ratio
        dx * dy * dz / (max_dim * max_dim * max_dim), // volume ratio
    ]);
    features.push(aspect);

    features
}

/// Batch encode multiple point clouds
pub fn encode_batch(point_clouds: &[PointCloud]) -> Vec<Vec<Multivector>> {
    point_clouds
        .iter()
        .map(|pc| encode_points(pc))
        .collect()
}

/// Batch encode as summary multivectors
pub fn encode_batch_summary(point_clouds: &[PointCloud]) -> Vec<Multivector> {
    point_clouds
        .iter()
        .map(encode_point_cloud)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_point() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        let mv = encode_point(&p);

        assert!((mv.components[1] - 1.0).abs() < 1e-10);
        assert!((mv.components[2] - 2.0).abs() < 1e-10);
        assert!((mv.components[3] - 3.0).abs() < 1e-10);
        assert!(mv.components[0].abs() < 1e-10); // scalar should be 0
    }

    #[test]
    fn test_encode_point_cloud() {
        // Create a simple centered point cloud
        let points = vec![
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(-1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, -1.0, 0.0),
        ];
        let pc = PointCloud::from_points(points);
        let mv = encode_point_cloud(&pc);

        // Centroid should be at origin
        assert!(mv.components[1].abs() < 1e-10);
        assert!(mv.components[2].abs() < 1e-10);
        assert!(mv.components[3].abs() < 1e-10);

        // Mean radial distance should be 1.0
        assert!((mv.components[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_encode_points() {
        let points = vec![
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, 0.0, 1.0),
        ];
        let pc = PointCloud::from_points(points);
        let mvs = encode_points(&pc);

        assert_eq!(mvs.len(), 3);
        assert!((mvs[0].components[1] - 1.0).abs() < 1e-10);
        assert!((mvs[1].components[2] - 1.0).abs() < 1e-10);
        assert!((mvs[2].components[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_invariance() {
        // Create a point cloud
        let points = vec![
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 2.0, 0.0),
            Point3D::new(0.0, 0.0, 3.0),
        ];
        let mut pc1 = PointCloud::from_points(points.clone());
        let mut pc2 = PointCloud::from_points(points);

        // Center both
        pc1.center();
        pc2.center();

        // Rotate one
        pc2.rotate_y(std::f64::consts::PI / 4.0);

        let mv1 = encode_point_cloud(&pc1);
        let mv2 = encode_point_cloud(&pc2);

        // Scalar component (mean radial distance) should be invariant
        assert!((mv1.components[0] - mv2.components[0]).abs() < 1e-6);
    }
}
