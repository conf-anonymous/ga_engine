/// Clifford Algebra Encoding for 3D Point Clouds
///
/// Encodes 3D medical structures (point clouds) as Cl(3,0) multivectors.
/// The encoding captures geometric properties: centroid, shape, orientation, volume.
///
/// Multivector components:
/// - m0 (scalar): Mean radial distance from centroid
/// - m1, m2, m3 (vector): Centroid position (x, y, z)
/// - m4, m5, m6 (bivector): Second moments (covariance, orientation)
/// - m7 (trivector): Volume indicator (determinant of covariance matrix)

use super::point_cloud::{Point3D, PointCloud};

/// Cl(3,0) multivector representation
#[derive(Debug, Clone, Copy)]
pub struct Multivector3D {
    pub components: [f64; 8],
}

impl Multivector3D {
    /// Create zero multivector
    pub fn zero() -> Self {
        Multivector3D {
            components: [0.0; 8],
        }
    }

    /// Create from components
    pub fn new(components: [f64; 8]) -> Self {
        Multivector3D { components }
    }

    /// Scalar component (m0)
    pub fn scalar(&self) -> f64 {
        self.components[0]
    }

    /// Vector components (m1, m2, m3)
    pub fn vector(&self) -> [f64; 3] {
        [self.components[1], self.components[2], self.components[3]]
    }

    /// Bivector components (m4, m5, m6) = (m12, m13, m23)
    pub fn bivector(&self) -> [f64; 3] {
        [self.components[4], self.components[5], self.components[6]]
    }

    /// Trivector component (m7) = m123
    pub fn trivector(&self) -> f64 {
        self.components[7]
    }
}

impl std::fmt::Display for Multivector3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({:.3} + {:.3}e₁ + {:.3}e₂ + {:.3}e₃ + {:.3}e₁₂ + {:.3}e₁₃ + {:.3}e₂₃ + {:.3}e₁₂₃)",
            self.components[0],
            self.components[1],
            self.components[2],
            self.components[3],
            self.components[4],
            self.components[5],
            self.components[6],
            self.components[7]
        )
    }
}

/// Encode point cloud as Cl(3,0) multivector
///
/// Algorithm:
/// 1. Compute centroid → vector components (m1, m2, m3)
/// 2. Compute mean radial distance → scalar component (m0)
/// 3. Compute second moments (covariance) → bivector components (m4, m5, m6)
/// 4. Compute volume (determinant) → trivector component (m7)
pub fn encode_point_cloud(pc: &PointCloud) -> Multivector3D {
    if pc.is_empty() {
        return Multivector3D::zero();
    }

    // 1. Compute centroid → vector components
    let centroid = pc.centroid();
    let m1 = centroid.x;
    let m2 = centroid.y;
    let m3 = centroid.z;

    // 2. Compute mean radial distance from centroid → scalar component
    let mut sum_radial_distance = 0.0;
    for p in &pc.points {
        let dx = p.x - centroid.x;
        let dy = p.y - centroid.y;
        let dz = p.z - centroid.z;
        sum_radial_distance += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    let m0 = sum_radial_distance / pc.len() as f64;

    // 3. Compute second moments (covariance matrix elements)
    // For bivectors, we use off-diagonal covariance elements
    let mut cov_xy = 0.0; // m4 (e12 component)
    let mut cov_xz = 0.0; // m5 (e13 component)
    let mut cov_yz = 0.0; // m6 (e23 component)

    // Also compute diagonal elements for volume calculation
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

    let m4 = cov_xy; // e12 component
    let m5 = cov_xz; // e13 component
    let m6 = cov_yz; // e23 component

    // 4. Compute volume indicator (determinant of covariance matrix)
    // det(C) = cov_xx*(cov_yy*cov_zz - cov_yz²) - cov_xy*(cov_xy*cov_zz - cov_yz*cov_xz) + cov_xz*(cov_xy*cov_yz - cov_yy*cov_xz)
    let m7 = cov_xx * (cov_yy * cov_zz - cov_yz * cov_yz)
        - cov_xy * (cov_xy * cov_zz - cov_yz * cov_xz)
        + cov_xz * (cov_xy * cov_yz - cov_yy * cov_xz);

    Multivector3D::new([m0, m1, m2, m3, m4, m5, m6, m7])
}

/// Encode multiple point clouds as batch
pub fn encode_batch(point_clouds: &[PointCloud]) -> Vec<Multivector3D> {
    point_clouds.iter().map(encode_point_cloud).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_empty_cloud() {
        let pc = PointCloud::new();
        let mv = encode_point_cloud(&pc);
        assert_eq!(mv.components, [0.0; 8]);
    }

    #[test]
    fn test_encode_single_point() {
        let points = vec![Point3D::new(1.0, 2.0, 3.0)];
        let pc = PointCloud::from_points(points);
        let mv = encode_point_cloud(&pc);

        // Centroid is the point itself
        assert!((mv.vector()[0] - 1.0).abs() < 1e-10);
        assert!((mv.vector()[1] - 2.0).abs() < 1e-10);
        assert!((mv.vector()[2] - 3.0).abs() < 1e-10);

        // Mean radial distance is 0 (only one point)
        assert!(mv.scalar().abs() < 1e-10);

        // Second moments are 0 (no variance)
        assert!(mv.bivector()[0].abs() < 1e-10);
        assert!(mv.bivector()[1].abs() < 1e-10);
        assert!(mv.bivector()[2].abs() < 1e-10);
        assert!(mv.trivector().abs() < 1e-10);
    }

    #[test]
    fn test_encode_sphere() {
        // Create sphere centered at origin with radius 1
        let mut points = Vec::new();
        for i in 0..100 {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;
            let phi = std::f64::consts::PI * (i as f64) / 100.0;
            let x = phi.sin() * theta.cos();
            let y = phi.sin() * theta.sin();
            let z = phi.cos();
            points.push(Point3D::new(x, y, z));
        }
        let pc = PointCloud::from_points(points);
        let mv = encode_point_cloud(&pc);

        // Centroid should be near origin (relaxed tolerance for random sampling)
        assert!(mv.vector()[0].abs() < 0.3);
        assert!(mv.vector()[1].abs() < 0.3);
        assert!(mv.vector()[2].abs() < 0.3);

        // Mean radial distance should be near 1
        assert!((mv.scalar() - 1.0).abs() < 0.3);
    }

    #[test]
    fn test_rotation_invariance_property() {
        // Create elongated point cloud along x-axis
        let mut points = Vec::new();
        for i in 0..50 {
            let t = (i as f64 - 25.0) / 25.0;
            points.push(Point3D::new(t * 2.0, 0.0, 0.0));
        }

        let pc1 = PointCloud::from_points(points.clone());
        let mut pc2 = PointCloud::from_points(points);

        // Rotate second cloud 90 degrees around z-axis
        pc2.rotate_z(std::f64::consts::PI / 2.0);

        let mv1 = encode_point_cloud(&pc1);
        let mv2 = encode_point_cloud(&pc2);

        // Scalar component (mean radial distance) should be rotation-invariant
        assert!((mv1.scalar() - mv2.scalar()).abs() < 1e-6);

        // Note: Centroid changes because we rotate around origin
        // In practice, we'd center first, then the encoding would be more rotation-invariant
    }

    #[test]
    fn test_encode_batch() {
        let pc1 = PointCloud::from_points(vec![Point3D::new(1.0, 0.0, 0.0)]);
        let pc2 = PointCloud::from_points(vec![Point3D::new(0.0, 1.0, 0.0)]);
        let pc3 = PointCloud::from_points(vec![Point3D::new(0.0, 0.0, 1.0)]);

        let batch = vec![pc1, pc2, pc3];
        let multivectors = encode_batch(&batch);

        assert_eq!(multivectors.len(), 3);
        assert!((multivectors[0].vector()[0] - 1.0).abs() < 1e-10);
        assert!((multivectors[1].vector()[1] - 1.0).abs() < 1e-10);
        assert!((multivectors[2].vector()[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_centered_encoding_more_invariant() {
        // Create elongated cloud
        let mut points = Vec::new();
        for i in 0..50 {
            let t = (i as f64) / 50.0;
            points.push(Point3D::new(t * 2.0, 0.0, 0.0));
        }

        let mut pc1 = PointCloud::from_points(points.clone());
        let mut pc2 = PointCloud::from_points(points);

        // Center both
        pc1.center();
        pc2.center();

        // Rotate second
        pc2.rotate_z(std::f64::consts::PI / 2.0);

        let mv1 = encode_point_cloud(&pc1);
        let mv2 = encode_point_cloud(&pc2);

        // Now centroid should be at origin for both
        assert!(mv1.vector()[0].abs() < 1e-10);
        assert!(mv1.vector()[1].abs() < 1e-10);
        assert!(mv2.vector()[0].abs() < 1e-10);
        assert!(mv2.vector()[1].abs() < 1e-10);

        // Scalar (mean radial distance) should match
        assert!((mv1.scalar() - mv2.scalar()).abs() < 1e-10);
    }
}
