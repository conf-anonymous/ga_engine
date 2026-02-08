//! Data augmentation for point clouds
//!
//! Provides standard augmentations used in point cloud classification:
//! - Random rotation (typically around gravity axis)
//! - Random scaling
//! - Random jittering (Gaussian noise)
//! - Random point dropout

use super::point_cloud::{PointCloud, Point3D};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

/// Configuration for point cloud augmentation
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    /// Enable rotation augmentation
    pub rotate: bool,
    /// Rotation range in degrees (symmetric around 0)
    pub rotation_range: f64,
    /// Rotate around all axes or just Y (gravity)
    pub rotate_all_axes: bool,

    /// Enable scaling augmentation
    pub scale: bool,
    /// Scale range [min, max]
    pub scale_range: (f64, f64),

    /// Enable jittering (Gaussian noise)
    pub jitter: bool,
    /// Jitter standard deviation
    pub jitter_std: f64,
    /// Jitter clip value (max displacement)
    pub jitter_clip: f64,

    /// Enable random point dropout
    pub dropout: bool,
    /// Max fraction of points to drop
    pub dropout_ratio: f64,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        // Standard augmentation config from PointNet/PointNet++ papers
        AugmentationConfig {
            rotate: true,
            rotation_range: 180.0, // Full rotation around Y axis
            rotate_all_axes: false,

            scale: true,
            scale_range: (0.8, 1.2),

            jitter: true,
            jitter_std: 0.02,
            jitter_clip: 0.05,

            dropout: false,
            dropout_ratio: 0.0,
        }
    }
}

impl AugmentationConfig {
    /// No augmentation
    pub fn none() -> Self {
        AugmentationConfig {
            rotate: false,
            rotation_range: 0.0,
            rotate_all_axes: false,
            scale: false,
            scale_range: (1.0, 1.0),
            jitter: false,
            jitter_std: 0.0,
            jitter_clip: 0.0,
            dropout: false,
            dropout_ratio: 0.0,
        }
    }

    /// Light augmentation (for fine-tuning)
    pub fn light() -> Self {
        AugmentationConfig {
            rotate: true,
            rotation_range: 15.0,
            rotate_all_axes: false,
            scale: true,
            scale_range: (0.95, 1.05),
            jitter: true,
            jitter_std: 0.01,
            jitter_clip: 0.03,
            dropout: false,
            dropout_ratio: 0.0,
        }
    }

    /// Strong augmentation (for training from scratch)
    pub fn strong() -> Self {
        AugmentationConfig {
            rotate: true,
            rotation_range: 180.0,
            rotate_all_axes: true, // SO(3) augmentation
            scale: true,
            scale_range: (0.66, 1.5),
            jitter: true,
            jitter_std: 0.02,
            jitter_clip: 0.05,
            dropout: true,
            dropout_ratio: 0.1,
        }
    }

    /// ModelNet40 standard augmentation
    pub fn modelnet40() -> Self {
        AugmentationConfig {
            rotate: true,
            rotation_range: 180.0, // Around Y axis
            rotate_all_axes: false,
            scale: true,
            scale_range: (0.8, 1.25),
            jitter: true,
            jitter_std: 0.01,
            jitter_clip: 0.05,
            dropout: false,
            dropout_ratio: 0.0,
        }
    }

    /// ScanObjectNN augmentation (needs more robustness)
    pub fn scanobjectnn() -> Self {
        AugmentationConfig {
            rotate: true,
            rotation_range: 180.0,
            rotate_all_axes: false, // Keep gravity axis
            scale: true,
            scale_range: (0.8, 1.2),
            jitter: true,
            jitter_std: 0.02, // More noise for real-world data
            jitter_clip: 0.05,
            dropout: true,
            dropout_ratio: 0.05,
        }
    }
}

/// Point cloud augmentation engine
pub struct Augmentation {
    config: AugmentationConfig,
}

impl Augmentation {
    pub fn new(config: AugmentationConfig) -> Self {
        Augmentation { config }
    }

    /// Apply all configured augmentations to a point cloud
    pub fn apply(&self, pc: &PointCloud) -> PointCloud {
        let mut result = pc.clone();

        if self.config.dropout && self.config.dropout_ratio > 0.0 {
            self.apply_dropout(&mut result);
        }

        if self.config.rotate && self.config.rotation_range > 0.0 {
            if self.config.rotate_all_axes {
                self.apply_random_rotation_so3(&mut result);
            } else {
                self.apply_random_rotation_y(&mut result);
            }
        }

        if self.config.scale {
            self.apply_random_scale(&mut result);
        }

        if self.config.jitter && self.config.jitter_std > 0.0 {
            self.apply_jitter(&mut result);
        }

        result
    }

    /// Random rotation around Y axis (gravity)
    fn apply_random_rotation_y(&self, pc: &mut PointCloud) {
        let mut rng = rand::thread_rng();
        let angle_deg = rng.gen_range(-self.config.rotation_range..self.config.rotation_range);
        let angle_rad = angle_deg.to_radians();
        pc.rotate_y(angle_rad);
    }

    /// Random SO(3) rotation (all axes)
    fn apply_random_rotation_so3(&self, pc: &mut PointCloud) {
        let mut rng = rand::thread_rng();

        // Generate random rotation using axis-angle
        // Sample uniformly from SO(3) using quaternion method
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let u3: f64 = rng.gen();

        let q0 = (1.0 - u1).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
        let q1 = (1.0 - u1).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let q2 = u1.sqrt() * (2.0 * std::f64::consts::PI * u3).sin();
        let q3 = u1.sqrt() * (2.0 * std::f64::consts::PI * u3).cos();

        // Convert quaternion to rotation matrix and apply
        for p in &mut pc.points {
            let x = p.x;
            let y = p.y;
            let z = p.z;

            // Rotation matrix from quaternion
            let r00 = 1.0 - 2.0 * (q2 * q2 + q3 * q3);
            let r01 = 2.0 * (q1 * q2 - q0 * q3);
            let r02 = 2.0 * (q1 * q3 + q0 * q2);

            let r10 = 2.0 * (q1 * q2 + q0 * q3);
            let r11 = 1.0 - 2.0 * (q1 * q1 + q3 * q3);
            let r12 = 2.0 * (q2 * q3 - q0 * q1);

            let r20 = 2.0 * (q1 * q3 - q0 * q2);
            let r21 = 2.0 * (q2 * q3 + q0 * q1);
            let r22 = 1.0 - 2.0 * (q1 * q1 + q2 * q2);

            p.x = r00 * x + r01 * y + r02 * z;
            p.y = r10 * x + r11 * y + r12 * z;
            p.z = r20 * x + r21 * y + r22 * z;
        }
    }

    /// Random uniform scaling
    fn apply_random_scale(&self, pc: &mut PointCloud) {
        let mut rng = rand::thread_rng();
        let scale = rng.gen_range(self.config.scale_range.0..self.config.scale_range.1);
        pc.scale(scale);
    }

    /// Gaussian jittering with clipping
    fn apply_jitter(&self, pc: &mut PointCloud) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.config.jitter_std).unwrap();

        for p in &mut pc.points {
            let jx = normal.sample(&mut rng).clamp(-self.config.jitter_clip, self.config.jitter_clip);
            let jy = normal.sample(&mut rng).clamp(-self.config.jitter_clip, self.config.jitter_clip);
            let jz = normal.sample(&mut rng).clamp(-self.config.jitter_clip, self.config.jitter_clip);

            p.x += jx;
            p.y += jy;
            p.z += jz;
        }
    }

    /// Random point dropout
    fn apply_dropout(&self, pc: &mut PointCloud) {
        if pc.is_empty() {
            return;
        }

        let mut rng = rand::thread_rng();
        let drop_count = (pc.len() as f64 * self.config.dropout_ratio) as usize;

        if drop_count == 0 {
            return;
        }

        // Randomly remove points
        for _ in 0..drop_count {
            if pc.len() <= 1 {
                break;
            }
            let idx = rng.gen_range(0..pc.len());
            pc.points.swap_remove(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_cloud() -> PointCloud {
        let points: Vec<Point3D> = (0..100)
            .map(|i| {
                let t = i as f64 / 100.0;
                Point3D::new(t, t * 0.5, t * 0.25)
            })
            .collect();
        PointCloud::from_points(points)
    }

    #[test]
    fn test_no_augmentation() {
        let pc = make_test_cloud();
        let aug = Augmentation::new(AugmentationConfig::none());
        let result = aug.apply(&pc);

        assert_eq!(result.len(), pc.len());
        for (p1, p2) in result.points.iter().zip(pc.points.iter()) {
            assert!((p1.x - p2.x).abs() < 1e-10);
            assert!((p1.y - p2.y).abs() < 1e-10);
            assert!((p1.z - p2.z).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rotation_preserves_distances() {
        let pc = make_test_cloud();
        let config = AugmentationConfig {
            rotate: true,
            rotation_range: 90.0,
            rotate_all_axes: false,
            ..AugmentationConfig::none()
        };
        let aug = Augmentation::new(config);

        // Original centroid distance
        let orig_centroid = pc.centroid();
        let orig_dist: f64 = pc.points.iter().map(|p| p.distance(&orig_centroid)).sum();

        let result = aug.apply(&pc);

        // Rotated centroid distance should be similar
        let new_centroid = result.centroid();
        let new_dist: f64 = result.points.iter().map(|p| p.distance(&new_centroid)).sum();

        // Allow small numerical error
        assert!((orig_dist - new_dist).abs() < 1e-6);
    }

    #[test]
    fn test_scale_changes_size() {
        let pc = make_test_cloud();
        let config = AugmentationConfig {
            scale: true,
            scale_range: (2.0, 2.0), // Fixed 2x scale
            ..AugmentationConfig::none()
        };
        let aug = Augmentation::new(config);
        let result = aug.apply(&pc);

        // Points should be 2x farther from origin
        for (p1, p2) in result.points.iter().zip(pc.points.iter()) {
            assert!((p1.x - 2.0 * p2.x).abs() < 1e-10);
            assert!((p1.y - 2.0 * p2.y).abs() < 1e-10);
            assert!((p1.z - 2.0 * p2.z).abs() < 1e-10);
        }
    }

    #[test]
    fn test_jitter_changes_points() {
        let pc = make_test_cloud();
        let config = AugmentationConfig {
            jitter: true,
            jitter_std: 0.1,
            jitter_clip: 0.2,
            ..AugmentationConfig::none()
        };
        let aug = Augmentation::new(config);
        let result = aug.apply(&pc);

        // Points should be different (with high probability)
        let mut any_different = false;
        for (p1, p2) in result.points.iter().zip(pc.points.iter()) {
            if (p1.x - p2.x).abs() > 1e-10 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "Jitter should modify at least one point");

        // But not too different (within clip range)
        for (p1, p2) in result.points.iter().zip(pc.points.iter()) {
            assert!((p1.x - p2.x).abs() <= 0.2 + 1e-10);
            assert!((p1.y - p2.y).abs() <= 0.2 + 1e-10);
            assert!((p1.z - p2.z).abs() <= 0.2 + 1e-10);
        }
    }

    #[test]
    fn test_dropout_removes_points() {
        let pc = make_test_cloud();
        let config = AugmentationConfig {
            dropout: true,
            dropout_ratio: 0.5,
            ..AugmentationConfig::none()
        };
        let aug = Augmentation::new(config);
        let result = aug.apply(&pc);

        // Should have fewer points
        assert!(result.len() < pc.len());
        // But not zero
        assert!(result.len() > 0);
    }
}
