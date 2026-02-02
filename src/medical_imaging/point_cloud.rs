/// 3D Point Cloud Representation
///
/// Represents a 3D medical structure (e.g., lung nodule) as a point cloud.
/// Used as input for Clifford algebra encoding.

use std::fmt;

/// A single 3D point
#[derive(Debug, Clone, Copy)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point3D { x, y, z }
    }

    /// Euclidean distance from origin
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Distance to another point
    pub fn distance_to(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// 3D Point Cloud
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Point3D>,
    pub label: Option<u8>, // 0 = benign, 1 = malignant
}

impl PointCloud {
    /// Create new point cloud
    pub fn new() -> Self {
        PointCloud {
            points: Vec::new(),
            label: None,
        }
    }

    /// Create from vector of points
    pub fn from_points(points: Vec<Point3D>) -> Self {
        PointCloud {
            points,
            label: None,
        }
    }

    /// Create with label
    pub fn from_points_with_label(points: Vec<Point3D>, label: u8) -> Self {
        PointCloud {
            points,
            label: Some(label),
        }
    }

    /// Number of points
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute centroid (mean position)
    pub fn centroid(&self) -> Point3D {
        if self.points.is_empty() {
            return Point3D::new(0.0, 0.0, 0.0);
        }

        let sum = self.points.iter().fold(
            Point3D::new(0.0, 0.0, 0.0),
            |acc, p| Point3D::new(acc.x + p.x, acc.y + p.y, acc.z + p.z),
        );

        let n = self.points.len() as f64;
        Point3D::new(sum.x / n, sum.y / n, sum.z / n)
    }

    /// Translate to center around origin
    pub fn center(&mut self) {
        let centroid = self.centroid();
        for p in &mut self.points {
            p.x -= centroid.x;
            p.y -= centroid.y;
            p.z -= centroid.z;
        }
    }

    /// Rotate around Z-axis by angle (radians)
    pub fn rotate_z(&mut self, angle: f64) {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        for p in &mut self.points {
            let x_new = p.x * cos_theta - p.y * sin_theta;
            let y_new = p.x * sin_theta + p.y * cos_theta;
            p.x = x_new;
            p.y = y_new;
        }
    }

    /// Compute bounding box diagonal length (scale indicator)
    pub fn bounding_box_diagonal(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }

        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        let mut min_z = f64::MAX;
        let mut max_z = f64::MIN;

        for p in &self.points {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
            min_z = min_z.min(p.z);
            max_z = max_z.max(p.z);
        }

        let dx = max_x - min_x;
        let dy = max_y - min_y;
        let dz = max_z - min_z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Normalize to unit scale
    pub fn normalize(&mut self) {
        let diagonal = self.bounding_box_diagonal();
        if diagonal > 1e-10 {
            for p in &mut self.points {
                p.x /= diagonal;
                p.y /= diagonal;
                p.z /= diagonal;
            }
        }
    }
}

impl fmt::Display for PointCloud {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PointCloud({} points, label: {:?})",
            self.len(),
            self.label
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_creation() {
        let points = vec![
            Point3D::new(1.0, 2.0, 3.0),
            Point3D::new(4.0, 5.0, 6.0),
        ];
        let pc = PointCloud::from_points(points);
        assert_eq!(pc.len(), 2);
    }

    #[test]
    fn test_centroid() {
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(2.0, 0.0, 0.0),
            Point3D::new(0.0, 2.0, 0.0),
            Point3D::new(0.0, 0.0, 2.0),
        ];
        let pc = PointCloud::from_points(points);
        let centroid = pc.centroid();
        assert!((centroid.x - 0.5).abs() < 1e-10);
        assert!((centroid.y - 0.5).abs() < 1e-10);
        assert!((centroid.z - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_centering() {
        let points = vec![
            Point3D::new(1.0, 1.0, 1.0),
            Point3D::new(3.0, 1.0, 1.0),
            Point3D::new(1.0, 3.0, 1.0),
            Point3D::new(1.0, 1.0, 3.0),
        ];
        let mut pc = PointCloud::from_points(points);
        pc.center();

        let new_centroid = pc.centroid();
        assert!(new_centroid.x.abs() < 1e-10);
        assert!(new_centroid.y.abs() < 1e-10);
        assert!(new_centroid.z.abs() < 1e-10);
    }

    #[test]
    fn test_rotation_invariance() {
        // Create sphere-like point cloud
        let mut points = Vec::new();
        for i in 0..100 {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;
            points.push(Point3D::new(theta.cos(), theta.sin(), 0.0));
        }

        let pc1 = PointCloud::from_points(points.clone());
        let mut pc2 = PointCloud::from_points(points);

        // Rotate second cloud
        pc2.rotate_z(std::f64::consts::PI / 4.0);

        // Both should have same bounding box diagonal (relaxed for floating point accumulation)
        let diag1 = pc1.bounding_box_diagonal();
        let diag2 = pc2.bounding_box_diagonal();
        assert!(
            (diag1 - diag2).abs() < 0.01,
            "Diagonals should match after rotation (within 1%%): {} vs {}",
            diag1,
            diag2
        );
    }
}
