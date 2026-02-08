//! Point cloud data structures for 3D classification
//!
//! Provides standardized point cloud representation compatible with
//! ModelNet40, ScanObjectNN, and other benchmarks.

use rand::Rng;
use std::path::Path;

/// A 3D point with optional features
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Optional normal vector
    pub nx: Option<f64>,
    pub ny: Option<f64>,
    pub nz: Option<f64>,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point3D {
            x,
            y,
            z,
            nx: None,
            ny: None,
            nz: None,
        }
    }

    pub fn with_normal(x: f64, y: f64, z: f64, nx: f64, ny: f64, nz: f64) -> Self {
        Point3D {
            x,
            y,
            z,
            nx: Some(nx),
            ny: Some(ny),
            nz: Some(nz),
        }
    }

    /// Squared distance from origin
    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Distance from origin
    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Squared distance to another point
    pub fn distance_squared(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Distance to another point
    pub fn distance(&self, other: &Point3D) -> f64 {
        self.distance_squared(other).sqrt()
    }

    /// Convert to array [x, y, z]
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Create from array
    pub fn from_array(arr: [f64; 3]) -> Self {
        Point3D::new(arr[0], arr[1], arr[2])
    }
}

/// A labeled point cloud for classification
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Point3D>,
    pub label: Option<usize>,
    pub class_name: Option<String>,
}

impl PointCloud {
    pub fn new() -> Self {
        PointCloud {
            points: Vec::new(),
            label: None,
            class_name: None,
        }
    }

    pub fn from_points(points: Vec<Point3D>) -> Self {
        PointCloud {
            points,
            label: None,
            class_name: None,
        }
    }

    pub fn from_points_with_label(points: Vec<Point3D>, label: usize) -> Self {
        PointCloud {
            points,
            label: Some(label),
            class_name: None,
        }
    }

    pub fn from_points_with_class(points: Vec<Point3D>, label: usize, class_name: String) -> Self {
        PointCloud {
            points,
            label: Some(label),
            class_name: Some(class_name),
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute centroid of point cloud
    pub fn centroid(&self) -> Point3D {
        if self.is_empty() {
            return Point3D::new(0.0, 0.0, 0.0);
        }

        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;

        for p in &self.points {
            cx += p.x;
            cy += p.y;
            cz += p.z;
        }

        let n = self.len() as f64;
        Point3D::new(cx / n, cy / n, cz / n)
    }

    /// Center point cloud at origin
    pub fn center(&mut self) {
        let c = self.centroid();
        for p in &mut self.points {
            p.x -= c.x;
            p.y -= c.y;
            p.z -= c.z;
        }
    }

    /// Normalize to unit sphere (max distance from origin = 1)
    pub fn normalize(&mut self) {
        if self.is_empty() {
            return;
        }

        // Find max distance from origin
        let max_dist = self
            .points
            .iter()
            .map(|p| p.magnitude())
            .fold(0.0f64, f64::max);

        if max_dist > 1e-10 {
            for p in &mut self.points {
                p.x /= max_dist;
                p.y /= max_dist;
                p.z /= max_dist;
            }
        }
    }

    /// Center and normalize (standard preprocessing)
    pub fn preprocess(&mut self) {
        self.center();
        self.normalize();
    }

    /// Randomly sample n points (with replacement if n > len)
    pub fn sample(&self, n: usize) -> PointCloud {
        let mut rng = rand::thread_rng();
        let mut sampled_points = Vec::with_capacity(n);

        if self.is_empty() {
            return PointCloud::from_points_with_label(sampled_points, self.label.unwrap_or(0));
        }

        for _ in 0..n {
            let idx = rng.gen_range(0..self.len());
            sampled_points.push(self.points[idx]);
        }

        PointCloud {
            points: sampled_points,
            label: self.label,
            class_name: self.class_name.clone(),
        }
    }

    /// Farthest Point Sampling (FPS) - deterministic, better coverage
    pub fn farthest_point_sample(&self, n: usize) -> PointCloud {
        if self.is_empty() || n == 0 {
            return PointCloud::from_points_with_label(Vec::new(), self.label.unwrap_or(0));
        }

        let n = n.min(self.len());
        let mut sampled_indices = Vec::with_capacity(n);
        let mut distances: Vec<f64> = vec![f64::MAX; self.len()];

        // Start with random point
        let mut rng = rand::thread_rng();
        let first_idx = rng.gen_range(0..self.len());
        sampled_indices.push(first_idx);

        // Iteratively add farthest point
        for _ in 1..n {
            let last_point = &self.points[*sampled_indices.last().unwrap()];

            // Update distances
            for (i, p) in self.points.iter().enumerate() {
                let d = p.distance_squared(last_point);
                distances[i] = distances[i].min(d);
            }

            // Find farthest point
            let farthest_idx = distances
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            sampled_indices.push(farthest_idx);
        }

        let sampled_points: Vec<Point3D> = sampled_indices
            .iter()
            .map(|&i| self.points[i])
            .collect();

        PointCloud {
            points: sampled_points,
            label: self.label,
            class_name: self.class_name.clone(),
        }
    }

    /// Rotate around Z axis (in radians)
    pub fn rotate_z(&mut self, angle: f64) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for p in &mut self.points {
            let x = p.x * cos_a - p.y * sin_a;
            let y = p.x * sin_a + p.y * cos_a;
            p.x = x;
            p.y = y;
        }
    }

    /// Rotate around Y axis (in radians)
    pub fn rotate_y(&mut self, angle: f64) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for p in &mut self.points {
            let x = p.x * cos_a + p.z * sin_a;
            let z = -p.x * sin_a + p.z * cos_a;
            p.x = x;
            p.z = z;
        }
    }

    /// Rotate around X axis (in radians)
    pub fn rotate_x(&mut self, angle: f64) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for p in &mut self.points {
            let y = p.y * cos_a - p.z * sin_a;
            let z = p.y * sin_a + p.z * cos_a;
            p.y = y;
            p.z = z;
        }
    }

    /// Scale by factor
    pub fn scale(&mut self, factor: f64) {
        for p in &mut self.points {
            p.x *= factor;
            p.y *= factor;
            p.z *= factor;
        }
    }

    /// Add random jitter to points
    pub fn jitter(&mut self, std: f64) {
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0.0, std).unwrap();

        for p in &mut self.points {
            use rand_distr::Distribution;
            p.x += normal.sample(&mut rng);
            p.y += normal.sample(&mut rng);
            p.z += normal.sample(&mut rng);
        }
    }

    /// Convert to flat array [x0, y0, z0, x1, y1, z1, ...]
    pub fn to_flat_array(&self) -> Vec<f64> {
        let mut arr = Vec::with_capacity(self.len() * 3);
        for p in &self.points {
            arr.push(p.x);
            arr.push(p.y);
            arr.push(p.z);
        }
        arr
    }

    /// Create from flat array
    pub fn from_flat_array(arr: &[f64], label: Option<usize>) -> Self {
        assert!(arr.len() % 3 == 0, "Array length must be multiple of 3");
        let points: Vec<Point3D> = arr
            .chunks(3)
            .map(|c| Point3D::new(c[0], c[1], c[2]))
            .collect();
        PointCloud {
            points,
            label,
            class_name: None,
        }
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}

/// A split of a dataset (train, test, or val)
#[derive(Debug, Clone)]
pub struct DatasetSplit {
    pub samples: Vec<PointCloud>,
    pub name: String,
}

impl DatasetSplit {
    pub fn new(samples: Vec<PointCloud>, name: &str) -> Self {
        DatasetSplit {
            samples,
            name: name.to_string(),
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get labels as vector
    pub fn labels(&self) -> Vec<usize> {
        self.samples
            .iter()
            .map(|s| s.label.unwrap_or(0))
            .collect()
    }

    /// Count samples per class
    pub fn class_counts(&self, num_classes: usize) -> Vec<usize> {
        let mut counts = vec![0; num_classes];
        for sample in &self.samples {
            if let Some(label) = sample.label {
                if label < num_classes {
                    counts[label] += 1;
                }
            }
        }
        counts
    }
}

/// Trait for datasets
pub trait Dataset {
    /// Number of classes
    fn num_classes(&self) -> usize;

    /// Class names
    fn class_names(&self) -> &[String];

    /// Get training split
    fn train(&self) -> &DatasetSplit;

    /// Get test split
    fn test(&self) -> &DatasetSplit;

    /// Total number of samples
    fn total_samples(&self) -> usize {
        self.train().len() + self.test().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_operations() {
        let p1 = Point3D::new(1.0, 0.0, 0.0);
        let p2 = Point3D::new(0.0, 1.0, 0.0);

        assert!((p1.magnitude() - 1.0).abs() < 1e-10);
        assert!((p1.distance(&p2) - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_point_cloud_center() {
        let points = vec![
            Point3D::new(1.0, 1.0, 1.0),
            Point3D::new(3.0, 3.0, 3.0),
        ];
        let mut pc = PointCloud::from_points(points);
        pc.center();

        let centroid = pc.centroid();
        assert!(centroid.x.abs() < 1e-10);
        assert!(centroid.y.abs() < 1e-10);
        assert!(centroid.z.abs() < 1e-10);
    }

    #[test]
    fn test_point_cloud_normalize() {
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(10.0, 0.0, 0.0),
        ];
        let mut pc = PointCloud::from_points(points);
        pc.normalize();

        let max_dist = pc.points.iter().map(|p| p.magnitude()).fold(0.0f64, f64::max);
        assert!((max_dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fps_sampling() {
        // Create a simple point cloud
        let points: Vec<Point3D> = (0..100)
            .map(|i| {
                let t = i as f64 / 100.0 * 2.0 * std::f64::consts::PI;
                Point3D::new(t.cos(), t.sin(), 0.0)
            })
            .collect();
        let pc = PointCloud::from_points(points);

        let sampled = pc.farthest_point_sample(10);
        assert_eq!(sampled.len(), 10);

        // FPS should give well-distributed points
        // Check that no two points are too close
        for i in 0..sampled.len() {
            for j in (i + 1)..sampled.len() {
                let d = sampled.points[i].distance(&sampled.points[j]);
                assert!(d > 0.3, "FPS points should be well-separated");
            }
        }
    }
}
