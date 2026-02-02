/// Synthetic 3D Shape Dataset Generator
///
/// Generates labeled 3D point clouds for training:
/// - Class 0: Spheres
/// - Class 1: Cubes
/// - Class 2: Pyramids
///
/// Used to validate geometric neural network before moving to real medical data.

use super::point_cloud::{Point3D, PointCloud};
use rand::Rng;
use rand::seq::SliceRandom;

/// Shape type for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeType {
    Sphere = 0,
    Cube = 1,
    Pyramid = 2,
}

impl ShapeType {
    pub fn label(&self) -> u8 {
        *self as u8
    }

    pub fn from_label(label: u8) -> Option<Self> {
        match label {
            0 => Some(ShapeType::Sphere),
            1 => Some(ShapeType::Cube),
            2 => Some(ShapeType::Pyramid),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ShapeType::Sphere => "Sphere",
            ShapeType::Cube => "Cube",
            ShapeType::Pyramid => "Pyramid",
        }
    }
}

/// Generate sphere point cloud
///
/// Creates points uniformly distributed on sphere surface.
/// Uses Fibonacci sphere algorithm for uniform distribution.
pub fn generate_sphere(num_points: usize, radius: f64) -> PointCloud {
    let mut points = Vec::with_capacity(num_points);
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;

    for i in 0..num_points {
        let theta = 2.0 * std::f64::consts::PI * (i as f64) / golden_ratio;
        let phi = (1.0 - 2.0 * (i as f64) / (num_points as f64 - 1.0)).acos();

        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();

        points.push(Point3D::new(x, y, z));
    }

    PointCloud::from_points_with_label(points, ShapeType::Sphere.label())
}

/// Generate cube point cloud
///
/// Creates points uniformly distributed on cube surface.
/// Cube has side length 2*size (centered at origin).
pub fn generate_cube(num_points: usize, size: f64) -> PointCloud {
    let mut rng = rand::thread_rng();
    let mut points = Vec::with_capacity(num_points);

    // 6 faces of the cube
    let points_per_face = num_points / 6;

    for face in 0..6 {
        for _ in 0..points_per_face {
            let u = rng.gen_range(-size..size);
            let v = rng.gen_range(-size..size);

            let point = match face {
                0 => Point3D::new(size, u, v),   // +X face
                1 => Point3D::new(-size, u, v),  // -X face
                2 => Point3D::new(u, size, v),   // +Y face
                3 => Point3D::new(u, -size, v),  // -Y face
                4 => Point3D::new(u, v, size),   // +Z face
                _ => Point3D::new(u, v, -size),  // -Z face
            };
            points.push(point);
        }
    }

    PointCloud::from_points_with_label(points, ShapeType::Cube.label())
}

/// Generate pyramid point cloud
///
/// Creates points on 4 triangular faces + square base.
/// Pyramid has base size 2*base_size, height = height.
pub fn generate_pyramid(num_points: usize, base_size: f64, height: f64) -> PointCloud {
    let mut rng = rand::thread_rng();
    let mut points = Vec::with_capacity(num_points);

    // 5 faces: 1 square base + 4 triangular sides
    let points_per_face = num_points / 5;

    // Base (square at z = 0)
    for _ in 0..points_per_face {
        let x = rng.gen_range(-base_size..base_size);
        let y = rng.gen_range(-base_size..base_size);
        points.push(Point3D::new(x, y, 0.0));
    }

    // 4 triangular faces
    // Apex at (0, 0, height)
    // Base corners at (±base_size, ±base_size, 0)

    for face in 0..4 {
        for _ in 0..points_per_face {
            let u = rng.gen::<f64>(); // Random barycentric coordinate
            let v = rng.gen::<f64>() * (1.0 - u);

            // Vertices of each triangular face
            let (v0, v1, v2) = match face {
                0 => (
                    Point3D::new(0.0, 0.0, height),
                    Point3D::new(base_size, base_size, 0.0),
                    Point3D::new(base_size, -base_size, 0.0),
                ),
                1 => (
                    Point3D::new(0.0, 0.0, height),
                    Point3D::new(base_size, -base_size, 0.0),
                    Point3D::new(-base_size, -base_size, 0.0),
                ),
                2 => (
                    Point3D::new(0.0, 0.0, height),
                    Point3D::new(-base_size, -base_size, 0.0),
                    Point3D::new(-base_size, base_size, 0.0),
                ),
                _ => (
                    Point3D::new(0.0, 0.0, height),
                    Point3D::new(-base_size, base_size, 0.0),
                    Point3D::new(base_size, base_size, 0.0),
                ),
            };

            // Barycentric interpolation
            let x = v0.x + u * (v1.x - v0.x) + v * (v2.x - v0.x);
            let y = v0.y + u * (v1.y - v0.y) + v * (v2.y - v0.y);
            let z = v0.z + u * (v1.z - v0.z) + v * (v2.z - v0.z);

            points.push(Point3D::new(x, y, z));
        }
    }

    PointCloud::from_points_with_label(points, ShapeType::Pyramid.label())
}

/// Generate random shape of specified type
pub fn generate_shape(shape_type: ShapeType, num_points: usize) -> PointCloud {
    match shape_type {
        ShapeType::Sphere => generate_sphere(num_points, 1.0),
        ShapeType::Cube => generate_cube(num_points, 1.0),
        ShapeType::Pyramid => generate_pyramid(num_points, 1.0, 1.5),
    }
}

/// Generate balanced dataset with equal samples per class
pub fn generate_dataset(
    samples_per_class: usize,
    points_per_sample: usize,
) -> Vec<PointCloud> {
    let mut dataset = Vec::new();

    for _ in 0..samples_per_class {
        dataset.push(generate_shape(ShapeType::Sphere, points_per_sample));
    }

    for _ in 0..samples_per_class {
        dataset.push(generate_shape(ShapeType::Cube, points_per_sample));
    }

    for _ in 0..samples_per_class {
        dataset.push(generate_shape(ShapeType::Pyramid, points_per_sample));
    }

    // Shuffle dataset
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    dataset.shuffle(&mut rng);

    dataset
}

/// Split dataset into train/test sets
pub fn train_test_split(
    dataset: Vec<PointCloud>,
    train_ratio: f64,
) -> (Vec<PointCloud>, Vec<PointCloud>) {
    let train_size = (dataset.len() as f64 * train_ratio) as usize;
    let (train, test) = dataset.split_at(train_size);
    (train.to_vec(), test.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_sphere() {
        let sphere = generate_sphere(100, 1.0);
        assert_eq!(sphere.len(), 100);
        assert_eq!(sphere.label, Some(0));

        // All points should be approximately radius 1.0 from origin
        for p in &sphere.points {
            let dist = p.magnitude();
            assert!((dist - 1.0).abs() < 0.01, "Distance: {}", dist);
        }
    }

    #[test]
    fn test_generate_cube() {
        let cube = generate_cube(120, 1.0);
        assert_eq!(cube.label, Some(1));

        // All points should be on cube surface (at least one coordinate = ±1.0)
        for p in &cube.points {
            let on_surface = p.x.abs() >= 0.99 || p.y.abs() >= 0.99 || p.z.abs() >= 0.99;
            assert!(on_surface, "Point not on surface: {:?}", p);
        }
    }

    #[test]
    fn test_generate_pyramid() {
        let pyramid = generate_pyramid(100, 1.0, 1.5);
        assert_eq!(pyramid.label, Some(2));
        assert_eq!(pyramid.len(), 100);

        // Check height range (base at z=0, apex at z=1.5)
        for p in &pyramid.points {
            assert!(p.z >= 0.0 && p.z <= 1.5, "Invalid z: {}", p.z);
        }
    }

    #[test]
    fn test_generate_dataset() {
        let dataset = generate_dataset(10, 50);
        assert_eq!(dataset.len(), 30); // 10 per class × 3 classes

        // Count labels
        let mut label_counts = [0; 3];
        for pc in &dataset {
            if let Some(label) = pc.label {
                label_counts[label as usize] += 1;
            }
        }

        // Should have 10 of each
        assert_eq!(label_counts, [10, 10, 10]);
    }

    #[test]
    fn test_train_test_split() {
        let dataset = generate_dataset(10, 50); // 30 total samples
        let (train, test) = train_test_split(dataset, 0.8);

        assert_eq!(train.len(), 24); // 80% of 30
        assert_eq!(test.len(), 6);   // 20% of 30
    }

    #[test]
    fn test_shape_distinguishability() {
        use super::super::clifford_encoding::encode_point_cloud;

        let sphere = generate_sphere(100, 1.0);
        let cube = generate_cube(120, 1.0);
        let pyramid = generate_pyramid(100, 1.0, 1.5);

        let mv_sphere = encode_point_cloud(&sphere);
        let mv_cube = encode_point_cloud(&cube);
        let mv_pyramid = encode_point_cloud(&pyramid);

        // Shapes should have different multivector encodings
        // (This is a sanity check - the NN will learn to distinguish them)
        println!("Sphere:  {}", mv_sphere);
        println!("Cube:    {}", mv_cube);
        println!("Pyramid: {}", mv_pyramid);

        // At least one component should be significantly different
        let mut different = false;
        for i in 0..8 {
            let diff_sc = (mv_sphere.components[i] - mv_cube.components[i]).abs();
            let diff_sp = (mv_sphere.components[i] - mv_pyramid.components[i]).abs();
            let diff_cp = (mv_cube.components[i] - mv_pyramid.components[i]).abs();

            if diff_sc > 0.1 || diff_sp > 0.1 || diff_cp > 0.1 {
                different = true;
                break;
            }
        }

        assert!(different, "Shapes should have distinguishable encodings");
    }
}
