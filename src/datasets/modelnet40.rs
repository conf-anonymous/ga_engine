//! ModelNet40 Dataset Loader
//!
//! Loads the ModelNet40 dataset for 3D point cloud classification.
//!
//! Dataset: https://modelnet.cs.princeton.edu/
//!
//! The dataset contains 12,311 CAD models from 40 object categories:
//! - Training: 9,840 samples
//! - Test: 2,468 samples
//!
//! Expected directory structure:
//! ```
//! modelnet40/
//! ├── airplane/
//! │   ├── train/
//! │   │   ├── airplane_0001.off
//! │   │   └── ...
//! │   └── test/
//! │       ├── airplane_0627.off
//! │       └── ...
//! ├── bathtub/
//! │   └── ...
//! └── ...
//! ```
//!
//! We also support pre-sampled point cloud formats (.txt, .npy, .pts)
//! commonly distributed as ModelNet40 variants.

use super::point_cloud::{Dataset, DatasetSplit, Point3D, PointCloud};
use rand::Rng;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// ModelNet40 class names in alphabetical order (standard ordering)
pub const MODELNET40_CLASSES: [&str; 40] = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
];

/// Configuration for loading ModelNet40
#[derive(Debug, Clone)]
pub struct ModelNet40Config {
    /// Path to ModelNet40 root directory
    pub root_path: PathBuf,
    /// Number of points to sample per model
    pub num_points: usize,
    /// Use farthest point sampling (better coverage) vs random sampling
    pub use_fps: bool,
    /// Preprocess (center + normalize) point clouds
    pub preprocess: bool,
    /// Format of the data files
    pub format: ModelNet40Format,
    /// Limit number of samples (for debugging)
    pub max_samples: Option<usize>,
}

/// Format of ModelNet40 data files
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelNet40Format {
    /// Original OFF mesh files (need to sample points from surface)
    Off,
    /// Pre-sampled point clouds in text format (x y z per line)
    Txt,
    /// Pre-sampled with normals (x y z nx ny nz per line)
    TxtWithNormals,
}

impl Default for ModelNet40Config {
    fn default() -> Self {
        ModelNet40Config {
            root_path: PathBuf::from("data/modelnet40"),
            num_points: 1024,
            use_fps: true,
            preprocess: true,
            format: ModelNet40Format::Txt,
            max_samples: None,
        }
    }
}

impl ModelNet40Config {
    pub fn new<P: AsRef<Path>>(root_path: P) -> Self {
        ModelNet40Config {
            root_path: root_path.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    pub fn with_num_points(mut self, n: usize) -> Self {
        self.num_points = n;
        self
    }

    pub fn with_format(mut self, format: ModelNet40Format) -> Self {
        self.format = format;
        self
    }

    pub fn with_max_samples(mut self, max: usize) -> Self {
        self.max_samples = Some(max);
        self
    }
}

/// ModelNet40 dataset
pub struct ModelNet40 {
    train_split: DatasetSplit,
    test_split: DatasetSplit,
    class_names: Vec<String>,
    config: ModelNet40Config,
}

impl ModelNet40 {
    /// Load ModelNet40 dataset from disk
    pub fn load(config: ModelNet40Config) -> Result<Self, ModelNet40Error> {
        let root = &config.root_path;

        if !root.exists() {
            return Err(ModelNet40Error::PathNotFound(root.clone()));
        }

        // Build class name to index mapping
        let class_names: Vec<String> = MODELNET40_CLASSES.iter().map(|s| s.to_string()).collect();
        let class_to_idx: HashMap<String, usize> = class_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Load train and test splits
        let train_samples = Self::load_split(root, "train", &class_to_idx, &config)?;
        let test_samples = Self::load_split(root, "test", &class_to_idx, &config)?;

        println!(
            "Loaded ModelNet40: {} train, {} test samples",
            train_samples.len(),
            test_samples.len()
        );

        Ok(ModelNet40 {
            train_split: DatasetSplit::new(train_samples, "train"),
            test_split: DatasetSplit::new(test_samples, "test"),
            class_names,
            config,
        })
    }

    /// Load a single split (train or test)
    fn load_split(
        root: &Path,
        split: &str,
        class_to_idx: &HashMap<String, usize>,
        config: &ModelNet40Config,
    ) -> Result<Vec<PointCloud>, ModelNet40Error> {
        let mut samples = Vec::new();
        let mut loaded_count = 0;

        for (class_name, &class_idx) in class_to_idx {
            // Try different directory structures
            let split_dir = Self::find_split_dir(root, class_name, split);

            if let Some(dir) = split_dir {
                let entries = fs::read_dir(&dir).map_err(|e| {
                    ModelNet40Error::IoError(format!("Failed to read {:?}: {}", dir, e))
                })?;

                for entry in entries {
                    if let Some(max) = config.max_samples {
                        if loaded_count >= max {
                            break;
                        }
                    }

                    let entry = entry.map_err(|e| {
                        ModelNet40Error::IoError(format!("Failed to read entry: {}", e))
                    })?;
                    let path = entry.path();

                    // Check file extension
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    let valid_ext = match config.format {
                        ModelNet40Format::Off => ext == "off",
                        ModelNet40Format::Txt | ModelNet40Format::TxtWithNormals => {
                            ext == "txt" || ext == "pts"
                        }
                    };

                    if !valid_ext {
                        continue;
                    }

                    // Load point cloud
                    match Self::load_file(&path, config) {
                        Ok(mut pc) => {
                            pc.label = Some(class_idx);
                            pc.class_name = Some(class_name.clone());

                            if config.preprocess {
                                pc.preprocess();
                            }

                            // Sample to target number of points
                            let pc = if config.use_fps {
                                pc.farthest_point_sample(config.num_points)
                            } else {
                                pc.sample(config.num_points)
                            };

                            samples.push(pc);
                            loaded_count += 1;
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to load {:?}: {}", path, e);
                        }
                    }
                }
            }
        }

        // Shuffle samples
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        samples.shuffle(&mut rng);

        Ok(samples)
    }

    /// Find the split directory (handles different directory structures)
    fn find_split_dir(root: &Path, class_name: &str, split: &str) -> Option<PathBuf> {
        // Structure 1: modelnet40/airplane/train/
        let path1 = root.join(class_name).join(split);
        if path1.exists() {
            return Some(path1);
        }

        // Structure 2: modelnet40/train/airplane/
        let path2 = root.join(split).join(class_name);
        if path2.exists() {
            return Some(path2);
        }

        // Structure 3: modelnet40_train/airplane/ or modelnet40_test/airplane/
        let path3 = root.parent()?.join(format!("modelnet40_{}", split)).join(class_name);
        if path3.exists() {
            return Some(path3);
        }

        None
    }

    /// Load a single file
    fn load_file(path: &Path, config: &ModelNet40Config) -> Result<PointCloud, ModelNet40Error> {
        match config.format {
            ModelNet40Format::Off => Self::load_off(path),
            ModelNet40Format::Txt => Self::load_txt(path, false),
            ModelNet40Format::TxtWithNormals => Self::load_txt(path, true),
        }
    }

    /// Load OFF mesh file and sample points
    fn load_off(path: &Path) -> Result<PointCloud, ModelNet40Error> {
        let file = File::open(path)
            .map_err(|e| ModelNet40Error::IoError(format!("Failed to open {:?}: {}", path, e)))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read header
        let first_line = lines
            .next()
            .ok_or_else(|| ModelNet40Error::ParseError("Empty file".to_string()))?
            .map_err(|e| ModelNet40Error::ParseError(e.to_string()))?;

        // Handle "OFF" on first line or combined with counts
        let counts_line = if first_line.trim() == "OFF" {
            lines
                .next()
                .ok_or_else(|| ModelNet40Error::ParseError("Missing counts line".to_string()))?
                .map_err(|e| ModelNet40Error::ParseError(e.to_string()))?
        } else if first_line.starts_with("OFF") {
            first_line[3..].trim().to_string()
        } else {
            return Err(ModelNet40Error::ParseError(format!(
                "Invalid OFF header: {}",
                first_line
            )));
        };

        // Parse vertex/face counts
        let counts: Vec<usize> = counts_line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if counts.len() < 2 {
            return Err(ModelNet40Error::ParseError(
                "Invalid counts line".to_string(),
            ));
        }

        let num_vertices = counts[0];
        let _num_faces = counts[1];

        // Read vertices
        let mut points = Vec::with_capacity(num_vertices);
        for _ in 0..num_vertices {
            let line = lines
                .next()
                .ok_or_else(|| ModelNet40Error::ParseError("Unexpected end of file".to_string()))?
                .map_err(|e| ModelNet40Error::ParseError(e.to_string()))?;

            let coords: Vec<f64> = line
                .split_whitespace()
                .take(3)
                .filter_map(|s| s.parse().ok())
                .collect();

            if coords.len() >= 3 {
                points.push(Point3D::new(coords[0], coords[1], coords[2]));
            }
        }

        // For now, just use vertices as point cloud
        // TODO: Sample from mesh surface for better coverage
        Ok(PointCloud::from_points(points))
    }

    /// Load pre-sampled point cloud from text file
    fn load_txt(path: &Path, with_normals: bool) -> Result<PointCloud, ModelNet40Error> {
        let file = File::open(path)
            .map_err(|e| ModelNet40Error::IoError(format!("Failed to open {:?}: {}", path, e)))?;
        let reader = BufReader::new(file);

        let mut points = Vec::new();

        for line in reader.lines() {
            let line =
                line.map_err(|e| ModelNet40Error::IoError(format!("Failed to read line: {}", e)))?;

            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Handle comma or space separation
            let parts: Vec<&str> = if line.contains(',') {
                line.split(',').collect()
            } else {
                line.split_whitespace().collect()
            };

            if parts.len() >= 3 {
                let x: f64 = parts[0].parse().unwrap_or(0.0);
                let y: f64 = parts[1].parse().unwrap_or(0.0);
                let z: f64 = parts[2].parse().unwrap_or(0.0);

                if with_normals && parts.len() >= 6 {
                    let nx: f64 = parts[3].parse().unwrap_or(0.0);
                    let ny: f64 = parts[4].parse().unwrap_or(0.0);
                    let nz: f64 = parts[5].parse().unwrap_or(0.0);
                    points.push(Point3D::with_normal(x, y, z, nx, ny, nz));
                } else {
                    points.push(Point3D::new(x, y, z));
                }
            }
        }

        if points.is_empty() {
            return Err(ModelNet40Error::ParseError(
                "No points found in file".to_string(),
            ));
        }

        Ok(PointCloud::from_points(points))
    }

    /// Get class name by index
    pub fn class_name(&self, idx: usize) -> Option<&str> {
        self.class_names.get(idx).map(|s| s.as_str())
    }

    /// Get class index by name
    pub fn class_index(&self, name: &str) -> Option<usize> {
        self.class_names.iter().position(|n| n == name)
    }

    /// Print dataset statistics
    pub fn print_stats(&self) {
        println!("\nModelNet40 Dataset Statistics:");
        println!("================================");
        println!("Number of classes: {}", self.num_classes());
        println!("Training samples: {}", self.train_split.len());
        println!("Test samples: {}", self.test_split.len());
        println!("Points per sample: {}", self.config.num_points);

        println!("\nClass distribution (train):");
        let train_counts = self.train_split.class_counts(self.num_classes());
        for (i, count) in train_counts.iter().enumerate() {
            if *count > 0 {
                println!("  {:20} {:4}", self.class_names[i], count);
            }
        }

        println!("\nClass distribution (test):");
        let test_counts = self.test_split.class_counts(self.num_classes());
        for (i, count) in test_counts.iter().enumerate() {
            if *count > 0 {
                println!("  {:20} {:4}", self.class_names[i], count);
            }
        }
    }
}

impl Dataset for ModelNet40 {
    fn num_classes(&self) -> usize {
        self.class_names.len()
    }

    fn class_names(&self) -> &[String] {
        &self.class_names
    }

    fn train(&self) -> &DatasetSplit {
        &self.train_split
    }

    fn test(&self) -> &DatasetSplit {
        &self.test_split
    }
}

/// Errors that can occur when loading ModelNet40
#[derive(Debug)]
pub enum ModelNet40Error {
    PathNotFound(PathBuf),
    IoError(String),
    ParseError(String),
}

impl std::fmt::Display for ModelNet40Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelNet40Error::PathNotFound(path) => {
                write!(f, "Path not found: {:?}", path)
            }
            ModelNet40Error::IoError(msg) => write!(f, "IO error: {}", msg),
            ModelNet40Error::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for ModelNet40Error {}

/// Generate synthetic ModelNet40-like dataset for testing
///
/// This creates a small synthetic dataset that mimics the structure
/// of ModelNet40 without requiring the actual dataset.
pub fn generate_synthetic_modelnet40(
    samples_per_class: usize,
    points_per_sample: usize,
    num_classes: usize,
) -> (DatasetSplit, DatasetSplit) {
    use rand::Rng;

    let num_classes = num_classes.min(40);
    let mut train_samples = Vec::new();
    let mut test_samples = Vec::new();

    let train_count = (samples_per_class as f64 * 0.8) as usize;
    let test_count = samples_per_class - train_count;

    for class_idx in 0..num_classes {
        // Generate training samples
        for _ in 0..train_count {
            let pc = generate_class_sample(class_idx, points_per_sample);
            train_samples.push(pc);
        }

        // Generate test samples
        for _ in 0..test_count {
            let pc = generate_class_sample(class_idx, points_per_sample);
            test_samples.push(pc);
        }
    }

    // Shuffle
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    train_samples.shuffle(&mut rng);
    test_samples.shuffle(&mut rng);

    (
        DatasetSplit::new(train_samples, "train"),
        DatasetSplit::new(test_samples, "test"),
    )
}

/// Generate a synthetic point cloud for a given class
fn generate_class_sample(class_idx: usize, num_points: usize) -> PointCloud {
    let mut rng = rand::thread_rng();

    // Each class has a distinct shape characteristic
    // We use a combination of:
    // 1. Base shape (sphere, cube, cylinder, etc.)
    // 2. Aspect ratio
    // 3. Noise level

    let base_shape = class_idx % 5;
    let aspect_x = 1.0 + 0.2 * ((class_idx / 5) as f64);
    let aspect_y = 1.0 + 0.1 * ((class_idx / 10) as f64);
    let noise = 0.02 + 0.01 * ((class_idx % 7) as f64);

    let mut points = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        let (mut x, mut y, mut z) = match base_shape {
            0 => {
                // Sphere
                let theta = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                let phi = rng.gen_range(0.0..std::f64::consts::PI);
                (
                    phi.sin() * theta.cos(),
                    phi.sin() * theta.sin(),
                    phi.cos(),
                )
            }
            1 => {
                // Cube
                let face = rng.gen_range(0..6);
                let u = rng.gen_range(-1.0..1.0);
                let v = rng.gen_range(-1.0..1.0);
                match face {
                    0 => (1.0, u, v),
                    1 => (-1.0, u, v),
                    2 => (u, 1.0, v),
                    3 => (u, -1.0, v),
                    4 => (u, v, 1.0),
                    _ => (u, v, -1.0),
                }
            }
            2 => {
                // Cylinder
                let theta = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                let h = rng.gen_range(-1.0..1.0);
                (theta.cos(), theta.sin(), h)
            }
            3 => {
                // Cone
                let theta = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                let h = rng.gen_range(0.0..1.0);
                let r = 1.0 - h;
                (r * theta.cos(), r * theta.sin(), h)
            }
            _ => {
                // Torus-like
                let theta = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                let phi = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                let r_major = 0.7;
                let r_minor = 0.3;
                (
                    (r_major + r_minor * phi.cos()) * theta.cos(),
                    (r_major + r_minor * phi.cos()) * theta.sin(),
                    r_minor * phi.sin(),
                )
            }
        };

        // Apply aspect ratio
        x *= aspect_x;
        y *= aspect_y;

        // Add noise
        x += rng.gen_range(-noise..noise);
        y += rng.gen_range(-noise..noise);
        z += rng.gen_range(-noise..noise);

        points.push(Point3D::new(x, y, z));
    }

    let mut pc = PointCloud::from_points_with_class(
        points,
        class_idx,
        MODELNET40_CLASSES.get(class_idx).unwrap_or(&"unknown").to_string(),
    );
    pc.preprocess();
    pc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dataset() {
        let (train, test) = generate_synthetic_modelnet40(10, 256, 10);

        assert_eq!(train.len(), 80); // 8 train per class × 10 classes
        assert_eq!(test.len(), 20); // 2 test per class × 10 classes

        // Check that samples have correct number of points
        for sample in train.samples.iter().chain(test.samples.iter()) {
            assert_eq!(sample.len(), 256);
            assert!(sample.label.unwrap() < 10);
        }
    }

    #[test]
    fn test_class_names() {
        assert_eq!(MODELNET40_CLASSES.len(), 40);
        assert_eq!(MODELNET40_CLASSES[0], "airplane");
        assert_eq!(MODELNET40_CLASSES[39], "xbox");
    }

    #[test]
    fn test_class_distinguishability() {
        // Different classes should produce different point distributions
        let pc0 = generate_class_sample(0, 100);
        let pc1 = generate_class_sample(1, 100);
        let pc2 = generate_class_sample(2, 100);

        // Compute simple statistics
        fn compute_stats(pc: &PointCloud) -> (f64, f64, f64) {
            let centroid = pc.centroid();
            let mean_dist: f64 = pc.points.iter().map(|p| p.distance(&centroid)).sum::<f64>()
                / pc.len() as f64;
            let z_range = pc.points.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max)
                - pc.points.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
            (centroid.magnitude(), mean_dist, z_range)
        }

        let stats0 = compute_stats(&pc0);
        let stats1 = compute_stats(&pc1);
        let stats2 = compute_stats(&pc2);

        // At least some statistics should differ between classes
        // (This is a weak test, but ensures the generation is not identical)
        let same_01 = (stats0.0 - stats1.0).abs() < 0.01
            && (stats0.1 - stats1.1).abs() < 0.01
            && (stats0.2 - stats1.2).abs() < 0.01;
        let same_02 = (stats0.0 - stats2.0).abs() < 0.01
            && (stats0.1 - stats2.1).abs() < 0.01
            && (stats0.2 - stats2.2).abs() < 0.01;

        assert!(
            !same_01 || !same_02,
            "Different classes should have different statistics"
        );
    }
}
