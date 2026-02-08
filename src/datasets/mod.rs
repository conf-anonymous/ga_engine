//! Dataset loaders for 3D point cloud benchmarks
//!
//! This module provides loaders for standard point cloud classification datasets:
//! - ModelNet40: 40-class CAD model classification
//! - ScanObjectNN: Real-world scanned object classification
//!
//! All datasets output point clouds in a standardized format suitable for
//! CliffordPointNet training and evaluation.

pub mod point_cloud;
pub mod modelnet40;
pub mod augmentation;

pub use point_cloud::{PointCloud, Point3D, Dataset, DatasetSplit};
pub use modelnet40::{ModelNet40, ModelNet40Config};
pub use augmentation::{Augmentation, AugmentationConfig};
