/// Medical Imaging Examples for Clifford FHE
///
/// This module demonstrates privacy-preserving 3D medical imaging classification
/// using batched Clifford FHE on GPU backends (Metal + CUDA).

pub mod point_cloud;
pub mod clifford_encoding;
pub mod synthetic_data;
pub mod plaintext_gnn;
pub mod simd_batching;
pub mod batched_gnn;
pub mod encrypted_inference;

// V2 CPU backend integration (requires v2 feature)
#[cfg(feature = "v2")]
pub mod encrypted_v2_cpu;

// V2 Metal GPU backend integration (requires v2-gpu-metal feature)
#[cfg(feature = "v2-gpu-metal")]
pub mod encrypted_metal;

// Re-export commonly used types
pub use point_cloud::{Point3D, PointCloud};
pub use clifford_encoding::{Multivector3D, encode_point_cloud, encode_batch};
pub use synthetic_data::{ShapeType, generate_dataset, train_test_split};
pub use plaintext_gnn::{GeometricNeuralNetwork, Trainer};
pub use simd_batching::BatchedMultivectors;
