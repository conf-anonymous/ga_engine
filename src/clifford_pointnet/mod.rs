//! CliffordPointNet: Privacy-Preserving 3D Point Cloud Classification
//!
//! A neural network architecture designed for encrypted point cloud inference
//! using Clifford algebra (Cl(3,0)) geometric products.
//!
//! ## Architecture
//!
//! ```text
//! Input: N points (x,y,z)
//!     ↓
//! Point Encoding → N multivectors in Cl(3,0)
//!     ↓
//! Clifford Embedding Layer (shared weights)
//!     ↓
//! Clifford Local Aggregation (geometric attention)
//!     ↓
//! Clifford Global Pooling (permutation invariant)
//!     ↓
//! Classification Head
//!     ↓
//! Output: Class logits
//! ```
//!
//! ## Key Features
//!
//! - Uses true Cl(3,0) geometric product (not simplified dot product)
//! - All operations are polynomial (FHE-compatible)
//! - Rotation equivariance by construction
//! - Permutation invariance via symmetric aggregation

pub mod multivector;
pub mod layers;
pub mod model;
pub mod training;
pub mod encoding;
pub mod simple_model;
pub mod serialization;
pub mod gp_classifier;

pub use multivector::Multivector;
pub use layers::{CliffordEmbedding, CliffordAggregation, CliffordPooling, CliffordClassifier};
pub use model::{CliffordPointNet, CliffordPointNetConfig};
pub use training::{Trainer, TrainingConfig, TrainingMetrics};
pub use encoding::encode_point_cloud;
pub use simple_model::{SimpleCliffordNet, train_simple_net};
pub use gp_classifier::{GPFeatureClassifier, encode_point_augmented, plaintext_geometric_product, compute_gp_features, train_gp_classifier};
