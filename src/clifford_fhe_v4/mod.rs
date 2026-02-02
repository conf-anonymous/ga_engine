/// Clifford FHE V4 - Packed Slot-Interleaved Layout
///
/// V4 implements a memory-efficient packed layout where all 8 Clifford algebra components
/// are stored in a single CKKS ciphertext with interleaved slots:
///
/// Slot layout: [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, e2₁, ...]
///
/// This provides:
/// - 8× memory reduction compared to naive component-separate layout
/// - Same memory footprint as standard CKKS
/// - 2-4× throughput improvement from batching
/// - Trade-off: 2-4× slower per operation due to rotation overhead
///
/// V4 uses V2 backend for all low-level operations (NTT, encoding, GPU ops).

#[cfg(feature = "v4")]
pub mod packed_multivector;

#[cfg(feature = "v4")]
pub mod packing;

#[cfg(feature = "v4")]
pub mod packing_butterfly;

#[cfg(feature = "v4")]
pub mod params;

#[cfg(feature = "v4")]
pub mod mult_table;

#[cfg(feature = "v4")]
pub mod geometric_ops;

#[cfg(feature = "v4")]
pub mod bootstrapping;

#[cfg(all(feature = "v4", feature = "v2-gpu-cuda"))]
pub mod cuda_adapter;

#[cfg(all(feature = "v4", feature = "v2-gpu-cuda"))]
pub mod packing_cuda;

// Re-export main types
#[cfg(feature = "v4")]
pub use packed_multivector::PackedMultivector;

// Export CUDA-specific packing functions when CUDA is enabled
#[cfg(all(feature = "v4", feature = "v2-gpu-cuda"))]
pub use packing_cuda::{pack_multivector_cuda as pack_multivector, unpack_multivector_cuda as unpack_multivector, extract_component_cuda as extract_component};

// Export Metal/CPU packing functions when CUDA is NOT enabled
#[cfg(all(feature = "v4", feature = "v2-gpu-metal", not(feature = "v2-gpu-cuda")))]
pub use packing::{pack_multivector, unpack_multivector, extract_component};

// Butterfly packing - keep shared for now (may need CUDA version later)
#[cfg(all(feature = "v4", any(feature = "v2-gpu-cuda", feature = "v2-gpu-metal")))]
pub use packing_butterfly::{pack_multivector_butterfly, unpack_multivector_butterfly};

#[cfg(feature = "v4")]
pub use params::PackedParams;
