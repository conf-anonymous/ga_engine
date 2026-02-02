/// Geometric Operations on Packed Multivectors
///
/// Implements Clifford algebra operations using diagonal multiply + rotation pattern.

use super::packed_multivector::PackedMultivector;
use super::mult_table::PackedMultTable;

// Import extract_component from parent module (which re-exports the right version)
#[cfg(any(feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
use super::extract_component;

/// Find a primitive 2N-th root of unity modulo q
///
/// For NTT-friendly primes q ≡ 1 (mod 2N), computes ψ = g^((q-1)/(2N))
/// where g is a generator of the multiplicative group mod q.
fn find_primitive_root_for_ntt(n: usize, q: u64) -> Result<u64, String> {
    // Verify q ≡ 1 (mod 2n)
    let two_n = (2 * n) as u64;
    if (q - 1) % two_n != 0 {
        return Err(format!(
            "q = {} is not NTT-friendly for n = {} (q-1 must be divisible by 2n)",
            q, n
        ));
    }

    // Try small candidates that are often generators
    for candidate in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        if is_primitive_root_candidate(candidate, n, q) {
            let exponent = (q - 1) / two_n;
            return Ok(mod_pow(candidate, exponent, q));
        }
    }

    // Extended search
    for candidate in 32..1000u64 {
        if is_primitive_root_candidate(candidate, n, q) {
            let exponent = (q - 1) / two_n;
            return Ok(mod_pow(candidate, exponent, q));
        }
    }

    Err(format!("Failed to find primitive root for q = {}, n = {}", q, n))
}

/// Check if g is suitable for generating primitive 2n-th root
fn is_primitive_root_candidate(g: u64, n: usize, q: u64) -> bool {
    // Check if g is a quadratic non-residue
    if mod_pow(g, (q - 1) / 2, q) == 1 {
        return false;
    }

    // Check if g^((q-1)/(2n)) generates the subgroup of order 2n
    let psi = mod_pow(g, (q - 1) / (2 * n as u64), q);

    // psi^n should equal -1 mod q
    let psi_n = mod_pow(psi, n as u64, q);
    if psi_n != q - 1 {
        return false;
    }

    // psi^(2n) should equal 1 mod q
    let psi_2n = mod_pow(psi, 2 * n as u64, q);
    psi_2n == 1
}

/// Modular exponentiation: base^exp mod m
fn mod_pow(base: u64, exp: u64, m: u64) -> u64 {
    let mut result = 1u128;
    let mut base = base as u128;
    let mut exp = exp;
    let m = m as u128;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }

    result as u64
}

#[cfg(feature = "v2-gpu-cuda")]
use crate::clifford_fhe_v2::backends::gpu_cuda::{
    ckks::{CudaCiphertext as Ciphertext, CudaCkksContext, CudaPlaintext as Plaintext},
    rotation_keys::CudaRotationKeys as RotationKeys,
};

#[cfg(all(feature = "v2-gpu-metal", not(feature = "v2-gpu-cuda")))]
use crate::clifford_fhe_v2::backends::gpu_metal::{
    ckks::{MetalCiphertext as Ciphertext, MetalCkksContext as CudaCkksContext, MetalPlaintext as Plaintext},
    rotation_keys::MetalRotationKeys as RotationKeys,
};

#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
use crate::clifford_fhe_v2::backends::cpu_optimized::{
    ckks::{Ciphertext, CpuCkksContext as CudaCkksContext, Plaintext},
};

/// Geometric product: a ⊗ b (packed version)
///
/// Algorithm:
/// 1. Unpack both packed multivectors into 8 component ciphertexts each
/// 2. Convert RNS ciphertexts to per-prime format for Metal geometric product
/// 3. For each prime modulus:
///    - Use Metal geometric product to compute all 8 output components
/// 4. Reconstruct RNS ciphertexts from per-prime results
/// 5. Pack the 8 result components back into a single PackedMultivector
///
/// This leverages the existing Metal GPU geometric product which handles
/// all 64 ciphertext multiplications in parallel on the GPU.
#[cfg(feature = "v2-gpu-metal")]
pub fn geometric_product_packed(
    a: &PackedMultivector,
    b: &PackedMultivector,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    use super::unpack_multivector;
    use crate::clifford_fhe_v2::backends::gpu_metal::geometric::MetalGeometricProduct;

    if !a.is_compatible(b) {
        return Err("Incompatible packed multivectors".to_string());
    }

    // Step 1: Unpack into component ciphertexts (RNS format) using butterfly transform
    use super::packing_butterfly::unpack_multivector_butterfly;
    let a_components = unpack_multivector_butterfly(a, rot_keys, ckks_ctx)?;
    let b_components = unpack_multivector_butterfly(b, rot_keys, ckks_ctx)?;

    let n = a.n;
    let level = a.level;

    // Use the actual number of primes from the unpacked ciphertexts, not from PackedMultivector
    let num_primes = a_components[0].num_primes;
    let moduli = &ckks_ctx.params.moduli[..num_primes];


    // Step 2: Reuse the existing Metal device from ckks_ctx instead of creating new ones
    // This is MUCH faster - avoids reinitializing Metal device for each prime
    let device = ckks_ctx.device().clone();

    // Step 2: Process each prime modulus separately
    let mut result_components_rns: Vec<Ciphertext> = vec![];

    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];

        // Find primitive root for this prime
        let root = find_primitive_root_for_ntt(n, q)?;

        // Create Metal geometric product computer using existing device
        let metal_gp = MetalGeometricProduct::new_with_device(device.clone(), n, q, root)?;

        // Step 2a: Extract polynomials for this prime from RNS representation
        let mut a_prime: [[Vec<u64>; 2]; 8] = Default::default();
        let mut b_prime: [[Vec<u64>; 2]; 8] = Default::default();

        for comp in 0..8 {
            // Extract c0 and c1 for this prime
            a_prime[comp][0] = extract_prime_from_strided(&a_components[comp].c0, n, num_primes, prime_idx);
            a_prime[comp][1] = extract_prime_from_strided(&a_components[comp].c1, n, num_primes, prime_idx);
            b_prime[comp][0] = extract_prime_from_strided(&b_components[comp].c0, n, num_primes, prime_idx);
            b_prime[comp][1] = extract_prime_from_strided(&b_components[comp].c1, n, num_primes, prime_idx);
        }

        // Step 2b: Compute geometric product for this prime
        let result_prime = metal_gp.geometric_product(&a_prime, &b_prime)?;

        // Step 2c: Store results for later RNS reconstruction
        if prime_idx == 0 {
            // Initialize result vectors
            // Use num_primes-1 as level since level is 0-indexed
            let result_level = num_primes - 1;
            for comp in 0..8 {
                result_components_rns.push(Ciphertext {
                    c0: vec![0u64; n * num_primes],
                    c1: vec![0u64; n * num_primes],
                    n,
                    num_primes,
                    level: result_level,
                    scale: a.scale * b.scale,
                });
            }
        }

        // Insert this prime's results into the flat RNS layout
        for comp in 0..8 {
            insert_prime_into_strided(&result_prime[comp][0], &mut result_components_rns[comp].c0, n, num_primes, prime_idx);
            insert_prime_into_strided(&result_prime[comp][1], &mut result_components_rns[comp].c1, n, num_primes, prime_idx);
        }
    }

    // Step 3: Pack result components back into a single PackedMultivector
    let result_array: [Ciphertext; 8] = [
        result_components_rns[0].clone(),
        result_components_rns[1].clone(),
        result_components_rns[2].clone(),
        result_components_rns[3].clone(),
        result_components_rns[4].clone(),
        result_components_rns[5].clone(),
        result_components_rns[6].clone(),
        result_components_rns[7].clone(),
    ];

    // Step 3: Pack result components back using butterfly transform
    use super::packing_butterfly::pack_multivector_butterfly;
    pack_multivector_butterfly(&result_array, a.batch_size, rot_keys, ckks_ctx)
}

/// Extract coefficients for a single prime from strided RNS layout
///
/// Strided layout (CUDA): [coeff_0 for all primes, coeff_1 for all primes, ...]
/// where each coefficient block is: [c_q0, c_q1, ..., c_qL]
/// Returns: [c0, c1, ..., c_{n-1}] for the specified prime
fn extract_prime_from_strided(strided: &[u64], n: usize, num_primes: usize, prime_idx: usize) -> Vec<u64> {
    let mut result = vec![0u64; n];
    for coeff_idx in 0..n {
        let idx = coeff_idx * num_primes + prime_idx;
        result[coeff_idx] = strided[idx];
    }
    result
}

/// Insert coefficients for a single prime into strided RNS layout
fn insert_prime_into_strided(prime_coeffs: &[u64], strided: &mut [u64], n: usize, num_primes: usize, prime_idx: usize) {
    for coeff_idx in 0..n {
        let idx = coeff_idx * num_primes + prime_idx;
        strided[idx] = prime_coeffs[coeff_idx];
    }
}

/// CUDA version - Full implementation using CudaGeometricProduct
#[cfg(all(feature = "v2-gpu-cuda", not(feature = "v2-gpu-metal")))]
pub fn geometric_product_packed(
    a: &PackedMultivector,
    b: &PackedMultivector,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    use super::unpack_multivector;
    use crate::clifford_fhe_v2::backends::gpu_cuda::geometric::CudaGeometricProduct;

    if !a.is_compatible(b) {
        return Err("Incompatible packed multivectors".to_string());
    }

    // Step 1: Unpack into component ciphertexts (RNS format) using butterfly transform
    use super::packing_butterfly::unpack_multivector_butterfly;
    let a_components = unpack_multivector_butterfly(a, rot_keys, ckks_ctx)?;
    let b_components = unpack_multivector_butterfly(b, rot_keys, ckks_ctx)?;

    let n = a.n;
    let level = a.level;

    // Use the actual number of primes from the unpacked ciphertexts
    let num_primes = a_components[0].num_primes;
    let moduli = &ckks_ctx.params().moduli[..num_primes];

    // Step 2: Process each prime modulus separately
    let mut result_components_rns = Vec::new();

    for prime_idx in 0..num_primes {
        let q = moduli[prime_idx];

        // Find primitive root for this prime
        let root = find_primitive_root_for_ntt(n, q)?;

        // Create CUDA geometric product computer
        let cuda_gp = CudaGeometricProduct::new(n, q, root)?;

        // Step 2a: Extract polynomials for this prime from RNS representation
        let mut a_prime: [[Vec<u64>; 2]; 8] = Default::default();
        let mut b_prime: [[Vec<u64>; 2]; 8] = Default::default();

        for comp in 0..8 {
            // Extract c0 and c1 for this prime
            a_prime[comp][0] = extract_prime_from_strided(&a_components[comp].c0, n, num_primes, prime_idx);
            a_prime[comp][1] = extract_prime_from_strided(&a_components[comp].c1, n, num_primes, prime_idx);
            b_prime[comp][0] = extract_prime_from_strided(&b_components[comp].c0, n, num_primes, prime_idx);
            b_prime[comp][1] = extract_prime_from_strided(&b_components[comp].c1, n, num_primes, prime_idx);
        }

        // Step 2b: Compute geometric product for this prime
        let result_prime = cuda_gp.geometric_product(&a_prime, &b_prime)?;

        // Step 2c: Store results for later RNS reconstruction
        if prime_idx == 0 {
            // Initialize result vectors
            let result_level = num_primes - 1;
            for comp in 0..8 {
                result_components_rns.push(Ciphertext {
                    c0: vec![0u64; n * num_primes],
                    c1: vec![0u64; n * num_primes],
                    n,
                    num_primes,
                    level: result_level,
                    scale: a.scale * b.scale,
                });
            }
        }

        // Insert this prime's results into the flat RNS layout
        for comp in 0..8 {
            insert_prime_into_strided(&result_prime[comp][0], &mut result_components_rns[comp].c0, n, num_primes, prime_idx);
            insert_prime_into_strided(&result_prime[comp][1], &mut result_components_rns[comp].c1, n, num_primes, prime_idx);
        }
    }

    // Step 3: Pack result components back into a single PackedMultivector
    let result_array: [Ciphertext; 8] = [
        result_components_rns[0].clone(),
        result_components_rns[1].clone(),
        result_components_rns[2].clone(),
        result_components_rns[3].clone(),
        result_components_rns[4].clone(),
        result_components_rns[5].clone(),
        result_components_rns[6].clone(),
        result_components_rns[7].clone(),
    ];

    // Pack result components back using butterfly transform
    use super::packing_butterfly::pack_multivector_butterfly;
    pack_multivector_butterfly(&result_array, a.batch_size, rot_keys, ckks_ctx)
}

/// CPU version (placeholder)
#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn geometric_product_packed(
    _a: &PackedMultivector,
    _b: &PackedMultivector,
    _ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    Err("geometric_product_packed not yet implemented for CPU backend".to_string())
}

/// Wedge product: a ∧ b = (ab - ba) / 2 (packed version)
///
/// Antisymmetric part of the geometric product.
/// Computes the exterior product which gives the oriented area/volume element.
#[cfg(feature = "v2-gpu-metal")]
pub fn wedge_product_packed(
    a: &PackedMultivector,
    b: &PackedMultivector,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    if !a.is_compatible(b) {
        return Err("Incompatible packed multivectors".to_string());
    }

    // wedge(a,b) = (geometric(a,b) - geometric(b,a)) / 2
    let ab = geometric_product_packed(a, b, rot_keys, ckks_ctx)?;
    let ba = geometric_product_packed(b, a, rot_keys, ckks_ctx)?;
    let diff = subtract_packed(&ab, &ba, ckks_ctx)?;

    // Multiply by 0.5
    let half = ckks_ctx.encode(&vec![0.5])?;
    let result_ct = diff.ct.multiply_plain(&half, ckks_ctx)?;

    Ok(PackedMultivector::new(
        result_ct,
        a.batch_size,
        a.n,
        a.num_primes,
        a.level,
        diff.scale * 0.5,
    ))
}

#[cfg(all(feature = "v2-gpu-cuda", not(feature = "v2-gpu-metal")))]
pub fn wedge_product_packed(
    a: &PackedMultivector,
    b: &PackedMultivector,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    if !a.is_compatible(b) {
        return Err("Incompatible packed multivectors".to_string());
    }

    // wedge(a,b) = (geometric(a,b) - geometric(b,a)) / 2
    let ab = geometric_product_packed(a, b, rot_keys, ckks_ctx)?;
    let ba = geometric_product_packed(b, a, rot_keys, ckks_ctx)?;
    let diff = subtract_packed(&ab, &ba, ckks_ctx)?;

    // Multiply by 0.5
    #[cfg(feature = "v2-gpu-cuda")]
    let half = ckks_ctx.encode(&vec![0.5], ckks_ctx.params().scale, a.level)?;

    let result_ct = diff.ct.multiply_plain(&half, ckks_ctx)?;

    Ok(PackedMultivector::new(
        result_ct,
        a.batch_size,
        a.n,
        a.num_primes,
        a.level,
        diff.scale * 0.5,
    ))
}

/// Inner product: a · b = (ab + ba) / 2 (packed version)
///
/// Symmetric part of the geometric product.
/// Generalizes the scalar/dot product to all grades.
#[cfg(feature = "v2-gpu-metal")]
pub fn inner_product_packed(
    a: &PackedMultivector,
    b: &PackedMultivector,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    if !a.is_compatible(b) {
        return Err("Incompatible packed multivectors".to_string());
    }

    // inner(a,b) = (geometric(a,b) + geometric(b,a)) / 2
    let ab = geometric_product_packed(a, b, rot_keys, ckks_ctx)?;
    let ba = geometric_product_packed(b, a, rot_keys, ckks_ctx)?;
    let sum = add_packed(&ab, &ba, ckks_ctx)?;

    // Multiply by 0.5
    let half = ckks_ctx.encode(&vec![0.5])?;
    let result_ct = sum.ct.multiply_plain(&half, ckks_ctx)?;

    Ok(PackedMultivector::new(
        result_ct,
        a.batch_size,
        a.n,
        a.num_primes,
        a.level,
        sum.scale * 0.5,
    ))
}

#[cfg(all(feature = "v2-gpu-cuda", not(feature = "v2-gpu-metal")))]
pub fn inner_product_packed(
    a: &PackedMultivector,
    b: &PackedMultivector,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    if !a.is_compatible(b) {
        return Err("Incompatible packed multivectors".to_string());
    }

    // inner(a,b) = (geometric(a,b) + geometric(b,a)) / 2
    let ab = geometric_product_packed(a, b, rot_keys, ckks_ctx)?;
    let ba = geometric_product_packed(b, a, rot_keys, ckks_ctx)?;
    let sum = add_packed(&ab, &ba, ckks_ctx)?;

    // Multiply by 0.5
    #[cfg(feature = "v2-gpu-cuda")]
    let half = ckks_ctx.encode(&vec![0.5], ckks_ctx.params().scale, a.level)?;

    let result_ct = sum.ct.multiply_plain(&half, ckks_ctx)?;

    Ok(PackedMultivector::new(
        result_ct,
        a.batch_size,
        a.n,
        a.num_primes,
        a.level,
        sum.scale * 0.5,
    ))
}

/// CPU versions (placeholder)
#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn wedge_product_packed(
    _a: &PackedMultivector,
    _b: &PackedMultivector,
    _ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    Err("wedge_product_packed not yet implemented for CPU backend".to_string())
}

#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn inner_product_packed(
    _a: &PackedMultivector,
    _b: &PackedMultivector,
    _ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    Err("inner_product_packed not yet implemented for CPU backend".to_string())
}

/// Addition: a + b (packed version)
///
/// Simple component-wise addition on the packed ciphertext.
/// Since all 8 components are interleaved in the same slots,
/// adding two packed ciphertexts adds all corresponding components.
#[cfg(any(feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
pub fn add_packed(
    a: &PackedMultivector,
    b: &PackedMultivector,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    if !a.is_compatible(b) {
        return Err("Incompatible packed multivectors".to_string());
    }

    // Add the underlying ciphertexts
    let result_ct = a.ct.add(&b.ct, ckks_ctx)?;

    Ok(PackedMultivector::new(
        result_ct,
        a.batch_size,
        a.n,
        a.num_primes,
        a.level,
        a.scale,
    ))
}

/// Subtraction: a - b (packed version)
///
/// Simple component-wise subtraction on the packed ciphertext.
/// Implemented as a + (-b) by negating b and adding.
#[cfg(any(feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
pub fn subtract_packed(
    a: &PackedMultivector,
    b: &PackedMultivector,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    if !a.is_compatible(b) {
        return Err("Incompatible packed multivectors".to_string());
    }

    // Negate b by multiplying by -1
    #[cfg(feature = "v2-gpu-cuda")]
    let neg_one = ckks_ctx.encode(&vec![-1.0], ckks_ctx.params().scale, a.level)?;
    #[cfg(all(feature = "v2-gpu-metal", not(feature = "v2-gpu-cuda")))]
    let neg_one = ckks_ctx.encode(&vec![-1.0])?;

    let neg_b = b.ct.multiply_plain(&neg_one, ckks_ctx)?;

    // Add a + (-b)
    let result_ct = a.ct.add(&neg_b, ckks_ctx)?;

    Ok(PackedMultivector::new(
        result_ct,
        a.batch_size,
        a.n,
        a.num_primes,
        a.level,
        a.scale,
    ))
}

/// CPU versions (placeholder)
#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn add_packed(
    _a: &PackedMultivector,
    _b: &PackedMultivector,
    _ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    Err("add_packed not yet implemented for CPU backend".to_string())
}

#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn subtract_packed(
    _a: &PackedMultivector,
    _b: &PackedMultivector,
    _ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    Err("subtract_packed not yet implemented for CPU backend".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Tests will be added once operations are implemented
}
