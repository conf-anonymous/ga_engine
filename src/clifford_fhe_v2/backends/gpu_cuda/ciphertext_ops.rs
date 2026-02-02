///! Extension methods for CudaCiphertext to match Metal API
///!
///! Implements methods like add(), rotate_by_steps(), multiply_plain()
///! directly on CudaCiphertext to provide the same API as MetalCiphertext.
///!
///! This allows V4 code to work seamlessly with CUDA backend.

use super::ckks::{CudaCiphertext, CudaCkksContext, CudaPlaintext};
use super::rotation_keys::CudaRotationKeys;

impl CudaCiphertext {
    /// Add two ciphertexts (component-wise polynomial addition)
    ///
    /// Mirrors MetalCiphertext::add() API
    pub fn add(&self, other: &Self, ctx: &CudaCkksContext) -> Result<Self, String> {
        // Delegate to CudaCkksContext::add
        ctx.add(self, other)
    }

    /// Subtract two ciphertexts
    ///
    /// Mirrors MetalCiphertext::subtract() API
    pub fn subtract(&self, other: &Self, ctx: &CudaCkksContext) -> Result<Self, String> {
        assert_eq!(self.n, other.n, "Dimensions must match");
        assert_eq!(self.level, other.level, "Levels must match");
        assert_eq!(self.num_primes, other.num_primes, "Number of primes must match");

        // Use subtract_polynomials_gpu for component-wise subtraction
        let num_primes_active = self.level + 1;

        let c0_sub = ctx.subtract_polynomials_gpu(&self.c0, &other.c0, num_primes_active)?;
        let c1_sub = ctx.subtract_polynomials_gpu(&self.c1, &other.c1, num_primes_active)?;

        Ok(Self {
            c0: c0_sub,
            c1: c1_sub,
            n: self.n,
            num_primes: self.num_primes,
            level: self.level,
            scale: self.scale,
        })
    }

    /// Multiply ciphertext by plaintext
    ///
    /// Mirrors MetalCiphertext::multiply_plain() API
    pub fn multiply_plain(
        &self,
        plaintext: &CudaPlaintext,
        ctx: &CudaCkksContext,
    ) -> Result<Self, String> {
        let num_primes_active = self.level + 1;

        if plaintext.num_primes < num_primes_active {
            return Err(format!(
                "Plaintext has {} primes but ciphertext is at level {} (needs {} primes)",
                plaintext.num_primes, self.level, num_primes_active
            ));
        }

        // Multiply c0 and c1 by plaintext (stride=num_primes, num_primes=num_primes)
        let c0_mult = ctx.pointwise_multiply_polynomials_gpu_strided(
            &self.c0,
            &plaintext.poly,
            self.num_primes,
            self.num_primes,
        )?;

        let c1_mult = ctx.pointwise_multiply_polynomials_gpu_strided(
            &self.c1,
            &plaintext.poly,
            self.num_primes,
            self.num_primes,
        )?;

        // Rescale to maintain scale
        let c0_rescaled = ctx.exact_rescale_gpu_strided(&c0_mult, self.level)?;
        let c1_rescaled = ctx.exact_rescale_gpu_strided(&c1_mult, self.level)?;

        // After rescaling, we drop one prime, so update both level and num_primes
        let new_level = self.level.saturating_sub(1);
        let new_num_primes = new_level + 1;

        Ok(Self {
            c0: c0_rescaled,
            c1: c1_rescaled,
            n: self.n,
            num_primes: new_num_primes,
            level: new_level,
            scale: self.scale * plaintext.scale / ctx.params().scale,
        })
    }

    /// Rotate ciphertext by given number of slots
    ///
    /// Mirrors MetalCiphertext::rotate_by_steps() API (3 parameters)
    /// Based on V3 CUDA bootstrap implementation
    pub fn rotate_by_steps(
        &self,
        step: i32,
        rot_keys: &CudaRotationKeys,
        ctx: &CudaCkksContext,
    ) -> Result<Self, String> {
        let n = self.n;
        let num_primes = self.num_primes;
        let level = self.level;

        // Convert to flat RNS layout
        let c0_flat = ctx.strided_to_flat(&self.c0, n, num_primes, num_primes);
        let c1_flat = ctx.strided_to_flat(&self.c1, n, num_primes, num_primes);

        // Access rotation context through rotation keys (like V3 CUDA bootstrap)
        let rot_ctx = rot_keys.rotation_context();

        // Apply Galois automorphism to c0 and c1 using GPU
        let c0_galois = rot_ctx.rotate_gpu(&c0_flat, step, num_primes)?;
        let c1_galois = rot_ctx.rotate_gpu(&c1_flat, step, num_primes)?;

        // Compute Galois element for this rotation
        let galois_elt = rot_ctx.galois_element(step);

        // Apply rotation key to c1(X^g) using GPU NTT
        let (c0_ks, c1_ks) = rot_keys.apply_rotation_key_gpu(
            &c1_galois,
            galois_elt,
            level,
            ctx.ntt_contexts(),
        )?;

        // Add c0(X^g) + c0_ks
        let c0_result = ctx.add_polynomials_gpu(&c0_galois, &c0_ks, num_primes)?;

        // Convert back from flat to strided layout
        let c0_strided = ctx.flat_to_strided(&c0_result, n, num_primes, num_primes);
        let c1_strided = ctx.flat_to_strided(&c1_ks, n, num_primes, num_primes);

        Ok(Self {
            c0: c0_strided,
            c1: c1_strided,
            n,
            num_primes,
            level,
            scale: self.scale,
        })
    }

    /// Batch rotation with hoisting (multiple rotations optimized)
    ///
    /// Mirrors MetalCiphertext::rotate_batch_with_hoisting() API (3 parameters)
    pub fn rotate_batch_with_hoisting(
        &self,
        steps: &[i32],
        rot_keys: &CudaRotationKeys,
        ctx: &CudaCkksContext,
    ) -> Result<Vec<Self>, String> {
        if steps.is_empty() {
            return Ok(vec![]);
        }

        // For now, implement as sequential rotations
        // TODO: Implement true hoisting optimization later
        let mut results = Vec::with_capacity(steps.len());

        for &step in steps {
            let rotated = self.rotate_by_steps(step, rot_keys, ctx)?;
            results.push(rotated);
        }

        Ok(results)
    }
}

