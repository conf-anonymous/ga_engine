//! Bootstrap Context - Main bootstrap API
//!
//! Provides the main interface for CKKS bootstrapping operations.

use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{SecretKey, EvaluationKey, KeyContext};
use super::sin_approx::{taylor_sin_coeffs, chebyshev_sin_coeffs};

/// Bootstrap parameters
///
/// Controls the precision and performance trade-offs in bootstrapping.
#[derive(Clone, Debug)]
pub struct BootstrapParams {
    /// Degree of sine polynomial approximation (15-31, must be odd)
    pub sin_degree: usize,

    /// Number of levels reserved for bootstrap operations
    pub bootstrap_levels: usize,

    /// Target precision after bootstrap (1e-6 to 1e-2)
    pub target_precision: f64,
}

impl BootstrapParams {
    /// Balanced bootstrap parameters (recommended)
    ///
    /// - Sine degree: 23 (good accuracy)
    /// - Bootstrap levels: 12
    /// - Target precision: 1e-4 (4 decimal places)
    ///
    /// # Example
    ///
    /// ```
    /// use ga_engine::clifford_fhe_v3::bootstrapping::BootstrapParams;
    ///
    /// let params = BootstrapParams::balanced();
    /// assert_eq!(params.sin_degree, 23);
    /// ```
    pub fn balanced() -> Self {
        BootstrapParams {
            sin_degree: 23,
            bootstrap_levels: 12,
            target_precision: 1e-4,
        }
    }

    /// Conservative bootstrap parameters (high precision)
    ///
    /// - Sine degree: 31 (high accuracy)
    /// - Bootstrap levels: 15
    /// - Target precision: 1e-6 (6 decimal places)
    ///
    /// Use when maximum precision is required, at cost of performance.
    pub fn conservative() -> Self {
        BootstrapParams {
            sin_degree: 31,
            bootstrap_levels: 15,
            target_precision: 1e-6,
        }
    }

    /// Fast bootstrap parameters (lower precision)
    ///
    /// - Sine degree: 15 (lower accuracy)
    /// - Bootstrap levels: 10
    /// - Target precision: 1e-2 (2 decimal places)
    ///
    /// Use when performance is critical and lower precision is acceptable.
    pub fn fast() -> Self {
        BootstrapParams {
            sin_degree: 15,
            bootstrap_levels: 10,
            target_precision: 1e-2,
        }
    }

    /// Validate bootstrap parameters
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid:
    /// - sin_degree must be odd and >= 5
    /// - bootstrap_levels must be >= 5 (10+ recommended for production)
    /// - target_precision must be > 0
    pub fn validate(&self) -> Result<(), String> {
        if self.sin_degree < 5 {
            return Err(format!("sin_degree must be >= 5, got {}", self.sin_degree));
        }
        if self.sin_degree % 2 == 0 {
            return Err(format!("sin_degree must be odd, got {}", self.sin_degree));
        }
        if self.bootstrap_levels < 5 {
            return Err(format!("bootstrap_levels must be >= 5, got {}", self.bootstrap_levels));
        }
        if self.bootstrap_levels < 10 {
            println!("  [WARNING] bootstrap_levels={} is below recommended minimum of 10", self.bootstrap_levels);
            println!("            Bootstrap may have reduced precision - OK for demos, NOT for production");
        }
        if self.target_precision <= 0.0 {
            return Err(format!("target_precision must be > 0, got {}", self.target_precision));
        }
        Ok(())
    }
}

/// Bootstrap context for CKKS bootstrapping
///
/// Manages all resources needed for bootstrapping operations:
/// - FHE parameters
/// - Bootstrap parameters
/// - Rotation keys (future)
/// - Precomputed sine coefficients
///
/// # Example
///
/// ```rust,ignore
/// use ga_engine::clifford_fhe_v3::bootstrapping::{BootstrapContext, BootstrapParams};
/// use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
///
/// // Setup parameters
/// let params = CliffordFHEParams::new_128bit();
/// let bootstrap_params = BootstrapParams::balanced();
///
/// // Create bootstrap context (generates rotation keys)
/// let bootstrap_ctx = BootstrapContext::new(
///     params,
///     bootstrap_params,
///     &secret_key,
/// )?;
///
/// // Bootstrap a noisy ciphertext
/// let ct_fresh = bootstrap_ctx.bootstrap(&ct_noisy)?;
/// ```
#[derive(Debug)]
pub struct BootstrapContext {
    params: CliffordFHEParams,
    bootstrap_params: BootstrapParams,
    sin_coeffs: Vec<f64>,
    rotation_keys: super::keys::RotationKeys,
    evk: EvaluationKey,
    key_ctx: KeyContext,
}

impl BootstrapContext {
    /// Create new bootstrap context
    ///
    /// Generates rotation keys and precomputes sine polynomial coefficients.
    ///
    /// # Arguments
    ///
    /// * `params` - FHE parameters (must have sufficient primes for bootstrap)
    /// * `bootstrap_params` - Bootstrap configuration
    /// * `secret_key` - Secret key for generating rotation keys
    ///
    /// # Returns
    ///
    /// Bootstrap context ready for bootstrapping operations
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Bootstrap parameters are invalid
    /// - FHE parameters have insufficient primes
    /// - Rotation key generation fails
    pub fn new(
        params: CliffordFHEParams,
        bootstrap_params: BootstrapParams,
        secret_key: &SecretKey,
    ) -> Result<Self, String> {
        // Validate bootstrap parameters
        bootstrap_params.validate()?;

        // Verify FHE parameters have sufficient primes
        // For production: bootstrap_levels + 3 for computation
        // For demos: bootstrap_levels + 1 minimum (at least 1 computation level)
        let min_computation_levels = if bootstrap_params.bootstrap_levels >= 10 { 3 } else { 1 };
        let required_primes = bootstrap_params.bootstrap_levels + min_computation_levels;

        if params.moduli.len() < required_primes {
            return Err(format!(
                "FHE parameters have {} primes, but {} required for bootstrap (bootstrap_levels={} + {})",
                params.moduli.len(),
                required_primes,
                bootstrap_params.bootstrap_levels,
                min_computation_levels
            ));
        }

        let actual_computation_levels = params.moduli.len() - bootstrap_params.bootstrap_levels - 1;
        if actual_computation_levels < 3 && bootstrap_params.bootstrap_levels >= 10 {
            println!("  [WARNING] Only {} computation levels available (3+ recommended)", actual_computation_levels);
        }

        println!("Creating bootstrap context:");
        println!("  Sine degree: {}", bootstrap_params.sin_degree);
        println!("  Bootstrap levels: {}", bootstrap_params.bootstrap_levels);
        println!("  Target precision: {}", bootstrap_params.target_precision);
        println!("  FHE primes: {}", params.moduli.len());

        // Precompute sine polynomial coefficients
        println!("  Precomputing sine polynomial coefficients...");
        let sin_coeffs = chebyshev_sin_coeffs(bootstrap_params.sin_degree);
        println!("  ✓ Sine coefficients computed ({} terms)", sin_coeffs.len());

        // Generate rotation keys for bootstrap
        println!("  Generating rotation keys...");
        let rotations = super::keys::required_rotations_for_bootstrap(params.n);
        let rotation_keys = super::keys::generate_rotation_keys(&rotations, secret_key, &params);
        println!("  ✓ Generated {} rotation keys", rotation_keys.num_keys());

        // Generate evaluation key for ciphertext multiplication
        println!("  Generating evaluation key...");
        let key_ctx = KeyContext::new(params.clone());
        let (_, _, evk) = key_ctx.keygen();
        println!("  ✓ Evaluation key generated");

        Ok(BootstrapContext {
            params,
            bootstrap_params,
            sin_coeffs,
            rotation_keys,
            evk,
            key_ctx,
        })
    }

    /// Bootstrap a ciphertext (refresh noise)
    ///
    /// Homomorphically decrypts and re-encrypts the ciphertext to remove noise,
    /// while keeping the plaintext data encrypted throughout.
    ///
    /// # Arguments
    ///
    /// * `ct` - Input ciphertext (may be noisy, low level)
    ///
    /// # Returns
    ///
    /// Fresh ciphertext with same plaintext, full levels restored, noise removed
    ///
    /// # Errors
    ///
    /// Returns error if bootstrap operations fail
    ///
    /// # Performance
    ///
    /// - CPU: ~1 second per ciphertext
    /// - GPU: ~200ms per ciphertext (future)
    pub fn bootstrap(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        println!("Starting bootstrap pipeline...");

        // Step 1: ModRaise - raise modulus to higher level
        println!("  [1/4] ModRaise...");
        let ct_raised = self.mod_raise(ct)?;

        // Step 2: CoeffToSlot - transform to evaluation form
        println!("  [2/4] CoeffToSlot...");
        let ct_slots = self.coeff_to_slot(&ct_raised)?;

        // Step 3: EvalMod - homomorphically evaluate modular reduction
        println!("  [3/4] EvalMod...");
        let ct_eval = self.eval_mod(&ct_slots)?;

        // Step 4: SlotToCoeff - transform back to coefficient form
        println!("  [4/4] SlotToCoeff...");
        let ct_coeffs = self.slot_to_coeff(&ct_eval)?;

        println!("  ✓ Bootstrap complete!");

        Ok(ct_coeffs)
    }

    /// Get sine polynomial coefficients
    ///
    /// Returns precomputed coefficients for sine approximation.
    pub fn sin_coeffs(&self) -> &[f64] {
        &self.sin_coeffs
    }

    /// Get bootstrap parameters
    pub fn bootstrap_params(&self) -> &BootstrapParams {
        &self.bootstrap_params
    }

    /// Get FHE parameters
    pub fn params(&self) -> &CliffordFHEParams {
        &self.params
    }

    // Internal operations

    fn mod_raise(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        // Use V3 mod_raise function
        use super::mod_raise::mod_raise;

        // ModRaise: Restore ciphertext to use ALL available primes
        // This gives maximum working room for bootstrap operations
        let target_level = self.params.moduli.len() - 1;  // Highest level (use all primes)

        if ct.level >= target_level {
            // Already at maximum level, no need to raise
            return Ok(ct.clone());
        }

        let target_moduli = &self.params.moduli[0..=target_level];
        mod_raise(ct, target_moduli)
    }

    fn coeff_to_slot(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        super::coeff_to_slot::coeff_to_slot(ct, &self.rotation_keys)
    }

    fn eval_mod(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        // Use the EvalMod implementation from eval_mod module
        let q = self.params.moduli[0];  // Use first prime as modulus
        super::eval_mod::eval_mod(
            ct,
            q,
            &self.sin_coeffs,
            &self.evk,
            &self.params,
            &self.key_ctx,
        )
    }

    fn slot_to_coeff(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        super::slot_to_coeff::slot_to_coeff(ct, &self.rotation_keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

    #[test]
    fn test_bootstrap_params_presets() {
        let balanced = BootstrapParams::balanced();
        assert_eq!(balanced.sin_degree, 23);
        assert_eq!(balanced.bootstrap_levels, 12);
        assert_eq!(balanced.target_precision, 1e-4);

        let conservative = BootstrapParams::conservative();
        assert_eq!(conservative.sin_degree, 31);

        let fast = BootstrapParams::fast();
        assert_eq!(fast.sin_degree, 15);
    }

    #[test]
    fn test_bootstrap_params_validation() {
        // Valid params
        let params = BootstrapParams::balanced();
        assert!(params.validate().is_ok());

        // Invalid: degree too small
        let mut params = BootstrapParams::balanced();
        params.sin_degree = 3;
        assert!(params.validate().is_err());

        // Invalid: degree even
        let mut params = BootstrapParams::balanced();
        params.sin_degree = 20;
        assert!(params.validate().is_err());

        // Warning: below recommended minimum (5 is allowed but warns)
        let mut params = BootstrapParams::balanced();
        params.bootstrap_levels = 5;
        assert!(params.validate().is_ok());  // Should pass with warning

        // Invalid: too few levels (below minimum of 5)
        let mut params = BootstrapParams::balanced();
        params.bootstrap_levels = 4;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_bootstrap_context_creation() {
        // Use large parameter set with many primes
        let params = CliffordFHEParams::new_128bit();  // 9 primes
        let key_ctx = KeyContext::new(params.clone());
        let (_, secret_key, _) = key_ctx.keygen();

        let bootstrap_params = BootstrapParams::fast();  // Requires 10+3=13 primes

        // This will fail because we only have 9 primes
        let result = BootstrapContext::new(params.clone(), bootstrap_params, &secret_key);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("primes"));

        // Try with balanced params (requires 12+3=15 primes) - also fails
        let bootstrap_params = BootstrapParams::balanced();
        let result = BootstrapContext::new(params, bootstrap_params, &secret_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_bootstrap_context_accessors() {
        // Use V3 minimal params which has 20 primes (sufficient for bootstrap)
        let params = CliffordFHEParams::new_v3_bootstrap_minimal();
        let key_ctx = KeyContext::new(params.clone());
        let (_, secret_key, _) = key_ctx.keygen();

        // Use custom bootstrap params that fit within 20 primes
        // With 20 primes: bootstrap_levels=12 means 20-12-1=7 computation levels (>= 3 required)
        let bootstrap_params = BootstrapParams {
            sin_degree: 15,
            bootstrap_levels: 12,
            target_precision: 1e-3,
        };

        // Create bootstrap context
        let ctx = BootstrapContext::new(params.clone(), bootstrap_params.clone(), &secret_key)
            .expect("Should create bootstrap context with V3 minimal params");

        // Test accessor methods
        assert_eq!(ctx.sin_coeffs().len(), 16); // degree 15 + 1 = 16 coefficients
        assert_eq!(ctx.bootstrap_params().sin_degree, 15);
        assert_eq!(ctx.bootstrap_params().bootstrap_levels, 12);
        assert_eq!(ctx.params().n, 8192);
        assert_eq!(ctx.params().moduli.len(), 20);

        // Verify sine coefficients are valid (odd function: even indices should be ~0)
        let coeffs = ctx.sin_coeffs();
        for i in (0..coeffs.len()).filter(|&i| i % 2 == 0) {
            assert!(coeffs[i].abs() < 1e-10,
                    "Coefficient {} should be ~0 for odd function, got {}", i, coeffs[i]);
        }

        // Verify first non-zero coefficient (x term) is near 1
        assert!((coeffs[1] - 1.0).abs() < 0.1,
                "x coefficient should be near 1.0, got {}", coeffs[1]);
    }

    #[test]
    fn test_sin_coeffs_precomputed() {
        // Test that sine coefficients are computed correctly
        // Use chebyshev_sin_coeffs directly (no need for full BootstrapContext)
        use crate::clifford_fhe_v3::bootstrapping::sin_approx::chebyshev_sin_coeffs;

        let sin_degree = 15;  // From BootstrapParams::fast()
        let coeffs = chebyshev_sin_coeffs(sin_degree);

        // Verify we got the right number of coefficients
        assert_eq!(coeffs.len(), sin_degree + 1);  // degree 15 + 1 = 16

        // Verify odd function (even powers except constant should be zero)
        for i in 0..coeffs.len() {
            if i % 2 == 0 && i > 0 {
                assert!(
                    coeffs[i].abs() < 1e-10,
                    "Coefficient {} should be ~0 (odd function), got {}",
                    i,
                    coeffs[i]
                );
            }
        }

        // Verify some non-zero odd coefficients exist (indices 1, 3, 5, ...)
        let odd_sum: f64 = (1..coeffs.len())
            .step_by(2)  // This gives 1, 3, 5, 7, ... (odd indices)
            .map(|i| coeffs[i].abs())
            .sum();
        assert!(
            odd_sum > 0.1,
            "Odd coefficients should be non-zero, sum = {}",
            odd_sum
        );

        // Verify first coefficient is 1.0 (x term in sin(x) = x - x³/3! + ...)
        assert!(
            (coeffs[1] - 1.0).abs() < 1e-10,
            "First non-zero coefficient should be 1.0, got {}",
            coeffs[1]
        );
    }
}
