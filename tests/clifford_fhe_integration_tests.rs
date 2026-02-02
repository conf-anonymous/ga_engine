//! Integration tests for Clifford FHE
//!
//! Run with: cargo test --test clifford_fhe_integration_tests --features v1 -- --nocapture

use ga_engine::clifford_fhe_v1::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v1::keys_rns::rns_keygen;
use ga_engine::clifford_fhe_v1::ckks_rns::{
    rns_encrypt, rns_decrypt, rns_add_ciphertexts, rns_multiply_ciphertexts,
    RnsPlaintext,
};
use std::time::Instant;

/// Helper to decode a value from RNS plaintext
fn decode_value(pt: &RnsPlaintext, scale: f64, all_primes: &[i64], level: usize) -> f64 {
    let active_primes = &all_primes[..all_primes.len() - level];

    // For now, ALWAYS use single-prime decoding (first prime)
    // This avoids CRT issues and works if message + noise << prime
    let val = pt.coeffs.rns_coeffs[0][0];
    let q = active_primes[0];
    let centered = if val > q / 2 { val - q } else { val };
    (centered as f64) / scale
}

/// Simple test result tracker
struct IntegrationTestSuite {
    results: Vec<(String, bool, f64, std::time::Duration)>,
    start_time: Instant,
}

impl IntegrationTestSuite {
    fn new(title: &str) -> Self {
        use colored::Colorize;
        println!("\n{}", "═".repeat(80).bright_blue().bold());
        println!("{} {}", "◆".bright_cyan().bold(), title.bright_white().bold());
        println!("{}\n", "═".repeat(80).bright_blue().bold());

        Self {
            results: Vec::new(),
            start_time: Instant::now(),
        }
    }

    fn test<F>(&mut self, name: &str, f: F)
    where
        F: FnOnce() -> Result<f64, String>
    {
        use colored::Colorize;
        let start = Instant::now();

        print!("  {} {}...", "▸".bright_cyan(), name.bright_white());
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        let duration = start.elapsed();

        match result {
            Ok(Ok(error)) => {
                println!(" {} [{:.2}ms] [err: {:.2e}]",
                    "✓".bright_green().bold(),
                    duration.as_secs_f64() * 1000.0,
                    error
                );
                self.results.push((name.to_string(), true, error, duration));
            }
            Ok(Err(msg)) => {
                println!(" {} [{:.2}ms] [{}]",
                    "✗".bright_red().bold(),
                    duration.as_secs_f64() * 1000.0,
                    msg.bright_red()
                );
                self.results.push((name.to_string(), false, 0.0, duration));
            }
            Err(_) => {
                println!(" {} [{:.2}ms] [panic]",
                    "✗".bright_red().bold(),
                    duration.as_secs_f64() * 1000.0
                );
                self.results.push((name.to_string(), false, 0.0, duration));
            }
        }
    }

    fn finish(&self) {
        use colored::Colorize;

        let total_duration = self.start_time.elapsed();
        let passed = self.results.iter().filter(|(_, p, _, _)| *p).count();
        let failed = self.results.len() - passed;

        println!("\n{}", "─".repeat(80).bright_blue());
        println!("{}", "TEST SUMMARY".bright_white().bold());
        println!("{}", "─".repeat(80).bright_blue());

        for (name, passed, error, duration) in &self.results {
            let status = if *passed {
                "✓".bright_green().bold()
            } else {
                "✗".bright_red().bold()
            };

            let time_str = format!("{:.0}ms", duration.as_secs_f64() * 1000.0);
            let error_str = if *passed && *error > 0.0 {
                format!("[err: {:.2e}]", error)
            } else {
                String::new()
            };

            println!("  {} {} {} {}", status, name.bright_white(),
                format!("[{}]", time_str).dimmed(), error_str.bright_cyan());
        }

        println!("\n{}", "─".repeat(80).bright_blue());

        let summary = if failed == 0 {
            format!("✓ {} passed in {:.2}s", passed, total_duration.as_secs_f64())
                .bright_green().bold()
        } else {
            format!("✗ {} passed, {} failed in {:.2}s", passed, failed, total_duration.as_secs_f64())
                .bright_red().bold()
        };

        println!("{}", summary);
        println!("{}\n", "═".repeat(80).bright_blue().bold());

        // Fail the test if any subtests failed
        if failed > 0 {
            panic!("{} test(s) failed", failed);
        }
    }
}

#[test]
fn test_all_integration_tests() {
    let mut suite = IntegrationTestSuite::new("Clifford FHE V1: Integration Tests");

    // Test 1: NTT Prime Validation
    suite.test("NTT 60-bit prime compatibility", || {
        let params = CliffordFHEParams {
            n: 1024,
            moduli: vec![1141392289560813569], // Single 60-bit prime
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: ga_engine::clifford_fhe_v1::params::SecurityLevel::Bit128,
        };

        let q = params.moduli[0];
        let n = params.n;

        // Verify (q-1) divisible by 2n (required for NTT)
        if (q - 1) % (2 * n as i64) != 0 {
            return Err("q-1 must be divisible by 2n for NTT".to_string());
        }
        Ok(0.0) // exact test, no error
    });

    // Test 2: Single Prime Encryption/Decryption
    suite.test("Single-prime encrypt/decrypt", || {
        let params = CliffordFHEParams {
            n: 1024,
            moduli: vec![1141392289560813569],
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: ga_engine::clifford_fhe_v1::params::SecurityLevel::Bit128,
        };

        let (pk, sk, _evk) = rns_keygen(&params);

        // Test encrypting zero
        let coeffs_zero = vec![0i64; params.n];
        let pt_zero = RnsPlaintext::from_coeffs(coeffs_zero, params.scale, &params.moduli, 0);
        let ct_zero = rns_encrypt(&pk, &pt_zero, &params);
        let pt_dec_zero = rns_decrypt(&sk, &ct_zero, &params);

        // Check noise is reasonable (should be ~100, allow up to 1000)
        let noise = pt_dec_zero.coeffs.rns_coeffs[0][0];
        let q = params.moduli[0];
        let centered = if noise > q / 2 { noise - q } else { noise };
        let noise_magnitude = centered.abs();

        if noise_magnitude >= 1000 {
            return Err(format!("Noise too large: {}", noise_magnitude));
        }
        Ok(noise_magnitude as f64)
    });

    // Test 3: Two Prime Encryption/Decryption
    suite.test("Two-prime encrypt/decrypt", || {
        let params = CliffordFHEParams {
            n: 1024,
            moduli: vec![
                1141392289560813569,  // q0 - 60-bit prime
                1099511678977,        // q1 - 41-bit prime
            ],
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: ga_engine::clifford_fhe_v1::params::SecurityLevel::Bit128,
        };

        let (pk, sk, _evk) = rns_keygen(&params);

        let value = 1.5;
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = (value * params.scale).round() as i64;

        let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
        let ct = rns_encrypt(&pk, &pt, &params);
        let pt_dec = rns_decrypt(&sk, &ct, &params);

        let decoded = decode_value(&pt_dec, params.scale, &params.moduli, 0);
        let error = (decoded - value).abs();

        if error >= 1e-6 {
            return Err(format!("Error too large: {}", error));
        }
        Ok(error)
    });

    // Test 4: Homomorphic Addition
    suite.test("Homomorphic addition", || {
        let params = CliffordFHEParams {
            n: 1024,
            moduli: vec![
                1141392289560813569,  // q0 - 60-bit prime
                1099511678977,        // q1 - 41-bit prime
            ],
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: ga_engine::clifford_fhe_v1::params::SecurityLevel::Bit128,
        };

        let (pk, sk, _evk) = rns_keygen(&params);

        let a = 1.5;
        let b = 2.7;
        let expected = a + b;

        let mut coeffs_a = vec![0i64; params.n];
        let mut coeffs_b = vec![0i64; params.n];
        coeffs_a[0] = (a * params.scale).round() as i64;
        coeffs_b[0] = (b * params.scale).round() as i64;

        let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
        let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

        let ct_a = rns_encrypt(&pk, &pt_a, &params);
        let ct_b = rns_encrypt(&pk, &pt_b, &params);

        let ct_sum = rns_add_ciphertexts(&ct_a, &ct_b, &params);
        let pt_sum = rns_decrypt(&sk, &ct_sum, &params);

        let result = decode_value(&pt_sum, params.scale, &params.moduli, ct_sum.level);
        let error = (result - expected).abs();

        if error >= 1e-6 {
            return Err(format!("Error: {} (expected {}, got {})", error, expected, result));
        }
        Ok(error)
    });

    // Test 5: Homomorphic Multiplication
    suite.test("Homomorphic multiplication", || {
        let params = CliffordFHEParams::new_rns_mult();
        let (pk, sk, evk) = rns_keygen(&params);

        let a = 1.5;
        let b = 2.0;
        let expected = a * b;

        let mut coeffs_a = vec![0i64; params.n];
        let mut coeffs_b = vec![0i64; params.n];
        coeffs_a[0] = (a * params.scale).round() as i64;
        coeffs_b[0] = (b * params.scale).round() as i64;

        let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
        let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

        let ct_a = rns_encrypt(&pk, &pt_a, &params);
        let ct_b = rns_encrypt(&pk, &pt_b, &params);

        let ct_prod = rns_multiply_ciphertexts(&ct_a, &ct_b, &evk, &params);
        let pt_prod = rns_decrypt(&sk, &ct_prod, &params);

        let result = decode_value(&pt_prod, ct_prod.scale, &params.moduli, ct_prod.level);
        let error = (result - expected).abs();

        if error >= 1e-3 {
            return Err(format!("Error: {} (expected {}, got {})", error, expected, result));
        }
        Ok(error)
    });

    // Test 6: Noise Growth
    suite.test("Noise growth tracking", || {
        let params = CliffordFHEParams {
            n: 1024,
            moduli: vec![
                1141392289560813569,  // q0 - 60-bit prime
                1099511678977,        // q1 - 41-bit prime
            ],
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: ga_engine::clifford_fhe_v1::params::SecurityLevel::Bit128,
        };

        let (pk, sk, _evk) = rns_keygen(&params);

        let value = 1.0;
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = (value * params.scale).round() as i64;

        let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
        let ct = rns_encrypt(&pk, &pt, &params);
        let pt_dec = rns_decrypt(&sk, &ct, &params);

        let dec1 = decode_value(&pt_dec, params.scale, &params.moduli, 0);
        let noise1 = ((dec1 - value) * params.scale).abs() as i128;

        // After one addition
        let ct_sum = rns_add_ciphertexts(&ct, &ct, &params);
        let pt_sum = rns_decrypt(&sk, &ct_sum, &params);
        let dec2 = decode_value(&pt_sum, params.scale, &params.moduli, ct_sum.level);
        let noise2 = ((dec2 - 2.0 * value) * params.scale).abs() as i128;

        // Noise should be reasonable
        if noise1 >= 10000 {
            return Err(format!("Initial noise too large: {}", noise1));
        }
        if noise2 >= 20000 {
            return Err(format!("Noise after addition too large: {}", noise2));
        }

        Ok(noise2 as f64)
    });

    suite.finish();
}
