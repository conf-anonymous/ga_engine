//! Trace the full relinearization loop to find where CPU and Metal diverge

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::KeyContext,
        multiplication::multiply_ciphertexts,
        ckks::{CkksContext, Ciphertext},
        rns::RnsRepresentation,
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use num_bigint::BigInt;
#[cfg(feature = "v2-gpu-metal")]
use num_traits::{One, Zero, ToPrimitive};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Tracing Relinearization Loop");
    println!("=============================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let device = Arc::new(MetalDevice::new()?);
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();

    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cpu_ctx = CkksContext::new(params.clone());

    // Generate Metal relin keys
    let ntt_contexts = metal_ctx.ntt_contexts();
    let metal_relin_keys = MetalRelinKeys::generate(
        device.clone(),
        &sk,
        &params,
        &ntt_contexts,
        20,
    )?;

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    // Create CPU ciphertext
    let cpu_pt_a = cpu_ctx.encode(&[2.0]);
    let cpu_pt_b = cpu_ctx.encode(&[3.0]);
    let cpu_ct_a = cpu_ctx.encrypt(&cpu_pt_a, &pk);
    let cpu_ct_b = cpu_ctx.encrypt(&cpu_pt_b, &pk);

    // Compute tensor product (same code as CPU)
    println!("Computing tensor products...\n");

    // CPU tensor product
    let ct0_ct0 = multiply_polys_cpu(&cpu_ct_a.c0, &cpu_ct_a.c0, &key_ctx, moduli);
    let ct0_ct1 = multiply_polys_cpu(&cpu_ct_a.c0, &cpu_ct_b.c1, &key_ctx, moduli);
    let ct1_ct0 = multiply_polys_cpu(&cpu_ct_a.c1, &cpu_ct_b.c0, &key_ctx, moduli);
    let ct1_ct1 = multiply_polys_cpu(&cpu_ct_a.c1, &cpu_ct_b.c1, &key_ctx, moduli);

    // d0 = ct0_a * ct0_b
    // d1 = ct0_a * ct1_b + ct1_a * ct0_b
    // d2 = ct1_a * ct1_b
    let d0 = ct0_ct0.clone();
    let mut d1 = Vec::with_capacity(n);
    for i in 0..n {
        d1.push(ct0_ct1[i].add(&ct1_ct0[i]));
    }
    let d2 = ct1_ct1;

    println!("CPU d2[0]: {:?}", d2[0].values);

    // Perform CPU relinearization manually
    println!("\n=== Manual CPU Relinearization ===");
    let (cpu_c0_final, cpu_c1_final) = manual_relin_cpu(
        &d0, &d1, &d2, &evk.evk0, &evk.evk1, 20, &key_ctx, moduli, n
    )?;

    // Verify CPU result
    // After multiplication, scale = scale1 × scale2 = (2^40)² = 2^80
    let scale_squared = (1u128 << 80) as f64;  // 2^80
    let cpu_result = verify_decryption(&cpu_c0_final, &cpu_c1_final, &sk, moduli, n, scale_squared)?;
    println!("\nCPU relin result (before rescale): {}\n", cpu_result);

    // Now do Metal relinearization with SAME tensor product
    println!("=== Manual Metal Relinearization ===");

    // Convert d0, d1, d2 to flat format for Metal
    let d0_flat = rns_to_flat(&d0, num_primes);
    let d1_flat = rns_to_flat(&d1, num_primes);
    let d2_flat = rns_to_flat(&d2, num_primes);

    println!("Metal d2_flat[0] across primes: {:?}",
        (0..num_primes).map(|j| d2_flat[0 * num_primes + j]).collect::<Vec<_>>());

    let (metal_evk0, metal_evk1) = metal_relin_keys.get_coeff_keys(level)?;
    let (base_w, _) = metal_relin_keys.gadget_params();

    let (metal_c0_final, metal_c1_final) = manual_relin_metal(
        &d0_flat, &d1_flat, &d2_flat, &metal_evk0, &metal_evk1,
        base_w, moduli, n, &metal_ctx
    )?;

    // Verify Metal result
    let metal_result = verify_decryption_flat(&metal_c0_final, &metal_c1_final, &sk, moduli, n, scale_squared)?;
    println!("\nMetal relin result (before rescale): {}", metal_result);

    println!("\n=== Comparison ===");
    println!("CPU result: {}", cpu_result);
    println!("Metal result: {}", metal_result);

    Ok(())
}

#[cfg(feature = "v2-gpu-metal")]
fn multiply_polys_cpu(
    a: &[RnsRepresentation],
    b: &[RnsRepresentation],
    key_ctx: &KeyContext,
    moduli: &[u64],
) -> Vec<RnsRepresentation> {
    let n = a.len();
    let num_primes = moduli.len();

    let mut result = vec![RnsRepresentation::new(vec![0; num_primes], moduli.to_vec()); n];

    for (j, &q) in moduli.iter().enumerate() {
        let ntt_ctx = &key_ctx.ntt_contexts[j];

        let a_poly: Vec<u64> = a.iter().map(|r| r.values[j]).collect();
        let b_poly: Vec<u64> = b.iter().map(|r| r.values[j]).collect();

        let prod = ntt_ctx.multiply_polynomials(&a_poly, &b_poly);

        for i in 0..n {
            result[i].values[j] = prod[i];
        }
    }

    result
}

#[cfg(feature = "v2-gpu-metal")]
fn rns_to_flat(poly: &[RnsRepresentation], num_primes: usize) -> Vec<u64> {
    let n = poly.len();
    let mut flat = vec![0u64; n * num_primes];
    for i in 0..n {
        for j in 0..num_primes {
            flat[i * num_primes + j] = poly[i].values[j];
        }
    }
    flat
}

#[cfg(feature = "v2-gpu-metal")]
fn flat_to_rns(flat: &[u64], n: usize, moduli: &[u64]) -> Vec<RnsRepresentation> {
    let num_primes = moduli.len();
    let mut poly = Vec::with_capacity(n);
    for i in 0..n {
        let mut vals = vec![0u64; num_primes];
        for j in 0..num_primes {
            vals[j] = flat[i * num_primes + j];
        }
        poly.push(RnsRepresentation::new(vals, moduli.to_vec()));
    }
    poly
}

#[cfg(feature = "v2-gpu-metal")]
fn manual_relin_cpu(
    d0: &[RnsRepresentation],
    d1: &[RnsRepresentation],
    d2: &[RnsRepresentation],
    evk0: &[Vec<RnsRepresentation>],
    evk1: &[Vec<RnsRepresentation>],
    base_w: u32,
    key_ctx: &KeyContext,
    moduli: &[u64],
    n: usize,
) -> Result<(Vec<RnsRepresentation>, Vec<RnsRepresentation>), String> {
    let num_primes = moduli.len();

    // Initialize c0 = d0, c1 = d1
    let mut c0 = d0.to_vec();
    let mut c1 = d1.to_vec();

    // Gadget decompose d2
    let d2_digits = gadget_decompose_cpu(d2, base_w, moduli);

    println!("  Number of digits: {}", d2_digits.len());
    println!("  First digit[0] coeff[0]: {:?}", d2_digits[0][0].values);

    // For each digit
    for (t, d2_digit) in d2_digits.iter().enumerate() {
        if t >= evk0.len() {
            break;
        }

        // term0 = d2_digit × evk0[t]
        let term0 = multiply_polys_cpu(d2_digit, &evk0[t], key_ctx, moduli);
        // term1 = d2_digit × evk1[t]
        let term1 = multiply_polys_cpu(d2_digit, &evk1[t], key_ctx, moduli);

        if t == 0 {
            println!("  CPU digit[{}] × evk0 = term0[0]: {:?}", t, term0[0].values);
            println!("  CPU c0[0] BEFORE: {:?}", c0[0].values);
        }

        // c0 -= term0, c1 += term1
        for i in 0..n {
            c0[i] = c0[i].sub(&term0[i]);
            c1[i] = c1[i].add(&term1[i]);
        }

        if t == 0 {
            println!("  CPU c0[0] AFTER: {:?}", c0[0].values);
        }
    }

    Ok((c0, c1))
}

#[cfg(feature = "v2-gpu-metal")]
fn manual_relin_metal(
    d0_flat: &[u64],
    d1_flat: &[u64],
    d2_flat: &[u64],
    evk0: &[Vec<u64>],
    evk1: &[Vec<u64>],
    base_w: u32,
    moduli: &[u64],
    n: usize,
    ctx: &MetalCkksContext,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let num_primes = moduli.len();

    // Initialize c0 = d0, c1 = d1
    let mut c0 = d0_flat.to_vec();
    let mut c1 = d1_flat.to_vec();

    // Gadget decompose d2
    let d2_digits = MetalCiphertext::gadget_decompose_flat(d2_flat, base_w, moduli, n)?;

    println!("  Number of digits: {}", d2_digits.len());
    println!("  First digit[0] coeff[0]: {:?}",
        (0..num_primes).map(|j| d2_digits[0][0 * num_primes + j]).collect::<Vec<_>>());

    // For each digit
    for (t, d2_digit) in d2_digits.iter().enumerate() {
        if t >= evk0.len() {
            break;
        }

        // term0 = d2_digit × evk0[t] using NTT
        let term0 = ctx.multiply_polys_flat_ntt_negacyclic(d2_digit, &evk0[t], moduli)?;
        // term1 = d2_digit × evk1[t] using NTT
        let term1 = ctx.multiply_polys_flat_ntt_negacyclic(d2_digit, &evk1[t], moduli)?;

        if t == 0 {
            println!("  Metal digit[{}] × evk0 = term0[0]: {:?}", t,
                (0..num_primes).map(|j| term0[0 * num_primes + j]).collect::<Vec<_>>());
            println!("  Metal c0[0] BEFORE: {:?}",
                (0..num_primes).map(|j| c0[0 * num_primes + j]).collect::<Vec<_>>());
        }

        // c0 -= term0, c1 += term1
        for i in 0..(n * num_primes) {
            let q = moduli[i % num_primes];
            c0[i] = if c0[i] >= term0[i] {
                c0[i] - term0[i]
            } else {
                q - (term0[i] - c0[i])
            };
            c1[i] = ((c1[i] as u128 + term1[i] as u128) % q as u128) as u64;
        }

        if t == 0 {
            println!("  Metal c0[0] AFTER: {:?}",
                (0..num_primes).map(|j| c0[0 * num_primes + j]).collect::<Vec<_>>());
        }
    }

    Ok((c0, c1))
}

#[cfg(feature = "v2-gpu-metal")]
fn gadget_decompose_cpu(
    poly: &[RnsRepresentation],
    base_w: u32,
    moduli: &[u64],
) -> Vec<Vec<RnsRepresentation>> {
    let n = poly.len();
    let num_primes = moduli.len();

    // Compute Q = product of all primes
    let q_prod_big: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
    let q_half_big = &q_prod_big / 2;
    let base_big = BigInt::one() << base_w;
    let half_base_big = &base_big / 2;

    // Determine number of digits
    let q_bits = q_prod_big.bits() as u32;
    let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

    let mut digits = vec![vec![RnsRepresentation::new(vec![0; num_primes], moduli.to_vec()); n]; num_digits];

    for i in 0..n {
        // CRT reconstruct
        let x_big = crt_reconstruct(&poly[i].values, moduli);

        // Center-lift
        let x_centered = if x_big > q_half_big {
            x_big - &q_prod_big
        } else {
            x_big
        };

        // Balanced decomposition
        let mut remainder = x_centered;

        for t in 0..num_digits {
            let dt_unbalanced = &remainder % &base_big;
            let dt = if dt_unbalanced > half_base_big {
                &dt_unbalanced - &base_big
            } else {
                dt_unbalanced
            };

            // Map to RNS
            for (j, &q) in moduli.iter().enumerate() {
                let q_big = BigInt::from(q);
                let mut dt_mod = &dt % &q_big;
                if dt_mod.sign() == num_bigint::Sign::Minus {
                    dt_mod += &q_big;
                }
                digits[t][i].values[j] = dt_mod.to_u64().unwrap();
            }

            remainder = (remainder - &dt) / &base_big;
        }
    }

    digits
}

#[cfg(feature = "v2-gpu-metal")]
fn crt_reconstruct(residues: &[u64], moduli: &[u64]) -> BigInt {
    let q_prod: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();

    let mut x = BigInt::zero();
    for (i, &ri) in residues.iter().enumerate() {
        let qi = BigInt::from(moduli[i]);
        let q_i = &q_prod / &qi;

        let qi_inv = mod_inverse(&q_i, &qi);

        let ri_big = BigInt::from(ri);
        let basis = (&q_i * &qi_inv) % &q_prod;
        let term = (ri_big * basis) % &q_prod;
        x = (&x + term) % &q_prod;
    }

    if x.sign() == num_bigint::Sign::Minus {
        x += &q_prod;
    }

    x
}

#[cfg(feature = "v2-gpu-metal")]
fn mod_inverse(a: &BigInt, m: &BigInt) -> BigInt {
    let mut t = BigInt::zero();
    let mut newt = BigInt::one();
    let mut r = m.clone();
    let mut newr = a.clone();

    while !newr.is_zero() {
        let quotient = &r / &newr;
        let temp_t = t.clone();
        t = newt.clone();
        newt = temp_t - &quotient * &newt;

        let temp_r = r.clone();
        r = newr.clone();
        newr = temp_r - &quotient * &newr;
    }

    if t.sign() == num_bigint::Sign::Minus {
        t += m;
    }

    t
}

#[cfg(feature = "v2-gpu-metal")]
fn verify_decryption(
    c0: &[RnsRepresentation],
    c1: &[RnsRepresentation],
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    moduli: &[u64],
    n: usize,
    scale: f64,
) -> Result<f64, String> {
    let num_primes = moduli.len();

    // Compute m = c0 + c1 * s
    let mut m = Vec::with_capacity(n);
    for i in 0..n {
        let mut vals = vec![0u64; num_primes];
        for (j, &q) in moduli.iter().enumerate() {
            let c0_val = c0[i].values[j];
            let c1_val = c1[i].values[j];
            let s_val = sk.coeffs[i].values[j];

            let c1_s = ((c1_val as u128 * s_val as u128) % q as u128) as u64;
            vals[j] = ((c0_val as u128 + c1_s as u128) % q as u128) as u64;
        }
        m.push(RnsRepresentation::new(vals, moduli.to_vec()));
    }

    // CRT reconstruct first coefficient and decode
    let m0 = crt_reconstruct(&m[0].values, moduli);
    let q_prod: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
    let q_half = &q_prod / 2;

    let m0_centered = if m0 > q_half {
        m0 - &q_prod
    } else {
        m0
    };

    // Convert to float
    let m0_float = m0_centered.to_f64().unwrap_or(0.0);

    Ok(m0_float / scale)
}

#[cfg(feature = "v2-gpu-metal")]
fn verify_decryption_flat(
    c0: &[u64],
    c1: &[u64],
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    moduli: &[u64],
    n: usize,
    scale: f64,
) -> Result<f64, String> {
    let num_primes = moduli.len();

    // Convert to RNS format
    let c0_rns = flat_to_rns(c0, n, moduli);
    let c1_rns = flat_to_rns(c1, n, moduli);

    verify_decryption(&c0_rns, &c1_rns, sk, moduli, n, scale)
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
