//! Step-by-step comparison of Metal vs CPU relinearization
//! Uses SAME ciphertext and SAME EVK to find where they diverge

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::{
    backends::gpu_metal::{
        ckks::{MetalCkksContext, MetalCiphertext},
        relin_keys::MetalRelinKeys,
        device::MetalDevice,
    },
    backends::cpu_optimized::{
        keys::{KeyContext, EvaluationKey},
        multiplication::multiply_ciphertexts,
        ckks::{CkksContext, Ciphertext},
        rns::RnsRepresentation,
        ntt::NttContext,
    },
    params::CliffordFHEParams,
};
#[cfg(feature = "v2-gpu-metal")]
use std::sync::Arc;

#[cfg(feature = "v2-gpu-metal")]
fn main() -> Result<(), String> {
    println!("Step-by-Step Metal vs CPU Relinearization");
    println!("==========================================\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let device = Arc::new(MetalDevice::new()?);
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();

    let metal_ctx = MetalCkksContext::new(params.clone())?;
    let cpu_ctx = CkksContext::new(params.clone());

    let n = params.n;
    let level = 2;
    let num_primes = level + 1;
    let moduli = &params.moduli[..num_primes];

    // Create CPU ciphertext
    let cpu_pt_a = cpu_ctx.encode(&[2.0]);
    let cpu_pt_b = cpu_ctx.encode(&[3.0]);
    let cpu_ct_a = cpu_ctx.encrypt(&cpu_pt_a, &pk);
    let cpu_ct_b = cpu_ctx.encrypt(&cpu_pt_b, &pk);

    // Convert CPU EVK to flat format for Metal
    let (evk0_flat, evk1_flat) = cpu_evk_to_flat(&evk, num_primes)?;

    // Convert CPU ciphertext to flat format
    let ct_a_flat = cpu_ct_to_flat(&cpu_ct_a)?;
    let ct_b_flat = cpu_ct_to_flat(&cpu_ct_b)?;

    println!("Using SAME ciphertext and SAME EVK for both CPU and Metal\n");

    // Step 1: Compute tensor product
    println!("=== Step 1: Tensor Product ===");

    // CPU tensor product
    let (cpu_d0, cpu_d1, cpu_d2) = compute_tensor_product_cpu(&cpu_ct_a, &cpu_ct_b, &key_ctx, moduli);
    println!("CPU d2[0]: {:?}", cpu_d2[0].values);

    // Metal tensor product
    let (metal_d0, metal_d1, metal_d2) = compute_tensor_product_metal(
        &ct_a_flat.0, &ct_a_flat.1,
        &ct_b_flat.0, &ct_b_flat.1,
        &metal_ctx, moduli, n
    )?;
    println!("Metal d2[0]: {:?}",
        (0..num_primes).map(|j| metal_d2[0 * num_primes + j]).collect::<Vec<_>>());

    // Compare
    let tensor_match = (0..num_primes).all(|j|
        cpu_d2[0].values[j] == metal_d2[0 * num_primes + j]
    );
    println!("Tensor product match: {}\n", if tensor_match { "✅" } else { "❌" });

    // Step 2: Gadget decomposition
    println!("=== Step 2: Gadget Decomposition ===");
    let base_w = evk.base_w;

    let cpu_digits = gadget_decompose_cpu(&cpu_d2, base_w, moduli);
    println!("CPU num_digits: {}", cpu_digits.len());
    println!("CPU digit[0][0]: {:?}", cpu_digits[0][0].values);

    let metal_digits = MetalCiphertext::gadget_decompose_flat(&metal_d2, base_w, moduli, n)?;
    println!("Metal num_digits: {}", metal_digits.len());
    println!("Metal digit[0][0]: {:?}",
        (0..num_primes).map(|j| metal_digits[0][0 * num_primes + j]).collect::<Vec<_>>());

    let decomp_match = (0..num_primes).all(|j|
        cpu_digits[0][0].values[j] == metal_digits[0][0 * num_primes + j]
    );
    println!("Gadget decomposition match: {}\n", if decomp_match { "✅" } else { "❌" });

    // Step 3: First digit × EVK multiplication
    println!("=== Step 3: First Digit × EVK ===");

    // CPU: digit[0] × evk0[0]
    let cpu_term0 = multiply_polys_cpu(&cpu_digits[0], &evk.evk0[0], &key_ctx, moduli);
    println!("CPU term0[0]: {:?}", cpu_term0[0].values);

    // Metal: digit[0] × evk0[0]
    let metal_term0 = metal_ctx.multiply_polys_flat_ntt_negacyclic(
        &metal_digits[0],
        &evk0_flat[0],
        moduli
    )?;
    println!("Metal term0[0]: {:?}",
        (0..num_primes).map(|j| metal_term0[0 * num_primes + j]).collect::<Vec<_>>());

    let term0_match = (0..num_primes).all(|j|
        cpu_term0[0].values[j] == metal_term0[0 * num_primes + j]
    );
    println!("term0 match: {}\n", if term0_match { "✅" } else { "❌" });

    // Step 4: Full relinearization
    println!("=== Step 4: Full Relinearization ===");

    // CPU relinearization
    let (cpu_c0, cpu_c1) = relinearize_cpu(&cpu_d0, &cpu_d1, &cpu_d2, &evk, &key_ctx, moduli);
    println!("CPU c0[0] after relin: {:?}", cpu_c0[0].values);
    println!("CPU c1[0] after relin: {:?}", cpu_c1[0].values);

    // Metal relinearization
    let (metal_c0, metal_c1) = relinearize_metal(
        &metal_d0, &metal_d1, &metal_d2,
        &evk0_flat, &evk1_flat,
        base_w, &metal_ctx, moduli, n
    )?;
    println!("Metal c0[0] after relin: {:?}",
        (0..num_primes).map(|j| metal_c0[0 * num_primes + j]).collect::<Vec<_>>());
    println!("Metal c1[0] after relin: {:?}",
        (0..num_primes).map(|j| metal_c1[0 * num_primes + j]).collect::<Vec<_>>());

    let relin_match = (0..num_primes).all(|j|
        cpu_c0[0].values[j] == metal_c0[0 * num_primes + j] &&
        cpu_c1[0].values[j] == metal_c1[0 * num_primes + j]
    );
    println!("Relinearization match: {}\n", if relin_match { "✅" } else { "❌" });

    if relin_match {
        println!("✅ All steps match!");
        Ok(())
    } else {
        println!("❌ Mismatch found!");
        Err("Steps don't match".to_string())
    }
}

#[cfg(feature = "v2-gpu-metal")]
fn cpu_evk_to_flat(evk: &EvaluationKey, num_primes: usize) -> Result<(Vec<Vec<u64>>, Vec<Vec<u64>>), String> {
    let num_digits = evk.evk0.len();
    let n = evk.evk0[0].len();

    let mut evk0_flat = Vec::with_capacity(num_digits);
    let mut evk1_flat = Vec::with_capacity(num_digits);

    for t in 0..num_digits {
        let mut e0 = vec![0u64; n * num_primes];
        let mut e1 = vec![0u64; n * num_primes];

        for i in 0..n {
            for j in 0..num_primes {
                e0[i * num_primes + j] = evk.evk0[t][i].values[j];
                e1[i * num_primes + j] = evk.evk1[t][i].values[j];
            }
        }

        evk0_flat.push(e0);
        evk1_flat.push(e1);
    }

    Ok((evk0_flat, evk1_flat))
}

#[cfg(feature = "v2-gpu-metal")]
fn cpu_ct_to_flat(ct: &Ciphertext) -> Result<(Vec<u64>, Vec<u64>), String> {
    let n = ct.c0.len();
    let num_primes = ct.level + 1;

    let mut c0 = vec![0u64; n * num_primes];
    let mut c1 = vec![0u64; n * num_primes];

    for i in 0..n {
        for j in 0..num_primes {
            c0[i * num_primes + j] = ct.c0[i].values[j];
            c1[i * num_primes + j] = ct.c1[i].values[j];
        }
    }

    Ok((c0, c1))
}

#[cfg(feature = "v2-gpu-metal")]
fn compute_tensor_product_cpu(
    ct_a: &Ciphertext,
    ct_b: &Ciphertext,
    key_ctx: &KeyContext,
    moduli: &[u64],
) -> (Vec<RnsRepresentation>, Vec<RnsRepresentation>, Vec<RnsRepresentation>) {
    let n = ct_a.c0.len();

    let ct0_ct0 = multiply_polys_cpu(&ct_a.c0, &ct_b.c0, key_ctx, moduli);
    let ct0_ct1 = multiply_polys_cpu(&ct_a.c0, &ct_b.c1, key_ctx, moduli);
    let ct1_ct0 = multiply_polys_cpu(&ct_a.c1, &ct_b.c0, key_ctx, moduli);
    let ct1_ct1 = multiply_polys_cpu(&ct_a.c1, &ct_b.c1, key_ctx, moduli);

    let d0 = ct0_ct0;
    let mut d1 = Vec::with_capacity(n);
    for i in 0..n {
        d1.push(ct0_ct1[i].add(&ct1_ct0[i]));
    }
    let d2 = ct1_ct1;

    (d0, d1, d2)
}

#[cfg(feature = "v2-gpu-metal")]
fn compute_tensor_product_metal(
    c0_a: &[u64], c1_a: &[u64],
    c0_b: &[u64], c1_b: &[u64],
    ctx: &MetalCkksContext,
    moduli: &[u64],
    n: usize,
) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>), String> {
    let num_primes = moduli.len();

    let ct0_ct0 = ctx.multiply_polys_flat_ntt_negacyclic(c0_a, c0_b, moduli)?;
    let ct0_ct1 = ctx.multiply_polys_flat_ntt_negacyclic(c0_a, c1_b, moduli)?;
    let ct1_ct0 = ctx.multiply_polys_flat_ntt_negacyclic(c1_a, c0_b, moduli)?;
    let ct1_ct1 = ctx.multiply_polys_flat_ntt_negacyclic(c1_a, c1_b, moduli)?;

    let d0 = ct0_ct0;
    let mut d1 = vec![0u64; n * num_primes];
    for i in 0..(n * num_primes) {
        let q = moduli[i % num_primes];
        d1[i] = ((ct0_ct1[i] as u128 + ct1_ct0[i] as u128) % q as u128) as u64;
    }
    let d2 = ct1_ct1;

    Ok((d0, d1, d2))
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

    for (j, &_q) in moduli.iter().enumerate() {
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
fn gadget_decompose_cpu(
    poly: &[RnsRepresentation],
    base_w: u32,
    moduli: &[u64],
) -> Vec<Vec<RnsRepresentation>> {
    use num_bigint::BigInt;
    use num_traits::{One, Zero, ToPrimitive};

    let n = poly.len();
    let num_primes = moduli.len();

    let q_prod_big: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();
    let q_half_big = &q_prod_big / 2;
    let base_big = BigInt::one() << base_w;
    let half_base_big = &base_big / 2;

    let q_bits = q_prod_big.bits() as u32;
    let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

    let mut digits = vec![vec![RnsRepresentation::new(vec![0; num_primes], moduli.to_vec()); n]; num_digits];

    for i in 0..n {
        let x_big = crt_reconstruct(&poly[i].values, moduli);

        let x_centered = if x_big > q_half_big {
            x_big - &q_prod_big
        } else {
            x_big
        };

        let mut remainder = x_centered;

        for t in 0..num_digits {
            let dt_unbalanced = &remainder % &base_big;
            let dt = if dt_unbalanced > half_base_big {
                &dt_unbalanced - &base_big
            } else {
                dt_unbalanced
            };

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
fn crt_reconstruct(residues: &[u64], moduli: &[u64]) -> num_bigint::BigInt {
    use num_bigint::BigInt;
    use num_traits::Zero;

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
fn mod_inverse(a: &num_bigint::BigInt, m: &num_bigint::BigInt) -> num_bigint::BigInt {
    use num_bigint::BigInt;
    use num_traits::{Zero, One};

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
fn relinearize_cpu(
    d0: &[RnsRepresentation],
    d1: &[RnsRepresentation],
    d2: &[RnsRepresentation],
    evk: &EvaluationKey,
    key_ctx: &KeyContext,
    moduli: &[u64],
) -> (Vec<RnsRepresentation>, Vec<RnsRepresentation>) {
    let n = d0.len();
    let base_w = evk.base_w;

    let mut c0 = d0.to_vec();
    let mut c1 = d1.to_vec();

    let d2_digits = gadget_decompose_cpu(d2, base_w, moduli);

    for (t, d2_digit) in d2_digits.iter().enumerate() {
        if t >= evk.evk0.len() {
            break;
        }

        let term0 = multiply_polys_cpu(d2_digit, &evk.evk0[t], key_ctx, moduli);
        let term1 = multiply_polys_cpu(d2_digit, &evk.evk1[t], key_ctx, moduli);

        for i in 0..n {
            c0[i] = c0[i].sub(&term0[i]);
            c1[i] = c1[i].add(&term1[i]);
        }
    }

    (c0, c1)
}

#[cfg(feature = "v2-gpu-metal")]
fn relinearize_metal(
    d0: &[u64],
    d1: &[u64],
    d2: &[u64],
    evk0: &[Vec<u64>],
    evk1: &[Vec<u64>],
    base_w: u32,
    ctx: &MetalCkksContext,
    moduli: &[u64],
    n: usize,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let num_primes = moduli.len();

    let mut c0 = d0.to_vec();
    let mut c1 = d1.to_vec();

    let d2_digits = MetalCiphertext::gadget_decompose_flat(d2, base_w, moduli, n)?;

    for (t, d2_digit) in d2_digits.iter().enumerate() {
        if t >= evk0.len() {
            break;
        }

        let term0 = ctx.multiply_polys_flat_ntt_negacyclic(d2_digit, &evk0[t], moduli)?;
        let term1 = ctx.multiply_polys_flat_ntt_negacyclic(d2_digit, &evk1[t], moduli)?;

        for i in 0..(n * num_primes) {
            let q = moduli[i % num_primes];
            c0[i] = if c0[i] >= term0[i] { c0[i] - term0[i] } else { q - (term0[i] - c0[i]) };
            c1[i] = ((c1[i] as u128 + term1[i] as u128) % q as u128) as u64;
        }
    }

    Ok((c0, c1))
}

#[cfg(not(feature = "v2-gpu-metal"))]
fn main() {
    println!("This example requires the 'v2-gpu-metal' feature.");
}
