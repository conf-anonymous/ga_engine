use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext as CpuNttContext;

fn main() {
    // Use tiny n=8 for manual verification
    let n = 8;
    let q = 1073872897u64;

    // Find psi and omega
    let psi = find_psi(n, q).unwrap();
    let omega = ((psi as u128 * psi as u128) % q as u128) as u64;
    let omega_inv = mod_inverse(omega, q).unwrap();
    let n_inv = mod_inverse(n as u64, q).unwrap();

    println!("n={}, q={}", n, q);
    println!("omega={}, omega_inv={}, n_inv={}", omega, omega_inv, n_inv);

    // Verify n_inv
    let check = ((n as u128 * n_inv as u128) % q as u128) as u64;
    println!("n * n_inv mod q = {} (should be 1)", check);

    // Create CPU NTT context
    let cpu_ntt = CpuNttContext::new(n, q);

    // Test polynomial
    let mut test = vec![0u64; n];
    test[0] = 1;
    test[1] = 2;

    println!("\nOriginal: {:?}", test);

    // CPU forward
    let mut cpu = test.clone();
    cpu_ntt.forward_ntt(&mut cpu);
    println!("CPU forward: {:?}", cpu);

    // Manual inverse on CPU data (step by step)
    // Step 1: Bit reverse
    let mut manual = cpu.clone();
    for i in 0..n {
        let j = bit_reverse(i, 3); // log_n = 3 for n=8
        if j > i {
            manual.swap(i, j);
        }
    }
    println!("After bit-reverse: {:?}", manual);

    // Step 2: Butterfly stages with omega_inv
    let mut m = 1;
    for stage in 0..3 {
        let m2 = m * 2;
        let w_m = pow_mod(omega_inv, (n / m2) as u64, q);
        println!("\nStage {}: m={}, m2={}, w_m={}", stage, m, m2, w_m);

        let mut k = 0;
        while k < n {
            let mut w = 1u64;
            for j in 0..m {
                let idx1 = k + j;
                let idx2 = idx1 + m;
                let u = manual[idx1];
                let t = mul_mod(w, manual[idx2], q);
                manual[idx1] = add_mod(u, t, q);
                manual[idx2] = sub_mod(u, t, q);

                if k == 0 && j < 2 {
                    println!("  j={}: idx1={}, idx2={}, u={}, t={}, w={}, result[{}]={}, result[{}]={}",
                        j, idx1, idx2, u, t, w, idx1, manual[idx1], idx2, manual[idx2]);
                }

                w = mul_mod(w, w_m, q);
            }
            k += m2;
        }
        println!("After stage {}: {:?}", stage, manual);
        m = m2;
    }

    // Step 3: Scale by n_inv
    for val in manual.iter_mut() {
        *val = mul_mod(*val, n_inv, q);
    }
    println!("\nAfter n_inv scaling: {:?}", manual);
    println!("Should be: {:?}", test);

    // Compare with CPU inverse
    let mut cpu2 = cpu.clone();
    cpu_ntt.inverse_ntt(&mut cpu2);
    println!("CPU inverse: {:?}", cpu2);
}

fn find_psi(n: usize, q: u64) -> Option<u64> {
    let two_n = 2 * n as u64;
    for g in 2..100u64 {
        let psi = pow_mod(g, (q - 1) / two_n, q);
        if pow_mod(psi, n as u64, q) == q - 1 {
            return Some(psi);
        }
    }
    None
}

fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % m as u128) as u64;
        }
        base = ((base as u128 * base as u128) % m as u128) as u64;
        exp >>= 1;
    }
    result
}

fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    let sum = a + b;
    if sum >= q { sum - q } else { sum }
}

fn sub_mod(a: u64, b: u64, q: u64) -> u64 {
    if a >= b { a - b } else { a + q - b }
}

fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (m as i128, a as i128);

    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }

    if r > 1 {
        return None;
    }
    if t < 0 {
        t += m as i128;
    }

    Some(t as u64)
}

fn bit_reverse(mut x: usize, log_n: usize) -> usize {
    let mut result = 0;
    for _ in 0..log_n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}
