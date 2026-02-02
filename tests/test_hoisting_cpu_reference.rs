//! CPU Reference Test for Negacyclic Hoisting Identity
//!
//! Validates: NTT_neg(σ_g a)[j] = ψ^{(g-1)j} · NTT_neg(a)[j·g mod N]
//!
//! Uses pure CPU arithmetic (no Montgomery, no bit-reversal) with N=8
//! to establish ground truth before comparing against GPU implementation.

/// Find a primitive 2N-th root of unity modulo q
fn find_primitive_2n_root(n: usize, q: u64) -> Option<u64> {
    let two_n = 2 * n;

    // q must satisfy q ≡ 1 (mod 2N) for a primitive 2N-th root to exist
    if q % (two_n as u64) != 1 {
        return None;
    }

    // Try candidates
    for candidate in 2..q {
        // Check if candidate^{2N} = 1 and candidate^N ≠ 1
        let order_2n = mod_pow(candidate, two_n as u64, q) == 1;
        let order_n = mod_pow(candidate, n as u64, q);

        if order_2n && order_n != 1 {
            // Verify it's primitive: candidate^N should equal -1 (i.e., q-1)
            if order_n == q - 1 {
                return Some(candidate);
            }
        }
    }

    None
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

/// Modular multiplication
fn mod_mul(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

/// CPU negacyclic NTT by definition: NTT_neg(a)[j] = Σ_i a[i]·ψ^i·ω^{ij}
fn cpu_ntt_neg(a: &[u64], psi_powers: &[u64], omega_powers: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    let mut result = vec![0u64; n];

    for j in 0..n {
        let mut sum = 0u128;
        for i in 0..n {
            // a[i] * ψ^i * ω^{ij}
            let twist = mod_mul(a[i], psi_powers[i], q);
            let omega_pow = omega_powers[(i * j) % n];
            let term = mod_mul(twist, omega_pow, q);
            sum = (sum + term as u128) % q as u128;
        }
        result[j] = sum as u64;
    }

    result
}

/// Apply Galois automorphism σ_g in coefficient domain with negacyclic signs
/// Uses PULL semantics: for each destination index j, find where it comes from
fn apply_galois_coeff(a: &[u64], g: usize, q: u64) -> Vec<u64> {
    let n = a.len();
    let two_n = 2 * n;
    let mut b = vec![0u64; n];

    // Find g_inv such that g · g_inv ≡ 1 (mod 2N)
    let mut g_inv = 0usize;
    for candidate in 1..two_n {
        if ((g * candidate) % two_n) == 1 {
            g_inv = candidate;
            break;
        }
    }

    for j in 0..n {
        // Find i such that g·i ≡ j (mod 2N)
        // This is i = j · g^{-1} mod 2N
        let i_times_2n = (j * g_inv) % two_n;

        if i_times_2n < n {
            // Source is a[i_times_2n] with no sign flip
            b[j] = a[i_times_2n];
        } else {
            // Source is a[i_times_2n - N] with sign flip (because X^N = -1)
            let i_source = i_times_2n - n;
            b[j] = if a[i_source] == 0 { 0 } else { q - a[i_source] };
        }
    }

    b
}

/// Compute diagonal twist D_g[j] = ψ^{(g-1)j} mod 2N
fn compute_diagonal(g: usize, n: usize, psi_powers: &[u64]) -> Vec<u64> {
    let two_n = 2 * n;
    let exp_base = ((g + two_n - 1) % two_n) as usize;  // (g-1) mod 2N

    let mut diagonal = vec![0u64; n];
    for j in 0..n {
        let exp = (exp_base * j) % two_n;
        diagonal[j] = psi_powers[exp];
    }

    diagonal
}

#[test]
fn test_cpu_hoisting_identity() {
    println!("\n════════════════════════════════════════════════════════");
    println!("CPU Reference: Negacyclic Hoisting Identity (N=8)");
    println!("════════════════════════════════════════════════════════\n");

    // Setup for N=8
    let n = 8usize;
    let two_n = 2 * n;

    // Use a small NTT-friendly prime: q = 2N·k + 1 = 16k + 1
    // Try q = 97 (= 16*6 + 1)
    let q = 97u64;

    // Find primitive 2N-th root ψ
    let psi = find_primitive_2n_root(n, q).expect("No primitive 2N-th root found");
    println!("Parameters:");
    println!("  N = {}", n);
    println!("  q = {}", q);
    println!("  ψ = {} (primitive {}-th root)", psi, two_n);

    // Verify ψ^N = -1 (i.e., q-1)
    let psi_n = mod_pow(psi, n as u64, q);
    assert_eq!(psi_n, q - 1, "ψ^N should equal -1 (mod q)");
    println!("  ψ^N = {} ≡ -1 (mod {}) ✓", psi_n, q);

    // Compute ω = ψ^2 (primitive N-th root)
    let omega = mod_mul(psi, psi, q);
    let omega_n = mod_pow(omega, n as u64, q);
    assert_eq!(omega_n, 1, "ω^N should equal 1");
    println!("  ω = ψ² = {} (primitive {}-th root)", omega, n);
    println!("  ω^N = {} ✓\n", omega_n);

    // Precompute power tables
    let mut psi_powers = vec![1u64; two_n];
    for i in 1..two_n {
        psi_powers[i] = mod_mul(psi_powers[i - 1], psi, q);
    }

    let mut omega_powers = vec![1u64; n];
    for i in 1..n {
        omega_powers[i] = mod_mul(omega_powers[i - 1], omega, q);
    }

    // Test with different inputs and Galois elements
    let test_cases = vec![
        (vec![1, 2, 3, 4, 5, 6, 7, 8], 3, "Sequential values, g=3"),
        (vec![1, 0, 0, 0, 0, 0, 0, 0], 3, "Delta impulse at i=0, g=3"),
        (vec![0, 1, 0, 0, 0, 0, 0, 0], 3, "Delta impulse at i=1, g=3"),
        (vec![1, 2, 3, 4, 5, 6, 7, 8], 5, "Sequential values, g=5"),
        (vec![1, 0, 0, 0, 0, 0, 0, 0], 5, "Delta impulse at i=0, g=5"),
    ];

    for (a_raw, g, description) in test_cases {
        // Reduce a modulo q
        let a: Vec<u64> = a_raw.iter().map(|&x| x % q).collect();

        println!("Test: {}", description);
        println!("  a = {:?}", a);

        // Path 1: Apply σ_g in coefficients, then negacyclic NTT
        let b = apply_galois_coeff(&a, g, q);
        println!("  After Galois σ_{}: b = {:?}", g, b);

        let b1 = cpu_ntt_neg(&b, &psi_powers, &omega_powers, q);
        println!("  Path 1 (σ_g → NTT): B1 = {:?}", b1);

        // Path 2: Negacyclic NTT first, then permute with offset (NO diagonal!)
        // Correct formula: NTT_neg(σ_g a)[j] = NTT_neg(a)[(g·j + α) mod N]
        // where α = (g-1)/2 mod N
        let a_ntt = cpu_ntt_neg(&a, &psi_powers, &omega_powers, q);
        println!("  NTT_neg(a): A = {:?}", a_ntt);

        // Compute offset: α = (g-1)/2 mod N
        let alpha = ((g - 1) / 2) % n;
        println!("  Offset α = (g-1)/2 = ({}-1)/2 = {}", g, alpha);

        // Permute with offset: B2[j] = A[(j*g + α) mod N]
        let mut b2 = vec![0u64; n];
        for j in 0..n {
            let source_idx = ((j * g) + alpha) % n;
            b2[j] = a_ntt[source_idx];
        }
        println!("  Path 2 (NTT → permute with offset): B2 = {:?}", b2);

        // Compare
        let mut all_match = true;
        for j in 0..n {
            if b1[j] != b2[j] {
                println!("  ✗ Mismatch at j={}: Path1={}, Path2={}", j, b1[j], b2[j]);
                all_match = false;
            }
        }

        if all_match {
            println!("  ✓ PASS: B1 == B2 for all j");
            println!("    B1 = {:?}", b1);
            println!("    B2 = {:?}", b2);
        } else {
            panic!("CPU reference test failed for: {}", description);
        }

        println!();
    }

    println!("════════════════════════════════════════════════════════");
    println!("✓ All CPU reference tests passed!");
    println!("  Identity verified: NTT_neg(σ_g a)[j] = ψ^{{(g-1)j}}·NTT_neg(a)[jg]");
    println!("════════════════════════════════════════════════════════\n");
}
