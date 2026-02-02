// Debug NTT root computation for prime[3]

#[test]
fn test_ntt_roots_for_prime3() {
    let q = 1099511693313u64;  // prime[3]
    let n = 1024usize;

    println!("Debugging NTT roots for q = {}", q);

    // Check basic properties
    println!("\n=== Basic Properties ===");
    println!("q - 1 = {}", q - 1);
    println!("2N = {}", 2 * n);
    println!("(q - 1) / (2N) = {}", (q - 1) / (2 * n) as u64);
    println!("(q - 1) % (2N) = {}", (q - 1) % (2 * n) as u64);

    // Manually compute primitive root
    println!("\n=== Finding Primitive Root ===");
    let phi = q - 1;
    let mut odd = phi;
    while odd % 2 == 0 {
        odd /= 2;
    }
    println!("phi = q - 1 = {}", phi);
    println!("odd part of phi = {}", odd);

    // Try finding primitive root
    let g = find_primitive_root(q);
    println!("Primitive root g = {}", g);

    // Verify it's actually primitive
    println!("\n=== Verifying Primitive Root ===");
    let g_to_phi_over_2 = mod_pow(g, phi / 2, q);
    println!("g^(phi/2) = {} (should be q-1 = {} for primitive root)", g_to_phi_over_2, q - 1);

    if g_to_phi_over_2 != q - 1 {
        println!("❌ Not a primitive root!");
    }

    // Compute psi (2N-th root of unity)
    println!("\n=== Computing psi (2N-th root of unity) ===");
    let two_n = 2u64 * (n as u64);
    let exp = (q - 1) / two_n;
    println!("psi = g^((q-1)/(2N)) = {}^{}", g, exp);
    let psi = mod_pow(g, exp, q);
    println!("psi = {}", psi);

    // Verify psi^(2N) = 1
    let psi_to_2n = mod_pow(psi, two_n, q);
    println!("psi^(2N) = {} (should be 1)", psi_to_2n);

    if psi_to_2n != 1 {
        println!("❌ psi is NOT a 2N-th root of unity!");
    } else {
        println!("✅ psi^(2N) = 1");
    }

    // Verify psi^N = -1
    let n_u64 = n as u64;
    let psi_to_n = mod_pow(psi, n_u64, q);
    println!("psi^N = {} (should be q-1 = {})", psi_to_n, q - 1);

    if psi_to_n != q - 1 {
        println!("❌ psi^N ≠ -1, so psi is not correct!");
    } else {
        println!("✅ psi^N = -1");
    }

    // Compute omega = psi^2
    println!("\n=== Computing omega = psi^2 ===");
    let omega = mod_mul(psi, psi, q);
    println!("omega = {}", omega);

    // Verify omega^N = 1
    let omega_to_n = mod_pow(omega, n_u64, q);
    println!("omega^N = {} (should be 1)", omega_to_n);

    if omega_to_n != 1 {
        println!("❌ omega is NOT an N-th root of unity!");
    } else {
        println!("✅ omega^N = 1");
    }

    // Check if omega is primitive N-th root
    let omega_to_n_over_2 = mod_pow(omega, n_u64 / 2, q);
    println!("omega^(N/2) = {} (should be q-1 = {} for primitive)", omega_to_n_over_2, q - 1);

    if omega_to_n_over_2 != q - 1 {
        println!("❌ omega is NOT a primitive N-th root!");
    } else {
        println!("✅ omega is primitive N-th root");
    }
}

fn mod_mul(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

fn mod_pow(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut result = 1u64;
    base = base % q;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, q);
        }
        base = mod_mul(base, base, q);
        exp >>= 1;
    }
    result
}

fn find_primitive_root(q: u64) -> u64 {
    let phi = q - 1;
    let mut odd = phi;
    while odd % 2 == 0 {
        odd /= 2;
    }

    for g in 2..q {
        if mod_pow(g, phi / 2, q) == 1 {
            continue;
        }
        if odd != 1 && mod_pow(g, phi / odd, q) == 1 {
            continue;
        }
        return g;
    }
    panic!("No primitive root found");
}
