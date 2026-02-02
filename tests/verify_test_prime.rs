//! Verify if 1141173990025715713 is prime

fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 || n == 3 { return true; }
    if n % 2 == 0 { return false; }

    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    let witnesses = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    'witness: for &a in &witnesses {
        if a >= n { continue; }

        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue 'witness;
        }

        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }

        return false;
    }

    true
}

fn mod_pow(base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u128;
    let mut b = (base % m) as u128;
    let m128 = m as u128;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % m128;
        }
        b = (b * b) % m128;
        exp >>= 1;
    }

    (result % m128) as u64
}

fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

#[test]
fn test_1141173990025715713() {
    let q = 1141173990025715713u64;
    let n = 1024u64;

    println!("Testing q = {}", q);
    println!("(q-1) mod 2N = {}", (q - 1) % (2 * n));
    println!("Is prime: {}", is_prime(q));

    assert_eq!((q - 1) % (2 * n), 0, "Not NTT-friendly");
    assert!(is_prime(q), "Not prime!");
}
