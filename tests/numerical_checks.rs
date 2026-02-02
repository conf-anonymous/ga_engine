// DISABLED: numerical_checks module no longer exists
#![cfg(feature = "numerical_checks_module_exists")]

use ga_engine::numerical_checks::dft::*;
use ga_engine::numerical_checks::multivector2::Multivector2;
use rand::Rng;
use rustfft::num_complex::Complex;

/// Helper: Convert real vector to classical complex vector
fn to_complex_vec(input: &[f64]) -> Vec<Complex<f64>> {
    input.iter().map(|&x| Complex::new(x, 0.0)).collect()
}

/// Helper: Convert real vector to GA multivector2 vector
fn to_ga_vec(input: &[f64]) -> Vec<Multivector2> {
    input
        .iter()
        .map(|&x| Multivector2::new(x, 0.0, 0.0, 0.0))
        .collect()
}

/// Test a single input
fn run_single_dft_test(input_real: &[f64]) {
    let input_complex = to_complex_vec(input_real);
    let input_ga = to_ga_vec(input_real);

    let output_classical = classical_dft(&input_complex);
    let output_ga = ga_dft(&input_ga);

    assert!(
        compare_complex_outputs(&output_classical, &output_ga),
        "Mismatch on input: {:?}",
        input_real
    );
}

/// Deterministic simple test
#[test]
fn test_dft_equivalence_simple() {
    let input_real = vec![1.0, 0.0, -1.0, 0.0];
    run_single_dft_test(&input_real);
}

/// Randomized tests
#[test]
fn test_dft_equivalence_randomized() {
    let mut rng = rand::thread_rng();
    let sizes = [4, 8, 16, 32, 64];
    for &size in &sizes {
        for _ in 0..10 {
            let input_real: Vec<f64> = (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect();
            run_single_dft_test(&input_real);
        }
    }
}

/// Edge case tests
#[test]
fn test_dft_equivalence_edge_cases() {
    let edge_cases = vec![
        vec![0.0; 4],
        vec![1.0; 4],
        vec![-1.0; 4],
        vec![1.0, -1.0, 1.0, -1.0],
        vec![1e-10, -1e-10, 1e10, -1e10],
    ];

    for case in edge_cases {
        run_single_dft_test(&case);
    }
}
