//! E8 Orbit Enumeration - Simple GA Approach
//!
//! Test: Can our current RotorND implementation beat matrix reflections?
//!
//! We'll use the simplest possible GA approach:
//! - Represent each reflection as a RotorND
//! - Use sandwich product to apply reflections
//! - Measure if this is faster than matrix approach

use ga_engine::lattice_reduction::e8_lattice::{E8Lattice, norm_squared};
use ga_engine::lattice_reduction::ga_reflection::reflect_e8_optimized;
use std::collections::HashSet;
use std::time::Instant;

/// Hash wrapper for [f64; 8] vectors
#[derive(Clone, Copy, Debug)]
struct FloatVec([f64; 8]);

impl PartialEq for FloatVec {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
    }
}

impl Eq for FloatVec {}

impl std::hash::Hash for FloatVec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for &x in &self.0 {
            let rounded = (x * 1e10).round() as i64;
            rounded.hash(state);
        }
    }
}

/// Enumerate orbit using optimized E8 reflection
fn enumerate_orbit_ga(v: &[f64; 8], simple_roots: &[[f64; 8]], max_orbit_size: usize) -> HashSet<FloatVec> {
    let mut orbit = HashSet::new();
    orbit.insert(FloatVec(*v));

    let mut prev_size = 0;
    let mut iteration = 0;

    while orbit.len() != prev_size && orbit.len() < max_orbit_size {
        prev_size = orbit.len();
        iteration += 1;

        let current_vectors: Vec<[f64; 8]> = orbit.iter().map(|fv| fv.0).collect();

        for vec in current_vectors {
            for root in simple_roots {
                let reflected = reflect_e8_optimized(&vec, root);
                orbit.insert(FloatVec(reflected));

                if orbit.len() >= max_orbit_size {
                    break;
                }
            }
        }

        if iteration % 5 == 0 {
            println!("  Iteration {}: {} vectors in orbit", iteration, orbit.len());
        }
    }

    orbit
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  E8 Orbit Enumeration - GA Optimized Reflection         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let e8 = E8Lattice::new();
    println!("{}", e8);
    println!();

    // Test the key case: e₁+e₂ (generates full 240-root orbit)
    let test_vec = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    println!("──────────────────────────────────────────────────────────");
    println!("Test vector: e₁+e₂");
    println!("  v = {:?}", &test_vec[..4]);
    println!("  ||v||² = {:.6}", norm_squared(&test_vec));
    println!();

    let max_orbit = 1000;

    println!("Enumerating orbit (GA optimized reflections)...");
    let start = Instant::now();

    let orbit = enumerate_orbit_ga(&test_vec, e8.simple_roots(), max_orbit);

    let elapsed = start.elapsed();

    println!();
    println!("Results:");
    println!("  Orbit size: {}", orbit.len());
    println!("  Time: {:?}", elapsed);
    println!("  Time per vector: {:?}", elapsed / orbit.len() as u32);
    println!();

    // Verify
    let original_norm_sq = norm_squared(&test_vec);
    let mut all_same_norm = true;
    for vec in orbit.iter().take(10) {
        let norm_sq = norm_squared(&vec.0);
        if (norm_sq - original_norm_sq).abs() > 1e-8 {
            println!("  ⚠️  Norm not preserved: {} vs {}", norm_sq, original_norm_sq);
            all_same_norm = false;
        }
    }
    if all_same_norm {
        println!("  ✓ All orbit vectors have same norm");
    }

    println!("──────────────────────────────────────────────────────────");
}
