//! Debug V3 prime ordering and scale tracking
//!
//! Run with:
//! ```bash
//! cargo run --release --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3 --example debug_v3_primes
//! ```

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

#[cfg(all(feature = "v3", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("\n=== DEBUG: V3 Prime Ordering and Scale Analysis ===\n");

    let params = CliffordFHEParams::new_v3_bootstrap_cuda_full()?;

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Num primes = {}", params.moduli.len());
    println!("  Scale = {:.2e} (2^{:.1})", params.scale, params.scale.log2());
    println!();

    println!("Moduli (ordered by index):");
    println!("┌───────┬─────────────────────────┬──────────────┬──────────────────────┐");
    println!("│ Index │ Prime Value             │ Bit Size     │ Type                 │");
    println!("├───────┼─────────────────────────┼──────────────┼──────────────────────┤");
    for (i, &q) in params.moduli.iter().enumerate() {
        let bits = (q as f64).log2();
        let prime_type = if bits > 55.0 { "Special (60-bit)" } else { "Scaling (45-bit)" };
        println!("│ {:>5} │ {:>23} │ {:>10.1} │ {:>20} │", i, q, bits, prime_type);
    }
    println!("└───────┴─────────────────────────┴──────────────┴──────────────────────┘");
    println!();

    println!("Rescaling behavior:");
    println!("When at level L and rescaling, we divide by moduli[L].");
    println!();

    let max_level = params.moduli.len() - 1;
    println!("┌───────────────┬──────────────────────┬──────────────┐");
    println!("│ Before Level  │ Prime Dropped        │ Bit Size     │");
    println!("├───────────────┼──────────────────────┼──────────────┤");
    for level in (1..=5).rev() {
        let dropped_idx = level;
        let q = params.moduli[dropped_idx];
        let bits = (q as f64).log2();
        println!("│ {:>13} │ moduli[{:>2}] = {:>9} │ {:>10.1} │", level, dropped_idx, q % 1_000_000_000, bits);
    }
    println!("└───────────────┴──────────────────────┴──────────────┘");
    println!();

    println!("Scale evolution through operations:");
    println!();

    let mut scale = params.scale;
    println!("Initial scale: {:.2e} (2^{:.1})", scale, scale.log2());
    println!();

    // Simulate multiplication + rescale cycle
    for level in (1..=5).rev() {
        let q_dropped = params.moduli[level];
        let scale_before_mult = scale;
        let scale_after_mult = scale * scale;
        let scale_after_rescale = scale_after_mult / q_dropped as f64;

        println!("At level {}:", level);
        println!("  Before mult: scale = {:.2e} (2^{:.1})", scale_before_mult, scale_before_mult.log2());
        println!("  After mult:  scale = {:.2e} (2^{:.1})", scale_after_mult, scale_after_mult.log2());
        println!("  Dropped q[{}] = {} (2^{:.1})", level, q_dropped, (q_dropped as f64).log2());
        println!("  After rescale: scale = {:.2e} (2^{:.1})", scale_after_rescale, scale_after_rescale.log2());
        println!();

        scale = scale_after_rescale;
    }

    println!("=== ANALYSIS ===");
    println!();
    println!("With V3 params (1× 60-bit + 29× 45-bit primes):");
    println!("- Initial scale: 2^45");
    println!("- After mult: 2^90");
    println!("- After rescale by ~45-bit: 2^90 / 2^45 = 2^45 (matches initial) ✓");
    println!();
    println!("The 60-bit prime is at index 0 and is NEVER dropped during normal operations.");
    println!("It serves as the 'special' prime that provides extra precision.");
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v3", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires: --no-default-features --features f64,nd,v2,v2-gpu-cuda,v3");
}
