//! Benchmark: S-box DDT Computation (Baseline vs GA)
//!
//! Tests the speedup of GA-based DDT computation on real-world S-boxes.

use ga_engine::cryptanalysis::sbox_ga::SBoxGA;
use std::time::Instant;

/// AES S-box (SubBytes transformation)
const AES_SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║        S-box DDT Computation: Baseline vs GA              ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Test on AES S-box (8-bit)
    println!("Testing AES S-box (8-bit, 256 elements)");
    println!("════════════════════════════════════════════════════════════");
    println!();

    let aes_sbox = SBoxGA::from_lut(AES_SBOX.to_vec(), 8);

    // Warmup
    println!("Warming up...");
    let _ = aes_sbox.compute_ddt_baseline();
    let _ = aes_sbox.compute_ddt_ga();
    println!();

    // Benchmark baseline
    println!("Benchmarking baseline DDT computation (100 trials)...");
    let trials = 100;
    let mut baseline_times = Vec::new();

    for _ in 0..trials {
        let start = Instant::now();
        let _ = aes_sbox.compute_ddt_baseline();
        baseline_times.push(start.elapsed());
    }

    let baseline_mean = baseline_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / trials as f64;
    let baseline_min = baseline_times.iter().map(|t| t.as_secs_f64()).fold(f64::INFINITY, f64::min);
    let baseline_max = baseline_times.iter().map(|t| t.as_secs_f64()).fold(f64::NEG_INFINITY, f64::max);

    println!("  Mean: {:.3} ms", baseline_mean * 1000.0);
    println!("  Min:  {:.3} ms", baseline_min * 1000.0);
    println!("  Max:  {:.3} ms", baseline_max * 1000.0);
    println!();

    // Benchmark GA
    println!("Benchmarking GA DDT computation (100 trials)...");
    let mut ga_times = Vec::new();

    for _ in 0..trials {
        let start = Instant::now();
        let _ = aes_sbox.compute_ddt_ga();
        ga_times.push(start.elapsed());
    }

    let ga_mean = ga_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / trials as f64;
    let ga_min = ga_times.iter().map(|t| t.as_secs_f64()).fold(f64::INFINITY, f64::min);
    let ga_max = ga_times.iter().map(|t| t.as_secs_f64()).fold(f64::NEG_INFINITY, f64::max);

    println!("  Mean: {:.3} ms", ga_mean * 1000.0);
    println!("  Min:  {:.3} ms", ga_min * 1000.0);
    println!("  Max:  {:.3} ms", ga_max * 1000.0);
    println!();

    // Compute speedup
    let speedup = baseline_mean / ga_mean;

    println!("════════════════════════════════════════════════════════════");
    println!("RESULTS");
    println!("════════════════════════════════════════════════════════════");
    println!();
    println!("  Baseline: {:.3} ms", baseline_mean * 1000.0);
    println!("  GA:       {:.3} ms", ga_mean * 1000.0);
    println!();
    println!("  SPEEDUP: {:.2}×", speedup);
    println!();

    if speedup >= 2.0 {
        println!("✓ SUCCESS: Achieved ≥2× speedup target!");
    } else if speedup >= 1.5 {
        println!("⚠ PARTIAL: {:.2}× speedup (target was 2×)", speedup);
    } else if speedup >= 1.1 {
        println!("⚠ MARGINAL: Only {:.2}× speedup", speedup);
    } else {
        println!("✗ NO SPEEDUP: {:.2}×", speedup);
    }
    println!();

    // Also test smaller S-boxes
    println!("════════════════════════════════════════════════════════════");
    println!("Testing 4-bit S-box (16 elements)");
    println!("════════════════════════════════════════════════════════════");
    println!();

    let toy_sbox = SBoxGA::from_lut(
        vec![0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
             0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7],
        4,
    );

    let trials_4bit = 10000;

    // Baseline
    let mut baseline_4bit_times = Vec::new();
    for _ in 0..trials_4bit {
        let start = Instant::now();
        let _ = toy_sbox.compute_ddt_baseline();
        baseline_4bit_times.push(start.elapsed());
    }
    let baseline_4bit_mean = baseline_4bit_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / trials_4bit as f64;

    // GA
    let mut ga_4bit_times = Vec::new();
    for _ in 0..trials_4bit {
        let start = Instant::now();
        let _ = toy_sbox.compute_ddt_ga();
        ga_4bit_times.push(start.elapsed());
    }
    let ga_4bit_mean = ga_4bit_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / trials_4bit as f64;

    let speedup_4bit = baseline_4bit_mean / ga_4bit_mean;

    println!("  Baseline: {:.3} µs", baseline_4bit_mean * 1_000_000.0);
    println!("  GA:       {:.3} µs", ga_4bit_mean * 1_000_000.0);
    println!("  SPEEDUP: {:.2}×", speedup_4bit);
    println!();

    // Final summary
    println!("════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("════════════════════════════════════════════════════════════");
    println!();
    println!("  4-bit S-box:  {:.2}× speedup", speedup_4bit);
    println!("  8-bit S-box:  {:.2}× speedup", speedup);
    println!();
    println!("  Average:      {:.2}×", (speedup_4bit + speedup) / 2.0);
    println!();
}
