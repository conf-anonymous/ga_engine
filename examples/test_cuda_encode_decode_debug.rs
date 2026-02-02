//! Debug version with detailed output

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;

    println!("=== Debug: CUDA vs CPU Encode/Decode ===\n");

    let params = CliffordFHEParams::new_128bit();

    // CUDA context
    let cuda_ctx = CudaCkksContext::new(params.clone())?;

    // CPU context for comparison
    let cpu_ctx = CkksContext::new(params.clone());

    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();

    let test_val = 2.0;
    println!("Test value: {}\n", test_val);

    let scale = params.scale;
    let level = params.moduli.len() - 1;

    // CUDA path
    println!("=== CUDA PATH ===");
    let pt_cuda = cuda_ctx.encode(&[test_val], scale, level)?;
    println!("Encoded plaintext[0] (first prime): {}", pt_cuda.poly[0]);

    let ct_cuda = cuda_ctx.encrypt(&pt_cuda, &pk)?;
    println!("Encrypted c0[0]: {}, c1[0]: {}", ct_cuda.c0[0], ct_cuda.c1[0]);

    let pt_dec_cuda = cuda_ctx.decrypt(&ct_cuda, &sk)?;
    println!("Decrypted plaintext[0] (first prime): {}", pt_dec_cuda.poly[0]);

    let result_cuda = cuda_ctx.decode(&pt_dec_cuda)?;
    println!("Decoded: {}", result_cuda[0]);
    println!("Error: {:.2e}\n", (result_cuda[0] - test_val).abs());

    // CPU path for comparison
    println!("=== CPU PATH (for comparison) ===");
    let pt_cpu = cpu_ctx.encode(&[test_val]);
    println!("Encoded plaintext[0] (first prime): {}", pt_cpu.coeffs[0].values[0]);

    let ct_cpu = cpu_ctx.encrypt(&pt_cpu, &pk);
    println!("Encrypted c0[0]: {}, c1[0]: {}", ct_cpu.c0[0].values[0], ct_cpu.c1[0].values[0]);

    let pt_dec_cpu = cpu_ctx.decrypt(&ct_cpu, &sk);
    println!("Decrypted plaintext[0] (first prime): {}", pt_dec_cpu.coeffs[0].values[0]);

    let result_cpu = cpu_ctx.decode(&pt_dec_cpu);
    println!("Decoded: {}", result_cpu[0]);
    println!("Error: {:.2e}", (result_cpu[0] - test_val).abs());

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires features: v2,v2-gpu-cuda");
}
