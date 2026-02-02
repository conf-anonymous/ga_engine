fn main() {
    // Tell the linker to link against the appropriate BLAS backend
    // Only use Accelerate framework on macOS
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
