/// Demonstration of V4 Packed Multiplication Table
///
/// Shows how the Clifford algebra multiplication table is structured
/// for the packed slot-interleaved layout.

#[cfg(feature = "v4")]
fn main() {
    use ga_engine::clifford_fhe_v4::mult_table::PackedMultTable;

    println!("═══════════════════════════════════════════════════════════");
    println!("    V4 Packed Multiplication Table Analysis");
    println!("═══════════════════════════════════════════════════════════\n");

    let table = PackedMultTable::new();

    println!("Clifford Algebra Cl(3,0) Basis:");
    println!("  Components: [s, e1, e2, e3, e12, e23, e31, I]\n");

    println!("Multiplication Table Structure:");
    println!("  Total terms: {} (8 components × 8 terms each)", table.total_terms());
    println!("  Memory: Much smaller than 8 separate ciphertexts!\n");

    // Show example: How to compute scalar component (component 0)
    println!("Example: Computing Scalar Component (component 0)");
    println!("  Result = (a⊗b)_scalar contains:");
    let scalar_terms = table.get_terms(0);
    for term in scalar_terms.iter() {
        let sign = if term.coeff > 0 { "+" } else { "-" };
        let comp_names = ["s", "e1", "e2", "e3", "e12", "e23", "e31", "I"];
        println!("    {} {}⊗{} ",
            sign,
            comp_names[term.a_comp],
            comp_names[term.b_comp]
        );
    }

    println!("\n  Notice: Vectors square to +1, bivectors to -1");

    // Show example: e1 component
    println!("\nExample: Computing e1 Component (component 1)");
    let e1_terms = table.get_terms(1);
    for term in e1_terms.iter() {
        let sign = if term.coeff > 0 { "+" } else { "-" };
        let comp_names = ["s", "e1", "e2", "e3", "e12", "e23", "e31", "I"];
        println!("    {} {}⊗{} (rotate a by {}, b by {})",
            sign,
            comp_names[term.a_comp],
            comp_names[term.b_comp],
            term.a_rotation,
            term.b_rotation
        );
    }

    println!("\nPacked Multiplication Algorithm:");
    println!("  For each output component i:");
    println!("    1. Initialize result to zero");
    println!("    2. For each term in table[i]:");
    println!("       a. Extract component a_comp from multivector A");
    println!("          (rotate A by -a_comp, apply mask)");
    println!("       b. Extract component b_comp from multivector B");
    println!("          (rotate B by -b_comp, apply mask)");
    println!("       c. Multiply extracted components");
    println!("       d. Scale by coefficient (+1 or -1)");
    println!("       e. Add to result");
    println!("    3. Pack result into output component i\n");

    println!("Performance Characteristics:");
    println!("  V2/V3: 64 ciphertext multiplications (8×8 separate cts)");
    println!("  V4:    64 diagonal multiplies + rotations (1 packed ct)");
    println!("  Trade-off: Similar operations, but 8× memory savings\n");

    println!("Next Steps:");
    println!("  1. ✓ Multiplication table defined");
    println!("  2. ⏭ Implement diagonal multiply helper");
    println!("  3. ⏭ Implement geometric_product_packed()");
    println!("  4. ⏭ Implement wedge/inner products\n");
}

#[cfg(not(feature = "v4"))]
fn main() {
    println!("This example requires feature: v4");
    println!("Run with: cargo run --features v4 --example test_v4_mult_table");
}
