use ga_engine::lattice_reduction::rotor_nd::RotorND;

fn main() {
    // Test 1: Simple 3D rotation
    println!("=== Test 1: 3D Rotation x → y ===");
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let r = RotorND::from_vectors(&a, &b);

    println!("Rotor: {}", r);
    println!("Unit check: {:.10}", r.verify_unit());

    let result = r.apply(&a);
    println!("Input:  {:?}", a);
    println!("Output: {:?}", result);

    let a_norm: f64 = a.iter().map(|x| x*x).sum::<f64>().sqrt();
    let r_norm: f64 = result.iter().map(|x| x*x).sum::<f64>().sqrt();
    println!("Input norm:  {:.10}", a_norm);
    println!("Output norm: {:.10}", r_norm);
    println!("Norm diff:   {:.10}", (a_norm - r_norm).abs());

    // Test 2: 4D rotation
    println!("\n=== Test 2: 4D Rotation ===");
    let a4 = vec![1.0, 0.0, 0.0, 0.0];
    let b4 = vec![0.0, 1.0, 0.0, 0.0];
    let r4 = RotorND::from_vectors(&a4, &b4);

    println!("4D Rotor unit check: {:.10}", r4.verify_unit());

    let v4 = vec![1.0, 2.0, 3.0, 4.0];
    let result4 = r4.apply(&v4);

    let v4_norm: f64 = v4.iter().map(|x| x*x).sum::<f64>().sqrt();
    let r4_norm: f64 = result4.iter().map(|x| x*x).sum::<f64>().sqrt();
    println!("Input:  {:?}", v4);
    println!("Output: {:?}", result4);
    println!("Input norm:  {:.10}", v4_norm);
    println!("Output norm: {:.10}", r4_norm);
    println!("Norm diff:   {:.10}", (v4_norm - r4_norm).abs());

    // Test 3: Identity should leave vector unchanged
    println!("\n=== Test 3: Identity Rotor ===");
    let r_id = RotorND::identity(3);
    let v = vec![1.0, 2.0, 3.0];
    let result_id = r_id.apply(&v);
    println!("Input:  {:?}", v);
    println!("Output: {:?}", result_id);
    let diff: f64 = v.iter().zip(result_id.iter()).map(|(a,b)| (a-b).abs()).sum();
    println!("Total diff: {:.10}", diff);
}
