/// CUDA Geometric Product Implementation
///
/// Homomorphic geometric product for Cl(3,0) using CUDA GPU acceleration.

use super::ntt::CudaNttContext;
use super::device::CudaDeviceContext;
use std::sync::Arc;

/// Structure constants for Cl(3,0) geometric product
/// Encodes the multiplication table: e_i * e_j = sign * e_k
pub struct Cl3StructureConstants {
    // For each output component [0..8], stores 8 terms: (input_a_idx, input_b_idx, sign)
    pub terms: [[(usize, usize, i8); 8]; 8],
}

impl Cl3StructureConstants {
    pub fn new() -> Self {
        // Clifford algebra Cl(3,0) multiplication table
        // Basis: {1, e1, e2, e3, e12, e13, e23, e123}
        // Indices: [0,  1,  2,  3,   4,   5,   6,    7]

        let mut terms = [[(0, 0, 0i8); 8]; 8];

        // Component 0 (scalar): 1*1, e1*e1, e2*e2, e3*e3, -e12*e12, -e13*e13, -e23*e23, -e123*e123
        terms[0] = [(0,0,1), (1,1,1), (2,2,1), (3,3,1), (4,4,-1), (5,5,-1), (6,6,-1), (7,7,-1)];

        // Component 1 (e1): 1*e1, e1*1, e12*e2, e13*e3, -e2*e12, -e3*e13, e123*e23, -e23*e123
        terms[1] = [(0,1,1), (1,0,1), (4,2,1), (5,3,1), (2,4,-1), (3,5,-1), (7,6,1), (6,7,-1)];

        // Component 2 (e2): 1*e2, e2*1, e12*e1, e23*e3, -e1*e12, -e123*e13, -e3*e23, e13*e123
        terms[2] = [(0,2,1), (2,0,1), (4,1,-1), (6,3,1), (1,4,1), (7,5,-1), (3,6,-1), (5,7,1)];

        // Component 3 (e3): 1*e3, e3*1, e13*e1, e23*e2, -e1*e13, -e2*e23, e123*e12, -e12*e123
        terms[3] = [(0,3,1), (3,0,1), (5,1,-1), (6,2,-1), (1,5,1), (2,6,1), (7,4,1), (4,7,-1)];

        // Component 4 (e12): 1*e12, e12*1, e1*e2, e123*e3, -e2*e1, -e3*e123, e13*e23, -e23*e13
        terms[4] = [(0,4,1), (4,0,1), (1,2,1), (7,3,1), (2,1,-1), (3,7,-1), (5,6,1), (6,5,-1)];

        // Component 5 (e13): 1*e13, e13*1, e1*e3, e123*e2, -e3*e1, e2*e123, -e12*e23, e23*e12
        terms[5] = [(0,5,1), (5,0,1), (1,3,1), (7,2,-1), (3,1,-1), (2,7,1), (4,6,-1), (6,4,1)];

        // Component 6 (e23): 1*e23, e23*1, e2*e3, e123*e1, -e3*e2, -e1*e123, e12*e13, -e13*e12
        terms[6] = [(0,6,1), (6,0,1), (2,3,1), (7,1,1), (3,2,-1), (1,7,-1), (4,5,1), (5,4,-1)];

        // Component 7 (e123): 1*e123, e123*1, e1*e23, e2*e13, e3*e12, -e23*e1, -e13*e2, -e12*e3
        terms[7] = [(0,7,1), (7,0,1), (1,6,1), (2,5,1), (3,4,1), (6,1,-1), (5,2,-1), (4,3,-1)];

        Cl3StructureConstants { terms }
    }
}

/// CUDA Geometric Product for Cl(3,0) multivectors
pub struct CudaGeometricProduct {
    device: Arc<CudaDeviceContext>,
    pub(crate) ntt_ctx: CudaNttContext,
    constants: Cl3StructureConstants,
}

impl CudaGeometricProduct {
    /// Create new CUDA geometric product context
    pub fn new(n: usize, q: u64, root: u64) -> Result<Self, String> {
        let device = Arc::new(CudaDeviceContext::new()?);
        let ntt_ctx = CudaNttContext::new(n, q, root)?;
        let constants = Cl3StructureConstants::new();

        Ok(CudaGeometricProduct {
            device,
            ntt_ctx,
            constants,
        })
    }

    /// Compute homomorphic geometric product on GPU
    ///
    /// Input: Two multivectors, each with 8 components (RNS form with 2 polynomials per component)
    /// Output: Result multivector (8 components, RNS form)
    pub fn geometric_product(
        &self,
        a_multivector: &[[Vec<u64>; 2]; 8],
        b_multivector: &[[Vec<u64>; 2]; 8],
    ) -> Result<[[Vec<u64>; 2]; 8], String> {
        let n = self.ntt_ctx.n;

        // Transform all input polynomials to NTT domain
        let mut a_ntt = vec![vec![vec![0u64; n]; 2]; 8];
        let mut b_ntt = vec![vec![vec![0u64; n]; 2]; 8];

        for i in 0..8 {
            for j in 0..2 {
                a_ntt[i][j].copy_from_slice(&a_multivector[i][j]);
                self.ntt_ctx.forward(&mut a_ntt[i][j])?;

                b_ntt[i][j].copy_from_slice(&b_multivector[i][j]);
                self.ntt_ctx.forward(&mut b_ntt[i][j])?;
            }
        }

        // Initialize result
        let mut result_ntt = vec![vec![vec![0u64; n]; 2]; 8];

        // For each output component, compute 8 cross-terms
        for out_idx in 0..8 {
            for &(a_idx, b_idx, sign) in &self.constants.terms[out_idx] {
                if sign == 0 {
                    continue;
                }

                // Multiply a[a_idx] * b[b_idx] in NTT domain (for both RNS components)
                for rns_idx in 0..2 {
                    let mut product = vec![0u64; n];
                    self.ntt_ctx.pointwise_multiply(
                        &a_ntt[a_idx][rns_idx],
                        &b_ntt[b_idx][rns_idx],
                        &mut product,
                    )?;

                    // Accumulate with sign
                    if sign > 0 {
                        for k in 0..n {
                            result_ntt[out_idx][rns_idx][k] =
                                (result_ntt[out_idx][rns_idx][k] + product[k]) % self.ntt_ctx.q;
                        }
                    } else {
                        for k in 0..n {
                            let val = product[k] % self.ntt_ctx.q;
                            if result_ntt[out_idx][rns_idx][k] >= val {
                                result_ntt[out_idx][rns_idx][k] -= val;
                            } else {
                                result_ntt[out_idx][rns_idx][k] += self.ntt_ctx.q - val;
                            }
                        }
                    }
                }
            }
        }

        // Transform back to coefficient domain
        let mut result = vec![vec![vec![0u64; n]; 2]; 8];
        for i in 0..8 {
            for j in 0..2 {
                result[i][j].copy_from_slice(&result_ntt[i][j]);
                self.ntt_ctx.inverse(&mut result[i][j])?;
            }
        }

        // Convert to array format
        let mut result_array: [[Vec<u64>; 2]; 8] = Default::default();
        for i in 0..8 {
            result_array[i][0] = result[i][0].clone();
            result_array[i][1] = result[i][1].clone();
        }

        Ok(result_array)
    }
}
