//! N-dimensional multivectors and geometric product.
//!
//! A multivector in N-dimensional Euclidean space has 2^N components,
//! one for each basis blade.

use crate::nd::types::Scalar;
use std::ops::{Add, Mul, Sub};

/// A full multivector in N dims: 2^N components in lexicographic blade order.
/// `data[0]` is the scalar part; `data[(1<<N)-1]` is the pseudoscalar.
#[derive(Debug, Clone, PartialEq)]
pub struct Multivector<const N: usize> {
    /// Raw components `[c_0, c_1, …, c_{2^N-1}]`
    pub data: Vec<Scalar>, // length must equal 2^N
}

impl<const N: usize> Multivector<N> {
    /// Construct from a raw component Vec (must be length 2^N).
    pub fn new(data: Vec<Scalar>) -> Self {
        let expected = 1 << N;
        assert!(
            data.len() == expected,
            "Multivector<{}> requires {} components, got {}",
            N,
            expected,
            data.len()
        );
        Self { data }
    }

    /// The zero multivector (all components zero).
    pub fn zero() -> Self {
        Multivector {
            data: vec![0.0 as Scalar; 1 << N],
        }
    }

    /// Geometric product: `self * other`, using inline sign/index computation.
    pub fn gp(&self, other: &Self) -> Self {
        let m = 1 << N;
        let mut out = vec![0.0 as Scalar; m];
        for i in 0..m {
            let a = self.data[i];
            if a == 0.0 { continue; }
            for j in 0..m {
                let b = other.data[j];
                if b == 0.0 { continue; }

                let k = i ^ j;

                let mut sgn = 1i32;
                for bit in 0..N {
                    if (i >> bit) & 1 != 0 {
                        let mut lower = j & ((1 << bit) - 1);
                        let mut cnt = 0;
                        while lower != 0 {
                            cnt += lower & 1;
                            lower >>= 1;
                        }
                        if cnt & 1 != 0 {
                            sgn = -sgn;
                        }
                    }
                }

                let sign = if sgn > 0 { 1.0 } else { -1.0 };
                out[k] += sign * a * b;
            }
        }
        Multivector { data: out }
    }
}

impl<const N: usize> Default for Multivector<N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: usize> Add for Multivector<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let data = self
            .data
            .into_iter()
            .zip(rhs.data)
            .map(|(a, b)| a + b)
            .collect();
        Multivector { data }
    }
}

impl<const N: usize> Sub for Multivector<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let data = self
            .data
            .into_iter()
            .zip(rhs.data)
            .map(|(a, b)| a - b)
            .collect();
        Multivector { data }
    }
}

impl<const N: usize> Mul<Scalar> for Multivector<N> {
    type Output = Self;
    fn mul(self, s: Scalar) -> Self {
        let data = self.data.into_iter().map(|v| v * s).collect();
        Multivector { data }
    }
}
