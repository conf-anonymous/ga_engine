//! SVP Challenge Format Parser
//!
//! Parses lattice files from the SVP Challenge (https://www.latticechallenge.org/svp-challenge/)
//!
//! # Format
//!
//! SVP Challenge lattices are stored as:
//! ```text
//! [[row1_val1 row1_val2 ... row1_valn]
//!  [row2_val1 row2_val2 ... row2_valn]
//!  ...
//!  [rown_val1 rown_val2 ... rown_valn]]
//! ```
//!
//! Each row is a basis vector. Values can be very large integers (128+ bits).
//!
//! # Note on Integer Size
//!
//! SVP Challenge uses arbitrary precision integers. For our purposes, we'll convert
//! to f64, which is sufficient for dimensions up to ~100 (the conversion loses some
//! precision but LLL is robust to this).
//!
//! For production cryptanalysis on larger dimensions, we'd need arbitrary precision
//! arithmetic throughout.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Parse SVP Challenge lattice file
///
/// # Arguments
///
/// * `path` - Path to the lattice file
///
/// # Returns
///
/// Basis vectors as Vec<Vec<f64>> where each inner Vec is a row (basis vector)
///
/// # Errors
///
/// Returns error if file cannot be read or parsed
pub fn parse_lattice_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<f64>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read entire file into string
    // SVP Challenge format has each row on a separate line:
    // [[row1]
    //  [row2]
    //  ...
    //  [rowN]
    // ]
    let mut content = String::new();
    for line in reader.lines() {
        let line = line?;
        content.push_str(&line);
        content.push('\n'); // Preserve newlines
    }

    // Normalize format:
    // - Replace "]\n[" with "] [" (between rows)
    // - Replace "]\n]" with "]]" (closing bracket)
    // - Then replace any remaining "\n" with " "
    let content = content
        .replace("]\n[", "] [")
        .replace("]\n]", "]]")
        .replace('\n', " ")
        .trim()
        .to_string();

    // Parse the matrix
    parse_lattice_string(&content)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Parse lattice from string
///
/// Handles the bracket format: [[a b c] [d e f] ...]
fn parse_lattice_string(s: &str) -> Result<Vec<Vec<f64>>, String> {
    let s = s.trim();

    // Remove outer brackets
    if !s.starts_with("[[") || !s.ends_with("]]") {
        return Err("Expected format [[...] [...] ...]".to_string());
    }

    let s = &s[2..s.len() - 2]; // Remove [[ and ]]

    // Split by "] [" to get individual rows
    let rows: Vec<&str> = s.split("] [").collect();

    let mut basis: Vec<Vec<f64>> = Vec::new();

    for (i, row_str) in rows.iter().enumerate() {
        let row_str = row_str.trim();

        // Parse numbers from this row
        let values: Result<Vec<f64>, _> = row_str
            .split_whitespace()
            .map(|s| {
                // Try parsing as integer first (handles large numbers better)
                // Then convert to f64
                if let Ok(val) = s.parse::<i64>() {
                    Ok(val as f64)
                } else {
                    // Very large number - parse as f64 directly
                    // This will lose precision but that's okay for our purposes
                    s.parse::<f64>()
                        .map_err(|_| format!("Failed to parse '{}' in row {}", s, i))
                }
            })
            .collect();

        let row = values.map_err(|e| format!("Row {}: {}", i, e))?;

        if !row.is_empty() {
            // Check dimension consistency
            if !basis.is_empty() && row.len() != basis[0].len() {
                return Err(format!(
                    "Row {} has {} elements, expected {}",
                    i,
                    row.len(),
                    basis[0].len()
                ));
            }
            basis.push(row);
        }
    }

    if basis.is_empty() {
        return Err("No basis vectors found".to_string());
    }

    Ok(basis)
}

/// Get dimension from lattice
pub fn get_dimension(basis: &[Vec<f64>]) -> usize {
    if basis.is_empty() {
        0
    } else {
        basis[0].len()
    }
}

/// Compute norm of a vector
pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Find shortest vector in basis
pub fn find_shortest_vector(basis: &[Vec<f64>]) -> (usize, f64) {
    let mut min_norm = f64::INFINITY;
    let mut min_idx = 0;

    for (i, v) in basis.iter().enumerate() {
        let n = norm(v);
        if n < min_norm {
            min_norm = n;
            min_idx = i;
        }
    }

    (min_idx, min_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_lattice() {
        let input = "[[1 2 3] [4 5 6] [7 8 9]]";
        let basis = parse_lattice_string(input).unwrap();

        assert_eq!(basis.len(), 3);
        assert_eq!(basis[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(basis[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(basis[2], vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_parse_large_integers() {
        let input = "[[1234567890123456789 0] [0 9876543210987654321]]";
        let basis = parse_lattice_string(input).unwrap();

        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0].len(), 2);
        // Large integers will be converted to f64 (with some loss of precision)
        assert!(basis[0][0] > 1e18);
        assert!(basis[1][1] > 9e18);
    }

    #[test]
    fn test_get_dimension() {
        let basis = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        assert_eq!(get_dimension(&basis), 3);
    }

    #[test]
    fn test_find_shortest() {
        let basis = vec![
            vec![10.0, 0.0],
            vec![1.0, 1.0],  // Shortest: norm ≈ 1.414
            vec![5.0, 5.0],
        ];

        let (idx, norm_val) = find_shortest_vector(&basis);
        assert_eq!(idx, 1);
        assert!((norm_val - 1.414).abs() < 0.01);
    }
}
