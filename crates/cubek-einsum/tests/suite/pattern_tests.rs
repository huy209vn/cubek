//! Pattern recognition tests.

use cubek_einsum::notation::parse_einsum;
use cubek_einsum::pattern::{recognize_pattern, FastPath};

#[test]
fn test_recognize_matmul() {
    let notation = parse_einsum("ij,jk->ik").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(matches!(pattern, Some(FastPath::Matmul { .. })));
}

#[test]
fn test_recognize_matmul_transpose_a() {
    let notation = parse_einsum("ji,jk->ik").unwrap();
    let pattern = recognize_pattern(&notation);
    match pattern {
        Some(FastPath::Matmul { transpose_a, transpose_b }) => {
            assert!(transpose_a);
            assert!(!transpose_b);
        }
        _ => panic!("expected matmul pattern"),
    }
}

#[test]
fn test_recognize_batched_matmul() {
    let notation = parse_einsum("bij,bjk->bik").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(matches!(pattern, Some(FastPath::BatchedMatmul { .. })));
}

#[test]
fn test_recognize_transpose() {
    let notation = parse_einsum("ij->ji").unwrap();
    let pattern = recognize_pattern(&notation);
    match pattern {
        Some(FastPath::Transpose { permutation }) => {
            assert_eq!(permutation, vec![1, 0]);
        }
        _ => panic!("expected transpose pattern"),
    }
}

#[test]
fn test_recognize_reduction() {
    let notation = parse_einsum("ij->i").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(matches!(pattern, Some(FastPath::Reduce { .. })));
}

#[test]
fn test_recognize_hadamard() {
    let notation = parse_einsum("ij,ij->ij").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(matches!(pattern, Some(FastPath::Hadamard)));
}

#[test]
fn test_recognize_outer_product() {
    let notation = parse_einsum("i,j->ij").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(matches!(pattern, Some(FastPath::OuterProduct)));
}

#[test]
fn test_recognize_dot_product() {
    let notation = parse_einsum("i,i->").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(matches!(pattern, Some(FastPath::DotProduct)));
}

#[test]
fn test_recognize_trace() {
    let notation = parse_einsum("ii->").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(matches!(pattern, Some(FastPath::Trace)));
}

#[test]
fn test_no_pattern_complex() {
    let notation = parse_einsum("ijk,jkl,klm->im").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(pattern.is_none());
}
