//! Contraction path optimization tests.

use cubek_einsum::notation::parse_einsum;
use cubek_einsum::optimization::{
    greedy_path, optimal_path, create_plan,
    CostModel, ContractionStrategy,
};

#[test]
fn test_greedy_two_tensors() {
    let notation = parse_einsum("ij,jk->ik").unwrap();
    let shapes: &[&[usize]] = &[&[100, 200], &[200, 300]];
    let cost_model = CostModel::default();

    let path = greedy_path(&notation, shapes, &cost_model);
    assert_eq!(path.len(), 1);
}

#[test]
fn test_greedy_three_tensors() {
    let notation = parse_einsum("ij,jk,kl->il").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];
    let cost_model = CostModel::default();

    let path = greedy_path(&notation, shapes, &cost_model);
    assert_eq!(path.len(), 2);
}

#[test]
fn test_optimal_two_tensors() {
    let notation = parse_einsum("ij,jk->ik").unwrap();
    let shapes: &[&[usize]] = &[&[100, 200], &[200, 300]];
    let cost_model = CostModel::default();

    let path = optimal_path(&notation, shapes, &cost_model);
    assert_eq!(path.len(), 1);
}

#[test]
fn test_optimal_three_tensors() {
    let notation = parse_einsum("ij,jk,kl->il").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];
    let cost_model = CostModel::default();

    let path = optimal_path(&notation, shapes, &cost_model);
    assert_eq!(path.len(), 2);
}

#[test]
fn test_plan_uses_fast_path_for_matmul() {
    let notation = parse_einsum("ij,jk->ik").unwrap();
    let shapes: &[&[usize]] = &[&[100, 200], &[200, 300]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);
    assert!(plan.uses_fast_path());
}

#[test]
fn test_plan_no_fast_path_for_chain() {
    let notation = parse_einsum("ij,jk,kl->il").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);
    assert!(!plan.uses_fast_path());
}

#[test]
fn test_cost_model() {
    let model = CostModel::default();

    let cost = model.compute_pairwise_cost(
        &[100, 200],
        &[200, 300],
        &['i', 'j'],
        &['j', 'k'],
        &['j'],
    );

    // FLOPs = M * N * K * 2 = 100 * 300 * 200 * 2 = 12,000,000
    assert_eq!(cost.flops, 12_000_000);
}
