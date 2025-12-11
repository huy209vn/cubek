//! Contraction path optimization for einsum.
//!
//! Implements multiple strategies for finding optimal contraction orderings:
//! - Greedy: O(n³) fast heuristic
//! - Dynamic Programming: Optimal for small n
//! - Branch and Bound: Good balance for medium n

mod cost;
mod greedy;
mod dynamic;
mod path;
mod plan;

pub use cost::{CostModel, ContractionCost};
pub use greedy::greedy_path;
pub use dynamic::optimal_path;
pub use path::{ContractionPath, ContractionStep};
pub use plan::{ExecutionPlan, ExecutionStep, ContractionStrategy, create_plan};
