//! GPU kernels for einsum operations.
//!
//! Contains implementations of:
//! - Element-wise operations (hadamard, outer product)
//! - Reduction operations (dot product, trace)
//! - Diagonal operations (extraction)

mod hadamard;
mod outer_product;
mod dot_product;
mod trace;
mod diagonal;

pub use hadamard::launch_hadamard;
pub use outer_product::launch_outer_product;
pub use dot_product::launch_dot_product;
pub use trace::launch_trace;
pub use diagonal::launch_diagonal;
