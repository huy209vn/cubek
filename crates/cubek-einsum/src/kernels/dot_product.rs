//! Dot product kernel.
//!
//! Computes scalar = sum(A ⊙ B) = A · B.
//! Implementation: hadamard product into workspace, then reduce.

use alloc::vec;

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use cubek_reduce::components::instructions::ReduceOperationConfig;

use crate::error::{EinsumError, EinsumResult};
use super::hadamard::launch_hadamard;

/// Launches the dot product kernel.
///
/// Computes `output = sum(lhs * rhs)` as a scalar.
/// Both inputs must have the same shape.
///
/// Implementation: computes hadamard product into workspace, then reduces.
pub fn launch_dot_product<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // Validate shapes match
    if lhs.shape != rhs.shape {
        return Err(EinsumError::launch("dot product requires same shape inputs"));
    }

    let num_elements: usize = lhs.shape.iter().product();
    if num_elements == 0 {
        return Ok(());
    }

    // Output should be scalar
    let output_size: usize = output.shape.iter().product();
    if output_size != 1 {
        return Err(EinsumError::launch(alloc::format!(
            "dot product output should be scalar, got size {}",
            output_size
        )));
    }

    // Step 1: Allocate workspace for hadamard product (flattened to 1D)
    let dtype = lhs.dtype;
    let workspace_shape = vec![num_elements];
    let mut hadamard_workspace = TensorHandle::zeros(client, workspace_shape, dtype);

    // Temporarily reshape inputs for hadamard kernel (they expect same shape)
    // Create flattened views
    let mut lhs_flat = lhs.clone();
    lhs_flat.shape = vec![num_elements];
    lhs_flat.strides = vec![1];

    let mut rhs_flat = rhs.clone();
    rhs_flat.shape = vec![num_elements];
    rhs_flat.strides = vec![1];

    // Step 2: Compute hadamard product into workspace
    launch_hadamard::<R, E>(client, &lhs_flat, &rhs_flat, &mut hadamard_workspace)?;

    // Step 3: Reduce workspace to scalar along axis 0
    let operation = ReduceOperationConfig::Sum;
    let elem_type = dtype.elem_type();
    let dtypes = operation.precision(elem_type);

    cubek_reduce::reduce(
        client,
        hadamard_workspace.as_ref(),
        output.as_ref(),
        0, // Reduce the single axis
        None, // Auto strategy
        operation,
        dtypes,
    ).map_err(|e| EinsumError::launch(alloc::format!("dot product reduce failed: {:?}", e)))
}

#[cfg(test)]
mod tests {
    // Integration tests require a runtime
}
