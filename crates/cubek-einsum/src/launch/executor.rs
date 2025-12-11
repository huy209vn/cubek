//! Einsum execution engine.
//!
//! Orchestrates parsing, optimization, and kernel dispatch.

use alloc::vec::Vec;
use alloc::vec;
use alloc::string::ToString;

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::ir::StorageType;
use cubecl::std::tensor::TensorHandle;

use cubek_matmul::{
    Strategy as MatmulStrategy,
    MatmulInputHandle, MatmulInputHandleRef,
    components::MatmulElems,
    tune_key::MatmulElemType,
};
use cubek_reduce::{
    ReduceDtypes,
    components::instructions::ReduceOperationConfig,
};

use crate::error::{EinsumError, EinsumResult};
use crate::notation::{parse_einsum, EinsumNotation, validate_notation};
use crate::notation::validation::validate_shapes;
use crate::optimization::{create_plan, ExecutionStep, ContractionStrategy};
use crate::pattern::FastPath;
use super::config::EinsumConfig;

/// Executes an einsum operation.
///
/// # Arguments
/// * `client` - The compute client
/// * `notation` - Einsum notation string (e.g., "ij,jk->ik")
/// * `inputs` - Input tensor handles
/// * `output` - Output tensor handle
/// * `config` - Optional configuration
///
/// # Example
///
/// ```ignore
/// let output = einsum(
///     &client,
///     "ij,jk->ik",
///     &[&a, &b],
///     &mut c,
///     None,
/// )?;
/// ```
pub fn einsum<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    notation_str: &str,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    config: Option<EinsumConfig>,
) -> EinsumResult<()> {
    let config = config.unwrap_or_default();

    // Parse notation
    let notation = parse_einsum(notation_str)?;

    // Validate notation
    validate_notation(&notation)?;

    // Extract shapes
    let shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape.as_slice()).collect();

    // Validate shapes if enabled
    if config.validate_shapes {
        let _ = validate_shapes(&notation, &shapes)?;
    }

    // Create execution plan
    let plan = create_plan(&notation, &shapes, config.strategy);

    // Execute plan
    execute_plan::<R, E>(client, &plan, inputs, output, &config)
}

/// Executes a pre-parsed einsum notation.
///
/// Useful when the same notation will be executed multiple times.
pub fn einsum_with_notation<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    notation: &EinsumNotation,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    config: Option<EinsumConfig>,
) -> EinsumResult<()> {
    let config = config.unwrap_or_default();

    // Extract shapes
    let shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape.as_slice()).collect();

    // Validate shapes if enabled
    if config.validate_shapes {
        let _ = validate_shapes(notation, &shapes)?;
    }

    // Create execution plan
    let plan = create_plan(notation, &shapes, config.strategy);

    // Execute plan
    execute_plan::<R, E>(client, &plan, inputs, output, &config)
}

/// Executes an execution plan.
fn execute_plan<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    plan: &crate::optimization::ExecutionPlan,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    config: &EinsumConfig,
) -> EinsumResult<()> {
    if plan.uses_fast_path() {
        // Single fast-path operation
        match &plan.steps()[0] {
            ExecutionStep::FastPath(fast_path) => {
                execute_fast_path::<R, E>(client, fast_path, inputs, output, config)
            }
            _ => Err(EinsumError::unsupported("invalid plan structure")),
        }
    } else {
        // General contraction path
        execute_contractions::<R, E>(client, plan, inputs, output, config)
    }
}

/// Executes a fast-path operation.
fn execute_fast_path<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    fast_path: &FastPath,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    _config: &EinsumConfig,
) -> EinsumResult<()> {
    match fast_path {
        FastPath::Matmul { transpose_a, transpose_b } => {
            execute_matmul::<R, E>(client, inputs, output, *transpose_a, *transpose_b, &[])
        }
        FastPath::BatchedMatmul { batch_dims, transpose_a, transpose_b } => {
            execute_matmul::<R, E>(client, inputs, output, *transpose_a, *transpose_b, batch_dims)
        }
        FastPath::Reduce { axes, .. } => {
            execute_reduce::<R, E>(client, inputs, output, axes)
        }
        FastPath::Transpose { permutation } => {
            execute_transpose::<R, E>(inputs, output, permutation)
        }
        FastPath::Hadamard => {
            execute_hadamard::<R, E>(client, inputs, output)
        }
        FastPath::OuterProduct => {
            execute_outer_product::<R, E>(client, inputs, output)
        }
        FastPath::DotProduct => {
            execute_dot_product::<R, E>(client, inputs, output)
        }
        FastPath::Trace => {
            execute_trace::<R, E>(client, inputs, output)
        }
        FastPath::DiagonalExtract => {
            execute_diagonal::<R, E>(client, inputs, output)
        }
    }
}

/// Executes matrix multiplication via cubek-matmul.
fn execute_matmul<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    transpose_a: bool,
    transpose_b: bool,
    _batch_dims: &[usize],
) -> EinsumResult<()> {
    if inputs.len() < 2 {
        return Err(EinsumError::unsupported("matmul requires 2 inputs"));
    }

    let mut lhs = inputs[0].clone();
    let mut rhs = inputs[1].clone();

    // Handle transposition by swapping the last two dimensions
    // cubek-matmul expects row-major layout, transposition is handled via strides
    if transpose_a {
        let ndim = lhs.shape.len();
        if ndim >= 2 {
            lhs.shape.swap(ndim - 2, ndim - 1);
            lhs.strides.swap(ndim - 2, ndim - 1);
        }
    }

    if transpose_b {
        let ndim = rhs.shape.len();
        if ndim >= 2 {
            rhs.shape.swap(ndim - 2, ndim - 1);
            rhs.strides.swap(ndim - 2, ndim - 1);
        }
    }

    // Create element type
    let elem_type = MatmulElemType::new(E::as_type_native_unchecked(), false);

    // Create MatmulElems from single dtype (all same type)
    let mut dtypes = MatmulElems::from_single_dtype(elem_type);

    // Create input handles
    let lhs_handle = MatmulInputHandle::Normal(lhs);
    let rhs_handle = MatmulInputHandle::Normal(rhs);

    // Use Auto strategy for best performance
    let strategy = MatmulStrategy::Auto;

    // Launch matmul
    cubek_matmul::launch(
        &strategy,
        client,
        lhs_handle,
        rhs_handle,
        output.clone(),
        dtypes,
    ).map_err(|e| EinsumError::launch(alloc::format!("matmul failed: {:?}", e)))
}

/// Executes reduction via cubek-reduce.
fn execute_reduce<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    axes: &[usize],
) -> EinsumResult<()> {
    if inputs.is_empty() {
        return Err(EinsumError::unsupported("reduce requires at least 1 input"));
    }

    let input = inputs[0];

    // For multiple axes, we need to reduce sequentially
    if axes.len() > 1 {
        return execute_multi_axis_reduce::<R, E>(client, input, output, axes);
    }

    if axes.is_empty() {
        // No reduction needed, just copy
        // For now, return error - this shouldn't happen in valid einsum
        return Err(EinsumError::unsupported("empty reduction axes"));
    }

    let axis = axes[0];

    // Get optimal precision for sum operation
    let operation = ReduceOperationConfig::Sum;
    let dtypes = operation.precision(E::as_type_native_unchecked());

    // Launch reduce
    cubek_reduce::reduce(
        client,
        input.as_ref(),
        output.as_ref(),
        axis,
        None, // Auto strategy
        operation,
        dtypes,
    ).map_err(|e| EinsumError::launch(alloc::format!("reduce failed: {:?}", e)))
}

/// Executes multi-axis reduction by reducing one axis at a time.
fn execute_multi_axis_reduce<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    input: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
    axes: &[usize],
) -> EinsumResult<()> {
    // Sort axes in descending order so we can reduce without invalidating indices
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_by(|a, b| b.cmp(a));

    let operation = ReduceOperationConfig::Sum;
    let dtypes = operation.precision(E::as_type_native_unchecked());

    // For now, we need intermediate tensors for multi-axis reduction
    // This is a simplified implementation - proper workspace management would be better

    let mut current_input = input.clone();

    for (i, &axis) in sorted_axes.iter().enumerate() {
        let is_last = i == sorted_axes.len() - 1;

        // Compute output shape for this reduction
        let mut reduced_shape = current_input.shape.clone();
        reduced_shape[axis] = 1;

        // Compute strides for reduced tensor
        let reduced_strides = compute_strides(&reduced_shape);

        let target = if is_last {
            output.as_ref()
        } else {
            // Need to allocate intermediate
            // For simplicity, we'll return an error for multi-axis for now
            return Err(EinsumError::unsupported(
                "multi-axis reduction requires workspace allocation - not yet implemented"
            ));
        };

        cubek_reduce::reduce(
            client,
            current_input.as_ref(),
            target,
            axis,
            None,
            operation,
            dtypes,
        ).map_err(|e| EinsumError::launch(alloc::format!("reduce failed: {:?}", e)))?;
    }

    Ok(())
}

/// Computes strides for a given shape (row-major order).
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Executes transpose operation (zero-copy via stride manipulation).
fn execute_transpose<R: Runtime, E: CubePrimitive + Numeric>(
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    permutation: &[usize],
) -> EinsumResult<()> {
    if inputs.is_empty() {
        return Err(EinsumError::unsupported("transpose requires 1 input"));
    }

    let input = inputs[0];

    // Apply permutation to shape and strides
    let new_shape: Vec<usize> = permutation.iter().map(|&i| input.shape[i]).collect();
    let new_strides: Vec<usize> = permutation.iter().map(|&i| input.strides[i]).collect();

    // Update output metadata
    // Note: This assumes output shares the same underlying buffer as input
    // For a true einsum operation, the caller should set up output appropriately
    output.shape = new_shape;
    output.strides = new_strides;

    Ok(())
}

/// Executes Hadamard (element-wise) product.
fn execute_hadamard<R: Runtime, E: CubePrimitive + Numeric>(
    _client: &ComputeClient<R>,
    _inputs: &[&TensorHandle<R>],
    _output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // TODO: Implement element-wise multiplication kernel
    // Could use a simple #[cube] kernel
    Err(EinsumError::unsupported("hadamard product not yet implemented"))
}

/// Executes outer product.
fn execute_outer_product<R: Runtime, E: CubePrimitive + Numeric>(
    _client: &ComputeClient<R>,
    _inputs: &[&TensorHandle<R>],
    _output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // TODO: Implement outer product kernel
    Err(EinsumError::unsupported("outer product not yet implemented"))
}

/// Executes dot product (reduction after element-wise multiply).
fn execute_dot_product<R: Runtime, E: CubePrimitive + Numeric>(
    _client: &ComputeClient<R>,
    _inputs: &[&TensorHandle<R>],
    _output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // TODO: Implement as hadamard + reduce
    Err(EinsumError::unsupported("dot product not yet implemented"))
}

/// Executes trace (sum of diagonal).
fn execute_trace<R: Runtime, E: CubePrimitive + Numeric>(
    _client: &ComputeClient<R>,
    _inputs: &[&TensorHandle<R>],
    _output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // TODO: Implement trace kernel
    Err(EinsumError::unsupported("trace not yet implemented"))
}

/// Executes diagonal extraction.
fn execute_diagonal<R: Runtime, E: CubePrimitive + Numeric>(
    _client: &ComputeClient<R>,
    _inputs: &[&TensorHandle<R>],
    _output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // TODO: Implement diagonal extraction kernel
    Err(EinsumError::unsupported("diagonal extraction not yet implemented"))
}

/// Executes a general contraction sequence.
fn execute_contractions<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    plan: &crate::optimization::ExecutionPlan,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    _config: &EinsumConfig,
) -> EinsumResult<()> {
    // For general contractions, we need workspace management
    // Each step produces an intermediate tensor that feeds into the next step

    let steps = plan.steps();
    if steps.is_empty() {
        return Ok(());
    }

    // For now, only support simple 2-tensor contractions that reduce to matmul
    // More complex chains require proper workspace allocation

    if steps.len() == 1 {
        if let ExecutionStep::Contraction { inputs: (i, j), contracted, result, .. } = &steps[0] {
            // Check if this is effectively a matmul
            if *i < inputs.len() && *j < inputs.len() {
                // Try to execute as matmul if structure matches
                // This is a simplified check - proper implementation would analyze the indices
                return execute_general_contraction::<R, E>(
                    client,
                    inputs[*i],
                    inputs[*j],
                    output,
                    contracted,
                    result,
                );
            }
        }
    }

    Err(EinsumError::unsupported(
        "general multi-step contractions require workspace management - not yet implemented"
    ))
}

/// Executes a general two-tensor contraction.
fn execute_general_contraction<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
    _contracted: &[char],
    _result: &[char],
) -> EinsumResult<()> {
    // For general contractions, we try to map to matmul
    // This is a simplified approach - proper implementation would generate custom kernels

    // For now, attempt matmul with auto-detected transposition
    let elem_type = MatmulElemType::new(E::as_type_native_unchecked(), false);
    let dtypes = MatmulElems::from_single_dtype(elem_type);

    let lhs_handle = MatmulInputHandle::Normal(lhs.clone());
    let rhs_handle = MatmulInputHandle::Normal(rhs.clone());

    cubek_matmul::launch(
        &MatmulStrategy::Auto,
        client,
        lhs_handle,
        rhs_handle,
        output.clone(),
        dtypes,
    ).map_err(|e| EinsumError::launch(alloc::format!("contraction failed: {:?}", e)))
}

#[cfg(test)]
mod tests {
    // Integration tests would go here, but require a runtime
}
