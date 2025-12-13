//! Hadamard (element-wise) product kernel.
//!
//! Computes C = A ⊙ B (element-wise multiplication).
//! Optimized with vectorized loads (4 elements per thread) for high memory bandwidth.

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use crate::error::{EinsumError, EinsumResult};

/// Block size for hadamard product.
const BLOCK_SIZE: u32 = 256;

/// Elements processed per thread for better memory throughput.
const ELEMENTS_PER_THREAD: u32 = 4;

/// Launches the hadamard (element-wise) product kernel.
///
/// Computes `output = lhs * rhs` element-wise.
/// Both inputs and output must have the same shape.
pub fn launch_hadamard<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // Validate shapes match
    if lhs.shape != rhs.shape {
        return Err(EinsumError::launch("hadamard requires same shape inputs"));
    }
    if lhs.shape != output.shape {
        return Err(EinsumError::launch("hadamard output shape mismatch"));
    }

    let num_elements: usize = lhs.shape.iter().product();
    if num_elements == 0 {
        return Ok(());
    }

    // Calculate launch config with vectorized processing
    let elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    let num_cubes = ((num_elements as u32) + elements_per_block - 1) / elements_per_block;

    let cube_dim = CubeDim::new(BLOCK_SIZE, 1, 1);
    let cube_count = CubeCount::Static(num_cubes, 1, 1);

    // Launch vectorized kernel
    unsafe {
        hadamard_kernel_vectorized::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            lhs.as_arg(1),
            rhs.as_arg(1),
            output.as_arg(1),
            ScalarArg::new(num_elements as u32),
            E::as_type_native_unchecked(),
        ).map_err(|e| EinsumError::launch(alloc::format!("hadamard kernel failed: {:?}", e)))
    }
}

/// Vectorized hadamard kernel - each thread processes 4 elements.
#[cube(launch_unchecked)]
fn hadamard_kernel_vectorized<E: Numeric>(
    lhs: &Tensor<Line<E>>,
    rhs: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    num_elements: u32,
    #[define(E)] _dtype: StorageType,
) {
    let tid = UNIT_POS_X;
    let block_id = CUBE_POS_X;
    let block_size = CUBE_DIM_X;

    // Each block processes BLOCK_SIZE * 4 elements
    let block_start = block_id * block_size * 4;
    let thread_start = block_start + tid;

    // Process 4 elements per thread with stride for coalesced access
    #[unroll]
    for i in 0..4u32 {
        let idx = thread_start + i * block_size;
        if idx < num_elements {
            let a = lhs[idx];
            let b = rhs[idx];
            output[idx] = a * b;
        }
    }
}

#[cfg(test)]
mod tests {
    // Integration tests require a runtime
}
