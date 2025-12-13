//! Hadamard (element-wise) product kernel.
//!
//! Computes C = A ⊙ B (element-wise multiplication).

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use crate::error::{EinsumError, EinsumResult};

/// Launch configuration for hadamard product.
const CUBE_DIM_DEFAULT: u32 = 256;

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

    // Compute launch config
    let cube_dim = CubeDim::new(CUBE_DIM_DEFAULT, 1, 1);
    let num_cubes = (num_elements as u32 + CUBE_DIM_DEFAULT - 1) / CUBE_DIM_DEFAULT;
    let cube_count = CubeCount::Static(num_cubes, 1, 1);

    // Launch kernel
    unsafe {
        hadamard_kernel::launch_unchecked::<R>(
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

#[cube(launch_unchecked)]
fn hadamard_kernel<E: Numeric>(
    lhs: &Tensor<Line<E>>,
    rhs: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    num_elements: u32,
    #[define(E)] _dtype: StorageType,
) {
    let global_id = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;

    if global_id < num_elements {
        let a = lhs[global_id];
        let b = rhs[global_id];
        output[global_id] = a * b;
    }
}

#[cfg(test)]
mod tests {
    // Integration tests require a runtime
}
