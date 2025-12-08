use crate::suite::layered::matmul_test_launcher::tensor_size;
use crate::suite::test_utils::Sample;
use cubecl::{TestRuntime, prelude::*};
use cubecl::{
    client::ComputeClient,
    server::{Allocation, AllocationDescriptor},
};
use cubek_matmul::components::{MatmulIdent, MatmulProblem, MatrixLayout};

use crate::suite::layered::matmul_test_launcher::{TensorRawParts, transpose};

pub(crate) fn tensor_raw_parts<E: CubeElement + Numeric + Sample>(
    client: &ComputeClient<TestRuntime>,
    problem: &MatmulProblem,
    ident: MatmulIdent,
) -> TensorRawParts<E> {
    match ident {
        MatmulIdent::Lhs => {
            let mut tensor_shape = problem.shape(MatmulIdent::Lhs);

            let handle = E::sample(client, &tensor_shape, 1234);

            let data = client.read_one_tensor(handle.as_copy_descriptor());
            let data = E::from_bytes(&data);
            let original_data = data.to_owned();

            let rank = tensor_shape.len();

            let data = match problem.lhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    tensor_shape.swap(rank - 1, rank - 2);
                    transpose::<E>(&original_data, problem.num_batches(), problem.m, problem.k)
                }
            };
            let descriptors = vec![(
                AllocationDescriptor::optimized(tensor_shape.as_slice(), E::type_size() as usize),
                E::as_bytes(&data),
            )];

            let mut tensors = client.create_tensors_from_slices(descriptors);
            let Allocation {
                handle,
                mut strides,
            } = tensors.remove(0);

            if matches!(problem.lhs_layout, MatrixLayout::ColMajor) {
                tensor_shape.swap(rank - 1, rank - 2);
                strides.swap(rank - 1, rank - 2);
            }

            let _offs = tensors.pop();
            let scale = tensors.pop().map(|it| it.handle);

            TensorRawParts {
                handle,
                scale,
                shape: tensor_shape,
                strides,
                original_data: Some(original_data),
            }
        }
        MatmulIdent::Rhs => {
            let mut tensor_shape = problem.shape(MatmulIdent::Rhs);

            let handle = E::sample(client, &tensor_shape, 5678);

            let data = client.read_one_tensor(handle.as_copy_descriptor());
            let data = E::from_bytes(&data);
            let original_data = data.to_owned();

            let rank = tensor_shape.len();

            let data = match problem.rhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    tensor_shape.swap(rank - 1, rank - 2);
                    transpose::<E>(&original_data, problem.num_batches(), problem.k, problem.n)
                }
            };

            let descriptors = vec![(
                AllocationDescriptor::optimized(tensor_shape.as_slice(), size_of::<E>()),
                E::as_bytes(&data),
            )];

            let mut tensors = client.create_tensors_from_slices(descriptors);
            let Allocation {
                handle,
                mut strides,
            } = tensors.remove(0);
            let _offs = tensors.pop();
            let scale = tensors.pop().map(|it| it.handle);

            if matches!(problem.rhs_layout, MatrixLayout::ColMajor) {
                tensor_shape.swap(rank - 1, rank - 2);
                strides.swap(rank - 1, rank - 2);
            }

            TensorRawParts {
                handle,
                scale,
                shape: tensor_shape,
                strides,
                original_data: Some(original_data),
            }
        }
        MatmulIdent::Out => {
            let zero = E::from_int(0);

            let data = vec![zero; tensor_size(problem, MatmulIdent::Out)];

            let tensor_shape = problem.shape(MatmulIdent::Out);

            let descriptors = vec![(
                AllocationDescriptor::optimized(tensor_shape.as_slice(), size_of::<E>()),
                E::as_bytes(&data),
            )];

            let mut tensors = client.create_tensors_from_slices(descriptors);
            let Allocation { handle, strides } = tensors.remove(0);
            let _offs = tensors.pop();
            let scale = tensors.pop().map(|it| it.handle);

            TensorRawParts {
                handle,
                scale,
                shape: tensor_shape,
                strides,
                original_data: None,
            }
        }
    }
}
