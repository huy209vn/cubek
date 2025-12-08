use cubecl::server::Allocation;
use cubecl::{TestRuntime, prelude::*};

use cubek_matmul::components::MatmulProblem;
use cubek_matmul::components::MatmulSelection;
use cubek_matmul::components::batch::BatchConfig;
use cubek_matmul::components::batch::BatchMatmulFamily;
use cubek_matmul::components::global::args::TensorMapArgs;
use cubek_matmul::components::global::args::{ConcreteInputsFactory, TensorMapInputs};
use cubek_matmul::components::{AvailableLineSizes, MatrixLayout};
use cubek_matmul::components::{MatmulElems, MatmulIdent};
use cubek_matmul::kernels::layered::Algorithm;
use cubek_matmul::{
    MatmulInputHandleRef,
    components::global::args::{ConcreteOutputFactory, TensorOutput},
};

use crate::suite::TestEG;
use crate::suite::test_utils::{Sample, assert_result};

use super::matmul_test_launcher::{TensorRawParts, tensor_size, transpose};

/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_tma_matmul_algorithm<A: Algorithm>(
    client: ComputeClient<TestRuntime>,
    mut problem: MatmulProblem,
    selection: MatmulSelection,
    dtypes: MatmulElems,
) {
    let env = std::env::var("MATMUL_TEST_MODE");

    let panic_on_launch_err = match env {
        Ok(val) => match val.as_str() {
            "panic" => true,
            "skip" => false,
            _ => false,
        },
        Err(_) => false,
    };

    let line_sizes = AvailableLineSizes::from_type_sizes(
        &client,
        dtypes.lhs_global.size(),
        dtypes.rhs_global.size(),
        dtypes.acc_global.size(),
    );
    let line_sizes = A::filter_line_sizes(line_sizes);
    let line_sizes = line_sizes
        .filter_lhs(|ls| *ls == 1)
        .filter_rhs(|ls| *ls == 1)
        .pick_max()
        .unwrap();
    // let dtypes = MatmulElems::new_with_tile::<P::MP, A::TileMatmul>();
    let config = match A::setup(&client, &problem, &selection, &line_sizes, &dtypes) {
        Ok(config) => config,
        Err(err) => {
            let msg = format!("Can't launch the test: {err}");
            if panic_on_launch_err {
                panic!("{msg}");
            } else {
                println!("{msg}");
                return;
            }
        }
    };

    let line_sizes = config.line_sizes();

    let lhs = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Lhs);
    let rhs = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Rhs);
    let out = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Out);

    problem.lhs_strides = lhs.strides.clone();
    problem.rhs_strides = rhs.strides.clone();

    let lhs_handle = MatmulInputHandleRef::Normal(
        unsafe {
            TensorHandleRef::from_raw_parts(
                &lhs.handle,
                &lhs.strides,
                &lhs.shape,
                dtypes.lhs_global.size(),
            )
        },
        *dtypes.lhs_global,
    );
    let rhs_handle = MatmulInputHandleRef::Normal(
        unsafe {
            TensorHandleRef::from_raw_parts(
                &rhs.handle,
                &rhs.strides,
                &rhs.shape,
                dtypes.rhs_global.size(),
            )
        },
        *dtypes.rhs_global,
    );
    let out_handle = unsafe {
        TensorHandleRef::from_raw_parts(
            &out.handle,
            &out.strides,
            &out.shape,
            dtypes.acc_global.size(),
        )
    };

    let inputs = TensorMapInputs::create(
        &client,
        &lhs_handle,
        &rhs_handle,
        &selection,
        &problem,
        &line_sizes,
        config,
        &dtypes,
    );
    let output = TensorOutput::create(
        &client,
        &out_handle,
        &selection,
        &problem,
        &line_sizes,
        config,
        &dtypes,
    );
    let cube_count_plan = config.hypercube_config().cube_count_plan(
        &problem,
        client.properties().hardware.max_cube_count.clone(),
    );

    let result = unsafe {
        A::BatchMatmul::launch_unchecked::<TensorMapArgs, TestRuntime>(
            &client,
            config.cube_dim(),
            cube_count_plan.resolve(),
            inputs,
            output,
            cube_count_plan.as_args(),
            config,
            &dtypes,
        )
    };

    match result {
        Ok(()) => {}
        Err(_err) => return,
    }

    assert_result(
        &lhs.original_data.unwrap(),
        &rhs.original_data.unwrap(),
        &problem,
        &client,
        out.handle,
        &out.shape,
        &out.strides,
    );
}

fn tensor_raw_parts<E: CubeElement + Numeric + Sample>(
    client: &ComputeClient<TestRuntime>,
    problem: &MatmulProblem,
    ident: MatmulIdent,
) -> TensorRawParts<E> {
    match ident {
        MatmulIdent::Lhs => {
            let mut shape = problem.shape(ident);

            let handle = E::sample(client, &shape, 1234);

            let data = client.read_one_tensor(handle.as_copy_descriptor());
            let data = E::from_bytes(&data);
            let original_data = data.to_owned();

            let rank = shape.len();

            let data = match problem.lhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    shape.swap(rank - 1, rank - 2);
                    transpose::<E>(&original_data, problem.num_batches(), problem.m, problem.k)
                }
            };

            let Allocation {
                handle,
                mut strides,
            } = client.create_tensor_from_slice(
                E::as_bytes(&data),
                &shape,
                E::type_size() as usize,
            );

            if matches!(problem.lhs_layout, MatrixLayout::ColMajor) {
                shape.swap(rank - 1, rank - 2);
                strides.swap(rank - 1, rank - 2);
            }

            TensorRawParts {
                handle,
                scale: None,
                shape,
                strides,
                original_data: Some(original_data),
            }
        }
        MatmulIdent::Rhs => {
            let mut shape = problem.shape(ident);

            let handle = E::sample(client, &shape, 5678);

            let data = client.read_one_tensor(handle.as_copy_descriptor());
            let data = E::from_bytes(&data);
            let original_data = data.to_owned();

            let rank = shape.len();

            let data = match problem.rhs_layout {
                MatrixLayout::RowMajor => original_data.clone(),
                MatrixLayout::ColMajor => {
                    shape.swap(rank - 1, rank - 2);
                    transpose::<E>(&original_data, problem.num_batches(), problem.k, problem.n)
                }
            };

            let Allocation {
                handle,
                mut strides,
            } = client.create_tensor_from_slice(
                E::as_bytes(&data),
                &shape,
                E::type_size() as usize,
            );

            if matches!(problem.rhs_layout, MatrixLayout::ColMajor) {
                shape.swap(rank - 1, rank - 2);
                strides.swap(rank - 1, rank - 2);
            }

            TensorRawParts {
                handle,
                scale: None,
                shape,
                strides,
                original_data: Some(original_data),
            }
        }
        MatmulIdent::Out => {
            let zero = E::from_int(0);

            let data = vec![zero; tensor_size(problem, MatmulIdent::Out)];

            let shape = problem.shape(MatmulIdent::Out);
            let Allocation { handle, strides } = client.create_tensor_from_slice(
                E::as_bytes(&data),
                &shape,
                E::type_size() as usize,
            );
            TensorRawParts {
                handle,
                scale: None,
                shape,
                strides,
                original_data: None,
            }
        }
    }
}
