use cubecl::prelude::*;
use cubecl::{
    TestRuntime,
    server::{self},
};

use crate::suite::TestEG;
use crate::suite::{test_utils::assert_result, test_utils_2::tensor_raw_parts};
use cubek_matmul::components::{
    MatmulElems,
    global::args::{ConcreteOutputFactory, TensorArgs, TensorOutput},
};
use cubek_matmul::components::{MatmulProblem, MatmulSelection};
use cubek_matmul::components::{MatrixLayout, global::args::ConcreteInputsFactory};
use cubek_matmul::components::{
    batch::{BatchConfig, BatchMatmulFamily},
    global::args::TensorInputs,
};
use cubek_matmul::kernels::layered::Algorithm;
use cubek_matmul::{
    MatmulInputHandleRef,
    components::{AvailableLineSizes, MatmulIdent},
};

#[derive(Debug)]
pub struct TensorRawParts<E: CubeElement> {
    pub handle: server::Handle,
    #[allow(unused)] //TODO: Fix
    pub scale: Option<server::Handle>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub original_data: Option<Vec<E>>,
}

/// Test the correctness of the specified Matmul on the given device,
/// against a naive CPU implementation over the given problem
pub fn test_matmul_algorithm<A: Algorithm>(
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
    let lhs = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Lhs);
    let rhs = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Rhs);
    let out = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Out);

    problem.lhs_strides = lhs.strides.clone();
    problem.rhs_strides = rhs.strides.clone();

    let line_sizes = AvailableLineSizes::from_type_sizes(
        &client,
        dtypes.lhs_global.size(),
        dtypes.rhs_global.size(),
        dtypes.acc_global.size(),
    );
    let line_sizes = A::filter_line_sizes(line_sizes);
    let line_sizes = line_sizes
        .filter_lhs_with_tensor(&lhs.strides, &lhs.shape, problem.lhs_layout)
        .filter_rhs_with_tensor(&rhs.strides, &rhs.shape, problem.rhs_layout)
        .filter_out_with_tensor(&out.strides, &out.shape)
        .pick_max()
        .unwrap();

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

    let props = &client.properties().hardware;
    if !props.max_cube_dim.can_contain(config.cube_dim())
        || config.cube_dim().num_elems() > props.max_units_per_cube
    {
        println!("Skipping test, too many resources requested");
        return;
    }

    let cube_count_plan = config.hypercube_config().cube_count_plan(
        &problem,
        client.properties().hardware.max_cube_count.clone(),
    );

    // let elem_size = ;
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

    let result = unsafe {
        A::BatchMatmul::launch_unchecked::<TensorArgs, TestRuntime>(
            &client,
            config.cube_dim(),
            cube_count_plan.resolve(),
            TensorInputs::create(
                &client,
                &lhs_handle,
                &rhs_handle,
                &selection,
                &problem,
                &line_sizes,
                config,
                &dtypes,
            ),
            TensorOutput::create(
                &client,
                &out_handle,
                &selection,
                &problem,
                &line_sizes,
                config,
                &dtypes,
            ),
            cube_count_plan.as_args(),
            config,
            &dtypes,
        )
    };

    match result {
        Ok(_) => {}
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

pub(crate) fn transpose<E: Copy>(array: &[E], batches: usize, rows: usize, cols: usize) -> Vec<E> {
    let mut result = vec![array[0]; array.len()];
    for b in 0..batches {
        for i in 0..rows {
            for j in 0..cols {
                result[(b * rows * cols) + j * rows + i] = array[(b * rows * cols) + i * cols + j];
            }
        }
    }
    result
}

/// Returns the total number of elements for the identified tensor, inferred by the problem definition
pub(crate) fn tensor_size(problem: &MatmulProblem, ident: MatmulIdent) -> usize {
    match ident {
        MatmulIdent::Lhs => problem.num_batches() * problem.m * problem.k,
        MatmulIdent::Rhs => problem.num_batches() * problem.k * problem.n,
        MatmulIdent::Out => problem.num_batches() * problem.m * problem.n,
    }
}

/// Returns the stride of the identified tensor, inferred by the problem definition
pub(crate) fn strides(problem: &MatmulProblem, ident: MatmulIdent) -> Vec<usize> {
    let shape = problem.shape(ident);
    let rank = shape.len();
    let mut strides = Vec::with_capacity(rank);

    let (last_batch, x, y) = match ident {
        MatmulIdent::Lhs => match problem.lhs_layout {
            MatrixLayout::RowMajor => (problem.m * problem.k, problem.k, 1),
            MatrixLayout::ColMajor => (problem.m * problem.k, 1, problem.m),
        },
        MatmulIdent::Rhs => match problem.rhs_layout {
            MatrixLayout::RowMajor => (problem.k * problem.n, problem.n, 1),
            MatrixLayout::ColMajor => (problem.k * problem.n, 1, problem.k),
        },
        MatmulIdent::Out => (problem.m * problem.n, problem.n, 1),
    };

    strides.push(y);
    strides.push(x);

    if rank > 2 {
        strides.push(last_batch);

        for b in shape.iter().rev().take(rank - 3) {
            strides.push(last_batch * b)
        }
    }

    strides.into_iter().rev().collect()
}
