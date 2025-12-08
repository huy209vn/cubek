use crate::suite::TestEG;
use crate::suite::test_utils::assert_result;
use crate::suite::test_utils_2::tensor_raw_parts;
use cubecl::frontend::CubePrimitive;
use cubecl::prelude::TensorHandleRef;
use cubecl::{Runtime, client::ComputeClient};
use cubek_matmul::MatmulInputHandleRef;
use cubek_matmul::components::batch::BatchConfig;
use cubek_matmul::components::batch::BatchMatmulFamily;
use cubek_matmul::components::global::args::ConcreteInputsFactory;
use cubek_matmul::components::global::args::ConcreteOutputFactory;
use cubek_matmul::components::global::args::{TensorArgs, TensorInputs, TensorOutput};
use cubek_matmul::components::{
    AvailableLineSizes, MatmulIdent, MatmulProblem, MatmulSelection, MatrixLayout,
};
use cubek_matmul::kernels::layered::Algorithm;
use cubek_matmul::{components::MatmulElems, kernels::naive};

type TestRuntime = cubecl::TestRuntime;

struct MatmulTestCase {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub batch: usize,
}

impl MatmulTestCase {
    fn to_problem(self) -> MatmulProblem {
        MatmulProblem {
            m: self.m,
            n: self.n,
            k: self.k,
            lhs_batches: vec![self.batch],
            rhs_batches: vec![self.batch],
            out_batches: vec![self.batch],
            lhs_strides: vec![self.m * self.k, self.k],
            rhs_strides: vec![self.k * self.n, self.n],
            lhs_layout: MatrixLayout::RowMajor,
            rhs_layout: MatrixLayout::RowMajor,
        }
    }
}

#[test]
pub fn test_small() {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 1,
    };

    test_naive(case);
}

#[test]
pub fn test_odd() {
    let case = MatmulTestCase {
        m: 1,
        k: 101,
        n: 255,
        batch: 1,
    };

    test_naive(case);
}

#[test]
pub fn test_large() {
    let case = MatmulTestCase {
        m: 256,
        k: 256,
        n: 256,
        batch: 1,
    };

    test_naive(case);
}

#[test]
pub fn test_with_check_bounds() {
    let case = MatmulTestCase {
        m: 60,
        k: 60,
        n: 60,
        batch: 1,
    };

    test_naive(case);
}

#[test]
pub fn test_with_batches() {
    let case = MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 3,
    };

    test_naive(case);
}

fn test_naive(case: MatmulTestCase) {
    let client = TestRuntime::client(&Default::default());
    let problem = case.to_problem();

    let lhs = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Lhs);
    let rhs = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Rhs);
    let out = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Out);

    // let lhs = case.random_lhs::<TestRuntime, FloatT>(&client);
    // let rhs = case.random_rhs::<TestRuntime, FloatT>(&client);

    // let expected = case.matmul_cpu::<TestRuntime, T>(&lhs, &rhs, &client);
    // let expected = matmul_cpu_reference(lhs, rhs, out);

    // let out: TensorHandle<TestRuntime> = case.empty_out::<TestRuntime, T>(&client);

    let lhs_handle = MatmulInputHandleRef::Normal(
        unsafe {
            TensorHandleRef::from_raw_parts(
                &lhs.handle,
                &lhs.strides,
                &lhs.shape,
                TestEG::type_size() as usize,
            )
        },
        TestEG::as_type_native_unchecked(),
    );
    let rhs_handle = MatmulInputHandleRef::Normal(
        unsafe {
            TensorHandleRef::from_raw_parts(
                &rhs.handle,
                &rhs.strides,
                &rhs.shape,
                TestEG::type_size() as usize,
            )
        },
        TestEG::as_type_native_unchecked(),
    );
    let out_handle = unsafe {
        TensorHandleRef::from_raw_parts(
            &out.handle,
            &out.strides,
            &out.shape,
            TestEG::type_size() as usize,
        )
    };

    naive::launch_ref(
        &client,
        &lhs_handle,
        &rhs_handle,
        &out_handle,
        &MatmulElems::new::<TestEG>(),
    )
    .unwrap();

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

// /// Test the correctness of the specified Matmul on the given device,
// /// against a naive CPU implementation over the given problem
// pub fn test_matmul_algorithm<A: Algorithm>(
//     client: ComputeClient<TestRuntime>,
//     mut problem: MatmulProblem,
//     selection: MatmulSelection,
//     dtypes: MatmulElems,
// ) {
//     let env = std::env::var("MATMUL_TEST_MODE");

//     let panic_on_launch_err = match env {
//         Ok(val) => match val.as_str() {
//             "panic" => true,
//             "skip" => false,
//             _ => false,
//         },
//         Err(_) => false,
//     };
//     let lhs = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Lhs);
//     let rhs = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Rhs);
//     let out = tensor_raw_parts::<TestEG>(&client, &problem, MatmulIdent::Out);

//     problem.lhs_strides = lhs.strides.clone();
//     problem.rhs_strides = rhs.strides.clone();

//     let line_sizes = AvailableLineSizes::from_type_sizes(
//         &client,
//         dtypes.lhs_global.size(),
//         dtypes.rhs_global.size(),
//         dtypes.acc_global.size(),
//     );
//     let line_sizes = A::filter_line_sizes(line_sizes);
//     let line_sizes = line_sizes
//         .filter_lhs_with_tensor(&lhs.strides, &lhs.shape, problem.lhs_layout)
//         .filter_rhs_with_tensor(&rhs.strides, &rhs.shape, problem.rhs_layout)
//         .filter_out_with_tensor(&out.strides, &out.shape)
//         .pick_max()
//         .unwrap();

//     // let dtypes = MatmulElems::new_with_tile::<P::MP, A::TileMatmul>();

//     let config = match A::setup(&client, &problem, &selection, &line_sizes, &dtypes) {
//         Ok(config) => config,
//         Err(err) => {
//             let msg = format!("Can't launch the test: {err}");
//             if panic_on_launch_err {
//                 panic!("{msg}");
//             } else {
//                 println!("{msg}");
//                 return;
//             }
//         }
//     };

//     let props = &client.properties().hardware;
//     if !props.max_cube_dim.can_contain(config.cube_dim())
//         || config.cube_dim().num_elems() > props.max_units_per_cube
//     {
//         println!("Skipping test, too many resources requested");
//         return;
//     }

//     let cube_count_plan = config.hypercube_config().cube_count_plan(
//         &problem,
//         client.properties().hardware.max_cube_count.clone(),
//     );

//     let lhs_handle = MatmulInputHandleRef::Normal(
//         unsafe {
//             TensorHandleRef::from_raw_parts(
//                 &lhs.handle,
//                 &lhs.strides,
//                 &lhs.shape,
//                 dtypes.lhs_global.size(),
//             )
//         },
//         *dtypes.lhs_global,
//     );
//     let rhs_handle = MatmulInputHandleRef::Normal(
//         unsafe {
//             TensorHandleRef::from_raw_parts(
//                 &rhs.handle,
//                 &rhs.strides,
//                 &rhs.shape,
//                 dtypes.rhs_global.size(),
//             )
//         },
//         *dtypes.rhs_global,
//     );
//     let out_handle = unsafe {
//         TensorHandleRef::from_raw_parts(
//             &out.handle,
//             &out.strides,
//             &out.shape,
//             dtypes.acc_global.size(),
//         )
//     };

//     let result = unsafe {
//         A::BatchMatmul::launch_unchecked::<TensorArgs, TestRuntime>(
//             &client,
//             config.cube_dim(),
//             cube_count_plan.resolve(),
//             TensorInputs::create(
//                 &client,
//                 &lhs_handle,
//                 &rhs_handle,
//                 &selection,
//                 &problem,
//                 &line_sizes,
//                 config,
//                 &dtypes,
//             ),
//             TensorOutput::create(
//                 &client,
//                 &out_handle,
//                 &selection,
//                 &problem,
//                 &line_sizes,
//                 config,
//                 &dtypes,
//             ),
//             cube_count_plan.as_args(),
//             config,
//             &dtypes,
//         )
//     };

//     match result {
//         Ok(_) => {}
//         Err(_err) => return,
//     }

//     assert_result(
//         &lhs.original_data.unwrap(),
//         &rhs.original_data.unwrap(),
//         &problem,
//         &client,
//         out.handle,
//         &out.shape,
//         &out.strides,
//     );
// }
