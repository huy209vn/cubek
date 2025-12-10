use cubecl::TestRuntime;
use cubecl::std::tensor::TensorHandle;
use cubecl::{CubeElement, client::ComputeClient};

use crate::suite::test_utils::cpu_reference::matmul_cpu_reference;
use crate::suite::test_utils::new_casted;
use cubek_matmul::components::{MatmulElems, MatmulProblem};

pub fn assert_result(
    lhs: &[f32],
    rhs: &[f32],
    problem: &MatmulProblem,
    client: &ComputeClient<TestRuntime>,
    out: &TensorHandle<TestRuntime>,
    dtypes: MatmulElems,
) {
    let epsilon = matmul_epsilon(&dtypes, 170.);
    let expected = matmul_cpu_reference(lhs, rhs, problem);

    if let Err(e) = assert_equals_approx(client, out, &expected, epsilon) {
        panic!("{}", e);
    }
}

fn matmul_epsilon(elems: &MatmulElems, safety_factor: f32) -> f32 {
    let total_eps = elems.lhs_global.dtype.epsilon()
        + elems.rhs_global.dtype.epsilon()
        + elems.acc_global.dtype.epsilon()
        + elems.lhs_stage.dtype.epsilon()
        + elems.rhs_stage.dtype.epsilon()
        + elems.acc_stage.dtype.epsilon()
        + elems.lhs_register.dtype.epsilon()
        + elems.rhs_register.dtype.epsilon()
        + elems.acc_register.dtype.epsilon();

    total_eps as f32 * safety_factor
}

/// Compares the content of a handle to a given slice of f32.
fn assert_equals_approx(
    client: &ComputeClient<TestRuntime>,
    out: &TensorHandle<TestRuntime>,
    expected: &[f32],
    epsilon: f32,
) -> Result<(), String> {
    let env = std::env::var("CUBEK_TEST_MODE");

    let print_instead_of_compare = match env {
        Ok(val) => matches!(val.as_str(), "print"),
        Err(_) => false,
    };

    // Obtain the data in f32 for not being generic over type
    let data_handle = new_casted(client, &out);
    let data_f32 =
        f32::from_bytes(&client.read_one_tensor(data_handle.as_copy_descriptor())).to_owned();

    for (i, (a, e)) in data_f32.iter().zip(expected.iter()).enumerate() {
        let allowed_error = (epsilon * e).max(epsilon);

        if print_instead_of_compare {
            println!("{:?}: {:?}, {:?}", i, a, e);
        } else {
            let actual_nan = f32::is_nan(*a);
            let expected_nan = f32::is_nan(*e);

            if actual_nan != expected_nan {
                if expected_nan {
                    return Err(format!("Expected NaN, got value={:?}", *a));
                } else {
                    return Err(format!("Expected value={:?}, got NaN", *e));
                }
            }

            let difference = f32::abs(a - e);

            if difference >= allowed_error {
                return Err(format!(
                    "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
                    i, *a, *e, difference, epsilon
                ));
            }
        }
    }

    if print_instead_of_compare {
        Err("".to_string())
    } else {
        Ok(())
    }
}
