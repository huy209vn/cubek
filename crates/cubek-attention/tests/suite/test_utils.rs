#![allow(clippy::needless_range_loop)]

use core::f32;
use std::fmt::Display;

use cubecl::{
    CubeElement, Runtime,
    client::ComputeClient,
    flex32,
    prelude::{CubePrimitive, Exp, Float, Numeric},
    server::{self},
    tf32,
};

use cubecl::std::tensor::TensorHandle;
use cubek_attention::components::{
    AttentionElems, AttentionIdent, AttentionPrecision, AttentionProblem, AttentionStorageTypes
};

use crate::suite::{attention_test_launcher::strides, cpu_reference::flash_attention_v2_cpu};

pub fn assert_result<R: Runtime>(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: Option<&[bool]>,
    problem: &AttentionProblem,
    client: &ComputeClient<R>,
    out: server::Handle,
    shape: &[usize],
    strides: &[usize],
) {
    let epsilon = 1e-2;
    let expected = flash_attention_v2_cpu(query, key, value, mask, problem);

    if let Err(e) = assert_equals_approx::<R>(client, out, shape, strides, &expected, epsilon) {
        panic!("{}", e);
    }
}

fn attention_epsilon(elems: &AttentionElems, safety_factor: f32) -> f32 {
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

// /// Compares the content of a handle to a given slice of f32.
// pub(crate) fn assert_equals_approx<R: Runtime, F: Float + CubeElement + Display>(
//     client: &ComputeClient<R>,
//     output: server::Handle,
//     shape: &[usize],
//     strides: &[usize],
//     expected: &[F],
//     epsilon: f32,
// ) -> Result<(), String> {
//     let env = std::env::var("CUBEK_TEST_MODE");

//     let print_instead_of_compare = match env {
//         Ok(val) => matches!(val.as_str(), "print"),
//         Err(_) => false,
//     };

//     let actual =
//         client.read_one_tensor(output.copy_descriptor(shape, strides, F::type_size() as usize));
//     let actual = F::from_bytes(&actual);

//     let epsilon = epsilon.max(F::EPSILON.to_f32().unwrap());

//     for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
//         let allowed_error = (epsilon * e.to_f32().unwrap()).max(epsilon);

//         // account for lower precision at higher values
//         if print_instead_of_compare {
//             println!("{:?}: {:?}, {:?}", i, a, e);
//         } else {
//             let actual_nan = f32::is_nan(a.to_f32().unwrap());
//             let expected_nan = f32::is_nan(e.to_f32().unwrap());

//             if actual_nan != expected_nan {
//                 if expected_nan {
//                     return Err(format!("Expected NaN, got value={:?}", *a));
//                 } else {
//                     return Err(format!("Expected value={:?}, got NaN", *e));
//                 }
//             }

//             let difference = f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap());

//             if difference >= allowed_error {
//                 return Err(format!(
//                     "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
//                     i, *a, *e, difference, epsilon
//                 ));
//             }
//         }
//     }

//     if print_instead_of_compare {
//         Err("".to_string())
//     } else {
//         Ok(())
//     }
// }

// pub trait CastInto<E> {
//     fn cast_into(self) -> E;
// }

// impl<E> CastInto<E> for E {
//     fn cast_into(self) -> E {
//         self
//     }
// }

// impl CastInto<f32> for half::f16 {
//     fn cast_into(self) -> f32 {
//         f32::from(self)
//     }
// }

// impl CastInto<f32> for half::bf16 {
//     fn cast_into(self) -> f32 {
//         f32::from(self)
//     }
// }

// impl CastInto<f32> for flex32 {
//     fn cast_into(self) -> f32 {
//         f32::from(self)
//     }
// }

// impl CastInto<half::bf16> for f32 {
//     fn cast_into(self) -> half::bf16 {
//         half::bf16::from_f32(self)
//     }
// }

// impl CastInto<half::bf16> for half::f16 {
//     fn cast_into(self) -> half::bf16 {
//         half::bf16::from_f32(self.to_f32())
//     }
// }

// impl CastInto<half::f16> for half::bf16 {
//     fn cast_into(self) -> half::f16 {
//         half::f16::from_f32(self.to_f32())
//     }
// }

// impl CastInto<half::f16> for f32 {
//     fn cast_into(self) -> half::f16 {
//         half::f16::from_f32(self)
//     }
// }

// impl CastInto<half::f16> for flex32 {
//     fn cast_into(self) -> half::f16 {
//         half::f16::from_f32(self.to_f32())
//     }
// }

// impl CastInto<half::bf16> for flex32 {
//     fn cast_into(self) -> half::bf16 {
//         half::bf16::from_f32(self.to_f32())
//     }
// }

// impl CastInto<flex32> for f32 {
//     fn cast_into(self) -> flex32 {
//         flex32::from_f32(self)
//     }
// }

// impl CastInto<f32> for tf32 {
//     fn cast_into(self) -> f32 {
//         self.to_f32()
//     }
// }

// impl CastInto<tf32> for f32 {
//     fn cast_into(self) -> tf32 {
//         tf32::from_f32(self)
//     }
// }

// impl CastInto<u16> for u8 {
//     fn cast_into(self) -> u16 {
//         self as u16
//     }
// }

// impl CastInto<i32> for u16 {
//     fn cast_into(self) -> i32 {
//         self as i32
//     }
// }

// impl CastInto<u8> for i32 {
//     fn cast_into(self) -> u8 {
//         self as u8
//     }
// }

// pub trait Sampleable: Sized + CubePrimitive {
//     fn sample<R: Runtime>(client: &ComputeClient<R>, shape: &[usize], seed: u64)
//     -> TensorHandle<R>;
// }

// macro_rules! sample_float {
//     ($($t:ty),*) => {
//         $(
//             impl Sampleable for $t
//             {
//                 fn sample<R: Runtime>(client: &ComputeClient<R>, shape: &[usize], seed: u64) -> TensorHandle<R> {
//                     cubek_random::seed(seed);
//                     let dtype = Self::as_type_native_unchecked();
//                     let output = TensorHandle::empty(client, shape.to_vec(), dtype);

//                     cubek_random::random_uniform(&client, f32::from_int(-1), f32::from_int(1), output.as_ref(), dtype).unwrap();

//                     output
//                 }
//             }
//         )*
//     };
// }

// sample_float!(half::f16);
// sample_float!(half::bf16);
// sample_float!(f32);
// sample_float!(f64);

// impl Sampleable for flex32 {
//     fn sample<R: Runtime>(
//         client: &ComputeClient<R>,
//         shape: &[usize],
//         seed: u64,
//     ) -> TensorHandle<R> {
//         cubek_random::seed(seed);
//         let dtype = f32::as_type_native_unchecked();
//         let output = TensorHandle::empty(client, shape.to_vec(), dtype);

//         cubek_random::random_uniform(
//             client,
//             f32::from_int(-1),
//             f32::from_int(1),
//             output.as_ref(),
//             dtype,
//         )
//         .unwrap();

//         output
//     }
// }

// impl Sampleable for tf32 {
//     fn sample<R: Runtime>(
//         client: &ComputeClient<R>,
//         shape: &[usize],
//         seed: u64,
//     ) -> TensorHandle<R> {
//         cubek_random::seed(seed);
//         let dtype = f32::as_type_native_unchecked();
//         let output = TensorHandle::empty(client, shape.to_vec(), dtype);

//         cubek_random::random_uniform(
//             client,
//             f32::from_int(-1),
//             f32::from_int(1),
//             output.as_ref(),
//             dtype,
//         )
//         .unwrap();

//         output
//     }
// }

// impl Sampleable for bool {
//     fn sample<R: Runtime>(
//         client: &ComputeClient<R>,
//         shape: &[usize],
//         seed: u64,
//     ) -> TensorHandle<R> {
//         cubek_random::seed(seed);
//         let dtype = bool::as_type_native_unchecked();
//         let output = TensorHandle::empty(client, shape.to_vec(), dtype);

//         cubek_random::random_bernoulli(client, 0.5, output.as_ref(), dtype).unwrap();

//         output
//     }
// }

// impl Sampleable for u8 {
//     fn sample<R: Runtime>(
//         client: &ComputeClient<R>,
//         shape: &[usize],
//         seed: u64,
//     ) -> TensorHandle<R> {
//         cubek_random::seed(seed);
//         let dtype = u8::as_type_native_unchecked();
//         let output = TensorHandle::empty(client, shape.to_vec(), dtype);

//         cubek_random::random_bernoulli(client, 0.5, output.as_ref(), dtype).unwrap();

//         output
//     }
// }
