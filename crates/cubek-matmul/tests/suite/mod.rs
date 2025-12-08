#![allow(missing_docs)]

pub type TestEG = half::f16;
pub type TestES = half::f16;
pub type TestEA = f32;
// TestEG: Float + CubeElement + Display + CastInto<TestES> + Sample + MatmulPrecision,
// TestES: Numeric + CastInto<TestEA>,
// TestEA: Numeric + CastInto<TestEG>,

pub mod layered;
pub mod naive;
pub mod test_utils;
pub mod test_utils_2;

mod unit {
    crate::testgen_matmul_unit!();
}
mod tma {
    crate::testgen_matmul_tma!();
}
mod plane_vecmat {
    crate::testgen_matmul_plane_vecmat!();
}

mod plane_accelerated {
    crate::testgen_matmul_plane_accelerated!();
}
