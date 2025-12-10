use crate::ReduceError;
use cubecl::{features::Plane, prelude::*};

// #[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
// pub struct ReduceStrategyLegacy {
//     /// If true and the compute client support plane instructions,
//     /// then try using them in the kernel. It could still be impossible to use
//     /// plane instructions depending on the memory layout of the tensors.
//     pub use_planes: bool,
//
//     /// If true, all units within a single cube cooperate to reduce a single item in the output.
//     /// Else, each unit or plane (if planes is true) reduce a single item by itself.
//     pub shared: bool,
// }

#[derive(Debug, Clone, Copy)]
pub enum ReduceStrategy {
    /// A unit is responsable to reduce a full vector.
    FullUnit,
    /// A plane is responsable to reduce a full vector.
    FullPlane {
        /// How the reduce is done by the plane.
        level: PlaneReduceLevel,
    },
    /// A cube is responsable to reduce a full vector.
    FullCube {
        /// How the reduce is done by the plane.
        use_planes: bool,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum PlaneReduceLevel {
    Unit,
    Plane,
}

impl ReduceStrategy {
    pub fn validate<R: Runtime>(self, client: &ComputeClient<R>) -> Result<Self, ReduceError> {
        let use_planes = match &self {
            ReduceStrategy::FullUnit => false,
            ReduceStrategy::FullPlane { level } => true,
            ReduceStrategy::FullCube { use_planes } => *use_planes,
        };

        if use_planes {
            if !support_plane(client) {
                return Err(ReduceError::PlanesUnavailable);
            }
            if !precise_plane_dim(client) {
                return Err(ReduceError::ImprecisePlaneDim);
            }
        }

        Ok(self)
    }
}

fn support_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().features.plane.contains(Plane::Ops)
}

fn precise_plane_dim<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let hw_props = &client.properties().hardware;
    hw_props.plane_size_min == hw_props.plane_size_max
}
