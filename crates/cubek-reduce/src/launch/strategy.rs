use crate::ReduceError;
use cubecl::{features::Plane, prelude::*};

#[derive(Debug, Clone, Copy)]
pub enum ReduceStrategy {
    /// A unit is responsable to reduce a full vector.
    FullUnit,
    /// A plane is responsable to reduce a full vector.
    FullPlane {
        /// How the accumulators are handled in a plane.
        independant: bool,
    },
    /// A cube is responsable to reduce a full vector.
    FullCube {
        /// How the reduce is done by the plane.
        use_planes: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneReduceLevel {
    Unit,
    Plane,
}

impl ReduceStrategy {
    pub fn validate<R: Runtime>(self, client: &ComputeClient<R>) -> Result<Self, ReduceError> {
        let use_planes = match &self {
            ReduceStrategy::FullUnit => false,
            ReduceStrategy::FullPlane { .. } => true,
            ReduceStrategy::FullCube { use_planes } => *use_planes,
        };

        if use_planes {
            if !support_plane(client) {
                return Err(ReduceError::PlanesUnavailable);
            }
        }

        Ok(self)
    }
}

fn support_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().features.plane.contains(Plane::Ops)
}
