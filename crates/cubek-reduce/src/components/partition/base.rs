use crate::{
    LineMode,
    components::{
        partition::{parallel::partition_parallel, perpendicular::partition_perpendicular},
        precision::ReducePrecision,
    },
    routines::{ReduceBlueprint, ReduceBlueprintKind},
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

/// A simple range to specify how to iterate a slice when performing a reduction.
#[derive(CubeType)]
pub struct ReducePartition {
    pub index_start: u32,
    pub index_step: u32,
    pub coordinate_start: u32,
    pub coordinate_end: u32,
    pub coordinate_step: u32,
}

#[derive(Clone)]
pub struct PartitionOption {
    pub split: PartitionSplit,
    pub line_mode: LineMode,
}

#[derive(Clone)]
pub enum PartitionSplit {
    /// The axis is not splitted and is executed entirely by a single unit.
    Unit,
    /// The axis is splitted across a plane.
    Plane,
    /// The axis is splitted across a cube.
    Cube,
}

impl PartitionOption {
    fn new(blueprint: ReduceBlueprint) -> PartitionOption {
        match blueprint.kind {
            ReduceBlueprintKind::Unit => PartitionOption {
                split: PartitionSplit::Unit,
                line_mode: blueprint.line_mode,
            },
            ReduceBlueprintKind::Plane(..) => PartitionOption {
                split: PartitionSplit::Plane,
                line_mode: blueprint.line_mode,
            },
            ReduceBlueprintKind::Cube(..) => PartitionOption {
                split: PartitionSplit::Cube,
                line_mode: blueprint.line_mode,
            },
        }
    }
}

#[cube]
impl ReducePartition {
    pub(crate) fn new<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] config: PartitionOption,
    ) -> ReducePartition {
        match comptime!(config.line_mode) {
            LineMode::Parallel => partition_parallel::<P, Out>(
                reduce_index,
                input,
                output,
                axis_reduce,
                input.line_size(),
                config,
            ),
            LineMode::Perpendicular => partition_perpendicular::<P, Out>(
                reduce_index,
                input,
                output,
                axis_reduce,
                input.line_size(),
                config,
            ),
        }
    }

    pub(crate) fn from_blueprint<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] blueprint: ReduceBlueprint,
    ) -> ReducePartition {
        let config = comptime!(PartitionOption::new(blueprint));
        ReducePartition::new::<P, Out>(reduce_index, input, output, axis_reduce, config)
    }
}
