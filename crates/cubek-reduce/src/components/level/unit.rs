use crate::{
    LineMode, ReduceInstruction, ReducePrecision,
    components::{
        instructions::{ReduceCoordinate, reduce_inplace},
        level::ReduceJob,
        partition::{PartitionOption, PartitionSplit, ReducePartition},
    },
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(Clone)]
pub struct UnitReduceConfig {
    pub line_size: u32,
    pub line_mode: LineMode,
}

impl UnitReduceConfig {
    pub fn new(input_line_size: u32, line_mode: LineMode) -> Self {
        Self {
            line_size: input_line_size,
            line_mode,
        }
    }
}

#[derive(CubeType)]
pub struct UnitReduce;

#[cube]
impl UnitReduce {
    pub fn plane_partitioning<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] config: UnitReduceConfig,
    ) -> ReducePartition {
        let line_mode = config.line_mode;

        ReducePartition::new::<P, Out>(
            reduce_index,
            input,
            output,
            axis_reduce,
            comptime!(PartitionOption {
                split: PartitionSplit::Plane,
                line_mode,
            }),
        )
    }
}

#[cube]
impl<P: ReducePrecision, I: ReduceInstruction<P>> ReduceJob<P, I> for UnitReduce {
    type Config = UnitReduceConfig;

    fn execute(
        input: &VirtualTensor<P::EI>,
        inst: &I,
        partition: ReducePartition,
        accumulator: &mut I::AccumulatorItem,
        #[comptime] config: Self::Config,
    ) {
        let mut index = partition.index_start;
        let requirements = I::requirements(inst);

        for coordinate in range_stepped(
            partition.coordinate_start,
            partition.coordinate_end,
            partition.coordinate_step,
        ) {
            let coordinates =
                ReduceCoordinate::new(coordinate, requirements, config.line_size, config.line_mode);

            reduce_inplace::<P, I>(inst, accumulator, input.read(index), coordinates, false);
            index += partition.index_step;
        }
    }
}
