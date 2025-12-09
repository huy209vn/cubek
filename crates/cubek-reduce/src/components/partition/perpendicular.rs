use crate::components::{
    partition::{PartitionOption, PartitionSplit, ReducePartition},
    precision::ReducePrecision,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[cube]
pub fn partition_perpendicular<P: ReducePrecision, Out: Numeric>(
    reduce_index: u32,
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    axis_reduce: u32,
    #[comptime] input_line_size: u32,
    #[comptime] config: PartitionOption,
) -> ReducePartition {
    let shape_axis = input.shape(axis_reduce);

    let mut index_start = 0;
    for axis in 0..input.rank() {
        let coordinate = output.coordinate(reduce_index * input_line_size, axis);
        index_start += coordinate * input.stride(axis);
    }
    index_start /= input_line_size;

    let index_step = input.stride(axis_reduce) / input_line_size;

    let coordinate_end = shape_axis;

    let coordinate_step = match comptime!(config.split) {
        PartitionSplit::Unit => 1u32.runtime(),
        PartitionSplit::Plane => CUBE_DIM_X,
        PartitionSplit::Cube => CUBE_DIM,
    };

    ReducePartition {
        index_start,
        index_step,
        coordinate_start: 0,
        coordinate_step,
        coordinate_end,
    }
}
