use crate::{
    LineMode, ReduceInstruction, ReducePrecision,
    components::{
        instructions::reduce_inplace,
        level::{self, plane::PlaneReduceConfig},
        partition::{PartitionOption, PartitionSplit, ReducePartition},
        writer,
    },
    routines::ReduceBlueprint,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub struct GlobalFullPlaneReduce;

#[cube]
impl GlobalFullPlaneReduce {
    pub fn execute<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        reduce_index: u32,
        inst: &I,
        #[comptime] blueprint: ReduceBlueprint,
    ) {
        let line_mode = blueprint.line_mode;
        let plane_blueprint = comptime!(match blueprint.global {
            crate::routines::GlobalReduceBlueprint::FullPlane(b) => b.clone(),
            _ => panic!(),
        });
        let input_line_size = input.line_size();
        let config = comptime!(PlaneReduceConfig::new(
            input_line_size,
            line_mode,
            plane_blueprint,
        ));
        let partition = GlobalFullPlaneReduce::partition::<P, Out>(
            reduce_index,
            input,
            output,
            axis_reduce,
            line_mode,
        );

        let accumulator =
            level::plane::reduce::<P, VirtualTensor<P::EI>, I>(input, inst, partition, config);

        let result = match plane_blueprint.independant {
            true => {
                let (item, coordinate) = I::read_accumulator(inst, &accumulator);
                let mut result = I::null_accumulator(inst, input_line_size);
                reduce_inplace::<P, I>(inst, &mut result, item, coordinate, true);
                result
            }
            false => accumulator,
        };

        writer::write::<P, Out, I>(
            output,
            result,
            reduce_index,
            input.shape(axis_reduce),
            blueprint,
            input.line_size(),
            inst,
        )
    }

    fn partition<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] line_mode: LineMode,
    ) -> ReducePartition {
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
