use crate::{
    ReduceInstruction, ReducePrecision,
    components::{
        level::{
            ReduceJob,
            unit::{UnitReduce, UnitReduceConfig},
        },
        partition::{PartitionOption, PartitionSplit, ReducePartition},
        writer,
    },
    routines::ReduceBlueprint,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub struct GlobalFullUnitReduce;

#[cube]
impl GlobalFullUnitReduce {
    pub fn execute<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        reduce_index: u32,
        inst: &I,
        #[comptime] blueprint: ReduceBlueprint,
    ) {
        let input_line_size = input.line_size();
        let config = comptime!(UnitReduceConfig::new(
            input.line_size(),
            blueprint.line_mode
        ));
        let partition = GlobalFullUnitReduce::partition::<P, Out>(
            reduce_index,
            input,
            output,
            axis_reduce,
            comptime!(config.clone()),
        );
        let mut accumulator = I::null_accumulator(inst, input_line_size);

        <UnitReduce as ReduceJob<P, I>>::execute(
            input,
            inst,
            partition,
            &mut accumulator,
            comptime!(config.clone()),
        );
        writer::write::<P, Out, I>(
            output,
            accumulator,
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
        #[comptime] config: UnitReduceConfig,
    ) -> ReducePartition {
        let line_mode = config.line_mode;

        ReducePartition::new::<P, Out>(
            reduce_index,
            input,
            output,
            axis_reduce,
            comptime!(PartitionOption {
                split: PartitionSplit::Unit,
                line_mode,
            }),
        )
    }
}
