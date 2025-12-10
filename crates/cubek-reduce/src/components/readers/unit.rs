use crate::{
    LineMode, ReduceInstruction, ReducePrecision,
    components::instructions::{ReduceCoordinate, ReduceRequirements},
};
use cubecl::{
    prelude::*,
    std::tensor::{
        View,
        layout::{Coords1d, plain::PlainLayout},
        r#virtual::VirtualTensor,
    },
};

#[derive(CubeType)]
pub enum UnitReader<P: ReducePrecision> {
    Parallel(ParallelUnitReader<P>),
    Perpendicular(PerpendicularUnitReader<P>),
}

#[derive(CubeType)]
pub struct ParallelUnitReader<P: ReducePrecision> {
    view: View<Line<P::EI>, Coords1d>,
    /// The global offset that points where the vector to reduce is located in global memory.
    vector_offset: u32,
    requirements: ReduceRequirements,
    #[cube(comptime)]
    line_size: u32,
}

#[derive(CubeType)]
pub struct PerpendicularUnitReader<P: ReducePrecision> {
    view: View<Line<P::EI>, Coords1d>,
    /// The global offset that points where the vector to reduce is located in global memory.
    vector_offset: u32,
    vector_offset_stride: u32,
    requirements: ReduceRequirements,
    #[cube(comptime)]
    line_size: u32,
}

#[cube]
impl<P: ReducePrecision> UnitReader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_axis: u32,
        reduce_index: u32,
        #[comptime] line_mode: LineMode,
    ) -> UnitReader<P> {
        match line_mode {
            LineMode::Parallel => {
                UnitReader::<P>::new_Parallel(ParallelUnitReader::<P>::new::<I, Out>(
                    input,
                    output,
                    inst,
                    reduce_index,
                ))
            }
            LineMode::Perpendicular => {
                UnitReader::<P>::new_Perpendicular(PerpendicularUnitReader::<P>::new::<I, Out>(
                    input,
                    output,
                    inst,
                    reduce_axis,
                    reduce_index,
                ))
            }
        }
    }

    pub fn read(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        match self {
            UnitReader::Parallel(reader) => reader.read(line_index),
            UnitReader::Perpendicular(reader) => reader.read(line_index),
        }
    }
}

#[cube]
impl<P: ReducePrecision> ParallelUnitReader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_index: u32,
    ) -> ParallelUnitReader<P> {
        let line_size = input.line_size();

        let mut vector_offset = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index, axis);
            vector_offset += coordinate * input.stride(axis);
        }
        vector_offset /= line_size;

        let requirements = I::requirements(inst);

        ParallelUnitReader::<P> {
            view: input.view(PlainLayout::new(input.len())),
            vector_offset,
            requirements,
            line_size,
        }
    }

    pub fn read(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        let global_index = self.vector_offset + line_index;
        let item = self.view[global_index];

        let coordinate = match comptime!(self.requirements.coordinates) {
            true => {
                let first = line_index * self.line_size;
                let mut coordinates = Line::empty(self.line_size);
                #[unroll]
                for j in 0..self.line_size {
                    coordinates[j] = first + j;
                }
                ReduceCoordinate::new_Required(coordinates)
            }
            false => ReduceCoordinate::new_NotRequired(),
        };

        (item, coordinate)
    }
}

#[cube]
impl<P: ReducePrecision> PerpendicularUnitReader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_axis: u32,
        reduce_index: u32,
    ) -> PerpendicularUnitReader<P> {
        let line_size = input.line_size();
        let output_index = reduce_index * line_size;

        let mut vector_offset = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(output_index, axis);
            vector_offset += coordinate * input.stride(axis);
        }
        vector_offset /= line_size;

        let requirements = I::requirements(inst);
        let vector_offset_stride = input.stride(reduce_axis) / line_size;

        PerpendicularUnitReader::<P> {
            view: input.view(PlainLayout::new(input.len())),
            vector_offset,
            vector_offset_stride,
            requirements,
            line_size,
        }
    }

    pub fn read(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        let global_index = self.vector_offset + line_index * self.vector_offset_stride;
        let item = self.view[global_index];

        let coordinate = match comptime!(self.requirements.coordinates) {
            true => ReduceCoordinate::new_Required(Line::empty(self.line_size).fill(line_index)),
            false => ReduceCoordinate::new_NotRequired(),
        };

        (item, coordinate)
    }
}
