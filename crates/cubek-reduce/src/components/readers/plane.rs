use crate::{
    BoundChecksInner, LineMode, ReduceInstruction, ReducePrecision,
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
pub enum PlaneReader<P: ReducePrecision> {
    Parallel(ParallelPlaneReader<P>),
    Perpendicular(PerpendicularPlaneReader<P>),
}

#[derive(CubeType)]
pub struct ParallelPlaneReader<P: ReducePrecision> {
    view: View<Line<P::EI>, Coords1d>,
    /// The global offset that points where the vector to reduce is located in global memory.
    batch_offset: u32,
    requirements: ReduceRequirements,
    #[cube(comptime)]
    line_size: u32,
    bound_checks: ReaderBoundChecksInfo<P>,
    len: u32,
}

#[derive(CubeType)]
pub struct PerpendicularPlaneReader<P: ReducePrecision> {
    view: View<Line<P::EI>, Coords1d>,
    /// The global offset that points where the vector to reduce is located in global memory.
    batch_offset: u32,
    vector_offset_stride: u32,
    requirements: ReduceRequirements,
    #[cube(comptime)]
    line_size: u32,
    bound_checks: ReaderBoundChecksInfo<P>,
    len: u32,
}

#[derive(CubeType)]
enum ReaderBoundChecksInfo<P: ReducePrecision> {
    NotRequired,
    Required(ReaderBoundChecks<P>),
}

#[derive(CubeType)]
struct ReaderBoundChecks<P: ReducePrecision> {
    #[cube(comptime)]
    bound_checks: BoundChecksInner,
    coordinate_max: u32,
    null_input: Line<P::EI>,
}

#[cube]
impl<P: ReducePrecision> PlaneReader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_axis: u32,
        reduce_index: u32,
        #[comptime] bound_checks: BoundChecksInner,
        #[comptime] line_mode: LineMode,
    ) -> PlaneReader<P> {
        match line_mode {
            LineMode::Parallel => {
                PlaneReader::<P>::new_Parallel(ParallelPlaneReader::<P>::new::<I, Out>(
                    input,
                    output,
                    inst,
                    reduce_axis,
                    reduce_index,
                    bound_checks,
                ))
            }
            LineMode::Perpendicular => {
                PlaneReader::<P>::new_Perpendicular(PerpendicularPlaneReader::<P>::new::<I, Out>(
                    input,
                    output,
                    inst,
                    reduce_axis,
                    reduce_index,
                    bound_checks,
                ))
            }
        }
    }

    pub fn read(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        match self {
            PlaneReader::Parallel(reader) => reader.read(line_index),
            PlaneReader::Perpendicular(reader) => reader.read(line_index),
        }
    }

    pub fn len(&self) -> u32 {
        match self {
            PlaneReader::Parallel(reader) => reader.len,
            PlaneReader::Perpendicular(reader) => reader.len,
        }
    }
}

#[cube]
impl<P: ReducePrecision> ParallelPlaneReader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_axis: u32,
        reduce_index: u32,
        #[comptime] bound_checks: BoundChecksInner,
    ) -> ParallelPlaneReader<P> {
        let line_size = input.line_size();

        let mut batch_offset = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index, axis);
            batch_offset += coordinate * input.stride(axis);
        }
        batch_offset /= line_size;

        let requirements = I::requirements(inst);

        let shape = input.shape(reduce_axis);

        let num_chunks = shape / line_size;
        let len = num_chunks.div_ceil(CUBE_DIM_X);

        let bound_checks = match comptime!(bound_checks) {
            BoundChecksInner::None => ReaderBoundChecksInfo::new_NotRequired(),
            BoundChecksInner::Mask | BoundChecksInner::Branch => {
                let coordinate_max = shape / line_size;
                let null_input = I::null_input(inst, line_size);

                ReaderBoundChecksInfo::new_Required(ReaderBoundChecks::<P> {
                    bound_checks,
                    coordinate_max,
                    null_input,
                })
            }
        };

        ParallelPlaneReader::<P> {
            view: input.view(PlainLayout::new(input.len())),
            batch_offset,
            requirements,
            line_size,
            bound_checks,
            len,
        }
    }

    pub fn read(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        let plane_offset = line_index * CUBE_DIM_X;
        let unit_offset = UNIT_POS_X;
        let offset = unit_offset + plane_offset + self.batch_offset;

        let item = match &self.bound_checks {
            ReaderBoundChecksInfo::NotRequired => self.view[offset],
            ReaderBoundChecksInfo::Required(checks) => match comptime!(checks.bound_checks) {
                BoundChecksInner::None => self.view[offset],
                BoundChecksInner::Mask => {
                    let mask = (plane_offset + UNIT_POS_X) < checks.coordinate_max;
                    let index = offset * u32::cast_from(mask);
                    select(mask, self.view[index], checks.null_input)
                }
                BoundChecksInner::Branch => {
                    if offset < checks.coordinate_max {
                        self.view[offset]
                    } else {
                        checks.null_input
                    }
                }
            },
        };

        let coordinate = match comptime!(self.requirements.coordinates) {
            true => ReduceCoordinate::new(
                (line_index * self.line_size * CUBE_DIM_X) + UNIT_POS_X * self.line_size,
                self.requirements,
                self.line_size,
                LineMode::Parallel,
            ),
            false => ReduceCoordinate::new_NotRequired(),
        };

        (item, coordinate)
    }
}

#[cube]
impl<P: ReducePrecision> PerpendicularPlaneReader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_axis: u32,
        reduce_index: u32,
        #[comptime] bound_checks: BoundChecksInner,
    ) -> PerpendicularPlaneReader<P> {
        let line_size = input.line_size();
        let output_index = reduce_index * line_size;

        let mut batch_offset = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(output_index, axis);
            batch_offset += coordinate * input.stride(axis);
        }
        batch_offset /= line_size;

        let requirements = I::requirements(inst);
        let vector_offset_stride = input.stride(reduce_axis) / line_size;
        let shape = input.shape(reduce_axis);
        let len = shape.div_ceil(CUBE_DIM_X);

        let bound_checks = match comptime!(bound_checks) {
            BoundChecksInner::None => ReaderBoundChecksInfo::new_NotRequired(),
            BoundChecksInner::Mask | BoundChecksInner::Branch => {
                let coordinate_max = shape;
                let null_input = I::null_input(inst, line_size);

                ReaderBoundChecksInfo::new_Required(ReaderBoundChecks::<P> {
                    bound_checks,
                    coordinate_max,
                    null_input,
                })
            }
        };

        PerpendicularPlaneReader::<P> {
            view: input.view(PlainLayout::new(input.len())),
            batch_offset,
            vector_offset_stride,
            requirements,
            line_size,
            bound_checks,
            len,
        }
    }

    pub fn read(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        let plane_offset = line_index * self.vector_offset_stride * CUBE_DIM_X;
        let unit_offset = UNIT_POS_X * self.vector_offset_stride;
        let offset = unit_offset + plane_offset + self.batch_offset;

        let item = match &self.bound_checks {
            ReaderBoundChecksInfo::NotRequired => self.view[offset],
            ReaderBoundChecksInfo::Required(checks) => match comptime!(checks.bound_checks) {
                BoundChecksInner::None => self.view[offset],
                BoundChecksInner::Mask => {
                    let base = line_index * CUBE_DIM_X;
                    let mask = (base + UNIT_POS_X) < checks.coordinate_max;
                    let index = offset * u32::cast_from(mask);
                    select(mask, self.view[index], checks.null_input)
                }
                BoundChecksInner::Branch => {
                    if offset < checks.coordinate_max {
                        self.view[offset]
                    } else {
                        checks.null_input
                    }
                }
            },
        };

        let coordinate = match comptime!(self.requirements.coordinates) {
            true => ReduceCoordinate::new_Required(
                Line::empty(self.line_size).fill(line_index + UNIT_POS_X),
            ),
            false => ReduceCoordinate::new_NotRequired(),
        };

        (item, coordinate)
    }
}
