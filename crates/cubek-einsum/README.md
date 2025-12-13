# CubeK Einsum

SOTA Einstein Summation (Einsum) implementation for GPU tensor operations.

## Overview

CubeK Einsum provides a high-level API for expressing complex tensor operations using Einstein summation notation, with automatic optimization and execution on GPU hardware.

## Features

### ✅ Implemented
- **Full einsum notation parsing** - Supports all standard syntax including ellipsis (`...`)
- **Pattern recognition** - Fast paths for common operations:
  - Matrix multiplication (`ij,jk->ik`)
  - Batched matrix multiplication (`bij,bjk->bik`)
  - Transpose (`ij->ji`)
  - Reduction (`ij->i`)
  - Hadamard product (`ij,ij->ij`)
  - Outer product (`i,j->ij`)
  - Dot product (`i,i->`)
  - Trace (`ii->`)
- **Contraction path optimization** - Greedy and dynamic programming algorithms
- **Workspace management** - Automatic allocation for multi-step contractions
- **General contraction execution** - Support for chain operations like `ij,jk,kl->il`

### 🚧 In Progress
- Branch and bound optimization algorithm
- Tensor core integration (WMMA)
- Autotuning framework
- Kernel fusion optimization

## Usage

### Basic Example

```rust
use cubecl::prelude::*;
use cubecl::runtime::CudaRuntime;
use cubecl::client::ComputeClient;
use cubek_einsum::einsum;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = CudaRuntime::new()?;
    let client = ComputeClient::new(&runtime)?;

    // Create tensors
    let mut a = TensorHandle::zeros(&client, [1024, 512], StorageType::F32);
    let mut b = TensorHandle::zeros(&client, [512, 256], StorageType::F32);
    let mut c = TensorHandle::zeros(&client, [1024, 256], StorageType::F32);

    // Matrix multiplication using einsum notation
    einsum::<CudaRuntime, f32>(
        &client,
        "ij,jk->ik",
        &[&a, &b],
        &mut c,
        None,
    )?;

    Ok(())
}
```

### Chain Contraction Example

```rust
// Chain of three matrix multiplications: A @ B @ C
let mut result = TensorHandle::zeros(&client, [64, 512], StorageType::F32);
einsum::<CudaRuntime, f32>(
    &client,
    "ij,jk,kl->il",
    &[&a, &b, &c],
    &mut result,
    None,
)?;
```

### Batched Operations Example

```rust
// Batched matrix multiplication with batch size 32
let mut a = TensorHandle::zeros(&client, [32, 64, 128], StorageType::F32); // batch x m x k
let mut b = TensorHandle::zeros(&client, [32, 128, 256], StorageType::F32); // batch x k x n
let mut c = TensorHandle::zeros(&client, [32, 64, 256], StorageType::F32); // batch x m x n

einsum::<CudaRuntime, f32>(
    &client,
    "bij,bjk->bik",
    &[&a, &b],
    &mut c,
    None,
)?;
```

## Notation Reference

| Notation | Operation | Example |
|----------|-----------|---------|
| `ij,jk->ik` | Matrix multiply | C[i,k] = sum_j A[i,j] * B[j,k] |
| `bij,bjk->bik` | Batched matmul | Batch dimension preserved |
| `ij->ji` | Transpose | Dimension reorder |
| `ii->` | Trace | Diagonal sum |
| `ij,ij->ij` | Hadamard product | Element-wise multiply |
| `i,j->ij` | Outer product | Rank expansion |
| `ij->i` | Row sum | Reduction over j |
| `ij->` | Total sum | Full reduction |

## Performance Characteristics

- **Fast paths**: Direct dispatch to optimized cubek kernels (matmul, reduce)
- **Chain contractions**: Automatic workspace management for intermediate results
- **Automatic optimization**: Intelligent path selection based on problem size
- **Memory efficient**: Reuse of tensor handles where possible

## Testing

Run the test suite:

```bash
cd cubek/crates/cubek-einsum
cargo test --lib
```

## Implementation Status

Based on the original specification, we're currently in **Phase 3 (Performance)** with these achievements:

- ✅ Foundation (Phase 1) - Complete
- ✅ Optimization (Phase 2) - Complete
- 🔄 Performance (Phase 3) - In progress
  - Workspace management: ✅ Implemented
  - General contraction execution: ✅ Implemented
  - Tensor core integration: ⏳ Not yet implemented
  - Autotuning: ⏳ Not yet implemented
- ❌ Polish (Phase 4) - Not started

## Future Work

1. **Branch and bound optimization** - Better algorithm for medium-sized expressions
2. **Tensor core integration** - WMMA support for matrix operations
3. **Autotuning framework** - Shape-aware tuning cache
4. **Kernel fusion** - Combining operations where beneficial
5. **Comprehensive benchmarks** - Performance comparison with cuTENSOR, opt_einsum