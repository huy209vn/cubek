//! Einsum benchmark suite.
//!
//! Benchmarks various einsum operations including:
//! - Matrix multiplication (fast path)
//! - Batched matrix multiplication (fast path)
//! - Tensor chain contractions (multi-step)
//! - Reductions
//! - Transposes

use cubecl::{
    benchmark::{Benchmark, BenchmarkComputations, BenchmarkDurations, TimingMethod},
    future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    einsum::{self, EinsumConfig, ContractionStrategy},
    random::random_uniform,
};

/// Einsum benchmark configuration.
#[allow(dead_code)]
struct EinsumBench<R: Runtime> {
    notation: &'static str,
    shapes: Vec<Vec<usize>>,
    strategy: ContractionStrategy,
    device: R::Device,
    client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for EinsumBench<R> {
    type Input = (Vec<TensorHandle<R>>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let dtype = f32::as_type_native_unchecked();

        // Create input tensors
        let inputs: Vec<TensorHandle<R>> = self.shapes.iter().map(|shape| {
            let tensor = TensorHandle::empty(&client, shape.clone(), dtype);
            random_uniform(&client, 0.0f32, 1.0f32, tensor.as_ref(), dtype).unwrap();
            tensor
        }).collect();

        // Compute output shape based on notation
        let output_shape = compute_output_shape(&self.notation, &self.shapes);
        let output = TensorHandle::empty(&client, output_shape, dtype);

        (inputs, output)
    }

    fn execute(&self, (inputs, mut output): Self::Input) -> Result<Self::Output, String> {
        let input_refs: Vec<&TensorHandle<R>> = inputs.iter().collect();
        let config = EinsumConfig {
            strategy: self.strategy,
            validate_shapes: false,
        };

        einsum::einsum::<R, f32>(
            &self.client,
            self.notation,
            &input_refs,
            &mut output,
            Some(config),
        ).map_err(|e| format!("{:?}", e))
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        let shape_str: Vec<String> = self.shapes.iter()
            .map(|s| format!("{:?}", s))
            .collect();
        format!(
            "{}-einsum-{}-shapes[{}]-{:?}",
            R::name(&client),
            self.notation.replace(",", "_").replace("->", "_to_"),
            shape_str.join(","),
            self.strategy
        ).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<cubecl::benchmark::ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "einsum-bench")
            .map(|it| it.1)
            .map_err(|err| format!("{err:?}"))
    }
}

/// Computes output shape from einsum notation and input shapes.
fn compute_output_shape(notation: &str, shapes: &[Vec<usize>]) -> Vec<usize> {
    use std::collections::HashMap;

    // Parse notation
    let parts: Vec<&str> = notation.split("->").collect();
    if parts.len() != 2 {
        return vec![1]; // Scalar output for implicit reduction
    }

    let inputs_str = parts[0];
    let output_str = parts[1];

    // Build dimension map
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    let input_subscripts: Vec<&str> = inputs_str.split(',').collect();

    for (i, subscript) in input_subscripts.iter().enumerate() {
        if i >= shapes.len() {
            break;
        }
        for (j, c) in subscript.chars().enumerate() {
            if j < shapes[i].len() && c.is_alphabetic() {
                dim_map.insert(c, shapes[i][j]);
            }
        }
    }

    // Build output shape
    if output_str.is_empty() {
        vec![1] // Scalar
    } else {
        output_str.chars()
            .filter(|c| c.is_alphabetic())
            .filter_map(|c| dim_map.get(&c).copied())
            .collect()
    }
}

/// Runs a single einsum benchmark.
#[allow(dead_code)]
fn run_one<R: Runtime>(
    device: R::Device,
    notation: &'static str,
    shapes: Vec<Vec<usize>>,
    strategy: ContractionStrategy,
) -> Result<(BenchmarkDurations, f64), String> {
    let client = R::client(&device);

    let bench = EinsumBench {
        notation,
        shapes: shapes.clone(),
        strategy,
        client: client.clone(),
        device: device.clone(),
    };

    println!("Einsum: {} with shapes {:?}", notation, shapes);
    println!("{}", bench.name());

    match bench.run(TimingMethod::System) {
        Ok(val) => {
            let flops = estimate_flops(notation, &shapes);
            let computed = BenchmarkComputations::new(&val);
            let tflops = flops as f64 / (computed.median.as_secs_f64() * 1e12);
            println!("TFLOPS: {:.3}", tflops);
            println!("Times: {val}");
            Ok((val, tflops))
        }
        Err(err) => {
            println!("Error: {err:?}");
            Err(err)
        }
    }
}

/// Estimates FLOPs for an einsum operation.
fn estimate_flops(notation: &str, shapes: &[Vec<usize>]) -> u64 {
    use std::collections::HashMap;

    let parts: Vec<&str> = notation.split("->").collect();
    let inputs_str = parts[0];

    // Build dimension map
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    let input_subscripts: Vec<&str> = inputs_str.split(',').collect();

    for (i, subscript) in input_subscripts.iter().enumerate() {
        if i >= shapes.len() {
            break;
        }
        for (j, c) in subscript.chars().enumerate() {
            if j < shapes[i].len() && c.is_alphabetic() {
                dim_map.insert(c, shapes[i][j]);
            }
        }
    }

    // FLOPs = 2 * product of all dimensions (for multiply-add)
    let total_size: u64 = dim_map.values().map(|&d| d as u64).product();
    2 * total_size
}

/// Benchmark matmul via einsum (should use fast path).
#[allow(unused)]
fn bench_matmul<R: Runtime>(device: R::Device) {
    println!("\n=== Matrix Multiplication (Fast Path) ===");
    for (m, k, n) in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)] {
        let _ = run_one::<R>(
            device.clone(),
            "ij,jk->ik",
            vec![vec![m, k], vec![k, n]],
            ContractionStrategy::Auto,
        );
    }
}

/// Benchmark batched matmul via einsum.
#[allow(unused)]
fn bench_batched_matmul<R: Runtime>(device: R::Device) {
    println!("\n=== Batched Matrix Multiplication (Fast Path) ===");
    for (b, m, k, n) in [(8, 512, 512, 512), (16, 256, 256, 256), (32, 128, 128, 128)] {
        let _ = run_one::<R>(
            device.clone(),
            "bij,bjk->bik",
            vec![vec![b, m, k], vec![b, k, n]],
            ContractionStrategy::Auto,
        );
    }
}

/// Benchmark tensor chain contraction.
#[allow(unused)]
fn bench_chain_contraction<R: Runtime>(device: R::Device) {
    println!("\n=== Chain Contraction (Multi-Step) ===");

    // 3-tensor chain: A @ B @ C
    let _ = run_one::<R>(
        device.clone(),
        "ij,jk,kl->il",
        vec![vec![256, 512], vec![512, 256], vec![256, 128]],
        ContractionStrategy::Auto,
    );

    // Compare greedy vs optimal
    println!("\n--- Greedy vs Optimal Path ---");
    let _ = run_one::<R>(
        device.clone(),
        "ij,jk,kl->il",
        vec![vec![10, 100], vec![100, 1000], vec![1000, 10]],
        ContractionStrategy::Greedy,
    );
    let _ = run_one::<R>(
        device.clone(),
        "ij,jk,kl->il",
        vec![vec![10, 100], vec![100, 1000], vec![1000, 10]],
        ContractionStrategy::Optimal,
    );
}

/// Benchmark reduction operations.
#[allow(unused)]
fn bench_reductions<R: Runtime>(device: R::Device) {
    println!("\n=== Reductions (Fast Path) ===");

    // Sum all elements
    let _ = run_one::<R>(
        device.clone(),
        "ij->",
        vec![vec![1024, 1024]],
        ContractionStrategy::Auto,
    );

    // Sum along axis
    let _ = run_one::<R>(
        device.clone(),
        "ij->i",
        vec![vec![1024, 1024]],
        ContractionStrategy::Auto,
    );

    // Trace
    let _ = run_one::<R>(
        device.clone(),
        "ii->",
        vec![vec![1024, 1024]],
        ContractionStrategy::Auto,
    );
}

/// Benchmark element-wise operations.
#[allow(unused)]
fn bench_elementwise<R: Runtime>(device: R::Device) {
    println!("\n=== Element-wise Operations ===");

    // Hadamard product
    let _ = run_one::<R>(
        device.clone(),
        "ij,ij->ij",
        vec![vec![1024, 1024], vec![1024, 1024]],
        ContractionStrategy::Auto,
    );

    // Outer product
    let _ = run_one::<R>(
        device.clone(),
        "i,j->ij",
        vec![vec![1024], vec![1024]],
        ContractionStrategy::Auto,
    );

    // Dot product
    let _ = run_one::<R>(
        device.clone(),
        "i,i->",
        vec![vec![1024 * 1024], vec![1024 * 1024]],
        ContractionStrategy::Auto,
    );
}

/// Benchmark attention-like patterns.
#[allow(unused)]
fn bench_attention_pattern<R: Runtime>(device: R::Device) {
    println!("\n=== Attention Patterns ===");

    // Attention scores: Q @ K^T
    let _ = run_one::<R>(
        device.clone(),
        "bhqd,bhkd->bhqk",
        vec![vec![8, 12, 512, 64], vec![8, 12, 512, 64]],
        ContractionStrategy::Auto,
    );

    // Attention output: scores @ V
    let _ = run_one::<R>(
        device.clone(),
        "bhqk,bhkd->bhqd",
        vec![vec![8, 12, 512, 512], vec![8, 12, 512, 64]],
        ContractionStrategy::Auto,
    );
}

/// Run all einsum benchmarks.
#[allow(unused)]
fn run_all_benches<R: Runtime>() {
    let device = R::Device::default();

    bench_matmul::<R>(device.clone());
    bench_batched_matmul::<R>(device.clone());
    bench_chain_contraction::<R>(device.clone());
    bench_reductions::<R>(device.clone());
    bench_elementwise::<R>(device.clone());
    bench_attention_pattern::<R>(device.clone());
}

fn main() {
    println!("===========================================");
    println!("         CubeK Einsum Benchmarks          ");
    println!("===========================================\n");

    run_all_benches::<cubecl::TestRuntime>();
}
