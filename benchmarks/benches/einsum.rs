//! Einsum benchmark suite.
//! (Updated: correct FLOP estimator + kernel vs system measurement + warmup)
use cubecl::profile::ProfileTicks;
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
use std::time::Duration;

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
            use_tensor_cores: true,
            autotune: false,
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
        // If no '->' provided, fall back to scalar (our benchmarks always use explicit ->).
        return vec![1];
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

/// Estimates FLOPs for an einsum operation (correct general formula).
///
/// Explanation:
///  - Let `inputs` be the input subscripts, `output` the output subscripts.
///  - Let `O` be the product of sizes of output indices.
///  - Let `S` be the product of sizes of summation (contracted) indices (indices present in inputs but not in output).
///  - Let `A` be the number of input tensors.
///
/// For each output element we compute `product_S` summands; each summand requires `(A - 1)` multiplies
/// (to multiply A inputs together), and summing across the `product_S` summands uses `product_S - 1` adds.
/// Therefore:
///   multiplies = (A - 1) * (O * S)
///   adds       = if S > 0 { (S - 1) * O } else { 0 }
///   flops      = multiplies + adds
fn estimate_flops(notation: &str, shapes: &[Vec<usize>]) -> u64 {
    use std::collections::HashMap;

    let parts: Vec<&str> = notation.split("->").collect();
    let inputs_str = parts[0];
    let output_str = if parts.len() == 2 { parts[1] } else { "" };

    // Build dimension map (same as compute_output_shape)
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

    // Collect all indices appearing in inputs
    let mut all_indices = Vec::new();
    for sub in &input_subscripts {
        for c in sub.chars().filter(|c| c.is_alphabetic()) {
            if !all_indices.contains(&c) {
                all_indices.push(c);
            }
        }
    }

    // Output indices set
    let output_indices: Vec<char> = output_str.chars().filter(|c| c.is_alphabetic()).collect();

    // Summation indices = all_indices \ output_indices
    let summation_indices: Vec<char> = all_indices
        .iter()
        .copied()
        .filter(|c| !output_indices.contains(c))
        .collect();

    // Compute products (use u128 to avoid overflow for large dims)
    let mut product_o: u128 = 1;
    for c in &output_indices {
        if let Some(&d) = dim_map.get(c) {
            product_o = product_o.saturating_mul(d as u128);
        } else {
            // missing dim info -> assume 1
            product_o = product_o.saturating_mul(1);
        }
    }

    let mut product_s: u128 = 1;
    for c in &summation_indices {
        if let Some(&d) = dim_map.get(c) {
            product_s = product_s.saturating_mul(d as u128);
        } else {
            product_s = product_s.saturating_mul(1);
        }
    }

    let num_inputs = input_subscripts.len() as u128;
    let product_all = product_o.saturating_mul(product_s);

    // multiplies
    let multiplies = if num_inputs >= 1 {
        (num_inputs - 1).saturating_mul(product_all)
    } else {
        0u128
    };

    // adds
    let adds = if product_s > 0 {
        product_o.saturating_mul(product_s.saturating_sub(1))
    } else {
        0u128
    };

    let flops = multiplies.saturating_add(adds);

    // clamp to u64 (if unrealistic dimensions overflow, cap at u64::MAX)
    if flops > u64::MAX as u128 {
        u64::MAX
    } else {
        flops as u64
    }
}

/// Runs a single einsum benchmark and reports both kernel TFLOPS and system timing.
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

    // -------------------------
    // Warmup
    // -------------------------
    let warm_args = bench.prepare();
    bench.execute(warm_args);
    bench.sync();

    // -------------------------
    // Kernel profiling
    // -------------------------
    let profile_args = bench.prepare();
    let profile_duration = bench.profile(profile_args.clone());


let profile_duration = bench.profile(profile_args)
    .map_err(|e| format!("profiling failed: {:?}", e))?;

let ticks = future::block_on(profile_duration.resolve());
let kernel_secs = ticks.duration().as_secs_f64();

    // -------------------------
    // System timing
    // -------------------------
    match bench.run(TimingMethod::System) {
        Ok(val) => {
            let flops = estimate_flops(notation, &shapes);
            let secs = kernel_secs.max(1e-12);
            let tflops = flops as f64 / (secs * 1e12);

            println!("TFLOPS (kernel): {:.3}", tflops);
            println!("Times (system): {val}");

            Ok((val, tflops))
        }
        Err(err) => Err(format!("{err:?}")),
    }
}
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
