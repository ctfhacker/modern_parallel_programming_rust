//! Adding two slices of f32s and printing the results
#![feature(portable_simd)]
#![feature(variant_count)]

use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
use std::simd::{LaneCount, Simd, SupportedLaneCount};

/// Size of the arrays being added together
const ARRAY_SIZE: usize = 19;

/// Number of iterations to execute the timing to average
const ITERS: usize = 0x1ffff;

/// Straight forward, naive implementation
fn naive(x: &[f32], y: &[f32], out: &mut [f32]) {
    assert!(x.len() == y.len());
    assert!(y.len() == out.len());

    for i in 0..x.len() {
        out[i] = x[i] + y[i];
    }
}

/// SIMD implementation using intrinsics
fn intrinsics(mut x: &[f32], mut y: &[f32], mut out: &mut [f32]) {
    const CHUNK_SIZE: usize = 8;

    unsafe {
        while x.len() >= CHUNK_SIZE {
            let x_chunk = _mm256_loadu_ps(x.as_ptr().cast());
            let y_chunk = _mm256_loadu_ps(y.as_ptr().cast());
            let sum_chunk = _mm256_add_ps(x_chunk, y_chunk);
            _mm256_storeu_ps(out.as_mut_ptr().cast(), sum_chunk);

            // Set the pointer to the next chunk in the slice
            x = &x[CHUNK_SIZE..];
            y = &y[CHUNK_SIZE..];
            out = &mut out[CHUNK_SIZE..];
        }
    }

    // Add out[i] = x[i] + [y] for the remainder of the chunks
    naive(x, y, out);
}

/// SIMD implementation using stdsimd with a variety number of possible lanes
fn stdsimd<const LANES: usize>(mut x: &[f32], mut y: &[f32], mut out: &mut [f32])
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // Ensure the input slices are the same size
    assert!(x.len() == y.len());
    assert!(y.len() == out.len());

    while x.len() >= LANES {
        // Get the SIMD values for the current slice
        let x_chunk = Simd::<f32, LANES>::from_slice(x);
        let y_chunk = Simd::<f32, LANES>::from_slice(y);

        // out[i] = x[i] + y[i]
        let sum_chunk = x_chunk + y_chunk;
        out[..LANES].copy_from_slice(&sum_chunk.to_array());

        // Set the pointer to the next chunk in the slice
        x = &x[LANES..];
        y = &y[LANES..];
        out = &mut out[LANES..];
    }

    // Add out[i] = x[i] + [y] for the remainder of the chunks
    naive(x, y, out);
}

/// Read the time stamp using rdtscp to ensure previous instructions have been executed
fn rdtsc() -> u64 {
    let mut x = 0;
    unsafe { std::arch::x86_64::__rdtscp(&mut x) }
}

/// Various stats we are keeping track of
#[derive(Debug)]
enum Stats {
    Naive,
    Intrinsics,
    StdSimdF32x8,
    StdSimdF32x16,
    Total,
}

fn main() {
    // Initialize the data arrays
    let mut x = [0.0_f32; ARRAY_SIZE];
    let mut y = [0.0_f32; ARRAY_SIZE];

    // Init the stats accumulator
    let mut stats = [0u64; std::mem::variant_count::<Stats>()];

    /// Macro used for timing work
    macro_rules! time {
        ($stat:ident, $work:expr) => {{
            // Start the timer
            let start = rdtsc();

            // Perform the work being timed
            let res = $work;

            // Add the elapsed time this work took
            let curr_stats_cycle = rdtsc() - start;

            stats[Stats::$stat as usize] += curr_stats_cycle;
            stats[Stats::Total as usize] += curr_stats_cycle;

            // Return the result from the work
            res
        }};
    }

    // Initialize the input X and Y data arrays with values
    #[allow(clippy::cast_precision_loss)]
    x.iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = i as f32 * 10. + 10.);
    #[allow(clippy::cast_precision_loss)]
    y.iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = i as f32 * 1000. + 1000.);

    for iter in 0..ITERS {
        // Initialize the data arrays
        let mut z_naive = [0.0_f32; ARRAY_SIZE];
        let mut z_intrinsics = [0.0_f32; ARRAY_SIZE];
        let mut z_stdsimd_8 = [0.0_f32; ARRAY_SIZE];
        let mut z_stdsimd_16 = [0.0_f32; ARRAY_SIZE];

        // Call the functions for the different implementations
        time!(Naive, naive(&x, &y, &mut z_naive));
        time!(Intrinsics, intrinsics(&x, &y, &mut z_intrinsics));
        time!(StdSimdF32x16, stdsimd::<16>(&x, &y, &mut z_stdsimd_16));
        time!(StdSimdF32x8, stdsimd::<8>(&x, &y, &mut z_stdsimd_8));

        if iter == 0 {
            // Pretty print the table
            const WIDTH: usize = 12;
            println!(
                "{:>w$} {:>w$} {:>w$} {:>w$} {:>w$} {:>w$} {:>w$} ",
                "i",
                "x",
                "y",
                "z_naive",
                "z_intrinsics",
                "z_stdsimd_8",
                "z_stdsimd_16",
                w = WIDTH
            );

            println!("{:-<w$}", "", w = WIDTH * 7);

            for i in 0..ARRAY_SIZE {
                print!("{i:WIDTH$} ");
                print!("{:WIDTH$.4} ", x[i]);
                print!("{:WIDTH$.4} ", y[i]);
                print!("{:WIDTH$.4} ", z_naive[i]);
                print!("{:WIDTH$.4} ", z_intrinsics[i]);
                print!("{:WIDTH$.4} ", z_stdsimd_8[i]);
                print!("{:WIDTH$.4} ", z_stdsimd_16[i]);
                println!();
            }
        }
    }

    /// Macro used to pretty print the statistics
    macro_rules! print_stat {
        ($stat:ident) => {{
            let curr_stat = stats[Stats::$stat as usize];

            eprintln!(
                "{:20} | Avg {:>10.2?} cycles/iter",
                format!("{:?}", Stats::$stat),
                curr_stat as f32 / ITERS as f32,
            );
        }};
    }

    println!(
        "+{:-^width$}+",
        format!(" Performance Stats | Array size {ARRAY_SIZE} "),
        width = 80
    );

    print_stat!(Naive);
    print_stat!(Intrinsics);
    print_stat!(StdSimdF32x8);
    print_stat!(StdSimdF32x16);
}
