# TurboQuant

An unofficial implementation of [TurboQuant, 2026](http://arxiv.org/abs/2504.19874), a high-performance vector quantization method for efficient compression of high-dimensional vectors.

## Overview

TurboQuant provides two quantization strategies:

- **TurboQuantMse**: Optimized for Mean-Squared Error (MSE) using random rotation and independent scalar quantization
- **TurboQuantProd**: Optimized for unbiased inner product estimation by combining MSE quantization with Quantized Johnson-Lindenstrauss (QJL) transform

Designed for applications involving high-dimensional Euclidean spaces such as LLM KV caches and vector databases.

## Features

- Zero-allocation hot paths with pre-allocated buffers
- SIMD optimization (x86_64) for matrix operations and centroid finding
- Parallel batch processing with Rayon
- Bit-packing for compact 1-bit storage
- Configurable bit-width per coordinate (1-8 bits)
- Hardcoded centroid fallbacks for 1 and 2-bit quantization

## Quick Start

```rust
use turboquant::{TurboQuantMse, TurboQuantProd, VectorQuantizer, BatchQuantizer};
use ndarray::Array1;

// MSE-optimized quantization (2 bits per coordinate)
let quantizer = TurboQuantMse::new(128, 2);
let vector = Array1::from_vec(vec![0.1, -0.5, 0.3, /* ... */]);
let quantized = quantizer.quantize(&vector.view());

let mut reconstructed = Array1::zeros(128);
quantizer.dequantize(&quantized, &mut reconstructed);

// Product quantization for inner product estimation (3 bits total)
let prod_quantizer = TurboQuantProd::new(128, 3);
let quantized = prod_quantizer.quantize(&vector.view());
```

## Running Tests

```bash
cargo test
```

Run benchmarks:
```bash
cargo bench
```

## Architecture

The library is organized into modules:

- **mse**: MSE-optimized quantization with rotation matrices
- **prod**: Product quantization combining MSE + QJL for inner products
- **centroids**: K-means centroid generation with hardcoded fallbacks
- **matrix**: Random rotation matrix generation via QR decomposition
- **qjl**: Quantized Johnson-Lindenstrauss transform for unbiased estimation
- **bitpack**: Bit-packing utilities for compact storage

## Usage Patterns

### Single Vector Quantization
```rust
let quantizer = TurboQuantMse::new(dimension, bits);
let quantized = quantizer.quantize(&vector.view());
quantizer.dequantize(&quantized, &mut output);
```

### Batch Processing (Parallel)
```rust
let vectors: Vec<ArrayView1<f32>> = /* ... */;
let quantized_batch = quantizer.quantize_batch(&vectors);
quantizer.dequantize_batch(&quantized_batch, &mut outputs);
```

### Zero-Allocation Hot Path
```rust
let mut indices = vec![0u8; d];
let mut rotated = vec![0.0; d];
let mut temp = vec![0.0; d];
let mut output = vec![0.0; d];

quantizer.quantize_into_buffer(&vector.view(), &mut indices, &mut rotated);
quantizer.dequantize_into_buffer(&indices, &mut output, &mut temp);
```

## Performance Characteristics

- **Memory**: Uses u8 indices for centroids (8 bits) with optional bit-packing for QJL signs (1 bit)
- **Speed**: SIMD-optimized matrix multiplication and centroid finding
- **Parallelism**: Rayon-based batch processing for multi-core utilization
- **Accuracy**: 
  - MSE variant: Minimizes reconstruction error
  - Product variant: Provides unbiased inner product estimates

