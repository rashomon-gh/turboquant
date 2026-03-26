Technical Specification: TurboQuant Implementation in Rust
1. Project Overview
This project requires implementing TurboQuant, an online, data-oblivious vector quantization algorithm optimized for high-dimensional Euclidean spaces (e.g., LLM KV caches, vector databases). The implementation will feature two variants:
TurboQuant$_{\text{mse}}$: Optimized for Mean-Squared Error (MSE) using random rotation and independent scalar quantization.
TurboQuant$_{\text{prod}}$: Optimized for unbiased inner product estimation by composing a $(b-1)$-bit TurboQuant$_{\text{mse}}$ with a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform on the residual error.
2. Agent Skills & Crate Ecosystem Requirements
The implementing agent must utilize the following Rust paradigms and libraries:
Linear Algebra: ndarray for tensor/matrix representation and faer (or nalgebra) for fast matrix factorizations (specifically QR decomposition for generating orthogonal rotation matrices).
Performance / Hot Path Constraints: The quantize and dequantize loops must be zero-allocation (no Vec::new() or Box). All output buffers must be pre-allocated and passed as mutable slices (&mut [T]).
Hardware Acceleration: Explicit use of SIMD (std::simd or core::arch) for finding the nearest centroids and computing vector dot products.
Concurrency: rayon for parallelizing the quantization of batches of vectors (e.g., token matrices).
Randomness: rand and rand_distr (specifically StandardNormal) for offline matrix generation.
---
3. Mathematical & Algorithmic Primitives
The agent must accurately implement the mathematical logic defined in the paper.
3.1. Matrix Generation (Offline Phase)
Rotation Matrix ($\Pi$): To generate $\Pi \in \mathbb{R}^{d \times d}$, sample a matrix with i.i.d. Normal $\mathcal{N}(0,1)$ entries, then apply QR decomposition. $\Pi$ is the resulting orthogonal $Q$ matrix.
Projection Matrix ($S$): To generate $S \in \mathbb{R}^{d \times d}$, simply sample i.i.d. Normal $\mathcal{N}(0,1)$ entries.
3.2. Centroid Generation (Offline Phase)
For a given dimension $d$ and bit-width $b$, coordinates of the rotated vectors follow a Beta distribution. The agent must compute or hardcode optimal 1-dimensional k-means centroids.
Note: For moderately high dimensions, the distribution approaches $\mathcal{N}(0, 1/d)$.
Hardcoded fallbacks provided in the paper: For $b=1$, centroids are $\{\pm\frac{\sqrt{2/\pi}}{\sqrt{d}}\}$. For $b=2$, centroids are $\{\pm\frac{0.453}{\sqrt{d}}, \pm\frac{1.51}{\sqrt{d}}\}$.
---
4. Rust Architecture & Boilerplate
The agent should use the following architectural scaffolding to begin implementation:
Rust
use ndarray::{Array1, Array2};  
// Assume `faer` or `ndarray-linalg` is used for QR decomposition
/// Represents the MSE-optimized TurboQuant algorithm.  
pub struct TurboQuantMse {  
/// Dimension of the input vectors (d)  
pub d: usize,  
/// Bit-width per coordinate (b)  
pub b: usize,  
/// Random rotation matrix (\Pi \in R^{d \times d})  
pub rotation_matrix: Array2<f32>,  
/// Precomputed optimal 1D k-means centroids (length 2^b)  
pub centroids: Vec<f32>,  
}
/// Represents the Inner Product-optimized TurboQuant algorithm.  
pub struct TurboQuantProd {  
/// Dimension of the input vectors (d)  
pub d: usize,  
/// Target total bit-width per coordinate (b)  
pub b: usize,  
/// The underlying MSE quantizer initialized with (b - 1) bits [cite: 270]  
pub mse_quantizer: TurboQuantMse,  
/// Random projection matrix (S \in R^{d \times d}) with N(0,1) entries  
pub projection_matrix: Array2<f32>,  
}
/// The result of an inner-product quantization[cite: 260, 274].  
pub struct ProdQuantizedVector {  
/// Indices from the MSE quantization stage  
pub idx: Vec<u8>,  
/// 1-bit QJL transform of the residual (-1 or 1, can be packed into bitsets)  
pub qjl_signs: Vec<i8>,  
/// L2 norm of the residual error (\gamma)  
pub gamma: f32,  
}
pub trait VectorQuantizer {  
type QuantizedType;
    /// Quantizes a single vector.   
    /// Note: In a production setting, this should accept a mutable out-parameter   
    /// to avoid allocations.  
    fn quantize(&self, x: \&Array1\<f32\>) \-\> Self::QuantizedType;

    /// Reconstructs a vector from its quantized representation.  
    fn dequantize(&self, quantized: \&Self::QuantizedType, out: &mut Array1\<f32\>);  
}
---
5. Algorithmic Workflow Requirements
5.1.
Implementing TurboQuantMse (Algorithm 1)
quantize(x):
Compute $y = \Pi \cdot x$ (Matrix-vector multiplication).
For each coordinate $y_j$, find the index $k$ that minimizes $|y_j - c_k|$, where $c$ is the centroids array.
Return the array of indices idx.
dequantize(idx, out):
Map each index in idx back to its centroid value to form a vector $\tilde{y}$ where $\tilde{y}_j = c_{idx_j}$.
Compute $\tilde{x} = \Pi^{\top} \cdot \tilde{y}$ and write to out.
5.2.
Implementing TurboQuantProd (Algorithm 2)
quantize(x):
Call mse_quantizer.quantize(x) to get idx.
Call mse_quantizer.dequantize(idx) to get $\tilde{x}_{mse}$.
Compute the residual vector: $r = x - \tilde{x}_{mse}$.
Compute $\gamma = \|r\|_2$ (L2 norm of the residual).
Compute the QJL transform: $qjl = \text{sign}(S \cdot r)$ (Matrix-vector mult followed by sign extraction).
Return ProdQuantizedVector { idx, qjl_signs: qjl, gamma }.
dequantize(quantized, out):
Compute $\tilde{x}_{mse}$ using mse_quantizer.dequantize(quantized.idx).
Compute the QJL reconstruction: $\tilde{x}_{qjl} = \frac{\sqrt{\pi/2}}{d} \cdot \gamma \cdot S^{\top} \cdot qjl$.
Compute final vector: $\tilde{x} = \tilde{x}_{mse} + \tilde{x}_{qjl}$ and write to out.
---
6. Implementation Milestones for the Coding Agent
Setup & Dependency Resolution: Initialize the Cargo.toml with ndarray, faer, rand, rand_distr, and rayon.
Offline Math: Implement the random rotation matrix generation (QR decomposition)  and centroid calculation/fallback generation.
Core Trait Implementation: Implement VectorQuantizer for TurboQuantMse. Write a unit test ensuring $D_{\text{mse}}$ remains within theoretical bounds.
Composition: Implement VectorQuantizer for TurboQuantProd. Write a unit test ensuring the expected inner product is unbiased ($\mathbb{E}[\langle y, \tilde{x} \rangle] \approx \langle y, x \rangle$).
Optimization: Refactor matrix multiplications to use BLAS/faer backends. Pack the qjl_signs (i8) into a compact bitset (u8 arrays) to achieve true 1-bit per coordinate memory usage.
7. Extension: Matrix Generation & QR Decomposition (Offline Phase)
Agent Instructions: To induce the required Beta distribution on the coordinates of the input vectors, TurboQuantMse requires a random rotation matrix $\Pi$. The agent must construct this by generating a dense square matrix with entries sampled from a Standard Normal distribution $\mathcal{N}(0,1)$, and then extracting the orthogonal matrix $Q$ via QR decomposition.
For the Rust implementation, combining ndarray with the ndarray-linalg crate (which binds to LAPACK/OpenBLAS) is the most efficient path for this offline setup.
Boilerplate (QR Decomposition):
Rust
use ndarray::Array2;
use ndarray_linalg::QR;
use rand::thread_rng;
use rand_distr::{StandardNormal, Distribution};
/// Generates a random rotation matrix (\Pi) of size d x d using QR decomposition.
/// This should be executed once during the offline initialization phase.
pub fn generate_rotation_matrix(d: usize) -> Array2<f32> {
    let mut rng \= thread\_rng();

    

    // 1\. Generate a d x d matrix with i.i.d. N(0,1) entries

    let gaussian\_matrix: Array2\<f32\> \= Array2::from\_shape\_simple\_fn((d, d), || {

        StandardNormal.sample(\&mut rng)

    });

    // 2\. Perform QR decomposition. 

    // We only need the orthogonal matrix 'Q', which represents our rotation matrix \\Pi.

    let (q, \_r) \= gaussian\_matrix.qr().expect("QR decomposition failed");

    

    q

}
/// Generates the random projection matrix (S) for the QJL transform.
/// Entries are i.i.d N(0,1). No QR decomposition is needed here.
pub fn generate_projection_matrix(d: usize) -> Array2<f32> {
    let mut rng \= thread\_rng();

    Array2::from\_shape\_simple\_fn((d, d), || {

        StandardNormal.sample(\&mut rng)

    })

}
---
8. Extension: QJL Transform Implementation (Inner Product)
Agent Instructions: The QJL transform is the core of TurboQuantProd's ability to provide unbiased inner product estimations. It operates on the residual vector $r$ (the difference between the original vector and the MSE-quantized vector).
Quantization: The agent must compute the sign of the dot product between the projection matrix $S$ and the residual vector $r$: $qjl = \text{sign}(S \cdot r)$.
Dequantization: The agent must reconstruct the residual using the transpose of the projection matrix $S^{\top}$, scaled by the L2 norm of the residual $\gamma$, the dimension $d$, and the constant $\sqrt{\pi/2}$.
Boilerplate (QJL Logic):
Rust
use ndarray::{Array1, Array2};
// Mathematical constant for QJL dequantization: sqrt(pi / 2)
const SQRT_PI_OVER_2: f32 = 1.2533141;
/// Computes the 1-bit QJL transform on the residual vector.
/// S is the projection matrix, r is the residual vector.
pub fn qjl_quantize(s_matrix: &Array2<f32>, r: &Array1<f32>) -> Vec<i8> {
    // Compute S \* r

    let projected \= s\_matrix.dot(r);

    

    // Extract the sign of each coordinate. 

    // Note: To achieve true 1-bit compression in production, 

    // these i8 values (-1 or 1\) should be packed into a bitset (Vec\<u8\>).

    projected.iter().map(|\&val| {

        if val \>= 0.0 { 1 } else { \-1 }

    }).collect()

}
/// Dequantizes the QJL output to approximate the original residual vector.
/// s_matrix is the projection matrix, qjl_signs are the quantized bits,
/// and gamma is the L2 norm of the original residual vector.
pub fn qjl_dequantize(
    s\_matrix: \&Array2\<f32\>, 

    qjl\_signs: &\[i8\], 

    gamma: f32, 

    d: usize

) -> Array1<f32> {
    // Convert signs back to f32 for matrix multiplication

    let signs\_f32 \= Array1::from\_vec(

        qjl\_signs.iter().map(|\&s| s as f32).collect()

    );

    // Compute S^T \* qjl\_signs

    let projected\_back \= s\_matrix.t().dot(\&signs\_f32);

    

    // Apply the QJL scaling factor: (\\sqrt{\\pi/2} / d) \* \\gamma \* (S^T \* z)

    let scale\_factor \= (SQRT\_PI\_OVER\_2 / (d as f32)) \* gamma;

    

    projected\_back \* scale\_factor

}
Integration Notes for the Agent:
Zero-Allocation Focus: The boilerplate above uses Array1::from_vec and collects into new Vecs for readability. The agent must optimize this in the final implementation to write directly to pre-allocated mutable slices passed into the functions, especially within the qjl_quantize and qjl_dequantize hot paths.
Bit-packing: The qjl_signs currently return Vec<i8>. The agent should be instructed to create a bit-packing utility to compress eight 1 or -1 states into a single u8 to achieve the exact memory footprint described in the paper.
