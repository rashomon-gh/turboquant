//! # Bit-Packing Utilities
//!
//! This module provides efficient bit-packing operations for compressing binary signs
//! (±1 values) into compact byte arrays. This is used by the QJL (Quantized Johnson-Lindenstrauss)
//! transform to achieve true 1-bit per coordinate storage.
//!
//! ## Overview
//!
//! The module supports packing arrays of `i8` signs (-1 or +1) into densely packed `u8` arrays,
//! where each bit represents one sign value:
//! - Bit set to 1: represents sign +1
//! - Bit set to 0: represents sign -1
//!
//! ## Example
//!
//! ```rust
//! use turboquant::bitpack::{pack_bits, unpack_bits, packed_len};
//!
//! let signs = vec![1i8, -1, 1, -1, 1, -1, 1, -1];
//! let packed = pack_bits(&signs);
//!
//! assert_eq!(packed.len(), 1); // 8 signs fit in 1 byte
//!
//! let mut unpacked = vec![0i8; 8];
//! unpack_bits(&packed, 8, &mut unpacked);
//! assert_eq!(signs, unpacked);
//! ```

/// Number of bits in a byte (constant used for bit-packing calculations).
pub const BITS_PER_BYTE: usize = 8;

/// Pack an array of signs (±1) into a compact byte array.
///
/// Each sign value is encoded as a single bit:
/// - +1 → bit 1
/// - -1 → bit 0
///
/// # Arguments
///
/// * `signs` - Array of sign values (-1 or +1)
///
/// # Returns
///
/// A vector of bytes with densely packed sign bits.
///
/// # Example
///
/// ```rust
/// use turboquant::bitpack::pack_bits;
///
/// let signs = vec![1i8, -1, 1, -1]; // 4 signs
/// let packed = pack_bits(&signs);
/// assert_eq!(packed.len(), 1); // Fits in 1 byte
/// ```
#[inline]
pub fn pack_bits(signs: &[i8]) -> Vec<u8> {
    let n = signs.len();
    let packed_len = n.div_ceil(BITS_PER_BYTE);
    let mut packed = vec![0u8; packed_len];

    for (i, &sign) in signs.iter().enumerate() {
        if sign > 0 {
            packed[i / BITS_PER_BYTE] |= 1 << (i % BITS_PER_BYTE);
        }
    }

    packed
}

/// Pack signs into a pre-allocated byte buffer (zero-allocation pattern).
///
/// This is the zero-allocation variant of `pack_bits` for hot paths where
/// memory allocation should be avoided.
///
/// # Arguments
///
/// * `signs` - Array of sign values (-1 or +1)
/// * `packed` - Pre-allocated output buffer (must be at least `packed_len(signs.len())` bytes)
///
/// # Panics
///
/// Panics in debug mode if `packed` is too small.
///
/// # Example
///
/// ```rust
/// use turboquant::bitpack::{pack_bits_into, packed_len};
///
/// let signs = vec![1i8, -1, 1, -1, 1, -1, 1, -1];
/// let mut packed = vec![0u8; packed_len(8)];
/// pack_bits_into(&signs, &mut packed);
/// ```
#[inline]
pub fn pack_bits_into(signs: &[i8], packed: &mut [u8]) {
    let packed_len = signs.len().div_ceil(BITS_PER_BYTE);
    debug_assert!(packed.len() >= packed_len);

    // Clear the buffer first
    for byte in packed.iter_mut() {
        *byte = 0;
    }

    for (i, &sign) in signs.iter().enumerate() {
        if sign > 0 {
            packed[i / BITS_PER_BYTE] |= 1 << (i % BITS_PER_BYTE);
        }
    }
}

/// Unpack a byte array into signs (±1 values).
///
/// Reconstructs the original sign array from packed bit representation.
///
/// # Arguments
///
/// * `packed` - Packed byte array
/// * `n` - Number of signs to unpack
/// * `signs` - Output buffer for unpacked signs (must be at least `n` elements)
///
/// # Example
///
/// ```rust
/// use turboquant::bitpack::{pack_bits, unpack_bits};
///
/// let signs = vec![1i8, -1, 1, -1];
/// let packed = pack_bits(&signs);
///
/// let mut unpacked = vec![0i8; 4];
/// unpack_bits(&packed, 4, &mut unpacked);
/// assert_eq!(signs, unpacked);
/// ```
#[inline]
pub fn unpack_bits(packed: &[u8], n: usize, signs: &mut [i8]) {
    debug_assert!(signs.len() >= n);

    for (i, sign) in signs.iter_mut().enumerate().take(n) {
        let byte_idx = i / BITS_PER_BYTE;
        let bit_idx = i % BITS_PER_BYTE;

        if byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1 {
            *sign = 1;
        } else {
            *sign = -1;
        }
    }
}

/// Unpack a single bit from a packed byte array.
///
/// # Arguments
///
/// * `packed` - Packed byte array
/// * `i` - Index of the bit to unpack
///
/// # Returns
///
/// The sign value at position `i` (-1 or +1).
///
/// # Example
///
/// ```rust
/// use turboquant::bitpack::{pack_bits, unpack_bit};
///
/// let signs = vec![1i8, -1, 1i8];
/// let packed = pack_bits(&signs);
///
/// assert_eq!(unpack_bit(&packed, 0), 1);
/// assert_eq!(unpack_bit(&packed, 1), -1);
/// ```
#[inline]
pub fn unpack_bit(packed: &[u8], i: usize) -> i8 {
    let byte_idx = i / BITS_PER_BYTE;
    let bit_idx = i % BITS_PER_BYTE;

    if byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1 {
        1
    } else {
        -1
    }
}

/// Calculate the number of bytes needed to pack `n` signs.
///
/// # Arguments
///
/// * `n` - Number of signs to pack
///
/// # Returns
///
/// The minimum number of bytes required.
///
/// # Example
///
/// ```rust
/// use turboquant::bitpack::packed_len;
///
/// assert_eq!(packed_len(8), 1);
/// assert_eq!(packed_len(9), 2);
/// assert_eq!(packed_len(16), 2);
/// ```
pub fn packed_len(n: usize) -> usize {
    n.div_ceil(BITS_PER_BYTE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let signs = vec![1i8, -1, 1, 1, -1, -1, 1, -1, 1, 1];
        let packed = pack_bits(&signs);

        let mut unpacked = vec![0i8; signs.len()];
        unpack_bits(&packed, signs.len(), &mut unpacked);

        assert_eq!(signs, unpacked);
    }

    #[test]
    fn test_pack_bits_into() {
        let signs = vec![1i8, -1, 1, 1, -1, -1, 1, -1];
        let mut packed = [0u8; 1];
        pack_bits_into(&signs, &mut packed);

        assert_eq!(packed[0], 0b01001101);
    }

    #[test]
    fn test_unpack_bit() {
        let packed = vec![0b01001101];

        assert_eq!(unpack_bit(&packed, 0), 1);
        assert_eq!(unpack_bit(&packed, 1), -1);
        assert_eq!(unpack_bit(&packed, 2), 1);
        assert_eq!(unpack_bit(&packed, 3), 1);
        assert_eq!(unpack_bit(&packed, 4), -1);
        assert_eq!(unpack_bit(&packed, 5), -1);
        assert_eq!(unpack_bit(&packed, 6), 1);
        assert_eq!(unpack_bit(&packed, 7), -1);
    }

    #[test]
    fn test_packed_len() {
        assert_eq!(packed_len(8), 1);
        assert_eq!(packed_len(9), 2);
        assert_eq!(packed_len(16), 2);
        assert_eq!(packed_len(17), 3);
    }
}
