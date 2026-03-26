pub const BITS_PER_BYTE: usize = 8;

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

#[inline]
pub fn pack_bits_into(signs: &[i8], packed: &mut [u8]) {
    let packed_len = signs.len().div_ceil(BITS_PER_BYTE);
    debug_assert!(packed.len() >= packed_len);

    for byte in packed.iter_mut() {
        *byte = 0;
    }

    for (i, &sign) in signs.iter().enumerate() {
        if sign > 0 {
            packed[i / BITS_PER_BYTE] |= 1 << (i % BITS_PER_BYTE);
        }
    }
}

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
