use std::cmp::min;

const BLOCK_TYPE_RAW: u8 = 0;
const BLOCK_TYPE_BYTE_RLE: u8 = 1;

fn encode_block_header(size: usize, block_type: u8, out: &mut Vec<u8>) {
    assert!(size >= 1);
    assert!(size <= 1024);
    let size1 = size - 1;
    if size1 <= 31 {
        out.push(size1 as u8 | (block_type << 5));
    } else {
        out.push(0xE0 | (block_type << 2) | ((size1 >> 8) as u8));
        out.push((size1 & 0xFF) as u8);
    }
}

fn encode_raw_block(data: &[u8], out: &mut Vec<u8>) {
    for i in 0..((data.len() + 1023) / 1024) {
        let block = &data[(i * 1024)..min((i + 1) * 1024, data.len())];
        encode_block_header(block.len(), BLOCK_TYPE_RAW, out);
        out.extend(block);
    }
}

fn encode_rle_block(value: u8, count: usize, out: &mut Vec<u8>) {
    for i in 0..((count + 1023) / 1024) {
        let end = min(i * 1024 + 1024, count);
        let size = end - i * 1024;
        encode_block_header(size, BLOCK_TYPE_BYTE_RLE, out);
        out.push(value)
    }
}

fn encode_block(data: &[u8], value: u8, rle_count: usize, out: &mut Vec<u8>) {
    if data.len() >= 1 {
        encode_raw_block(data, out);
    }
    encode_rle_block(value, rle_count, out);
}

// Compress data into format used by Super Metroid for graphics, tilemaps, etc.
// This is a simple encoder that only uses two block types: raw blocks and byte-level RLE (run length encoding).
// Supporting other block types could allow for a smaller compressed output.
pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut block_data: Vec<u8> = Vec::new();
    let mut prev: isize = -1; // Previous byte value (-1 used just as a dummy value at the beginning)
    let mut rle_count: usize = 0;
    for &x in data {
        if x as isize == prev {
            rle_count += 1;
        } else {
            if rle_count >= 3 {
                encode_block(&block_data, prev as u8, rle_count, &mut out);
                block_data.clear();
            } else {
                for _ in 0..rle_count {
                    block_data.push(prev as u8);
                }
            }
            rle_count = 1;
            prev = x as isize;
        }
    }
    encode_block(&block_data, prev as u8, rle_count, &mut out);
    out.push(0xFF);
    out
}
