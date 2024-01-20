use anyhow::{bail, Result};

use super::suffix_tree::SuffixTree;

// Threshold for minimum matching chunk size (bytes) to encode a source copy block:
const SOURCE_MATCH_THRESHOLD: usize = 4;

#[derive(Debug)]
enum BPSBlock {
    Unchanged {
        dst_start: usize,
        length: usize,
    },
    SourceCopy {
        src_start: usize,
        dst_start: usize,
        length: usize,
    },
    TargetCopy {
        src_start: usize,
        dst_start: usize,
        length: usize,
    },
    Data {
        dst_start: usize,
        data: Vec<u8>
    },
}

#[derive(Debug)]
pub struct BPSPatch {
    blocks: Vec<BPSBlock>,
}

impl BPSPatch {
    pub fn new(data: Vec<u8>) -> Result<Self> {
        let mut decoder = BPSDecoder::new(data);
        decoder.decode()?;
        Ok(BPSPatch {
            blocks: decoder.blocks,
        })
    }

    pub fn apply(&self, source: &[u8], output: &mut [u8]) {
        for block in &self.blocks {
            match block {
                &BPSBlock::Unchanged { .. } => {
                    // These blocks won't be loaded and wouldn't do anything anyway.
                }
                &BPSBlock::SourceCopy { src_start, dst_start, length } => {
                    let src_slice = &source[src_start..(src_start+length)];
                    let dst_slice = &mut output[dst_start..(dst_start+length)];
                    dst_slice.copy_from_slice(src_slice);
                },
                &BPSBlock::TargetCopy { src_start, dst_start, length } => {
                    for i in 0..length {
                        output[dst_start + i] = output[src_start + i];
                    }
                },
                BPSBlock::Data { dst_start, data } => {
                    output[*dst_start..(*dst_start + data.len())].copy_from_slice(data);
                }
            }           
        }
    }

    // This function isn't used now, but could be useful later, so keeping it around.
    pub fn get_modified_ranges(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = vec![];
        for block in &self.blocks {
            match block {
                &BPSBlock::Unchanged { .. } => {}
                &BPSBlock::SourceCopy { src_start, dst_start, length } => {
                    if src_start != dst_start {
                        out.push((dst_start, dst_start + length));
                    }
                }
                &BPSBlock::TargetCopy { dst_start, length, .. } => {
                    out.push((dst_start, dst_start + length));
                }
                BPSBlock::Data { dst_start, data } => {
                    out.push((*dst_start, *dst_start + data.len()));
                }
            }
        }
        out
    }
}

struct BPSDecoder {
    patch_bytes: Vec<u8>,
    patch_pos: usize,
    output_pos: usize,
    src_pos: usize,
    dst_pos: usize,
    blocks: Vec<BPSBlock>,
}

impl BPSDecoder {
    fn new(patch_bytes: Vec<u8>) -> Self {
        BPSDecoder {
            patch_bytes,
            patch_pos: 0,
            output_pos: 0,
            src_pos: 0,
            dst_pos: 0,
            blocks: vec![],
        }
    }

    fn decode(&mut self) -> Result<()> {
        assert!(self.read_n(4)? == "BPS1".as_bytes());
        let _src_size = self.decode_number()?;
        let _dst_size = self.decode_number()?;
        let metadata_size = self.decode_number()?;
        self.patch_pos += metadata_size;
        while self.patch_pos < self.patch_bytes.len() - 12 {
            let block = self.decode_block()?;
            if let BPSBlock::Unchanged { .. } = block {
                // Skip blocks that do not change the output.
            } else {
                self.blocks.push(block);
            }
        }
        assert!(self.patch_pos == self.patch_bytes.len() - 12);
        // Ignore the checksums at the end of the file.
        Ok(())
    }

    fn read(&mut self) -> Result<u8> {
        if self.patch_pos >= self.patch_bytes.len() {
            bail!("BPS read past end of data");
        }
        let x = self.patch_bytes[self.patch_pos];
        self.patch_pos += 1;
        Ok(x)
    }

    fn read_n(&mut self, n: usize) -> Result<Vec<u8>> {
        if self.patch_pos + n > self.patch_bytes.len() {
            bail!("BPS read_n past end of data");
        }
        let out = self.patch_bytes[self.patch_pos..(self.patch_pos + n)].to_vec();
        self.patch_pos += n;
        Ok(out)
    }

    fn decode_block(&mut self) -> Result<BPSBlock> {
        let cmd = self.decode_number()?;
        let action = cmd & 3;
        let length = (cmd >> 2) + 1;
        match action {
            0 => {
                let block = BPSBlock::Unchanged {
                    dst_start: self.output_pos,
                    length, 
                };
                self.output_pos += length;
                return Ok(block);
            },
            1 => {
                let block = BPSBlock::Data {
                    dst_start: self.output_pos,
                    data: self.read_n(length)?,
                };
                self.output_pos += length;
                return Ok(block);
            },
            2 => {
                let raw_offset = self.decode_number()?;
                let offset_neg = raw_offset & 1 == 1;
                let offset_abs = (raw_offset >> 1) as isize;
                let offset = if offset_neg { -offset_abs } else { offset_abs };
                self.src_pos = (self.src_pos as isize + offset) as usize;
                let block = BPSBlock::SourceCopy {
                    src_start: self.src_pos,
                    dst_start: self.output_pos,
                    length,
                };
                self.src_pos += length;
                self.output_pos += length;
                return Ok(block);
            },
            3 => {
                let raw_offset = self.decode_number()?;
                let offset_neg = raw_offset & 1 == 1;
                let offset_abs = (raw_offset >> 1) as isize;
                let offset = if offset_neg { -offset_abs } else { offset_abs };
                self.dst_pos = (self.dst_pos as isize + offset) as usize;
                let block = BPSBlock::TargetCopy {
                    src_start: self.dst_pos,
                    dst_start: self.output_pos,
                    length,
                };
                self.dst_pos += length;
                self.output_pos += length;
                return Ok(block);
            },
            _ => panic!("Unexpected BPS action: {}", action)
        }
    }

    fn decode_number(&mut self) -> Result<usize> {
        let mut out: usize = 0;
        let mut shift: usize = 1;
        for _ in 0..10 {
            let x = self.read()?;
            out += ((x & 0x7f) as usize) * shift;
            if x & 0x80 != 0 {
                return Ok(out);
            }
            shift <<= 7;
            out += shift;
        }
        bail!("invalid BPS");
    }
}


pub struct BPSEncoder<'a> {
    source_suffix_tree: &'a SuffixTree,
    target: &'a [u8],
    modified_ranges: &'a [(usize, usize)],
    pub patch_bytes: Vec<u8>,
    src_pos: usize,
    input_pos: usize,
    count_data_bytes: usize,
    count_copy_bytes: usize,
}

fn compute_crc32(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

// This is a simplified BPS encoder that is not overly concerned with minimizing patch size,
// mainly just with capturing dependencies on the source data (even when relocated) to avoid 
// source data being copied into the patch. The key difference from other encoders (and the reason
// we rolled a custom one here) is that this encoder enforces that only bytes that differ from the source
// will get touched by the patch; this is important in order to ensure that we can correctly 
// and efficiently layer multiple patches on top of each other (assuming they affect disjoint sets of 
// bytes). For a similar reason, this encoder also doesn't create blocks that copy from the target
// (i.e. previously output data).
impl<'a> BPSEncoder<'a> {
    pub fn new(source_suffix_tree: &'a SuffixTree, target: &'a [u8], modified_ranges: &'a [(usize, usize)]) -> Self {
        Self {
            source_suffix_tree,
            target,
            modified_ranges,
            patch_bytes: vec![],
            src_pos: 0,
            input_pos: 0,
            count_copy_bytes: 0,
            count_data_bytes: 0,
        }       
    }

    pub fn encode(&mut self) {
        self.write_n("BPS1".as_bytes());
        self.encode_number(self.source_suffix_tree.data.len());
        self.encode_number(self.target.len());
        self.encode_number(0); // metadata size
        for r in self.modified_ranges {
            self.encode_range(r.0, r.1);
        }
        if self.input_pos < self.source_suffix_tree.data.len() {
            self.encode_unchanged(self.source_suffix_tree.data.len() - self.input_pos);
        }
        self.write_n(&compute_crc32(&self.source_suffix_tree.data).to_le_bytes());
        self.write_n(&compute_crc32(&self.target).to_le_bytes());
        self.write_n(&compute_crc32(&self.patch_bytes).to_le_bytes());
        self.check_encoding();
        // let count_total_bytes = self.count_copy_bytes + self.count_data_bytes;
        // let data_frac = (self.count_data_bytes as f32) / (count_total_bytes as f32 + 1e-12);
        // println!("{}/{} data bytes ({:.1}%)", self.count_data_bytes, count_total_bytes, data_frac * 100.0);
    }

    fn encode_range(&mut self, mut start_addr: usize, end_addr: usize) {
        // println!("range: {} {}", start_addr, end_addr);
        // Unchanged block:
        if self.input_pos < start_addr {
            let length = start_addr - self.input_pos;
            self.encode_unchanged(length);
            self.input_pos += length;
        }

        // Data and source copy blocks:
        while start_addr < end_addr {
            let (source_start, match_length) = self.source_suffix_tree.find_longest_prefix(&self.target[start_addr..end_addr]);
            // println!("start_addr={start_addr}, match_length={match_length}, end_addr={end_addr}");
            if match_length as usize >= SOURCE_MATCH_THRESHOLD {
                if start_addr > self.input_pos {
                    self.encode_data(&self.target[self.input_pos..start_addr]);
                }
                self.encode_source_copy(source_start as usize, match_length as usize);
                self.input_pos = start_addr + match_length as usize;
                start_addr = start_addr + match_length as usize;
            } else {
                start_addr += 1;
            }    
        }
        if end_addr > self.input_pos {
            self.encode_data(&self.target[self.input_pos..end_addr]);
            self.input_pos = end_addr;
        }
    }

    fn encode_block_header(&mut self, action: usize, length: usize) {
        let x = action | ((length - 1) << 2);
        self.encode_number(x);
    }

    fn encode_unchanged(&mut self, length: usize) {
        self.encode_block_header(0, length);
    }

    fn encode_data(&mut self, data: &[u8]) {
        // println!("data: dst={}, length={}: {:?}", self.input_pos, data.len(), data);
        self.encode_block_header(1, data.len());
        self.write_n(data);
        self.count_data_bytes += data.len();
    }

    fn encode_source_copy(&mut self, idx: usize, length: usize) {
        // println!("source copy: dst={}, src={}, length={}", self.input_pos, idx, length);
        self.encode_block_header(2, length);
        let relative_idx = (idx as isize) - (self.src_pos as isize);
        if relative_idx < 0 {
            self.encode_number(1 | (((-relative_idx) as usize) << 1));
        } else {
            self.encode_number((relative_idx as usize) << 1);
        }
        self.src_pos = idx + length;
        self.count_copy_bytes += length;
    }

    fn write(&mut self, b: u8) {
        self.patch_bytes.push(b);
    }

    fn write_n(&mut self, data: &[u8]) {
        self.patch_bytes.extend(data);
    }

    fn encode_number(&mut self, mut x: usize) {
        for _ in 0..10 {
            let b = (x & 0x7f) as u8;
            x >>= 7;
            if x == 0 {
              self.write(0x80 | b);
              break;
            }
            self.write(b);
            x -= 1;
        }
    }

    fn check_encoding(&self) {
        // Check decoding using our decoder:
        let patch = BPSPatch::new(self.patch_bytes.clone()).unwrap();
        // println!("{:?}", patch);
        let mut output = self.source_suffix_tree.data.clone();
        patch.apply(&self.source_suffix_tree.data, &mut output);
        // for i in 0..self.target.len() {
        //     if self.target[i] != output[i] {
        //         println!("at {}: source={}, target={}, output={}", i, self.source_suffix_tree.data[i], self.target[i], output[i]);
        //     }
        // }
        assert!(&self.target == &output);

        // // Check decoding using Flips:
        // let patch = flips::BpsPatch::new(&self.patch_bytes);
        // let output = patch.apply(&self.source_suffix_tree.data).unwrap();
        // assert_eq!(output.as_bytes(), self.target);
    }
}