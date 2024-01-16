use anyhow::{bail, Result};

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
        let mut decoder = BPSPatchDecoder::new(data);
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

struct BPSPatchDecoder {
    patch_bytes: Vec<u8>,
    patch_pos: usize,
    output_pos: usize,
    src_pos: usize,
    dst_pos: usize,
    blocks: Vec<BPSBlock>,
}

impl BPSPatchDecoder {
    fn new(patch_bytes: Vec<u8>) -> Self {
        BPSPatchDecoder {
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


// struct BPSPatchEncoder {
//     source_prefix_tree: PrefixTree,
//     patch_bytes: Vec<u8>,
//     src_pos: usize,
//     dst_pos: usize,
//     input_pos: usize,
// }

// struct BPSPatch