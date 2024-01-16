use anyhow::{bail, Result};

#[derive(Debug)]
enum BPSBlock {
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

    fn apply_byte(&self, source: &[u8], output: &mut [u8], idx: usize, b: u8) {
        // With this method of applying BPS patches, we are careful to only apply changes to bytes that different from the source.
        // This is important because otherwise when we layer multiple patches on top of each other, they could overwrite each other
        // even if they are disjoint, because the Flips encoder that we're using does not guarantee that unchanged bytes won't 
        // be touched by the patch.
        if b != source[idx] {
            output[idx] = b;
        }
    }

    pub fn apply(&self, source: &[u8], output: &mut [u8]) {
        for block in &self.blocks {
            match block {
                &BPSBlock::SourceCopy { src_start, dst_start, length } => {
                    if src_start == dst_start {
                        // Skip copying over from the source to the destination, since it has no change.
                        // This allows us to efficiently and correctly apply hundreds of small patches on top of each other.
                    } else {
                        for i in 0..length {
                            self.apply_byte(source, output, dst_start + i, source[src_start + i]);
                        }
                    }
                },
                &BPSBlock::TargetCopy { src_start, dst_start, length } => {
                    for i in 0..length {
                        self.apply_byte(source, output, dst_start + i, output[src_start + i]);
                    }
                },
                BPSBlock::Data { dst_start, data } => {
                    for i in 0..data.len() {
                        self.apply_byte(source, output, dst_start + i, data[i]);
                    }
                }
            }           
        }
    }

    pub fn get_modified_ranges(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = vec![];
        for block in &self.blocks {
            match block {
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
            self.blocks.push(block);
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
                let block = BPSBlock::SourceCopy { 
                    src_start: self.output_pos,
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

