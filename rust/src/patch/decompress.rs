use super::Rom;
use anyhow::{Result, bail};

pub fn decompress(rom: &Rom, mut addr: usize) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();
    loop {
        let byte = rom.read_u8(addr)?;
        addr += 1;
        if byte == 0xFF {
            return Ok(out);
        }
        let mut block_type = byte >> 5;
        let size: usize;
        if block_type != 7 {
            size = ((byte & 0x1F) + 1) as usize;
        } else {
            size = (((byte & 3) << 8 | rom.read_u8(addr)?) + 1) as usize;
            addr += 1;
            block_type = (byte >> 2) & 7;
        }
        // println!("{addr}: len={0}, type={block_type}, size={size}", out.len());

        match block_type {
            0 => {
                // Raw block
                out.extend(rom.read_n(addr, size)?);
                addr += size;
            },
            1 => {
                // Byte-level RLE block
                let value = rom.read_u8(addr)? as u8;
                addr += 1;
                out.extend(&vec![value; size]);                
            },
            2 => {
                // Word-level RLE block
                let b0 = rom.read_u8(addr)? as u8;
                let b1 = rom.read_u8(addr + 1)? as u8;
                addr += 2;
                out.extend(&[b0, b1].repeat(size >> 1));
                if size & 1 == 1 {
                    out.push(b0);
                }
            },
            3 => {
                // Incrementing sequence
                let mut b = rom.read_u8(addr)? as u8;
                addr += 1;
                for _ in 0..size {
                    out.push(b);
                    b += 1;
                }
            },
            4 => {
                // Copy earlier output, with absolute offset:
                let offset = rom.read_u16(addr)? as usize;
                assert!(offset < out.len());
                addr += 2;
                for i in offset..(offset + size) {
                    out.push(out[i]);
                }
            },
            5 => {
                // Copy complement of earlier output, with absolute offset:
                let offset = rom.read_u16(addr)? as usize;
                assert!(offset < out.len());
                addr += 2;
                for i in offset..(offset + size) {
                    out.push(out[i] ^ 0xFF);
                }
            },
            6 => {
                // Copy earlier output, with relative offset:
                let rel = rom.read_u8(addr)? as usize;
                addr += 1;
                assert!(rel <= out.len());
                let offset = out.len() - rel;
                for i in offset..(offset + size) {
                    out.push(out[i]);
                }
            },
            7 => {
                // Copy complement of earlier output, with relative offset:
                let rel = rom.read_u8(addr)? as usize;
                addr += 1;
                assert!(rel <= out.len());
                let offset = out.len() - rel;
                for i in offset..(offset + size) {
                    out.push(out[i] ^ 0xFF);
                }
            },
            _ => {
                bail!("Unexpected/impossible block type: {block_type}");
            }
        }
    }
}
