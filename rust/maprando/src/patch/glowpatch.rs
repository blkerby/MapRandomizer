use crate::patch::Rom;
use anyhow::{bail, ensure, Result};

pub enum GlowPatchSection {
    Direct {
        offset: usize,
        data: Vec<u8>,
    },
    Indirect {
        offset: usize,
        read_length: u8,
        sections: Vec<GlowPatchSection>,
    },
}

impl GlowPatchSection {
    pub fn write(&self, rom: &mut Rom, base_offset: usize) -> Result<()> {
        match self {
            Self::Direct { offset, data } => {
                let total_offset = offset + base_offset;
                rom.write_n(total_offset, &data)?;
                Ok(())
            },
            Self::Indirect { offset, read_length, sections } => {
                let total_offset = offset + base_offset;
                let addr_data = rom
                    .read_n(total_offset, *read_length as usize)?
                    .to_vec();
                let addr = as_usize_le(&addr_data) as usize;
                for i in 0..sections.len() {
                    sections[i].write(rom, addr)?;
                }
                Ok(())
            },
        }
    }
}

fn as_usize_le(array: &[u8]) -> usize {
    let mut value: usize = 0;
    for i in 0..array.len() {
        value |= (array[i] as usize) << (8 * i);
    }
    return value;
}

struct GlowPatchReader {
    data: Vec<u8>,
    pos: usize,
}

impl GlowPatchReader {
    fn new(data: &Vec<u8>) -> Self {
        return GlowPatchReader {
            data: data.clone(),
            pos: 0,
        };
    }

    fn read_u8(&mut self) -> Result<u8> {
        ensure!(
            self.pos + 1 <= self.data.len(),
            "read_u8 address out of bounds"
        );
        let value = self.data[self.pos];
        self.pos += 1;
        Ok(value)
    }

    fn read_n(&mut self, n: usize) -> Result<&[u8]> {
        ensure!(
            self.pos + n <= self.data.len(),
            "read_n address out of bounds"
        );
        let value = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(value)
    }

    fn read_varint(&mut self) -> Result<usize> {
        let len = self.read_u8()? as usize;
        Ok(as_usize_le(self.read_n(len)?))
    }
}

pub fn parse_glowpatch(patch: &Vec<u8>) -> Result<Vec<GlowPatchSection>> {
    let mut reader = GlowPatchReader::new(patch);
    let len = reader.read_varint()?;
    let mut sections: Vec<GlowPatchSection> = vec![];

    for _i in 0..len {
        sections.push(parse_section(&mut reader)?);
    }

    Ok(sections)
}

fn parse_section(patch: &mut GlowPatchReader) -> Result<GlowPatchSection> {
    let kind = patch.read_u8()?;
    let section: GlowPatchSection = match kind {
        0u8 => parse_direct_section(patch)?,
        1u8 => parse_indirect_section(patch)?,
        _ => bail!(
            "Unexpected patch section kind header: {} at position: {:X}",
            kind,
            patch.pos - 1
        ),
    };

    Ok(section)
}

fn parse_direct_section(patch: &mut GlowPatchReader) -> Result<GlowPatchSection> {
    let offset = patch.read_varint()?;
    let len = patch.read_varint()?;
    let data = patch.read_n(len)?;
    let section = GlowPatchSection::Direct {
        offset: offset,
        data: data.to_vec(),
    };
    Ok(section)
}

fn parse_indirect_section(patch: &mut GlowPatchReader) -> Result<GlowPatchSection> {
    let offset = patch.read_varint()?;
    let read_length = patch.read_u8()?;
    let len = patch.read_varint()?;
    let mut sections: Vec<GlowPatchSection> = vec![];
    for _i in 0..len {
        sections.push(parse_section(patch)?);
    }

    let section = GlowPatchSection::Indirect {
        offset: offset,
        read_length: read_length,
        sections: sections,
    };

    Ok(section)
}
