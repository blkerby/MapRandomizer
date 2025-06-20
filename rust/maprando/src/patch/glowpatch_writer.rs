// Code to build the glow patch JSON ("reduced_flashing.json") is at https://github.com/ardnaxelarak/super_metroid_deflash

use crate::patch::Rom;
use anyhow::Result;
use maprando_game::glowpatch::{GlowPatch, GlowPatchSection};

pub fn write_glowpatch(rom: &mut Rom, patch: &GlowPatch) -> Result<()> {
    for section in &patch.sections {
        write_glowpatch_section(rom, section, 0)?;
    }
    Ok(())
}

fn write_glowpatch_section(
    rom: &mut Rom,
    section: &GlowPatchSection,
    base_offset: usize,
) -> Result<()> {
    match section {
        GlowPatchSection::Direct { offset, data } => {
            let total_offset = *offset as usize + base_offset;
            rom.write_n(total_offset, data)?;
            Ok(())
        }
        GlowPatchSection::Indirect {
            offset,
            read_length,
            sections,
        } => {
            let total_offset = *offset as usize + base_offset;
            let addr_data = rom.read_n(total_offset, *read_length as usize)?.to_vec();
            let addr = from_bytes_le(&addr_data) as usize;
            for subsection in sections {
                write_glowpatch_section(rom, subsection, addr)?;
            }
            Ok(())
        }
    }
}

fn from_bytes_le(array: &[u8]) -> u64 {
    let mut value: u64 = 0;
    for (i, &x) in array.iter().enumerate() {
        value |= (x as u64) << (8 * i);
    }
    value
}
