use anyhow::{Result, ensure};

use super::{pc2snes, snes2pc};

/// SNES free regions in bank $83 used for room FX data (same as mosaic retiling).
pub const FX_ALLOC_SNES_REGIONS: &[(usize, usize)] = &[
    (0x838000, 0x8388FC),
    (0x839AC2, 0x83A0A4),
    (0x83A0D4, 0x83A18A),
    (0x83F000, 0x840000),
];
pub const FX_BYTES_PER_ENTRY: usize = 16;

pub struct FxAllocator {
    next_addr: usize,
    region_idx: usize,
}

impl FxAllocator {
    pub fn new() -> Self {
        let (start, _) = FX_ALLOC_SNES_REGIONS[0];
        Self {
            next_addr: snes2pc(start),
            region_idx: 0,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Result<usize> {
        loop {
            let (_, end_snes) = FX_ALLOC_SNES_REGIONS[self.region_idx];
            let end_pc = snes2pc(end_snes);
            if self.next_addr + size <= end_pc {
                let addr = self.next_addr;
                self.next_addr += size;
                return Ok(addr);
            }
            self.region_idx += 1;
            ensure!(
                self.region_idx < FX_ALLOC_SNES_REGIONS.len(),
                "Ran out of bank $83 space for room FX"
            );
            let (start, _) = FX_ALLOC_SNES_REGIONS[self.region_idx];
            self.next_addr = snes2pc(start);
        }
    }
}

pub fn write_room_fx(
    rom: &mut super::Rom,
    fx_allocator: &mut FxAllocator,
    room_ptr: usize,
    fx_bytes: &[u8],
) -> Result<()> {
    use super::get_room_state_ptrs;

    ensure!(
        fx_bytes.len() == FX_BYTES_PER_ENTRY,
        "FX data must be {FX_BYTES_PER_ENTRY} bytes for room {room_ptr:x}"
    );
    let state_ptrs = get_room_state_ptrs(rom, room_ptr)?;
    ensure!(!state_ptrs.is_empty(), "Room {room_ptr:x} has no states");

    let fx_addr = fx_allocator.allocate(FX_BYTES_PER_ENTRY + 2)?;
    rom.write_n(fx_addr, fx_bytes)?;
    rom.write_u16(fx_addr + FX_BYTES_PER_ENTRY, 0xFFFF)?;
    let fx_ptr = (pc2snes(fx_addr) & 0xFFFF) as isize;
    for (_, state_ptr) in &state_ptrs {
        rom.write_u16(state_ptr + 6, fx_ptr)?;
    }
    Ok(())
}

pub fn read_default_fx(rom: &super::Rom, room_ptr: usize) -> Result<Vec<u8>> {
    use anyhow::{Context, bail};

    use super::get_room_state_ptrs;

    let state_ptrs = get_room_state_ptrs(rom, room_ptr)?;
    let (_, state_ptr) = state_ptrs.last().context("No standard state")?;
    let fx_ptr = rom.read_u16(state_ptr + 6)? as usize;
    if fx_ptr == 0 || fx_ptr == 0xFFFF {
        bail!("Room {room_ptr:x} has no FX pointer");
    }
    let fx_pc = snes2pc(0x830000 + fx_ptr);
    let mut fx = vec![];
    for i in 0..4 {
        let offset = fx_pc + i * FX_BYTES_PER_ENTRY;
        let door = rom.read_u16(offset)?;
        if door == 0xFFFF {
            break;
        }
        for j in 0..FX_BYTES_PER_ENTRY {
            fx.push(rom.read_u8(offset + j)? as u8);
        }
        if door == 0 {
            return Ok(fx);
        }
    }
    if fx.len() >= FX_BYTES_PER_ENTRY {
        Ok(fx[..FX_BYTES_PER_ENTRY].to_vec())
    } else {
        bail!("Could not find default FX entry for room {room_ptr:x}");
    }
}
