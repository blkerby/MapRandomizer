use anyhow::{Context, Result, bail, ensure};
use hashbrown::HashMap;
use log::info;

use maprando_game::{GameData, RoomPtr};

use super::{Rom, get_room_state_ptrs, pc2snes, snes2pc};
use crate::randomize::Randomization;

/// SNES free regions in bank $83 used for room FX data (same as mosaic retiling).
const FX_ALLOC_SNES_REGIONS: &[(usize, usize)] = &[
    (0x838000, 0x8388FC),
    (0x839AC2, 0x83A0A4),
    (0x83A0D4, 0x83A18A),
    (0x83F000, 0x840000),
];
const FX_BYTES_PER_ENTRY: usize = 16;

struct FxAllocator {
    next_addr: usize,
    region_idx: usize,
}

impl FxAllocator {
    fn new() -> Self {
        let (start, _) = FX_ALLOC_SNES_REGIONS[0];
        Self {
            next_addr: snes2pc(start),
            region_idx: 0,
        }
    }

    fn allocate(&mut self, size: usize) -> Result<usize> {
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
                "Ran out of bank $83 space for water FX"
            );
            let (start, _) = FX_ALLOC_SNES_REGIONS[self.region_idx];
            self.next_addr = snes2pc(start);
        }
    }
}

pub struct WaterFxPatcher<'a> {
    rom: &'a mut Rom,
    game_data: &'a GameData,
    randomization: &'a Randomization,
    fx_allocator: FxAllocator,
}

impl<'a> WaterFxPatcher<'a> {
    pub fn new(rom: &'a mut Rom, game_data: &'a GameData, randomization: &'a Randomization) -> Self {
        Self {
            rom,
            game_data,
            randomization,
            fx_allocator: FxAllocator::new(),
        }
    }

    pub fn apply(&mut self) -> Result<()> {
        if self.randomization.water_assignments.is_empty() {
            return Ok(());
        }

        let donor_templates: HashMap<usize, Vec<u8>> = self
            .randomization
            .water_assignments
            .values()
            .map(|a| a.donor_room_ptr)
            .collect::<hashbrown::HashSet<_>>()
            .into_iter()
            .map(|ptr| {
                let template = read_default_water_fx(self.rom, ptr)
                    .with_context(|| format!("Unable to read water FX template from room {ptr:x}"))?;
                Ok((ptr, template))
            })
            .collect::<Result<_>>()?;

        for (&room_id, assignment) in &self.randomization.water_assignments {
            let room_ptr = self.game_data.room_ptr_by_id[&room_id];
            if !self.randomization.map.room_mask[self.game_data.room_idx_by_id[&room_id]] {
                continue;
            }
            let template = &donor_templates[&assignment.donor_room_ptr];
            let room_idx = self.game_data.room_idx_by_id[&room_id];
            let surface = compute_liquid_surface(
                self.game_data.room_geometry[room_idx].map.len(),
                template,
            );
            let fx_bytes = build_water_fx_bytes(template, surface);
            self.write_room_water_fx(room_ptr, &fx_bytes)?;
            if self.randomization.water_assignments.len() <= 10 {
                info!(
                    "Applied water FX to room {} ({room_id}) at {room_ptr:x}, surface={surface:x}",
                    room_name(self.game_data, room_id)
                );
            }
        }
        if self.randomization.water_assignments.len() > 10 {
            info!(
                "Applied water FX to {} rooms",
                self.randomization.water_assignments.len()
            );
        }
        Ok(())
    }

    fn write_room_water_fx(&mut self, room_ptr: RoomPtr, fx_bytes: &[u8]) -> Result<()> {
        ensure!(
            fx_bytes.len() == FX_BYTES_PER_ENTRY,
            "Water FX data must be {FX_BYTES_PER_ENTRY} bytes for room {room_ptr:x}"
        );
        let state_ptrs = get_room_state_ptrs(self.rom, room_ptr)?;
        ensure!(!state_ptrs.is_empty(), "Room {room_ptr:x} has no states");

        let fx_addr = self.fx_allocator.allocate(FX_BYTES_PER_ENTRY + 2)?;
        self.rom.write_n(fx_addr, fx_bytes)?;
        self.rom.write_u16(fx_addr + FX_BYTES_PER_ENTRY, 0xFFFF)?;
        let fx_ptr = (pc2snes(fx_addr) & 0xFFFF) as isize;
        for (_, state_ptr) in &state_ptrs {
            self.rom.write_u16(state_ptr + 6, fx_ptr)?;
        }
        Ok(())
    }
}

fn room_name(game_data: &GameData, room_id: usize) -> String {
    game_data.room_json_map[&room_id]["name"]
        .as_str()
        .unwrap_or("?")
        .to_string()
}

fn read_default_water_fx(rom: &Rom, room_ptr: usize) -> Result<Vec<u8>> {
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

/// Liquid surface Y from the top of the room. Low values place the surface near the
/// ceiling, which fills the room with water (matches Fish Tank's `0x0010`).
const FULL_FLOOD_SURFACE_Y: u16 = 0x0010;

fn compute_liquid_surface(_room_screen_rows: usize, _template: &[u8]) -> u16 {
    FULL_FLOOD_SURFACE_Y
}

fn build_water_fx_bytes(template: &[u8], surface: u16) -> Vec<u8> {
    let mut fx = template.to_vec();
    if fx.len() < FX_BYTES_PER_ENTRY {
        fx.resize(FX_BYTES_PER_ENTRY, 0);
    }
    fx.truncate(FX_BYTES_PER_ENTRY);
    fx[0] = 0;
    fx[1] = 0;
    fx[2..4].copy_from_slice(&surface.to_le_bytes());
    fx[4..6].copy_from_slice(&surface.to_le_bytes());
    fx[9] = 6; // water FX type
    fx
}
