use super::Allocator;
use crate::{
    game_data::{GameData, AreaIdx, TilesetIdx, RoomPtr},
    patch::{snes2pc, pc2snes, Rom, get_room_state_ptrs},
};
use anyhow::{bail, Result};
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

pub fn apply_retiling(rom: &mut Rom, game_data: &GameData) -> Result<()> {
    let retiled_theme_data = game_data.retiled_theme_data.as_ref().unwrap();

    let new_tile_table_snes = 0x8FF900;
    let new_tile_pointers_snes = 0x8FFD00;
    let tile_pointers_free_space_end = 0x8FFE00;

    let mut allocator = Allocator::new(vec![
        // (snes2pc(0xBAC629), snes2pc(0xC2C2BB)), // Vanilla tile GFX, tilemaps, and palettes, which we overwrite
        (snes2pc(0xBAC629), snes2pc(0xCF8000)), // Vanilla tile GFX, tilemaps, palettes, and level data, which we overwrite
        (snes2pc(0xE18000), snes2pc(0xE20000)),
        (snes2pc(0xE2B000), snes2pc(0xE30000)),
        (snes2pc(0xE3B000), snes2pc(0xE40000)),
        (snes2pc(0xE4B000), snes2pc(0xE50000)),
        (snes2pc(0xE5B000), snes2pc(0xE60000)),
        (snes2pc(0xE6B000), snes2pc(0xE70000)),
        (snes2pc(0xE7B000), snes2pc(0xE80000)),
        (snes2pc(0xE99000), snes2pc(0xEA0000)),
        (snes2pc(0xEA8000), snes2pc(0xFFFFFF)),
    ]);

    let mut pal_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut gfx8_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut gfx16_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut level_data_map: HashMap<Vec<u8>, usize> = HashMap::new();

    // let mut tile_table: Vec<u8> = rom.read_n(snes2pc(0x8FE6A2), next_tile_idx * 9)?.to_vec();
    let base_theme = &retiled_theme_data.themes["Base"];
    let num_tilesets = base_theme.sce_tilesets.keys().max().unwrap() + 1;
    assert!(new_tile_pointers_snes + 2 * num_tilesets <= tile_pointers_free_space_end);

    for (&tileset_idx, tileset) in base_theme.sce_tilesets.iter() {
        if tileset_idx >= 0x0F && tileset_idx <= 0x14 {
            // Skip Ceres tilesets
            continue;
        }
        let compressed_pal = &tileset.compressed_palette;
        let pal_addr = match pal_map.entry(compressed_pal.clone()) {
            Entry::Occupied(x) => *x.get(),
            Entry::Vacant(view) => {
                let addr = allocator.allocate(compressed_pal.len())?;
                view.insert(addr);
                addr
            }
        };
        rom.write_n(pal_addr, &compressed_pal)?;

        let compressed_gfx8 = &tileset.compressed_gfx8x8;
        let gfx8_addr = match gfx8_map.entry(compressed_gfx8.clone()) {
            Entry::Occupied(x) => *x.get(),
            Entry::Vacant(view) => {
                let addr = allocator.allocate(compressed_gfx8.len())?;
                view.insert(addr);
                addr
            }
        };
        rom.write_n(gfx8_addr, &compressed_gfx8)?;

        let compressed_gfx16 = &tileset.compressed_gfx16x16;
        let gfx16_addr = match gfx16_map.entry(compressed_gfx16.clone()) {
            Entry::Occupied(x) => *x.get(),
            Entry::Vacant(view) => {
                let addr = allocator.allocate(compressed_gfx16.len())?;
                view.insert(addr);
                addr
            }
        };
        rom.write_n(gfx16_addr, &compressed_gfx16)?;

        let tile_table_entry_addr = new_tile_table_snes + 9 * tileset_idx;
        rom.write_u24(snes2pc(tile_table_entry_addr), pc2snes(gfx16_addr) as isize)?;
        rom.write_u24(snes2pc(tile_table_entry_addr + 3), pc2snes(gfx8_addr) as isize)?;
        rom.write_u24(snes2pc(tile_table_entry_addr + 6), pc2snes(pal_addr) as isize)?;
        rom.write_u16(snes2pc(new_tile_pointers_snes + 2 * tileset_idx), 
        (tile_table_entry_addr & 0xFFFF) as isize)?;
    }
    println!("Number of tilesets: {}", num_tilesets);
    println!("Number of unique pal: {}", pal_map.len());
    println!("Number of unique gfx8: {}", gfx8_map.len());
    println!("Number of unique gfx16: {}", gfx16_map.len());
    assert!(num_tilesets * 9  <= new_tile_pointers_snes - new_tile_table_snes);

    rom.write_u16(
        snes2pc(0x82DF03),
        (new_tile_pointers_snes & 0xFFFF) as isize,
    )?;

    let mut room_ptr_map: HashMap<(usize, usize), RoomPtr> = HashMap::new();
    let mut bg_ptr_map: HashMap<(usize, usize, usize), u16> = HashMap::new();
    for &room_ptr in game_data.room_ptr_by_id.values() {
        let area = rom.read_u8(room_ptr + 1)?;
        let index = rom.read_u8(room_ptr)?;
        room_ptr_map.insert((area as usize, index as usize), room_ptr);

        let state_ptrs = get_room_state_ptrs(&rom, room_ptr)?;
        for (state_idx, (_event_ptr, state_ptr)) in state_ptrs.iter().enumerate() {
            let bg_ptr = rom.read_u16(state_ptr + 22)? as u16;
            bg_ptr_map.insert((area as usize, index as usize, state_idx), bg_ptr);
        }
    }

    let theme = &retiled_theme_data.themes["OuterCrateria"];
    for room in &theme.rooms {
        let room_ptr_opt = room_ptr_map.get(&(room.area, room.index));
        if room_ptr_opt.is_none() {
            continue;
        }
        let room_ptr = *room_ptr_opt.unwrap();
        let state_ptrs = get_room_state_ptrs(&rom, room_ptr)?;
        if state_ptrs.len() != room.states.len() {
            panic!("Number of states in {} ({}) does not match vanilla ({})", room.path, room.states.len(), state_ptrs.len());
        }
        for state_idx in 0..state_ptrs.len() {
            let state = &room.states[state_idx];
            let (_event_ptr, state_ptr) = state_ptrs[state_idx];

            rom.write_u8(state_ptr + 3, state.tileset_idx as isize)?;
            if let Some(bg_ref) = &state.bgdata_reference {
                let bg_ptr = bg_ptr_map[&(bg_ref.room_area, bg_ref.room_index, bg_ref.room_state_index)];
                rom.write_u16(state_ptr + 22, bg_ptr as isize)?;
            } else {
                rom.write_u16(state_ptr + 22, 0)?;
            }

            let level_data = &state.compressed_level_data;
            let level_data_addr = match level_data_map.entry(level_data.clone()) {
                Entry::Occupied(x) => *x.get(),
                Entry::Vacant(view) => {
                    let addr = allocator.allocate(level_data.len())?;
                    view.insert(addr);
                    addr
                }
            };
            rom.write_n(level_data_addr, &level_data)?;
    
            println!("{:x} {:x}", pc2snes(level_data_addr), state_ptr);
            rom.write_u24(state_ptr, pc2snes(level_data_addr) as isize)?;
        }
    }

    Ok(())
}
