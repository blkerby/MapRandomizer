use std::{alloc::GlobalAlloc, path::Path};

use super::Allocator;
use crate::{
    game_data::{GameData, AreaIdx, TilesetIdx, RoomPtr, smart_xml::{Layer2Type}, themed_retiling::{FX1, FX1Reference}, DoorPtr},
    patch::{snes2pc, pc2snes, Rom, get_room_state_ptrs, apply_ips_patch},
};
use anyhow::{bail, Result, Context};
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

fn get_fx_data(fx_list: &[FX1], fx_door_ptr_map: &HashMap<FX1Reference, DoorPtr>) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = vec![];
    for fx in fx_list {
        if fx.fx1_data.default {
            out.extend(0u16.to_le_bytes());
        } else {
            let fx_ref = fx.fx1_reference.context("Missing door info for non-default FX")?;
            let door_ptr = *fx_door_ptr_map.get(&fx_ref).context("Unrecognized FX door info")?;
            out.extend((door_ptr as u16).to_le_bytes());
        }
        out.extend((fx.fx1_data.surfacestart as u16).to_le_bytes());
        out.extend((fx.fx1_data.surfacenew as u16).to_le_bytes());
        out.extend((fx.fx1_data.surfacespeed as u16).to_le_bytes());
        out.extend([fx.fx1_data.surfacedelay as u8]);
        out.extend([fx.fx1_data.type_ as u8]);
        out.extend([fx.fx1_data.transparency1_a as u8]);
        out.extend([fx.fx1_data.transparency2_b as u8]);
        out.extend([fx.fx1_data.liquidflags_c as u8]);
        out.extend([fx.fx1_data.paletteflags as u8]);
        out.extend([fx.fx1_data.animationflags as u8]);
        out.extend([fx.fx1_data.paletteblend as u8]);
    }
    if out.len() == 0 {
        out.extend(vec![0xFF, 0xFF]);
    }
    Ok(out)
}

pub fn apply_retiling(rom: &mut Rom, game_data: &GameData, theme_name: &str) -> Result<()> {
    // "theme" is just a temporary argument, to hard-code a constant theme through the whole game.
    // It will be eliminated once we have all the themes are are ready to assign them based on area.
    let patch_names = vec![
        "Scrolling Sky v1.5",
        "Area FX",
        "Bowling",
    ];
    for name in &patch_names {
        let patch_path_str = format!("../patches/ips/{}.ips", name);
        apply_ips_patch(rom, Path::new(&patch_path_str))?;    
    }

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
        (snes2pc(0xEA8000), snes2pc(0xF80000)),
    ]);

    let mut fx_allocator = Allocator::new(vec![
        (snes2pc(0x838000), snes2pc(0x8388FC)),
        (snes2pc(0x839AC2), snes2pc(0x83A18A)),
        (snes2pc(0x83F000), snes2pc(0x840000)),
    ]);

    let mut pal_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut gfx8_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut gfx16_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut level_data_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut fx_data_map: HashMap<Vec<u8>, usize> = HashMap::new();

    let base_theme = &retiled_theme_data.themes["Base"];
    let num_tilesets = base_theme.sce_tilesets.keys().max().unwrap() + 1;
    assert!(new_tile_pointers_snes + 2 * num_tilesets <= tile_pointers_free_space_end);

    // Write CRE 8x8 tile graphics and update pointers to it:
    let cre_gfx8x8_addr = allocator.allocate(retiled_theme_data.cre_tileset.compressed_gfx8x8.len())?;
    rom.write_n(cre_gfx8x8_addr, &retiled_theme_data.cre_tileset.compressed_gfx8x8)?;
    rom.write_u8(snes2pc(0x82E415), (pc2snes(cre_gfx8x8_addr) >> 16) as isize)?;
    rom.write_u16(snes2pc(0x82E419), (pc2snes(cre_gfx8x8_addr) & 0xFFFF) as isize)?;
    rom.write_u8(snes2pc(0x82E797), (pc2snes(cre_gfx8x8_addr) >> 16) as isize)?;
    rom.write_u16(snes2pc(0x82E79B), (pc2snes(cre_gfx8x8_addr) & 0xFFFF) as isize)?;
    
    // Write CRE 16x16 tile graphics and update pointers to it:
    let cre_gfx16x16_addr = allocator.allocate(retiled_theme_data.cre_tileset.compressed_gfx16x16.len())?;
    rom.write_n(cre_gfx16x16_addr, &retiled_theme_data.cre_tileset.compressed_gfx16x16)?;
    rom.write_u8(snes2pc(0x82E83D), (pc2snes(cre_gfx16x16_addr) >> 16) as isize)?;
    rom.write_u16(snes2pc(0x82E841), (pc2snes(cre_gfx16x16_addr) & 0xFFFF) as isize)?;
    rom.write_u8(snes2pc(0x82EAED), (pc2snes(cre_gfx16x16_addr) >> 16) as isize)?;
    rom.write_u16(snes2pc(0x82EAF1), (pc2snes(cre_gfx16x16_addr) & 0xFFFF) as isize)?;
    
    // Write SCE tileset:
    for (&tileset_idx, tileset) in base_theme.sce_tilesets.iter() {
        if tileset_idx >= 0x0F && tileset_idx <= 0x14 {
            // Skip Ceres tilesets
            continue;
        }

        // Write SCE palette:
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

        // Write SCE 8x8 graphics:
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

        // Write SCE 16x16 graphics:
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

        // Update tileset pointers:
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

    // Create mapping of room pointers, BGData pointers, and FX door pointers:
    let mut room_ptr_map: HashMap<(usize, usize), RoomPtr> = HashMap::new();
    let mut bg_ptr_map: HashMap<(usize, usize, usize), u16> = HashMap::new();
    let mut fx_door_ptr_map: HashMap<FX1Reference, DoorPtr> = HashMap::new();
    for &room_ptr in game_data.raw_room_id_by_ptr.keys() {
        let area = rom.read_u8(room_ptr + 1)?;
        let index = rom.read_u8(room_ptr)?;
        room_ptr_map.insert((area as usize, index as usize), room_ptr);

        let state_ptrs = get_room_state_ptrs(&rom, room_ptr)?;
        for (state_idx, (_event_ptr, state_ptr)) in state_ptrs.iter().enumerate() {
            let bg_ptr = rom.read_u16(state_ptr + 22)? as u16;
            bg_ptr_map.insert((area as usize, index as usize, state_idx), bg_ptr);

            let fx_ptr = rom.read_u16(state_ptr + 6)? as usize;
            for i in 0..4 {
                let door_ptr = rom.read_u16(snes2pc(0x830000 + fx_ptr + i * 16))?;
                if door_ptr == 0 {
                    break;
                }
                fx_door_ptr_map.insert(FX1Reference {
                    room_area: area as usize,
                    room_index: index as usize,
                    state_index: state_idx as usize,
                    fx_index: i,
                }, door_ptr as DoorPtr);
            }
        }
    }

    // Write room data:
    let theme = &retiled_theme_data.themes[theme_name];
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

            // Write the tileset index
            rom.write_u8(state_ptr + 3, state.tileset_idx as isize)?;

            // Write (or clear) the BGData pointer:
            if state.layer2_type == Layer2Type::BGData {
                let bg_ref = state.bgdata_reference.as_ref().unwrap();
                let bg_ptr = bg_ptr_map[&(bg_ref.room_area, bg_ref.room_index, bg_ref.room_state_index)];
                rom.write_u16(state_ptr + 22, bg_ptr as isize)?;
            } else {
                rom.write_u16(state_ptr + 22, 0)?;
            }

            // Write BG scroll speeds:
            let mut speed_x = state.bg_scroll_speed_x;
            let mut speed_y = state.bg_scroll_speed_y;
            if state.layer2_type == Layer2Type::BGData { 
                speed_x |= 0x01;
                speed_y |= 0x01;
            }
            rom.write_u8(state_ptr + 12, speed_x as isize)?;
            rom.write_u8(state_ptr + 13, speed_y as isize)?;

            // Write level data:
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
            rom.write_u24(state_ptr, pc2snes(level_data_addr) as isize)?;

            // Write FX:
            let fx_data = get_fx_data(&state.fx1, &fx_door_ptr_map)?;
            let fx_data_addr = match fx_data_map.entry(fx_data.clone()) {
                Entry::Occupied(x) => *x.get(),
                Entry::Vacant(view) => {
                    let addr = fx_allocator.allocate(fx_data.len())?;
                    view.insert(addr);
                    addr
                }
            };
            rom.write_n(fx_data_addr, &fx_data)?;
            rom.write_u16(state_ptr + 6, (pc2snes(fx_data_addr) & 0xFFFF) as isize)?;

            // Write setup & main ASM pointers:
            rom.write_u16(state_ptr + 18, state.main_asm as isize)?;
            rom.write_u16(state_ptr + 24, state.setup_asm as isize)?;
        }
    }

    Ok(())
}
