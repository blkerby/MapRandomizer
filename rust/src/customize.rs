use crate::{
    game_data::{AreaIdx, GameData, TilesetIdx},
    patch::{compress::compress, snes2pc, Rom},
};
use anyhow::Result;
use hashbrown::HashMap;
use std::cmp::max;

pub struct CustomizeSettings {
    pub area_themed_palette: bool,
}

fn encode_palette(pal: &[[u8; 3]; 128]) -> [u8; 256] {
    let mut out = [0u8; 256];
    for i in 0..128 {
        let r = pal[i][0] as u16 / 8;
        let g = pal[i][1] as u16 / 8;
        let b = pal[i][2] as u16 / 8;
        let w = r | (g << 5) | (b << 10);
        out[i * 2] = (w & 0xFF) as u8;
        out[i * 2 + 1] = (w >> 8) as u8;
    }
    out
}

// Returns list of (event_ptr, state_ptr):
fn get_room_state_ptrs(rom: &Rom, room_ptr: usize) -> Result<Vec<(usize, usize)>> {
    let mut pos = 11;
    let mut ptr_pairs: Vec<(usize, usize)> = Vec::new();
    loop {
        let ptr = rom.read_u16(room_ptr + pos)? as usize;
        if ptr == 0xE5E6 {
            // This is the standard state, which is the last one.
            ptr_pairs.push((ptr, room_ptr + pos + 2));
            return Ok(ptr_pairs);
        } else if ptr == 0xE612 || ptr == 0xE629 {
            // This is an event state.
            let state_ptr = 0x70000 + rom.read_u16(room_ptr + pos + 3)?;
            ptr_pairs.push((ptr, state_ptr as usize));
            pos += 5;
        } else {
            // This is another kind of state.
            let state_ptr = 0x70000 + rom.read_u16(room_ptr + pos + 2)?;
            ptr_pairs.push((ptr, state_ptr as usize));
            pos += 4;
        }
    }
}

fn get_room_map_area(rom: &Rom, room_ptr: usize) -> Result<usize> {
    let room_index = rom.read_u8(room_ptr)? as usize;
    let vanilla_area = rom.read_u8(room_ptr + 1)? as usize;
    let area_data_base_ptr = snes2pc(0x8FE99B);
    let area_data_ptr = rom.read_u16(area_data_base_ptr + vanilla_area * 2)? as usize;
    let map_area = rom.read_u8(snes2pc(0x8F0000 + area_data_ptr) + room_index)? as usize;
    Ok(map_area)
}

fn replace_room_tilesets(
    rom: &mut Rom,
    game_data: &GameData,
    tile_map: &HashMap<(AreaIdx, TilesetIdx), TilesetIdx>,
) -> Result<()> {
    for &room_ptr in game_data.room_ptr_by_id.values() {
        let area = get_room_map_area(rom, room_ptr)?;
        for (_event_ptr, state_ptr) in get_room_state_ptrs(rom, room_ptr)? {
            let old_tileset_idx = rom.read_u8(state_ptr + 3)? as usize;
            if tile_map.contains_key(&(area, old_tileset_idx)) {
                let new_tileset_idx = tile_map[&(area, old_tileset_idx)];
                println!("area={area}, room_ptr={room_ptr:x}, old_tileset_idx={old_tileset_idx}, new_tileset_idx={new_tileset_idx}");
                // rom.write_u8(state_ptr + 3, old_tileset_idx as isize)?;    
                rom.write_u8(state_ptr + 3, new_tileset_idx as isize)?;    
            } else {
                println!("unchanged: area={area}, room_ptr={room_ptr:x}, old_tileset_idx={old_tileset_idx}");
            }
        }
    }
    Ok(())
}

pub fn apply_area_themed_palettes(rom: &mut Rom, game_data: &GameData) -> Result<()> {
    let new_tile_table_snes = 0x8FF900;
    let new_tile_pointers_snes = 0x8FFC00;
    let pal_free_space_start_snes = 0xE18000;
    let pal_free_space_end_snes = pal_free_space_start_snes + 0x8000;
    let mut pal_free_space_snes = pal_free_space_start_snes;
    rom.data
        .resize(max(snes2pc(pal_free_space_end_snes), rom.data.len()), 0xFF);

    let mut next_tile_idx = 29;
    let mut tile_table: Vec<u8> = rom.read_n(snes2pc(0x8FE6A2), next_tile_idx * 9)?.to_vec();
    let mut tile_map: HashMap<(AreaIdx, TilesetIdx), TilesetIdx> = HashMap::new();
    for (area_idx, area_palette_data) in game_data.palette_data.iter().enumerate() {
        for (&tileset_idx, pal) in area_palette_data {
            let encoded_pal = encode_palette(pal);
            let compressed_pal = compress(&encoded_pal);
            rom.write_n(snes2pc(pal_free_space_snes), &compressed_pal)?;

            let data = tile_table[(tileset_idx * 9)..(tileset_idx * 9 + 6)].to_vec();
            tile_table.extend(&data);
            tile_table.extend(&pal_free_space_snes.to_le_bytes()[0..3]);
            tile_map.insert((area_idx, tileset_idx), next_tile_idx);

            next_tile_idx += 1;
            pal_free_space_snes += compressed_pal.len();
        }
    }
    println!("Tileset table size: {}, next_tile_idx={next_tile_idx}", tile_table.len());
    assert!(pal_free_space_snes <= pal_free_space_end_snes);
    assert!(tile_table.len() <= new_tile_pointers_snes - new_tile_table_snes);

    rom.write_n(snes2pc(new_tile_table_snes), &tile_table)?;
    for i in 0..tile_table.len() / 9 {
        rom.write_u16(
            snes2pc(new_tile_pointers_snes + 2 * i),
            ((new_tile_table_snes + 9 * i) & 0xFFFF) as isize,
        )?;
    }

    rom.write_u16(
        snes2pc(0x82DF03),
        (new_tile_pointers_snes & 0xFFFF) as isize,
    )?;
    replace_room_tilesets(rom, game_data, &tile_map)?;

    Ok(())
}

pub fn customize_rom(
    rom: &mut Rom,
    seed_patch: &[u8],
    settings: &CustomizeSettings,
    game_data: &GameData,
) -> Result<()> {
    let patch = ips::Patch::parse(seed_patch).unwrap();
    // .with_context(|| format!("Unable to parse patch {}", patch_path.display()))?;
    for hunk in patch.hunks() {
        rom.write_n(hunk.offset(), hunk.payload())?;
    }
    if settings.area_themed_palette {
        apply_area_themed_palettes(rom, game_data)?;
    }
    Ok(())
}
