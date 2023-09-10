use crate::{
    game_data::{AreaIdx, GameData, TilesetIdx},
    patch::{compress::compress, snes2pc, pc2snes, Rom},
};
use anyhow::{Result, bail};
use hashbrown::HashMap;
use hashbrown::hash_map::Entry;

fn encode_palette(pal: &[[u8; 3]]) -> Vec<u8> {
    let mut out: Vec<u8> = vec![];
    for i in 0..128 {
        let r = pal[i][0] as u16 / 8;
        let g = pal[i][1] as u16 / 8;
        let b = pal[i][2] as u16 / 8;
        let w = r | (g << 5) | (b << 10);
        out.push((w & 0xFF) as u8);
        out.push((w >> 8) as u8);
    }
    out
}

pub fn decode_palette(pal_bytes: &[u8]) -> [[u8; 3]; 128] {
    let mut out = [[0u8; 3]; 128];
    for i in 0..128 {
        let c = pal_bytes[i * 2] as u16 | ((pal_bytes[i * 2 + 1] as u16) << 8);
        let r = (c & 31) * 8;
        let g = ((c >> 5) & 31) * 8;
        let b = ((c >> 10) & 31) * 8;
        out[i] = [r as u8, g as u8, b as u8];
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

fn encode_color(color: [u8; 3]) -> u16 {
    let r = color[0] as u16;
    let g = color[1] as u16;
    let b = color[2] as u16;
    r | (g << 5) | (b << 10)
}

fn decode_color(word: u16) -> [u8; 3] {
    let mut out = [0u8; 3];
    out[0] = (word & 0x1F) as u8;
    out[1] = ((word >> 5) & 0x1F) as u8;
    out[2] = ((word >> 10) & 0x1F) as u8;
    out
}

fn make_palette_blends_gray(rom: &mut Rom) -> Result<()> {
    // Adjust palette blends (e.g. water backgrounds) to grayscale to make them
    // have less effect on the color of the room.
    for i in 1..8 {
        for j in 0..15 {
            let ptr_snes = 0x89AA02 + (i * 16 + j) * 2;
            let color_word = rom.read_u16(snes2pc(ptr_snes))? as u16;
            let color = decode_color(color_word);
            let avg = (color[0] + color[1] + color[2]) / 3;
            let gray = [avg; 3];
            let gray_color = encode_color(gray);
            rom.write_u16(snes2pc(ptr_snes), gray_color as isize)?;
        }
    }
    Ok(())
}

fn replace_room_tilesets(
    rom: &mut Rom,
    game_data: &GameData,
    tile_map: &HashMap<(AreaIdx, TilesetIdx), TilesetIdx>,
) -> Result<()> {
    for room_json in game_data.room_json_map.values() {
        let room_ptr =
            parse_int::parse::<usize>(room_json["roomAddress"].as_str().unwrap()).unwrap();
        let vanilla_area = rom.read_u8(room_ptr + 1)? as usize;
        let map_area = get_room_map_area(rom, room_ptr)?;
        for (_event_ptr, state_ptr) in get_room_state_ptrs(rom, room_ptr)? {
            let old_tileset_idx = rom.read_u8(state_ptr + 3)? as usize;
            if tile_map.contains_key(&(map_area, old_tileset_idx)) {
                let new_tileset_idx = tile_map[&(map_area, old_tileset_idx)];
                rom.write_u8(state_ptr + 3, new_tileset_idx as isize)?;
            }

            if vanilla_area != map_area {
                // Remove palette glows for non-vanilla rooms:
                let fx_ptr_snes = rom.read_u16(state_ptr + 6)? as usize + 0x830000;
                let fx_door_select = rom.read_u16(snes2pc(fx_ptr_snes))?;

                if fx_door_select != 0xFFFF {
                    let mut pal_fx_bitflags = rom.read_u8(snes2pc(fx_ptr_snes + 13))?;

                    if vanilla_area == 2 {
                        pal_fx_bitflags &= 1;  // Norfair room: only keep the heat FX bit
                    } else if vanilla_area != 4 {  // Keep palette FX for Maridia rooms (e.g. waterfalls)
                        pal_fx_bitflags = 0;
                    }
                    rom.write_u8(snes2pc(fx_ptr_snes + 13), pal_fx_bitflags)?;    
                }
            }
        }
    }
    Ok(())
}

fn fix_phantoon_power_on(rom: &mut Rom, game_data: &GameData) -> Result<()> {
    // Fix palette transition that happens in Phantoon's Room after defeating Phantoon.
    let phantoon_room_ptr = 0x7CD13;
    let phantoon_area = get_room_map_area(rom, phantoon_room_ptr)?;
    if phantoon_area >= 6 {
        bail!("Invalid Phantoon area: {phantoon_area}")
    }
    if phantoon_area != 3 {
        let powered_on_palette = &game_data.tileset_palette_themes[phantoon_area][&4].palette;
        let encoded_palette = encode_palette(powered_on_palette);
        rom.write_n(snes2pc(0xA7CA61), &encoded_palette[0..224])?;
        rom.write_u16(snes2pc(0xA7CA7B), 0x48FB)?; // 2bpp palette 3, color 1: pink color for E-tanks (instead of black)
        rom.write_u16(snes2pc(0xA7CA97), 0x7FFF)?; // 2bpp palette 6, color 3: white color for HUD text/digits
    }
    Ok(())
}

fn lighten_firefleas(rom: &mut Rom) -> Result<()> {
    // Reduce the darkening effect per fireflea kill (so that in many of the palettes the
    // room won't go completely black so soon).
    let darkness_shades = [
        0x00, 0x00, 0x00, 0x03, 0x00, 0x06, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x12,
    ];
    rom.write_n(snes2pc(0x88B070), &darkness_shades)?;
    Ok(())
}

fn fix_mother_brain(rom: &mut Rom, game_data: &GameData) -> Result<()> {
    // Copy new room palette to where it's needed so it doesn't get overwritten
    // during cutscenes:
    let mother_brain_room_ptr = 0x7DD58;
    let area = get_room_map_area(rom, mother_brain_room_ptr)?;
    if area != 5 {
        let theme = &game_data.tileset_palette_themes[area][&14];
        // let encoded_palette = encode_palette(palette);
        // rom.write_n(snes2pc(0xA9D082), &encoded_palette[104..128])?;
    
        for i in 0..6 {
            let faded_palette: Vec<[u8; 3]> = theme.palette
                .iter()
                .map(|&c| c.map(|x| (x as usize * (6 - i as usize) / 6) as u8))
                .collect();
            let encoded_faded_palette = encode_palette(&faded_palette);
            rom.write_n(snes2pc(0xADF283 + i * 56), &encoded_faded_palette[98..126])?;
            rom.write_n(snes2pc(0xADF283 + i * 56 + 28), &encoded_faded_palette[162..190])?;
        }    
    }

    // Disable red background flashing at escape start:
    rom.write_n(snes2pc(0xA9B295), &[0xE8u8; 28])?;  // NOP:...:NOP

    // // Disable lights off before Metroid death cutscene
    // rom.write_u8(snes2pc(0xADF209), 0x6B)?;  // RTL

    // // Disable lights back on after Metroid death cutscene
    // rom.write_u8(snes2pc(0xADF24B), 0x6B)?;  // RTL

    Ok(())
}

struct AllocatorBlock {
    _start_addr: usize,
    end_addr: usize,
    current_addr: usize
}

struct Allocator {
    blocks: Vec<AllocatorBlock>,
}

impl Allocator {
    pub fn new(blocks: Vec<(usize, usize)>) -> Self {
        Allocator {
            blocks: blocks.into_iter().map(|(start, end)| AllocatorBlock {
                _start_addr: start,
                end_addr: end,
                current_addr: start,
            }).collect()
        }
    }

    pub fn allocate(&mut self, size: usize) -> Result<usize> {
        for block in &mut self.blocks {
            if block.end_addr - block.current_addr >= size {
                let addr = block.current_addr;
                block.current_addr += size;
                // println!("success: allocated {} bytes: ending at {:x}", size, pc2snes(block.current_addr));
                return Ok(addr);
            }
        }
        bail!("Failed to allocate {} bytes", size);
    }
}

pub fn apply_area_themed_palettes(rom: &mut Rom, game_data: &GameData) -> Result<()> {
    let new_tile_table_snes = 0x8FF900;
    let new_tile_pointers_snes = 0x8FFD00;
    let tile_pointers_free_space_end = 0x8FFE00;

    let mut allocator = Allocator::new(vec![
        (snes2pc(0xBAC629), snes2pc(0xC2C2BB)),  // Vanilla tile GFX, tilemaps, and palettes, which we overwrite
        (snes2pc(0xE18000), snes2pc(0xE20000)),
        (snes2pc(0xEA8000), snes2pc(0xF00000)),
    ]);

    let mut pal_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut gfx8_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut gfx16_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut tile_idx_map: HashMap<(usize, usize, usize), usize> = HashMap::new();

    let mut next_tile_idx = 0;
    let mut tile_table: Vec<u8> = rom.read_n(snes2pc(0x8FE6A2), next_tile_idx * 9)?.to_vec();
    let mut tile_map: HashMap<(AreaIdx, TilesetIdx), TilesetIdx> = HashMap::new();
    for (area_idx, area_theme_data) in game_data.tileset_palette_themes.iter().enumerate() {
        for (&tileset_idx, theme) in area_theme_data {
            let encoded_pal = encode_palette(&theme.palette);
            let compressed_pal = compress(&encoded_pal);
            let pal_addr = match pal_map.entry(encoded_pal.clone()) {
                Entry::Occupied(x) => {
                    *x.get()
                },
                Entry::Vacant(view) => {
                    let addr = allocator.allocate(compressed_pal.len())?;
                    view.insert(addr);
                    addr
                }
            };
            rom.write_n(pal_addr, &compressed_pal)?;

            let compressed_gfx8 = compress(&theme.gfx8x8);
            let gfx8_addr = match gfx8_map.entry(theme.gfx8x8.clone()) {
                Entry::Occupied(x) => {
                    *x.get()
                },
                Entry::Vacant(view) => {
                    let addr = allocator.allocate(compressed_gfx8.len())?;
                    view.insert(addr);
                    addr
                }
            };
            rom.write_n(gfx8_addr, &compressed_gfx8)?;

            let compressed_gfx16 = compress(&theme.gfx16x16);
            let gfx16_addr = match gfx16_map.entry(theme.gfx16x16.clone()) {
                Entry::Occupied(x) => {
                    *x.get()
                }
                Entry::Vacant(view) => {
                    let addr = allocator.allocate(compressed_gfx16.len())?;
                    view.insert(addr);
                    addr
                }
            };
            rom.write_n(gfx16_addr, &compressed_gfx16)?;


            // let data = tile_table[(tileset_idx * 9)..(tileset_idx * 9 + 6)].to_vec();
            // tile_table.extend(&data);
            let tile_idx = match tile_idx_map.entry((gfx16_addr, gfx8_addr, pal_addr)) {
                Entry::Occupied(x) => {
                    *x.get()
                }
                Entry::Vacant(view) => {
                    let idx = next_tile_idx;
                    view.insert(idx);
                    tile_table.extend(&pc2snes(gfx16_addr).to_le_bytes()[0..3]);
                    tile_table.extend(&pc2snes(gfx8_addr).to_le_bytes()[0..3]);
                    tile_table.extend(&pc2snes(pal_addr).to_le_bytes()[0..3]);     
                    next_tile_idx += 1;
                    idx
                }
            };
            tile_map.insert((area_idx, tileset_idx), tile_idx);
        }
    }
    println!("Number of unique pal: {}", pal_map.len());
    println!("Number of unique gfx8: {}", gfx8_map.len());
    println!("Number of unique gfx16: {}", gfx16_map.len());
    println!("Number of unique tile_idx: {}", tile_idx_map.len());
    println!(
        "Tileset table size: {}, next_tile_idx={next_tile_idx}",
        tile_table.len()
    );
    assert!(tile_table.len() <= new_tile_pointers_snes - new_tile_table_snes);
    assert!(new_tile_pointers_snes + 2 * tile_table.len() / 9 <= tile_pointers_free_space_end);

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
    make_palette_blends_gray(rom)?;
    fix_phantoon_power_on(rom, game_data)?;
    lighten_firefleas(rom)?;
    fix_mother_brain(rom, game_data)?;
    Ok(())
}
