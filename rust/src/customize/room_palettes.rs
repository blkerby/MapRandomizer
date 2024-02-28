use std::path::Path;

use crate::{
    game_data::{AreaIdx, GameData, TilesetIdx},
    patch::{apply_ips_patch, compress::compress, pc2snes, snes2pc, Rom},
};
use super::Allocator;
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

pub fn apply_area_themed_palettes(rom: &mut Rom, game_data: &GameData) -> Result<()> {
    apply_ips_patch(rom, Path::new(&"../patches/ips/area_palettes.ips"))?;
    make_palette_blends_gray(rom)?;
    fix_phantoon_power_on(rom, game_data)?;
    lighten_firefleas(rom)?;
    fix_mother_brain(rom, game_data)?;
    Ok(())
}
