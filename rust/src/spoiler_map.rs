use anyhow::Result;
use hashbrown::HashMap;
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use std::io::Cursor;

use crate::{
    game_data::{GameData, Map, AreaIdx},
    patch::{snes2pc, xy_to_map_offset, Rom, map_tiles::{TilemapOffset, TilemapWord}},
    patch::map_tiles::TILE_GFX_ADDR_4BPP,
};

// fn read_tile_2bpp(rom: &Rom, base_addr: usize, idx: usize) -> Result<[[u8; 8]; 8]> {
//     let mut out: [[u8; 8]; 8] = [[0; 8]; 8];
//     for y in 0..8 {
//         let addr = base_addr + idx * 16 + y * 2;
//         let data_low = rom.read_u8(addr)?;
//         let data_high = rom.read_u8(addr + 1)?;
//         for x in 0..8 {
//             let bit_low = (data_low >> (7 - x)) & 1;
//             let bit_high = (data_high >> (7 - x)) & 1;
//             let c = bit_low | (bit_high << 1);
//             out[y][x] = c as u8;
//         }
//     }
//     Ok(out)
// }

fn read_tile_4bpp(rom: &Rom, base_addr: usize, idx: usize) -> Result<[[u8; 8]; 8]> {
    let mut out: [[u8; 8]; 8] = [[0; 8]; 8];
    for y in 0..8 {
        let addr = base_addr + idx * 32 + y * 2;
        let data_0 = rom.read_u8(addr)?;
        let data_1 = rom.read_u8(addr + 1)?;
        let data_2 = rom.read_u8(addr + 16)?;
        let data_3 = rom.read_u8(addr + 17)?;
        for x in 0..8 {
            let bit_0 = (data_0 >> (7 - x)) & 1;
            let bit_1 = (data_1 >> (7 - x)) & 1;
            let bit_2 = (data_2 >> (7 - x)) & 1;
            let bit_3 = (data_3 >> (7 - x)) & 1;
            let c = bit_0 | (bit_1 << 1) | (bit_2 << 2) | (bit_3 << 3);
            out[y][x] = c as u8;
        }
    }
    Ok(out)
}

fn render_tile(rom: &Rom, tilemap_word: u16, map_area: usize) -> Result<[[u8; 8]; 8]> {
    let idx = (tilemap_word & 0x3FF) as usize;
    let x_flip = tilemap_word & 0x4000 != 0;
    let y_flip = tilemap_word & 0x8000 != 0;
    // let tile = read_tile_2bpp(rom, snes2pc(0x9AB200), idx)?;
    let tile = read_tile_4bpp(rom, snes2pc(TILE_GFX_ADDR_4BPP + map_area * 0x10000), idx)?;
    let mut out = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            let x1 = if !x_flip { x } else { 7 - x };
            let y1 = if !y_flip { y } else { 7 - y };
            out[y][x] = tile[y1][x1];
        }
    }
    Ok(out)
}

fn get_rgb(r: isize, g: isize, b: isize)-> Rgb<u8> {
    Rgb([(r * 255 / 31) as u8, (g * 255 / 31) as u8, (b * 255 / 31) as u8])
}

fn get_color(value: u8, area: usize) -> Rgb<u8> {
    let cool_area_color = match area {
        0 => get_rgb(18, 0, 27), // Crateria
        1 => get_rgb(0, 18, 0), // Brinstar
        2 => get_rgb(23, 0, 0), // Norfair
        3 => get_rgb(16, 17, 0), // Wrecked Ship
        4 => get_rgb(3, 12, 29), // Maridia
        5 => get_rgb(21, 12, 0), // Tourian
        _ => panic!("Unexpected area {}", area),
    };
    let hot_area_color = match area {
        0 => get_rgb(27, 15, 31), // Crateria
        1 => get_rgb(12, 25, 12), // Brinstar
        2 => get_rgb(31, 12, 12), // Norfair
        3 => get_rgb(23, 23, 11), // Wrecked Ship
        4 => get_rgb(12, 20, 31), // Maridia
        5 => get_rgb(29, 17, 12), // Tourian
        _ => panic!("Unexpected area {}", area),
    };
    match value {
        0 => get_rgb(0, 0, 0),
        1 => cool_area_color,
        2 => hot_area_color,
        3 => get_rgb(31, 31, 31), // Wall/passage (white)
        4 => get_rgb(0, 0, 0), // Opaque black (used in elevators, covers up dotted grid background)
        6 => get_rgb(31, 12, 0), // Yellow (orange) door (Power Bomb, Spazer)
        7 => get_rgb(27, 2, 27), // Red (pink) door (Missile, Wave)
        8 => get_rgb(4, 13, 31), // Blue door (Ice)
        12 => get_rgb(0, 0, 0), // Door lock shadow covering wall (black)
        13 => get_rgb(31, 31, 31), // Item dots (white)
        14 => get_rgb(6, 26, 6), // Green door (Super, Plasma)
        15 => get_rgb(18, 12, 14), // Gray door (including Charge)
        _ => panic!("Unexpected color value {}", value),
    }
}

pub struct SpoilerMaps {
    pub assigned: Vec<u8>,
    pub vanilla: Vec<u8>,
    pub grid: Vec<u8>,
}

fn get_map_overrides(rom: &Rom) -> Result<HashMap<(AreaIdx, TilemapOffset), TilemapWord>> {
    let mut out = HashMap::new();
    let base_ptr_pc = snes2pc(0x83B000);
    for area_idx in 0..6 {
        let data_ptr_snes = rom.read_u16(base_ptr_pc + 2 * area_idx)?;
        let data_ptr_pc = snes2pc(0x830000 + data_ptr_snes as usize);
        let size = rom.read_u16(base_ptr_pc + 12 + 2 * area_idx)? as usize;
        for i in 0..size {
            let offset = rom.read_u16(data_ptr_pc + 6 * i + 2)?;
            let word = rom.read_u16(data_ptr_pc + 6 * i + 4)?;
            out.insert((area_idx, offset as TilemapOffset), word as TilemapWord);
        }
    }
    Ok(out)
}

pub fn get_spoiler_map(
    rom: &Rom,
    map: &Map,
    game_data: &GameData,
) -> Result<SpoilerMaps> {
    let max_tiles = 72;
    let width = (max_tiles + 2) * 8;
    let height = (max_tiles + 2) * 8;
    let mut img_assigned = RgbImage::new(width, height);
    let mut img_vanilla = RgbImage::new(width, height);
    let mut img_grid = RgbaImage::new(width, height);
    let grid_val = Rgba([0x29, 0x29, 0x29, 0xFF]);
    let map_overrides = get_map_overrides(rom)?;

    for y in (7..height).step_by(8) {
        for x in (0..width).step_by(2) {
            img_grid.put_pixel(x, y, grid_val);
        }
    }
    for x in (0..width).step_by(8) {
        for y in (1..height).step_by(2) {
            img_grid.put_pixel(x, y, grid_val);
        }
    }

    for room_idx in 0..map.rooms.len() {
        let room = &game_data.room_geometry[room_idx];
        let room_ptr = room.rom_address;
        let map_area = map.area[room_idx];
        let vanilla_area = rom.read_u8(room_ptr + 1)? as usize;
        let area_room_x = rom.read_u8(room_ptr + 2)?;
        let area_room_y = rom.read_u8(room_ptr + 3)?;
        let global_room_x = map.rooms[room_idx].0;
        let global_room_y = map.rooms[room_idx].1;
        for (local_y, row) in room.map.iter().enumerate() {
            for (local_x, &cell) in row.iter().enumerate() {
                if cell == 0 && room_idx != game_data.toilet_room_idx {
                    continue;
                }
                let cell_x = area_room_x + local_x as isize;
                let cell_y = area_room_y + local_y as isize;
                let offset = xy_to_map_offset(cell_x, cell_y);
                let cell_ptr = game_data.area_map_ptrs[map_area] + offset;
                let mut tilemap_word = rom.read_u16(cell_ptr as usize)? as u16;
                if let Some(new_word) = map_overrides.get(&(map_area, offset as TilemapOffset)) {
                    tilemap_word = *new_word;
                }
                let tile = render_tile(rom, tilemap_word, map_area)?;
                for y in 0..8 {
                    for x in 0..8 {
                        let x1 = (global_room_x + local_x + 1) * 8 + x;
                        let y1 = (global_room_y + local_y + 1) * 8 + y;
                        if tile[y][x] != 0 {
                            img_grid.put_pixel(x1 as u32, y1 as u32, Rgba([0x00, 0x00, 0x00, 0x00]));
                        }
                        img_vanilla.put_pixel(
                            x1 as u32,
                            y1 as u32,
                            get_color(tile[y][x], vanilla_area),
                        );
                        img_assigned.put_pixel(x1 as u32, y1 as u32, get_color(tile[y][x], map_area));
                    }
                }
            }
        }
    }

    let mut vec_assigned: Vec<u8> = Vec::new();
    img_assigned.write_to(
        &mut Cursor::new(&mut vec_assigned),
        image::ImageOutputFormat::Png,
    )?;

    let mut vec_vanilla: Vec<u8> = Vec::new();
    img_vanilla.write_to(
        &mut Cursor::new(&mut vec_vanilla),
        image::ImageOutputFormat::Png,
    )?;

    let mut vec_grid: Vec<u8> = Vec::new();
    img_grid.write_to(
        &mut Cursor::new(&mut vec_grid),
        image::ImageOutputFormat::Png,
    )?;
    Ok(SpoilerMaps {
        assigned: vec_assigned,
        vanilla: vec_vanilla,
        grid: vec_grid,
    })
}
