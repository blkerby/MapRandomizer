use std::io::Cursor;
use anyhow::Result;
use image::{Rgb, RgbImage};

use crate::{game_data::{Map, GameData}, patch::{Rom, xy_to_map_offset, snes2pc}};


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
            let c = bit_0 | (bit_1 << 1) | (bit_2 << 1) | (bit_3 << 1);
            out[y][x] = c as u8;
        }
    }
    Ok(out)
}


fn render_tile(rom: &Rom, tilemap_word: u16) -> Result<[[u8; 8]; 8]> {
    let idx = (tilemap_word & 0x3FF) as usize;
    let x_flip = tilemap_word & 0x4000 != 0;
    let y_flip = tilemap_word & 0x8000 != 0;
    // let tile = read_tile_2bpp(rom, snes2pc(0x9AB200), idx)?;
    let tile = read_tile_4bpp(rom, snes2pc(0xB68000), idx)?;
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

fn get_color(value: u8, area: usize) -> Rgb<u8> {
    match value {
        0 => Rgb([0x00, 0x00, 0x00]),
        1 => {
            match area {
                0 => Rgb([0x80, 0x10, 0xD8]),  // Crateria
                1 => Rgb([0x00, 0xA8, 0x00]),  // Brinstar
                2 => Rgb([0xC8, 0x00, 0x00]),  // Norfair
                3 => Rgb([0xB0, 0xB8, 0x00]),  // Wrecked Ship
                4 => Rgb([0x30, 0x60, 0xF8]),  // Maridia
                5 => Rgb([0xA0, 0xA0, 0xA0]),  // Tourian
                _ => panic!("Unexpected area {}", area)
            }
        },
        2 => Rgb([0xFF, 0xFF, 0xFF]),
        3 => Rgb([0x00, 0x00, 0x00]),
        _ => panic!("Unexpected color value {}", value)
    }
}

pub fn get_spoiler_map(rom: &Rom, map: &Map, game_data: &GameData, use_vanilla_area: bool) -> Result<Vec<u8>> {
    let max_tiles = 72;
    let width = (max_tiles + 2) * 8;
    let height = (max_tiles + 2) * 8;
    let mut img = RgbImage::new(width, height);

    for room_idx in 0..map.rooms.len() {
        let room = &game_data.room_geometry[room_idx];
        let room_ptr = room.rom_address;
        let map_area = map.area[room_idx];
        let vanilla_area = rom.read_u8(room_ptr + 1)? as usize;
        let area_room_x = rom.read_u8(room_ptr + 2)?;
        let mut area_room_y = rom.read_u8(room_ptr + 3)?;
        if room.name == "Aqueduct" {
            area_room_y -= 4;
        }
        let global_room_x = map.rooms[room_idx].0;
        let global_room_y = map.rooms[room_idx].1;
        for (local_y, row) in room.map.iter().enumerate() {
            for (local_x, &cell) in row.iter().enumerate() {
                if cell == 0 {
                    continue;
                }
                let cell_x = area_room_x + local_x as isize;
                let cell_y = area_room_y + local_y as isize;
                let offset = xy_to_map_offset(cell_x, cell_y);
                let cell_ptr = game_data.area_map_ptrs[map_area] + offset;
                let tilemap_word = rom.read_u16(cell_ptr as usize)? as u16;
                let tile = render_tile(rom, tilemap_word)?;
                for y in 0..8 {
                    for x in 0..8 {
                        let x1 = (global_room_x + local_x + 1) * 8 + x;
                        let y1 = (global_room_y + local_y + 1) * 8 + y;
                        if use_vanilla_area {
                            img.put_pixel(x1 as u32, y1 as u32, get_color(tile[y][x], vanilla_area));
                        } else {
                            img.put_pixel(x1 as u32, y1 as u32, get_color(tile[y][x], map_area));
                        }
                    }
                }
            }
        }
    }
    
    let mut output_bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut output_bytes), image::ImageOutputFormat::Png)?;
    Ok(output_bytes)
}