use std::path::Path;

use crate::patch::compress::compress;
use maprando_game::{read_image, IndexedVec};

use super::{decompress::decompress, pc2snes, snes2pc, PcAddr, Rom};
use anyhow::{bail, ensure, Result};
use hashbrown::HashMap;
use ndarray::{concatenate, Array2, Array3, Axis};
use slice_of_array::prelude::*;

pub struct TitlePatcher<'a> {
    rom: &'a mut Rom,
    pub next_free_space_pc: usize,
    pub end_free_space_pc: usize,
}

fn rgb_to_u16(rgb: (u8, u8, u8)) -> u16 {
    let (r, g, b) = rgb;
    (r as u16) | (g as u16) << 5 | (b as u16) << 10
}

struct Graphics {
    palette: Vec<(u8, u8, u8)>,
    tiles: Vec<[[u8; 8]; 8]>, // indices into `palette`
    tilemap: Array2<u8>,      // indices into `tiles`
}

fn encode_mode7_graphics(image: &Array3<u8>) -> Result<Graphics> {
    let (height, width, _) = image.dim();

    let mut tilemap: Array2<u8> = Array2::zeros([height / 8, width / 8]);
    let mut tile_isv: IndexedVec<[[[u8; 3]; 8]; 8]> = IndexedVec::default();

    let mut process_tile = |tile_y, tile_x| {
        let mut tile: [[[u8; 3]; 8]; 8] = [[[0; 3]; 8]; 8];
        for y in 0..8 {
            for x in 0..8 {
                for c in 0..3 {
                    tile[y][x][c] = image[[tile_y * 8 + y, tile_x * 8 + x, c]];
                }
            }
        }
        let tile_idx = tile_isv.add(&tile) as u8;
        tilemap[[tile_y, tile_x]] = tile_idx;
    };

    // Process the map station tiles first (required for the animation patch)
    process_tile(20, 15);
    process_tile(20, 16);
    process_tile(21, 15);
    process_tile(21, 16);
    process_tile(16, 15); // blank tile
    for tile_y in 0..(height / 8) {
        for tile_x in 0..(width / 8) {
            process_tile(tile_y, tile_x);
        }
    }
    ensure!(tile_isv.keys.len() <= 256);

    let mut new_tiles: Vec<[[u8; 8]; 8]> = Vec::new();
    let mut color_isv: IndexedVec<(u8, u8, u8)> = IndexedVec::default();
    color_isv.add(&(0, 0, 0)); // Keep the black color
    for tile in &tile_isv.keys {
        let mut new_tile: [[u8; 8]; 8] = [[0; 8]; 8];
        for y in 0..8 {
            for x in 0..8 {
                let pixel = tile[y][x].map(|x| x as isize);
                let r = pixel[0] / 8;
                let g = pixel[1] / 8;
                let b = pixel[2] / 8;
                let idx = color_isv.add(&(r as u8, g as u8, b as u8)) as u8;
                new_tile[y][x] = idx;
            }
        }
        new_tiles.push(new_tile);
    }

    Ok(Graphics {
        palette: color_isv.keys,
        tiles: new_tiles,
        tilemap,
    })
}

fn decode_tile_4bpp(tile: &[u8; 32]) -> [[u8; 8]; 8] {
    let mut out: [[u8; 8]; 8] = [[0; 8]; 8];
    for y in 0..8 {
        let i = y * 2;
        for x in 0..8 {
            let b0 = (tile[i] >> (7 - x)) & 1;
            let b1 = (tile[i + 1] >> (7 - x)) & 1;
            let b2 = (tile[i + 16] >> (7 - x)) & 1;
            let b3 = (tile[i + 17] >> (7 - x)) & 1;
            out[y][x] = b0 | b1 << 1 | b2 << 2 | b3 << 3;
        }
    }
    out
}

fn encode_tile_4bpp(tile: &[[u8; 8]; 8]) -> [u8; 32] {
    let mut out: [u8; 32] = [0; 32];
    let offsets = [0, 1, 16, 17];
    for p in 0..4 {
        for y in 0..8 {
            let mut c = 0;
            for x in 0..8 {
                c |= ((tile[y][x] >> p) & 1) << (7 - x);
            }
            out[y * 2 + offsets[p]] = c;
        }
    }
    out
}

#[derive(Debug)]
struct SpriteMapEntry {
    x: i16,       // x offset in pixels: -256 to +255
    y: i8,        // y offset in pixels -128 to +127
    c: u16,       // character (tile) index: 0 to 511
    palette: u8,  // 0 to 7
    priority: u8, // 0 to 3
    size_16: bool,
    x_flip: bool,
    y_flip: bool,
}

impl<'a> TitlePatcher<'a> {
    pub fn new(rom: &'a mut Rom) -> Self {
        Self {
            rom,
            next_free_space_pc: snes2pc(0xE98400),
            end_free_space_pc: snes2pc(0xEA8000),
        }
    }

    fn write_to_free_space(&mut self, data: &[u8]) -> Result<PcAddr> {
        let free_space = self.next_free_space_pc;
        self.rom.write_n(free_space, data)?;
        self.next_free_space_pc += data.len();
        if self.next_free_space_pc > self.end_free_space_pc {
            bail!("Not enough free space for title screen data");
        }
        Ok(free_space)
    }

    fn write_palette(&mut self, addr: usize, palette: &[(u8, u8, u8)]) -> Result<()> {
        let pal_u16: Vec<u16> = palette.iter().map(|&rgb| rgb_to_u16(rgb)).collect();
        for (i, c) in pal_u16.iter().copied().enumerate() {
            self.rom.write_u16(addr + i * 2, c as isize)?;
        }
        Ok(())
    }

    fn write_title_background_tiles(&mut self, tiles: &[[[u8; 8]; 8]]) -> Result<()> {
        let flat_tiles = tiles.flat().flat();
        let compressed = compress(flat_tiles);
        let gfx_pc_addr = self.write_to_free_space(&compressed)?;
        let gfx_snes_addr = pc2snes(gfx_pc_addr);
        // Update reference to tile GFX:
        self.rom
            .write_u8(snes2pc(0x8B9BA8), (gfx_snes_addr >> 16) as isize)?;
        self.rom
            .write_u16(snes2pc(0x8B9BAC), (gfx_snes_addr & 0xFFFF) as isize)?;
        Ok(())
    }

    fn write_title_background_tilemap(&mut self, tilemap: &Array2<u8>) -> Result<()> {
        let tilemap_horizontal_padding: Array2<u8> = Array2::zeros([28, 96]) + 4;
        let padded_tilemap0 = concatenate(
            Axis(1),
            &[tilemap.view(), tilemap_horizontal_padding.view()],
        )?;
        let tilemap_vertical_padding: Array2<u8> = Array2::zeros([4, 128]) + 4;
        let padded_tilemap = concatenate(
            Axis(0),
            &[padded_tilemap0.view(), tilemap_vertical_padding.view()],
        )?;
        let compressed = compress(padded_tilemap.as_standard_layout().as_slice().unwrap());
        let tilemap_pc_addr = self.write_to_free_space(&compressed)?;
        let tilemap_snes_addr = pc2snes(tilemap_pc_addr);
        // Update reference to tilemap:
        self.rom
            .write_u8(snes2pc(0x8B9BB9), (tilemap_snes_addr >> 16) as isize)?;
        self.rom
            .write_u16(snes2pc(0x8B9BBD), (tilemap_snes_addr & 0xFFFF) as isize)?;
        Ok(())
    }

    pub fn patch_title_blue_light(&mut self) -> Result<()> {
        // If we want to alter the subtractive effect while the camera pans, before zooming out:
        // self.rom.write_u8(snes2pc(0x8B8675), 0x20 | 0x10)?;  // red value to subtract = 0x10
        // self.rom.write_u8(snes2pc(0x8B8679), 0x40 | 0x10)?;  // green value to subtract = 0x10
        // self.rom.write_u8(snes2pc(0x8B867D), 0x80 | 0x00)?;  // blue value to subtract = 0x00

        // Don't disable the effect for the second camera pan (top right to top left):
        self.rom.write_n(snes2pc(0x8B9D43), &[0xEA; 3])?;

        Ok(())
    }

    pub fn patch_title_gradient(&mut self) -> Result<()> {
        // Make the title gradient gray (vs. blue-tinted) and less extreme
        for i in 3..16 {
            let base_addr_snes = self.rom.read_u16(snes2pc(0x8CBC5D + i * 2))? + 0x8c0000;
            for j in 0..256 {
                let addr_pc = snes2pc((base_addr_snes + j * 2) as usize);
                let mut num_lines = self.rom.read_u8(addr_pc)?;
                let mut c = self.rom.read_u8(addr_pc + 1)?;
                let mut color_plane_mask = c & 0xE0;
                let mut intensity = c & 0x1F;
                if color_plane_mask == 0xE0 {
                    // Effect applied to all 3 color planes (used for subtractive mode, at top of screen)
                    // Reduce the effect to avoid too much darkening at the top of the screen:
                    intensity /= 3;
                    if intensity > 3 {
                        intensity = 3;
                    }
                } else if color_plane_mask == 0xC0 {
                    // Effect applied to only blue and green color planes (used for additive mode, at bottom of screen)
                    // Apply the effect to all 3 color planes, producing a grayscale gradient instead of cyan.
                    // Reduce the intensity of the effect, to avoid washing out the bottom of the screen.
                    color_plane_mask = 0xE0;
                    if intensity > 3 {
                        // Stretch out the length of each gradient band beyond this point:
                        num_lines += 3;
                    }
                } else if c == 0 {
                    break;
                } else {
                    panic!("Unexpected title screen gradient control: {c:x}");
                }
                c = color_plane_mask | intensity;
                self.rom.write_u8(addr_pc, num_lines)?;
                self.rom.write_u8(addr_pc + 1, c)?;
            }
        }
        Ok(())
    }

    pub fn patch_title_background(&mut self, img: &Array3<u8>) -> Result<()> {
        assert!(img.dim() == (224, 256, 3));

        // Compute title background palette, tile GFX, and tilemap:
        let graphics = encode_mode7_graphics(img)?;
        println!(
            "Title background distinct colors: {}",
            graphics.palette.len()
        );

        // Write palette, tile GFX, and tilemap. Here write the palette in place,
        // while we write the GFX and tilemap to free space and update references to them.
        self.write_palette(0x661E9, &graphics.palette)?;
        self.write_title_background_tiles(&graphics.tiles)?;
        self.write_title_background_tilemap(&graphics.tilemap)?;

        // Skip palette FX handler
        self.rom.write_n(snes2pc(0x8B9A34), &[0xEA; 4])?;

        // Use white color for Nintendo copyright text (otherwise it would stay black since we skip the palette FX handler)
        self.rom.write_u16(0x661E9 + 0xC9 * 2, 0x7FFF)?;

        Ok(())
    }

    fn read_compressed_tiles(&self, pc_addr: usize) -> Result<Vec<[[u8; 8]; 8]>> {
        let decompressed = decompress(self.rom, pc_addr)?;
        let mut tiles: Vec<[[u8; 8]; 8]> = Vec::new();
        assert!(decompressed.len() == 16384);
        for i in 0..512 {
            tiles.push(decode_tile_4bpp(
                &decompressed[(i * 32)..(i * 32 + 32)].try_into()?,
            ));
        }
        Ok(tiles)
    }

    fn read_spritemap(&self, mut pc_addr: usize) -> Result<Vec<SpriteMapEntry>> {
        let num_tiles = self.rom.read_u16(pc_addr)?;
        let mut out: Vec<SpriteMapEntry> = Vec::new();
        pc_addr += 2;
        for _ in 0..num_tiles {
            let x0 = self.rom.read_u16(pc_addr)?;
            let mut x = x0 & 0x1FF;
            if x >= 256 {
                x -= 512;
            }
            let size_16 = x0 >> 15 != 0;
            let mut y = self.rom.read_u8(pc_addr + 2)?;
            if y >= 128 {
                y -= 256;
            }
            let a = self.rom.read_u8(pc_addr + 3)?;
            let b = self.rom.read_u8(pc_addr + 4)?;
            let y_flip = b >> 7 != 0;
            let x_flip = (b >> 6) & 1 != 0;
            let palette = (b >> 1) & 7;
            let priority = (b >> 4) & 3;
            let c = ((b & 1) << 8) | a;
            out.push(SpriteMapEntry {
                x: x as i16,
                y: y as i8,
                c: c as u16,
                palette: palette as u8,
                priority: priority as u8,
                size_16,
                x_flip,
                y_flip,
            });
            pc_addr += 5;
        }
        Ok(out)
    }

    fn write_spritemap(&mut self, mut pc_addr: usize, spritemap: &[SpriteMapEntry]) -> Result<()> {
        self.rom.write_u16(pc_addr, spritemap.len() as isize)?;
        pc_addr += 2;
        for entry in spritemap {
            let mut x0 = (entry.x as isize + 0x200) & 0x1FF;
            if entry.size_16 {
                x0 |= 0x8000;
            }
            self.rom.write_u16(pc_addr, x0)?;
            self.rom
                .write_u8(pc_addr + 2, (entry.y as isize + 0x100) & 0xFF)?;
            self.rom.write_u8(pc_addr + 3, (entry.c as isize) & 0xFF)?;
            self.rom.write_u8(
                pc_addr + 4,
                ((entry.c as isize) >> 8)
                    | ((entry.palette as isize) << 1)
                    | ((entry.priority as isize) << 4)
                    | (if entry.x_flip { 1 << 6 } else { 0 })
                    | (if entry.y_flip { 1 << 7 } else { 0 }),
            )?;
            pc_addr += 5;
        }
        Ok(())
    }

    pub fn patch_title_foreground(&mut self) -> Result<()> {
        // Start by loading the vanilla tiles & spritemap, for "Super Metroid" title:
        let mut tiles = self.read_compressed_tiles(snes2pc(0x9580D8))?;
        let mut spritemap = self.read_spritemap(snes2pc(0x8C879D))?;

        // Now we will patch the tiles & spritemap by adding "Map Rando" to the same sprite.
        // First load the image:
        let image_path = Path::new("../gfx/title/maprando.png");
        let img = read_image(image_path)?;
        assert!(img.dim() == (224, 256, 3));

        // We don't modify the palette, just reuse colors from the existing palette.
        // There are only 3 colors in the PNG, and we manually map them over (This is a
        // little silly, because we could just use an indexed PNG to begin with, instead of RGB.):
        let mut pal_map: HashMap<(u8, u8, u8), u8> = HashMap::new();
        pal_map.insert((0, 1, 0), 0);
        pal_map.insert((205, 207, 152), 13);
        pal_map.insert((206, 208, 153), 1);

        // Indexes of 16 x 16 tiles that are free for us to use for "Map Rando" subtitle:
        let free_tiles: Vec<usize> = vec![
            0xA0, 0xA2, 0xA4, 0xA6, 0xA8, 0xAA, 0xAC, 0xAE, 0xC0, 0xC2, 0xC4, 0xC6, 0xC8, 0xCA,
            0xCC, 0xCE, 0xE0, 0xE2, 0xE4, 0xE6, 0xE8, 0xEA, 0xEC, 0xEE, 0x102, 0x104, 0x106, 0x108,
            0x10A, 0x10C, 0x10E, 0x122, 0x124, 0x126, 0x128, 0x12A, 0x12C, 0x12E, 0x140, 0x142,
            0x144, 0x146, 0x148, 0x14A, 0x14C, 0x14E, 0x160, 0x162, 0x164, 0x180, 0x182, 0x184,
            0x1A0, 0x1A2, 0x1A4, 0x1AC, 0x1AE, 0x1CC, 0x1CE, 0x1E0, 0x1E2, 0x1E4, 0x1E6, 0x1E8,
            0x1EA, 0x1EC, 0x1EE,
        ];

        let get_tile = |tile_y: usize, tile_x: usize| -> [[u8; 8]; 8] {
            let mut tile = [[0; 8]; 8];
            for y in 0..8 {
                for x in 0..8 {
                    let r = img[[tile_y * 8 + y, tile_x * 8 + x, 0]];
                    let g = img[[tile_y * 8 + y, tile_x * 8 + x, 1]];
                    let b = img[[tile_y * 8 + y, tile_x * 8 + x, 2]];
                    let c = pal_map[&(r, g, b)];
                    tile[y][x] = c;
                }
            }
            tile
        };

        let mut free_tile_idx = 0;
        let y_shift = 0x10; // Offset to shift the title down from where it is in vanilla
        for tile_y in 0..14 {
            for tile_x in 0..16 {
                let tile_00 = get_tile(tile_y * 2, tile_x * 2);
                let tile_01 = get_tile(tile_y * 2, tile_x * 2 + 1);
                let tile_10 = get_tile(tile_y * 2 + 1, tile_x * 2);
                let tile_11 = get_tile(tile_y * 2 + 1, tile_x * 2 + 1);
                let z = [[0; 8]; 8];
                if tile_00 == z && tile_01 == z && tile_10 == z && tile_11 == z {
                    continue;
                }
                let tile_idx = free_tiles[free_tile_idx];
                free_tile_idx += 1;
                let entry = SpriteMapEntry {
                    x: (tile_x as isize * 16 - 0x80) as i16,
                    y: (tile_y as isize * 16 - (0x30 + y_shift)) as i8,
                    c: tile_idx as u16,
                    palette: 2,
                    priority: 1,
                    size_16: true,
                    x_flip: false,
                    y_flip: false,
                };
                spritemap.push(entry);
                tiles[tile_idx] = tile_00;
                tiles[tile_idx + 1] = tile_01;
                tiles[tile_idx + 16] = tile_10;
                tiles[tile_idx + 17] = tile_11;
            }
        }

        // Write the tiles & spritemap to a new location in free space, since they would
        // no longer fit in the original location:
        let encoded_tiles: Vec<[u8; 32]> = tiles.iter().map(encode_tile_4bpp).collect();
        let new_gfx_pc_addr = self.write_to_free_space(&compress(encoded_tiles.flat()))?;
        let new_gfx_snes_addr = pc2snes(new_gfx_pc_addr);
        let new_spritemap_snes_addr = 0x8CF3E9;
        self.write_spritemap(snes2pc(new_spritemap_snes_addr), &spritemap)?;

        // Update pointer to the tile GFX data:
        self.rom
            .write_u8(snes2pc(0x8B9BCA), (new_gfx_snes_addr >> 16) as isize)?;
        self.rom
            .write_u16(snes2pc(0x8B9BCE), (new_gfx_snes_addr & 0xFFFF) as isize)?;

        // Update pointers to spritemap:
        self.rom.write_u16(
            snes2pc(0x8BA0C7),
            (new_spritemap_snes_addr & 0xFFFF) as isize,
        )?;
        self.rom.write_u16(
            snes2pc(0x8BA0CD),
            (new_spritemap_snes_addr & 0xFFFF) as isize,
        )?;

        // Shift the title spritemap down:
        self.rom.write_u16(snes2pc(0x8B9B21), 0x30 + y_shift)?;
        self.rom.write_u16(snes2pc(0x8B9EBA), 0x30 + y_shift)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::patch::title::decode_tile_4bpp;
    use crate::patch::title::encode_tile_4bpp;

    #[test]
    fn decode_encode_4bpp() {
        let data: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let encoded = decode_tile_4bpp(&data);
        let decoded = encode_tile_4bpp(&encoded);
        assert_eq!(decoded, data);
    }
}
