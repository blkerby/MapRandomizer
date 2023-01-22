use std::path::Path;

use crate::{game_data::IndexedVec, compress::compress};

use std::cmp::min;
use super::{PcAddr, Rom, pc2snes, snes2pc};
use anyhow::{Context, Result};
use image::{io::Reader as ImageReader, Rgb};
use ndarray::{Array2, Array3, concatenate, Axis};
use slice_of_array::prelude::*;

pub struct TitlePatcher<'a> {
    rom: &'a mut Rom,
    next_free_space_pc: usize,
}

fn read_image(path: &Path) -> Result<Array3<u8>> {
    let img = ImageReader::open(path)
        .with_context(|| format!("Unable to open image: {}", path.display()))?
        .decode()
        .with_context(|| format!("Unable to decode image: {}", path.display()))?
        .to_rgb8();
    let width = img.width() as usize;
    let height = img.height() as usize;
    let mut arr: Array3<u8> = Array3::zeros([height, width, 3]);
    for y in 0..height {
        for x in 0..width {
            let &Rgb([r, g, b]) = img.get_pixel(x as u32, y as u32);
            arr[[y, x, 0]] = r;
            arr[[y, x, 1]] = g;
            arr[[y, x, 2]] = b;
        }
    }
    Ok(arr)
}

fn rgb_to_u16(rgb: (u8, u8, u8)) -> u16 {
    let (r, g, b) = rgb;
    (r as u16) | (g as u16) << 5 | (b as u16) << 10
}

struct Graphics {
    palette: Vec<(u8, u8, u8)>,
    tiles: Vec<[[u8; 8]; 8]>,    // indices into `palette`
    tilemap: Array2<u8>,        // indices into `tiles`
}

fn encode_graphics(image: &Array3<u8>, quantize_factor: isize) -> Graphics {
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
    process_tile(16, 15);  // blank tile
    for tile_y in 0..(height / 8) {
        for tile_x in 0..(width / 8) {
            process_tile(tile_y, tile_x);
        }
    }
    assert!(tile_isv.keys.len() <= 256);

    let mut new_tiles: Vec<[[u8; 8]; 8]> = Vec::new();
    let mut color_isv: IndexedVec<(u8, u8, u8)> = IndexedVec::default();
    color_isv.add(&(0, 0, 0));  // Keep the black color
    for tile in &tile_isv.keys {
        let mut new_tile: [[u8; 8]; 8] = [[0; 8]; 8];
        for y in 0..8 {
            for x in 0..8 {
                let pixel = tile[y][x].map(|x| x as isize);
                // let r = (pixel[0] / 8 + quantize_factor / 2) / quantize_factor * quantize_factor;
                // let g = (pixel[1] / 8 + quantize_factor / 2) / quantize_factor * quantize_factor;
                // let b = (pixel[2] / 8 + quantize_factor / 2) / quantize_factor * quantize_factor;
                let r = min(31, (pixel[0] as isize + quantize_factor * 4) / (quantize_factor * 8) * quantize_factor);
                let g = min(31, (pixel[1] as isize + quantize_factor * 4) / (quantize_factor * 8) * quantize_factor);
                let b = min(31, (pixel[2] as isize + quantize_factor * 4) / (quantize_factor * 8) * quantize_factor);
                // let r = (pixel[0] / 8) / quantize_factor * quantize_factor;
                // let g = (pixel[1] / 8) / quantize_factor * quantize_factor;
                // let b = (pixel[2] / 8) / quantize_factor * quantize_factor;
                let idx = color_isv.add(&(r as u8, g as u8, b as u8)) as u8;
                new_tile[y][x] = idx;
            }    
        }
        new_tiles.push(new_tile);
    }


    Graphics {
        palette: color_isv.keys,
        tiles: new_tiles,
        tilemap
    }
}

impl<'a> TitlePatcher<'a> {
    pub fn new(rom: &'a mut Rom) -> Self {
        Self {
            rom,
            next_free_space_pc: 0x1C0000,
        }
    }

    fn write_to_free_space(&mut self, data: &[u8]) -> Result<PcAddr> {
        let free_space = self.next_free_space_pc;
        self.rom.write_n(free_space, data)?;
        self.next_free_space_pc += data.len();
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
        let compressed = compress(&flat_tiles);
        let gfx_pc_addr = self.write_to_free_space(&compressed)?;        
        let gfx_snes_addr = pc2snes(gfx_pc_addr);
        // Update reference to tile GFX:
        self.rom.write_u8(snes2pc(0x8B9BA8), (gfx_snes_addr >> 16) as isize)?;
        self.rom.write_u16(snes2pc(0x8B9BAC), (gfx_snes_addr & 0xFFFF) as isize)?;
        Ok(())
    }

    fn write_title_background_tilemap(&mut self, tilemap: &Array2<u8>) -> Result<()> {
        let tilemap_horizontal_padding: Array2<u8> = Array2::zeros([28, 96]) + 4;
        let padded_tilemap0 = concatenate(Axis(1), &[tilemap.view(), tilemap_horizontal_padding.view()])?;
        let tilemap_vertical_padding: Array2<u8> = Array2::zeros([4, 128]) + 4;
        let padded_tilemap = concatenate(Axis(0), &[padded_tilemap0.view(), tilemap_vertical_padding.view()])?;        
        let compressed = compress(&padded_tilemap.as_standard_layout().as_slice().unwrap());
        let tilemap_pc_addr = self.write_to_free_space(&compressed)?;        
        let tilemap_snes_addr = pc2snes(tilemap_pc_addr);
        // Update reference to tilemap:
        self.rom.write_u8(snes2pc(0x8B9BB9), (tilemap_snes_addr >> 16) as isize)?;
        self.rom.write_u16(snes2pc(0x8B9BBD), (tilemap_snes_addr & 0xFFFF) as isize)?;
        Ok(()) 
    }

    pub fn patch_title_background(&mut self) -> Result<()> {
        let image_path = Path::new("../gfx/title/Title3.png");
        let quantize_factor = 8;

        let img = read_image(image_path)?;
        assert!(img.dim() == (224, 256, 3));

        // Compute title background palette, tile GFX, and tilemap:
        let graphics = encode_graphics(&img, quantize_factor);
        println!("Title background distinct colors: {}", graphics.palette.len());
      
        // Write palette, tile GFX, and tilemap:
        self.write_palette(0x661E9, &graphics.palette)?;
        self.write_title_background_tiles(&graphics.tiles)?;
        self.write_title_background_tilemap(&graphics.tilemap)?;

        // Skip palette FX handler
        self.rom.write_n(snes2pc(0x8B9A34), &[0xEA; 4])?;

        // Use white color for Nintendo copyright text (otherwise it would stay black since we skip the palette FX handler)
        self.rom.write_u16(0x661E9 + 0xC9 * 2, 0x7FFF)?;
    
        Ok(())
    }
}
