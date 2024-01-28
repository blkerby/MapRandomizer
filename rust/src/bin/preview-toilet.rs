use hashbrown::HashMap;
use image::{Rgb, RgbImage};
use maprando::game_data::smart_xml::{self, from_hex_words, BGData, Screen};
use std::path::Path;

use anyhow::{Context, Result};

#[derive(Copy, Clone)]
struct Tile8x8 {
    idx: usize,
    palette: usize,
    flip_x: bool,
    flip_y: bool,
    priority: bool,
}

#[derive(Copy, Clone)]
struct Tile16x16 {
    top_left: Tile8x8,
    top_right: Tile8x8,
    bottom_left: Tile8x8,
    bottom_right: Tile8x8,
}

struct CRETileset {
    gfx: Vec<[[u8; 8]; 8]>,
    tiles: Vec<Tile16x16>,
}

struct SCETileset {
    palette: Vec<[u8; 3]>,
    gfx: Vec<[[u8; 8]; 8]>,
    tiles: Vec<Tile16x16>,
}

fn decode_8x8_tile_data_4bpp(data: &[u8]) -> Result<[[u8; 8]; 8]> {
    let mut out: [[u8; 8]; 8] = [[0; 8]; 8];
    for y in 0..8 {
        let addr = y * 2;
        let data_0 = data[addr];
        let data_1 = data[addr + 1];
        let data_2 = data[addr + 16];
        let data_3 = data[addr + 17];
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

fn decode_8x8_tile(data: &[u8]) -> Tile8x8 {
    let x = (data[0] as usize) | ((data[1] as usize) << 8);
    Tile8x8 {
        idx: (x & 0x3FF) as usize,
        palette: ((x >> 10) & 7) as usize,
        priority: ((x >> 13) & 1) == 1,
        flip_x: ((x >> 14) & 1) == 1,
        flip_y: ((x >> 15) & 1) == 1,
    }
}

fn decode_16x16_tile(data: &[u8]) -> Tile16x16 {
    Tile16x16 {
        top_left: decode_8x8_tile(&data[0..2]),
        top_right: decode_8x8_tile(&data[2..4]),
        bottom_left: decode_8x8_tile(&data[4..6]),
        bottom_right: decode_8x8_tile(&data[6..8]),
    }
}

fn decode_color(data: u16) -> [u8; 3] {
    let r = data & 0x1f;
    let g = (data >> 5) & 0x1f;
    let b = (data >> 10) & 0x1f;
    [(r * 8) as u8, (g * 8) as u8, (b * 8) as u8]
}

fn load_palette(palette_path: &Path) -> Result<Vec<[u8; 3]>> {
    let palette_bytes = std::fs::read(&palette_path)
        .with_context(|| format!("Unable to load palette at {}", palette_path.display()))?;
    let mut palette: Vec<[u8; 3]> = vec![];
    let mut i = 0;
    while i < palette_bytes.len() {
        let data = (palette_bytes[i] as u16) | ((palette_bytes[i + 1] as u16) << 8);
        let color = decode_color(data);
        palette.push(color);
        i += 2;
    }
    Ok(palette)
}

fn load_8x8_gfx(gfx8x8_path: &Path) -> Result<Vec<[[u8; 8]; 8]>> {
    let gfx8x8_bytes = std::fs::read(&gfx8x8_path)
        .with_context(|| format!("Unable to load CRE 8x8 gfx at {}", gfx8x8_path.display()))?;
    let mut gfx: Vec<[[u8; 8]; 8]> = vec![];
    let mut i = 0;
    while i < gfx8x8_bytes.len() {
        let tile = decode_8x8_tile_data_4bpp(&gfx8x8_bytes[i..(i + 32)])?;
        gfx.push(tile);
        i += 32;
    }
    Ok(gfx)
}

fn load_16x16_gfx(gfx16x16_path: &Path) -> Result<Vec<Tile16x16>> {
    let gfx16x16_bytes = std::fs::read(&gfx16x16_path).with_context(|| {
        format!(
            "Unable to load CRE 16x16 tiles at {}",
            gfx16x16_path.display()
        )
    })?;
    let mut tiles: Vec<Tile16x16> = vec![];
    let mut i = 0;
    while i < gfx16x16_bytes.len() {
        let tile = decode_16x16_tile(&gfx16x16_bytes[i..(i + 8)]);
        tiles.push(tile);
        i += 8;
    }
    Ok(tiles)
}

fn load_cre_tileset(tileset_path: &Path) -> Result<CRETileset> {
    let gfx = load_8x8_gfx(&tileset_path.join("8x8tiles.gfx"))?;
    let tiles = load_16x16_gfx(&tileset_path.join("16x16tiles.ttb"))?;
    Ok(CRETileset { gfx, tiles })
}

fn load_sce_tileset(tileset_path: &Path, cre_tileset: &CRETileset) -> Result<SCETileset> {
    let palette = load_palette(&tileset_path.join("palette.snes"))?;
    let mut gfx = load_8x8_gfx(&tileset_path.join("8x8tiles.gfx"))?;
    let sce_tiles = load_16x16_gfx(&tileset_path.join("16x16tiles.ttb"))?;
    
    gfx.extend(cre_tileset.gfx.clone());
    let mut tiles = cre_tileset.tiles.clone();
    tiles.extend(sce_tiles);
    Ok(SCETileset {
        palette,
        gfx,
        tiles,
    })
}

fn render_tile_8x8(image: &mut RgbImage, x0: usize, y0: usize, tile: Tile8x8, tileset: &SCETileset) {
    let gfx = &tileset.gfx[tile.idx];
    for y in 0..8 {
        for x in 0..8 {
            let x1 = if tile.flip_x { 7 - x } else { x };
            let y1 = if tile.flip_y { 7 - y } else { y };
            if gfx[y1][x1] == 0 {
                // Skip transparent pixels
                continue;
            }
            let color_idx = tile.palette * 16 + (gfx[y1][x1] as usize);
            let color = tileset.palette[color_idx];
            image.put_pixel((x0 + x) as u32, (y0 + y) as u32, Rgb(color));
        }
    }
}

fn render_tile_16x16(image: &mut RgbImage, x0: usize, y0: usize, tile: Tile16x16, tileset: &SCETileset) {
    render_tile_8x8(image, x0, y0, tile.top_left, tileset);
    render_tile_8x8(image, x0 + 8, y0, tile.top_right, tileset);
    render_tile_8x8(image, x0, y0 + 8, tile.bottom_left, tileset);
    render_tile_8x8(image, x0 + 8, y0 + 8, tile.bottom_right, tileset);
}

fn render_tileset(tileset: &SCETileset) -> RgbImage {
    let width = 512;
    let height = (tileset.tiles.len() / 2) as u32;
    let mut image: RgbImage = RgbImage::new(width, height);
    for i in 0..tileset.tiles.len() {
        let x0 = (i % 32) * 16;
        let y0 = (i / 32) * 16;
        render_tile_16x16(&mut image, x0, y0, tileset.tiles[i], tileset);
    }
    image
}

fn render_bgdata(bgdata: &BGData, image: &mut RgbImage, tileset: &SCETileset) -> Result<()> {
    for data in &bgdata.data {
        if data.type_ != "DECOMP" {
            continue;
        }
        let mut tiles = vec![];
        for (i, word_str) in data.source.split_ascii_whitespace().enumerate() {
            let word = u16::from_str_radix(word_str, 16)?;
            let tile = decode_8x8_tile(&word.to_le_bytes());
            tiles.push(tile);
            // let x = (i % 32) * 8;
            // let y = (i / 32) * 8;
            // if y >= 256 {
            //     println!("{}", y);
            // }
            // if x < image.width() as usize && y < image.height() as usize {

            //     render_tile_8x8(image, x, y, tile, tileset);
            // }
        }
        if tiles.len() == 1024 {
            for screen_y in 0..image.height() as usize / 256 {
                for screen_x in 0..image.width() as usize / 256 {
                    for (i, &tile) in tiles.iter().enumerate() {
                        let x = screen_x * 256 + (i % 32) * 8;
                        let y = screen_y * 256 + (i / 32) * 8;
                        render_tile_8x8(image, x, y, tile, tileset);
                    }
                }
            }
            // println!("{}", tiles.len());
        } else if tiles.len() == 2048 {
            for screen_y in 0..image.height() as usize / 256 {
                for screen_x2 in 0..image.width() as usize / 512 {
                    for (i, &tile) in tiles.iter().enumerate() {
                        if i < 1024 {
                            let x = screen_x2 * 512 + (i % 32) * 8;
                            let y = screen_y * 256 + (i / 32) * 8;
                            render_tile_8x8(image, x, y, tile, tileset);
                        } else {
                            let x = screen_x2 * 512 + 256 + (i % 32) * 8;
                            let y = screen_y * 256 + ((i - 1024) / 32) * 8;
                            render_tile_8x8(image, x, y, tile, tileset);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn render_screens(screens: &[Screen], image: &mut RgbImage, tileset: &SCETileset) {
    for screen in screens {
        let x0 = screen.x * 16;
        let y0 = screen.y * 16;

        for (i, data) in screen.data.iter().enumerate() {
            let x = (i % 16) + x0;
            let y = (i / 16) + y0;
            let tile_idx = (data & 0x3FF) as usize;
            let tile_flip_x = (data & 0x400) != 0;
            let tile_flip_y = (data & 0x800) != 0;
            let mut tile = tileset.tiles[tile_idx];
            if tile_flip_x {
                (tile.top_left, tile.top_right) = (tile.top_right, tile.top_left);
                (tile.bottom_left, tile.bottom_right) = (tile.bottom_right, tile.bottom_left);
                tile.top_left.flip_x = !tile.top_left.flip_x;
                tile.top_right.flip_x = !tile.top_right.flip_x;
                tile.bottom_left.flip_x = !tile.bottom_left.flip_x;
                tile.bottom_right.flip_x = !tile.bottom_right.flip_x;
            }
            if tile_flip_y {
                (tile.top_left, tile.bottom_left) = (tile.bottom_left, tile.top_left);
                (tile.top_right, tile.bottom_right) = (tile.bottom_right, tile.top_right);
                tile.top_left.flip_y = !tile.top_left.flip_y;
                tile.top_right.flip_y = !tile.top_right.flip_y;
                tile.bottom_left.flip_y = !tile.bottom_left.flip_y;
                tile.bottom_right.flip_y = !tile.bottom_right.flip_y;
            }
            render_tile_16x16(image, x * 16, y * 16, tile, tileset);
        }
    }
}

fn render_tube(image: &mut RgbImage, screen_x: usize, tileset: &SCETileset) {
    let tube_left_tile = Tile16x16 {
        top_left: Tile8x8 { idx: 0x377, palette: 0, flip_x: true, flip_y: false, priority: false },
        top_right: Tile8x8 { idx: 0x376, palette: 0, flip_x: true, flip_y: false, priority: false },
        bottom_left: Tile8x8 { idx: 0x377, palette: 0, flip_x: true, flip_y: false, priority: false },
        bottom_right: Tile8x8 { idx: 0x376, palette: 0, flip_x: true, flip_y: false, priority: false },
    };
    let tube_right_tile = Tile16x16 {
        top_left: Tile8x8 { idx: 0x376, palette: 0, flip_x: false, flip_y: false, priority: false },
        top_right: Tile8x8 { idx: 0x377, palette: 0, flip_x: false, flip_y: false, priority: false },
        bottom_left: Tile8x8 { idx: 0x376, palette: 0, flip_x: false, flip_y: false, priority: false },
        bottom_right: Tile8x8 { idx: 0x377, palette: 0, flip_x: false, flip_y: false, priority: false },
    };
    for tile_y in 0..image.height() / 16 {
        let y = (tile_y * 16) as usize;
        let x = screen_x * 256 + 128 - 16;
        render_tile_16x16(image, x, y, tube_left_tile, tileset);
        render_tile_16x16(image, x + 16, y, tube_right_tile, tileset);
    }
}

fn render_room(room_path: &Path, output_rooms_dir: &Path, tilesets: &HashMap<usize, SCETileset>) -> Result<()> {
    let room_filename = room_path.file_name().unwrap().to_str().unwrap().to_owned();
    let room_name = room_filename.strip_suffix(".xml").context("Expecting room filename to end in .xml")?;
    let room_str = std::fs::read_to_string(&room_path)
        .with_context(|| format!("Unable to load room at {}", room_path.display()))?;
    let room: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str())
        .with_context(|| format!("Unable to parse XML in {}", room_path.display()))?;

    if room.height < 2 {
        return Ok(());
    }
    println!("Processing {}", room_path.display());
    for (state_idx, state_xml) in room.states.state.iter().enumerate() {
        if state_idx != 0 {
            // For now, skip room states other than the first one.
            continue;
        }
        for tube_screen_x in 0..room.width {
            let tileset_idx = state_xml.gfx_set;
            let tileset = tilesets.get(&tileset_idx);
            if tileset.is_none() {
                println!("Skipping {}", room_name);
                return Ok(());
            }
            let tileset = tileset.unwrap();
            let width = room.width * 256;
            let height = room.height * 256;
            let mut image: RgbImage = RgbImage::new(width as u32, height as u32);

            render_bgdata(&state_xml.bg_data, &mut image, tileset)?;
            render_screens(&state_xml.level_data.layer_2.screen, &mut image, tileset);
            render_tube(&mut image, tube_screen_x, tileset);
            render_screens(&state_xml.level_data.layer_1.screen, &mut image, tileset);
            image.save(output_rooms_dir.join(format!("{}-{}.png", room_name, tube_screen_x)))?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let mosaic_dir = Path::new("../Mosaic");
    let base_dir = mosaic_dir.join("Projects/Base");
    let cre_tileset_dir = base_dir.join("Export/Tileset/CRE/00/");
    // let sce_tilesets_dir = base_dir.join("Export/Tileset/SCE");
    let sce_tilesets_dir = base_dir.join("Export/Tileset/SCE");
    let base_rooms_dir = base_dir.join("Export/Rooms/");
    let output_dir = Path::new("../tmp/preview");
    let output_tilesets_dir = &output_dir.join("tilesets");
    let output_rooms_dir = &output_dir.join("rooms");

    std::fs::create_dir_all(output_tilesets_dir)?;
    std::fs::create_dir_all(output_rooms_dir)?;
    let cre_tileset = load_cre_tileset(&cre_tileset_dir)?;
    let mut sce_tilesets: HashMap<usize, SCETileset> = HashMap::new();

    for tileset_path in std::fs::read_dir(sce_tilesets_dir)? {
        let tileset_path = tileset_path?;
        let name = tileset_path.file_name().to_str().unwrap().to_owned();
        let tileset_idx = usize::from_str_radix(&name, 16)?;
        if tileset_idx >= 0x0F && tileset_idx <= 0x14 {
            // Skip Ceres tilesets
            continue;
        }
        let tileset = load_sce_tileset(&tileset_path.path(), &cre_tileset)?;

        let img = render_tileset(&tileset);
        img.save(output_tilesets_dir.join(format!("{}.png", tileset_idx)))?;

        sce_tilesets.insert(tileset_idx, tileset);
    }


    for room_path in std::fs::read_dir(base_rooms_dir)? {
        let room_path = room_path?.path();
        render_room(&room_path, &output_rooms_dir, &sce_tilesets)?;
    }
    Ok(())
}
