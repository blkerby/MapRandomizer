use anyhow::Result;
use hashbrown::HashMap;
use std::path::Path;
use serde_xml_rs::from_str;

use crate::{patch::compress::compress, game_data::smart_xml::{self, Layer2Type}};

use super::{RoomPtr, smart_xml::Screen};

pub struct RetiledRoomState {
    pub tileset_idx: u8,
    pub compressed_level_data: Vec<u8>,
    pub bgdata_ptr: Option<u16>,
    // TODO: add FX
}

pub struct RetiledRoom {
    pub area: usize,
    pub index: usize,
    pub states: Vec<RetiledRoomState>,
}

pub struct RetiledCRETileset {
    pub compressed_gfx8x8: Vec<u8>,
    pub compressed_gfx16x16: Vec<u8>,
}

pub struct RetiledSCETileset {
    pub compressed_palette: Vec<u8>,
    pub compressed_gfx8x8: Vec<u8>,
    pub compressed_gfx16x16: Vec<u8>,
}

pub struct RetiledTheme {
    pub name: String,
    pub rooms: Vec<RetiledRoom>,
    pub sce_tilesets: HashMap<usize, RetiledSCETileset>,
}

pub struct RetiledThemeData {
    pub themes: HashMap<String, RetiledTheme>,
    pub cre_tileset: RetiledCRETileset,
}

fn extract_screen_words(screen: &Screen, out: &mut [u8], width: usize, height: usize) {
    let base_pos = (screen.y * width + screen.x) * 16 * 2;
    assert!(screen.data.len() == 256);
    for y in 0..16 {
        for x in 0..16 {
            let c = screen.data[y * 16 + x];
            let pos = base_pos + (y * width * 16 + x) * 2;
            out[pos] = (c & 0xFF) as u8;
            out[pos + 1] = (c >> 8) as u8;
        }
    }
}

fn extract_screen_bytes(screen: &Screen, out: &mut [u8], width: usize, height: usize) {
    let base_pos = (screen.y * width + screen.x) * 16;
    assert!(screen.data.len() == 256);
    for y in 0..16 {
        for x in 0..16 {
            let c = screen.data[y * 16 + x];
            let pos = base_pos + y * width * 16 + x;
            out[pos] = c as u8;
        }  
    }
}

fn extract_all_screen_words(screens: &[Screen], out: &mut [u8], width: usize, height: usize) {
    for screen in screens {
        extract_screen_words(screen, out, width, height);
    }
}

fn extract_all_screen_bytes(screens: &[Screen], out: &mut [u8], width: usize, height: usize) {
    for screen in screens {
        extract_screen_bytes(screen, out, width, height);
    }
}

fn load_room(room_path: &Path) -> Result<RetiledRoom> {
    let room_str = std::fs::read_to_string(room_path)?;
    println!("{}: {}", room_path.display(), room_str.len());
    let room: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str()).unwrap();
    let mut states: Vec<RetiledRoomState> = vec![];
    for state_xml in &room.states.state {
        let height = state_xml.level_data.height;
        let width = state_xml.level_data.width;
        let num_tiles = height * width * 256;
        let level_data_size = if state_xml.layer2_type == Layer2Type::Layer2 { num_tiles * 5 } else { num_tiles * 3};
        let mut level_data = vec![0u8; level_data_size];
        extract_all_screen_words(&state_xml.level_data.layer_1.screen, &mut level_data, width, height);
        extract_all_screen_bytes(&state_xml.level_data.bts.screen, &mut level_data[num_tiles * 2..], width, height);
        if state_xml.layer2_type == Layer2Type::Layer2 {
            extract_all_screen_words(&state_xml.level_data.layer_2.screen, &mut level_data[num_tiles * 3..], width, height);
        }
        let compressed_level_data = compress(&level_data);
        let state = RetiledRoomState {
            tileset_idx: state_xml.gfx_set as u8,
            compressed_level_data,
            bgdata_ptr: None,
        };
        states.push(state);
    }
    // println!("{:?}", room);
    Ok(RetiledRoom {
        area: room.area,
        index: room.index,
        states: vec![],
    })
}

fn load_all_rooms(project_path: &Path) -> Result<Vec<RetiledRoom>> {
    let rooms_dir_path = project_path.join("Export/Rooms/");
    let mut out = vec![];

    for room_file_path in std::fs::read_dir(rooms_dir_path)? {
        let room_file_path = room_file_path?;
        out.push(load_room(&room_file_path.path())?);
    }
    Ok(out)
}

fn load_cre_tileset(mosaic_path: &Path) -> Result<RetiledCRETileset> {
    let tileset_path = mosaic_path.join("Projects/Base/Export/Tileset/CRE/00/");

    let gfx8x8_path = tileset_path.join("8x8tiles.gfx");
    let gfx8x8_bytes = std::fs::read(gfx8x8_path)?;
    let compressed_gfx8x8 = compress(&gfx8x8_bytes);

    let gfx16x16_path = tileset_path.join("16x16tiles.ttb");
    let gfx16x16_bytes = std::fs::read(gfx16x16_path)?;
    let compressed_gfx16x16 = compress(&gfx16x16_bytes);

    Ok(RetiledCRETileset {
        compressed_gfx8x8,
        compressed_gfx16x16,
    })
}

fn load_all_sce_tilesets(project_path: &Path) -> Result<HashMap<usize, RetiledSCETileset>> {
    let tilesets_path = project_path.join("Export/Tileset/SCE");
    let mut out = HashMap::new();

    for tileset_dir in std::fs::read_dir(tilesets_path)? {
        let tileset_dir = tileset_dir?;
        let tileset_idx =
            usize::from_str_radix(tileset_dir.file_name().to_str().unwrap(), 16)?;
        let tileset_path = tileset_dir.path();

        let palette_path = tileset_path.join("palette.snes");
        let palette_bytes = std::fs::read(palette_path)?;
        let compressed_palette = compress(&palette_bytes);

        let gfx8x8_path = tileset_path.join("8x8tiles.gfx");
        let gfx8x8_bytes = std::fs::read(gfx8x8_path)?;
        let compressed_gfx8x8 = compress(&gfx8x8_bytes);
    
        let gfx16x16_path = tileset_path.join("16x16tiles.ttb");
        let gfx16x16_bytes = std::fs::read(gfx16x16_path)?;
        let compressed_gfx16x16 = compress(&gfx16x16_bytes);

        let tileset = RetiledSCETileset {
            compressed_palette,
            compressed_gfx8x8,
            compressed_gfx16x16,
        };
        out.insert(tileset_idx, tileset);
    }
    Ok(out)
}

fn load_theme(mosaic_path: &Path, theme_name: &str) -> Result<RetiledTheme> {
    let project_path = mosaic_path.join("Projects").join(theme_name);
    let sce_tilesets = load_all_sce_tilesets(&project_path)?;
    let rooms = load_all_rooms(&project_path)?;

    Ok(RetiledTheme {
        name: theme_name.to_string(),
        rooms,
        sce_tilesets,
    })
}

pub fn load_theme_data(mosaic_path: &Path) -> Result<RetiledThemeData> {
    let cre_tileset = load_cre_tileset(mosaic_path)?;
    let mut themes = HashMap::new();
    themes.insert("Base".to_string(), load_theme(mosaic_path, "Base")?);
    Ok(RetiledThemeData {
        themes,
        cre_tileset,
    })
}
