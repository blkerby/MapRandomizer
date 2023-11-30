use anyhow::{Context, Result};
use hashbrown::HashMap;
use std::path::Path;

use crate::game_data::smart_xml::{self, Layer2Type};

use super::smart_xml::Screen;

// TODO: don't hard-code this
static COMPRESSED_LOOKUP_PATH: &'static str = "../compressed_data";

fn compress_lookup(data: &[u8], name: &str) -> Result<Vec<u8>> {
    let digest = crypto_hash::hex_digest(crypto_hash::Algorithm::SHA256, &data);
    let path = Path::new(COMPRESSED_LOOKUP_PATH).join(digest);
    let data = std::fs::read(path).with_context(|| {
        format!(
            "Unable to read compressed data for {}. Need to re-run compress-retiling?",
            name
        )
    })?;
    Ok(data)
}

#[derive(Debug, Clone)]
pub struct BGDataReference {
    pub room_area: usize,
    pub room_index: usize,
    pub room_state_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FX1Reference {
    pub room_area: usize,
    pub room_index: usize,
    pub state_index: usize,
    pub fx_index: usize,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct FX1Door {
    pub room_area: usize,
    pub room_index: usize,
    pub door_index: usize,
}

pub struct FX1 {
    pub fx1_reference: Option<FX1Reference>,
    pub fx1_door: Option<FX1Door>,
    pub fx1_data: smart_xml::FX1,
}

pub struct RetiledRoomState {
    pub tileset_idx: u8,
    pub compressed_level_data: Vec<u8>,
    pub layer2_type: smart_xml::Layer2Type,
    pub bgdata_content: smart_xml::BGData,
    pub bgdata_reference: Option<BGDataReference>,
    pub bg_scroll_speed_x: u8,
    pub bg_scroll_speed_y: u8,
    pub fx1: Vec<FX1>,
    pub setup_asm: u16,
    pub main_asm: u16,
}

pub struct RetiledRoom {
    pub path: String,
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

fn extract_screen_words(screen: &Screen, out: &mut [u8], width: usize, _height: usize) {
    let base_pos = (screen.y * width * 256 + screen.x * 16) * 2;
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

fn extract_screen_bytes(screen: &Screen, out: &mut [u8], width: usize, _height: usize) {
    let base_pos = screen.y * width * 256 + screen.x * 16;
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

pub fn extract_uncompressed_level_data(state_xml: &smart_xml::RoomState) -> Vec<u8> {
    let height = state_xml.level_data.height;
    let width = state_xml.level_data.width;
    let num_tiles = height * width * 256;
    let level_data_size = if state_xml.layer2_type == Layer2Type::Layer2 {
        2 + num_tiles * 5
    } else {
        2 + num_tiles * 3
    };
    let mut level_data = vec![0u8; level_data_size];
    level_data[0] = ((num_tiles * 2) & 0xFF) as u8;
    level_data[1] = ((num_tiles * 2) >> 8) as u8;
    extract_all_screen_words(
        &state_xml.level_data.layer_1.screen,
        &mut level_data[2..],
        width,
        height,
    );
    extract_all_screen_bytes(
        &state_xml.level_data.bts.screen,
        &mut level_data[2 + num_tiles * 2..],
        width,
        height,
    );
    if state_xml.layer2_type == Layer2Type::Layer2 {
        extract_all_screen_words(
            &state_xml.level_data.layer_2.screen,
            &mut level_data[2 + num_tiles * 3..],
            width,
            height,
        );
    }
    level_data
}

fn make_fx1(fx1: &smart_xml::FX1) -> FX1 {
    FX1 {
        fx1_reference: None,
        fx1_door: if fx1.default {
            None
        } else {
            Some(FX1Door {
                room_area: fx1.roomarea,
                room_index: fx1.roomindex,
                door_index: fx1.fromdoor,
            })
        },
        fx1_data: fx1.clone(),
    }
}

fn load_room(room_path: &Path) -> Result<RetiledRoom> {
    let room_str = std::fs::read_to_string(room_path)
        .with_context(|| format!("Unable to load room at {}", room_path.display()))?;
    let room: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str())
        .with_context(|| format!("Unable to parse XML in {}", room_path.display()))?;
    let mut states: Vec<RetiledRoomState> = vec![];
    for state_xml in &room.states.state {
        let level_data = extract_uncompressed_level_data(state_xml);
        let compressed_level_data = compress_lookup(&level_data, room_path.to_str().unwrap())?;

        let state = RetiledRoomState {
            tileset_idx: state_xml.gfx_set as u8,
            compressed_level_data,
            layer2_type: state_xml.layer2_type,
            bgdata_content: state_xml.bg_data.clone(),
            bgdata_reference: None,
            bg_scroll_speed_x: state_xml.layer2_xscroll as u8,
            bg_scroll_speed_y: state_xml.layer2_yscroll as u8,
            fx1: state_xml.fx1s.fx1.iter().map(make_fx1).collect(),
            setup_asm: state_xml.layer1_2 as u16,
            main_asm: state_xml.fx2 as u16,
        };
        states.push(state);
    }
    Ok(RetiledRoom {
        path: room_path.as_os_str().to_str().unwrap().to_string(),
        area: room.area,
        index: room.index,
        states,
    })
}

fn load_all_rooms(project_path: &Path) -> Result<Vec<RetiledRoom>> {
    let rooms_dir_path = project_path.join("Export/Rooms/");
    let mut out = vec![];
    let room_it = std::fs::read_dir(&rooms_dir_path)
        .with_context(|| format!("Unable to list rooms at {}", rooms_dir_path.display()))?;
    for room_file_path in room_it {
        let room_file_path = room_file_path?;
        out.push(load_room(&room_file_path.path())?);
    }
    Ok(out)
}

fn load_cre_tileset(mosaic_path: &Path) -> Result<RetiledCRETileset> {
    let tileset_path = mosaic_path.join("Projects/Base/Export/Tileset/CRE/00/");

    let gfx8x8_path = tileset_path.join("8x8tiles.gfx");
    let gfx8x8_bytes = std::fs::read(&gfx8x8_path)
        .with_context(|| format!("Unable to load CRE 8x8 gfx at {}", gfx8x8_path.display()))?;
    let compressed_gfx8x8 = compress_lookup(&gfx8x8_bytes, gfx8x8_path.to_str().unwrap())?;

    let gfx16x16_path = tileset_path.join("16x16tiles.ttb");
    let gfx16x16_bytes = std::fs::read(&gfx16x16_path).with_context(|| {
        format!(
            "Unable to load CRE 16x16 gfx at {}",
            gfx16x16_path.display()
        )
    })?;
    let compressed_gfx16x16 = compress_lookup(&gfx16x16_bytes, gfx16x16_path.to_str().unwrap())?;

    Ok(RetiledCRETileset {
        compressed_gfx8x8,
        compressed_gfx16x16,
    })
}

fn load_all_sce_tilesets(project_path: &Path) -> Result<HashMap<usize, RetiledSCETileset>> {
    let tilesets_path = project_path.join("Export/Tileset/SCE");
    let mut out = HashMap::new();
    let tileset_it = std::fs::read_dir(&tilesets_path)
        .with_context(|| format!("Unable to list tilesets at {}", tilesets_path.display()))?;
    for tileset_dir in tileset_it {
        let tileset_dir = tileset_dir?;
        let tileset_idx = usize::from_str_radix(tileset_dir.file_name().to_str().unwrap(), 16)?;
        let tileset_path = tileset_dir.path();

        let palette_path = tileset_path.join("palette.snes");
        let palette_bytes = std::fs::read(&palette_path)
            .with_context(|| format!("Unable to read palette at {}", palette_path.display()))?;
        let compressed_palette = compress_lookup(&palette_bytes, palette_path.to_str().unwrap())?;

        let gfx8x8_path = tileset_path.join("8x8tiles.gfx");
        let gfx8x8_bytes = std::fs::read(&gfx8x8_path)
            .with_context(|| format!("Unable to read 8x8 gfx at {}", gfx8x8_path.display()))?;
        let compressed_gfx8x8 = compress_lookup(&gfx8x8_bytes, gfx8x8_path.to_str().unwrap())?;

        let gfx16x16_path = tileset_path.join("16x16tiles.ttb");
        let gfx16x16_bytes = std::fs::read(&gfx16x16_path)
            .with_context(|| format!("Unable to read 16x16 gfx at {}", gfx16x16_path.display()))?;
        let compressed_gfx16x16 =
            compress_lookup(&gfx16x16_bytes, gfx16x16_path.to_str().unwrap())?;

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
    let base_path = mosaic_path.join("Projects").join("Base");
    let project_path = mosaic_path.join("Projects").join(theme_name);
    let sce_tilesets = load_all_sce_tilesets(&base_path)?;
    let rooms = load_all_rooms(&project_path)?;

    Ok(RetiledTheme {
        name: theme_name.to_string(),
        rooms,
        sce_tilesets,
    })
}

fn make_bgdata_mapping(base_theme: &RetiledTheme) -> HashMap<smart_xml::BGData, BGDataReference> {
    let mut out = HashMap::new();
    for room in &base_theme.rooms {
        for (i, state) in room.states.iter().enumerate() {
            out.insert(
                state.bgdata_content.clone(),
                BGDataReference {
                    room_area: room.area,
                    room_index: room.index,
                    room_state_index: i,
                },
            );
        }
    }
    out
}

fn resolve_bgdata_references(
    theme: &mut RetiledTheme,
    bg_mapping: &HashMap<smart_xml::BGData, BGDataReference>,
) {
    for room in &mut theme.rooms {
        for (i, state) in (&mut room.states).iter_mut().enumerate() {
            let r = bg_mapping.get(&state.bgdata_content);
            if let Some(bg_ref) = r {
                state.bgdata_reference = Some(bg_ref.clone());
            } else {
                panic!("Unrecognized BGData in room {}, state {}", room.path, i);
            }
        }
    }
}

fn make_fx_mapping(base_theme: &RetiledTheme) -> HashMap<FX1Door, FX1Reference> {
    let mut out = HashMap::new();
    for room in &base_theme.rooms {
        for (i, state) in room.states.iter().enumerate() {
            for (j, fx) in state.fx1.iter().enumerate() {
                if let Some(door) = &fx.fx1_door {
                    out.insert(
                        door.clone(),
                        FX1Reference {
                            room_area: room.area,
                            room_index: room.index,
                            state_index: i,
                            fx_index: j,
                        },
                    );
                }
            }
        }
    }
    out
}

fn resolve_fx_references(theme: &mut RetiledTheme, fx_mapping: &HashMap<FX1Door, FX1Reference>) {
    for room in &mut theme.rooms {
        for (i, state) in (&mut room.states).iter_mut().enumerate() {
            for (j, fx) in state.fx1.iter_mut().enumerate() {
                if let Some(door) = &fx.fx1_door {
                    let r = fx_mapping.get(door);

                    if let Some(fx_ref) = r {
                        fx.fx1_reference = Some(fx_ref.clone());
                    } else {
                        panic!(
                            "Unrecognized FX door in room {}, state {}, FX {}",
                            room.path, i, j
                        );
                    }
                }
            }
        }
    }
}

pub fn load_theme_data(mosaic_path: &Path) -> Result<RetiledThemeData> {
    let cre_tileset = load_cre_tileset(mosaic_path)?;
    let theme_names = [
        "Base",
        "OuterCrateria",
        "InnerCrateria",
        "GreenBrinstar",
        "UpperNorfair",
    ];
    let mut themes = HashMap::new();
    for name in theme_names {
        themes.insert(name.to_string(), load_theme(mosaic_path, name)?);
    }
    let bg_mapping = make_bgdata_mapping(&themes["Base"]);
    for theme in themes.values_mut() {
        resolve_bgdata_references(theme, &bg_mapping);
    }
    let fx_mapping = make_fx_mapping(&themes["Base"]);
    for theme in themes.values_mut() {
        resolve_fx_references(theme, &fx_mapping);
    }
    Ok(RetiledThemeData {
        themes,
        cre_tileset,
    })
}
