pub mod logic;

use hashbrown::{HashSet, HashMap};
use serde_derive::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use log::info;

use crate::game_data::{self, GameData, Map};
use crate::randomize::Randomizer;
use crate::seed_repository::SeedRepository;

use self::logic::LogicData;

pub const VERSION: usize = 113;
pub const HQ_VIDEO_URL_ROOT: &'static str = "https://storage.googleapis.com/super-metroid-map-rando-videos-webm";

#[derive(Serialize, Deserialize, Clone)]
pub struct Preset {
    pub name: String,
    pub shinespark_tiles: f32,
    pub heated_shinespark_tiles: f32,
    pub shinecharge_leniency_frames: usize,
    pub resource_multiplier: f32,
    pub escape_timer_multiplier: f32,
    pub gate_glitch_leniency: usize,
    pub door_stuck_leniency: usize,
    pub phantoon_proficiency: f32,
    pub draygon_proficiency: f32,
    pub ridley_proficiency: f32,
    pub botwoon_proficiency: f32,
    pub mother_brain_proficiency: f32,
    pub tech: Vec<String>,
    pub notable_strats: Vec<String>,
}

pub struct PresetData {
    pub preset: Preset,
    pub tech_setting: Vec<(String, bool)>,
    pub implicit_tech: HashSet<String>,
    pub notable_strat_setting: Vec<(String, bool)>,
}

pub struct MapRepository {
    pub base_path: PathBuf,
    pub filenames: Vec<String>,
}


#[derive(Deserialize, Clone)]
pub struct SamusSpriteInfo {
    pub name: String,
    pub display_name: String,
    pub credits_name: Option<String>,
    pub authors: Vec<String>,
}

#[derive(Deserialize, Clone)]
pub struct SamusSpriteCategory {
    pub category_name: String,
    pub sprites: Vec<SamusSpriteInfo>,
}

#[derive(Deserialize, Clone)]
pub struct MosaicTheme {
    pub name: String,
    pub display_name: String,
}


#[derive(Clone)]
pub struct VersionInfo {
    pub version: usize,
    pub dev: bool,
}

pub struct AppData {
    pub game_data: GameData,
    pub preset_data: Vec<PresetData>,
    pub implicit_tech: HashSet<String>,
    pub map_repositories: HashMap<String, MapRepository>,
    pub seed_repository: SeedRepository,
    pub visualizer_files: Vec<(String, Vec<u8>)>, // (path, contents)
    pub tech_gif_listing: HashSet<String>,
    pub notable_gif_listing: HashSet<String>,
    pub samus_sprite_categories: Vec<SamusSpriteCategory>,
    pub logic_data: LogicData,
    // pub samus_customizer: SamusSpriteCustomizer,
    pub debug: bool,
    pub version_info: VersionInfo,
    pub static_visualizer: bool,
    pub etank_colors: Vec<Vec<String>>,  // colors in HTML hex format, e.g "#ff0000"
    pub mosaic_themes: Vec<MosaicTheme>,
    pub parallelism: usize,
}

impl MapRepository {
    pub fn new(name: &str, base_path: &Path) -> Result<Self> {
        let mut filenames: Vec<String> = Vec::new();
        for path in std::fs::read_dir(base_path)? {
            filenames.push(path?.file_name().into_string().unwrap());
        }
        filenames.sort();
        info!("{}: {} maps available ({})", name, filenames.len(), base_path.display());
        Ok(MapRepository {
            base_path: base_path.to_owned(),
            filenames,
        })
    }

    pub fn get_map(&self, attempt_num_rando: usize, seed: usize, game_data: &GameData) -> Result<Map> {
        let idx = seed % self.filenames.len();
        let path = self.base_path.join(&self.filenames[idx]);
        let map_string = std::fs::read_to_string(&path)
            .with_context(|| format!("[attempt {attempt_num_rando}] Unable to read map file at {}", path.display()))?;
        info!("[attempt {attempt_num_rando}] Map: {}", path.display());
        let mut map: Map = serde_json::from_str(&map_string)
            .with_context(|| format!("[attempt {attempt_num_rando}] Unable to parse map file at {}", path.display()))?;
        
        // Make Toilet area/subarea align with its intersecting room(s):
        // TODO: Push this upstream into the map generation
        let toilet_intersections = Randomizer::get_toilet_intersections(&map, game_data);
        if toilet_intersections.len() > 0 {
            let area = map.area[toilet_intersections[0]];
            let subarea = map.subarea[toilet_intersections[0]];
            for i in 1..toilet_intersections.len() {
                if map.area[toilet_intersections[i]] != area {
                    panic!("Mismatched areas for Toilet intersection");
                }
                if map.subarea[toilet_intersections[i]] != subarea {
                    panic!("Mismatched subareas for Toilet intersection");
                }
            }
            map.area[game_data.toilet_room_idx] = area;
            map.subarea[game_data.toilet_room_idx] = subarea;
        }
        Ok(map)
    }
}
