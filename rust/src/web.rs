pub mod logic;

use hashbrown::HashSet;
use serde_derive::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use log::info;

use crate::game_data::{GameData, Map};
use crate::seed_repository::SeedRepository;

use self::logic::LogicData;

pub const VERSION: usize = 81;

#[derive(Serialize, Deserialize, Clone)]
pub struct Preset {
    pub name: String,
    pub shinespark_tiles: usize,
    pub resource_multiplier: f32,
    pub escape_timer_multiplier: f32,
    pub gate_glitch_leniency: usize,
    pub phantoon_proficiency: f32,
    pub draygon_proficiency: f32,
    pub ridley_proficiency: f32,
    pub botwoon_proficiency: f32,
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

pub struct AppData {
    pub game_data: GameData,
    pub preset_data: Vec<PresetData>,
    pub ignored_notable_strats: HashSet<String>,
    pub implicit_tech: HashSet<String>,
    pub map_repository: MapRepository,
    pub seed_repository: SeedRepository,
    pub visualizer_files: Vec<(String, Vec<u8>)>, // (path, contents)
    pub tech_gif_listing: HashSet<String>,
    pub notable_gif_listing: HashSet<String>,
    pub samus_sprite_categories: Vec<SamusSpriteCategory>,
    pub logic_data: LogicData,
    // pub samus_customizer: SamusSpriteCustomizer,
    pub debug: bool,
    pub static_visualizer: bool,
    pub etank_colors: Vec<Vec<String>>,  // colors in HTML hex format, e.g "#ff0000"
}

impl MapRepository {
    pub fn new(base_path: &Path) -> Result<Self> {
        let mut filenames: Vec<String> = Vec::new();
        for path in std::fs::read_dir(base_path)? {
            filenames.push(path?.file_name().into_string().unwrap());
        }
        filenames.sort();
        info!("{} maps available", filenames.len());
        Ok(MapRepository {
            base_path: base_path.to_owned(),
            filenames,
        })
    }

    pub fn get_map(&self, seed: usize) -> Result<Map> {
        let idx = seed % self.filenames.len();
        let path = self.base_path.join(&self.filenames[idx]);
        let map_string = std::fs::read_to_string(&path)
            .with_context(|| format!("Unable to read map file at {}", path.display()))?;
        info!("Map: {}", path.display());
        let map: Map = serde_json::from_str(&map_string)
            .with_context(|| format!("Unable to parse map file at {}", path.display()))?;
        Ok(map)
    }

    pub fn get_vanilla_map(&self) -> Result<Map> {
        let path = Path::new("data/vanilla_map.json");
        let map_string = std::fs::read_to_string(&path)
            .with_context(|| format!("Unable to read map file at {}", path.display()))?;
        info!("Map: {}", path.display());
        let map: Map = serde_json::from_str(&map_string)
            .with_context(|| format!("Unable to parse map file at {}", path.display()))?;
        Ok(map)
    }
}
