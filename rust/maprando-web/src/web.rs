pub mod about;
pub mod generate;
pub mod home;
pub mod logic;
pub mod randomize;
pub mod releases;
pub mod seed;

use crate::logic_helper::LogicData;
use hashbrown::HashMap;
use maprando::{
    customize::{mosaic::MosaicTheme, samus_sprite::SamusSpriteCategory},
    map_repository::MapRepository,
    preset::{NotableData, Preset, TechData},
    seed_repository::SeedRepository,
};
use maprando_game::GameData;
use serde::Serialize;

pub const VERSION: usize = 115;

#[derive(Serialize)]
pub struct PresetData {
    pub preset: Preset,
    pub tech_setting: Vec<(TechData, bool)>,
    pub notable_setting: Vec<(NotableData, bool)>,
}

#[derive(Clone)]
pub struct VersionInfo {
    pub version: usize,
    pub dev: bool,
}

pub struct AppData {
    pub game_data: GameData,
    pub preset_data: Vec<PresetData>,
    pub map_repositories: HashMap<String, MapRepository>,
    pub seed_repository: SeedRepository,
    pub visualizer_files: Vec<(String, Vec<u8>)>, // (path, contents)
    pub video_storage_url: String,
    pub video_storage_path: Option<String>,
    pub samus_sprite_categories: Vec<SamusSpriteCategory>,
    pub logic_data: LogicData,
    pub debug: bool,
    pub port: u16,
    pub version_info: VersionInfo,
    pub static_visualizer: bool,
    pub etank_colors: Vec<Vec<String>>, // colors in HTML hex format, e.g "#ff0000"
    pub mosaic_themes: Vec<MosaicTheme>,
}
