use hashbrown::{HashMap, HashSet};
use maprando::customize::mosaic::MosaicTheme;
use maprando::customize::samus_sprite::SamusSpriteCategory;
use maprando::game_data::GameData;
use maprando::map_repository::MapRepository;
use maprando::preset::Preset;
use maprando::seed_repository::SeedRepository;

use crate::logic::LogicData;

pub const VERSION: usize = 114;
pub const HQ_VIDEO_URL_ROOT: &'static str =
    "https://storage.googleapis.com/super-metroid-map-rando-videos-webm";

pub struct PresetData {
    pub preset: Preset,
    pub tech_setting: Vec<(String, bool)>,
    pub implicit_tech: HashSet<String>,
    pub notable_strat_setting: Vec<(String, bool)>,
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
    pub debug: bool,
    pub version_info: VersionInfo,
    pub static_visualizer: bool,
    pub etank_colors: Vec<Vec<String>>, // colors in HTML hex format, e.g "#ff0000"
    pub mosaic_themes: Vec<MosaicTheme>,
}
