use anyhow::Result;
use maprando_game::{Item, NotableId, RoomId, TechId};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct RandomizerSettings {
    pub version: usize,
    pub name: Option<String>,
    pub skill_assumption_settings: SkillAssumptionSettings,
    pub item_progression_settings: ItemProgressionSettings,
    pub quality_of_life_settings: QualityOfLifeSettings,
    pub objectives_mode: ObjectivesMode,
    pub map_layout: String,
    pub doors_mode: DoorsMode,
    pub start_location_mode: StartLocationMode,
    pub save_animals: SaveAnimals,
    pub other_settings: OtherSettings,
    #[serde(default)]
    pub debug: bool,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct SkillAssumptionSettings {
    pub preset: Option<String>,
    pub shinespark_tiles: f32,
    pub heated_shinespark_tiles: f32,
    pub speed_ball_tiles: f32,
    pub shinecharge_leniency_frames: i32,
    pub resource_multiplier: f32,
    pub gate_glitch_leniency: i32,
    pub door_stuck_leniency: i32,
    pub phantoon_proficiency: f32,
    pub draygon_proficiency: f32,
    pub ridley_proficiency: f32,
    pub botwoon_proficiency: f32,
    pub mother_brain_proficiency: f32,
    pub escape_timer_multiplier: f32,
    pub tech_settings: Vec<TechSetting>,
    pub notable_settings: Vec<NotableSetting>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct TechSetting {
    pub id: TechId,
    pub name: String,
    pub enabled: bool,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct NotableSetting {
    pub room_id: RoomId,
    pub notable_id: NotableId,
    pub room_name: String,
    pub notable_name: String,
    pub enabled: bool,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct ItemProgressionSettings {
    pub preset: Option<String>,
    pub progression_rate: ProgressionRate,
    pub item_placement_style: ItemPlacementStyle,
    pub item_priority_strength: ItemPriorityStrength,
    pub random_tank: bool,
    pub spazer_before_plasma: bool,
    pub item_pool_preset: Option<ItemPoolPreset>,
    pub stop_item_placement_early: bool,
    pub ammo_collect_fraction: f32,
    pub item_pool: Vec<ItemCount>,
    pub starting_items_preset: Option<StartingItemsPreset>,
    pub starting_items: Vec<ItemCount>,
    pub key_item_priority: Vec<KeyItemPrioritySetting>,
    pub filler_items: Vec<FillerItemPrioritySetting>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct ItemCount {
    pub item: Item,
    pub count: usize,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct KeyItemPrioritySetting {
    pub item: Item,
    pub priority: KeyItemPriority,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct FillerItemPrioritySetting {
    pub item: Item,
    pub priority: FillerItemPriority,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub enum ItemPoolPreset {
    Full,
    Reduced,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub enum StartingItemsPreset {
    None,
    All,
}
#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct QualityOfLifeSettings {
    pub preset: Option<String>,
    // Map:
    pub item_markers: ItemMarkers,
    pub mark_map_stations: bool,
    pub room_outline_revealed: bool,
    pub opposite_area_revealed: bool,
    // End game:
    pub mother_brain_fight: MotherBrainFight,
    pub supers_double: bool,
    pub escape_movement_items: bool,
    pub escape_refill: bool,
    pub escape_enemies_cleared: bool,
    // Faster transitions:
    pub fast_elevators: bool,
    pub fast_doors: bool,
    pub fast_pause_menu: bool,
    // Samus control
    pub respin: bool,
    pub infinite_space_jump: bool,
    pub momentum_conservation: bool,
    // Tweaks to unintuitive vanilla behavior:
    pub all_items_spawn: bool,
    pub acid_chozo: bool,
    pub remove_climb_lava: bool,
    // Other:
    pub buffed_drops: bool,
    pub early_save: bool,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct OtherSettings {
    pub wall_jump: WallJump,
    pub etank_refill: ETankRefill,
    pub area_assignment: AreaAssignment,
    pub item_dot_change: ItemDotChange,
    pub transition_letters: bool,
    pub door_locks_size: DoorLocksSize,
    pub maps_revealed: MapsRevealed,
    pub map_station_reveal: MapStationReveal,
    pub energy_free_shinesparks: bool,
    pub ultra_low_qol: bool,
    pub race_mode: bool,
    pub random_seed: Option<usize>,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub enum ProgressionRate {
    Slow,
    Uniform,
    Fast,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub enum ItemPlacementStyle {
    Neutral,
    Forced,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub enum ItemPriorityStrength {
    Moderate,
    Heavy,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Hash, Eq, Default)]
pub enum KeyItemPriority {
    Early,
    #[default]
    Default,
    Late,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum FillerItemPriority {
    No,
    Semi,
    Yes,
    Early,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum DoorLocksSize {
    Small,
    Large,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ItemMarkers {
    Simple,
    Majors,
    Uniques,
    #[serde(rename = "3-Tiered")]
    ThreeTiered,
    #[serde(rename = "4-Tiered")]
    FourTiered,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ItemDotChange {
    Fade,
    Disappear,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ObjectivesMode {
    None,
    Bosses,
    Minibosses,
    Metroids,
    Chozos,
    Pirates,
    Random,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum DoorsMode {
    Blue,
    Ammo,
    Beam,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum StartLocationMode {
    Ship,
    Random,
    Escape,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum AreaAssignment {
    Standard,
    Random,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum WallJump {
    Vanilla,
    Collectible,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ETankRefill {
    Disabled,
    Vanilla,
    Full,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum MapsRevealed {
    No,
    Partial,
    Full,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum MapStationReveal {
    Partial,
    Full,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum SaveAnimals {
    No,
    Optional,
    Yes,
    Random,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum MotherBrainFight {
    Vanilla,
    Short,
    Skip,
}

pub fn parse_randomizer_settings(settings_json: &str) -> Result<RandomizerSettings> {
    let mut des = serde_json::Deserializer::from_str(settings_json);
    let settings = serde_path_to_error::deserialize(&mut des)?;
    Ok(settings)
}
