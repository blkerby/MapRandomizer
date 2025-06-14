use anyhow::Result;
use maprando_game::{Item, NotableId, RoomId, TechId};
use serde::{Deserialize, Serialize};

use crate::customize::CustomizeSettings;

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct RandomizerSettings {
    pub version: usize,
    pub name: Option<String>,
    pub skill_assumption_settings: SkillAssumptionSettings,
    pub item_progression_settings: ItemProgressionSettings,
    pub quality_of_life_settings: QualityOfLifeSettings,
    pub objective_settings: ObjectiveSettings,
    pub map_layout: String,
    pub doors_mode: DoorsMode,
    pub start_location_settings: StartLocationSettings,
    pub save_animals: SaveAnimals,
    pub other_settings: OtherSettings,
    #[serde(default)]
    pub debug: bool,
}
impl RandomizerSettings {
    pub fn apply_overrides(&mut self, customize_settings: &CustomizeSettings) {
        // Should the item_dot_change, transition_letters, and door_locks_size settings be fully
        // removed from the RandomizerSettings struct? This will make it so RandomizerSettings has
        // only settings chosen on the generate page and CustomizeSettings only has settings chosen
        // on the customize page, which was the case before these options were moved to the
        // customize page. It will also clean up this method by getting rid of these three options
        // and leaving only the actual overridden options. However, it seems these three options are
        // heavily integrated into the make_rom step, which only takes RandomizerSettings. So to
        // fully move these three options to CustomizeSettings, either make_rom will need to be
        // modified to take both RandomizerSettings and CustomizeSettings, or both make_rom and
        // customize_rom will need to be refactored so the patches related to these three options
        // now occur in customize_rom.
        self.other_settings.item_dot_change = customize_settings.item_dot_change;
        self.other_settings.transition_letters = customize_settings.transition_letters;
        self.other_settings.door_locks_size = customize_settings.door_locks_size;

        if !self.other_settings.race_mode {
            if let Some(mark_map_stations) = customize_settings.overrides.mark_map_stations {
                self.quality_of_life_settings.mark_map_stations = mark_map_stations;
            }
        }
    }
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
    pub bomb_into_cf_leniency: i32,
    pub jump_into_cf_leniency: i32,
    pub spike_xmode_leniency: i32,
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
    pub fanfares: Fanfares,
    // Samus control
    pub respin: bool,
    pub infinite_space_jump: bool,
    pub momentum_conservation: bool,
    // Tweaks to unintuitive vanilla behavior:
    pub all_items_spawn: bool,
    pub acid_chozo: bool,
    pub remove_climb_lava: bool,
    // Energy and reserves
    pub etank_refill: ETankRefill,
    pub energy_station_reserves: bool,
    pub disableable_etanks: bool,
    pub reserve_backward_transfer: bool,
    // Other:
    pub buffed_drops: bool,
    pub early_save: bool,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub enum Objective {
    Kraid,
    Phantoon,
    Draygon,
    Ridley,
    SporeSpawn,
    Crocomire,
    Botwoon,
    GoldenTorizo,
    MetroidRoom1,
    MetroidRoom2,
    MetroidRoom3,
    MetroidRoom4,
    BombTorizo,
    BowlingStatue,
    AcidChozoStatue,
    PitRoom,
    BabyKraidRoom,
    PlasmaRoom,
    MetalPiratesRoom,
}

impl Objective {
    pub fn get_flag_name(&self) -> &'static str {
        use Objective::*;
        match self {
            Kraid => "f_DefeatedKraid",
            Phantoon => "f_DefeatedPhantoon",
            Draygon => "f_DefeatedDraygon",
            Ridley => "f_DefeatedRidley",
            SporeSpawn => "f_DefeatedSporeSpawn",
            Crocomire => "f_DefeatedCrocomire",
            Botwoon => "f_DefeatedBotwoon",
            GoldenTorizo => "f_DefeatedGoldenTorizo",
            MetroidRoom1 => "f_KilledMetroidRoom1",
            MetroidRoom2 => "f_KilledMetroidRoom2",
            MetroidRoom3 => "f_KilledMetroidRoom3",
            MetroidRoom4 => "f_KilledMetroidRoom4",
            BombTorizo => "f_DefeatedBombTorizo",
            BowlingStatue => "f_UsedBowlingStatue",
            AcidChozoStatue => "f_UsedAcidChozoStatue",
            PitRoom => "f_ClearedPitRoom",
            BabyKraidRoom => "f_ClearedBabyKraidRoom",
            PlasmaRoom => "f_ClearedPlasmaRoom",
            MetalPiratesRoom => "f_ClearedMetalPiratesRoom",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum ObjectiveSetting {
    No,
    Maybe,
    Yes,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct ObjectiveOption {
    pub objective: Objective,
    pub setting: ObjectiveSetting,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum ObjectiveScreen {
    Disabled,
    Enabled,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct ObjectiveSettings {
    pub preset: Option<String>,
    pub objective_options: Vec<ObjectiveOption>,
    pub min_objectives: i32,
    pub max_objectives: i32,
    pub objective_screen: ObjectiveScreen,
}

pub struct ObjectiveGroup {
    pub name: String,
    pub objectives: Vec<(String, String)>, // (internal name, display name)
}

pub fn get_objective_groups() -> Vec<ObjectiveGroup> {
    vec![
        ObjectiveGroup {
            name: "Bosses".to_string(),
            objectives: vec![
                ("Kraid", "Kraid"),
                ("Phantoon", "Phantoon"),
                ("Draygon", "Draygon"),
                ("Ridley", "Ridley"),
            ]
            .into_iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect(),
        },
        ObjectiveGroup {
            name: "Minibosses".to_string(),
            objectives: vec![
                ("SporeSpawn", "Spore Spawn"),
                ("Crocomire", "Crocomire"),
                ("Botwoon", "Botwoon"),
                ("GoldenTorizo", "Golden Torizo"),
            ]
            .into_iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect(),
        },
        ObjectiveGroup {
            name: "Pirates".to_string(),
            objectives: vec![
                ("PitRoom", "Pit Room"),
                ("BabyKraidRoom", "Baby Kraid"),
                ("PlasmaRoom", "Plasma Room"),
                ("MetalPiratesRoom", "Metal Pirates"),
            ]
            .into_iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect(),
        },
        ObjectiveGroup {
            name: "Chozos".to_string(),
            objectives: vec![
                ("BombTorizo", "Bomb Torizo"),
                ("BowlingStatue", "Bowling"),
                ("AcidChozoStatue", "Acid Statue"),
            ]
            .into_iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect(),
        },
        ObjectiveGroup {
            name: "Metroids".to_string(),
            objectives: vec![
                ("MetroidRoom1", "Metroids 1"),
                ("MetroidRoom2", "Metroids 2"),
                ("MetroidRoom3", "Metroids 3"),
                ("MetroidRoom4", "Metroids 4"),
            ]
            .into_iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect(),
        },
    ]
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct StartLocationSettings {
    pub mode: StartLocationMode,
    pub room_id: Option<usize>,
    pub node_id: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct OtherSettings {
    pub wall_jump: WallJump,
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
pub enum Fanfares {
    Vanilla,
    Trimmed,
    Off,
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
    Custom,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum AreaAssignment {
    Ordered,
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
