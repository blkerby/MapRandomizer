use std::fmt::Display;

use anyhow::{Context, Result, bail};
use hashbrown::HashMap;
use maprando_game::{Item, NotableId, RoomId, TechId};
use serde::{Deserialize, Serialize};

use crate::preset::PresetData;

const VERSION: usize = include!("../../VERSION");

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

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct SkillAssumptionSettings {
    pub preset: Option<String>,
    pub shinespark_tiles: f32,
    pub heated_shinespark_tiles: f32,
    pub speed_ball_tiles: f32,
    pub shinecharge_leniency_frames: i32,
    pub resource_multiplier: f32,
    pub farm_time_limit: f32,
    pub gate_glitch_leniency: i32,
    pub door_stuck_leniency: i32,
    pub bomb_into_cf_leniency: i32,
    pub jump_into_cf_leniency: i32,
    pub flash_suit_distance: u8,
    pub blue_suit_distance: u8,
    pub spike_suit_leniency: i32,
    pub spike_xmode_leniency: i32,
    pub spike_speed_keep_leniency: i32,
    pub elevator_cf_leniency: i32,
    pub crystal_spark_leniency: i32,
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
pub enum DisableETankSetting {
    Off,
    Standard,
    Unrestricted,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct QualityOfLifeSettings {
    pub preset: Option<String>,
    // Map:
    pub initial_map_reveal_settings: InitialMapRevealSettings,
    pub item_markers: ItemMarkers,
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
    pub disableable_etanks: DisableETankSetting,
    pub reserve_backward_transfer: bool,
    // Other:
    pub buffed_drops: bool,
    pub early_save: bool,
    pub persist_flash_suit: bool,
    pub persist_blue_suit: bool,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub enum MapRevealLevel {
    No,
    Partial,
    Full,
}

impl Display for MapRevealLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct InitialMapRevealSettings {
    pub preset: Option<String>,
    pub map_stations: MapRevealLevel,
    pub save_stations: MapRevealLevel,
    pub refill_stations: MapRevealLevel,
    pub ship: MapRevealLevel,
    pub objectives: MapRevealLevel,
    pub area_transitions: MapRevealLevel,
    pub items1: MapRevealLevel,
    pub items2: MapRevealLevel,
    pub items3: MapRevealLevel,
    pub items4: MapRevealLevel,
    pub other: MapRevealLevel,
    pub all_areas: bool,
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
    pub door_locks_size: DoorLocksSize,
    pub map_station_reveal: MapStationReveal,
    pub energy_free_shinesparks: bool,
    pub ultra_low_qol: bool,
    pub disable_spikesuit: bool,
    pub disable_bluesuit: bool,
    pub enable_major_glitches: bool,
    pub speed_booster: SpeedBooster,
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
    Local,
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
pub enum AreaAssignmentPreset {
    Standard,
    Size,
    Depth,
    Random,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum AreaAssignmentBaseOrder {
    Size,
    Depth,
    Random,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct AreaAssignment {
    pub preset: Option<AreaAssignmentPreset>,
    pub base_order: AreaAssignmentBaseOrder,
    pub ship_in_crateria: bool,
    pub mother_brain_in_tourian: bool,
}

impl AreaAssignment {
    pub fn from_preset(preset: AreaAssignmentPreset) -> Self {
        match preset {
            AreaAssignmentPreset::Standard => AreaAssignment {
                preset: Some(preset),
                base_order: AreaAssignmentBaseOrder::Size,
                ship_in_crateria: true,
                mother_brain_in_tourian: true,
            },
            AreaAssignmentPreset::Size => AreaAssignment {
                preset: Some(preset),
                base_order: AreaAssignmentBaseOrder::Size,
                ship_in_crateria: false,
                mother_brain_in_tourian: false,
            },
            AreaAssignmentPreset::Depth => AreaAssignment {
                preset: Some(preset),
                base_order: AreaAssignmentBaseOrder::Depth,
                ship_in_crateria: false,
                mother_brain_in_tourian: false,
            },
            AreaAssignmentPreset::Random => AreaAssignment {
                preset: Some(preset),
                base_order: AreaAssignmentBaseOrder::Random,
                ship_in_crateria: false,
                mother_brain_in_tourian: false,
            },
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum WallJump {
    Vanilla,
    Collectible,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum SpeedBooster {
    Vanilla,
    Split,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ETankRefill {
    Disabled,
    Vanilla,
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

fn assign_presets(settings: &mut serde_json::Value, preset_data: &PresetData) -> Result<()> {
    if let Some(preset) = settings["skill_assumption_settings"]["preset"].as_str() {
        let preset = preset.to_owned();
        for p in &preset_data.skill_presets {
            if p.preset.as_ref() == Some(&preset) {
                *settings.get_mut("skill_assumption_settings").unwrap() = serde_json::to_value(p)?;
            }
        }
    }
    if let Some(preset) = settings["item_progression_settings"]["preset"].as_str() {
        let preset = preset.to_owned();
        for p in &preset_data.item_progression_presets {
            if p.preset.as_ref() == Some(&preset) {
                *settings.get_mut("item_progression_settings").unwrap() = serde_json::to_value(p)?;
            }
        }
    }
    if let Some(preset) = settings["quality_of_life_settings"]["preset"].as_str() {
        let preset = preset.to_owned();
        for p in &preset_data.quality_of_life_presets {
            if p.preset.as_ref() == Some(&preset) {
                *settings.get_mut("quality_of_life_settings").unwrap() = serde_json::to_value(p)?;
            }
        }
    }
    if let Some(preset) = settings["objective_settings"]["preset"].as_str() {
        let preset = preset.to_owned();
        for p in &preset_data.objective_presets {
            if p.preset.as_ref() == Some(&preset) {
                *settings.get_mut("objective_settings").unwrap() = serde_json::to_value(p)?;
            }
        }
    }
    if let Some(preset) = settings["name"].as_str() {
        let preset = preset.to_owned();
        for p in &preset_data.full_presets {
            if p.name.as_ref() == Some(&preset) {
                *settings = serde_json::to_value(p)?;
            }
        }
    }
    Ok(())
}

fn upgrade_tech_settings(settings: &mut serde_json::Value, preset_data: &PresetData) -> Result<()> {
    // This updates the names of tech, discards any obsolete tech settings, and disables
    // any new tech that are not referenced in the settings.
    let mut tech_map: HashMap<TechId, bool> = HashMap::new();
    for tech_setting in settings["skill_assumption_settings"]["tech_settings"]
        .as_array()
        .context("missing tech_settings")?
    {
        let tech_id = tech_setting["id"]
            .as_i64()
            .context("tech_setting missing id field")? as TechId;
        let enabled = tech_setting["enabled"]
            .as_bool()
            .context("tech_setting missing enabled field")?;
        tech_map.insert(tech_id, enabled);
    }

    let mut new_tech_settings: Vec<TechSetting> = vec![];
    for t in &preset_data
        .default_preset
        .skill_assumption_settings
        .tech_settings
    {
        new_tech_settings.push(TechSetting {
            id: t.id,
            name: t.name.clone(),
            enabled: tech_map.get(&t.id).copied().unwrap_or(false),
        });
    }
    *settings
        .get_mut("skill_assumption_settings")
        .unwrap()
        .get_mut("tech_settings")
        .unwrap() = serde_json::to_value(new_tech_settings)?;

    Ok(())
}

fn upgrade_notable_settings(
    settings: &mut serde_json::Value,
    preset_data: &PresetData,
) -> Result<()> {
    // This updates the names of notables, discards any obsolete notables settings, and disables
    // any new notables that are not referenced in the settings.
    let mut notable_map: HashMap<(RoomId, NotableId), bool> = HashMap::new();
    for notable_setting in settings["skill_assumption_settings"]["notable_settings"]
        .as_array()
        .context("missing notable_settings")?
    {
        let room_id = notable_setting["room_id"]
            .as_i64()
            .context("notable_setting missing room_id field")? as RoomId;
        let notable_id = notable_setting["notable_id"]
            .as_i64()
            .context("notable_setting missing notable_id field")?
            as RoomId;
        let enabled = notable_setting["enabled"]
            .as_bool()
            .context("notable_setting missing enabled field")?;
        notable_map.insert((room_id, notable_id), enabled);
    }

    let mut new_notable_settings: Vec<NotableSetting> = vec![];
    for s in &preset_data
        .default_preset
        .skill_assumption_settings
        .notable_settings
    {
        new_notable_settings.push(NotableSetting {
            room_id: s.room_id,
            notable_id: s.notable_id,
            room_name: s.room_name.clone(),
            notable_name: s.notable_name.clone(),
            enabled: notable_map
                .get(&(s.room_id, s.notable_id))
                .copied()
                .unwrap_or(false),
        });
    }
    *settings
        .get_mut("skill_assumption_settings")
        .unwrap()
        .get_mut("notable_settings")
        .unwrap() = serde_json::to_value(new_notable_settings)?;

    Ok(())
}

fn upgrade_other_skill_settings(settings: &mut serde_json::Value) -> Result<()> {
    let skill_assumption_settings = settings
        .get_mut("skill_assumption_settings")
        .context("missing skill_assumption_settings")?
        .as_object_mut()
        .context("skill_assumption_settings is not object")?;
    if !skill_assumption_settings.contains_key("bomb_into_cf_leniency") {
        skill_assumption_settings.insert("bomb_into_cf_leniency".to_string(), 5.into());
    }
    if !skill_assumption_settings.contains_key("jump_into_cf_leniency") {
        skill_assumption_settings.insert("jump_into_cf_leniency".to_string(), 9.into());
    }
    if !skill_assumption_settings.contains_key("spike_xmode_leniency") {
        skill_assumption_settings.insert("spike_xmode_leniency".to_string(), 2.into());
    }
    if !skill_assumption_settings.contains_key("farm_time_limit") {
        skill_assumption_settings.insert("farm_time_limit".to_string(), (60.0).into());
    }
    if !skill_assumption_settings.contains_key("flash_suit_distance") {
        skill_assumption_settings.insert("flash_suit_distance".to_string(), (255).into());
    }
    if !skill_assumption_settings.contains_key("blue_suit_distance") {
        skill_assumption_settings.insert("blue_suit_distance".to_string(), (255).into());
    }
    if !skill_assumption_settings.contains_key("spike_suit_leniency") {
        skill_assumption_settings.insert("spike_suit_leniency".to_string(), 2.into());
    }
    if !skill_assumption_settings.contains_key("spike_speed_keep_leniency") {
        skill_assumption_settings.insert("spike_speed_keep_leniency".to_string(), 4.into());
    }
    if !skill_assumption_settings.contains_key("elevator_cf_leniency") {
        skill_assumption_settings.insert("elevator_cf_leniency".to_string(), 8.into());
    }
    if !skill_assumption_settings.contains_key("crystal_spark_leniency") {
        skill_assumption_settings.insert("crystal_spark_leniency".to_string(), 8.into());
    }

    Ok(())
}

fn upgrade_item_progression_settings(settings: &mut serde_json::Value) -> Result<()> {
    let item_progression_settings = settings
        .get_mut("item_progression_settings")
        .context("missing item_progression_settings")?
        .as_object_mut()
        .context("item_progression_settings is not object")?;
    if !item_progression_settings.contains_key("ammo_collect_fraction") {
        item_progression_settings.insert("ammo_collect_fraction".to_string(), (0.7).into());
    }

    Ok(())
}

fn upgrade_initial_map_reveal_settings(settings: &mut serde_json::Value) -> Result<()> {
    if settings["quality_of_life_settings"]
        .as_object()
        .unwrap()
        .contains_key("initial_map_reveal_settings")
    {
        return Ok(());
    }

    let maps_revealed = settings["other_settings"]
        .as_object_mut()
        .context("missing 'other_settings'")?["maps_revealed"]
        .as_str()
        .context("expected 'maps_revealed' to be a string")?
        .to_owned();

    let qol_settings = settings["quality_of_life_settings"]
        .as_object_mut()
        .unwrap();
    if let Some(mark_map_stations) = qol_settings["mark_map_stations"].as_bool() {
        let reveal_level = match maps_revealed.as_str() {
            "No" => MapRevealLevel::No,
            "Partial" => MapRevealLevel::Partial,
            "Full" => MapRevealLevel::Full,
            _ => bail!("Unexpected value of 'maps_revealed'"),
        };
        let initial_map_reveal_settings = InitialMapRevealSettings {
            preset: Some(match reveal_level {
                MapRevealLevel::No => {
                    if mark_map_stations {
                        "Maps".to_string()
                    } else {
                        "No".to_string()
                    }
                }
                MapRevealLevel::Partial => "Partial".to_string(),
                MapRevealLevel::Full => "Full".to_string(),
            }),
            map_stations: if reveal_level == MapRevealLevel::No {
                if mark_map_stations {
                    MapRevealLevel::Full
                } else {
                    MapRevealLevel::No
                }
            } else {
                reveal_level
            },
            save_stations: reveal_level,
            refill_stations: reveal_level,
            ship: reveal_level,
            objectives: reveal_level,
            area_transitions: reveal_level,
            items1: reveal_level,
            items2: reveal_level,
            items3: reveal_level,
            items4: reveal_level,
            other: reveal_level,
            all_areas: reveal_level != MapRevealLevel::No,
        };
        qol_settings.insert(
            "initial_map_reveal_settings".to_string(),
            serde_json::to_value(initial_map_reveal_settings)?,
        );
    };

    Ok(())
}

fn upgrade_qol_settings(settings: &mut serde_json::Value) -> Result<()> {
    let etank_refill = settings["other_settings"]["etank_refill"]
        .as_str()
        .unwrap_or("Vanilla")
        .to_string();
    let qol_settings = settings
        .get_mut("quality_of_life_settings")
        .context("missing quality_of_life_settings")?
        .as_object_mut()
        .context("quality_of_life_settings is not object")?;
    if !qol_settings.contains_key("fanfares") {
        qol_settings.insert("fanfares".to_string(), "Off".into());
    }
    if !qol_settings.contains_key("etank_refill") {
        qol_settings.insert("etank_refill".to_string(), etank_refill.into());
    }
    if !qol_settings.contains_key("energy_station_reserves") {
        qol_settings.insert("energy_station_reserves".to_string(), false.into());
    }
    if !qol_settings.contains_key("disableable_etanks") {
        qol_settings.insert("disableable_etanks".to_string(), "Off".into());
    } else {
        match qol_settings["disableable_etanks"].as_bool() {
            Some(false) => {
                qol_settings.insert("disableable_etanks".to_string(), "Off".into());
            }
            Some(true) => {
                qol_settings.insert("disableable_etanks".to_string(), "Standard".into());
            }
            None => {}
        };
    }
    if !qol_settings.contains_key("reserve_backward_transfer") {
        qol_settings.insert("reserve_backward_transfer".to_string(), false.into());
    }
    if !qol_settings.contains_key("persist_flash_suit") {
        qol_settings.insert("persist_flash_suit".to_string(), false.into());
    }
    if !qol_settings.contains_key("persist_blue_suit") {
        qol_settings.insert("persist_blue_suit".to_string(), false.into());
    }
    upgrade_initial_map_reveal_settings(settings)?;
    Ok(())
}

fn upgrade_map_setting(settings: &mut serde_json::Value) -> Result<()> {
    if settings["map_layout"].as_str() == Some("Tame") {
        *settings.get_mut("map_layout").unwrap() = "Standard".into();
    }
    Ok(())
}

fn upgrade_start_location_setings(settings: &mut serde_json::Value) -> Result<()> {
    if !settings
        .as_object()
        .unwrap()
        .contains_key("start_location_settings")
    {
        let start_location_mode: String = settings["start_location_mode"].as_str().unwrap().into();
        settings.as_object_mut().unwrap().insert(
            "start_location_settings".to_string(),
            serde_json::Value::Object(
                vec![("mode".to_string(), start_location_mode.into())]
                    .into_iter()
                    .collect(),
            ),
        );
    }
    Ok(())
}

fn upgrade_animals_setting(settings: &mut serde_json::Value) -> Result<()> {
    if settings["save_animals"].as_str() == Some("Maybe") {
        *settings.get_mut("save_animals").unwrap() = "Optional".into();
    }
    Ok(())
}

fn upgrade_objective_settings(
    settings: &mut serde_json::Value,
    preset_data: &PresetData,
) -> Result<()> {
    let settings_obj = settings
        .as_object_mut()
        .context("expected settings to be object")?;

    if !settings_obj.contains_key("objective_settings") {
        settings_obj.insert(
            "objective_settings".to_string(),
            serde_json::to_value(preset_data.objective_presets[1].clone()).unwrap(),
        );
    }
    if settings_obj.contains_key("objectives_mode") {
        *settings_obj
            .get_mut("objective_settings")
            .unwrap()
            .get_mut("preset")
            .unwrap() = settings_obj["objectives_mode"].as_str().unwrap().into();
    }
    if !settings_obj["objective_settings"]
        .as_object()
        .unwrap()
        .contains_key("objective_screen")
    {
        settings_obj
            .get_mut("objective_settings")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("objective_screen".to_string(), "Enabled".into());
    }
    Ok(())
}

fn upgrade_other_settings(settings: &mut serde_json::Value) -> Result<()> {
    let settings_obj = settings
        .as_object_mut()
        .context("expected settings to be object")?;

    let other_settings = settings_obj
        .get_mut("other_settings")
        .context("missing other_settings")?
        .as_object_mut()
        .context("expected other_settings to be object")?;

    let area_assignment = other_settings
        .get_mut("area_assignment")
        .context("missing area_assignment")?;

    if area_assignment.is_string() {
        let preset_str = area_assignment.as_str().unwrap();
        let preset = match preset_str {
            "Standard" => AreaAssignmentPreset::Standard,
            "Ordered" => AreaAssignmentPreset::Size,
            "Random" => AreaAssignmentPreset::Random,
            _ => bail!("Unrecognized area assignment preset: {}", preset_str),
        };
        *area_assignment = serde_json::to_value(AreaAssignment::from_preset(preset))?;
    }

    if other_settings.get("disable_spikesuit").is_none() {
        other_settings.insert("disable_spikesuit".to_string(), false.into());
    }

    if other_settings.get("disable_bluesuit").is_none() {
        other_settings.insert("disable_bluesuit".to_string(), false.into());
    }

    if other_settings.get("enable_major_glitches").is_none() {
        other_settings.insert("enable_major_glitches".to_string(), false.into());
    }

    if other_settings.get("speed_booster").is_none() {
        other_settings.insert("speed_booster".to_string(), "Vanilla".into());
    }

    Ok(())
}

pub fn try_upgrade_settings(
    settings_str: String,
    preset_data: &PresetData,
    apply_presets: bool,
) -> Result<(String, RandomizerSettings)> {
    let mut settings: serde_json::Value = serde_json::from_str(&settings_str)?;

    upgrade_objective_settings(&mut settings, preset_data)?;
    if apply_presets {
        assign_presets(&mut settings, preset_data)?;
    }
    upgrade_tech_settings(&mut settings, preset_data)?;
    upgrade_notable_settings(&mut settings, preset_data)?;
    upgrade_other_skill_settings(&mut settings)?;
    upgrade_item_progression_settings(&mut settings)?;
    upgrade_qol_settings(&mut settings)?;
    upgrade_map_setting(&mut settings)?;
    upgrade_other_settings(&mut settings)?;
    upgrade_start_location_setings(&mut settings)?;
    upgrade_animals_setting(&mut settings)?;

    // Update version field to current version:
    *settings
        .get_mut("version")
        .context("missing version field")? = VERSION.into();

    // Validate that the upgraded settings will parse as a RandomizerSettings struct:
    let settings_str = settings.to_string();
    let settings_out = parse_randomizer_settings(&settings_str)?;
    let settings_out_str = serde_json::to_string(&settings_out)?;
    Ok((settings_out_str, settings_out))
}
