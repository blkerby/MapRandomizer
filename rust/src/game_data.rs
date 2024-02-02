pub mod smart_xml;

use anyhow::{bail, ensure, Context, Result};
// use log::info;
use crate::customize::room_palettes::decode_palette;
use crate::patch::title::read_image;
use hashbrown::{HashMap, HashSet};
use json::{self, JsonValue};
use num_enum::TryFromPrimitive;
use serde::Serialize;
use serde_derive::Deserialize;
use std::borrow::ToOwned;
use std::fs::File;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use strum::VariantNames;
use strum_macros::{EnumString, EnumVariantNames};
use log::{info, error};

#[derive(Deserialize, Clone)]
pub struct Map {
    pub rooms: Vec<(usize, usize)>, // (x, y) of upper-left corner of room on map
    pub doors: Vec<(
        (Option<usize>, Option<usize>), // Source (exit_ptr, entrance_ptr)
        (Option<usize>, Option<usize>), // Destination (exit_ptr, entrance_ptr)
        bool,                           // bidirectional
    )>,
    pub area: Vec<usize>,    // Area number: 0, 1, 2, 3, 4, or 5
    pub subarea: Vec<usize>, // Subarea number: 0 or 1
}

pub type TechId = usize; // Index into GameData.tech_isv.keys: distinct tech names from sm-json-data
pub type StratId = usize; // Index into GameData.notable_strats_isv.keys: distinct notable strat names from sm-json-data
pub type ItemId = usize; // Index into GameData.item_isv.keys: 21 distinct item names
pub type ItemIdx = usize; // Index into the game's item bit array (in RAM at 7E:D870)
pub type FlagId = usize; // Index into GameData.flag_isv.keys: distinct game flag names from sm-json-data
pub type RoomId = usize; // Room ID from sm-json-data
pub type RoomPtr = usize; // Room pointer (PC address of room header)
pub type RoomStateIdx = usize; // Room state index
pub type NodeId = usize; // Node ID from sm-json-data (only unique within a room)
pub type NodePtr = usize; // nodeAddress from sm-json-data: for items this is the PC address of PLM, for doors it is PC address of door data
pub type VertexId = usize; // Index into GameData.vertex_isv.keys: (room_id, node_id, obstacle_bitmask) combinations
pub type ItemLocationId = usize; // Index into GameData.item_locations: 100 nodes each containing an item
pub type ObstacleMask = usize; // Bitmask where `i`th bit (from least significant) indicates `i`th obstacle cleared within a room
pub type WeaponMask = usize; // Bitmask where `i`th bit indicates availability of (or vulnerability to) `i`th weapon.
pub type Capacity = i32; // Data type used to represent quantities of energy, ammo, etc.
pub type ItemPtr = usize; // PC address of item in PLM list
pub type DoorPtr = usize; // PC address of door data for exiting given door
pub type DoorPtrPair = (Option<DoorPtr>, Option<DoorPtr>); // PC addresses of door data for exiting & entering given door (from vanilla door connection)
pub type TilesetIdx = usize; // Tileset index
pub type AreaIdx = usize; // Area index (0..5)
pub type LinkIdx = u32;

#[derive(Default, Clone)]
pub struct IndexedVec<T: Hash + Eq> {
    pub keys: Vec<T>,
    pub index_by_key: HashMap<T, usize>,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    EnumString,
    EnumVariantNames,
    TryFromPrimitive,
    Serialize,
    Deserialize,
)]
#[repr(usize)]
// Note: the ordering of these items is significant; it must correspond to the ordering of PLM types:
pub enum Item {
    ETank,        // 0
    Missile,      // 1
    Super,        // 2
    PowerBomb,    // 3
    Bombs,        // 4
    Charge,       // 5
    Ice,          // 6
    HiJump,       // 7
    SpeedBooster, // 8
    Wave,         // 9
    Spazer,       // 10
    SpringBall,   // 11
    Varia,        // 12
    Gravity,      // 13
    XRayScope,    // 14
    Plasma,       // 15
    Grapple,      // 16
    SpaceJump,    // 17
    ScrewAttack,  // 18
    Morph,        // 19
    ReserveTank,  // 20
    WallJump,     // 21
    Nothing,      // 22
}

impl Item {
    pub fn is_unique(self) -> bool {
        ![
            Item::Missile,
            Item::Super,
            Item::PowerBomb,
            Item::ETank,
            Item::ReserveTank,
            Item::Nothing,
        ]
        .contains(&self)
    }
}

#[derive(Clone, Debug)]
pub enum Requirement {
    Free,
    Never,
    Tech(TechId),
    Strat(StratId),
    Item(ItemId),
    Flag(FlagId),
    NotFlag(FlagId),
    Walljump,
    ShineCharge {
        used_tiles: f32,
        heated: bool,
    },
    ShineChargeLeniencyFrames(i32),
    Shinespark {
        shinespark_tech_id: usize,
        frames: i32,
        excess_frames: i32,
    },
    HeatFrames(i32),
    LavaFrames(i32),
    GravitylessLavaFrames(i32),
    AcidFrames(i32),
    MetroidFrames(i32),
    Damage(i32),
    // Energy(i32),
    Missiles(i32),
    MissilesCapacity(i32),
    SupersCapacity(i32),
    PowerBombsCapacity(i32),
    RegularEnergyCapacity(i32),
    ReserveEnergyCapacity(i32),
    Supers(i32),
    PowerBombs(i32),
    EnergyRefill,
    ReserveRefill,
    MissileRefill,
    SuperRefill,
    PowerBombRefill,
    AmmoStationRefill,
    GateGlitchLeniency {
        green: bool,
        heated: bool,
    },
    HeatedDoorStuckLeniency {
        heat_frames: i32,
    },
    EnergyDrain,
    ReserveTrigger {
        min_reserve_energy: i32,
        max_reserve_energy: i32,
    },
    EnemyKill {
        count: i32,
        vul: EnemyVulnerabilities,
    },
    PhantoonFight {},
    DraygonFight {
        can_be_very_patient_tech_id: usize,
    },
    RidleyFight {
        can_be_very_patient_tech_id: usize,
    },
    BotwoonFight {
        second_phase: bool,
    },
    AdjacentRunway {
        room_id: RoomId,
        node_id: NodeId,
        used_tiles: f32,
        use_frames: Option<i32>,
        physics: Option<String>,
        override_runway_requirements: bool,
    },
    AdjacentJumpway {
        room_id: RoomId,
        node_id: NodeId,
        jumpway_type: String,
        min_height: Option<f32>,
        max_height: Option<f32>,
        max_left_position: Option<f32>,
        min_right_position: Option<f32>,
    },
    CanComeInCharged {
        room_id: RoomId,
        node_id: NodeId,
        frames_remaining: i32,
        unusable_tiles: i32,
    },
    ComeInWithRMode {
        room_id: RoomId,
        node_ids: Vec<NodeId>,
    },
    ComeInWithGMode {
        room_id: RoomId,
        node_ids: Vec<NodeId>,
        mode: String,
        mobility: String,
        artificial_morph: bool,
    },
    DoorUnlocked {
        room_id: RoomId,
        node_id: NodeId,
    },
    And(Vec<Requirement>),
    Or(Vec<Requirement>),
}

impl Requirement {
    pub fn make_and(reqs: Vec<Requirement>) -> Requirement {
        let mut out_reqs: Vec<Requirement> = vec![];
        for req in reqs {
            if let Requirement::Never = req {
                return Requirement::Never;
            } else if let Requirement::Free = req {
                continue;
            }
            out_reqs.push(req);
        }
        if out_reqs.len() == 0 {
            Requirement::Free
        } else if out_reqs.len() == 1 {
            out_reqs.into_iter().next().unwrap()
        } else {
            Requirement::And(out_reqs)
        }
    }

    pub fn make_or(reqs: Vec<Requirement>) -> Requirement {
        let mut out_reqs: Vec<Requirement> = vec![];
        for req in reqs {
            if let Requirement::Never = req {
                continue;
            } else if let Requirement::Free = req {
                return Requirement::Free;
            }
            out_reqs.push(req);
        }
        if out_reqs.len() == 0 {
            Requirement::Never
        } else if out_reqs.len() == 1 {
            out_reqs.into_iter().next().unwrap()
        } else {
            Requirement::Or(out_reqs)
        }
    }

    pub fn make_shinecharge(tiles: f32, heated: bool) -> Requirement {
        if tiles < 11.0 {
            // An effective runway length of 11 is the minimum possible length of shortcharge supported in the logic.
            // Strats requiring shorter runways than this are discarded to save processing time during generation.
            // Technically it is humanly viable to go as low as about 10.5, but below 11 the precision needed is so much
            // that it would not be reasonable to require on any settings.
            Requirement::Never
        } else {
            Requirement::ShineCharge { used_tiles: tiles, heated }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Runway {
    // TODO: add more details like slopes
    pub name: String,
    pub length: i32,
    pub open_end: i32,
    pub requirement: Requirement,
    pub physics: String,
    pub heated: bool,
    pub usable_coming_in: bool,
}

#[derive(Clone, Debug)]
pub struct Jumpway {
    pub name: String,
    pub jumpway_type: String,
    pub height: f32,
    pub left_position: Option<f32>,
    pub right_position: Option<f32>,
    pub requirement: Requirement,
}

#[derive(Debug, Clone)]
pub struct CanLeaveCharged {
    // TODO: add more details like slopes
    pub frames_remaining: i32,
    pub used_tiles: i32,
    pub open_end: i32,
    pub requirement: Requirement,
}

#[derive(Debug)]
pub struct LeaveWithGModeSetup {
    pub knockback: bool,
    pub requirement: Requirement,
}

#[derive(Debug)]
pub struct LeaveWithGMode {
    pub artificial_morph: bool,
    pub requirement: Requirement,
}

#[derive(Debug)]
pub struct GModeImmobile {
    pub requirement: Requirement,
}

#[derive(Clone, Debug)]
pub struct Link {
    pub from_vertex_id: VertexId,
    pub to_vertex_id: VertexId,
    pub requirement: Requirement,
    pub entrance_condition: Option<EntranceCondition>,
    pub exit_condition: Option<ExitCondition>,
    pub bypasses_door_shell: bool,
    pub notable_strat_name: Option<String>,
    pub strat_name: String,
    pub strat_notes: Vec<String>,
    pub sublinks: Vec<Link>,
}

#[derive(Deserialize, Default, Clone, Debug)]
pub struct RoomGeometryDoor {
    pub direction: String,
    pub x: usize,
    pub y: usize,
    pub exit_ptr: Option<usize>,
    pub entrance_ptr: Option<usize>,
    pub subtype: String,
    pub offset: Option<usize>,
}

#[derive(Deserialize, Default, Clone, Debug)]
pub struct RoomGeometryItem {
    pub x: usize,
    pub y: usize,
    pub addr: usize,
}

pub type RoomGeometryRoomIdx = usize;
pub type RoomGeometryDoorIdx = usize;
pub type RoomGeometryPartIdx = usize;

#[derive(Deserialize, Default, Clone, Debug)]
pub struct RoomGeometry {
    pub name: String,
    pub area: usize,
    pub rom_address: usize,
    pub twin_rom_address: Option<usize>,
    pub map: Vec<Vec<u8>>,
    pub doors: Vec<RoomGeometryDoor>,
    pub parts: Vec<Vec<RoomGeometryDoorIdx>>,
    pub durable_part_connections: Vec<(RoomGeometryPartIdx, RoomGeometryPartIdx)>,
    pub transient_part_connections: Vec<(RoomGeometryPartIdx, RoomGeometryPartIdx)>,
    pub items: Vec<RoomGeometryItem>,
    pub node_tiles: Vec<(usize, Vec<(usize, usize)>)>,
    pub twin_node_tiles: Option<Vec<(usize, Vec<(usize, usize)>)>>,
    pub heated: bool,
}

#[derive(Deserialize)]
pub struct EscapeTimingDoor {
    pub name: String,
    pub direction: String,
    pub x: usize,
    pub y: usize,
    pub part_idx: usize,
    pub door_idx: usize,
    pub node_id: usize,
}

#[derive(Deserialize, Copy, Clone)]
#[serde(rename_all = "snake_case")]
pub enum EscapeConditionRequirement {
    // #[serde(rename = "enemies_cleared")]
    EnemiesCleared,
    // #[serde(rename = "can_use_powerbombs")]
    CanUsePowerBombs,
    // #[serde(rename = "can_moonfall")]
    CanMoonfall,
    // #[serde(rename = "can_reverse_gate")]
    CanReverseGate,
    // #[serde(rename = "can_acid_dive")]
    CanAcidDive,
    // #[serde(rename = "can_off_camera_shot")]
    CanOffCameraShot,
    // #[serde(rename = "can_kago")]
    CanKago,
    // #[serde(rename = "can_hero_shot")]
    CanHeroShot,
}

#[derive(Deserialize)]
pub struct EscapeTimingCondition {
    pub requires: Vec<EscapeConditionRequirement>,
    pub in_game_time: f32,
}

#[derive(Deserialize)]
pub struct EscapeTiming {
    pub to_door: EscapeTimingDoor,
    pub in_game_time: Option<f32>,
    #[serde(default)]
    pub conditions: Vec<EscapeTimingCondition>,
}

#[derive(Deserialize)]
pub struct EscapeTimingGroup {
    pub from_door: EscapeTimingDoor,
    pub to: Vec<EscapeTiming>,
}

#[derive(Deserialize)]
pub struct EscapeTimingRoom {
    pub room_name: String,
    pub timings: Vec<EscapeTimingGroup>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct StartLocation {
    pub name: String,
    pub room_id: usize,
    pub node_id: usize,
    pub door_load_node_id: Option<usize>,
    pub x: f32,
    pub y: f32,
    pub requires: Option<Vec<serde_json::Value>>,
    pub note: Option<Vec<String>>,
    pub camera_offset_x: Option<f32>,
    pub camera_offset_y: Option<f32>,
    #[serde(skip_deserializing)]
    pub requires_parsed: Option<Requirement>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct HubLocation {
    pub name: String,
    pub room_id: usize,
    pub node_id: usize,
    pub requires: Option<Vec<serde_json::Value>>,
    pub note: Option<Vec<String>>,
    #[serde(skip_deserializing)]
    pub requires_parsed: Option<Requirement>,
}

#[derive(Clone, Debug)]
pub struct EnemyVulnerabilities {
    pub hp: i32,
    pub non_ammo_vulnerabilities: WeaponMask,
    pub missile_damage: i32,
    pub super_damage: i32,
    pub power_bomb_damage: i32,
}

pub struct ThemedPaletteTileset {
    pub palette: [[u8; 3]; 128],
    pub gfx8x8: Vec<u8>,
    pub gfx16x16: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct RunwayGeometry {
    pub length: f32,
    pub open_end: f32,
    pub gentle_up_tiles: f32,
    pub gentle_down_tiles: f32,
    pub steep_up_tiles: f32,
    pub steep_down_tiles: f32,
    pub starting_down_tiles: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Physics {
    Air,
    Water,
    Lava,
    Acid,
}

fn parse_physics(physics: &str) -> Result<Physics> {
    Ok(match physics {
        "air" => Physics::Air,
        "water" => Physics::Water,
        "lava" => Physics::Lava,
        "acid" => Physics::Acid,
        _ => bail!(format!("Unrecognized physics '{}'", physics)),
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DoorPosition {
    Left,
    Right,
    Top,
    Bottom,
}

fn parse_door_position(door_position: &str) -> Result<DoorPosition> {
    Ok(match door_position {
        "left" => DoorPosition::Left,
        "right" => DoorPosition::Right,
        "top" => DoorPosition::Top,
        "bottom" => DoorPosition::Bottom,
        _ => bail!(format!("Unrecognized door position '{}'", door_position)),
    })
}

#[derive(Clone, Debug)]
pub struct GModeRegainMobility {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SparkPosition {
    Top,
    Bottom,
    Any,
}

#[derive(Clone, Debug)]
pub enum ExitCondition {
    LeaveWithRunway {
        effective_length: f32,
        heated: bool,
        physics: Option<Physics>,
    },
    LeaveShinecharged {
        frames_remaining: Option<i32>,
        physics: Option<Physics>,
    },
    LeaveWithSpark {
        position: SparkPosition,
    },
    LeaveWithGModeSetup {
        knockback: bool,
    },
    LeaveWithGMode {
        morphed: bool,
    },
    LeaveWithStoredFallSpeed {
        fall_speed_in_tiles: i32,
    },
    LeaveWithDoorFrameBelow {
        height: f32,
        heated: bool,
    },
    LeaveWithPlatformBelow {
        height: f32,
        left_position: f32,
        right_position: f32,
    },
}

fn parse_spark_position(s: Option<&str>) -> Result<SparkPosition> {
    Ok(match s {
        Some("top") => SparkPosition::Top,
        Some("bottom") => SparkPosition::Bottom,
        None => SparkPosition::Any,
        _ => bail!("Unrecognized spark position: {}", s.unwrap())
    })
}

fn parse_exit_condition(
    exit_json: &JsonValue,
    heated: bool,
    physics: Option<Physics>,
) -> Result<ExitCondition> {
    ensure!(exit_json.is_object());
    ensure!(exit_json.len() == 1);
    let (key, value) = exit_json.entries().next().unwrap();
    ensure!(value.is_object());
    match key {
        "leaveWithRunway" => {
            let runway_geometry = parse_runway_geometry(value)?;
            let runway_effective_length = compute_runway_effective_length(&runway_geometry);
            Ok(ExitCondition::LeaveWithRunway {
                effective_length: runway_effective_length,
                heated,
                physics,
            })
        }
        "leaveShinecharged" => Ok(ExitCondition::LeaveShinecharged {
            frames_remaining: value["framesRemaining"].as_i32(),
            physics,
        }),
        "leaveWithSpark" => Ok(ExitCondition::LeaveWithSpark {
            position: parse_spark_position(value["position"].as_str())?,
        }),
        "leaveWithGModeSetup" => Ok(ExitCondition::LeaveWithGModeSetup {
            knockback: value["knockback"].as_bool().unwrap_or(true),
        }),
        "leaveWithGMode" => Ok(ExitCondition::LeaveWithGMode {
            morphed: value["morphed"]
                .as_bool()
                .context("Expecting boolean 'morphed'")?,
        }),
        "leaveWithStoredFallSpeed" => Ok(ExitCondition::LeaveWithStoredFallSpeed {
            fall_speed_in_tiles: value["fallSpeedInTiles"]
                .as_i32()
                .context("Expecting integer 'fallSpeedInTiles")?,
        }),
        "leaveWithDoorFrameBelow" => Ok(ExitCondition::LeaveWithDoorFrameBelow { 
            height: value["height"].as_f32().context("Expecting number 'height'")?,
            heated,
        }),
        "leaveWithPlatformBelow" => Ok(ExitCondition::LeaveWithPlatformBelow { 
            height: value["height"].as_f32().context("Expecting number 'height'")?,
            left_position: value["leftPosition"].as_f32().context("Expecting number 'leftPosition'")?,
            right_position: value["rightPosition"].as_f32().context("Expecting number 'rightPosition'")?,
        }),
        _ => {
            bail!(format!("Unrecognized exit condition: {}", key));
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GModeMode {
    Direct,
    Indirect,
    Any,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GModeMobility {
    Mobile,
    Immobile,
    Any,
}

#[derive(Clone, Debug)]
pub enum EntranceCondition {
    ComeInNormally {},
    ComeInRunning {
        speed_booster: Option<bool>,
        min_tiles: f32,
        max_tiles: f32,
    },
    ComeInJumping {
        speed_booster: Option<bool>,
        min_tiles: f32,
        max_tiles: f32,
    },
    ComeInShinecharging {
        effective_length: f32,
        heated: bool,
    },
    ComeInShinecharged {
        frames_required: i32,
    },
    ComeInShinechargedJumping {
        frames_required: i32,
    },
    ComeInWithSpark {
        position: SparkPosition,
    },
    ComeInSpeedballing {
        effective_runway_length: f32,
    },
    ComeInWithTemporaryBlue {},
    ComeInStutterShinecharging {
        min_tiles: f32,
    },
    ComeInWithBombBoost {},
    ComeInWithDoorStuckSetup {
        heated: bool,
    },
    ComeInWithRMode {},
    ComeInWithGMode {
        mode: GModeMode,
        morphed: bool,
        mobility: GModeMobility,
    },
    ComeInWithStoredFallSpeed {
        fall_speed_in_tiles: i32,
    },
    ComeInWithWallJumpBelow {
        min_height: f32,
    },
    ComeInWithSpaceJumpBelow {},
    ComeInWithPlatformBelow {
        min_height: f32,
        max_height: f32,
        max_left_position: f32,
        min_right_position: f32,
    },
}

fn parse_runway_geometry(runway: &JsonValue) -> Result<RunwayGeometry> {
    Ok(RunwayGeometry {
        length: runway["length"].as_f32().context("Expecting 'length'")?,
        open_end: runway["openEnd"].as_f32().context("Expecting 'openEnd'")?,
        gentle_up_tiles: runway["gentleUpTiles"].as_f32().unwrap_or(0.0),
        gentle_down_tiles: runway["gentleDownTiles"].as_f32().unwrap_or(0.0),
        steep_up_tiles: runway["steepUpTiles"].as_f32().unwrap_or(0.0),
        steep_down_tiles: runway["steepDownTiles"].as_f32().unwrap_or(0.0),
        starting_down_tiles: runway["startingDownTiles"].as_f32().unwrap_or(0.0),
    })
}

fn parse_runway_geometry_shinecharge(runway: &JsonValue) -> Result<RunwayGeometry> {
    // for some reason "canShinecharge" requirements use "usedTiles" instead of "length":
    Ok(RunwayGeometry {
        length: runway["usedTiles"]
            .as_f32()
            .context("Expecting 'usedTiles'")?,
        open_end: runway["openEnd"].as_f32().context("Expecting 'openEnd'")?,
        gentle_up_tiles: runway["gentleUpTiles"].as_f32().unwrap_or(0.0),
        gentle_down_tiles: runway["gentleDownTiles"].as_f32().unwrap_or(0.0),
        steep_up_tiles: runway["steepUpTiles"].as_f32().unwrap_or(0.0),
        steep_down_tiles: runway["steepDownTiles"].as_f32().unwrap_or(0.0),
        starting_down_tiles: runway["startingDownTiles"].as_f32().unwrap_or(0.0),
    })
}

fn compute_runway_effective_length(geom: &RunwayGeometry) -> f32 {
    let effective_length =
        geom.length - geom.starting_down_tiles - 9.0 / 16.0 * (1.0 - geom.open_end)
            + 1.0 / 3.0 * geom.steep_up_tiles
            + 1.0 / 7.0 * geom.steep_down_tiles
            + 5.0 / 27.0 * geom.gentle_up_tiles
            + 5.0 / 59.0 * geom.gentle_down_tiles;
    effective_length
}

fn parse_entrance_condition(entrance_json: &JsonValue, heated: bool) -> Result<EntranceCondition> {
    ensure!(entrance_json.is_object());
    ensure!(entrance_json.len() == 1);
    let (key, value) = entrance_json.entries().next().unwrap();
    ensure!(value.is_object());
    match key {
        "comeInNormally" => Ok(EntranceCondition::ComeInNormally {}),
        "comeInRunning" => Ok(EntranceCondition::ComeInRunning {
            speed_booster: value["speedBooster"].as_bool(),
            min_tiles: value["minTiles"]
                .as_f32()
                .context("Expecting number 'minTiles'")?,
            max_tiles: value["maxTiles"].as_f32().unwrap_or(255.0),
        }),
        "comeInJumping" => Ok(EntranceCondition::ComeInJumping {
            speed_booster: value["speedBooster"].as_bool(),
            min_tiles: value["minTiles"]
                .as_f32()
                .context("Expecting number 'minTiles'")?,
            max_tiles: value["maxTiles"].as_f32().unwrap_or(255.0),
        }),
        "comeInShinecharging" => {
            let runway_geometry = parse_runway_geometry(value)?;
            // Subtract 0.25 tiles since the door transition skips over approximately that much distance beyond the door shell tile:
            let runway_effective_length = compute_runway_effective_length(&runway_geometry) - 0.25;
            Ok(EntranceCondition::ComeInShinecharging {
                effective_length: runway_effective_length,
                heated,
            })
        }
        "comeInShinecharged" => Ok(EntranceCondition::ComeInShinecharged {
            frames_required: value["framesRequired"]
                .as_i32()
                .context("Expecting integer 'framesRequired'")?,
        }),
        "comeInShinechargedJumping" => Ok(EntranceCondition::ComeInShinechargedJumping {
            frames_required: value["framesRequired"]
                .as_i32()
                .context("Expecting integer 'framesRequired'")?,
        }),
        "comeInWithSpark" => Ok(EntranceCondition::ComeInWithSpark {
            position: parse_spark_position(value["position"].as_str())?,
        }),
        "comeInStutterShinecharging" => Ok(EntranceCondition::ComeInStutterShinecharging {
            min_tiles: value["minTiles"]
                .as_f32()
                .context("Expecting number 'minTiles'")?,
        }),
        "comeInWithBombBoost" => Ok(EntranceCondition::ComeInWithBombBoost {}),
        "comeInWithDoorStuckSetup" => Ok(EntranceCondition::ComeInWithDoorStuckSetup { heated }),
        "comeInSpeedballing" => {
            let runway_geometry = parse_runway_geometry(&value["runway"])?;
            let effective_runway_length = compute_runway_effective_length(&runway_geometry);
            Ok(EntranceCondition::ComeInSpeedballing {
                effective_runway_length,
            })
        }
        "comeInWithTemporaryBlue" => Ok(EntranceCondition::ComeInWithTemporaryBlue {}),
        "comeInWithRMode" => Ok(EntranceCondition::ComeInWithRMode {}),
        "comeInWithGMode" => {
            let mode = match value["mode"].as_str().context("Expected string 'mode'")? {
                "direct" => GModeMode::Direct,
                "indirect" => GModeMode::Indirect,
                "any" => GModeMode::Any,
                m => bail!("Unrecognized 'mode': {}", m),
            };
            let morphed = value["morphed"]
                .as_bool()
                .context("Expected bool 'morphed'")?;
            let mobility = match value["mobility"].as_str().unwrap_or("any") {
                "mobile" => GModeMobility::Mobile,
                "immobile" => GModeMobility::Immobile,
                "any" => GModeMobility::Any,
                m => bail!("Unrecognized 'mobility': {}", m),
            };
            Ok(EntranceCondition::ComeInWithGMode {
                mode,
                morphed,
                mobility,
            })
        }
        "comeInWithStoredFallSpeed" => Ok(EntranceCondition::ComeInWithStoredFallSpeed {
            fall_speed_in_tiles: value["fallSpeedInTiles"]
                .as_i32()
                .context("Expecting integer 'fallSpeedInTiles")?,
        }),
        "comeInWithWallJumpBelow" => Ok(EntranceCondition::ComeInWithWallJumpBelow {
            min_height: value["minHeight"].as_f32().context("Expecting number 'minHeight'")?,
        }),
        "comeInWithSpaceJumpBelow" => Ok(EntranceCondition::ComeInWithSpaceJumpBelow {}),
        "comeInWithPlatformBelow" => Ok(EntranceCondition::ComeInWithPlatformBelow {
            min_height: value["minHeight"].as_f32().unwrap_or(0.0),
            max_height: value["maxHeight"].as_f32().unwrap_or(f32::INFINITY),
            max_left_position: value["maxLeftPosition"].as_f32().unwrap_or(f32::INFINITY),
            min_right_position: value["minRightPosition"].as_f32().unwrap_or(f32::NEG_INFINITY),
        }),
        _ => {
            bail!(format!("Unrecognized entrance condition: {}", key));
        }
    }
}

#[derive(Default)]
pub struct LinksDataGroup {
    pub links: Vec<Link>,
    pub links_by_src: Vec<Vec<(LinkIdx, Link)>>,
    pub links_by_dst: Vec<Vec<(LinkIdx, Link)>>,
}

impl LinksDataGroup {
    pub fn new(links: Vec<Link>, num_vertices: usize, start_idx: usize) -> Self {
        let mut links_by_src: Vec<Vec<(LinkIdx, Link)>> = vec![Vec::new(); num_vertices];
        let mut links_by_dst: Vec<Vec<(LinkIdx, Link)>> = vec![Vec::new(); num_vertices];

        for (idx, link) in links.iter().enumerate() {
            let mut reversed_link = link.clone();
            std::mem::swap(
                &mut reversed_link.from_vertex_id,
                &mut reversed_link.to_vertex_id,
            );
            links_by_dst[reversed_link.from_vertex_id]
                .push(((start_idx + idx) as LinkIdx, reversed_link));
            links_by_src[link.from_vertex_id].push(((start_idx + idx) as LinkIdx, link.clone()));
        }
        Self {
            links,
            links_by_src,
            links_by_dst,
        }
    }
}

fn get_ignored_notable_strats() -> HashSet<String> {
    [
        "Breaking the Maridia Tube Gravity Jump", // not usable because of canRiskPermanentLossOfAccess
        "Metroid Room 1 PB Dodge Kill (Left to Right)",
        "Metroid Room 1 PB Dodge Kill (Right to Left)",
        "Metroid Room 2 PB Dodge Kill (Bottom to Top)",
        "Metroid Room 3 PB Dodge Kill (Left to Right)",
        "Metroid Room 3 PB Dodge Kill (Right to Left)",
        "Metroid Room 4 Three PB Kill (Top to Bottom)",
        "Metroid Room 4 Six PB Dodge Kill (Bottom to Top)",
        "Metroid Room 4 Three PB Dodge Kill (Bottom to Top)",
        "Wrecked Ship Main Shaft Partial Covern Ice Clip", // not usable because of canRiskPermanentLossOfAccess
        "Mickey Mouse Crumble Jump IBJ",  // only useful with CF clip strat, or if we change item progression rules
        "Green Brinstar Main Shaft Moonfall Spark",  // does not seem to be viable with the vanilla door connection
    ]
    .iter()
    .map(|x| x.to_string())
    .collect()
}

type TitleScreenImage = ndarray::Array3<u8>;

#[derive(Default)]
pub struct TitleScreenData {
    pub top_left: Vec<TitleScreenImage>,
    pub top_right: Vec<TitleScreenImage>,
    pub bottom_left: Vec<TitleScreenImage>,
    pub bottom_right: Vec<TitleScreenImage>,
    pub map_station: TitleScreenImage,
}

// TODO: Clean this up, e.g. pull out a separate structure to hold
// temporary data used only during loading, replace any
// remaining JsonValue types in the main struct with something
// more structured; combine maps with the same keys; also maybe unify the room geometry data
// with sm-json-data and cut back on the amount of different
// keys/IDs/indexes for rooms, nodes, and doors.
#[derive(Default)]
pub struct GameData {
    sm_json_data_path: PathBuf,
    pub tech_isv: IndexedVec<String>,
    pub notable_strat_isv: IndexedVec<String>,
    pub flag_isv: IndexedVec<String>,
    pub item_isv: IndexedVec<String>,
    weapon_isv: IndexedVec<String>,
    enemy_attack_damage: HashMap<(String, String), Capacity>,
    enemy_vulnerabilities: HashMap<String, EnemyVulnerabilities>,
    enemy_json: HashMap<String, JsonValue>,
    weapon_json_map: HashMap<String, JsonValue>,
    non_ammo_weapon_mask: WeaponMask,
    tech_json_map: HashMap<String, JsonValue>,
    pub helper_json_map: HashMap<String, JsonValue>,
    tech: HashMap<String, Option<Requirement>>,
    pub helpers: HashMap<String, Option<Requirement>>,
    pub room_json_map: HashMap<RoomId, JsonValue>,
    pub room_obstacle_idx_map: HashMap<RoomId, HashMap<String, usize>>,
    pub ignored_notable_strats: HashSet<String>,
    pub node_json_map: HashMap<(RoomId, NodeId), JsonValue>,
    pub node_spawn_at_map: HashMap<(RoomId, NodeId), NodeId>,
    pub node_runways_map: HashMap<(RoomId, NodeId), Vec<Runway>>,
    pub node_jumpways_map: HashMap<(RoomId, NodeId), Vec<Jumpway>>,
    pub node_can_leave_charged_map: HashMap<(RoomId, NodeId), Vec<CanLeaveCharged>>,
    pub node_leave_with_gmode_map: HashMap<(RoomId, NodeId), Vec<LeaveWithGMode>>,
    pub node_leave_with_gmode_setup_map: HashMap<(RoomId, NodeId), Vec<LeaveWithGModeSetup>>,
    pub node_gmode_immobile_map: HashMap<(RoomId, NodeId), GModeImmobile>,
    pub reverse_node_ptr_map: HashMap<NodePtr, (RoomId, NodeId)>,
    pub node_ptr_map: HashMap<(RoomId, NodeId), NodePtr>,
    pub node_exits: HashMap<(RoomId, NodeId), Vec<Link>>,
    pub node_gmode_regain_mobility: HashMap<(RoomId, NodeId), Vec<(Link, GModeRegainMobility)>>,
    pub node_lock_req_json: HashMap<(RoomId, NodeId), JsonValue>,
    pub unlocked_node_map: HashMap<(RoomId, NodeId), NodeId>,
    pub room_num_obstacles: HashMap<RoomId, usize>,
    pub door_ptr_pair_map: HashMap<DoorPtrPair, (RoomId, NodeId)>,
    pub unlocked_door_ptr_pair_map: HashMap<DoorPtrPair, (RoomId, NodeId)>,
    pub reverse_door_ptr_pair_map: HashMap<(RoomId, NodeId), DoorPtrPair>,
    pub door_position: HashMap<(RoomId, NodeId), DoorPosition>,
    pub vertex_isv: IndexedVec<(RoomId, NodeId, ObstacleMask)>,
    pub item_locations: Vec<(RoomId, NodeId)>,
    pub item_vertex_ids: Vec<Vec<VertexId>>,
    pub flag_locations: Vec<(RoomId, NodeId, FlagId)>,
    pub flag_vertex_ids: Vec<Vec<VertexId>>,
    pub target_vertices: IndexedVec<VertexId>,
    pub save_locations: Vec<(RoomId, NodeId)>,
    pub links: Vec<Link>,
    pub base_links: Vec<Link>,
    pub base_links_data: LinksDataGroup,
    pub seed_links: Vec<Link>,
    pub room_geometry: Vec<RoomGeometry>,
    pub room_and_door_idxs_by_door_ptr_pair:
        HashMap<DoorPtrPair, (RoomGeometryRoomIdx, RoomGeometryDoorIdx)>,
    pub room_ptr_by_id: HashMap<RoomId, RoomPtr>,
    pub room_id_by_ptr: HashMap<RoomPtr, RoomId>,
    pub raw_room_id_by_ptr: HashMap<RoomPtr, RoomId>, // Does not replace twin room pointer with corresponding main room pointer
    pub room_idx_by_ptr: HashMap<RoomPtr, RoomGeometryRoomIdx>,
    pub room_idx_by_name: HashMap<String, RoomGeometryRoomIdx>,
    pub node_tile_coords: HashMap<(RoomId, NodeId), Vec<(usize, usize)>>,
    pub node_coords: HashMap<(RoomId, NodeId), (usize, usize)>,
    pub room_shape: HashMap<RoomId, (usize, usize)>,
    pub area_names: Vec<String>,
    pub area_map_ptrs: Vec<isize>,
    pub tech_description: HashMap<String, String>,
    pub tech_dependencies: HashMap<String, Vec<String>>,
    pub strat_dependencies: HashMap<String, Vec<String>>,
    pub strat_area: HashMap<String, String>,
    pub strat_room: HashMap<String, String>,
    pub strat_description: HashMap<String, String>,
    pub tileset_palette_themes: Vec<HashMap<TilesetIdx, ThemedPaletteTileset>>,
    pub escape_timings: Vec<EscapeTimingRoom>,
    pub start_locations: Vec<StartLocation>,
    pub hub_locations: Vec<HubLocation>,
    pub heat_run_tech_id: TechId, // Cached since it is used frequently in graph traversal, and to avoid needing to store it in every HeatFrames req.
    pub wall_jump_tech_id: TechId,
    pub title_screen_data: TitleScreenData,
}

impl<T: Hash + Eq> IndexedVec<T> {
    pub fn add<U: ToOwned<Owned = T> + ?Sized>(&mut self, name: &U) -> usize {
        if !self.index_by_key.contains_key(&name.to_owned()) {
            let idx = self.keys.len();
            self.index_by_key.insert(name.to_owned(), self.keys.len());
            self.keys.push(name.to_owned());
            idx
        } else {
            self.index_by_key[&name.to_owned()]
        }
    }
}

fn read_json(path: &Path) -> Result<JsonValue> {
    let file = File::open(path).with_context(|| format!("unable to open {}", path.display()))?;
    let json_str = std::io::read_to_string(file)
        .with_context(|| format!("unable to read {}", path.display()))?;
    let json_data =
        json::parse(&json_str).with_context(|| format!("unable to parse {}", path.display()))?;
    Ok(json_data)
}

// TODO: Take steep slopes into account here:
pub fn get_effective_runway_length(used_tiles: f32, open_end: f32) -> f32 {
    used_tiles + open_end * 0.5
}

#[derive(Default)]
struct RequirementContext<'a> {
    room_id: RoomId,
    _from_node_id: NodeId, // Usable for debugging
    room_heated: bool,
    from_obstacles_bitmask: ObstacleMask,
    obstacles_idx_map: Option<&'a HashMap<String, usize>>,
}

impl GameData {
    fn load_tech(&mut self) -> Result<()> {
        let mut full_tech_json = read_json(&self.sm_json_data_path.join("tech.json"))?;
        ensure!(full_tech_json["techCategories"].is_array());
        full_tech_json["techCategories"].members_mut().find(|x| x["name"] == "Shots").unwrap()["techs"].push(json::object!{
            "name": "canHyperGateShot",
            "requires": [],
            "note": [
                "Can shoot blue & green gates from either side using Hyper Beam during the escape.",
                "This is easy to do; this tech just represents knowing it can be done.",
                "This is based on a randomizer patch applied on all settings (as in the vanilla game it isn't possible to open green gates using Hyper Beam.)"
            ]
        })?;
        full_tech_json["techCategories"].members_mut().find(|x| x["name"] == "Movement").unwrap()["techs"].push(json::object!{
            "name": "canEscapeMorphLocation",
            "requires": [],
            "devNote": "A special internal tech that is auto-enabled when using vanilla map, to ensure there is at least one bireachable item."
        })?;
        Self::override_can_awaken_zebes_tech_note(&mut full_tech_json)?;
        for tech_category in full_tech_json["techCategories"].members_mut() {
            ensure!(tech_category["techs"].is_array());
            for tech_json in tech_category["techs"].members() {
                self.load_tech_rec(tech_json)?;
            }
        }
        self.heat_run_tech_id = *self.tech_isv.index_by_key.get("canHeatRun").unwrap();
        self.wall_jump_tech_id = *self.tech_isv.index_by_key.get("canWalljump").unwrap();
        Ok(())
    }

    fn override_can_awaken_zebes_tech_note(full_tech_json: &mut JsonValue) -> Result<()> {
        let tech_category = full_tech_json["techCategories"].members_mut().find(|x| x["name"] == "Meta").unwrap();
        let tech = tech_category["techs"].members_mut().find(|x| x["name"] == "canAwakenZebes").unwrap();
        let tech_notes = &mut tech["note"];
        let notes = vec![
            "Understanding game behavior related to how the planet is awakened.",
            "The planet is awakened by unlocking any gray door locked by killing enemies in the room (not including bosses or minibosses).",
            "Pit Room, Baby Kraid Room, Metal Pirates Room, and Plasma Room are the places where this can be done in the randomizer.",
            "Awakening the planet causes enemies to spawn in Parlor, Climb, Morph Ball Room, Construction Zone, and Blue Brinstar Energy Tank Room.",
            "It also causes the item in the left side of Morph Ball Room and in The Final Missile to spawn.",
            "The item and enemies in Pit Room do not spawn until entering with Morph and Missiles collected, regardless of whether the planet is awake.",
            "If the quality-of-life option 'All items spawn from start' is enabled, as it is by default, then the items will already be spawned anyway, but awakening the planet can still matter because of its effects on enemies.",
        ];
        tech_notes.clear();
        for note in notes {
            tech_notes.push(JsonValue::String(note.to_owned()))?;
        }
        Ok(())
    }

    fn load_tech_rec(&mut self, tech_json: &JsonValue) -> Result<()> {
        let name = tech_json["name"]
            .as_str()
            .context("Missing 'name' in tech")?;
        self.tech_isv.add(name);

        let desc = if tech_json["note"].is_string() {
            tech_json["note"].as_str().unwrap().to_string()
        } else if tech_json["note"].is_array() {
            let notes: Vec<String> = tech_json["note"]
                .members()
                .map(|x| x.as_str().unwrap().to_string())
                .collect();
            notes.join(" ")
        } else {
            String::new()
        };

        self.tech_description.insert(name.to_string(), desc);
        self.tech_json_map
            .insert(name.to_string(), tech_json.clone());
        if tech_json.has_key("extensionTechs") {
            ensure!(tech_json["extensionTechs"].is_array());
            for ext_tech in tech_json["extensionTechs"].members() {
                self.load_tech_rec(ext_tech)?;
            }
        }
        Ok(())
    }

    fn extract_tech_dependencies(&self, req: &Requirement) -> HashSet<String> {
        match req {
            Requirement::Tech(tech_id) => vec![self.tech_isv.keys[*tech_id].clone()]
                .into_iter()
                .collect(),
            Requirement::And(sub_reqs) => {
                let mut out: HashSet<String> = HashSet::new();
                for r in sub_reqs {
                    out.extend(self.extract_tech_dependencies(r));
                }
                out
            }
            _ => HashSet::new(),
        }
    }

    fn get_tech_requirement(&mut self, tech_name: &str) -> Result<Requirement> {
        if let Some(req_opt) = self.tech.get(tech_name) {
            if let Some(req) = req_opt {
                return Ok(req.clone());
            } else {
                bail!("Circular dependence in tech: {}", tech_name);
            }
        }
        // if self.tech.contains_key(tech_name) {
        //     return self.tech[tech_name].clone().unwrap();
        // }

        // Temporarily insert a None value to act as a sentinel for detecting circular dependencies:
        self.tech.insert(tech_name.to_string(), None);

        let tech_json = &self.tech_json_map[tech_name].clone();
        let req = if tech_json.has_key("requires") {
            let ctx = RequirementContext::default();
            let mut reqs =
                self.parse_requires_list(tech_json["requires"].members().as_slice(), &ctx)?;
            reqs.push(Requirement::Tech(self.tech_isv.index_by_key[tech_name]));
            Requirement::make_and(reqs)
        } else {
            Requirement::Tech(self.tech_isv.index_by_key[tech_name])
        };
        *self.tech.get_mut(tech_name).unwrap() = Some(req.clone());
        Ok(req)
    }

    fn load_items_and_flags(&mut self) -> Result<()> {
        let item_json = read_json(&self.sm_json_data_path.join("items.json"))?;

        for item_name in Item::VARIANTS {
            self.item_isv.add(&item_name.to_string());
        }
        self.item_isv.add("WallJump");
        ensure!(item_json["gameFlags"].is_array());
        for flag_name in item_json["gameFlags"].members() {
            self.flag_isv.add(flag_name.as_str().unwrap());
        }

        // Add randomizer-specific flags:
        self.flag_isv.add("f_AllItemsSpawn");
        self.flag_isv.add("f_AcidChozoWithoutSpaceJump");

        Ok(())
    }

    fn load_weapons(&mut self) -> Result<()> {
        let weapons_json = read_json(&self.sm_json_data_path.join("weapons/main.json"))?;
        ensure!(weapons_json["weapons"].is_array());
        for weapon_json in weapons_json["weapons"].members() {
            let name = weapon_json["name"].as_str().unwrap();
            if weapon_json["situational"].as_bool().unwrap() {
                continue;
            }
            self.weapon_json_map
                .insert(name.to_string(), weapon_json.clone());
            self.weapon_isv.add(name);
        }

        self.non_ammo_weapon_mask = 0;
        for (i, weapon) in self.weapon_isv.keys.iter().enumerate() {
            let weapon_json = &self.weapon_json_map[weapon];
            if !weapon_json.has_key("shotRequires") {
                self.non_ammo_weapon_mask |= 1 << i;
            }
        }
        Ok(())
    }

    fn load_enemies(&mut self) -> Result<()> {
        for file in ["main.json", "bosses/main.json"] {
            let enemies_json = read_json(&self.sm_json_data_path.join("enemies").join(file))?;
            ensure!(enemies_json["enemies"].is_array());
            for enemy_json in enemies_json["enemies"].members() {
                let enemy_name = enemy_json["name"].as_str().unwrap();
                ensure!(enemy_json["attacks"].is_array());
                for attack in enemy_json["attacks"].members() {
                    let attack_name = attack["name"].as_str().unwrap();
                    let damage = attack["baseDamage"].as_i32().unwrap() as Capacity;
                    self.enemy_attack_damage
                        .insert((enemy_name.to_string(), attack_name.to_string()), damage);
                }
                self.enemy_vulnerabilities.insert(
                    enemy_name.to_string(),
                    self.get_enemy_vulnerabilities(enemy_json)?,
                );
                self.enemy_json
                    .insert(enemy_name.to_string(), enemy_json.clone());
            }
        }
        Ok(())
    }

    fn get_enemy_damage_multiplier(&self, enemy_json: &JsonValue, weapon_name: &str) -> f32 {
        for multiplier in enemy_json["damageMultipliers"].members() {
            if multiplier["weapon"] == weapon_name {
                return multiplier["value"].as_f32().unwrap();
            }
        }
        1.0
    }

    fn get_enemy_damage_weapon(
        &self,
        enemy_json: &JsonValue,
        weapon_name: &str,
        vul_mask: WeaponMask,
    ) -> i32 {
        let multiplier = self.get_enemy_damage_multiplier(enemy_json, weapon_name);
        let weapon_idx = self.weapon_isv.index_by_key[weapon_name];
        if vul_mask & (1 << weapon_idx) == 0 {
            return 0;
        }
        match weapon_name {
            "Missile" => (100.0 * multiplier) as i32,
            "Super" => (300.0 * multiplier) as i32,
            "PowerBomb" => (400.0 * multiplier) as i32,
            _ => panic!("Unsupported weapon: {}", weapon_name),
        }
    }

    fn get_enemy_vulnerabilities(&self, enemy_json: &JsonValue) -> Result<EnemyVulnerabilities> {
        ensure!(enemy_json["invul"].is_array());
        let invul: HashSet<String> = enemy_json["invul"]
            .members()
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        let mut vul_mask = 0;
        'weapon: for (i, weapon_name) in self.weapon_isv.keys.iter().enumerate() {
            let weapon_json = &self.weapon_json_map[weapon_name];
            if invul.contains(weapon_name) {
                continue;
            }
            ensure!(weapon_json["categories"].is_array());
            for cat in weapon_json["categories"]
                .members()
                .map(|x| x.as_str().unwrap())
            {
                if invul.contains(cat) {
                    continue 'weapon;
                }
            }
            vul_mask |= 1 << i;
        }

        Ok(EnemyVulnerabilities {
            non_ammo_vulnerabilities: vul_mask & self.non_ammo_weapon_mask,
            hp: enemy_json["hp"].as_i32().unwrap(),
            missile_damage: self.get_enemy_damage_weapon(enemy_json, "Missile", vul_mask),
            super_damage: self.get_enemy_damage_weapon(enemy_json, "Super", vul_mask),
            power_bomb_damage: self.get_enemy_damage_weapon(enemy_json, "PowerBomb", vul_mask),
        })
    }

    fn load_helpers(&mut self) -> Result<()> {
        let helpers_json = read_json(&self.sm_json_data_path.join("helpers.json"))?;
        ensure!(helpers_json["helperCategories"].is_array());
        for category_json in helpers_json["helperCategories"].members() {
            ensure!(category_json["helpers"].is_array());
            for helper in category_json["helpers"].members() {
                self.helper_json_map
                    .insert(helper["name"].as_str().unwrap().to_owned(), helper.clone());
            }
        }
        Ok(())
    }

    fn get_helper(&mut self, name: &str) -> Result<Requirement> {
        if self.helpers.contains_key(name) {
            if self.helpers[name].is_none() {
                bail!("Circular dependence in helper {}", name);
            }
            return Ok(self.helpers[name].clone().unwrap());
        }
        self.helpers.insert(name.to_owned(), None);
        let json_value = self.helper_json_map[name].clone();
        ensure!(json_value["requires"].is_array());
        let ctx = RequirementContext::default();
        let req = Requirement::make_and(
            self.parse_requires_list(&json_value["requires"].members().as_slice(), &ctx)?,
        );
        *self.helpers.get_mut(name).unwrap() = Some(req.clone());
        Ok(req)
    }

    fn parse_requires_list(
        &mut self,
        req_jsons: &[JsonValue],
        ctx: &RequirementContext,
    ) -> Result<Vec<Requirement>> {
        let mut reqs: Vec<Requirement> = Vec::new();
        for req_json in req_jsons {
            reqs.push(self.parse_requirement(req_json, ctx)?);
        }
        Ok(reqs)
    }

    // TODO: Find some way to have this not need to be mutable (e.g. resolve the helper dependencies in a first pass)
    fn parse_requirement(
        &mut self,
        req_json: &JsonValue,
        ctx: &RequirementContext,
    ) -> Result<Requirement> {
        if req_json.is_string() {
            let value = req_json.as_str().unwrap();
            if value == "never" {
                return Ok(Requirement::Never);
            } else if value == "canWalljump" {
                return Ok(Requirement::Walljump);
            } else if value == "i_ammoRefill" {
                return Ok(Requirement::AmmoStationRefill);
            } else if value == "i_BlueGateGlitchLeniency" {
                return Ok(Requirement::GateGlitchLeniency {
                    green: false,
                    heated: false,
                });
            } else if value == "i_GreenGateGlitchLeniency" {
                return Ok(Requirement::GateGlitchLeniency {
                    green: true,
                    heated: false,
                });
            } else if value == "i_HeatedBlueGateGlitchLeniency" {
                return Ok(Requirement::GateGlitchLeniency {
                    green: false,
                    heated: true,
                });
            } else if value == "i_HeatedGreenGateGlitchLeniency" {
                return Ok(Requirement::GateGlitchLeniency {
                    green: true,
                    heated: true,
                });
            } else if let Some(&item_id) = self.item_isv.index_by_key.get(value) {
                return Ok(Requirement::Item(item_id as ItemId));
            } else if let Some(&flag_id) = self.flag_isv.index_by_key.get(value) {
                return Ok(Requirement::Flag(flag_id as FlagId));
            } else if self.tech_json_map.contains_key(value) {
                return self.get_tech_requirement(value);
            } else if self.helper_json_map.contains_key(value) {
                return self.get_helper(value);
            }
        } else if req_json.is_object() && req_json.len() == 1 {
            let (key, value) = req_json.entries().next().unwrap();
            if key == "or" {
                ensure!(value.is_array());
                return Ok(Requirement::make_or(
                    self.parse_requires_list(value.members().as_slice(), ctx)?,
                ));
            } else if key == "and" {
                ensure!(value.is_array());
                return Ok(Requirement::make_and(
                    self.parse_requires_list(value.members().as_slice(), ctx)?,
                ));
            } else if key == "not" {
                if let Some(&flag_id) = self.flag_isv.index_by_key.get(value.as_str().unwrap()) {
                    return Ok(Requirement::NotFlag(flag_id));
                } else {
                    panic!("Unrecognized flag in 'not': {}", value);
                }
            } else if key == "ammo" {
                let ammo_type = value["type"]
                    .as_str()
                    .expect(&format!("missing/invalid ammo type in {}", req_json));
                let count = value["count"]
                    .as_i32()
                    .expect(&format!("missing/invalid ammo count in {}", req_json));
                if ammo_type == "Missile" {
                    return Ok(Requirement::Missiles(count as Capacity));
                } else if ammo_type == "Super" {
                    return Ok(Requirement::Supers(count as Capacity));
                } else if ammo_type == "PowerBomb" {
                    return Ok(Requirement::PowerBombs(count as Capacity));
                } else {
                    bail!("Unexpected ammo type in {}", req_json);
                }
            } else if key == "resourceCapacity" {
                ensure!(value.members().len() == 1);
                let value0 = value.members().next().unwrap();
                let resource_type = value0["type"]
                    .as_str()
                    .expect(&format!("missing/invalid resource type in {}", req_json));
                let count = value0["count"]
                    .as_i32()
                    .expect(&format!("missing/invalid resource count in {}", req_json));
                if resource_type == "Missile" {
                    return Ok(Requirement::MissilesCapacity(count as Capacity));
                } else if resource_type == "Super" {
                    return Ok(Requirement::SupersCapacity(count as Capacity));
                } else if resource_type == "PowerBomb" {
                    return Ok(Requirement::PowerBombsCapacity(count as Capacity));
                } else if resource_type == "RegularEnergy" {
                    return Ok(Requirement::RegularEnergyCapacity(count as Capacity));
                } else if resource_type == "ReserveEnergy" {
                    return Ok(Requirement::ReserveEnergyCapacity(count as Capacity));
                } else {
                    bail!("Unexpected ammo type in {}", req_json);
                }
            } else if key == "refill" {
                let mut req_list_and: Vec<Requirement> = vec![];
                for resource_type_json in value.members() {
                    let resource_type = resource_type_json.as_str().unwrap();
                    if resource_type == "Missile" {
                        req_list_and.push(Requirement::MissileRefill);
                    } else if resource_type == "Super" {
                        req_list_and.push(Requirement::SuperRefill);
                    } else if resource_type == "PowerBomb" {
                        req_list_and.push(Requirement::PowerBombRefill);
                    } else if resource_type == "RegularEnergy" {
                        req_list_and.push(Requirement::EnergyRefill);
                    } else if resource_type == "ReserveEnergy" {
                        req_list_and.push(Requirement::ReserveRefill);
                    } else if resource_type == "Energy" {
                        req_list_and.push(Requirement::EnergyRefill);
                        req_list_and.push(Requirement::ReserveRefill);
                    } else {
                        bail!("Unrecognized refill resource type: {}", resource_type);
                    }
                }
                return Ok(Requirement::make_and(req_list_and));
            } else if key == "ammoDrain" {
                // We patch out the ammo drain from the Mother Brain fight.
                return Ok(Requirement::Free);
            } else if key == "shinespark" {
                let frames = value["frames"]
                    .as_i32()
                    .expect(&format!("missing/invalid frames in {}", req_json));
                let excess_frames = value["excessFrames"].as_i32().unwrap_or(0);
                return Ok(Requirement::Shinespark {
                    shinespark_tech_id: self.tech_isv.index_by_key["canShinespark"],
                    frames,
                    excess_frames,
                });
            } else if key == "canShineCharge" {
                let runway_geometry = parse_runway_geometry_shinecharge(value)?;
                let effective_length = compute_runway_effective_length(&runway_geometry);
                return Ok(Requirement::make_shinecharge(effective_length, ctx.room_heated));
            } else if key == "heatFrames" {
                let frames = value  
                    .as_i32()
                    .expect(&format!("invalid heatFrames in {}", req_json));
                return Ok(Requirement::HeatFrames(frames));
            } else if key == "gravitylessHeatFrames" {
                // In Map Rando, Gravity doesn't affect heat frames, so this is treated the
                // same as "heatFrames".
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid gravitylessHeatFrames in {}", req_json));
                return Ok(Requirement::HeatFrames(frames));
            } else if key == "lavaFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid lavaFrames in {}", req_json));
                return Ok(Requirement::LavaFrames(frames));
            } else if key == "gravitylessLavaFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid gravitylessLavaFrames in {}", req_json));
                return Ok(Requirement::GravitylessLavaFrames(frames));
            } else if key == "acidFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid acidFrames in {}", req_json));
                return Ok(Requirement::AcidFrames(frames));
                // return Ok(Requirement::Damage(3 * frames / 2));
            } else if key == "metroidFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid metroidFrames in {}", req_json));
                return Ok(Requirement::MetroidFrames(frames));
            } else if key == "draygonElectricityFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid draygonElectricityFrames in {}", req_json));
                return Ok(Requirement::Damage(frames));
            } else if key == "samusEaterFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid samusEaterFrames in {}", req_json));
                return Ok(Requirement::Damage(frames / 8));
            } else if key == "spikeHits" {
                let hits = value
                    .as_i32()
                    .expect(&format!("invalid spikeHits in {}", req_json));
                return Ok(Requirement::Damage(hits * 60));
            } else if key == "thornHits" {
                let hits = value
                    .as_i32()
                    .expect(&format!("invalid thornHits in {}", req_json));
                return Ok(Requirement::Damage(hits * 16));
            } else if key == "hibashiHits" {
                let hits = value
                    .as_i32()
                    .expect(&format!("invalid hibashiHits in {}", req_json));
                return Ok(Requirement::Damage(hits * 30));
            } else if key == "enemyDamage" {
                let enemy_name = value["enemy"].as_str().unwrap().to_string();
                let attack_name = value["type"].as_str().unwrap().to_string();
                let hits = value["hits"].as_i32().unwrap() as Capacity;
                let base_damage = self.enemy_attack_damage[&(enemy_name, attack_name)];
                return Ok(Requirement::Damage(hits * base_damage));
            } else if key == "enemyKill" {
                // We only consider enemy kill methods that are non-situational and do not require ammo.
                // TODO: Consider all methods.
                let mut enemy_set: HashSet<String> = HashSet::new();
                let mut enemy_list: Vec<(String, i32)> = Vec::new();
                ensure!(value["enemies"].is_array());
                for enemy_group in value["enemies"].members() {
                    ensure!(enemy_group.is_array());
                    let mut last_enemy_name: Option<String> = None;
                    let mut cnt = 0;
                    for enemy in enemy_group.members() {
                        let enemy_name = enemy.as_str().unwrap().to_string();
                        enemy_set.insert(enemy_name.clone());
                        if Some(&enemy_name) == last_enemy_name.as_ref() {
                            cnt += 1;
                        } else {
                            if cnt > 0 {
                                enemy_list.push((last_enemy_name.unwrap(), cnt));
                            }
                            last_enemy_name = Some(enemy_name);
                            cnt = 1;
                        }
                    }
                    if cnt > 0 {
                        enemy_list.push((last_enemy_name.unwrap(), cnt));
                    }
                }

                if enemy_set.contains("Phantoon") {
                    return Ok(Requirement::PhantoonFight {});
                } else if enemy_set.contains("Draygon") {
                    return Ok(Requirement::DraygonFight {
                        can_be_very_patient_tech_id: self.tech_isv.index_by_key["canBeVeryPatient"],
                    });
                } else if enemy_set.contains("Ridley") {
                    return Ok(Requirement::RidleyFight {
                        can_be_very_patient_tech_id: self.tech_isv.index_by_key["canBeVeryPatient"],
                    });
                } else if enemy_set.contains("Botwoon 1") {
                    return Ok(Requirement::BotwoonFight {
                        second_phase: false,
                    });
                } else if enemy_set.contains("Botwoon 2") {
                    return Ok(Requirement::BotwoonFight { second_phase: true });
                }

                let mut allowed_weapons: WeaponMask = if value.has_key("explicitWeapons") {
                    ensure!(value["explicitWeapons"].is_array());
                    let mut weapon_mask = 0;
                    for weapon_name in value["explicitWeapons"].members() {
                        if self
                            .weapon_isv
                            .index_by_key
                            .contains_key(weapon_name.as_str().unwrap())
                        {
                            weapon_mask |=
                                1 << self.weapon_isv.index_by_key[weapon_name.as_str().unwrap()];
                        }
                    }
                    weapon_mask
                } else {
                    (1 << self.weapon_isv.keys.len()) - 1
                };
                if value.has_key("excludedWeapons") {
                    ensure!(value["excludedWeapons"].is_array());
                    for weapon_name in value["excludedWeapons"].members() {
                        if self
                            .weapon_isv
                            .index_by_key
                            .contains_key(weapon_name.as_str().unwrap())
                        {
                            allowed_weapons &=
                                !(1 << self.weapon_isv.index_by_key[weapon_name.as_str().unwrap()]);
                        }
                    }
                }
                let mut reqs: Vec<Requirement> = Vec::new();
                for (enemy_name, count) in &enemy_list {
                    let mut vul = self.enemy_vulnerabilities[enemy_name].clone();
                    vul.non_ammo_vulnerabilities &= allowed_weapons;
                    if allowed_weapons & (1 << self.weapon_isv.index_by_key["Missile"]) == 0 {
                        vul.missile_damage = 0;
                    }
                    if allowed_weapons & (1 << self.weapon_isv.index_by_key["Super"]) == 0 {
                        vul.super_damage = 0;
                    }
                    if allowed_weapons & (1 << self.weapon_isv.index_by_key["PowerBomb"]) == 0 {
                        vul.power_bomb_damage = 0;
                    }
                    reqs.push(Requirement::EnemyKill {
                        count: *count,
                        vul: vul,
                    });
                }
                return Ok(Requirement::make_and(reqs));
            } else if key == "energyAtMost" {
                ensure!(value.as_i32().unwrap() == 1);
                return Ok(Requirement::EnergyDrain);
            } else if key == "previousNode" {
                // Currently this is used only in the Early Supers quick crumble and Mission Impossible strats and is
                // redundant in both cases, so we treat it as free.
                return Ok(Requirement::Free);
            } else if key == "previousStratProperty" {
                // This is only used in one place in Crumble Shaft, where it doesn't seem to be necessary.
                return Ok(Requirement::Free);
            } else if key == "obstaclesCleared" {
                ensure!(value.is_array());
                if let Some(obstacles_idx_map) = ctx.obstacles_idx_map {
                    for obstacle_name_json in value.members() {
                        let obstacle_name = obstacle_name_json.as_str().unwrap();
                        if let Some(obstacle_idx) = obstacles_idx_map.get(obstacle_name) {
                            if (1 << obstacle_idx) & ctx.from_obstacles_bitmask == 0 {
                                return Ok(Requirement::Never);
                            }
                        } else {
                            bail!("Obstacle name {} not found", obstacle_name);
                        }
                    }
                    return Ok(Requirement::Free);
                } else {
                    // No obstacle state in context. This happens with cross-room strats. We're not ready to
                    // deal with obstacles yet here, so we just keep these out of logic.
                    return Ok(Requirement::Never);
                }
            } else if key == "obstaclesNotCleared" {
                ensure!(value.is_array());
                if let Some(obstacles_idx_map) = ctx.obstacles_idx_map {
                    for obstacle_name_json in value.members() {
                        let obstacle_name = obstacle_name_json.as_str().unwrap();
                        if let Some(obstacle_idx) = obstacles_idx_map.get(obstacle_name) {
                            if (1 << obstacle_idx) & ctx.from_obstacles_bitmask != 0 {
                                return Ok(Requirement::Never);
                            }
                        } else {
                            bail!("Obstacle name {} not found", obstacle_name);
                        }
                    }
                    return Ok(Requirement::Free);
                } else {
                    // No obstacle state in context. This happens with cross-room strats, in which case
                    // all obstacles should be cleared:
                    return Ok(Requirement::Free);
                }
            } else if key == "adjacentRunway" {
                if ctx.from_obstacles_bitmask != 0 {
                    return Ok(Requirement::Never);
                }
                let physics: Option<String> = if value.has_key("physics") {
                    ensure!(value["physics"].len() == 1);
                    Some(value["physics"][0].as_str().unwrap().to_string())
                } else {
                    None
                };
                let use_frames: Option<i32> = if value.has_key("useFrames") {
                    Some(
                        value["useFrames"]
                            .as_i32()
                            .context("Expecting integer for useFrames")?,
                    )
                } else {
                    None
                };
                let mut unlocked_node_id = value["fromNode"].as_usize().unwrap();
                if self
                    .unlocked_node_map
                    .contains_key(&(ctx.room_id, unlocked_node_id))
                {
                    unlocked_node_id = self.unlocked_node_map[&(ctx.room_id, unlocked_node_id)];
                }

                return Ok(Requirement::AdjacentRunway {
                    room_id: ctx.room_id,
                    node_id: unlocked_node_id,
                    used_tiles: value["usedTiles"].as_f32().unwrap(),
                    use_frames,
                    physics: physics,
                    override_runway_requirements: value["overrideRunwayRequirements"]
                        .as_bool()
                        .unwrap_or(false),
                });
            } else if key == "adjacentJumpway" {
                // TODO: implement this
                if ctx.from_obstacles_bitmask != 0 {
                    return Ok(Requirement::Never);
                }
                let jumpway_type: String = value["jumpwayType"].as_str().unwrap().to_string();
                let min_height: Option<f32> = if value.has_key("minHeight") {
                    Some(
                        value["minHeight"]
                            .as_f32()
                            .context("Expecting number for minHeight")?,
                    )
                } else {
                    None
                };
                let max_height: Option<f32> = if value.has_key("maxHeight") {
                    Some(
                        value["maxHeight"]
                            .as_f32()
                            .context("Expecting number for maxHeight")?,
                    )
                } else {
                    None
                };
                let max_left_position: Option<f32> = if value.has_key("maxLeftPosition") {
                    Some(
                        value["maxLeftPosition"]
                            .as_f32()
                            .context("Expecting number for maxLeftPosition")?,
                    )
                } else {
                    None
                };
                let min_right_position: Option<f32> = if value.has_key("minRightPosition") {
                    Some(
                        value["minRightPosition"]
                            .as_f32()
                            .context("Expecting number for minRightPosition")?,
                    )
                } else {
                    None
                };

                let mut unlocked_node_id = value["fromNode"].as_usize().unwrap();
                if self
                    .unlocked_node_map
                    .contains_key(&(ctx.room_id, unlocked_node_id))
                {
                    unlocked_node_id = self.unlocked_node_map[&(ctx.room_id, unlocked_node_id)];
                }

                return Ok(Requirement::AdjacentJumpway {
                    room_id: ctx.room_id,
                    node_id: unlocked_node_id,
                    jumpway_type: jumpway_type,
                    min_height,
                    max_height,
                    max_left_position,
                    min_right_position,
                });
            } else if key == "canComeInCharged" {
                if ctx.from_obstacles_bitmask != 0 {
                    return Ok(Requirement::Never);
                }
                let frames_remaining = value["framesRemaining"]
                    .as_i32()
                    .with_context(|| format!("missing/invalid framesRemaining in {}", req_json))?;
                let unusable_tiles = value["unusableTiles"].as_i32().unwrap_or(0);
                // if value["fromNode"].as_usize().unwrap() != ctx.src_node_id {
                //     println!("In roomId={}, canComeInCharged fromNode={}, from nodeId={}", ctx.room_id,
                //         value["fromNode"].as_usize().unwrap(), ctx.src_node_id);
                // }
                let mut unlocked_node_id = value["fromNode"].as_usize().unwrap();
                if self
                    .unlocked_node_map
                    .contains_key(&(ctx.room_id, unlocked_node_id))
                {
                    unlocked_node_id = self.unlocked_node_map[&(ctx.room_id, unlocked_node_id)];
                }
                return Ok(Requirement::CanComeInCharged {
                    room_id: ctx.room_id,
                    node_id: unlocked_node_id,
                    // node_id: ctx.src_node_id,
                    frames_remaining,
                    unusable_tiles,
                });
                // return Ok(Requirement::Never);
            } else if key == "comeInWithRMode" {
                if ctx.from_obstacles_bitmask != 0 {
                    return Ok(Requirement::Never);
                }
                let mut node_ids: Vec<NodeId> = Vec::new();
                for from_node in value["fromNodes"].members() {
                    let mut unlocked_node_id = from_node.as_usize().unwrap();
                    if self
                        .unlocked_node_map
                        .contains_key(&(ctx.room_id, unlocked_node_id))
                    {
                        unlocked_node_id = self.unlocked_node_map[&(ctx.room_id, unlocked_node_id)];
                    }
                    node_ids.push(unlocked_node_id);
                }

                return Ok(Requirement::ComeInWithRMode {
                    room_id: ctx.room_id,
                    node_ids,
                });
            } else if key == "comeInWithGMode" {
                if ctx.from_obstacles_bitmask != 0 {
                    return Ok(Requirement::Never);
                }
                let mut node_ids: Vec<NodeId> = Vec::new();
                for from_node in value["fromNodes"].members() {
                    let mut unlocked_node_id = from_node.as_usize().unwrap();
                    if self
                        .unlocked_node_map
                        .contains_key(&(ctx.room_id, unlocked_node_id))
                    {
                        unlocked_node_id = self.unlocked_node_map[&(ctx.room_id, unlocked_node_id)];
                    }
                    node_ids.push(unlocked_node_id);
                }
                let mode = value["mode"]
                    .as_str()
                    .with_context(|| format!("missing/invalid artificialMorph in {}", req_json))?;
                let artificial_morph = value["artificialMorph"]
                    .as_bool()
                    .with_context(|| format!("missing/invalid artificialMorph in {}", req_json))?;
                let mobility = value["mobility"].as_str().unwrap_or("any");

                return Ok(Requirement::ComeInWithGMode {
                    room_id: ctx.room_id,
                    node_ids,
                    mode: mode.to_string(),
                    artificial_morph,
                    mobility: mobility.to_string(),
                });
            } else if key == "resetRoom" {
                if ctx.from_obstacles_bitmask != 0 {
                    return Ok(Requirement::Never);
                }
                let mut node_ids: Vec<NodeId> = Vec::new();
                for from_node in value["nodes"].members() {
                    let mut unlocked_node_id = from_node.as_usize().unwrap();
                    if self
                        .unlocked_node_map
                        .contains_key(&(ctx.room_id, unlocked_node_id))
                    {
                        unlocked_node_id = self.unlocked_node_map[&(ctx.room_id, unlocked_node_id)];
                    }
                    node_ids.push(unlocked_node_id);
                }
                let mut reqs_or: Vec<Requirement> = vec![];
                for node_id in node_ids {
                    reqs_or.push(Requirement::DoorUnlocked {
                        room_id: ctx.room_id,
                        node_id,
                    });
                }
                return Ok(Requirement::make_or(reqs_or));
            } else if key == "doorUnlockedAtNode" {
                let mut unlocked_node_id = value.as_usize().unwrap();
                if self
                    .unlocked_node_map
                    .contains_key(&(ctx.room_id, unlocked_node_id))
                {
                    unlocked_node_id = self.unlocked_node_map[&(ctx.room_id, unlocked_node_id)];
                }
                return Ok(Requirement::DoorUnlocked {
                    room_id: ctx.room_id,
                    node_id: unlocked_node_id,
                });
            } else if key == "itemNotCollectedAtNode" {
                // TODO: implement this
                return Ok(Requirement::Free);
            } else if key == "autoReserveTrigger" {
                return Ok(Requirement::ReserveTrigger { 
                    min_reserve_energy: value["minReserveEnergy"].as_i32().unwrap_or(1),
                    max_reserve_energy:  value["maxReserveEnergy"].as_i32().unwrap_or(400),
                })
            }
        }
        bail!("Unable to parse requirement: {}", req_json);
    }

    fn load_regions(&mut self) -> Result<()> {
        let region_pattern =
            self.sm_json_data_path.to_str().unwrap().to_string() + "/region/**/*.json";
        for entry in glob::glob(&region_pattern).unwrap() {
            if let Ok(path) = entry {
                let path_str = path.to_str().with_context(|| {
                    format!("Unable to convert path to string: {}", path.display())
                })?;
                if path_str.contains("ceres") || path_str.contains("roomDiagrams") {
                    continue;
                }

                let room_json = read_json(&path)?;
                let room_name = room_json["name"].clone();
                let preprocessed_room_json = self
                    .preprocess_room(&room_json)
                    .with_context(|| format!("Preprocessing room {}", room_name))?;
                self.process_room(&preprocessed_room_json)
                    .with_context(|| format!("Processing room {}", room_name))?;
            } else {
                bail!("Error processing region path: {}", entry.err().unwrap());
            }
        }

        let ignored_notable_strats = get_ignored_notable_strats();
        if !ignored_notable_strats.is_subset(&self.ignored_notable_strats) {
            let diff: Vec<String> = ignored_notable_strats
                .difference(&self.ignored_notable_strats)
                .cloned()
                .collect();
            panic!("Unrecognized ignored notable strats: {:?}", diff);
        }
    
        Ok(())
    }

    fn override_morph_ball_room(&mut self, room_json: &mut JsonValue) {
        // Override the Careful Jump strat to get out from Morph Ball location:
        //
        room_json["strats"]
            .members_mut()
            .find(|x| {
                x["link"][0].as_i32().unwrap() == 4
                    && x["link"][1].as_i32().unwrap() == 2
                    && x["name"].as_str().unwrap() == "Careful Jump"
            })
            .unwrap()
            .insert(
                "requires",
                json::array![{
                    "or": [
                        "canCarefulJump",
                        "canEscapeMorphLocation",
                    ]
                }],
            )
            .unwrap();
    }

    fn override_shaktool_room(&mut self, room_json: &mut JsonValue) {
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 3 {
                // Adding a dummy lock on Shaktool done digging event, so that the code in `preprocess_room`
                // can pick it up and construct a corresponding obstacle for the flag (as it expects there
                // to be a lock).
                node_json["locks"] = json::array![{
                    "name": "Shaktool Lock",
                    "lockType": "triggeredEvent",
                    "unlockStrats": [
                        {
                            "name": "Base",
                            "notable": false,
                            "requires": [],
                        }
                    ]
                }];
            }
        }
    }

    fn override_metal_pirates_room(&mut self, room_json: &mut JsonValue) {
        // Add lock on right door of Metal Pirates Room:
        let mut found = false;
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 2 {
                found = true;
                node_json["locks"] = json::array![
                  {
                    "name": "Metal Pirates Grey Lock (to Wasteland)",
                    "lockType": "killEnemies",
                    "unlockStrats": [
                      {
                        "name": "Base",
                        "notable": false,
                        "requires": [ {"obstaclesCleared": ["A"]} ]
                      }
                    ],
                    "yields": [ "f_ZebesAwake" ]
                  }
                ];
            }
        }
        assert!(found);
    }

    fn override_tourian_save_room(&mut self, room_json: &mut JsonValue) {
        // Remove the "save" utility, as we have a map here instead.
        let mut found = false;
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 2 {
                node_json.remove("utility");
                found = true;
            }
        }
        assert!(found);
    }

    fn preprocess_room(&mut self, room_json: &JsonValue) -> Result<JsonValue> {
        // We apply some changes to the sm-json-data specific to Map Rando.

        let ignored_notable_strats = get_ignored_notable_strats();

        let mut new_room_json = room_json.clone();
        ensure!(room_json["nodes"].is_array());
        let mut next_node_id = room_json["nodes"]
            .members()
            .map(|x| x["id"].as_usize().unwrap())
            .max()
            .unwrap()
            + 1;
        let mut extra_nodes: Vec<JsonValue> = Vec::new();
        let mut extra_strats: Vec<JsonValue> = Vec::new();
        let room_id = room_json["id"].as_usize().unwrap();

        if room_json["name"].as_str().unwrap() == "Upper Tourian Save Room" {
            new_room_json["name"] = JsonValue::String("Tourian Map Room".to_string());
        }

        // Rooms where we want the logic to take into account the gray door locks (elsewhere the gray doors are changed to blue):
        // Be sure to keep this consistent with patches where the gray doors are actually changed in the ROM, in
        // "patch.rs" and "gray_doors.asm".
        let door_lock_allowed_room_ids = [
            12,  // Pit Room
            82,  // Baby Kraid Room
            84,  // Kraid Room
            139, // Metal Pirates Room
            142, // Ridley's Room
            150, // Golden Torizo Room
            193, // Draygon's Room
            219, // Plasma Room
        ];

        // Flags for which we want to add an obstacle in the room, to allow progression through (or back out of) the room
        // after setting the flag on the same graph traversal step (which cannot take into account the new flag).
        let obstacle_flags = [
            "f_DefeatedKraid",
            "f_DefeatedDraygon",
            "f_DefeatedRidley",
            "f_DefeatedGoldenTorizo",
            "f_DefeatedCrocomire",
            "f_DefeatedSporeSpawn",
            "f_DefeatedBotwoon",
            "f_MaridiaTubeBroken",
            "f_ShaktoolDoneDigging",
            "f_UsedAcidChozoStatue",
        ];

        // TODO: handle overrides in a more structured/robust way
        if room_id == 222 {
            self.override_shaktool_room(&mut new_room_json);
        } else if room_id == 38 {
            self.override_morph_ball_room(&mut new_room_json);
        } else if room_id == 139 {
            self.override_metal_pirates_room(&mut new_room_json);
        } else if room_id == 225 {
            self.override_tourian_save_room(&mut new_room_json);
        }

        let mut obstacle_flag: Option<String> = None;

        for node_json in new_room_json["nodes"].members_mut() {
            let node_id = node_json["id"].as_usize().unwrap();

            // TODO: get rid of dummy nodes that we're creating here.
            if node_json.has_key("locks")
                && (!["door", "entrance"].contains(&node_json["nodeType"].as_str().unwrap())
                    || door_lock_allowed_room_ids.contains(&room_id))
            {
                ensure!(node_json["locks"].len() == 1);
                let base_node_name = node_json["name"].as_str().unwrap().to_string();
                let lock = node_json["locks"][0].clone();
                let mut yields = node_json["yields"].clone();
                if lock["yields"] != JsonValue::Null {
                    yields = lock["yields"].clone();
                }
                node_json.remove("locks");
                let mut unlocked_node_json = node_json.clone();
                if yields != JsonValue::Null {
                    node_json.remove("yields");
                }
                node_json["name"] = JsonValue::String(base_node_name.clone() + " (locked)");
                node_json["nodeType"] = JsonValue::String("junction".to_string());

                unlocked_node_json["id"] = next_node_id.into();

                // Pit Room is a special case: since the doors can always be freely opened, we don't use
                // the unlocked nodes in the door edges.
                if room_json["name"] != "Pit Room" {
                    self.unlocked_node_map
                        .insert((room_id, node_id), next_node_id.into());
                }
                // Adding spawnAt helps shorten/clean spoiler log but interferes with the implicit leaveWithGMode:
                // unlocked_node_json["spawnAt"] = node_id.into();
                unlocked_node_json["name"] =
                    JsonValue::String(base_node_name.clone() + " (unlocked)");
                if yields != JsonValue::Null {
                    unlocked_node_json["yields"] = yields.clone();
                }

                let mut unlock_strats = lock["unlockStrats"].clone();
                if lock["name"].as_str().unwrap() == "Phantoon Fight" {
                    unlock_strats = json::array![
                        {
                            "name": "Base",
                            "requires": [
                                {"enemyKill":{
                                    "enemies": [
                                        [ "Phantoon" ]
                                    ]
                                }},
                            ]
                        }
                    ];
                }

                extra_nodes.push(unlocked_node_json);

                let unlock_reqs = json::object! {
                    "or": unlock_strats.members()
                            .map(|x| json::object!{"and": x["requires"].clone()})
                            .collect::<Vec<JsonValue>>()
                };
                if room_id != 12 {
                    // Exclude lock requirements in Pit Room since with randomizer changes these become free
                    self.node_lock_req_json
                        .insert((room_id, node_id), unlock_reqs);
                }

                if lock.has_key("lock") {
                    ensure!(lock["lock"].is_array());
                    for strat in &mut unlock_strats.members_mut() {
                        for req in lock["lock"].members() {
                            strat["requires"].push(req.clone())?;
                        }
                    }
                }

                if yields != JsonValue::Null
                    && obstacle_flags.contains(&yields[0].as_str().unwrap())
                {
                    obstacle_flag = Some(yields[0].as_str().unwrap().to_string())
                }

                // Strats from locked node to unlocked node
                for strat in unlock_strats.members() {
                    let mut new_strat = strat.clone();
                    new_strat["link"] = json::array![node_id, next_node_id];
                    if let Some(obs) = &obstacle_flag {
                        if !new_strat.has_key("clearsObstacles") {
                            new_strat["clearsObstacles"] = json::array![];
                        }
                        new_strat["clearsObstacles"].push(JsonValue::String(obs.clone()))?;
                    }
                    extra_strats.push(new_strat);
                }

                // Strat back from unlocked node to locked node
                let strat_backward = json::object! {
                    "link": [next_node_id, node_id],
                    "name": "Base",
                    "notable": false,
                    "requires": [],
                };
                extra_strats.push(strat_backward);

                next_node_id += 1;
            }
        }

        for extra_node in extra_nodes {
            new_room_json["nodes"].push(extra_node).unwrap();
        }
        for strat in extra_strats {
            new_room_json["strats"].push(strat).unwrap();
        }

        if obstacle_flag.is_some() {
            let obstacle_flag_name = obstacle_flag.as_ref().unwrap();
            if !new_room_json.has_key("obstacles") {
                new_room_json["obstacles"] = json::array![];
            }
            new_room_json["obstacles"]
                .push(json::object! {
                    "id": obstacle_flag_name.to_string(),
                    "name": obstacle_flag_name.to_string(),
                })
                .unwrap();
            ensure!(new_room_json["strats"].is_array());

            // For each strat requiring one of the "obstacle flags" listed above, create an alternative strat
            // depending on the corresponding obstacle instead:
            let mut new_strats: Vec<JsonValue> = Vec::new();
            for strat in new_room_json["strats"].members_mut() {
                let json_obstacle_flag_name = JsonValue::String(obstacle_flag_name.clone());
                let mut new_strat = strat.clone();
                let mut found = false;
                for req in new_strat["requires"].members_mut() {
                    if req == &json_obstacle_flag_name {
                        *req = json::object! {
                            "obstaclesCleared": [obstacle_flag_name.to_string()]
                        };
                        found = true;
                    }
                }
                if found {
                    new_strats.push(new_strat);
                }
            }
            for strat in new_strats {
                new_room_json["strats"].push(strat).unwrap();
            }
        }

        for strat_json in new_room_json["strats"].members_mut() {
            let strat_name = strat_json["name"].as_str().unwrap();
            if ignored_notable_strats.contains(strat_name) {
                if strat_json["notable"].as_bool() == Some(true) {
                    self.ignored_notable_strats.insert(strat_name.to_string());
                }
                strat_json["notable"] = JsonValue::Boolean(false);
            }
        }
        Ok(new_room_json)
    }

    pub fn get_obstacle_data(
        &self,
        strat_json: &JsonValue,
        room_json: &JsonValue,
        from_obstacles_bitmask: ObstacleMask,
        obstacles_idx_map: &HashMap<String, usize>,
        requires_json: &mut Vec<JsonValue>,
    ) -> Result<ObstacleMask> {
        let mut to_obstacles_bitmask = from_obstacles_bitmask;
        if strat_json.has_key("obstacles") {
            ensure!(strat_json["obstacles"].is_array());
            for obstacle in strat_json["obstacles"].members() {
                let obstacle_idx = obstacles_idx_map[obstacle["id"].as_str().unwrap()];
                to_obstacles_bitmask |= 1 << obstacle_idx;
                if (1 << obstacle_idx) & from_obstacles_bitmask == 0 {
                    ensure!(obstacle["requires"].is_array());
                    requires_json.extend(obstacle["requires"].members().map(|x| x.clone()));
                    let room_obstacle = &room_json["obstacles"][obstacle_idx];
                    if room_obstacle.has_key("requires") {
                        ensure!(room_obstacle["requires"].is_array());
                        requires_json
                            .extend(room_obstacle["requires"].members().map(|x| x.clone()));
                    }
                    if obstacle.has_key("additionalObstacles") {
                        ensure!(obstacle["additionalObstacles"].is_array());
                        for additional_obstacle_id in obstacle["additionalObstacles"].members() {
                            let additional_obstacle_idx =
                                obstacles_idx_map[additional_obstacle_id.as_str().unwrap()];
                            to_obstacles_bitmask |= 1 << additional_obstacle_idx;
                        }
                    }
                }
            }
        }
        if strat_json.has_key("clearsObstacles") {
            ensure!(strat_json["clearsObstacles"].is_array());
            for obstacle_id in strat_json["clearsObstacles"].members() {
                let obstacle_idx = obstacles_idx_map[obstacle_id.as_str().unwrap()];
                to_obstacles_bitmask |= 1 << obstacle_idx;
            }
        }
        if strat_json.has_key("resetsObstacles") {
            ensure!(strat_json["resetsObstacles"].is_array());
            for obstacle_id in strat_json["resetsObstacles"].members() {
                let obstacle_idx = obstacles_idx_map[obstacle_id.as_str().unwrap()];
                to_obstacles_bitmask &= !(1 << obstacle_idx);
            }
        }
        Ok(to_obstacles_bitmask)
    }

    pub fn parse_note(&self, note: &JsonValue) -> Vec<String> {
        if note.is_string() {
            vec![note.as_str().unwrap().to_string()]
        } else if note.is_array() {
            note.members()
                .map(|x| x.as_str().unwrap().to_string())
                .collect()
        } else {
            vec![]
        }
    }

    fn get_node_physics(&self, node_json: &JsonValue) -> Result<String> {
        // TODO: handle case with multiple environments
        ensure!(node_json["doorEnvironments"].is_array());
        ensure!(node_json["doorEnvironments"].len() == 1);
        return Ok(node_json["doorEnvironments"][0]["physics"]
            .as_str()
            .unwrap()
            .to_string());
    }

    fn get_room_heated(&self, room_json: &JsonValue, node_id: NodeId) -> Result<bool> {
        if !room_json.has_key("roomEnvironments") {
            // TODO: add roomEnvironments to Toilet Bowl so we don't need to skip this here
            return Ok(false);
        }
        ensure!(room_json["roomEnvironments"].is_array());
        for env in room_json["roomEnvironments"].members() {
            if env.has_key("entranceNodes") {
                ensure!(env["entranceNodes"].is_array());
                if !env["entranceNodes"].members().any(|x| x == node_id) {
                    continue;
                }
            }
            return Ok(env["heated"]
                .as_bool()
                .context("Expecting 'heated' to be a bool")?);
        }
        bail!("No match for node {} in roomEnvironments", node_id);
    }

    pub fn all_links(&self) -> impl Iterator<Item = &Link> {
        self.links
            .iter()
            .chain(self.node_exits.values().flatten())
    }

    fn process_room(&mut self, room_json: &JsonValue) -> Result<()> {
        let room_id = room_json["id"].as_usize().unwrap();
        self.room_json_map.insert(room_id, room_json.clone());

        let mut room_ptr =
            parse_int::parse::<usize>(room_json["roomAddress"].as_str().unwrap()).unwrap();
        self.raw_room_id_by_ptr.insert(room_ptr, room_id);
        if room_ptr == 0x7D408 {
            room_ptr = 0x7D5A7; // Treat Toilet Bowl as part of Aqueduct
        } else if room_ptr == 0x7D69A {
            room_ptr = 0x7D646; // Treat East Pants Room as part of Pants Room
        } else if room_ptr == 0x7968F {
            room_ptr = 0x793FE; // Treat Homing Geemer Room as part of West Ocean
        } else {
            self.room_id_by_ptr.insert(room_ptr, room_id);
        }
        self.room_ptr_by_id.insert(room_id, room_ptr);

        // Process obstacles:
        let obstacles_idx_map: HashMap<String, usize> = if room_json.has_key("obstacles") {
            ensure!(room_json["obstacles"].is_array());
            room_json["obstacles"]
                .members()
                .enumerate()
                .map(|(i, x)| (x["id"].as_str().unwrap().to_string(), i))
                .collect()
        } else {
            HashMap::new()
        };
        let num_obstacles = obstacles_idx_map.len();
        self.room_num_obstacles.insert(room_id, num_obstacles);
        self.room_obstacle_idx_map
            .insert(room_id, obstacles_idx_map.clone());

        // Process nodes:
        ensure!(room_json["nodes"].is_array());
        for node_json in room_json["nodes"].members() {
            let node_id = node_json["id"].as_usize().unwrap();
            self.node_json_map
                .insert((room_id, node_id), node_json.clone());
            if node_json.has_key("nodeAddress") {
                let mut node_ptr =
                    parse_int::parse::<usize>(node_json["nodeAddress"].as_str().unwrap()).unwrap();
                // Convert East Pants Room door pointers to corresponding Pants Room pointers
                if node_ptr == 0x1A7BC {
                    node_ptr = 0x1A798;
                } else if node_ptr == 0x1A7B0 {
                    node_ptr = 0x1A7A4;
                }
                self.node_ptr_map.insert((room_id, node_id), node_ptr);
                if (room_id, node_id) != (32, 7) && (room_id, node_id) != (32, 8) {
                    self.reverse_node_ptr_map.insert(node_ptr, (room_id, node_id));
                }
            }
            for obstacle_bitmask in 0..(1 << num_obstacles) {
                self.vertex_isv.add(&(room_id, node_id, obstacle_bitmask));
            }
        }
        for node_json in room_json["nodes"].members() {
            let node_id = node_json["id"].as_usize().unwrap();
            if node_json.has_key("runways") {
                ensure!(node_json["runways"].is_array());
                let mut runway_vec: Vec<Runway> = vec![];
                for runway_json in node_json["runways"].members() {
                    ensure!(runway_json["strats"].is_array());
                    for strat_json in runway_json["strats"].members() {
                        ensure!(strat_json["requires"].is_array());
                        let requires_json: Vec<JsonValue> = strat_json["requires"]
                            .members()
                            .map(|x| x.clone())
                            .collect();
                        let mut ctx = RequirementContext::default();
                        ctx.room_id = room_id;
                        let requirement =
                            Requirement::make_and(self.parse_requires_list(&requires_json, &ctx)?);
                        if strat_json.has_key("obstacles") {
                            // TODO: handle obstacles in runways
                            continue;
                        }
                        let heated = self.get_room_heated(room_json, node_id)?;
                        let physics_res = self.get_node_physics(node_json);
                        if let Ok(physics) = physics_res {
                            let runway = Runway {
                                name: runway_json["name"].as_str().unwrap().to_string(),
                                length: runway_json["length"].as_i32().unwrap(),
                                open_end: runway_json["openEnd"].as_i32().unwrap(),
                                requirement,
                                physics: physics.clone(),
                                heated,
                                usable_coming_in: runway_json["usableComingIn"]
                                    .as_bool()
                                    .unwrap_or(true),
                            };
                            // info!("Runway: {:?}", runway);
                            runway_vec.push(runway);

                            // Temporary while migration is in process -- Create new-style exit-condition strat:
                            let vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, 0)];
                            let lock_req = if let Some(lock_req_json) =
                                self.node_lock_req_json.get(&(room_id, node_id))
                            {
                                // This accounts for requirements to unlock a gray door before performing a cross-room
                                // strat through it:
                                self.parse_requirement(&lock_req_json.clone(), &ctx)?
                            } else {
                                Requirement::Free
                            };

                            let runway_geometry = parse_runway_geometry(runway_json)?;
                            let effective_length =
                                compute_runway_effective_length(&runway_geometry);
                            let exit_condition = ExitCondition::LeaveWithRunway {
                                effective_length,
                                heated,
                                physics: Some(parse_physics(&physics)?),
                            };

                            let link = Link {
                                from_vertex_id: vertex_id,
                                to_vertex_id: vertex_id,
                                requirement: lock_req,
                                entrance_condition: None,
                                exit_condition: Some(exit_condition),
                                bypasses_door_shell: false,
                                notable_strat_name: None,
                                strat_name: strat_json["name"].as_str().unwrap().to_string(),
                                strat_notes: vec![],
                                sublinks: vec![],
                            };
                            self.node_exits
                                .entry((room_id, node_id))
                                .or_insert(vec![])
                                .push(link);
                        } else {
                            // info!("Invalid physics in runway: {} - {}", room_json["name"], runway_json["name"])
                        }
                    }
                }
                self.node_runways_map.insert((room_id, node_id), runway_vec);
            } else {
                self.node_runways_map.insert((room_id, node_id), vec![]);
            }

            if node_json.has_key("jumpways") {
                ensure!(node_json["jumpways"].is_array());
                let mut jumpway_vec: Vec<Jumpway> = vec![];
                for jumpway_json in node_json["jumpways"].members() {
                    ensure!(jumpway_json["requires"].is_array());
                    let requires_json: Vec<JsonValue> = jumpway_json["requires"]
                        .members()
                        .map(|x| x.clone())
                        .collect();
                    let mut ctx = RequirementContext::default();
                    ctx.room_id = room_id;
                    let requirement =
                        Requirement::make_and(self.parse_requires_list(&requires_json, &ctx)?);

                    let jumpway = Jumpway {
                        name: jumpway_json["name"].as_str().unwrap().to_string(),
                        jumpway_type: jumpway_json["jumpwayType"].as_str().unwrap().to_string(),
                        height: jumpway_json["height"].as_f32().unwrap(),
                        left_position: jumpway_json["leftPosition"].as_f32(),
                        right_position: jumpway_json["rightPosition"].as_f32(),
                        requirement,
                    };
                    jumpway_vec.push(jumpway);
                }
                self.node_jumpways_map
                    .insert((room_id, node_id), jumpway_vec);
            } else {
                self.node_jumpways_map.insert((room_id, node_id), vec![]);
            }

            if node_json.has_key("leaveWithGModeSetup") {
                ensure!(node_json["leaveWithGModeSetup"].is_array());
                let mut leave_with_gmode_setup_vec: Vec<LeaveWithGModeSetup> = vec![];
                for leave_with_gmode_setup_json in node_json["leaveWithGModeSetup"].members() {
                    ensure!(leave_with_gmode_setup_json["strats"].is_array());
                    for strat_json in leave_with_gmode_setup_json["strats"].members() {
                        ensure!(strat_json["requires"].is_array());
                        let requires_json: Vec<JsonValue> = strat_json["requires"]
                            .members()
                            .map(|x| x.clone())
                            .collect();
                        let mut ctx = RequirementContext::default();
                        ctx.room_id = room_id;
                        let knockback = leave_with_gmode_setup_json["knockback"]
                            .as_bool()
                            .unwrap_or(true);
                        let requirement =
                            Requirement::make_and(self.parse_requires_list(&requires_json, &ctx)?);
                        let leave_with_gmode_setup = LeaveWithGModeSetup {
                            knockback,
                            requirement: requirement.clone(),
                        };
                        leave_with_gmode_setup_vec.push(leave_with_gmode_setup);

                        // Temporary while migration is in process -- Create new-style exit-condition strat:
                        let vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, 0)];
                        let lock_req = if let Some(lock_req_json) =
                            self.node_lock_req_json.get(&(room_id, node_id))
                        {
                            // This accounts for requirements to unlock a gray door before performing a cross-room
                            // strat through it:
                            self.parse_requirement(&lock_req_json.clone(), &ctx)?
                        } else {
                            Requirement::Free
                        };
                        let exit_condition = ExitCondition::LeaveWithGModeSetup { knockback };
                        let link = Link {
                            from_vertex_id: vertex_id,
                            to_vertex_id: vertex_id,
                            requirement: Requirement::make_and(vec![requirement, lock_req]),
                            entrance_condition: None,
                            exit_condition: Some(exit_condition),
                            bypasses_door_shell: false,
                            notable_strat_name: None,
                            strat_name: strat_json["name"].as_str().unwrap().to_string(),
                            strat_notes: vec![],
                            sublinks: vec![],
                        };
                        self.node_exits
                            .entry((room_id, node_id))
                            .or_insert(vec![])
                            .push(link);
                    }
                }
                self.node_leave_with_gmode_setup_map
                    .insert((room_id, node_id), leave_with_gmode_setup_vec);
            } else {
                self.node_leave_with_gmode_setup_map
                    .insert((room_id, node_id), vec![]);
            }

            // Explicit leaveWithGMode:
            if node_json.has_key("leaveWithGMode") {
                ensure!(node_json["leaveWithGMode"].is_array());
                let mut leave_with_gmode_vec: Vec<LeaveWithGMode> = vec![];
                for leave_with_gmode_json in node_json["leaveWithGMode"].members() {
                    ensure!(leave_with_gmode_json["strats"].is_array());
                    for strat_json in leave_with_gmode_json["strats"].members() {
                        ensure!(strat_json["requires"].is_array());
                        let requires_json: Vec<JsonValue> = strat_json["requires"]
                            .members()
                            .map(|x| x.clone())
                            .collect();
                        let mut ctx = RequirementContext::default();
                        ctx.room_id = room_id;
                        let requirement =
                            Requirement::make_and(self.parse_requires_list(&requires_json, &ctx)?);
                        let artificial_morph = leave_with_gmode_json["leavesWithArtificialMorph"]
                            .as_bool()
                            .context("Expecting field leavesWithArtificialMorph")?;
                        let leave_with_gmode = LeaveWithGMode {
                            artificial_morph,
                            requirement: requirement.clone(),
                        };
                        leave_with_gmode_vec.push(leave_with_gmode);

                        // Temporary while migration is in process -- Create new-style exit-condition strat:
                        let vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, 0)];
                        let lock_req = if let Some(lock_req_json) =
                            self.node_lock_req_json.get(&(room_id, node_id))
                        {
                            // This accounts for requirements to unlock a gray door before performing a cross-room
                            // strat through it:
                            self.parse_requirement(&lock_req_json.clone(), &ctx)?
                        } else {
                            Requirement::Free
                        };
                        let exit_condition = ExitCondition::LeaveWithGMode {
                            morphed: artificial_morph,
                        };
                        let link = Link {
                            from_vertex_id: vertex_id,
                            to_vertex_id: vertex_id,
                            requirement: Requirement::make_and(vec![requirement, lock_req]),
                            entrance_condition: None,
                            exit_condition: Some(exit_condition),
                            bypasses_door_shell: false,
                            notable_strat_name: None,
                            strat_name: strat_json["name"].as_str().unwrap().to_string(),
                            strat_notes: vec![],
                            sublinks: vec![],
                        };
                        self.node_exits
                            .entry((room_id, node_id))
                            .or_insert(vec![])
                            .push(link);
                    }
                }
                self.node_leave_with_gmode_map
                    .insert((room_id, node_id), leave_with_gmode_vec);
            } else {
                self.node_leave_with_gmode_map
                    .insert((room_id, node_id), vec![]);
            }

            // Implicit leaveWithGMode:
            if !node_json.has_key("spawnAt") && node_json["nodeType"].as_str().unwrap() == "door" {
                // Old style of cross-room strat:
                for artificial_morph in [false, true] {
                    self.node_leave_with_gmode_map
                        .get_mut(&(room_id, node_id))
                        .unwrap()
                        .push(LeaveWithGMode {
                            artificial_morph,
                            requirement: Requirement::ComeInWithGMode {
                                room_id,
                                node_ids: vec![node_id],
                                mode: "direct".to_string(),
                                mobility: "any".to_string(),
                                artificial_morph,
                            },
                        });
                }

                // New style of cross-room strat:
                let vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, 0)];
                for morphed in [false, true] {
                    let exit_condition = ExitCondition::LeaveWithGMode { morphed };
                    let link = Link {
                        from_vertex_id: vertex_id,
                        to_vertex_id: vertex_id,
                        requirement: Requirement::Free,
                        entrance_condition: Some(EntranceCondition::ComeInWithGMode {
                            mode: GModeMode::Direct,
                            morphed,
                            mobility: GModeMobility::Any,
                        }),
                        exit_condition: Some(exit_condition),
                        bypasses_door_shell: false,
                        notable_strat_name: None,
                        strat_name: "G-Mode Go Back Through Door".to_string(),
                        strat_notes: vec![],
                        sublinks: vec![],
                    };
                    self.node_exits
                        .entry((room_id, node_id))
                        .or_insert(vec![])
                        .push(link);
                }
            }

            if node_json.has_key("gModeImmobile") {
                let gmode_immobile_json = &node_json["gModeImmobile"];
                ensure!(gmode_immobile_json["requires"].is_array());
                let requires_json: Vec<JsonValue> = gmode_immobile_json["requires"]
                    .members()
                    .map(|x| x.clone())
                    .collect();
                let mut ctx = RequirementContext::default();
                ctx.room_id = room_id;
                let requirement =
                    Requirement::make_and(self.parse_requires_list(&requires_json, &ctx)?);
                let gmode_immobile = GModeImmobile {
                    requirement: requirement.clone(),
                };
                self.node_gmode_immobile_map
                    .insert((room_id, node_id), gmode_immobile);

                // Temporary while migration is in process -- Create new-style strat with gModeRegainMobility:
                let vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, 0)];
                let link = Link {
                    from_vertex_id: vertex_id,
                    to_vertex_id: vertex_id,
                    requirement: requirement,
                    entrance_condition: None,
                    exit_condition: None,
                    bypasses_door_shell: false,
                    notable_strat_name: None,
                    strat_name: "G-Mode Immobile".to_string(),
                    strat_notes: vec![],
                    sublinks: vec![],
                };
                self.node_gmode_regain_mobility
                    .entry((room_id, node_id))
                    .or_insert(vec![])
                    .push((link, GModeRegainMobility {}));
            }

            if node_json.has_key("spawnAt") {
                let spawn_node_id = node_json["spawnAt"].as_usize().unwrap();
                self.node_spawn_at_map
                    .insert((room_id, node_id), spawn_node_id);
            }
        }

        // Process roomwide reusable strats:
        let mut roomwide_notable: HashMap<String, JsonValue> = HashMap::new();
        for strat in room_json["reusableRoomwideNotable"].members() {
            roomwide_notable.insert(strat["name"].as_str().unwrap().to_string(), strat.clone());
        }

        // Process strats:
        ensure!(room_json["strats"].is_array());
        for strat_json in room_json["strats"].members() {
            let from_node_id = strat_json["link"][0].as_usize().unwrap();
            let to_node_id = strat_json["link"][1].as_usize().unwrap();
            // TODO: deal with heated room more explicitly for Volcano Room, instead of guessing based on node ID:
            let from_heated = self.get_room_heated(room_json, from_node_id)?;

            let to_heated = self.get_room_heated(room_json, to_node_id)?;
            let physics_res = self.get_node_physics(&self.node_json_map[&(room_id, to_node_id)]);
            let physics: Option<Physics> = if let Ok(physics_str) = &physics_res {
                Some(parse_physics(&physics_str)?)
            } else {
                None
            };

            let entrance_condition: Option<EntranceCondition> =
                if strat_json.has_key("entranceCondition") {
                    ensure!(strat_json["entranceCondition"].is_object());
                    Some(parse_entrance_condition(
                        &strat_json["entranceCondition"],
                        from_heated,
                    )?)
                } else {
                    None
                };
            let exit_condition: Option<ExitCondition> = if strat_json.has_key("exitCondition") {
                ensure!(strat_json["exitCondition"].is_object());
                Some(parse_exit_condition(
                    &strat_json["exitCondition"],
                    to_heated,
                    physics,
                )?)
            } else {
                None
            };
            let gmode_regain_mobility: Option<GModeRegainMobility> =
                if strat_json.has_key("gModeRegainMobility") {
                    ensure!(strat_json["gModeRegainMobility"].is_object());
                    Some(GModeRegainMobility {})
                } else {
                    None
                };

            for from_obstacles_bitmask in 0..(1 << num_obstacles) {
                if entrance_condition.is_some() && from_obstacles_bitmask != 0 {
                    continue;
                }
                ensure!(strat_json["requires"].is_array());
                let mut requires_json: Vec<JsonValue> = strat_json["requires"]
                    .members()
                    .map(|x| x.clone())
                    .collect();

                let to_obstacles_bitmask = self.get_obstacle_data(
                    strat_json,
                    room_json,
                    from_obstacles_bitmask,
                    &obstacles_idx_map,
                    &mut requires_json,
                )?;
                let ctx = RequirementContext {
                    room_id,
                    _from_node_id: from_node_id,
                    room_heated: from_heated || to_heated,
                    from_obstacles_bitmask,
                    obstacles_idx_map: Some(&obstacles_idx_map),
                };
                let mut requires_vec = self.parse_requires_list(&requires_json, &ctx)?;
                let strat_name = strat_json["name"].as_str().unwrap().to_string();
                let strat_notes = self.parse_note(&strat_json["note"]);
                let notable = strat_json["notable"].as_bool().unwrap_or(false);
                let mut notable_strat_name = strat_name.clone();
                if notable {
                    let mut notable_strat_note: Vec<String> = strat_notes.clone();
                    if strat_json.has_key("reusableRoomwideNotable") {
                        notable_strat_name = strat_json["reusableRoomwideNotable"]
                            .as_str()
                            .unwrap()
                            .to_string();
                        if !roomwide_notable.contains_key(&notable_strat_name) {
                            bail!(
                                "Unrecognized reusable notable strat name: {}",
                                notable_strat_name
                            );
                        }
                        notable_strat_note =
                            self.parse_note(&roomwide_notable[&notable_strat_name]["note"]);
                    }
                    let strat_id = self.notable_strat_isv.add(&notable_strat_name);
                    requires_vec.push(Requirement::Strat(strat_id));
                    let area = format!(
                        "{} - {}",
                        room_json["area"].as_str().unwrap(),
                        room_json["subarea"].as_str().unwrap()
                    );
                    self.strat_area.insert(notable_strat_name.clone(), area);
                    self.strat_room.insert(
                        notable_strat_name.clone(),
                        room_json["name"].as_str().unwrap().to_string(),
                    );
                    self.strat_description
                        .insert(notable_strat_name.clone(), notable_strat_note.join(" "));
                }

                if exit_condition.is_some() {
                    if let Some(lock_req_json) = self.node_lock_req_json.get(&(room_id, to_node_id))
                    {
                        // This accounts for requirements to unlock a gray door before performing a cross-room
                        // strat through it:
                        // TODO: Also consider the possibility of the gray door being open due to having just
                        // entered
                        requires_vec.push(self.parse_requirement(&lock_req_json.clone(), &ctx)?);
                    }
                }

                let bypasses_door_shell = strat_json["bypassesDoorShell"].as_bool().unwrap_or(false);
                if bypasses_door_shell {
                    requires_vec.push(Requirement::Tech(self.tech_isv.index_by_key["canSkipDoorLock"]));
                }

                let requirement = Requirement::make_and(requires_vec);
                if let Requirement::Never = requirement {
                    continue;
                }
                let from_vertex_id =
                    self.vertex_isv.index_by_key[&(room_id, from_node_id, from_obstacles_bitmask)];
                let to_vertex_id =
                    self.vertex_isv.index_by_key[&(room_id, to_node_id, to_obstacles_bitmask)];
                let link = Link {
                    from_vertex_id,
                    to_vertex_id,
                    requirement: requirement.clone(),
                    entrance_condition: entrance_condition.clone(),
                    exit_condition: exit_condition.clone(),
                    bypasses_door_shell,
                    notable_strat_name: if notable {
                        Some(notable_strat_name)
                    } else {
                        None
                    },
                    strat_name: strat_name.clone(),
                    strat_notes,
                    sublinks: vec![],
                };
                if gmode_regain_mobility.is_some() {
                    if entrance_condition.is_some() || exit_condition.is_some() {
                        bail!("gModeRegainMobility combined with entranceCondition or exitCondition is not allowed.");
                    }
                    if from_node_id != to_node_id {
                        bail!("gModeRegainMobility `from` and `to` node must be equal.");
                    }
                    self.node_gmode_regain_mobility
                        .entry((room_id, to_node_id))
                        .or_insert(vec![])
                        .push((link, gmode_regain_mobility.clone().unwrap()))
                } else if exit_condition.is_some() {
                    self.node_exits
                        .entry((room_id, to_node_id))
                        .or_insert(vec![])
                        .push(link);
                } else {
                    self.links.push(link);
                }

                // Temporary while in the middle of migration -- create old-style runways, etc.:
                if from_node_id == to_node_id {
                    match exit_condition {
                        Some(ExitCondition::LeaveWithRunway {
                            heated, physics, ..
                        }) => {
                            if let Ok(physics_str) = &physics_res {
                                let mut runway_reqs = vec![requirement];
                                if physics != Some(Physics::Air) {
                                    runway_reqs.push(Requirement::Item(Item::Gravity as usize));
                                }
                                // println!("{}", strat_json["exitCondition"].pretty(2));
                                self.node_runways_map
                                    .entry((room_id, to_node_id))
                                    .or_insert(vec![])
                                    .push(Runway {
                                        name: strat_name,
                                        length: strat_json["exitCondition"]["leaveWithRunway"]
                                            ["length"]
                                            .as_f32()
                                            .unwrap()
                                            as i32,
                                        open_end: strat_json["exitCondition"]["leaveWithRunway"]
                                            ["openEnd"]
                                            .as_i32()
                                            .unwrap(),
                                        requirement: Requirement::make_and(runway_reqs),
                                        physics: physics_str.to_string(),
                                        heated: heated,
                                        usable_coming_in: false,
                                    });
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    fn load_connections(&mut self) -> Result<()> {
        let connection_pattern =
            self.sm_json_data_path.to_str().unwrap().to_string() + "/connection/**/*.json";
        for entry in glob::glob(&connection_pattern)? {
            if let Ok(path) = entry {
                if !path.to_str().unwrap().contains("ceres") {
                    self.process_connections(&read_json(&path)?)?;
                }
            } else {
                bail!("Error processing connection path: {}", entry.err().unwrap());
            }
        }
        Ok(())
    }

    fn process_connections(&mut self, connection_file_json: &JsonValue) -> Result<()> {
        ensure!(connection_file_json["connections"].is_array());
        for connection in connection_file_json["connections"].members() {
            ensure!(connection["nodes"].is_array());
            ensure!(connection["nodes"].len() == 2);
            let src_pair = (
                connection["nodes"][0]["roomid"].as_usize().unwrap(),
                connection["nodes"][0]["nodeid"].as_usize().unwrap(),
            );
            let dst_pair = (
                connection["nodes"][1]["roomid"].as_usize().unwrap(),
                connection["nodes"][1]["nodeid"].as_usize().unwrap(),
            );
            self.add_connection(src_pair, dst_pair, &connection["nodes"][0]);
            self.add_connection(dst_pair, src_pair, &connection["nodes"][1]);
        }
        Ok(())
    }

    fn add_connection(
        &mut self,
        mut src: (RoomId, NodeId),
        dst: (RoomId, NodeId),
        conn: &JsonValue,
    ) {
        let src_ptr = self.node_ptr_map.get(&src).map(|x| *x);
        let dst_ptr = self.node_ptr_map.get(&dst).map(|x| *x);
        let is_bridge = src == (32, 7) || src == (32, 8);
        if src_ptr.is_some() || dst_ptr.is_some() {
            if !is_bridge {
                self.door_ptr_pair_map.insert((src_ptr, dst_ptr), src);
                self.reverse_door_ptr_pair_map
                    .insert(src, (src_ptr, dst_ptr));    
            }
            let pos = parse_door_position(conn["position"].as_str().unwrap()).unwrap();
            self.door_position.insert(src, pos);
            if self.unlocked_node_map.contains_key(&src) {
                let src_room_id = src.0;
                src = (src_room_id, self.unlocked_node_map[&src])
            }
            if !is_bridge {
                self.unlocked_door_ptr_pair_map
                    .insert((src_ptr, dst_ptr), src);
            }
            self.door_position.insert(src, pos);
        }
    }

    fn populate_target_locations(&mut self) -> Result<()> {
        // Flags that are relevant to track in the randomizer:
        let flag_set: HashSet<String> = [
            "f_ZebesAwake",
            "f_MaridiaTubeBroken",
            "f_ShaktoolDoneDigging",
            "f_UsedAcidChozoStatue",
            "f_DefeatedBotwoon",
            "f_DefeatedCrocomire",
            "f_DefeatedSporeSpawn",
            "f_DefeatedGoldenTorizo",
            "f_DefeatedKraid",
            "f_DefeatedPhantoon",
            "f_DefeatedDraygon",
            "f_DefeatedRidley",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();

        for (&(room_id, node_id), node_json) in &self.node_json_map {
            if node_json["nodeType"] == "item" {
                self.item_locations.push((room_id, node_id));
            }
            if node_json.has_key("utility") {
                if node_json["utility"].members().any(|x| x == "save") {
                    if room_id != 304 {
                        // room_id: 304 is the broken save room, which is not a logical save for the purposes of
                        // guaranteed early save station, which is all this is currently used for.
                        self.save_locations.push((room_id, node_id));
                    }
                }
            }
            if node_json.has_key("yields") {
                ensure!(node_json["yields"].len() >= 1);
                let flag_id = self.flag_isv.index_by_key[node_json["yields"][0].as_str().unwrap()];
                if flag_set.contains(&self.flag_isv.keys[flag_id]) {
                    let mut unlocked_node_id = node_id;
                    if self.unlocked_node_map.contains_key(&(room_id, node_id)) {
                        unlocked_node_id = self.unlocked_node_map[&(room_id, node_id)];
                    }
                    self.flag_locations
                        .push((room_id, unlocked_node_id, flag_id));
                }
            }
        }

        for &(room_id, node_id) in &self.item_locations {
            let num_obstacles = self.room_num_obstacles[&room_id];
            let mut vertex_ids: Vec<VertexId> = Vec::new();
            for obstacle_bitmask in 0..(1 << num_obstacles) {
                let vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, obstacle_bitmask)];
                vertex_ids.push(vertex_id);
                self.target_vertices.add(&vertex_id);
            }
            self.item_vertex_ids.push(vertex_ids);
        }

        for &(room_id, node_id, _flag_id) in &self.flag_locations {
            let num_obstacles = self.room_num_obstacles[&room_id];
            let mut vertex_ids: Vec<VertexId> = Vec::new();
            for obstacle_bitmask in 0..(1 << num_obstacles) {
                let vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, obstacle_bitmask)];
                vertex_ids.push(vertex_id);
                self.target_vertices.add(&vertex_id);
            }
            self.flag_vertex_ids.push(vertex_ids);
        }

        for &(room_id, node_id) in &self.save_locations {
            let vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, 0)];
            self.target_vertices.add(&vertex_id);
        }

        Ok(())
    }

    pub fn get_weapon_mask(&self, items: &[bool]) -> WeaponMask {
        let mut weapon_mask = 0;
        let implicit_item_requires: HashSet<String> =
            vec!["PowerBeam", "canUseGrapple", "canSpecialBeamAttack"]
                .into_iter()
                .map(|x| x.to_string())
                .collect();
        // TODO: possibly make this more efficient. We could avoid dealing with strings
        // and just use a pre-computed item bitmask per weapon. But not sure yet if it matters.
        'weapon: for (i, weapon_name) in self.weapon_isv.keys.iter().enumerate() {
            let weapon = &self.weapon_json_map[weapon_name];
            assert!(weapon["useRequires"].is_array());
            for item_name_json in weapon["useRequires"].members() {
                let item_name = item_name_json.as_str().unwrap();
                if implicit_item_requires.contains(item_name) {
                    continue;
                }
                let item_idx = self.item_isv.index_by_key[item_name];
                if !items[item_idx] {
                    continue 'weapon;
                }
            }
            weapon_mask |= 1 << i;
        }
        weapon_mask
    }

    fn load_escape_timings(&mut self, path: &Path) -> Result<()> {
        let escape_timings_str = std::fs::read_to_string(path)
            .with_context(|| format!("Unable to load escape timings at {}", path.display()))?;
        self.escape_timings = serde_json::from_str(&escape_timings_str)?;
        assert_eq!(self.escape_timings.len(), self.room_geometry.len());
        Ok(())
    }

    fn load_start_locations(&mut self, path: &Path) -> Result<()> {
        let start_locations_str = std::fs::read_to_string(path)
            .with_context(|| format!("Unable to load start locations at {}", path.display()))?;
        let mut start_locations: Vec<StartLocation> = serde_json::from_str(&start_locations_str)?;
        for loc in &mut start_locations {
            if loc.requires.is_none() {
                loc.requires_parsed = Some(Requirement::Free);
            } else {
                let mut req_json_list: Vec<JsonValue> = vec![];
                for req in loc.requires.as_ref().unwrap() {
                    let req_str = req.to_string();
                    let req_json = json::parse(&req_str)
                        .with_context(|| format!("Error parsing requires in {:?}", loc))?;
                    req_json_list.push(req_json);
                }
                let ctx = RequirementContext::default();
                let req_list = self.parse_requires_list(&req_json_list, &ctx)?;
                loc.requires_parsed = Some(Requirement::make_and(req_list));
            }
            if !self
                .vertex_isv
                .index_by_key
                .contains_key(&(loc.room_id, loc.node_id, 0))
            {
                panic!("Bad starting location: {:?}", loc);
            }
        }
        self.start_locations = start_locations;
        Ok(())
    }

    fn load_hub_locations(&mut self, path: &Path) -> Result<()> {
        let hub_locations_str = std::fs::read_to_string(path)
            .with_context(|| format!("Unable to load hub locations at {}", path.display()))?;
        let mut hub_locations: Vec<HubLocation> = serde_json::from_str(&hub_locations_str)?;
        for loc in &mut hub_locations {
            if loc.requires.is_none() {
                loc.requires_parsed = Some(Requirement::Free);
            } else {
                let mut req_json_list: Vec<JsonValue> = vec![];
                for req in loc.requires.as_ref().unwrap() {
                    let req_str = req.to_string();
                    let req_json = json::parse(&req_str)
                        .with_context(|| format!("Error parsing requires in {:?}", loc))?;
                    req_json_list.push(req_json);
                }
                let ctx = RequirementContext::default();
                let req_list = self.parse_requires_list(&req_json_list, &ctx)?;
                loc.requires_parsed = Some(Requirement::make_and(req_list));
            }
            if !self.vertex_isv.index_by_key.contains_key(&(loc.room_id, loc.node_id, 0)) {
                panic!("Bad hub location: {:?}", loc);
            }
        }
        self.hub_locations = hub_locations;
        Ok(())
    }

    fn load_room_geometry(&mut self, path: &Path) -> Result<()> {
        let room_geometry_str = std::fs::read_to_string(path)
            .with_context(|| format!("Unable to load room geometry at {}", path.display()))?;
        let room_geometry: Vec<RoomGeometry> = serde_json::from_str(&room_geometry_str)?;
        for (room_idx, room) in room_geometry.iter().enumerate() {
            self.room_idx_by_name.insert(room.name.clone(), room_idx);
            self.room_idx_by_ptr.insert(room.rom_address, room_idx);
            if let Some(twin_rom_address) = room.twin_rom_address {
                self.room_idx_by_ptr.insert(twin_rom_address, room_idx);
            }
            for (door_idx, door) in room.doors.iter().enumerate() {
                let door_ptr_pair = (door.exit_ptr, door.entrance_ptr);
                self.room_and_door_idxs_by_door_ptr_pair
                    .insert(door_ptr_pair, (room_idx, door_idx));
                let (room_id, node_id) = self.door_ptr_pair_map[&door_ptr_pair];
                self.node_coords.insert((room_id, node_id), (door.x, door.y));
            }
            for item in &room.items {
                let (room_id, node_id) = self.reverse_node_ptr_map[&item.addr];
                self.node_coords.insert((room_id, node_id), (item.x, item.y));
            }

            let room_id = self.room_id_by_ptr[&room.rom_address];
            let mut max_x = 0;
            let mut max_y = 0;
            for (node_id, tiles) in &room.node_tiles {
                self.node_tile_coords
                    .insert((room_id, *node_id), tiles.clone());
                let node_max_x = tiles.iter().map(|x| x.0).max().unwrap();
                let node_max_y = tiles.iter().map(|x| x.1).max().unwrap();
                if node_max_x > max_x {
                    max_x = node_max_x;
                }
                if node_max_y > max_y {
                    max_y = node_max_y;
                }
            }
            self.room_shape.insert(room_id, (max_x + 1, max_y + 1));

            if let Some(twin_rom_address) = room.twin_rom_address {
                let room_id = self.raw_room_id_by_ptr[&twin_rom_address];
                for (node_id, tiles) in room.twin_node_tiles.as_ref().unwrap() {
                    self.node_tile_coords.insert((room_id, *node_id), tiles.clone());
                }
            }
        }
        self.room_geometry = room_geometry;
        Ok(())
    }

    // fn load_palette(&mut self, json_path: &Path) -> Result<()> {
    //     let file = File::open(json_path)?;
    //     let json_value: serde_json::Value = serde_json::from_reader(file)?;
    //     for area_json in json_value.as_array().unwrap() {
    //         let mut pal_map: HashMap<TilesetIdx, ThemedTileset> = HashMap::new();
    //         for (tileset_idx_str, palette) in area_json.as_object().unwrap().iter() {
    //             let tileset_idx: usize = tileset_idx_str.parse()?;
    //             let mut pal = [[0u8; 3]; 128];
    //             for (i, color) in palette.as_array().unwrap().iter().enumerate() {
    //                 let color_arr = color.as_array().unwrap();
    //                 let r = color_arr[0].as_i64().unwrap();
    //                 let g = color_arr[1].as_i64().unwrap();
    //                 let b = color_arr[2].as_i64().unwrap();
    //                 pal[i][0] = r as u8;
    //                 pal[i][1] = g as u8;
    //                 pal[i][2] = b as u8;
    //             }

    //             // for i in 0..128 {
    //             //     for j in 0..3 {
    //             //         pal[i][j] = 0;
    //             //     }
    //             // }
    //             pal_map.insert(tileset_idx, ThemedTileset { palette: pal });
    //         }
    //         self.tileset_palette_themes.push(pal_map);
    //     }
    //     Ok(())
    // }

    fn load_themes(&mut self, base_path: &Path) -> Result<()> {
        let ignored_tileset_idxs = vec![
            1, // Red Crateria
            15, 16, 17, 18, 19, 20, // Ceres
        ];
        for (_area_idx, area) in [
            "crateria",
            "brinstar",
            "norfair",
            "wrecked_ship",
            "maridia",
            "tourian",
        ]
        .into_iter()
        .enumerate()
        {
            let sce_path = base_path.join(area).join("Export/Tileset/SCE");
            let mut pal_map: HashMap<TilesetIdx, ThemedPaletteTileset> = HashMap::new();
            let tilesets_it = std::fs::read_dir(&sce_path).with_context(|| {
                format!("Unable to read Mosaic tilesets at {}", sce_path.display())
            })?;
            for tileset_dir in tilesets_it {
                let tileset_dir = tileset_dir?;
                let tileset_idx =
                    usize::from_str_radix(tileset_dir.file_name().to_str().unwrap(), 16)?;
                // if !tileset_idxs[area_idx].contains(&tileset_idx) {
                //     continue;
                // }
                if ignored_tileset_idxs.contains(&tileset_idx) {
                    continue;
                }
                let tileset_path = tileset_dir.path();
                let palette_path = tileset_path.join("palette.snes");
                let palette_bytes = std::fs::read(&palette_path).with_context(|| {
                    format!(
                        "Unable to load Mosaic palette at {}",
                        palette_path.display()
                    )
                })?;
                let palette = decode_palette(&palette_bytes);

                let gfx8x8_path = tileset_path.join("8x8tiles.gfx");
                let gfx8x8_bytes = std::fs::read(&gfx8x8_path).with_context(|| {
                    format!("Unable to load Mosaic 8x8 gfx at {}", gfx8x8_path.display())
                })?;

                let gfx16x16_path = tileset_path.join("16x16tiles.ttb");
                let gfx16x16_bytes = std::fs::read(&gfx16x16_path).with_context(|| {
                    format!(
                        "Unable to load Mosaic 16x16 gfx at {}",
                        gfx16x16_path.display()
                    )
                })?;

                pal_map.insert(
                    tileset_idx,
                    ThemedPaletteTileset {
                        palette,
                        gfx8x8: gfx8x8_bytes,
                        gfx16x16: gfx16x16_bytes,
                    },
                );
            }
            self.tileset_palette_themes.push(pal_map);
        }
        Ok(())
    }

    fn extract_all_tech_dependencies(&mut self) -> Result<()> {
        let tech_vec = self.tech_isv.keys.clone();
        for tech in &tech_vec {
            let req = self.get_tech_requirement(tech)?;
            let deps: Vec<String> = self
                .extract_tech_dependencies(&req)
                .into_iter()
                .filter(|x| x != tech)
                .collect();
            self.tech_dependencies.insert(tech.clone(), deps);
        }
        Ok(())
    }

    fn extract_all_strat_dependencies(&mut self) -> Result<()> {
        let links: Vec<Link> = self.all_links().cloned().collect();
        for link in &links {
            if let Some(notable_strat_name) = link.notable_strat_name.clone() {
                let deps: HashSet<String> = self.extract_tech_dependencies(&link.requirement);
                self.strat_dependencies
                    .insert(notable_strat_name.clone(), deps.into_iter().collect());
            }
        }
        Ok(())
    }

    fn is_base_req(req: &Requirement) -> bool {
        match req {
            Requirement::AdjacentRunway { .. } => false,
            Requirement::AdjacentJumpway { .. } => false,
            Requirement::CanComeInCharged { .. } => false,
            Requirement::ComeInWithRMode { .. } => false,
            Requirement::ComeInWithGMode { .. } => false,
            Requirement::DoorUnlocked { .. } => false,
            Requirement::And(and_reqs) => and_reqs.iter().all(|x| Self::is_base_req(x)),
            Requirement::Or(or_reqs) => or_reqs.iter().all(|x| Self::is_base_req(x)),
            _ => true,
        }
    }

    fn is_base_link(link: &Link) -> bool {
        if link.entrance_condition.is_some() || link.bypasses_door_shell {
            return false;
        }
        Self::is_base_req(&link.requirement)
    }

    fn filter_links(&mut self) {
        for link in &self.links {
            if Self::is_base_link(link) {
                self.base_links.push(link.clone());
            } else {
                self.seed_links.push(link.clone());
            }
        }
        self.base_links_data =
            LinksDataGroup::new(self.base_links.clone(), self.vertex_isv.keys.len(), 0);
    }

    pub fn load_title_screens(&mut self, path: &Path) -> Result<()> {
        info!("Loading title screens");
        let file_it = path
            .read_dir()
            .with_context(|| format!("Unable to read title screen directory at {}", path.display()))?;
        for file in file_it {
            let file = file?;
            let filename = file.file_name().into_string().unwrap();
            let img = read_image(&file.path())?;

            if filename.starts_with("TL") {
                self.title_screen_data.top_left.push(img);
            } else if filename.starts_with("TR") {
                self.title_screen_data.top_right.push(img);
            } else if filename.starts_with("BL") {
                self.title_screen_data.bottom_left.push(img);
            } else if filename.starts_with("BR") {
                self.title_screen_data.bottom_right.push(img);
            } else if filename == "map.png" {
                self.title_screen_data.map_station = img;
            }
        }
        Ok(())
    }

    pub fn load(
        sm_json_data_path: &Path,
        room_geometry_path: &Path,
        palette_theme_path: &Path,
        escape_timings_path: &Path,
        start_locations_path: &Path,
        hub_locations_path: &Path,
        mosaic_path: &Path,
        title_screen_path: &Path,
    ) -> Result<GameData> {
        let mut game_data = GameData::default();
        game_data.sm_json_data_path = sm_json_data_path.to_owned();

        game_data.load_items_and_flags()?;
        game_data.load_tech()?;
        game_data.load_helpers()?;

        // Patch the h_heatProof and h_heatResistant to take into account the complementary suit
        // patch, where only Varia (and not Gravity) provides heat protection:
        *game_data.helper_json_map.get_mut("h_heatProof").unwrap() = json::object! {
            "name": "h_heatProof",
            "requires": ["Varia"],
        };
        *game_data
            .helper_json_map
            .get_mut("h_heatResistant")
            .unwrap() = json::object! {
            "name": "h_heatResistant",
            "requires": ["Varia"],
        };
        // Both Varia and Gravity are required to provide full lava protection:
        *game_data.helper_json_map.get_mut("h_lavaProof").unwrap() = json::object! {
            "name": "h_lavaProof",
            "requires": ["Varia", "Gravity"],
        };
        // Gate glitch leniency
        *game_data
            .helper_json_map
            .get_mut("h_BlueGateGlitchLeniency")
            .unwrap() = json::object! {
            "name": "h_BlueGateGlitchLeniency",
            "requires": ["i_BlueGateGlitchLeniency"],
        };
        *game_data
            .helper_json_map
            .get_mut("h_GreenGateGlitchLeniency")
            .unwrap() = json::object! {
            "name": "h_GreenGateGlitchLeniency",
            "requires": ["i_GreenGateGlitchLeniency"],
        };
        *game_data
            .helper_json_map
            .get_mut("h_HeatedBlueGateGlitchLeniency")
            .unwrap() = json::object! {
            "name": "h_BlueGateGlitchLeniency",
            "requires": ["i_HeatedBlueGateGlitchLeniency"],
        };
        *game_data
            .helper_json_map
            .get_mut("h_HeatedGreenGateGlitchLeniency")
            .unwrap() = json::object! {
            "name": "h_GreenGateGlitchLeniency",
            "requires": ["i_HeatedGreenGateGlitchLeniency"],
        };
        // Other:
        *game_data
            .helper_json_map
            .get_mut("h_AllItemsSpawned")
            .unwrap() = json::object! {
            "name": "h_AllItemsSpawned",
            "requires": ["f_AllItemsSpawn"]  // internal flag "f_AllItemsSpawn" gets set at start if QoL option enabled
        };
        *game_data
            .helper_json_map
            .get_mut("h_EverestMorphTunnelExpanded")
            .unwrap() = json::object! {
            "name": "h_EverestMorphTunnelExpanded",
            "requires": []  // This morph tunnel is always expanded, so there are no requirements.
        };
        *game_data
            .helper_json_map
            .get_mut("h_canActivateAcidChozo")
            .unwrap() = json::object! {
            "name": "h_canActivateAcidChozo",
            "requires": [{
                "or": ["SpaceJump", "f_AcidChozoWithoutSpaceJump"]
            }],
        };
        *game_data
            .helper_json_map
            .get_mut("h_ShaktoolVanillaFlag")
            .unwrap() = json::object! {
            "name": "h_ShaktoolVanillaFlag",
            "requires": ["never"],
        };
        *game_data
            .helper_json_map
            .get_mut("h_ShaktoolCameraFix")
            .unwrap() = json::object! {
            "name": "h_ShaktoolCameraFix",
            "requires": [],
        };
        *game_data
            .helper_json_map
            .get_mut("h_KraidCameraFix")
            .unwrap() = json::object! {
            "name": "h_KraidCameraFix",
            "requires": [],
        };
        // Ammo station refill
        *game_data
            .helper_json_map
            .get_mut("h_useMissileRefillStation")
            .unwrap() = json::object! {
            "name": "h_useMissileRefillStation",
            "requires": ["i_ammoRefill"],
        };
        // Wall on right side of Tourian Escape Room 1 does not spawn in the randomizer:
        *game_data
            .helper_json_map
            .get_mut("h_AccessTourianEscape1RightDoor")
            .unwrap() = json::object! {
            "name": "h_AccessTourianEscape1RightDoor",
            "requires": [],
        };

        game_data.load_weapons()?;
        game_data.load_enemies()?;
        game_data.load_regions()?;
        game_data.filter_links();
        game_data.load_connections()?;
        game_data.populate_target_locations()?;
        game_data.extract_all_tech_dependencies()?;
        game_data.extract_all_strat_dependencies()?;

        game_data
            .load_room_geometry(room_geometry_path)
            .context("Unable to load room geometry")?;
        game_data.load_escape_timings(escape_timings_path)?;
        game_data.load_start_locations(start_locations_path)?;
        game_data.load_hub_locations(hub_locations_path)?;
        game_data.area_names = vec![
            "Crateria",
            "Brinstar",
            "Norfair",
            "Wrecked Ship",
            "Maridia",
            "Tourian",
        ]
        .into_iter()
        .map(|x| x.to_owned())
        .collect();
        game_data.area_map_ptrs = vec![
            0x1A9000, // Crateria
            0x1A8000, // Brinstar
            0x1AA000, // Norfair
            0x1AB000, // Wrecked ship
            0x1AC000, // Maridia
            0x1AD000, // Tourian
        ];
        // game_data.load_palette(palette_path)?;
        game_data.load_themes(palette_theme_path)?;
        game_data.load_title_screens(title_screen_path)?;

        Ok(game_data)
    }
}
