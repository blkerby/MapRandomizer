pub mod smart_xml;

use anyhow::{bail, ensure, Context, Result};
// use log::info;
use crate::customize::room_palettes::decode_palette;
use crate::patch::title::read_image;
use crate::randomize::DoorType;
use hashbrown::{HashMap, HashSet};
use json::{self, JsonValue};
use log::{error, info};
use num_enum::TryFromPrimitive;
use serde::Serialize;
use serde_derive::Deserialize;
use std::borrow::ToOwned;
use std::fs::File;
use std::hash::Hash;
use std::ops::IndexMut;
use std::path::{Path, PathBuf};
use strum::VariantNames;
use strum_macros::{EnumString, EnumVariantNames};

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Requirement {
    Free,
    Never,
    Tech(TechId),
    Strat(StratId),
    Item(ItemId),
    Flag(FlagId),
    NotFlag(FlagId),
    Objective(usize),
    Walljump,
    ShineCharge {
        used_tiles: Float,
        heated: bool,
    },
    ShineChargeFrames(i32),
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
    EnergyRefill(i32),
    RegularEnergyRefill(i32),
    ReserveRefill(i32),
    MissileRefill(i32),
    SuperRefill(i32),
    PowerBombRefill(i32),
    AmmoStationRefill,
    AmmoStationRefillAll,
    LowerNorfairElevatorDownFrames,
    LowerNorfairElevatorUpFrames,
    MainHallElevatorFrames,
    ShinesparksCostEnergy,
    SupersDoubleDamageMotherBrain,
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
    MotherBrain2Fight {
        can_be_very_patient_tech_id: usize,
    },
    DoorType {
        room_id: RoomId,
        node_id: NodeId,
        door_type: DoorType,
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
            Requirement::ShineCharge {
                used_tiles: Float::new(tiles),
                heated,
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Link {
    pub from_vertex_id: VertexId,
    pub to_vertex_id: VertexId,
    pub requirement: Requirement,
    pub notable_strat_name: Option<String>,
    pub strat_name: String,
    pub strat_notes: Vec<String>,
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EnemyVulnerabilities {
    pub hp: i32,
    pub non_ammo_vulnerabilities: WeaponMask,
    pub missile_damage: i32,
    pub super_damage: i32,
    pub power_bomb_damage: i32,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoorOrientation {
    Left,
    Right,
    Up,
    Down,
}

fn parse_door_orientation(door_orientation: &str) -> Result<DoorOrientation> {
    Ok(match door_orientation {
        "left" => DoorOrientation::Left,
        "right" => DoorOrientation::Right,
        "top" => DoorOrientation::Up,
        "bottom" => DoorOrientation::Down,
        _ => bail!(format!("Unrecognized door position '{}'", door_orientation)),
    })
}

#[derive(Clone, Debug)]
pub struct GModeRegainMobility {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SparkPosition {
    Top,
    Bottom,
    Any,
}

// Hashable wrapper for f32 based on its bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Float {
    data: u32,
}

impl Float {
    pub fn new(x: f32) -> Self {
        Float { data: x.to_bits() }
    }

    pub fn get(&self) -> f32 {
        f32::from_bits(self.data)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TemporaryBlueDirection {
    Left,
    Right,
    Any,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlueOption {
    Yes,
    No,
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExitCondition {
    LeaveNormally {},
    LeaveWithRunway {
        effective_length: Float,
        heated: bool,
        physics: Option<Physics>,
        from_exit_node: bool,
    },
    LeaveShinecharged {
        physics: Option<Physics>,
    },
    LeaveWithTemporaryBlue {
        direction: TemporaryBlueDirection,
    },
    LeaveWithSpark {
        position: SparkPosition,
    },
    LeaveSpinning {
        remote_runway_length: Float,
        blue: BlueOption,
        heated: bool,
    },
    LeaveWithMockball {
        remote_runway_length: Float,
        landing_runway_length: Float,
        blue: BlueOption,
    },
    LeaveWithSpringBallBounce {
        remote_runway_length: Float,
        landing_runway_length: Float,
        blue: BlueOption,
        movement_type: BounceMovementType
    },
    LeaveSpaceJumping {
        remote_runway_length: Float,
        blue: BlueOption,
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
        height: Float,
        heated: bool,
    },
    LeaveWithPlatformBelow {
        height: Float,
        left_position: Float,
        right_position: Float,
    },
    LeaveWithGrappleTeleport {
        block_positions: Vec<(u16, u16)>,
    },
}

fn parse_spark_position(s: Option<&str>) -> Result<SparkPosition> {
    Ok(match s {
        Some("top") => SparkPosition::Top,
        Some("bottom") => SparkPosition::Bottom,
        None => SparkPosition::Any,
        _ => bail!("Unrecognized spark position: {}", s.unwrap()),
    })
}

fn parse_temporary_blue_direction(s: Option<&str>) -> Result<TemporaryBlueDirection> {
    Ok(match s {
        Some("left") => TemporaryBlueDirection::Left,
        Some("right") => TemporaryBlueDirection::Right,
        Some("any") => TemporaryBlueDirection::Any,
        None => TemporaryBlueDirection::Any,
        _ => bail!("Unrecognized temporary blue direction: {}", s.unwrap()),
    })
}

fn parse_blue_option(s: Option<&str>) -> Result<BlueOption> {
    Ok(match s {
        Some("yes") => BlueOption::Yes,
        Some("no") => BlueOption::No,
        Some("any") => BlueOption::Any,
        None => BlueOption::Any,
        _ => bail!("Unrecognized blue option: {}", s.unwrap()),
    })
}

fn parse_bounce_movement_type(s: &str) -> Result<BounceMovementType> {
    Ok(match s {
        "controlled" => BounceMovementType::Controlled,
        "uncontrolled" => BounceMovementType::Uncontrolled,
        "any" => BounceMovementType::Any,
        _ => bail!("Unrecognized bounce movementType: {}", s),
    })
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GModeMode {
    Direct,
    Indirect,
    Any,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GModeMobility {
    Mobile,
    Immobile,
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ToiletCondition {
    No,
    Yes,
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EntranceCondition {
    pub through_toilet: ToiletCondition,
    pub main: MainEntranceCondition,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BounceMovementType {
    Controlled,
    Uncontrolled,
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MainEntranceCondition {
    ComeInNormally {},
    ComeInRunning {
        speed_booster: Option<bool>,
        min_tiles: Float,
        max_tiles: Float,
    },
    ComeInJumping {
        speed_booster: Option<bool>,
        min_tiles: Float,
        max_tiles: Float,
    },
    ComeInSpaceJumping {
        speed_booster: Option<bool>,
        min_tiles: Float,
        max_tiles: Float,
    },
    ComeInShinecharging {
        effective_length: Float,
        heated: bool,
    },
    ComeInShinecharged {},
    ComeInShinechargedJumping {},
    ComeInWithSpark {
        position: SparkPosition,
    },
    ComeInSpeedballing {
        effective_runway_length: Float,
    },
    ComeInWithTemporaryBlue {
        direction: TemporaryBlueDirection,
    },
    ComeInBlueSpinning {
        min_tiles: Float,
        unusable_tiles: Float,
    },
    ComeInWithMockball {
        adjacent_min_tiles: Float,
        remote_and_landing_min_tiles: Vec<(Float, Float)>,
    },
    ComeInWithSpringBallBounce {
        adjacent_min_tiles: Float,
        remote_and_landing_min_tiles: Vec<(Float, Float)>,
        movement_type: BounceMovementType,
    },

    ComeInStutterShinecharging {
        min_tiles: Float,
    },
    ComeInWithBombBoost {},
    ComeInWithDoorStuckSetup {
        heated: bool,
        door_orientation: DoorOrientation,
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
        min_height: Float,
    },
    ComeInWithSpaceJumpBelow {},
    ComeInWithPlatformBelow {
        min_height: Float,
        max_height: Float,
        max_left_position: Float,
        min_right_position: Float,
    },
    ComeInWithGrappleTeleport {
        block_positions: Vec<(u16, u16)>,
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
        "Wrecked Ship Main Shaft Partial Covern Ice Clip", // not usable because of canRiskPermanentLossOfAccess
        "Mickey Mouse Crumble Jump IBJ", // only useful with CF clip strat, or if we change item progression rules
        "Green Brinstar Main Shaft Moonfall Spark", // does not seem to be viable with the vanilla door connection
        "Waterway Grapple Teleport Inside Wall",
    ]
    .iter()
    .map(|x| x.to_string())
    .collect()
}

fn get_logical_gray_door_room_ids() -> Vec<RoomId> {
    vec![
        // Pirate rooms:
        12,  // Pit Room
        82,  // Baby Kraid Room
        139, // Metal Pirates Room
        219, // Plasma Room
        // Boss/miniboss rooms:
        84,  // Kraid Room
        193, // Draygon's Room
        142, // Ridley's Room
        150, // Golden Torizo Room
    ]
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum VertexAction {
    #[default]
    Nothing,  // This should never be constructed, just here because we need a default value
    // A MaybeExit vertex is created when a strat has unlockDoors but no exit condition; it has the option of 
    // unlocking the door (potentially with lower requirements than the implicit ones) or not, and then
    // exiting the room or not.
    MaybeExit(ExitCondition, Requirement),
    Exit(ExitCondition),
    Enter(EntranceCondition),
    DoorUnlock(NodeId, VertexId),
    ItemCollect(NodeId),
    FlagSet(FlagId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct VertexKey {
    pub room_id: RoomId,
    pub node_id: NodeId,
    pub obstacle_mask: ObstacleMask,
    pub actions: Vec<VertexAction>,
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
    pub reverse_node_ptr_map: HashMap<NodePtr, (RoomId, NodeId)>,
    pub node_ptr_map: HashMap<(RoomId, NodeId), NodePtr>,
    pub node_door_unlock: HashMap<(RoomId, NodeId), Vec<VertexId>>,
    pub node_entrance_conditions: HashMap<(RoomId, NodeId), Vec<(VertexId, EntranceCondition)>>,
    pub node_exit_conditions: HashMap<(RoomId, NodeId), Vec<(VertexId, ExitCondition)>>,
    pub node_gmode_regain_mobility: HashMap<(RoomId, NodeId), Vec<(Link, GModeRegainMobility)>>,
    pub room_num_obstacles: HashMap<RoomId, usize>,
    pub door_ptr_pair_map: HashMap<DoorPtrPair, (RoomId, NodeId)>,
    pub reverse_door_ptr_pair_map: HashMap<(RoomId, NodeId), DoorPtrPair>,
    pub door_position: HashMap<(RoomId, NodeId), DoorOrientation>,
    pub vertex_isv: IndexedVec<VertexKey>,
    pub item_locations: Vec<(RoomId, NodeId)>,
    pub item_vertex_ids: Vec<Vec<VertexId>>,
    pub flag_ids: Vec<FlagId>,
    pub flag_vertex_ids: Vec<Vec<VertexId>>,
    pub save_locations: Vec<(RoomId, NodeId)>,
    pub links: Vec<Link>,
    pub base_links_data: LinksDataGroup,
    pub room_geometry: Vec<RoomGeometry>,
    pub room_and_door_idxs_by_door_ptr_pair:
        HashMap<DoorPtrPair, (RoomGeometryRoomIdx, RoomGeometryDoorIdx)>,
    pub room_ptr_by_id: HashMap<RoomId, RoomPtr>,
    pub room_id_by_ptr: HashMap<RoomPtr, RoomId>,
    pub raw_room_id_by_ptr: HashMap<RoomPtr, RoomId>, // Does not replace twin room pointer with corresponding main room pointer
    pub room_idx_by_ptr: HashMap<RoomPtr, RoomGeometryRoomIdx>,
    pub room_idx_by_name: HashMap<String, RoomGeometryRoomIdx>,
    pub toilet_room_idx: usize,
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
    pub escape_timings: Vec<EscapeTimingRoom>,
    pub start_locations: Vec<StartLocation>,
    pub hub_locations: Vec<HubLocation>,
    pub heat_run_tech_id: TechId, // Cached since it is used frequently in graph traversal, and to avoid needing to store it in every HeatFrames req.
    pub wall_jump_tech_id: TechId,
    pub manage_reserves_tech_id: TechId,
    pub pause_abuse_tech_id: TechId,
    pub mother_brain_defeated_flag_id: usize,
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

#[derive(Default, Clone)]
struct RequirementContext<'a> {
    room_id: RoomId,
    _from_node_id: NodeId, // Usable for debugging
    to_node_id: NodeId,
    room_heated: bool,
    from_obstacles_bitmask: ObstacleMask,
    obstacles_idx_map: Option<&'a HashMap<String, usize>>,
    unlocks_doors_json: Option<&'a JsonValue>,
    node_implicit_door_unlocks: Option<&'a HashMap<NodeId, bool>>,
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
        self.manage_reserves_tech_id =
            *self.tech_isv.index_by_key.get("canManageReserves").unwrap();
        self.pause_abuse_tech_id = *self.tech_isv.index_by_key.get("canPauseAbuse").unwrap();
        self.mother_brain_defeated_flag_id = self.flag_isv.index_by_key["f_DefeatedMotherBrain"];
        Ok(())
    }

    fn override_can_awaken_zebes_tech_note(full_tech_json: &mut JsonValue) -> Result<()> {
        let tech_category = full_tech_json["techCategories"]
            .members_mut()
            .find(|x| x["name"] == "Meta")
            .unwrap();
        let tech = tech_category["techs"]
            .members_mut()
            .find(|x| x["name"] == "canAwakenZebes")
            .unwrap();
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
        self.flag_isv.add("f_UsedBowlingStatue");
        self.flag_isv.add("f_ClearedPitRoom");
        self.flag_isv.add("f_ClearedBabyKraidRoom");
        self.flag_isv.add("f_ClearedPlasmaRoom");
        self.flag_isv.add("f_ClearedMetalPiratesRoom");

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
    fn get_unlocks_doors_req(
        &mut self,
        node_id: NodeId,
        ctx: &RequirementContext,
    ) -> Result<Requirement> {
        let door_type_methods = vec![
            (
                DoorType::Red,
                vec![
                    (vec!["missiles", "ammo"], Requirement::Missiles(5), Requirement::HeatFrames(50)),
                    (vec!["super", "ammo"], Requirement::Supers(1), Requirement::Free),
                ],
            ),
            (
                DoorType::Green,
                vec![(vec!["super", "ammo"], Requirement::Supers(1), Requirement::Free)],
            ),
            (
                DoorType::Yellow,
                vec![(
                    vec!["powerbomb", "ammo"],
                    Requirement::make_and(vec![
                        Requirement::Item(Item::Morph as ItemId),
                        Requirement::PowerBombs(1),
                    ]),
                    Requirement::HeatFrames(110),
                )],
            ),
            (DoorType::Grey, vec![(vec!["grey"], Requirement::Free, Requirement::Free)]),
        ];

        let room_id = ctx.room_id;
        let to_node_id = ctx.to_node_id;
        let empty_array = json::array![];
        let unlocks_doors_json = ctx.unlocks_doors_json.unwrap_or(&empty_array);

        let mut door_reqs = vec![
            Requirement::DoorUnlocked { room_id, node_id },
            Requirement::DoorType {
                room_id,
                node_id,
                door_type: DoorType::Blue,
            },
        ];
        ensure!(unlocks_doors_json.is_array());
        
        // Disallow using "unlocksDoors" inside its own requirements, to avoid an infinite recursion.
        // TODO: Figure out how to more properly handle "resetRoom" requirements inside of "unlocksDoors".
        let mut ctx1 = ctx.clone();
        ctx1.unlocks_doors_json = None;

        for (door_type, unlock_methods) in &door_type_methods {
            let mut door_type_reqs = vec![];
            for (keys, implicit_req, heat_req) in unlock_methods {
                let mut req: Option<Requirement> = None;
                for &key in keys {
                    for u in unlocks_doors_json.members() {
                        if u["nodeId"].as_usize().unwrap_or(ctx.to_node_id) == node_id {
                            if !u["types"].is_array() {
                                println!("{}", unlocks_doors_json);
                            }
                            ensure!(u["types"].is_array());
                            if u["types"].members().any(|t| t == key) {
                                if req.is_some() {
                                    bail!("Overlapping unlocksDoors for '{}', room_id={}, node_id={}: {:?}", key, room_id, node_id, unlocks_doors_json);
                                }
                                let requires: &[JsonValue] = u["requires"].members().as_slice();
                                let mut req_list = self.parse_requires_list(requires, &ctx1)?;
                                if u["useImplicitRequires"].as_bool().unwrap_or(false) {
                                    req_list.push(implicit_req.clone());
                                }
                                req = Some(Requirement::make_and(req_list));
                            }
                        }
                    }
                }
                if let Some(req) = req {
                    door_type_reqs.push(req);
                } else if door_type != &DoorType::Grey {
                    if ctx.node_implicit_door_unlocks.unwrap()[&node_id] {
                        if ctx.room_heated {
                            door_type_reqs.push(Requirement::make_and(vec![
                                implicit_req.clone(),
                                heat_req.clone(),
                            ]));
                        } else {
                            door_type_reqs.push(implicit_req.clone())
                        }
                    }
                }
            }
            let method_req = Requirement::make_and(vec![
                Requirement::DoorType {
                    room_id,
                    node_id,
                    door_type: *door_type,
                },
                Requirement::make_or(door_type_reqs),
            ]);
            door_reqs.push(method_req);
        }
        Ok(Requirement::make_or(door_reqs))
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
            } else if value == "i_ammoRefillAll" {
                return Ok(Requirement::AmmoStationRefillAll);
            } else if value == "i_SupersDoubleDamageMotherBrain" {
                return Ok(Requirement::SupersDoubleDamageMotherBrain);
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
            } else if value == "i_Objective1Complete" {
                return Ok(Requirement::Objective(0));
            } else if value == "i_Objective2Complete" {
                return Ok(Requirement::Objective(1));
            } else if value == "i_Objective3Complete" {
                return Ok(Requirement::Objective(2));
            } else if value == "i_Objective4Complete" {
                return Ok(Requirement::Objective(3));
            } else if value == "i_LowerNorfairElevatorDownwardFrames" {
                return Ok(Requirement::LowerNorfairElevatorDownFrames);
            } else if value == "i_LowerNorfairElevatorUpwardFrames" {
                return Ok(Requirement::LowerNorfairElevatorUpFrames);
            } else if value == "i_MainHallElevatorFrames" {
                return Ok(Requirement::MainHallElevatorFrames);
            } else if value == "i_ShinesparksCostEnergy" {
                return Ok(Requirement::ShinesparksCostEnergy);
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
                        req_list_and.push(Requirement::MissileRefill(9999));
                    } else if resource_type == "Super" {
                        req_list_and.push(Requirement::SuperRefill(9999));
                    } else if resource_type == "PowerBomb" {
                        req_list_and.push(Requirement::PowerBombRefill(9999));
                    } else if resource_type == "RegularEnergy" {
                        req_list_and.push(Requirement::RegularEnergyRefill(9999));
                    } else if resource_type == "ReserveEnergy" {
                        req_list_and.push(Requirement::ReserveRefill(9999));
                    } else if resource_type == "Energy" {
                        req_list_and.push(Requirement::EnergyRefill(9999));
                    } else {
                        bail!("Unrecognized refill resource type: {}", resource_type);
                    }
                }
                return Ok(Requirement::make_and(req_list_and));
            } else if key == "partialRefill" {
                let resource_type = value["type"].as_str().unwrap();
                let limit = value["limit"].as_i32().unwrap();
                let req = if resource_type == "Missile" {
                    Requirement::MissileRefill(limit)
                } else if resource_type == "Super" {
                    Requirement::SuperRefill(limit)
                } else if resource_type == "PowerBomb" {
                    Requirement::PowerBombRefill(limit)
                } else if resource_type == "RegularEnergy" {
                    Requirement::RegularEnergyRefill(limit)
                } else if resource_type == "ReserveEnergy" {
                    Requirement::ReserveRefill(limit)
                } else if resource_type == "Energy" {
                    Requirement::EnergyRefill(limit)
                } else {
                    bail!(
                        "Unrecognized partialRefill resource type: {}",
                        resource_type
                    );
                };
                return Ok(req);
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
                return Ok(Requirement::make_shinecharge(
                    effective_length,
                    ctx.room_heated,
                ));
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
                } else if enemy_set.contains("Mother Brain 2") {
                    return Ok(Requirement::MotherBrain2Fight {
                        can_be_very_patient_tech_id: self.tech_isv.index_by_key["canBeVeryPatient"],
                    });
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
            } else if key == "resetRoom" {
                if ctx.from_obstacles_bitmask != 0 {
                    return Ok(Requirement::Never);
                }
                let mut node_ids: Vec<NodeId> = Vec::new();
                for from_node in value["nodes"].members() {
                    node_ids.push(from_node.as_usize().unwrap());
                }
                let mut reqs_or: Vec<Requirement> = vec![];
                for node_id in node_ids {
                    reqs_or.push(self.get_unlocks_doors_req(node_id, ctx)?);
                }
                return Ok(Requirement::make_or(reqs_or));
            } else if key == "doorUnlockedAtNode" {
                let node_id = value.as_usize().unwrap();
                // if ctx.unlocks_doors_json.is_some() {
                //     info!("node_id={node_id}, unlocksDoors={}", ctx.unlocks_doors_json.unwrap());
                // } else {
                //     info!("node_id={node_id}, unlocksDoors=None");
                // }
                return self.get_unlocks_doors_req(node_id, ctx);
            } else if key == "itemNotCollectedAtNode" {
                // TODO: implement this
                return Ok(Requirement::Free);
            } else if key == "autoReserveTrigger" {
                return Ok(Requirement::ReserveTrigger {
                    min_reserve_energy: value["minReserveEnergy"].as_i32().unwrap_or(1),
                    max_reserve_energy: value["maxReserveEnergy"].as_i32().unwrap_or(400),
                });
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

    fn override_pit_room(&mut self, room_json: &mut JsonValue) {
        // Add yielded flag "f_ClearedPitRoom" to gray door unlocks:
        for node_json in room_json["nodes"].members_mut() {
            if [1, 2].contains(&node_json["id"].as_i32().unwrap()) {
                node_json["locks"][0]["yields"] = json::array!["f_ZebesAwake", "f_ClearedPitRoom"]
            }
        }
    }

    fn override_baby_kraid_room(&mut self, room_json: &mut JsonValue) {
        // Add yielded flag "f_ClearedBabyKraidRoom" to gray door unlocks:
        for node_json in room_json["nodes"].members_mut() {
            if [1, 2].contains(&node_json["id"].as_i32().unwrap()) {
                node_json["locks"][0]["yields"] =
                    json::array!["f_ZebesAwake", "f_ClearedBabyKraidRoom"]
            }
        }
    }

    fn override_plasma_room(&mut self, room_json: &mut JsonValue) {
        // Add yielded flag "f_ClearedPlasmaRoom" to gray door unlocks:
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 1 {
                node_json["locks"][0]["yields"] =
                    json::array!["f_ZebesAwake", "f_ClearedPlasmaRoom"]
            }
        }
    }

    fn override_metal_pirates_room(&mut self, room_json: &mut JsonValue) {
        // Add yielded flag "f_ClearedMetalPiratesRoom" to gray door unlock:
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 1 {
                node_json["locks"][0]["yields"] =
                    json::array!["f_ZebesAwake", "f_ClearedMetalPiratesRoom"]
            }
        }

        // Add lock on right door:
        let mut found = false;
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 2 {
                found = true;
                node_json["nodeSubType"] = "grey".into();
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
                    "yields": ["f_ZebesAwake", "f_ClearedMetalPiratesRoom"]
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

    fn override_mother_brain_room(&mut self, room_json: &mut JsonValue) {
        // Add a requirement for objectives to be completed in order to cross the barriers
        for x in room_json["strats"].members_mut() {
            if x["link"][0].as_i32().unwrap() == 2 && x["link"][1].as_i32().unwrap() != 2 {
                x["requires"].push("i_Objective1Complete").unwrap();
                x["requires"].push("i_Objective2Complete").unwrap();
                x["requires"].push("i_Objective3Complete").unwrap();
                x["requires"].push("i_Objective4Complete").unwrap();
            }
        }

        // Override the MB2 boss fight requirements
        let mut found = false;
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 4 {
                node_json["locks"][0]["unlockStrats"] = json::array![{
                    "name": "Base",
                    "notable": false,
                    "requires": [
                        {"enemyKill": {"enemies": [["Mother Brain 2"]]}}
                    ]
                }];
                found = true;
            }
        }
        assert!(found);
    }

    fn override_bowling_alley(&mut self, room_json: &mut JsonValue) {
        // Add flag on Bowling Statue node
        let mut found = false;
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 6 {
                found = true;
                node_json["yields"] = json::array!["f_UsedBowlingStatue"];
                node_json["locks"] = json::array![{
                    "name": "Use Statue",
                    "lockType": "gameFlag",
                    "unlockStrats": [{
                        "name": "Base",
                        "notable": false,
                        "requires": []
                    }]
                }];
            }
        }
        assert!(found);
    }

    fn override_metroid_room_1(&mut self, room_json: &mut JsonValue) {
        // Remove the "f_ZebesAwake" flag from the gray door unlock, since in the randomizer there's not actually a gray door here:
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 1 {
                assert!(node_json["locks"][0]["yields"].is_array());
                node_json["locks"][0]["yields"] = json::array!["f_KilledMetroidRoom1"]
            }
        }
    }

    fn override_metroid_room_2(&mut self, room_json: &mut JsonValue) {
        // Remove the "f_ZebesAwake" flag from the gray door unlock, since in the randomizer there's not actually a gray door here:
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 2 {
                assert!(node_json["locks"][0]["yields"].is_array());
                node_json["locks"][0]["yields"] = json::array!["f_KilledMetroidRoom2"]
            }
        }
    }

    fn override_metroid_room_3(&mut self, room_json: &mut JsonValue) {
        // Remove the "f_ZebesAwake" flag from the gray door unlock, since in the randomizer there's not actually a gray door here:
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 2 {
                assert!(node_json["locks"][0]["yields"].is_array());
                node_json["locks"][0]["yields"] = json::array!["f_KilledMetroidRoom3"]
            }
        }
    }

    fn override_metroid_room_4(&mut self, room_json: &mut JsonValue) {
        // Remove the "f_ZebesAwake" flag from the gray door unlock, since in the randomizer there's not actually a gray door here:
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 2 {
                assert!(node_json["locks"][0]["yields"].is_array());
                node_json["locks"][0]["yields"] = json::array!["f_KilledMetroidRoom4"]
            }
        }
    }

    fn get_default_unlocks_door(&self, room_json: &JsonValue, node_id: usize, to_node_id: usize) -> Result<JsonValue> {
        let mut unlocks_door = if self.get_room_heated(room_json, node_id)? {
            json::array![
                {"types": ["missiles"], "requires": [{"heatFrames": 50}]},
                {"types": ["super"], "requires": []},
                {"types": ["powerbomb"], "requires": [{"heatFrames": 110}]}
            ]
        } else {
            json::array![
                {"types": ["ammo"], "requires": []},
            ]
        };
        if node_id != to_node_id {
            for u in unlocks_door.members_mut() {
                u["nodeId"] = node_id.into();
            }    
        }
        Ok(unlocks_door)
    }

    fn preprocess_room(&mut self, room_json: &JsonValue) -> Result<JsonValue> {
        // We apply some changes to the sm-json-data specific to Map Rando.

        let ignored_notable_strats = get_ignored_notable_strats();

        let mut new_room_json = room_json.clone();
        ensure!(room_json["nodes"].is_array());
        let mut extra_strats: Vec<JsonValue> = Vec::new();
        let room_id = room_json["id"].as_usize().unwrap();

        if room_json["name"].as_str().unwrap() == "Upper Tourian Save Room" {
            new_room_json["name"] = JsonValue::String("Tourian Map Room".to_string());
        }

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

        match room_id {
            222 => self.override_shaktool_room(&mut new_room_json),
            38 => self.override_morph_ball_room(&mut new_room_json),
            225 => self.override_tourian_save_room(&mut new_room_json),
            238 => self.override_mother_brain_room(&mut new_room_json),
            161 => self.override_bowling_alley(&mut new_room_json),
            12 => self.override_pit_room(&mut new_room_json),
            82 => self.override_baby_kraid_room(&mut new_room_json),
            139 => self.override_metal_pirates_room(&mut new_room_json),
            219 => self.override_plasma_room(&mut new_room_json),
            226 => self.override_metroid_room_1(&mut new_room_json),
            227 => self.override_metroid_room_2(&mut new_room_json),
            228 => self.override_metroid_room_3(&mut new_room_json),
            229 => self.override_metroid_room_4(&mut new_room_json),
            _ => {}
        }

        let mut obstacle_flag: Option<String> = None;
        let logical_gray_door_room_ids = get_logical_gray_door_room_ids();

        for node_json in new_room_json["nodes"].members_mut() {
            let node_id = node_json["id"].as_usize().unwrap();
            let node_type = node_json["nodeType"].as_str().unwrap();
            if ["door", "exit"].contains(&node_type) && node_json["useImplicitDoorUnlocks"].as_bool() != Some(false) {
                extra_strats.push(json::object!{
                    "link": [node_id, node_id],
                    "name": "Base (Unlock Door)",
                    "requires": [],
                    "unlocksDoors": self.get_default_unlocks_door(room_json, node_id, node_id)?,
                });    
            }
            if ["door", "entrance"].contains(&node_type) {
                let spawn_node_id = node_json["spawnAt"].as_usize().unwrap_or(node_id);
                extra_strats.push(json::object!{
                    "link": [node_id, spawn_node_id],
                    "name": "Base (Come In Normally)",
                    "entranceCondition": {
                        "comeInNormally": {}
                    },
                    "requires": []
                });
            }

            if node_type == "item" && !node_json.has_key("locks") {
                node_json["locks"] = json::array![
                    {
                      "name": "Dummy Item Lock",
                      "lockType": "gameFlag",
                      "unlockStrats": [
                        {
                          "name": "Base",
                          "notable": false,
                          "requires": [],
                        }
                      ]
                    }
                ];
            }

            // println!("yields: {:?}", node_json["locks"][0]["yields[0].as_str());
            if node_json["name"] == "Kraid" {
                println!("{}", node_json);
            }

            if node_json.has_key("locks")
                && (!["door", "entrance"].contains(&node_json["nodeType"].as_str().unwrap())
                    || logical_gray_door_room_ids.contains(&room_id))
            {
                ensure!(node_json["locks"].len() == 1);
                let lock = node_json["locks"][0].clone();
                let mut unlock_strats = lock["unlockStrats"].clone();
                let yields = if lock["yields"] != JsonValue::Null {
                    lock["yields"].clone()
                } else {
                    node_json["yields"].clone()
                };

                if node_json["name"] == "Kraid" {
                    println!("yields: {}", yields);
                }
    
                if yields != JsonValue::Null
                    && obstacle_flags.contains(&yields[0].as_str().unwrap())
                {
                    obstacle_flag = Some(yields[0].as_str().unwrap().to_owned());
                }
                if (room_id, node_id) == (158, 2) {
                    // Override Phantoon fight requirement
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

                for unlock_strat in unlock_strats.members() {
                    let mut new_strat = unlock_strat.clone();
                    new_strat["link"] = json::array![node_id, node_id];

                    match (
                        node_json["nodeType"].as_str().unwrap(),
                        node_json["nodeSubType"].as_str().unwrap(),
                    ) {
                        ("door", node_sub_type) => {
                            if node_sub_type == "grey" && get_logical_gray_door_room_ids().contains(&room_id) {
                                new_strat["unlocksDoors"] = json::array![
                                    {"types": ["grey"], "requires": []}
                                ];
                            }
                        }
                        ("event", "flag" | "boss") | ("junction", _) => {
                            new_strat["setsFlags"] = yields.clone();
                            if yields != JsonValue::Null && obstacle_flags.contains(&yields[0].as_str().unwrap()) {
                                new_strat["clearsObstacles"] = json::array![yields[0].as_str().unwrap()];
                            }
                        }
                        ("item", _) => {
                            new_strat["collectsItems"] = json::array![node_id];
                        }
                        ("utility", _) => {
                            continue;
                        }
                        (node_type, node_subtype) => {
                            panic!(
                                "Unexpected node type/subtype for lock: {} {}",
                                node_type, node_subtype
                            );
                        }
                    }
                    extra_strats.push(new_strat);
                }
            }
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
                if new_strat.has_key("unlocksDoors") {
                    ensure!(new_strat["unlocksDoors"].is_array());
                    for unlock in new_strat["unlocksDoors"].members_mut() {
                        for req in unlock["requires"].members_mut() {
                            if req == &json_obstacle_flag_name {
                                *req = json::object! {
                                    "obstaclesCleared": [obstacle_flag_name.to_string()]
                                };
                                found = true;
                            }    
                        }
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
            let strat_name = strat_json["name"].as_str().unwrap().to_string();
            let from_node_id = strat_json["link"][0].as_usize().unwrap();
            let to_node_id = strat_json["link"][1].as_usize().unwrap();

            // TODO: fix this:
            // if from_node_id == to_node_id && strat_json.has_key("exitCondition") && !strat_json.has_key("unlocksDoors") {
            //     strat_json["unlocksDoors"] = self.get_default_unlocks_door(room_json, to_node_id, to_node_id)?;
            // }

            if ignored_notable_strats.contains(&strat_name) {
                if strat_json["notable"].as_bool() == Some(true) {
                    self.ignored_notable_strats.insert(strat_name.to_string());
                }
                strat_json["notable"] = JsonValue::Boolean(false);
            }

            if let Some(reusable_name) = strat_json["reusableRoomwideNotable"].as_str() {
                if ignored_notable_strats.contains(reusable_name) {
                    if strat_json["notable"].as_bool() == Some(true) {
                        self.ignored_notable_strats
                            .insert(reusable_name.to_string());
                    }
                    strat_json["notable"] = JsonValue::Boolean(false);
                }
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
        self.links.iter()
    }

    fn parse_exit_condition(
        &self,
        exit_json: &JsonValue,
        strat_json: &JsonValue,
        heated: bool,
        physics: Option<Physics>,
    ) -> Result<(ExitCondition, Requirement)> {
        ensure!(exit_json.is_object());
        ensure!(exit_json.len() == 1);
        let (key, value) = exit_json.entries().next().unwrap();
        ensure!(value.is_object());
        let from_node_id = strat_json["link"][0].as_usize().unwrap();
        let to_node_id = strat_json["link"][1].as_usize().unwrap();
        let mut req = Requirement::Free;
        let exit_condition = match key {
            "leaveNormally" => ExitCondition::LeaveNormally {},
            "leaveWithRunway" => {
                let runway_geometry = parse_runway_geometry(value)?;
                let runway_effective_length = compute_runway_effective_length(&runway_geometry);
                ExitCondition::LeaveWithRunway {
                    effective_length: Float::new(runway_effective_length),
                    heated,
                    physics,
                    from_exit_node: from_node_id == to_node_id,
                }
            }
            "leaveShinecharged" => {
                if let Some(frames_remaining) = value["framesRemaining"].as_i32() {
                    req = Requirement::ShineChargeFrames(180 - frames_remaining);
                }
                ExitCondition::LeaveShinecharged {
                    physics,
                }
            },
            "leaveWithTemporaryBlue" => ExitCondition::LeaveWithTemporaryBlue {
                direction: parse_temporary_blue_direction(value["direction"].as_str())?,
            },
            "leaveWithSpark" => ExitCondition::LeaveWithSpark {
                position: parse_spark_position(value["position"].as_str())?,
            },
            "leaveSpinning" => {
                let remote_runway_geometry = parse_runway_geometry(&value["remoteRunway"])?;
                let remote_runway_effective_length = compute_runway_effective_length(&remote_runway_geometry);
                ExitCondition::LeaveSpinning { 
                    remote_runway_length: Float::new(remote_runway_effective_length),
                    blue: parse_blue_option(value["blue"].as_str())?,
                    heated,
                }
            },
            "leaveWithMockball" => {
                let remote_runway_geometry = parse_runway_geometry(&value["remoteRunway"])?;
                let remote_runway_effective_length = compute_runway_effective_length(&remote_runway_geometry);
                let landing_runway_geometry = parse_runway_geometry(&value["landingRunway"])?;
                let landing_runway_effective_length = compute_runway_effective_length(&landing_runway_geometry);
                ExitCondition::LeaveWithMockball { 
                    remote_runway_length: Float::new(remote_runway_effective_length),
                    landing_runway_length: Float::new(landing_runway_effective_length),
                    blue: parse_blue_option(value["blue"].as_str())?,
                }
            },
            "leaveWithSpringBallBounce" => {
                let remote_runway_geometry = parse_runway_geometry(&value["remoteRunway"])?;
                let remote_runway_effective_length = compute_runway_effective_length(&remote_runway_geometry);
                let landing_runway_geometry = parse_runway_geometry(&value["landingRunway"])?;
                let landing_runway_effective_length = compute_runway_effective_length(&landing_runway_geometry);
                ExitCondition::LeaveWithSpringBallBounce { 
                    remote_runway_length: Float::new(remote_runway_effective_length),
                    landing_runway_length: Float::new(landing_runway_effective_length),
                    blue: parse_blue_option(value["blue"].as_str())?,
                    movement_type: parse_bounce_movement_type(value["movementType"].as_str().unwrap())?,
                }
            },
            "leaveSpaceJumping" => {
                let remote_runway_geometry = parse_runway_geometry(&value["remoteRunway"])?;
                let remote_runway_effective_length = compute_runway_effective_length(&remote_runway_geometry);
                ExitCondition::LeaveSpaceJumping { 
                    remote_runway_length: Float::new(remote_runway_effective_length),
                    blue: parse_blue_option(value["blue"].as_str())?,
                }
            },
            "leaveWithGModeSetup" => ExitCondition::LeaveWithGModeSetup {
                knockback: value["knockback"].as_bool().unwrap_or(true),
            },
            "leaveWithGMode" => ExitCondition::LeaveWithGMode {
                morphed: value["morphed"]
                    .as_bool()
                    .context("Expecting boolean 'morphed'")?,
            },
            "leaveWithStoredFallSpeed" => ExitCondition::LeaveWithStoredFallSpeed {
                fall_speed_in_tiles: value["fallSpeedInTiles"]
                    .as_i32()
                    .context("Expecting integer 'fallSpeedInTiles")?,
            },
            "leaveWithDoorFrameBelow" => ExitCondition::LeaveWithDoorFrameBelow {
                height: Float::new(value["height"]
                    .as_f32()
                    .context("Expecting number 'height'")?),
                heated,
            },
            "leaveWithPlatformBelow" => ExitCondition::LeaveWithPlatformBelow {
                height: Float::new(value["height"]
                    .as_f32()
                    .context("Expecting number 'height'")?),
                left_position: Float::new(value["leftPosition"]
                    .as_f32()
                    .context("Expecting number 'leftPosition'")?),
                right_position: Float::new(value["rightPosition"]
                    .as_f32()
                    .context("Expecting number 'rightPosition'")?),
            },
            "leaveWithGrappleTeleport" => ExitCondition::LeaveWithGrappleTeleport {
                block_positions: value["blockPositions"]
                    .members()
                    .map(|x| (x[0].as_u16().unwrap(), x[1].as_u16().unwrap()))
                    .collect(),
            },
            _ => {
                bail!(format!("Unrecognized exit condition: {}", key));
            }
        };
        Ok((exit_condition, req))
    }
    
    fn parse_entrance_condition(&self, entrance_json: &JsonValue, room_id: RoomId, from_node_id: NodeId, heated: bool) -> Result<(EntranceCondition, Requirement)> {
        ensure!(entrance_json.is_object());
        let through_toilet = if entrance_json.has_key("comesThroughToilet") {
            ensure!(entrance_json.len() == 2);
            match entrance_json["comesThroughToilet"].as_str().unwrap() {
                "no" => ToiletCondition::No,
                "yes" => ToiletCondition::Yes,
                "any" => ToiletCondition::Any,
                _ => panic!(
                    "Unexpected comesThroughToilet value: {}",
                    entrance_json["comesThroughToilet"].as_str().unwrap()
                ),
            }
        } else {
            ensure!(entrance_json.len() == 1);
            ToiletCondition::No
        };
        let (key, value) = entrance_json.entries().next().unwrap();
        ensure!(value.is_object());
        let mut req = Requirement::Free;
        let main = match key {
            "comeInNormally" => MainEntranceCondition::ComeInNormally {},
            "comeInRunning" => MainEntranceCondition::ComeInRunning {
                speed_booster: value["speedBooster"].as_bool(),
                min_tiles: Float::new(value["minTiles"]
                    .as_f32()
                    .context("Expecting number 'minTiles'")?),
                max_tiles: Float::new(value["maxTiles"].as_f32().unwrap_or(255.0)),
            },
            "comeInJumping" => MainEntranceCondition::ComeInJumping {
                speed_booster: value["speedBooster"].as_bool(),
                min_tiles: Float::new(value["minTiles"]
                    .as_f32()
                    .context("Expecting number 'minTiles'")?),
                max_tiles: Float::new(value["maxTiles"].as_f32().unwrap_or(255.0)),
            },
            "comeInSpaceJumping" => MainEntranceCondition::ComeInSpaceJumping {
                speed_booster: value["speedBooster"].as_bool(),
                min_tiles: Float::new(value["minTiles"]
                    .as_f32()
                    .context("Expecting number 'minTiles'")?),
                max_tiles: Float::new(value["maxTiles"].as_f32().unwrap_or(255.0)),
            },
            "comeInShinecharging" => {
                let runway_geometry = parse_runway_geometry(value)?;
                // Subtract 0.25 tiles since the door transition skips over approximately that much distance beyond the door shell tile,
                // Subtract another 1 tile for leniency since taps are harder to time across a door transition:
                let runway_effective_length =
                    (compute_runway_effective_length(&runway_geometry) - 1.25).max(0.0);
                MainEntranceCondition::ComeInShinecharging {
                    effective_length: Float::new(runway_effective_length),
                    heated,
                }
            }
            "comeInShinecharged" => {
                let frames_required = value["framesRequired"].as_i32()
                    .context("Expecting integer 'framesRequired'")?;
                req = Requirement::ShineChargeFrames(frames_required);
                MainEntranceCondition::ComeInShinecharged { }
            },
            "comeInShinechargedJumping" => {
                let frames_required = value["framesRequired"].as_i32()
                    .context("Expecting integer 'framesRequired'")?;
                req = Requirement::ShineChargeFrames(frames_required);
                MainEntranceCondition::ComeInShinechargedJumping { }
            },
            "comeInWithSpark" => MainEntranceCondition::ComeInWithSpark {
                position: parse_spark_position(value["position"].as_str())?,
            },
            "comeInStutterShinecharging" => MainEntranceCondition::ComeInStutterShinecharging {
                min_tiles: Float::new(value["minTiles"]
                    .as_f32()
                    .context("Expecting number 'minTiles'")?),
            },
            "comeInWithBombBoost" => MainEntranceCondition::ComeInWithBombBoost {},
            "comeInWithDoorStuckSetup" => {
                let node_json = &self.node_json_map[&(room_id, from_node_id)];
                let door_orientation = parse_door_orientation(node_json["doorOrientation"].as_str().unwrap())?;
                MainEntranceCondition::ComeInWithDoorStuckSetup { heated, door_orientation }
            }
            "comeInSpeedballing" => {
                let runway_geometry = parse_runway_geometry(&value["runway"])?;
                // Subtract 0.25 tiles since the door transition skips over approximately that much distance beyond the door shell tile,
                // Subtract another 1 tile for leniency since taps and/or speedball are harder to time across a door transition:
                let effective_runway_length =
                    (compute_runway_effective_length(&runway_geometry) - 1.25).max(0.0);
                MainEntranceCondition::ComeInSpeedballing {
                    effective_runway_length: Float::new(effective_runway_length),
                }
            }
            "comeInWithTemporaryBlue" => MainEntranceCondition::ComeInWithTemporaryBlue {
                direction: parse_temporary_blue_direction(value["direction"].as_str())?,
            },
            "comeInBlueSpinning" => {
                MainEntranceCondition::ComeInBlueSpinning { 
                    min_tiles: Float::new(value["minTiles"]
                        .as_f32()
                        .unwrap_or(0.0)),
                    unusable_tiles: Float::new(value["unusableTiles"]
                        .as_f32()
                        .unwrap_or(0.0)),
                }
            },
            "comeInWithMockball" => {
                MainEntranceCondition::ComeInWithMockball { 
                    adjacent_min_tiles: Float::new(value["adjacentMinTiles"]
                        .as_f32()
                        .unwrap_or(255.0)),
                    remote_and_landing_min_tiles: value["remoteAndLandingMinTiles"]
                        .members()
                        .map(|x| (Float::new(x[0].as_f32().unwrap()), Float::new(x[1].as_f32().unwrap())))
                        .collect(),
                }
            },
            "comeInWithSpringBallBounce" => {
                MainEntranceCondition::ComeInWithSpringBallBounce { 
                    adjacent_min_tiles: Float::new(value["adjacentMinTiles"]
                        .as_f32()
                        .unwrap_or(255.0)),
                    remote_and_landing_min_tiles: value["remoteAndLandingMinTiles"]
                        .members()
                        .map(|x| (Float::new(x[0].as_f32().unwrap()), Float::new(x[1].as_f32().unwrap())))
                        .collect(),
                    movement_type: parse_bounce_movement_type(value["movementType"].as_str().unwrap())?,
                }
            },
            "comeInWithRMode" => MainEntranceCondition::ComeInWithRMode {},
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
                MainEntranceCondition::ComeInWithGMode {
                    mode,
                    morphed,
                    mobility,
                }
            }
            "comeInWithStoredFallSpeed" => MainEntranceCondition::ComeInWithStoredFallSpeed {
                fall_speed_in_tiles: value["fallSpeedInTiles"]
                    .as_i32()
                    .context("Expecting integer 'fallSpeedInTiles")?,
            },
            "comeInWithWallJumpBelow" => MainEntranceCondition::ComeInWithWallJumpBelow {
                min_height: Float::new(value["minHeight"]
                    .as_f32()
                    .context("Expecting number 'minHeight'")?),
            },
            "comeInWithSpaceJumpBelow" => MainEntranceCondition::ComeInWithSpaceJumpBelow {},
            "comeInWithPlatformBelow" => MainEntranceCondition::ComeInWithPlatformBelow {
                min_height: Float::new(value["minHeight"].as_f32().unwrap_or(0.0)),
                max_height: Float::new(value["maxHeight"].as_f32().unwrap_or(f32::INFINITY)),
                max_left_position: Float::new(value["maxLeftPosition"].as_f32().unwrap_or(f32::INFINITY)),
                min_right_position: Float::new(value["minRightPosition"]
                    .as_f32()
                    .unwrap_or(f32::NEG_INFINITY)),
            },
            "comeInWithGrappleTeleport" => MainEntranceCondition::ComeInWithGrappleTeleport {
                block_positions: value["blockPositions"]
                    .members()
                    .map(|x| (x[0].as_u16().unwrap(), x[1].as_u16().unwrap()))
                    .collect(),
            },
            _ => {
                bail!(format!("Unrecognized entrance condition: {}", key));
            }
        };
        Ok((EntranceCondition {
            through_toilet,
            main,
        }, req))
    }
    
    fn process_room(&mut self, room_json: &JsonValue) -> Result<()> {
        let room_id = room_json["id"].as_usize().unwrap();
        self.room_json_map.insert(room_id, room_json.clone());

        let mut room_ptr =
            parse_int::parse::<usize>(room_json["roomAddress"].as_str().unwrap()).unwrap();
        self.raw_room_id_by_ptr.insert(room_ptr, room_id);
        if room_ptr == 0x7D69A {
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
                    self.reverse_node_ptr_map
                        .insert(node_ptr, (room_id, node_id));
                }
            }
        }
        let mut node_implicit_door_unlocks: HashMap<NodeId, bool> = HashMap::new();
        for node_json in room_json["nodes"].members() {
            let node_id = node_json["id"].as_usize().unwrap();

            node_implicit_door_unlocks.insert(node_id, node_json["useImplicitDoorUnlocks"].as_bool().unwrap_or(true));

            // Implicit leaveWithGMode:
            if !node_json.has_key("spawnAt") && node_json["nodeType"].as_str().unwrap() == "door" {
                for morphed in [false, true] {
                    let from_vertex_id = self.vertex_isv.add(&VertexKey {
                        room_id,
                        node_id,
                        obstacle_mask: 0,
                        actions: vec![VertexAction::Enter(EntranceCondition {
                            through_toilet: ToiletCondition::No,
                            main: MainEntranceCondition::ComeInWithGMode {
                                mode: GModeMode::Direct,
                                morphed,
                                mobility: GModeMobility::Any,
                            },
                        })],
                    });
                    let to_vertex_id = self.vertex_isv.add(&VertexKey {
                        room_id,
                        node_id,
                        obstacle_mask: 0,
                        actions: vec![VertexAction::Exit(ExitCondition::LeaveWithGMode { morphed })],
                    });
                    let link = Link {
                        from_vertex_id,
                        to_vertex_id,
                        requirement: Requirement::Free,
                        notable_strat_name: None,
                        strat_name: "G-Mode Go Back Through Door".to_string(),
                        strat_notes: vec![],
                    };
                    self.links.push(link);
                }
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
            let to_node_json = self.node_json_map[&(room_id, to_node_id)].clone();

            let to_heated = self.get_room_heated(room_json, to_node_id)?;
            let physics_res = self.get_node_physics(&self.node_json_map[&(room_id, to_node_id)]);
            let physics: Option<Physics> = if let Ok(physics_str) = &physics_res {
                Some(parse_physics(&physics_str)?)
            } else {
                None
            };

            let (entrance_condition, entrance_req) =
                if strat_json.has_key("entranceCondition") {
                    ensure!(strat_json["entranceCondition"].is_object());
                    let (e, r) = self.parse_entrance_condition(
                        &strat_json["entranceCondition"],
                        room_id,
                        from_node_id,
                        from_heated,
                    )?;
                    (Some(e), Some(r))
                } else {
                    (None, None)
                };
            let (exit_condition, exit_req) = if strat_json.has_key("exitCondition") {
                ensure!(strat_json["exitCondition"].is_object());
                let (e, r) = self.parse_exit_condition(
                    &strat_json["exitCondition"],
                    strat_json,
                    to_heated,
                    physics,
                )?;
                (Some(e), Some(r))
            } else {
                (None, None)
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
                    to_node_id: to_node_id,
                    room_heated: from_heated || to_heated,
                    from_obstacles_bitmask,
                    obstacles_idx_map: Some(&obstacles_idx_map),
                    unlocks_doors_json: if strat_json.has_key("unlocksDoors") {
                        Some(&strat_json["unlocksDoors"])
                    } else {
                        None
                    },
                    node_implicit_door_unlocks: Some(&node_implicit_door_unlocks),
                };
                let mut requires_vec = vec![];
                if let Some(r) = &entrance_req {
                    requires_vec.push(r.clone());
                }
                requires_vec.extend(self.parse_requires_list(&requires_json, &ctx)?);
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

                let bypasses_door_shell =
                    strat_json["bypassesDoorShell"].as_bool().unwrap_or(false);
                if bypasses_door_shell {
                    requires_vec.push(Requirement::Tech(
                        self.tech_isv.index_by_key["canSkipDoorLock"],
                    ));
                }

                let mut from_actions: Vec<VertexAction> = vec![];
                let mut to_actions: Vec<VertexAction> = vec![];

                if let Some(e) = &entrance_condition {
                    from_actions.push(VertexAction::Enter(e.clone()));
                }

                if strat_json.has_key("setsFlags") {
                    for flag_json in strat_json["setsFlags"].members() {
                        let flag = flag_json.as_str().unwrap();
                        let flag_id = self.flag_isv.index_by_key[flag];
                        to_actions.push(VertexAction::FlagSet(flag_id));
                    }
                }

                if strat_json.has_key("collectsItems") {
                    for item_json in strat_json["collectsItems"].members() {
                        let item_node_id = item_json.as_usize().unwrap();
                        to_actions.push(VertexAction::ItemCollect(item_node_id));
                    }                    
                }

                let mut maybe_exit_req: Option<Requirement> = None;
                if let Some(e) = &exit_condition {
                    to_actions.push(VertexAction::Exit(e.clone()));
                    requires_vec.push(exit_req.clone().unwrap());
                } else if ["door", "exit"].contains(&to_node_json["nodeType"].as_str().unwrap()) && strat_json.has_key("unlocksDoors") {
                    if let Ok(req) = self.get_unlocks_doors_req(to_node_id, &ctx) {
                        maybe_exit_req = Some(req);
                        to_actions.push(VertexAction::MaybeExit(ExitCondition::LeaveNormally {}, maybe_exit_req.clone().unwrap()));
                    }
                }

                if !bypasses_door_shell && exit_condition.is_some() {
                    let unlock_to_door_req = self.get_unlocks_doors_req(to_node_id, &ctx)?;
                    requires_vec.push(unlock_to_door_req);
                }

                let requirement = Requirement::make_and(requires_vec);
                if let Requirement::Never = requirement {
                    continue;
                }
                let from_vertex_id = self.vertex_isv.add(&VertexKey {
                    room_id,
                    node_id: from_node_id,
                    obstacle_mask: from_obstacles_bitmask,
                    actions: from_actions.clone(),
                });
                let to_vertex_id = self.vertex_isv.add(&VertexKey {
                    room_id,
                    node_id: to_node_id,
                    obstacle_mask: if exit_condition.is_some() { 0 } else { to_obstacles_bitmask },
                    actions: to_actions.clone(),
                });
                let link = Link {
                    from_vertex_id,
                    to_vertex_id,
                    requirement: requirement.clone(),
                    notable_strat_name: if notable {
                        Some(notable_strat_name)
                    } else {
                        None
                    },
                    strat_name: strat_name.clone(),
                    strat_notes,
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
                } else {
                    self.links.push(link.clone());
                }

                if let Some(r) = &maybe_exit_req {
                    let exit_to_vertex_id = self.vertex_isv.add(&VertexKey {
                        room_id,
                        node_id: to_node_id,
                        obstacle_mask: 0,
                        actions: vec![VertexAction::Exit(ExitCondition::LeaveNormally {})],
                    });
                    self.links.push(Link {
                        from_vertex_id: to_vertex_id,
                        to_vertex_id: exit_to_vertex_id,
                        requirement: r.clone(),
                        notable_strat_name: None,
                        strat_name: "Base (Maybe Exit -> Exit)".to_string(),
                        strat_notes: vec![],
                    });
                }

                if exit_condition.is_none() && !to_actions.is_empty() {
                    let plain_to_vertex_id = self.vertex_isv.add(&VertexKey {
                        room_id,
                        node_id: to_node_id,
                        obstacle_mask: to_obstacles_bitmask,
                        actions: vec![],
                    });
                    self.links.push(Link {
                        from_vertex_id: to_vertex_id,
                        to_vertex_id: plain_to_vertex_id,
                        requirement: Requirement::Free,
                        notable_strat_name: None,
                        strat_name: "Base (Action -> Plain)".to_string(),
                        strat_notes: vec![],
                    });
                }

                if strat_json.has_key("unlocksDoors") {
                    let mut unlock_node_id_set: HashSet<usize> = HashSet::new();
                    ensure!(strat_json["unlocksDoors"].is_array());
                    for unlock_json in strat_json["unlocksDoors"].members() {
                        if unlock_json.has_key("nodeId") {
                            unlock_node_id_set.insert(unlock_json["nodeId"].as_usize().unwrap());
                        } else {
                            unlock_node_id_set.insert(to_node_id);
                        }
                    }
                    for unlock_node_id in unlock_node_id_set {
                        let unlock_vertex_id = self.vertex_isv.add(&VertexKey {
                            room_id,
                            node_id: to_node_id,
                            obstacle_mask: to_obstacles_bitmask,
                            actions: vec![VertexAction::DoorUnlock(unlock_node_id, to_vertex_id)],
                        }); 
                        let unlock_req = self.get_unlocks_doors_req(unlock_node_id, &ctx)?;
                        self.links.push(
                            Link {
                                from_vertex_id: to_vertex_id,
                                to_vertex_id: unlock_vertex_id,
                                requirement: unlock_req,
                                notable_strat_name: None,
                                strat_name: "Base (Unlock)".to_string(),
                                strat_notes: vec![],
                            }
                        );
                        self.links.push(
                            Link {
                                from_vertex_id: unlock_vertex_id,
                                to_vertex_id: to_vertex_id,
                                requirement: Requirement::Free,
                                notable_strat_name: None,
                                strat_name: "Base (Return from Unlock)".to_string(),
                                strat_notes: vec![],
                            }                            
                        );
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
        let is_west_ocean_bridge = src == (32, 7) || src == (32, 8);
        if src_ptr.is_some() || dst_ptr.is_some() {
            if !is_west_ocean_bridge {
                self.door_ptr_pair_map.insert((src_ptr, dst_ptr), src);
                self.reverse_door_ptr_pair_map
                    .insert(src, (src_ptr, dst_ptr));
            }
            let pos = parse_door_orientation(conn["position"].as_str().unwrap()).unwrap();
            self.door_position.insert(src, pos);
        }
    }

    fn populate_target_locations(&mut self) -> Result<()> {
        // Flags that are relevant to track in the randomizer:
        self.flag_ids = vec![
            "f_ZebesAwake",
            "f_MaridiaTubeBroken",
            "f_ShaktoolDoneDigging",
            "f_UsedAcidChozoStatue",
            "f_UsedBowlingStatue",
            "f_ClearedPitRoom",
            "f_ClearedBabyKraidRoom",
            "f_ClearedPlasmaRoom",
            "f_ClearedMetalPiratesRoom",
            "f_DefeatedBombTorizo",
            "f_DefeatedBotwoon",
            "f_DefeatedCrocomire",
            "f_DefeatedSporeSpawn",
            "f_DefeatedGoldenTorizo",
            "f_DefeatedKraid",
            "f_DefeatedPhantoon",
            "f_DefeatedDraygon",
            "f_DefeatedRidley",
            "f_KilledMetroidRoom1",
            "f_KilledMetroidRoom2",
            "f_KilledMetroidRoom3",
            "f_KilledMetroidRoom4",
            "f_KilledZebetites1",
            "f_KilledZebetites2",
            "f_KilledZebetites3",
            "f_KilledZebetites4",
            "f_MotherBrainGlassBroken",
            "f_DefeatedMotherBrain",
        ].into_iter().map(|x| self.flag_isv.index_by_key[x]).collect();

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
        }

        let mut item_location_vertex_map: HashMap<(RoomId, NodeId), Vec<VertexId>> = HashMap::new();
        let mut flag_location_vertex_map: HashMap<FlagId, Vec<VertexId>> = HashMap::new();
        for (vertex_id, vertex_key) in self.vertex_isv.keys.iter().enumerate() {
            for action in &vertex_key.actions {
                match action {
                    VertexAction::Nothing => panic!("Unexpected VertexAction::Nothing"),
                    VertexAction::MaybeExit(_, _) => {},
                    VertexAction::Exit(exit_condition) => {
                        self.node_exit_conditions.entry((vertex_key.room_id, vertex_key.node_id))
                            .or_default()
                            .push((vertex_id, exit_condition.clone()))
                    }
                    VertexAction::Enter(entrance_condition) => {
                        self.node_entrance_conditions.entry((vertex_key.room_id, vertex_key.node_id))
                            .or_default()
                            .push((vertex_id, entrance_condition.clone()))
                    }
                    VertexAction::DoorUnlock(door_node_id, _) => {
                        self.node_door_unlock.entry((vertex_key.room_id, *door_node_id))
                            .or_default()
                            .push(vertex_id)
                    }
                    VertexAction::ItemCollect(node_id) => {
                        item_location_vertex_map
                            .entry((vertex_key.room_id, *node_id))
                            .or_default()
                            .push(vertex_id);
                    }
                    VertexAction::FlagSet(flag_id) => {
                        flag_location_vertex_map
                            .entry(*flag_id)
                            .or_default()
                            .push(vertex_id);    
                    }
                }
            }
        }

        for &(room_id, node_id) in &self.item_locations {
            let vertex_ids = &item_location_vertex_map[&(room_id, node_id)];
            self.item_vertex_ids.push(vertex_ids.clone());
        }

        for &flag_id in &self.flag_ids {
            let empty_vec = vec![];
            let vertices = flag_location_vertex_map.get(&flag_id).unwrap_or(&empty_vec);
            self.flag_vertex_ids.push(vertices.clone())
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
            if !self.vertex_isv.index_by_key.contains_key(&VertexKey {
                room_id: loc.room_id,
                node_id: loc.node_id,
                obstacle_mask: 0,
                actions: vec![],
            }) {
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
            if !self.vertex_isv.index_by_key.contains_key(&VertexKey {
                room_id: loc.room_id,
                node_id: loc.node_id,
                obstacle_mask: 0,
                actions: vec![],
            }) {
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
            if room.name == "Toilet" {
                self.toilet_room_idx = room_idx;
            }
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
                self.node_coords
                    .insert((room_id, node_id), (door.x, door.y));
            }
            for item in &room.items {
                let (room_id, node_id) = self.reverse_node_ptr_map[&item.addr];
                self.node_coords
                    .insert((room_id, node_id), (item.x, item.y));
            }

            let room_id = *self.room_id_by_ptr.get(&room.rom_address).context(format!(
                "room_id_by_ptr missing entry {:x}",
                room.rom_address
            ))?;
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
                    self.node_tile_coords
                        .insert((room_id, *node_id), tiles.clone());
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

    fn make_links_data(&mut self) {
        self.base_links_data =
            LinksDataGroup::new(self.links.clone(), self.vertex_isv.keys.len(), 0);
    }

    pub fn load_title_screens(&mut self, path: &Path) -> Result<()> {
        info!("Loading title screens");
        let file_it = path.read_dir().with_context(|| {
            format!(
                "Unable to read title screen directory at {}",
                path.display()
            )
        })?;
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
            "name": "h_HeatedBlueGateGlitchLeniency",
            "requires": ["i_HeatedBlueGateGlitchLeniency"],
        };
        *game_data
            .helper_json_map
            .get_mut("h_HeatedGreenGateGlitchLeniency")
            .unwrap() = json::object! {
            "name": "h_HeatedGreenGateGlitchLeniency",
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
            .get_mut("h_canActivateBombTorizo")
            .unwrap() = json::object! {
            "name": "h_canActivateBombTorizo",
            "requires": [],
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
        *game_data
            .helper_json_map
            .get_mut("h_CrocomireCameraFix")
            .unwrap() = json::object! {
            "name": "h_CrocomireCameraFix",
            "requires": [],
        };
        *game_data
            .helper_json_map
            .get_mut("h_EtecoonDoorSpawnFix")
            .unwrap() = json::object! {
            "name": "h_EtecoonDoorSpawnFix",
            "requires": [],
        };
        *game_data
            .helper_json_map
            .get_mut("h_SupersDoubleDamageMotherBrain")
            .unwrap() = json::object! {
            "name": "h_SupersDoubleDamageMotherBrain",
            "requires": ["i_SupersDoubleDamageMotherBrain"],
        };
        // Ammo station refill
        *game_data
            .helper_json_map
            .get_mut("h_useMissileRefillStation")
            .unwrap() = json::object! {
            "name": "h_useMissileRefillStation",
            "requires": ["i_ammoRefill"],
        };
        *game_data
            .helper_json_map
            .get_mut("h_MissileRefillStationAllAmmo")
            .unwrap() = json::object! {
            "name": "h_MissileRefillStationAllAmmo",
            "requires": ["i_ammoRefillAll"],
        };

        // Wall on right side of Tourian Escape Room 1 does not spawn in the randomizer:
        *game_data
            .helper_json_map
            .get_mut("h_AccessTourianEscape1RightDoor")
            .unwrap() = json::object! {
            "name": "h_AccessTourianEscape1RightDoor",
            "requires": [],
        };

        // Elevator heat frames depend on if "Fast elevator" quality-of-life option is enabled; tech can also affect
        // the downward heat frames in Lower Norfair Elevator Room since there is a pause trick to reduce them.
        *game_data
            .helper_json_map
            .get_mut("h_LowerNorfairElevatorDownwardFrames")
            .unwrap() = json::object! {
            "name": "h_LowerNorfairElevatorDownwardFrames",
            "requires": ["i_LowerNorfairElevatorDownwardFrames"],
        };
        *game_data
            .helper_json_map
            .get_mut("h_LowerNorfairElevatorUpwardFrames")
            .unwrap() = json::object! {
            "name": "h_LowerNorfairElevatorUpwardFrames",
            "requires": ["i_LowerNorfairElevatorUpwardFrames"],
        };
        *game_data
            .helper_json_map
            .get_mut("h_MainHallElevatorFrames")
            .unwrap() = json::object! {
            "name": "h_MainHallElevatorFrames",
            "requires": [
                "i_MainHallElevatorFrames",
                {"or":[
                    "h_heatResistant",
                    {"resourceCapacity": [{"type": "RegularEnergy", "count": 149}]}
                ]}
            ],
        };
        *game_data
            .helper_json_map
            .get_mut("h_ShinesparksCostEnergy")
            .unwrap() = json::object! {
            "name": "i_ShinesparksCostEnergy",
            "requires": ["i_ShinesparksCostEnergy"],
        };

        game_data.load_weapons()?;
        game_data.load_enemies()?;
        game_data.load_regions()?;
        game_data.make_links_data();
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
        game_data.load_title_screens(title_screen_path)?;

        for (vertex_id, key) in game_data.vertex_isv.keys.iter().enumerate() {
            if (key.room_id, key.node_id) == (219, 1) {
                println!("{}: {:?}", vertex_id, key);

                let to_ids: Vec<VertexId> = game_data.base_links_data.links_by_src[vertex_id].iter().map(|x| x.1.to_vertex_id).collect();
                println!("{} -> {:?}", vertex_id, to_ids);
                let from_ids: Vec<VertexId> = game_data.base_links_data.links_by_dst[vertex_id].iter().map(|x| x.1.to_vertex_id).collect();
                println!("{} <- {:?}", vertex_id, from_ids);

            }
        }

        Ok(game_data)
    }
}
