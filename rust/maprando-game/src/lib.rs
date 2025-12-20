// The changes suggested by this lint usually make the code more cluttered and less clear:
#![allow(clippy::needless_range_loop)]
// TODO: consider removing this later. It's not a bad lint but I don't want to deal with it now.
#![allow(clippy::too_many_arguments)]

pub mod glowpatch;
pub mod smart_xml;

use crate::glowpatch::GlowPatch;
use anyhow::{Context, Result, bail, ensure};
use hashbrown::{HashMap, HashSet};
use image::{Rgb, io::Reader as ImageReader};
use json::{self, JsonValue};
use log::{error, info, warn};
use ndarray::Array3;
use num_enum::TryFromPrimitive;
use serde::{Deserialize, Serialize};
use std::borrow::ToOwned;
use std::fmt::{self, Debug, Formatter};
use std::fs::File;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use strum::VariantNames;
use strum_macros::{EnumString, VariantNames};

pub const TECH_ID_CAN_WALLJUMP: TechId = 76;
pub const TECH_ID_CAN_HEAT_RUN: TechId = 6;
pub const TECH_ID_CAN_MID_AIR_MORPH: TechId = 32;
pub const TECH_ID_CAN_SPEEDBALL: TechId = 42;
pub const TECH_ID_CAN_MOCKBALL: TechId = 41;
pub const TECH_ID_CAN_MANAGE_RESERVES: TechId = 18;
pub const TECH_ID_CAN_PAUSE_ABUSE: TechId = 19;
pub const TECH_ID_CAN_SHINECHARGE_MOVEMENT: TechId = 136;
pub const TECH_ID_CAN_SHINESPARK: TechId = 132;
pub const TECH_ID_CAN_HORIZONTAL_SHINESPARK: TechId = 133;
pub const TECH_ID_CAN_MIDAIR_SHINESPARK: TechId = 134;
pub const TECH_ID_CAN_BE_PATIENT: TechId = 1;
pub const TECH_ID_CAN_BE_VERY_PATIENT: TechId = 2;
pub const TECH_ID_CAN_BE_EXTREMELY_PATIENT: TechId = 3;
pub const TECH_ID_CAN_SKIP_DOOR_LOCK: TechId = 184;
pub const TECH_ID_CAN_DISABLE_EQUIPMENT: TechId = 12;
pub const TECH_ID_CAN_SPRING_BALL_BOUNCE: TechId = 38;
pub const TECH_ID_CAN_STUTTER_WATER_SHINECHARGE: TechId = 151;
pub const TECH_ID_CAN_TEMPORARY_BLUE: TechId = 146;
pub const TECH_ID_CAN_STATIONARY_SPIN_JUMP: TechId = 63;
pub const TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK: TechId = 157;
pub const TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK_FROM_WATER: TechId = 158;
pub const TECH_ID_CAN_ENTER_R_MODE: TechId = 161;
pub const TECH_ID_CAN_ENTER_G_MODE: TechId = 162;
pub const TECH_ID_CAN_ENTER_G_MODE_IMMOBILE: TechId = 163;
pub const TECH_ID_CAN_ARTIFICIAL_MORPH: TechId = 164;
pub const TECH_ID_CAN_HEATED_G_MODE: TechId = 198;
pub const TECH_ID_CAN_MOONFALL: TechId = 25;
pub const TECH_ID_CAN_PRECISE_GRAPPLE: TechId = 51;
pub const TECH_ID_CAN_GRAPPLE_JUMP: TechId = 52;
pub const TECH_ID_CAN_GRAPPLE_TELEPORT: TechId = 55;
pub const TECH_ID_CAN_SAMUS_EATER_TELEPORT: TechId = 194;
pub const TECH_ID_CAN_SUPER_SINK: TechId = 204;
pub const TECH_ID_CAN_KAGO: TechId = 107;
pub const TECH_ID_CAN_SUITLESS_LAVA_DIVE: TechId = 5;
pub const TECH_ID_CAN_HERO_SHOT: TechId = 130;
pub const TECH_ID_CAN_OFF_SCREEN_SUPER_SHOT: TechId = 126;
pub const TECH_ID_CAN_BOMB_HORIZONTALLY: TechId = 87;
pub const TECH_ID_CAN_MOONDANCE: TechId = 26;
pub const TECH_ID_CAN_EXTENDED_MOONDANCE: TechId = 27;
pub const TECH_ID_CAN_ENEMY_STUCK_MOONFALL: TechId = 28;
pub const TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP: TechId = 197;
pub const TECH_ID_CAN_SPIKE_SUIT: TechId = 141;
pub const TECH_ID_CAN_ELEVATOR_CRYSTAL_FLASH: TechId = 178;
pub const TECH_ID_CAN_CARRY_FLASH_SUIT: TechId = 207;
pub const TECH_ID_CAN_TRICKY_CARRY_FLASH_SUIT: TechId = 142;
pub const TECH_ID_CAN_HYPER_GATE_SHOT: TechId = 10001;

#[allow(clippy::type_complexity)]
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Map {
    #[serde(default)]
    pub room_mask: Vec<bool>,
    pub rooms: Vec<(usize, usize)>, // (x, y) of upper-left corner of room on map
    pub doors: Vec<(
        (Option<usize>, Option<usize>), // Source (exit_ptr, entrance_ptr)
        (Option<usize>, Option<usize>), // Destination (exit_ptr, entrance_ptr)
        bool,                           // bidirectional
    )>,
    pub area: Vec<usize>,    // Area number: 0, 1, 2, 3, 4, or 5
    pub subarea: Vec<usize>, // Subarea number: 0 or 1
    #[serde(default)] // Default for backward compatibility
    pub subsubarea: Vec<usize>, // Subsubarea number: 0 or 1
}

pub type TechId = i32; // Tech ID from sm-json-data
pub type TechIdx = usize; // Index into GameData.tech_isv.keys: distinct tech names from sm-json-data
pub type NotableIdx = usize; // Index into GameData.notable_strats_isv.keys: distinct pairs (room_id, notable_id) from sm-json-data
pub type ItemId = usize; // Index into GameData.item_isv.keys: 21 distinct item names
pub type ItemIdx = usize; // Index into the game's item bit array (in RAM at 7E:D870)
pub type FlagId = usize; // Index into GameData.flag_isv.keys: distinct game flag names from sm-json-data
pub type RoomId = usize; // Room ID from sm-json-data
pub type RoomPtr = usize; // Room pointer (PC address of room header)
pub type RoomStateIdx = usize; // Room state index
pub type NodeId = usize; // Node ID from sm-json-data (only unique within a room)
pub type StartLocationId = usize; // Index into GameData.start_locations
pub type NodePtr = usize; // nodeAddress from sm-json-data: for items this is the PC address of PLM, for doors it is PC address of door data
pub type StratId = usize; // Strat ID from sm-json-data (only unique within a room)
pub type NotableId = usize; // Notable ID from sm-json-data (only unique within a room)
pub type VertexId = usize; // Index into GameData.vertex_isv.keys: (room_id, node_id, obstacle_bitmask) combinations
pub type ItemLocationId = usize; // Index into GameData.item_locations: 100 nodes each containing an item
pub type LinkLength = u32; // Length of a link (based on difficulty)
pub type ObstacleMask = usize; // Bitmask where `i`th bit (from least significant) indicates `i`th obstacle cleared within a room
pub type WeaponMask = usize; // Bitmask where `i`th bit indicates availability of (or vulnerability to) `i`th weapon.
pub type Capacity = i16; // Data type used to represent quantities of energy, ammo, etc.
pub type ItemPtr = usize; // PC address of item in PLM list
pub type DoorPtr = usize; // PC address of door data for exiting given door
pub type DoorPtrPair = (Option<DoorPtr>, Option<DoorPtr>); // PC addresses of door data for exiting & entering given door (from vanilla door connection)
pub type TilesetIdx = usize; // Tileset index
pub type AreaIdx = usize; // Area index (0..5)
pub type StepTrailId = i32;
pub type LinkIdx = i32;
pub type TraversalId = usize; // Index into Traversal.past_steps

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
    VariantNames,
    TryFromPrimitive,
    Serialize,
    Deserialize,
    PartialOrd,
    Ord,
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EnemyDrop {
    pub nothing_weight: Float,
    pub small_energy_weight: Float,
    pub large_energy_weight: Float,
    pub missile_weight: Float,
    pub super_weight: Float,
    pub power_bomb_weight: Float,
    pub count: Capacity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BeamType {
    Charge,
    Ice,
    Wave,
    Spazer,
    Plasma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DoorType {
    Blue,
    Red,
    Green,
    Yellow,
    Gray,
    Beam(BeamType),
    Wall,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RidleyStuck {
    None,
    Top,
    Bottom,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Requirement {
    Free,
    Never,
    Tech(TechIdx),
    Notable(NotableIdx),
    Item(ItemId),
    Flag(FlagId),
    NotFlag(FlagId),
    MotherBrainBarrierClear(usize),
    DisableableETank,
    Walljump,
    ShineCharge {
        used_tiles: Float,
        heated: bool,
    },
    SpeedBall {
        used_tiles: Float,
        heated: bool,
    },
    GetBlueSpeed {
        used_tiles: Float,
        heated: bool,
    },
    ShineChargeFrames(Capacity),
    Shinespark {
        shinespark_tech_idx: usize,
        frames: Capacity,
        excess_frames: Capacity,
    },
    HeatFrames(Capacity),
    SuitlessHeatFrames(Capacity),
    SimpleHeatFrames(Capacity),
    HeatFramesWithEnergyDrops(Capacity, Vec<EnemyDrop>, Vec<EnemyDrop>),
    LavaFrames(Capacity),
    LavaFramesWithEnergyDrops(Capacity, Vec<EnemyDrop>, Vec<EnemyDrop>),
    GravitylessLavaFrames(Capacity),
    AcidFrames(Capacity),
    GravitylessAcidFrames(Capacity),
    MetroidFrames(Capacity),
    CycleFrames(Capacity),
    SimpleCycleFrames(Capacity),
    Farm {
        requirement: Box<Requirement>,
        enemy_drops: Vec<EnemyDrop>,
        enemy_drops_buffed: Vec<EnemyDrop>,
        full_energy: bool,
        full_missiles: bool,
        full_supers: bool,
        full_power_bombs: bool,
    },
    Damage(Capacity),
    MissilesAvailable(Capacity),
    SupersAvailable(Capacity),
    PowerBombsAvailable(Capacity),
    RegularEnergyAvailable(Capacity),
    ReserveEnergyAvailable(Capacity),
    EnergyAvailable(Capacity),
    MissilesCapacity(Capacity),
    SupersCapacity(Capacity),
    PowerBombsCapacity(Capacity),
    RegularEnergyCapacity(Capacity),
    ReserveEnergyCapacity(Capacity),
    MissilesMissingAtMost(Capacity),
    SupersMissingAtMost(Capacity),
    PowerBombsMissingAtMost(Capacity),
    RegularEnergyMissingAtMost(Capacity),
    ReserveEnergyMissingAtMost(Capacity),
    EnergyMissingAtMost(Capacity),
    Energy(Capacity),
    RegularEnergy(Capacity),
    ReserveEnergy(Capacity),
    Missiles(Capacity),
    Supers(Capacity),
    PowerBombs(Capacity),
    EnergyRefill(Capacity),
    RegularEnergyRefill(Capacity),
    ReserveRefill(Capacity),
    MissileRefill(Capacity),
    SuperRefill(Capacity),
    PowerBombRefill(Capacity),
    ClimbWithoutLava,
    AmmoStationRefill,
    AmmoStationRefillAll,
    EnergyStationRefill,
    RegularEnergyDrain(Capacity),
    ReserveEnergyDrain(Capacity),
    MissileDrain(Capacity),
    LowerNorfairElevatorDownFrames,
    LowerNorfairElevatorUpFrames,
    MainHallElevatorFrames,
    EquipmentScreenCycleFrames,
    ShinesparksCostEnergy,
    SupersDoubleDamageMotherBrain,
    GateGlitchLeniency {
        green: bool,
        heated: bool,
    },
    HeatedDoorStuckLeniency {
        heat_frames: Capacity,
    },
    SpikeSuitSpikeHitLeniency,
    SpikeSuitThornHitLeniency,
    SpikeSuitSamusEaterLeniency,
    SpikeSuitPowerBombLeniency,
    XModeSpikeHitLeniency {},
    XModeThornHitLeniency {},
    FramePerfectXModeThornHitLeniency,
    FramePerfectDoubleXModeThornHitLeniency,
    SpeedKeepSpikeHitLeniency,
    ElevatorCFLeniency,
    BombIntoCrystalFlashClipLeniency {},
    JumpIntoCrystalFlashClipLeniency {},
    ReserveTrigger {
        min_reserve_energy: Capacity,
        max_reserve_energy: Capacity,
        heated: bool,
    },
    EnemyKill {
        count: Capacity,
        vul: EnemyVulnerabilities,
    },
    PhantoonFight {},
    DraygonFight {
        can_be_patient_tech_idx: usize,
        can_be_very_patient_tech_idx: usize,
        can_be_extremely_patient_tech_idx: usize,
    },
    RidleyFight {
        can_be_patient_tech_idx: usize,
        can_be_very_patient_tech_idx: usize,
        can_be_extremely_patient_tech_idx: usize,
        power_bombs: bool,
        g_mode: bool,
        stuck: RidleyStuck,
    },
    BotwoonFight {
        second_phase: bool,
    },
    MotherBrain2Fight {
        can_be_very_patient_tech_id: usize,
        r_mode: bool,
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
    UnlockDoor {
        room_id: RoomId,
        node_id: NodeId,
        requirement_red: Box<Requirement>,
        requirement_green: Box<Requirement>,
        requirement_yellow: Box<Requirement>,
        requirement_charge: Box<Requirement>,
    },
    ResetRoom {
        room_id: RoomId,
        node_id: NodeId,
    },
    EscapeMorphLocation,
    DoorTransition,
    GainFlashSuit,
    UseFlashSuit {
        carry_flash_suit_tech_idx: TechIdx,
    },
    NoFlashSuit,
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
            } else if let Requirement::And(and_reqs) = req {
                out_reqs.extend(and_reqs);
            } else {
                out_reqs.push(req);
            }
        }
        if out_reqs.is_empty() {
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
            } else if let Requirement::Or(or_reqs) = req {
                out_reqs.extend(or_reqs);
            } else {
                out_reqs.push(req);
            }
        }
        if out_reqs.is_empty() {
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

    pub fn make_blue_speed(tiles: f32, heated: bool) -> Requirement {
        if tiles < 11.0 {
            // An effective runway length of 11 is the minimum possible length of shortcharge supported in the logic.
            // Strats requiring shorter runways than this are discarded to save processing time during generation.
            // Technically it is humanly viable to go as low as about 10.5, but below 11 the precision needed is so much
            // that it would not be reasonable to require on any settings.
            Requirement::Never
        } else {
            Requirement::GetBlueSpeed {
                used_tiles: Float::new(tiles),
                heated,
            }
        }
    }

    pub fn print_pretty(&self, indent: usize, game_data: &GameData) {
        let spaces = " ".repeat(indent);
        print!("{spaces}");
        match self {
            &Requirement::Tech(tech_idx) => {
                let tech_id = game_data.tech_isv.keys[tech_idx];
                print!("Tech({})", game_data.tech_names[&tech_id]);
            }
            &Requirement::Item(item_idx) => {
                print!("Item({})", game_data.item_isv.keys[item_idx]);
            }
            &Requirement::Flag(flag_idx) => {
                print!("Flag({})", game_data.flag_isv.keys[flag_idx]);
            }
            Requirement::And(reqs) => {
                println!("And(");
                for r in reqs {
                    r.print_pretty(indent + 2, game_data);
                    println!(",");
                }
                print!("{spaces})")
            }
            Requirement::Or(reqs) => {
                println!("Or(");
                for r in reqs {
                    r.print_pretty(indent + 2, game_data);
                    println!(",");
                }
                print!("{spaces})")
            }
            other => {
                print!("{other:?}");
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Link {
    pub from_vertex_id: VertexId,
    pub to_vertex_id: VertexId,
    pub requirement: Requirement,
    pub start_with_shinecharge: bool,
    pub end_with_shinecharge: bool,
    pub difficulty: u8,
    pub length: LinkLength,
    pub strat_id: Option<usize>,
    pub strat_name: String,
    // TODO: Remove this field since this data can be looked up elsewhere:
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

#[allow(clippy::type_complexity)]
#[derive(Deserialize, Default, Clone, Debug)]
pub struct RoomGeometry {
    pub room_id: usize,
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

#[derive(Deserialize, Clone)]
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
    EnemiesCleared,
    CanMidAirMorph,
    CanUsePowerBombs,
    CanMoonfall,
    CanReverseGate,
    CanAcidDive,
    CanOffCameraShot,
    CanKago,
    CanHeroShot,
    CanOneTapShortcharge,
}

#[derive(Deserialize, Clone)]
pub struct EscapeTimingCondition {
    pub requires: Vec<EscapeConditionRequirement>,
    pub in_game_time: f32,
}

#[derive(Deserialize, Clone)]
pub struct EscapeTiming {
    pub to_door: EscapeTimingDoor,
    pub in_game_time: Option<f32>,
    #[serde(default)]
    pub conditions: Vec<EscapeTimingCondition>,
}

#[derive(Deserialize, Clone)]
pub struct EscapeTimingGroup {
    pub from_door: EscapeTimingDoor,
    pub to: Vec<EscapeTiming>,
}

#[derive(Deserialize, Clone)]
pub struct EscapeTimingRoom {
    pub room_id: RoomId,
    pub room_name: String,
    pub timings: Vec<EscapeTimingGroup>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
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
    #[serde(skip_serializing, skip_deserializing)]
    pub requires_parsed: Option<Requirement>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct HubLocation {
    pub room_id: usize,
    pub node_id: usize,
    pub vertex_id: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EnemyVulnerabilities {
    pub hp: Capacity,
    pub non_ammo_vulnerabilities: WeaponMask,
    pub missile_damage: Capacity,
    pub super_damage: Capacity,
    pub power_bomb_damage: Capacity,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
        "up" => DoorOrientation::Up,
        "down" => DoorOrientation::Down,
        _ => bail!(format!(
            "Unrecognized door orientation '{}'",
            door_orientation
        )),
    })
}

#[derive(Clone, Debug)]
pub struct GModeRegainMobility {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SparkPosition {
    Top,
    Bottom,
    Any,
}

// Hashable wrapper for f32 based on its bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

impl Debug for Float {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "Float({})", self.get())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporaryBlueDirection {
    Left,
    Right,
    Any,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlueOption {
    Yes,
    No,
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum GrappleSwingBlockEnvironment {
    #[default]
    Air,
    Water,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GrappleSwingBlock {
    position: (Float, Float),
    environment: GrappleSwingBlockEnvironment,
    obstructions: Vec<(i32, i32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GrappleJumpPosition {
    Left,
    Right,
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExitCondition {
    LeaveNormally {},
    LeaveWithRunway {
        effective_length: Float,
        min_extra_run_speed: Float,
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
        door_orientation: DoorOrientation,
    },
    LeaveSpinning {
        remote_runway_length: Float,
        blue: BlueOption,
        heated: bool,
        min_extra_run_speed: Float,
        max_extra_run_speed: Float,
    },
    LeaveWithMockball {
        remote_runway_length: Float,
        landing_runway_length: Float,
        blue: BlueOption,
        heated: bool,
        min_extra_run_speed: Float,
        max_extra_run_speed: Float,
    },
    LeaveWithSpringBallBounce {
        remote_runway_length: Float,
        landing_runway_length: Float,
        blue: BlueOption,
        heated: bool,
        movement_type: BounceMovementType,
        min_extra_run_speed: Float,
        max_extra_run_speed: Float,
    },
    LeaveSpaceJumping {
        remote_runway_length: Float,
        blue: BlueOption,
        heated: bool,
        min_extra_run_speed: Float,
        max_extra_run_speed: Float,
    },
    LeaveWithGModeSetup {
        knockback: bool,
        heated: bool,
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
    LeaveWithSidePlatform {
        effective_length: Float,
        height: Float,
        obstruction: (u16, u16),
        environment: SidePlatformEnvironment,
    },
    LeaveWithGrappleSwing {
        blocks: Vec<GrappleSwingBlock>,
    },
    LeaveWithGrappleJump {
        position: GrappleJumpPosition,
    },
    LeaveWithGrappleTeleport {
        block_positions: Vec<(u16, u16)>,
    },
    LeaveWithSamusEaterTeleport {
        floor_positions: Vec<(u16, u16)>,
        ceiling_positions: Vec<(u16, u16)>,
    },
    LeaveWithSuperSink {},
}

fn parse_spark_position(s: Option<&str>) -> Result<SparkPosition> {
    Ok(match s {
        Some("top") => SparkPosition::Top,
        Some("bottom") => SparkPosition::Bottom,
        None => SparkPosition::Any,
        _ => bail!("Unrecognized spark position: {}", s.unwrap()),
    })
}

fn parse_grapple_jump_position(s: Option<&str>) -> Result<GrappleJumpPosition> {
    Ok(match s {
        Some("left") => GrappleJumpPosition::Left,
        Some("right") => GrappleJumpPosition::Right,
        Some("any") => GrappleJumpPosition::Any,
        _ => bail!("Unrecognized grapple jump position: {}", s.unwrap()),
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GModeMode {
    Direct,
    Indirect,
    Any,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GModeMobility {
    Mobile,
    Immobile,
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToiletCondition {
    No,
    Yes,
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntranceCondition {
    pub through_toilet: ToiletCondition,
    pub main: MainEntranceCondition,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BounceMovementType {
    Controlled,
    Uncontrolled,
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SidePlatformEnvironment {
    #[default]
    Any,
    Air,
    Water,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SidePlatformEntrance {
    pub min_tiles: Float,
    pub speed_booster: Option<bool>,
    pub min_height: Float,
    pub max_height: Float,
    pub obstructions: Vec<(u16, u16)>,
    pub environment: SidePlatformEnvironment,
    pub requirement: Requirement,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
        min_tiles: Float,
        heated: bool,
    },
    ComeInGettingBlueSpeed {
        effective_length: Float,
        min_tiles: Float,
        heated: bool,
        min_extra_run_speed: Float,
        max_extra_run_speed: Float,
    },
    ComeInShinecharged {},
    ComeInShinechargedJumping {},
    ComeInWithSpark {
        position: SparkPosition,
        door_orientation: DoorOrientation,
    },
    ComeInSpeedballing {
        effective_runway_length: Float,
        min_extra_run_speed: Float,
        max_extra_run_speed: Float,
        heated: bool,
    },
    ComeInWithTemporaryBlue {
        direction: TemporaryBlueDirection,
    },
    ComeInSpinning {
        unusable_tiles: Float,
        min_extra_run_speed: Float,
        max_extra_run_speed: Float,
    },
    ComeInBlueSpinning {
        unusable_tiles: Float,
        min_extra_run_speed: Float,
        max_extra_run_speed: Float,
    },
    ComeInBlueSpaceJumping {
        min_extra_run_speed: Float,
        max_extra_run_speed: Float,
    },
    ComeInWithMockball {
        speed_booster: Option<bool>,
        adjacent_min_tiles: Float,
        remote_and_landing_min_tiles: Vec<(Float, Float)>,
    },
    ComeInWithSpringBallBounce {
        speed_booster: Option<bool>,
        adjacent_min_tiles: Float,
        remote_and_landing_min_tiles: Vec<(Float, Float)>,
        movement_type: BounceMovementType,
    },
    ComeInWithBlueSpringBallBounce {
        min_extra_run_speed: Float,
        max_extra_run_speed: Float,
        min_landing_tiles: Float,
        movement_type: BounceMovementType,
    },
    ComeInStutterShinecharging {
        min_tiles: Float,
    },
    ComeInStutterGettingBlueSpeed {
        min_tiles: Float,
    },
    ComeInWithBombBoost {},
    ComeInWithDoorStuckSetup {
        heated: bool,
        door_orientation: DoorOrientation,
    },
    ComeInWithRMode {
        heated: bool,
    },
    ComeInWithGMode {
        mode: GModeMode,
        morphed: bool,
        mobility: GModeMobility,
        heated: bool,
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
    ComeInWithSidePlatform {
        platforms: Vec<SidePlatformEntrance>,
    },
    ComeInWithGrappleSwing {
        blocks: Vec<GrappleSwingBlock>,
    },
    ComeInWithGrappleJump {
        position: GrappleJumpPosition,
    },
    ComeInWithGrappleTeleport {
        block_positions: Vec<(u16, u16)>,
    },
    ComeInWithSamusEaterTeleport {
        floor_positions: Vec<(u16, u16)>,
        ceiling_positions: Vec<(u16, u16)>,
    },
    ComeInWithSuperSink {},
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
    geom.length - geom.starting_down_tiles - 9.0 / 16.0 * (1.0 - geom.open_end)
        + 1.0 / 3.0 * geom.steep_up_tiles
        + 1.0 / 7.0 * geom.steep_down_tiles
        + 5.0 / 27.0 * geom.gentle_up_tiles
        + 5.0 / 59.0 * geom.gentle_down_tiles
}

#[derive(Default, Clone)]
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

fn get_logical_gray_door_node_ids() -> Vec<(RoomId, NodeId)> {
    // Gray door nodes that will be modeled in logic, i.e. their `locks` requirement must be met to go through them, and
    // and an obstacle will be added to the room to track them staying open when entering through them.
    // This excludes Bomb Torizo Room and Pit Room since those gray doors are free to open (and Pit Room ones don't always exist).
    vec![
        // Pirate rooms:
        (82, 1),  // Baby Kraid Room left door
        (82, 2),  // Baby Kraid Room right door
        (139, 1), // Metal Pirates Room left door
        (139, 2), // Metal Pirates Room right door
        (219, 1), // Plasma Room left door
        // Boss rooms:
        (84, 1),  // Kraid Room left door
        (84, 2),  // Kraid Room right door
        (158, 1), // Phantoon's Room left door
        (193, 1), // Draygon's Room left door
        (193, 2), // Draygon's Room right door
        (142, 1), // Ridley's Room left door
        (142, 2), // Ridley's Room right door
        // Miniboss rooms:
        (57, 2),  // Spore Spawn Room bottom door
        (122, 2), // Crocomire's Room top door
        (185, 1), // Botwoon's Room left door
        (150, 2), // Golden Torizo Room
    ]
}

fn get_flagged_gray_door_node_ids() -> Vec<(RoomId, NodeId)> {
    // Gray doors which have a side effect of setting flags that we want to model:
    vec![
        // Pirate rooms:
        (12, 1),  // Pit Room left door
        (82, 2),  // Baby Kraid Room right door
        (139, 1), // Metal Pirates Room left door
        (139, 2), // Metal Pirates Room right door
        (219, 1), // Plasma Room
        (226, 1), // Metroid Room 1
        (227, 2), // Metroid Room 2
        (228, 2), // Metroid Room 3
        (229, 2), // Metroid Room 4
    ]
}

fn parse_hex(v: &JsonValue, default: f32) -> Result<f32> {
    if v.is_null() {
        return Ok(default);
    } else if !v.is_string() {
        bail!("Unexpected type of value in parse_hex: {}", v);
    }
    let s = v.as_str().unwrap();
    if !s.starts_with("$") {
        bail!("hex value should start with '$': {}", s);
    }
    let s: &str = s.split_at(1).1;
    let pair: Vec<&str> = s.split(".").collect();
    if pair.len() != 2 {
        bail!("Unexpected format in parse_hex: {}", s);
    }
    let x = i64::from_str_radix(pair[0], 16)?;
    let y = i64::from_str_radix(pair[1], 16)?;
    let p = 16i64.pow(pair[1].len() as u32);
    let f = x as f32 + (y as f32 / p as f32);
    Ok(f)
}

type TitleScreenImage = ndarray::Array3<u8>;

#[derive(Default, Clone)]
pub struct TitleScreenData {
    pub top_left: Vec<TitleScreenImage>,
    pub top_right: Vec<TitleScreenImage>,
    pub bottom_left: Vec<TitleScreenImage>,
    pub bottom_right: Vec<TitleScreenImage>,
    pub map_station: TitleScreenImage,
}

pub fn read_image(path: &Path) -> Result<Array3<u8>> {
    let img = ImageReader::open(path)
        .with_context(|| format!("Unable to open image: {}", path.display()))?
        .decode()
        .with_context(|| format!("Unable to decode image: {}", path.display()))?
        .to_rgb8();
    let width = img.width() as usize;
    let height = img.height() as usize;
    let mut arr: Array3<u8> = Array3::zeros([height, width, 3]);
    for y in 0..height {
        for x in 0..width {
            let &Rgb([r, g, b]) = img.get_pixel(x as u32, y as u32);
            arr[[y, x, 0]] = r;
            arr[[y, x, 1]] = g;
            arr[[y, x, 2]] = b;
        }
    }
    Ok(arr)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum VertexAction {
    #[default]
    Nothing, // This should never be constructed, just here because we need a default value
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct VertexKey {
    pub room_id: RoomId,
    pub node_id: NodeId,
    pub obstacle_mask: ObstacleMask,
    pub actions: Vec<VertexAction>,
}

#[derive(Clone)]
pub struct NotableInfo {
    pub room_id: RoomId,
    pub notable_id: NotableId,
    pub name: String,
    pub note: String,
}

#[derive(Deserialize, Clone)]
pub struct StratVideo {
    pub room_id: usize,
    pub strat_id: usize,
    pub video_id: usize,
    pub created_user: String,
    pub note: String,
    pub dev_note: String,
}

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum DoorLockType {
    Gray,
    Wall,
    // Ammo doors:
    Red,
    Green,
    Yellow,
    // Beam doors:
    Charge,
    Ice,
    Wave,
    Spazer,
    Plasma,
}

#[derive(Copy, Clone, Debug, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum MapTileEdge {
    #[default]
    Empty,
    QolEmpty,
    Passage,
    QolPassage,
    Door,
    QolDoor,
    Wall,
    QolWall,
    ElevatorEntrance,
    Sand,
    QolSand,
    // Extension used at runtime:
    LockedDoor(DoorLockType),
}

#[derive(Copy, Clone, Debug, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum MapTileInterior {
    #[default]
    Empty,
    Item,
    DoubleItem,
    HiddenItem,
    ElevatorPlatformHigh,
    ElevatorPlatformLow,
    SaveStation,
    MapStation,
    EnergyRefill,
    AmmoRefill,
    DoubleRefill,
    Ship,
    Event,
    // Extensions added at runtime:
    Objective,
    AmmoItem,
    MediumItem,
    MajorItem,
}

impl MapTileInterior {
    pub fn is_item(&self) -> bool {
        use MapTileInterior::*;
        matches!(
            self,
            Item | DoubleItem | HiddenItem | AmmoItem | MediumItem | MajorItem
        )
    }
}

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum MapTileSpecialType {
    SlopeUpFloorLow,
    SlopeUpFloorHigh,
    SlopeUpCeilingLow,
    SlopeUpCeilingHigh,
    SlopeDownFloorLow,
    SlopeDownFloorHigh,
    SlopeDownCeilingLow,
    SlopeDownCeilingHigh,
    Tube,
    Elevator,
    Black,
    // Extensions added at runtime:
    AreaTransition(AreaIdx, Direction),
}

#[derive(Clone, Copy, Debug, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum MapLiquidType {
    #[default]
    None,
    Water,
    Lava,
    Acid,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MapTile {
    pub coords: (usize, usize),
    pub area: Option<usize>,
    #[serde(default)]
    pub left: MapTileEdge,
    #[serde(default)]
    pub right: MapTileEdge,
    #[serde(default)]
    pub top: MapTileEdge,
    #[serde(default)]
    pub bottom: MapTileEdge,
    #[serde(default)]
    pub interior: MapTileInterior,
    #[serde(default)]
    pub heated: bool,
    #[serde(default)]
    pub liquid_type: MapLiquidType,
    pub liquid_level: Option<f32>,
    pub special_type: Option<MapTileSpecialType>,
    // Extensions added at runtime:
    #[serde(default)]
    pub faded: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MapTileData {
    pub room_id: usize,
    pub room_name: String,
    #[serde(default)]
    pub liquid_type: MapLiquidType,
    pub liquid_level: Option<f32>,
    #[serde(default)]
    pub heated: bool,
    pub map_tiles: Vec<MapTile>,
}

#[derive(Deserialize)]
struct MapTileDataFile {
    rooms: Vec<MapTileData>,
}

type GfxTile1Bpp = [u8; 8];

#[derive(Default, Clone)]
pub struct VariableWidthFont {
    pub gfx: Vec<GfxTile1Bpp>,
    pub widths: Vec<u8>,
    pub char_isv: IndexedVec<char>,
}

#[derive(Clone)]
pub struct ExitInfo {
    pub vertex_id: VertexId,
    pub exit_condition: ExitCondition,
    pub exit_req: Requirement,
}

// TODO: Clean this up, e.g. pull out a separate structure to hold
// temporary data used only during loading, replace any
// remaining JsonValue types in the main struct with something
// more structured; combine maps with the same keys; also maybe unify the room geometry data
// with sm-json-data and cut back on the amount of different
// keys/IDs/indexes for rooms, nodes, and doors.
#[derive(Default, Clone)]
pub struct GameData {
    sm_json_data_path: PathBuf,
    pub tech_isv: IndexedVec<TechId>,
    pub notable_isv: IndexedVec<(RoomId, NotableId)>,
    pub notable_info: Vec<NotableInfo>,
    pub flag_isv: IndexedVec<String>,
    pub item_isv: IndexedVec<String>,
    weapon_isv: IndexedVec<String>,
    weapon_categories: HashMap<String, Vec<String>>, // map from weapon category to specific weapons with that category
    enemy_attack_damage: HashMap<(String, String), Capacity>,
    enemy_vulnerabilities: HashMap<String, EnemyVulnerabilities>,
    enemy_json: HashMap<String, JsonValue>,
    enemy_json_buffed: HashMap<String, JsonValue>,
    weapon_json_map: HashMap<String, JsonValue>,
    non_ammo_weapon_mask: WeaponMask,
    pub tech_json_map: HashMap<TechId, JsonValue>,
    pub tech_names: HashMap<TechId, String>,
    pub tech_id_by_name: HashMap<String, TechId>,
    pub notable_id_by_name: HashMap<(RoomId, String), NotableId>,
    pub helper_json_map: HashMap<String, JsonValue>,
    pub helper_category_map: HashMap<String, String>,
    pub tech_requirement: HashMap<(TechId, bool), Option<Requirement>>,
    pub helpers: HashMap<String, Option<Requirement>>,
    pub room_json_map: HashMap<RoomId, JsonValue>,
    pub room_obstacle_idx_map: HashMap<RoomId, HashMap<String, usize>>,
    pub room_full_area: HashMap<RoomId, String>,
    pub node_json_map: HashMap<(RoomId, NodeId), JsonValue>,
    pub node_spawn_at_map: HashMap<(RoomId, NodeId), NodeId>,
    pub reverse_node_ptr_map: HashMap<NodePtr, (RoomId, NodeId)>,
    pub node_ptr_map: HashMap<(RoomId, NodeId), NodePtr>,
    pub node_door_unlock: HashMap<(RoomId, NodeId), Vec<VertexId>>,
    pub node_entrance_conditions: HashMap<(RoomId, NodeId), Vec<(VertexId, EntranceCondition)>>,
    pub node_exit_conditions: HashMap<(RoomId, NodeId), Vec<ExitInfo>>,
    pub node_gmode_regain_mobility: HashMap<(RoomId, NodeId), Vec<(Link, GModeRegainMobility)>>,
    pub node_reset_room_requirement: HashMap<(RoomId, NodeId), Requirement>,
    pub room_num_obstacles: HashMap<RoomId, usize>,
    pub door_ptr_pair_map: HashMap<DoorPtrPair, (RoomId, NodeId)>,
    pub reverse_door_ptr_pair_map: HashMap<(RoomId, NodeId), DoorPtrPair>,
    pub door_position: HashMap<(RoomId, NodeId), DoorOrientation>,
    pub vertex_isv: IndexedVec<VertexKey>,
    pub grey_lock_map: HashMap<(RoomId, NodeId), JsonValue>,
    pub item_locations: Vec<(RoomId, NodeId)>,
    pub item_vertex_ids: Vec<Vec<VertexId>>,
    pub flag_ids: Vec<FlagId>,
    pub flag_vertex_ids: Vec<Vec<VertexId>>,
    pub save_locations: Vec<(RoomId, NodeId)>,
    pub links: Vec<Link>,
    pub base_links_data: LinksDataGroup,
    pub room_geometry: Vec<RoomGeometry>,
    pub room_ptrs: Vec<RoomPtr>,
    pub room_and_door_idxs_by_door_ptr_pair:
        HashMap<DoorPtrPair, (RoomGeometryRoomIdx, RoomGeometryDoorIdx)>,
    pub room_ptr_by_id: HashMap<RoomId, RoomPtr>,
    pub room_id_by_ptr: HashMap<RoomPtr, RoomId>,
    pub raw_room_id_by_ptr: HashMap<RoomPtr, RoomId>, // Does not replace twin room pointer with corresponding main room pointer
    pub room_idx_by_ptr: HashMap<RoomPtr, RoomGeometryRoomIdx>,
    pub room_idx_by_id: HashMap<RoomId, RoomGeometryRoomIdx>,
    pub toilet_room_idx: usize,
    pub node_tile_coords: HashMap<(RoomId, NodeId), Vec<(usize, usize)>>,
    pub node_coords: HashMap<(RoomId, NodeId), (usize, usize)>,
    pub room_shape: HashMap<RoomId, (usize, usize)>,
    pub area_names: Vec<String>,
    pub area_map_ptrs: Vec<isize>,
    pub tech_description: HashMap<TechId, String>,
    pub tech_dependencies: HashMap<TechId, Vec<TechId>>,
    pub escape_timings: Vec<EscapeTimingRoom>,
    pub start_locations: Vec<StartLocation>,
    pub start_location_id_map: HashMap<(RoomId, NodeId), StartLocationId>,
    pub hub_farms: Vec<(VertexId, Requirement)>,
    pub heat_run_tech_idx: TechIdx, // Cached since it is used frequently in graph traversal, and to avoid needing to store it in every HeatFrames req.
    pub speed_ball_tech_idx: TechIdx, // Cached since it is used frequently in graph traversal, and to avoid needing to store it in every HeatFrames req.
    pub wall_jump_tech_idx: TechIdx,
    pub manage_reserves_tech_idx: TechIdx,
    pub pause_abuse_tech_idx: TechIdx,
    pub mother_brain_defeated_flag_id: usize,
    pub title_screen_data: TitleScreenData,
    pub room_name_font: VariableWidthFont,
    pub reduced_flashing_patch: GlowPatch,
    pub strat_videos: HashMap<(RoomId, StratId), Vec<StratVideo>>,
    pub map_tile_data: Vec<MapTileData>,
    pub area_order: Vec<String>,
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
    strat_name: &'a str,
    to_node_id: NodeId,
    room_heated: bool,
    from_obstacles_bitmask: ObstacleMask,
    obstacles_idx_map: Option<&'a HashMap<String, usize>>,
    unlocks_doors_json: Option<&'a JsonValue>,
    node_implicit_door_unlocks: Option<&'a HashMap<NodeId, bool>>,
    notable_map: Option<&'a HashMap<String, NotableIdx>>,
}

impl GameData {
    fn load_tech(&mut self) -> Result<()> {
        let mut full_tech_json = read_json(&self.sm_json_data_path.join("tech.json"))?;
        ensure!(full_tech_json["techCategories"].is_array());
        full_tech_json["techCategories"].members_mut().find(|x| x["name"] == "Shots").unwrap()["techs"].push(json::object!{
            "id": 10001,
            "name": "canHyperGateShot",
            "techRequires": [],
            "otherRequires": [],
            "note": [
                "The ability to shoot blue & green gates from either side using Hyper Beam during the escape.",
                "This is easy to do; this tech just represents knowing it can be done.",
                "This is based on a randomizer patch applied on all settings,",
                "as in the vanilla game it isn't possible to open green gates using Hyper Beam."
            ]
        })?;
        Self::override_can_awaken_zebes_tech_note(&mut full_tech_json)?;
        for tech_category in full_tech_json["techCategories"].members_mut() {
            ensure!(tech_category["techs"].is_array());
            for tech_json in tech_category["techs"].members() {
                self.load_tech_rec(tech_json)?;
            }
        }
        self.heat_run_tech_idx = *self
            .tech_isv
            .index_by_key
            .get(&TECH_ID_CAN_HEAT_RUN)
            .unwrap();
        self.speed_ball_tech_idx = *self
            .tech_isv
            .index_by_key
            .get(&TECH_ID_CAN_SPEEDBALL)
            .unwrap();
        self.wall_jump_tech_idx = *self
            .tech_isv
            .index_by_key
            .get(&TECH_ID_CAN_WALLJUMP)
            .unwrap();
        self.manage_reserves_tech_idx = *self
            .tech_isv
            .index_by_key
            .get(&TECH_ID_CAN_MANAGE_RESERVES)
            .unwrap();
        self.pause_abuse_tech_idx = *self
            .tech_isv
            .index_by_key
            .get(&TECH_ID_CAN_PAUSE_ABUSE)
            .unwrap();
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
        if tech_json["id"].as_i64().is_none() {
            error!("Tech {} missing ID", tech_json["name"]);
            return Ok(());
        }
        let id = tech_json["id"].as_i64().unwrap() as TechId;
        self.tech_isv.add(&id);

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

        self.tech_description.insert(id, desc);
        self.tech_json_map.insert(id, tech_json.clone());
        let tech_name = tech_json["name"].as_str().unwrap().to_string();
        self.tech_names.insert(id, tech_name.clone());
        self.tech_id_by_name.insert(tech_name.clone(), id);
        if tech_json.has_key("extensionTechs") {
            ensure!(tech_json["extensionTechs"].is_array());
            for ext_tech in tech_json["extensionTechs"].members() {
                self.load_tech_rec(ext_tech)?;
            }
        }
        Ok(())
    }

    fn extract_tech_dependencies(&self, req: &Requirement) -> HashSet<TechId> {
        match req {
            Requirement::Tech(tech_idx) => {
                vec![self.tech_isv.keys[*tech_idx]].into_iter().collect()
            }
            Requirement::And(sub_reqs) => {
                let mut out: HashSet<TechId> = HashSet::new();
                for r in sub_reqs {
                    out.extend(self.extract_tech_dependencies(r));
                }
                out
            }
            _ => HashSet::new(),
        }
    }

    fn get_tech_requirement(
        &mut self,
        tech_name: &str,
        include_other_requires: bool,
    ) -> Result<Requirement> {
        let tech_id = self.tech_id_by_name[tech_name];
        if let Some(req_opt) = self
            .tech_requirement
            .get(&(tech_id, include_other_requires))
        {
            if let Some(req) = req_opt {
                return Ok(req.clone());
            } else {
                bail!("Circular dependence in tech: {} ({})", tech_name, tech_id);
            }
        }

        // Temporarily insert a None value to act as a sentinel for detecting circular dependencies:
        self.tech_requirement
            .insert((tech_id, include_other_requires), None);

        let tech_json = &self
            .tech_json_map
            .get(&tech_id)
            .context(format!("Tech not found: {tech_name} ({tech_id})"))?
            .clone();

        let mut reqs: Vec<Requirement> = vec![if tech_id == TECH_ID_CAN_WALLJUMP {
            Requirement::Walljump
        } else {
            Requirement::Tech(self.tech_isv.index_by_key[&tech_id])
        }];
        let ctx = RequirementContext::default();
        ensure!(tech_json["techRequires"].is_array());
        for req in tech_json["techRequires"].members() {
            if req.is_string() {
                reqs.push(
                    self.get_tech_requirement(req.as_str().unwrap(), include_other_requires)
                        .context(format!("Parsing tech requirement '{req}'"))?,
                );
            } else if req.has_key("tech") {
                reqs.push(
                    self.get_tech_requirement(req["tech"].as_str().unwrap(), false)
                        .context(format!("Parsing pure tech requirement '{req}'"))?,
                );
            } else {
                bail!("Unexpected requirement type in techRequires: {}", req);
            }
        }
        if include_other_requires {
            reqs.extend(
                self.parse_requires_list(tech_json["otherRequires"].members().as_slice(), &ctx)?,
            );
        }
        let combined_req = Requirement::make_and(reqs);
        *self
            .tech_requirement
            .get_mut(&(tech_id, include_other_requires))
            .unwrap() = Some(combined_req.clone());
        Ok(combined_req)
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
            self.weapon_json_map
                .insert(name.to_string(), weapon_json.clone());
            self.weapon_isv.add(name);
            let mut categories: Vec<String> = weapon_json["categories"]
                .members()
                .map(|x| x.as_str().unwrap().to_string())
                .collect();
            categories.push(name.to_string());
            for category in categories {
                if !self.weapon_categories.contains_key(&category) {
                    self.weapon_categories.insert(category.clone(), vec![]);
                }
                self.weapon_categories
                    .get_mut(&category)
                    .unwrap()
                    .push(name.to_string());
            }
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
        // Overridden enemy drop rates for buffed drop QoL:
        // (enemy ID, small energy, big energy, missiles, nothing, supers, power bombs)
        // Covern buff is ignored, to reduce the loss-of-access issue when power is on.
        let buffed_drop_overrides = vec![
            (83, 0x3C, 0x3C, 0x32, 0x05, 0x3C, 0x14), // Gamet
            (23, 0x14, 0x41, 0x1E, 0x00, 0x78, 0x14), // Zeb
            (30, 0x14, 0x41, 0x1E, 0x00, 0x78, 0x14), // Geega
            (24, 0x00, 0x8C, 0x05, 0x00, 0x64, 0x0A), // Zebbo
            (75, 0x00, 0x64, 0x3C, 0x05, 0x46, 0x14), // Zoa
            // (51, 0x32, 0x5F, 0x32, 0x00, 0x14, 0x28), // Covern
            (8, 0x23, 0x5F, 0x3C, 0x05, 0x28, 0x14), // Kago
        ];
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
                let vul = self.get_enemy_vulnerabilities(enemy_json)?;
                self.enemy_vulnerabilities
                    .insert(enemy_name.to_string(), vul);
                self.enemy_json
                    .insert(enemy_name.to_string(), enemy_json.clone());

                let mut enemy_json_buffed = enemy_json.clone();
                for &(enemy_id, small, big, missiles, nothing, supers, pbs) in
                    &buffed_drop_overrides
                {
                    if enemy_id == enemy_json["id"].as_usize().unwrap() {
                        enemy_json_buffed["drops"] = json::object! {
                            "noDrop": nothing,
                            "smallEnergy": small,
                            "bigEnergy": big,
                            "missile": missiles,
                            "super": supers,
                            "powerBomb": pbs,
                        };
                    }
                }
                self.enemy_json_buffed
                    .insert(enemy_name.to_string(), enemy_json_buffed.clone());
            }
        }
        Ok(())
    }

    fn get_enemy_damage_multiplier(&self, enemy_json: &JsonValue, weapon_name: &str) -> f32 {
        for multiplier in enemy_json["damageMultipliers"].members() {
            let category = multiplier["weapon"].as_str().unwrap();
            if !self.weapon_categories.contains_key(category) {
                error!(
                    "Weapon category '{}' not found, in enemy JSON: {}",
                    category,
                    enemy_json.pretty(2)
                );
            }
            if self.weapon_categories[category].contains(&weapon_name.to_string()) {
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
    ) -> Capacity {
        let multiplier = self.get_enemy_damage_multiplier(enemy_json, weapon_name);
        let weapon_idx = self.weapon_isv.index_by_key[weapon_name];
        if vul_mask & (1 << weapon_idx) == 0 {
            return 0;
        }
        match weapon_name {
            "Missile" => (100.0 * multiplier) as Capacity,
            "Super" => (300.0 * multiplier) as Capacity,
            "PowerBomb" => (400.0 * multiplier) as Capacity,
            _ => panic!("Unsupported weapon: {weapon_name}"),
        }
    }

    fn get_enemy_vulnerabilities(&self, enemy_json: &JsonValue) -> Result<EnemyVulnerabilities> {
        ensure!(enemy_json["invul"].is_array());
        let invul: HashSet<String> = enemy_json["invul"]
            .members()
            .map(|x| x.to_string())
            .collect();
        let mut vul_mask = 0;
        'weapon: for (i, weapon_name) in self.weapon_isv.keys.iter().enumerate() {
            let weapon_json = &self.weapon_json_map[weapon_name];
            if invul.contains(weapon_name) {
                continue;
            }
            if weapon_json["situational"].as_bool().unwrap() {
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
            hp: enemy_json["hp"].as_i32().unwrap() as Capacity,
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
                if self
                    .helper_json_map
                    .contains_key(helper["name"].as_str().unwrap())
                {
                    bail!(
                        "Duplicate helper definition: {}",
                        helper["name"].as_str().unwrap()
                    );
                }
                self.helper_json_map
                    .insert(helper["name"].as_str().unwrap().to_owned(), helper.clone());
                self.helper_category_map.insert(
                    helper["name"].as_str().unwrap().to_owned(),
                    category_json["name"].as_str().unwrap().to_owned(),
                );
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
            self.parse_requires_list(json_value["requires"].members().as_slice(), &ctx)?,
        );
        *self.helpers.get_mut(name).unwrap() = Some(req.clone());
        Ok(req)
    }

    fn get_unlocks_door_type_req(
        &mut self,
        door_type: DoorType,
        node_id: NodeId,
        ctx: &RequirementContext,
    ) -> Result<Requirement> {
        let unlock_methods = match door_type {
            DoorType::Blue => vec![],
            DoorType::Red => vec![
                (
                    vec!["missiles", "ammo"],
                    Requirement::Missiles(5),
                    Requirement::HeatFrames(50),
                ),
                (
                    vec!["super", "ammo"],
                    Requirement::Supers(1),
                    Requirement::Free,
                ),
            ],
            DoorType::Green => vec![(
                vec!["super", "ammo"],
                Requirement::Supers(1),
                Requirement::Free,
            )],
            DoorType::Yellow => vec![(
                vec!["powerbomb", "ammo"],
                Requirement::make_and(vec![
                    Requirement::Item(Item::Morph as ItemId),
                    Requirement::PowerBombs(1),
                ]),
                Requirement::HeatFrames(110),
            )],
            DoorType::Beam(BeamType::Charge) => vec![(
                vec!["charge"],
                Requirement::Item(Item::Charge as ItemId),
                Requirement::HeatFrames(60),
            )],
            DoorType::Gray | DoorType::Beam(_) | DoorType::Wall => {
                panic!("Unexpected DoorType in get_unlocks_door_type_req: {door_type:?}")
            }
        };
        let room_id = ctx.room_id;
        let empty_array = json::array![];
        let unlocks_doors_json = ctx.unlocks_doors_json.unwrap_or(&empty_array);

        ensure!(unlocks_doors_json.is_array());

        // Disallow using "unlocksDoors" inside its own requirements, to avoid an infinite recursion.
        // TODO: Figure out how to more properly handle "resetRoom" requirements inside of "unlocksDoors".
        let mut ctx1 = ctx.clone();
        ctx1.unlocks_doors_json = None;

        let mut reqs_list = vec![];
        for (keys, implicit_req, heat_req) in unlock_methods {
            let mut req: Option<Requirement> = None;
            for key in keys {
                for u in unlocks_doors_json.members() {
                    if u["nodeId"].as_usize().unwrap_or(ctx.to_node_id) == node_id {
                        ensure!(u["types"].is_array());
                        if u["types"].members().any(|t| t == key) {
                            if req.is_some() {
                                bail!(
                                    "Overlapping unlocksDoors for '{}', room_id={}, node_id={}: {:?}",
                                    key,
                                    room_id,
                                    node_id,
                                    unlocks_doors_json
                                );
                            }
                            let requires: &[JsonValue] = u["requires"].members().as_slice();
                            let mut req_list = self.parse_requires_list(requires, &ctx1)?;
                            if u["useImplicitRequires"].as_bool().unwrap_or(true) {
                                req_list.push(implicit_req.clone());
                            }
                            req = Some(Requirement::make_and(req_list));
                        }
                    }
                }
            }
            if let Some(req) = req {
                reqs_list.push(req);
            } else if ctx.node_implicit_door_unlocks.unwrap()[&node_id] {
                if ctx.room_heated {
                    reqs_list.push(Requirement::make_and(vec![
                        implicit_req.clone(),
                        heat_req.clone(),
                    ]));
                } else {
                    reqs_list.push(implicit_req.clone())
                }
            }
        }
        Ok(Requirement::make_or(reqs_list))
    }

    fn get_unlocks_doors_req(
        &mut self,
        node_id: NodeId,
        ctx: &RequirementContext,
    ) -> Result<Requirement> {
        if let Some(grey_unlock_req_json) = self.grey_lock_map.get(&(ctx.room_id, node_id)).cloned()
        {
            let grey_unlock_req = self.parse_requirement(&grey_unlock_req_json, ctx)?;
            return Ok(grey_unlock_req);
        }
        Ok(Requirement::UnlockDoor {
            room_id: ctx.room_id,
            node_id,
            requirement_red: Box::new(self.get_unlocks_door_type_req(
                DoorType::Red,
                node_id,
                ctx,
            )?),
            requirement_green: Box::new(self.get_unlocks_door_type_req(
                DoorType::Green,
                node_id,
                ctx,
            )?),
            requirement_yellow: Box::new(self.get_unlocks_door_type_req(
                DoorType::Yellow,
                node_id,
                ctx,
            )?),
            requirement_charge: Box::new(self.get_unlocks_door_type_req(
                DoorType::Beam(BeamType::Charge),
                node_id,
                ctx,
            )?),
        })
    }

    fn parse_requires_list(
        &mut self,
        req_jsons: &[JsonValue],
        ctx: &RequirementContext,
    ) -> Result<Vec<Requirement>> {
        let mut reqs: Vec<Requirement> = Vec::new();
        for req_json in req_jsons {
            reqs.push(
                self.parse_requirement(req_json, ctx)
                    .with_context(|| format!("Processing requirement {req_json}"))?,
            );
        }
        Ok(reqs)
    }

    fn parse_enemy_drops(&self, value: &JsonValue, buffed: bool) -> Vec<EnemyDrop> {
        let mut enemy_drops = vec![];
        assert!(value.is_array());
        for drop in value.members() {
            let enemy_name = drop["enemy"].as_str().unwrap();
            let enemy_json = if buffed {
                &self
                    .enemy_json_buffed
                    .get(enemy_name)
                    .unwrap_or_else(|| panic!("Unknown enemy: {}", enemy_name))
            } else {
                &self
                    .enemy_json
                    .get(enemy_name)
                    .unwrap_or_else(|| panic!("Unknown enemy: {}", enemy_name))
            };
            let drops_json = &enemy_json["drops"];
            let amount_of_drops = enemy_json["amountOfDrops"].as_isize().unwrap() as Capacity;
            let count = drop["count"].as_i32().unwrap() as Capacity;
            let nothing_weight = drops_json["noDrop"]
                .as_f32()
                .context(format!("missing noDrop for {enemy_name}"))
                .unwrap();
            let small_energy_weight = drops_json["smallEnergy"]
                .as_f32()
                .context(format!("missing smallEnergy for {enemy_name}"))
                .unwrap();
            let large_energy_weight = drops_json["bigEnergy"]
                .as_f32()
                .context(format!("missing bigEnergy for {enemy_name}"))
                .unwrap();
            let missile_weight = drops_json["missile"]
                .as_f32()
                .context(format!("missing missile for {enemy_name}"))
                .unwrap();
            let super_weight = drops_json["super"]
                .as_f32()
                .context(format!("missing super for {enemy_name}"))
                .unwrap();
            let power_bomb_weight = drops_json["powerBomb"]
                .as_f32()
                .context(format!("missing powerBomb for {enemy_name}"))
                .unwrap();
            let total_weight = nothing_weight
                + small_energy_weight
                + large_energy_weight
                + missile_weight
                + super_weight
                + power_bomb_weight;
            let enemy_drop = EnemyDrop {
                nothing_weight: Float::new(nothing_weight / total_weight),
                small_energy_weight: Float::new(small_energy_weight / total_weight),
                large_energy_weight: Float::new(large_energy_weight / total_weight),
                missile_weight: Float::new(missile_weight / total_weight),
                super_weight: Float::new(super_weight / total_weight),
                power_bomb_weight: Float::new(power_bomb_weight / total_weight),
                count: count * amount_of_drops,
            };
            enemy_drops.push(enemy_drop);
        }
        enemy_drops
    }

    // TODO: Find some way to have this not need to be mutable (e.g. resolve the helper dependencies in a first pass)
    #[allow(clippy::if_same_then_else)]
    fn parse_requirement(
        &mut self,
        req_json: &JsonValue,
        ctx: &RequirementContext,
    ) -> Result<Requirement> {
        if req_json.is_string() {
            let value = req_json.as_str().unwrap();
            if value == "never" {
                return Ok(Requirement::Never);
            } else if value == "free" {
                // Defined for internal use in the randomizer
                return Ok(Requirement::Free);
            } else if value == "canWalljump" {
                return Ok(Requirement::Walljump);
            } else if value == "i_ClimbWithoutLava" {
                return Ok(Requirement::ClimbWithoutLava);
            } else if value == "i_ammoRefill" {
                return Ok(Requirement::AmmoStationRefill);
            } else if value == "i_ammoRefillAll" {
                return Ok(Requirement::AmmoStationRefillAll);
            } else if value == "i_energyStationRefill" {
                return Ok(Requirement::EnergyStationRefill);
            } else if value == "i_SupersDoubleDamageMotherBrain" {
                return Ok(Requirement::SupersDoubleDamageMotherBrain);
            } else if value == "i_blueGateGlitchLeniency" {
                return Ok(Requirement::GateGlitchLeniency {
                    green: false,
                    heated: false,
                });
            } else if value == "i_greenGateGlitchLeniency" {
                return Ok(Requirement::GateGlitchLeniency {
                    green: true,
                    heated: false,
                });
            } else if value == "i_heatedBlueGateGlitchLeniency" {
                return Ok(Requirement::GateGlitchLeniency {
                    green: false,
                    heated: true,
                });
            } else if value == "i_heatedGreenGateGlitchLeniency" {
                return Ok(Requirement::GateGlitchLeniency {
                    green: true,
                    heated: true,
                });
            } else if value == "i_elevatorCrystalFlashLeniency" {
                return Ok(Requirement::ElevatorCFLeniency);
            } else if value == "i_bombIntoCrystalFlashClipLeniency" {
                return Ok(Requirement::BombIntoCrystalFlashClipLeniency {});
            } else if value == "i_jumpIntoCrystalFlashClipLeniency" {
                return Ok(Requirement::JumpIntoCrystalFlashClipLeniency {});
            } else if value == "i_spikeSuitSpikeHitLeniency" {
                return Ok(Requirement::SpikeSuitSpikeHitLeniency {});
            } else if value == "i_spikeSuitThornHitLeniency" {
                return Ok(Requirement::SpikeSuitThornHitLeniency {});
            } else if value == "i_spikeSuitSamusEaterLeniency" {
                return Ok(Requirement::SpikeSuitSamusEaterLeniency {});
            } else if value == "i_spikeSuitPowerBombLeniency" {
                return Ok(Requirement::SpikeSuitPowerBombLeniency {});
            } else if value == "i_XModeSpikeHitLeniency" {
                return Ok(Requirement::XModeSpikeHitLeniency {});
            } else if value == "i_XModeThornHitLeniency" {
                return Ok(Requirement::XModeThornHitLeniency {});
            } else if value == "i_FramePerfectXModeThornHitLeniency" {
                return Ok(Requirement::FramePerfectXModeThornHitLeniency);
            } else if value == "i_FramePerfectDoubleXModeThornHitLeniency" {
                return Ok(Requirement::FramePerfectDoubleXModeThornHitLeniency {});
            } else if value == "i_speedKeepSpikeHitLeniency" {
                return Ok(Requirement::SpeedKeepSpikeHitLeniency);
            } else if value == "i_MotherBrainBarrier1Clear" {
                return Ok(Requirement::MotherBrainBarrierClear(0));
            } else if value == "i_MotherBrainBarrier2Clear" {
                return Ok(Requirement::MotherBrainBarrierClear(1));
            } else if value == "i_MotherBrainBarrier3Clear" {
                return Ok(Requirement::MotherBrainBarrierClear(2));
            } else if value == "i_MotherBrainBarrier4Clear" {
                return Ok(Requirement::MotherBrainBarrierClear(3));
            } else if value == "i_LowerNorfairElevatorDownwardFrames" {
                return Ok(Requirement::LowerNorfairElevatorDownFrames);
            } else if value == "i_LowerNorfairElevatorUpwardFrames" {
                return Ok(Requirement::LowerNorfairElevatorUpFrames);
            } else if value == "i_MainHallElevatorFrames" {
                return Ok(Requirement::MainHallElevatorFrames);
            } else if value == "i_equipmentScreenCycleFrames" {
                return Ok(Requirement::EquipmentScreenCycleFrames);
            } else if value == "i_ShinesparksCostEnergy" {
                return Ok(Requirement::ShinesparksCostEnergy);
            } else if value == "i_canEscapeMorphLocation" {
                return Ok(Requirement::EscapeMorphLocation);
            } else if let Some(&item_id) = self.item_isv.index_by_key.get(value) {
                return Ok(Requirement::Item(item_id as ItemId));
            } else if let Some(&flag_id) = self.flag_isv.index_by_key.get(value) {
                return Ok(Requirement::Flag(flag_id as FlagId));
            } else if self.tech_id_by_name.contains_key(value) {
                return self.get_tech_requirement(value, true);
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
                    panic!("Unrecognized flag in 'not': {value}");
                }
            } else if key == "ammo" {
                let ammo_type = value["type"]
                    .as_str()
                    .unwrap_or_else(|| panic!("missing/invalid ammo type in {req_json}"));
                let count = value["count"]
                    .as_i32()
                    .unwrap_or_else(|| panic!("missing/invalid ammo count in {req_json}"));
                if ammo_type == "Missile" {
                    return Ok(Requirement::Missiles(count as Capacity));
                } else if ammo_type == "Super" {
                    return Ok(Requirement::Supers(count as Capacity));
                } else if ammo_type == "PowerBomb" {
                    return Ok(Requirement::PowerBombs(count as Capacity));
                } else {
                    bail!("Unexpected ammo type in {}", req_json);
                }
            } else if key == "resourceAvailable" {
                let mut reqs = vec![];
                ensure!(value.is_array());
                for value0 in value.members() {
                    let resource_type = value0["type"]
                        .as_str()
                        .unwrap_or_else(|| panic!("missing/invalid resource type in {req_json}"));
                    let count = value0["count"]
                        .as_i32()
                        .unwrap_or_else(|| panic!("missing/invalid resource count in {req_json}"));
                    if resource_type == "Missile" {
                        reqs.push(Requirement::MissilesAvailable(count as Capacity));
                    } else if resource_type == "Super" {
                        reqs.push(Requirement::SupersAvailable(count as Capacity));
                    } else if resource_type == "PowerBomb" {
                        reqs.push(Requirement::PowerBombsAvailable(count as Capacity));
                    } else if resource_type == "RegularEnergy" {
                        reqs.push(Requirement::RegularEnergyAvailable(count as Capacity));
                    } else if resource_type == "ReserveEnergy" {
                        reqs.push(Requirement::ReserveEnergyAvailable(count as Capacity));
                    } else if resource_type == "Energy" {
                        reqs.push(Requirement::EnergyAvailable(count as Capacity));
                    } else {
                        bail!("Unexpected resource type in {}", req_json);
                    }
                }
                return Ok(Requirement::make_and(reqs));
            } else if key == "resourceCapacity" {
                ensure!(value.members().len() == 1);
                let value0 = value.members().next().unwrap();
                let resource_type = value0["type"]
                    .as_str()
                    .unwrap_or_else(|| panic!("missing/invalid resource type in {req_json}"));
                let count = value0["count"]
                    .as_i32()
                    .unwrap_or_else(|| panic!("missing/invalid resource count in {req_json}"));
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
                    bail!("Unexpected resource type in {}", req_json);
                }
            } else if key == "resourceMaxCapacity" {
                ensure!(value.members().len() == 1);
                let value0 = value.members().next().unwrap();
                let resource_type = value0["type"]
                    .as_str()
                    .unwrap_or_else(|| panic!("missing/invalid resource type in {req_json}"));
                let count = value0["count"]
                    .as_i32()
                    .unwrap_or_else(|| panic!("missing/invalid resource count in {req_json}"));
                if resource_type == "RegularEnergy" {
                    return Ok(Requirement::make_and(vec![
                        Requirement::DisableableETank,
                        Requirement::RegularEnergyDrain(count as Capacity),
                    ]));
                } else if resource_type == "Missile" {
                    // TODO: change this if we add an ability to disable Missiles.
                    return Ok(Requirement::Never);
                } else {
                    bail!("Unexpected resource type in {}", req_json);
                }
            } else if key == "resourceAtMost" {
                let mut reqs: Vec<Requirement> = vec![];
                for r in value.members() {
                    let resource_type = r["type"]
                        .as_str()
                        .unwrap_or_else(|| panic!("missing/invalid resource type in {req_json}"));
                    let count = r["count"]
                        .as_i32()
                        .unwrap_or_else(|| panic!("missing/invalid resource count in {req_json}"));
                    if resource_type == "RegularEnergy" {
                        assert!(count > 0);
                        reqs.push(Requirement::RegularEnergyDrain(count as Capacity));
                    } else if resource_type == "Energy" {
                        // TODO: remove this case or handle it properly.
                        reqs.push(Requirement::RegularEnergyDrain(count as Capacity));
                    } else if resource_type == "ReserveEnergy" {
                        reqs.push(Requirement::ReserveEnergyDrain(count as Capacity));
                    } else if resource_type == "Missile" {
                        reqs.push(Requirement::MissileDrain(count as Capacity));
                    } else {
                        bail!("Unexpected resource type in {}", req_json);
                    }
                }
                return Ok(Requirement::make_and(reqs));
            } else if key == "resourceMissingAtMost" {
                let mut reqs: Vec<Requirement> = vec![];
                for r in value.members() {
                    let resource_type = r["type"]
                        .as_str()
                        .unwrap_or_else(|| panic!("missing/invalid resource type in {req_json}"));
                    let count = r["count"]
                        .as_i32()
                        .unwrap_or_else(|| panic!("missing/invalid resource count in {req_json}"));
                    if resource_type == "Missile" {
                        reqs.push(Requirement::MissilesMissingAtMost(count as Capacity));
                    } else if resource_type == "Super" {
                        reqs.push(Requirement::SupersMissingAtMost(count as Capacity));
                    } else if resource_type == "PowerBomb" {
                        reqs.push(Requirement::PowerBombsMissingAtMost(count as Capacity));
                    } else if resource_type == "RegularEnergy" {
                        reqs.push(Requirement::RegularEnergyMissingAtMost(count as Capacity));
                    } else if resource_type == "ReserveEnergy" {
                        reqs.push(Requirement::ReserveEnergyMissingAtMost(count as Capacity));
                    } else if resource_type == "Energy" {
                        reqs.push(Requirement::EnergyMissingAtMost(count as Capacity));
                    } else {
                        bail!("Unexpected resource type in {}", req_json);
                    }
                }
                return Ok(Requirement::make_and(reqs));
            } else if key == "resourceConsumed" {
                let mut reqs = vec![];
                ensure!(value.is_array());
                for value0 in value.members() {
                    let resource_type = value0["type"]
                        .as_str()
                        .unwrap_or_else(|| panic!("missing/invalid resource type in {req_json}"));
                    let count = value0["count"]
                        .as_i32()
                        .unwrap_or_else(|| panic!("missing/invalid resource count in {req_json}"));
                    if resource_type == "RegularEnergy" {
                        reqs.push(Requirement::RegularEnergy(count as Capacity));
                    } else if resource_type == "ReserveEnergy" {
                        reqs.push(Requirement::ReserveEnergy(count as Capacity));
                    } else if resource_type == "Energy" {
                        reqs.push(Requirement::Energy(count as Capacity));
                    } else {
                        bail!("Unexpected resource type in {}", req_json);
                    }
                }
                return Ok(Requirement::make_and(reqs));
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
                    Requirement::MissileRefill(limit as Capacity)
                } else if resource_type == "Super" {
                    Requirement::SuperRefill(limit as Capacity)
                } else if resource_type == "PowerBomb" {
                    Requirement::PowerBombRefill(limit as Capacity)
                } else if resource_type == "RegularEnergy" {
                    Requirement::RegularEnergyRefill(limit as Capacity)
                } else if resource_type == "ReserveEnergy" {
                    Requirement::ReserveRefill(limit as Capacity)
                } else if resource_type == "Energy" {
                    Requirement::EnergyRefill(limit as Capacity)
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
                    .unwrap_or_else(|| panic!("missing/invalid frames in {req_json}"))
                    as Capacity;
                let excess_frames = value["excessFrames"].as_i32().unwrap_or(0) as Capacity;
                return Ok(Requirement::Shinespark {
                    shinespark_tech_idx: self.tech_isv.index_by_key[&TECH_ID_CAN_SHINESPARK],
                    frames,
                    excess_frames,
                });
            } else if key == "getBlueSpeed" {
                let runway_geometry = parse_runway_geometry_shinecharge(value)?;
                let effective_length = compute_runway_effective_length(&runway_geometry);
                return Ok(Requirement::make_blue_speed(
                    effective_length,
                    ctx.room_heated,
                ));
            } else if key == "speedBall" {
                let runway_geometry = parse_runway_geometry(value)?;
                let effective_length = compute_runway_effective_length(&runway_geometry);
                return Ok(Requirement::SpeedBall {
                    used_tiles: Float::new(effective_length),
                    heated: ctx.room_heated,
                });
            } else if key == "canShineCharge" {
                let runway_geometry = parse_runway_geometry_shinecharge(value)?;
                let effective_length = compute_runway_effective_length(&runway_geometry);
                return Ok(Requirement::make_shinecharge(
                    effective_length,
                    ctx.room_heated,
                ));
            } else if key == "shineChargeFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid shineChargeFrames in {req_json}"));
                return Ok(Requirement::ShineChargeFrames(frames as Capacity));
            } else if key == "heatFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid heatFrames in {req_json}"));
                return Ok(Requirement::HeatFrames(frames as Capacity));
            } else if key == "suitlessHeatFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid suitlessHeatFrames in {req_json}"));
                return Ok(Requirement::SuitlessHeatFrames(frames as Capacity));
            } else if key == "simpleHeatFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid simpleHeatFrames in {req_json}"));
                return Ok(Requirement::SimpleHeatFrames(frames as Capacity));
            } else if key == "gravitylessHeatFrames" {
                // In Map Rando, Gravity doesn't affect heat frames, so this is treated the same as "heatFrames".
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid gravitylessHeatFrames in {req_json}"));
                return Ok(Requirement::HeatFrames(frames as Capacity));
            } else if key == "lavaFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid lavaFrames in {req_json}"));
                return Ok(Requirement::LavaFrames(frames as Capacity));
            } else if key == "gravitylessLavaFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid gravitylessLavaFrames in {req_json}"));
                return Ok(Requirement::GravitylessLavaFrames(frames as Capacity));
            } else if key == "gravitylessAcidFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid gravitylessAcidFrames in {req_json}"));
                return Ok(Requirement::GravitylessAcidFrames(frames as Capacity));
            } else if key == "acidFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid acidFrames in {req_json}"));
                return Ok(Requirement::AcidFrames(frames as Capacity));
            } else if key == "metroidFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid metroidFrames in {req_json}"));
                return Ok(Requirement::MetroidFrames(frames as Capacity));
            } else if key == "draygonElectricityFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid draygonElectricityFrames in {req_json}"));
                return Ok(Requirement::Damage(frames as Capacity));
            } else if key == "samusEaterCycles" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid samusEaterCycles in {req_json}"));
                return Ok(Requirement::Damage(frames as Capacity * 16));
            } else if key == "cycleFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid cycleFrames in {req_json}"));
                return Ok(Requirement::CycleFrames(frames as Capacity));
            } else if key == "simpleCycleFrames" {
                let frames = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid simpleCycleFrames in {req_json}"));
                return Ok(Requirement::SimpleCycleFrames(frames as Capacity));
            } else if key == "spikeHits" {
                let hits = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid spikeHits in {req_json}"));
                return Ok(Requirement::Damage(hits as Capacity * 60));
            } else if key == "thornHits" {
                let hits = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid thornHits in {req_json}"));
                return Ok(Requirement::Damage(hits as Capacity * 16));
            } else if key == "electricityHits" {
                let hits = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid electricityHits in {req_json}"));
                return Ok(Requirement::Damage(hits as Capacity * 30));
            } else if key == "hibashiHits" {
                let hits = value
                    .as_i32()
                    .unwrap_or_else(|| panic!("invalid hibashiHits in {req_json}"));
                return Ok(Requirement::Damage(hits as Capacity * 30));
            } else if key == "enemyDamage" {
                let enemy_name = value["enemy"].as_str().unwrap().to_string();
                let attack_name = value["type"].as_str().unwrap().to_string();
                let hits = value["hits"].as_i32().unwrap() as Capacity;
                let base_damage = self
                    .enemy_attack_damage
                    .get(&(enemy_name.clone(), attack_name.clone()))
                    .with_context(|| {
                        format!("Missing enemy attack damage for {enemy_name} - {attack_name}:")
                    })?;
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
                        can_be_patient_tech_idx: self.tech_isv.index_by_key
                            [&TECH_ID_CAN_BE_PATIENT],
                        can_be_very_patient_tech_idx: self.tech_isv.index_by_key
                            [&TECH_ID_CAN_BE_VERY_PATIENT],
                        can_be_extremely_patient_tech_idx: self.tech_isv.index_by_key
                            [&TECH_ID_CAN_BE_EXTREMELY_PATIENT],
                    });
                } else if enemy_set.contains("Botwoon 1") {
                    return Ok(Requirement::BotwoonFight {
                        second_phase: false,
                    });
                } else if enemy_set.contains("Botwoon 2") {
                    return Ok(Requirement::BotwoonFight { second_phase: true });
                } else if enemy_set.contains("Mother Brain 2") {
                    // Here we check the first obstacle ("A") to see if we're in R-mode.
                    // Also we only want the R-mode logic to take effect with the R-Mode strats;
                    // currently, we do this in a hacky way by checking the strat name.
                    // TODO: Make logical requirements for boss fights, so that we could express the requirements
                    // properly in the sm-json-data, e.g. like {"fightMotherBrain": {"rMode": true}}.
                    let r_mode =
                        (ctx.from_obstacles_bitmask & 1) == 1 && ctx.strat_name.contains("R-Mode");
                    return Ok(Requirement::MotherBrain2Fight {
                        can_be_very_patient_tech_id: self.tech_isv.index_by_key
                            [&TECH_ID_CAN_BE_VERY_PATIENT],
                        r_mode,
                    });
                }

                let mut allowed_weapons: WeaponMask = if value.has_key("explicitWeapons") {
                    ensure!(value["explicitWeapons"].is_array());
                    let mut weapon_mask = 0;
                    for weapon_name in value["explicitWeapons"].members() {
                        weapon_mask |=
                            1 << self.weapon_isv.index_by_key[weapon_name.as_str().unwrap()];
                    }
                    weapon_mask
                } else {
                    (1 << self.weapon_isv.keys.len()) - 1
                };
                if value.has_key("excludedWeapons") {
                    ensure!(value["excludedWeapons"].is_array());
                    for weapon_name in value["excludedWeapons"].members() {
                        allowed_weapons &=
                            !(1 << self.weapon_isv.index_by_key[weapon_name.as_str().unwrap()]);
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
                        if allowed_weapons
                            & (1 << self.weapon_isv.index_by_key["PowerBombPeriphery"])
                            == 0
                        {
                            vul.power_bomb_damage = 0;
                        } else {
                            vul.power_bomb_damage /= 2;
                        }
                    }
                    reqs.push(Requirement::EnemyKill {
                        count: *count as Capacity,
                        vul,
                    });
                }
                return Ok(Requirement::make_and(reqs));
            } else if key == "ridleyKill" {
                let power_bombs = value["powerBombs"].as_bool().unwrap_or(false);
                let g_mode = value["gMode"].as_bool().unwrap_or(false);
                let stuck = match value["stuck"].as_str() {
                    None => RidleyStuck::None,
                    Some("top") => RidleyStuck::Top,
                    Some("bottom") => RidleyStuck::Bottom,
                    _ => panic!(
                        "unexpected ridleyFight `stuck` value: {}",
                        value["stuck"].as_str().unwrap()
                    ),
                };
                return Ok(Requirement::RidleyFight {
                    can_be_patient_tech_idx: self.tech_isv.index_by_key[&TECH_ID_CAN_BE_PATIENT],
                    can_be_very_patient_tech_idx: self.tech_isv.index_by_key
                        [&TECH_ID_CAN_BE_VERY_PATIENT],
                    can_be_extremely_patient_tech_idx: self.tech_isv.index_by_key
                        [&TECH_ID_CAN_BE_EXTREMELY_PATIENT],
                    power_bombs,
                    g_mode,
                    stuck,
                });
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
                let mut node_ids: Vec<NodeId> = Vec::new();
                for from_node in value["nodes"].members() {
                    node_ids.push(from_node.as_usize().unwrap());
                }
                let mut reqs_or: Vec<Requirement> = vec![];
                for node_id in node_ids {
                    reqs_or.push(Requirement::make_and(vec![
                        self.get_unlocks_doors_req(node_id, ctx)?,
                        Requirement::ResetRoom {
                            room_id: ctx.room_id,
                            node_id,
                        },
                    ]));
                }
                return Ok(Requirement::make_or(reqs_or));
            } else if key == "doorUnlockedAtNode" {
                let node_id = value.as_usize().unwrap();
                return self.get_unlocks_doors_req(node_id, ctx);
            } else if key == "itemNotCollectedAtNode" {
                // TODO: implement this
                return Ok(Requirement::Free);
            } else if key == "itemCollectedAtNode" {
                // TODO: implement this
                return Ok(Requirement::Never);
            } else if key == "autoReserveTrigger" {
                return Ok(Requirement::ReserveTrigger {
                    min_reserve_energy: value["minReserveEnergy"].as_i32().unwrap_or(1) as Capacity,
                    max_reserve_energy: value["maxReserveEnergy"].as_i32().unwrap_or(400)
                        as Capacity,
                    heated: ctx.room_heated,
                });
            } else if key == "gainFlashSuit" {
                return Ok(Requirement::GainFlashSuit);
            } else if key == "noFlashSuit" {
                return Ok(Requirement::NoFlashSuit);
            } else if key == "useFlashSuit" {
                return Ok(Requirement::UseFlashSuit {
                    carry_flash_suit_tech_idx: self.tech_isv.index_by_key
                        [&TECH_ID_CAN_CARRY_FLASH_SUIT],
                });
            } else if key == "gainBlueSuit" {
                // Not yet implemented.
                return Ok(Requirement::Never);
            } else if key == "noBlueSuit" {
                // Not yet implemented.
                return Ok(Requirement::Free);
            } else if key == "haveBlueSuit" {
                // Not yet implemented.
                return Ok(Requirement::Never);
            } else if key == "blueSuitShinecharge" {
                // Not yet implemented.
                return Ok(Requirement::Never);
            } else if key == "tech" {
                return self.get_tech_requirement(value.as_str().unwrap(), false);
            } else if key == "notable" {
                let notable_name = value.as_str().unwrap().to_string();
                let notable_idx = *ctx
                    .notable_map
                    .unwrap()
                    .get(&notable_name)
                    .context(format!("Undefined notable: {}", &notable_name))?;
                return Ok(Requirement::Notable(notable_idx));
            } else if key == "heatFramesWithEnergyDrops" {
                let frames = value["frames"].as_i32().unwrap() as Capacity;
                let enemy_drops = self.parse_enemy_drops(&value["drops"], false);
                let enemy_drops_buffed = self.parse_enemy_drops(&value["drops"], true);
                return Ok(Requirement::HeatFramesWithEnergyDrops(
                    frames,
                    enemy_drops,
                    enemy_drops_buffed,
                ));
            } else if key == "lavaFramesWithEnergyDrops" {
                let frames = value["frames"].as_i32().unwrap() as Capacity;
                let enemy_drops = self.parse_enemy_drops(&value["drops"], false);
                let enemy_drops_buffed = self.parse_enemy_drops(&value["drops"], true);
                return Ok(Requirement::LavaFramesWithEnergyDrops(
                    frames,
                    enemy_drops,
                    enemy_drops_buffed,
                ));
            } else if key == "disableEquipment" {
                let mut reqs = vec![Requirement::Tech(
                    self.tech_isv.index_by_key[&TECH_ID_CAN_DISABLE_EQUIPMENT],
                )];
                if value == "ETank" {
                    reqs.push(Requirement::DisableableETank);
                }
                return Ok(Requirement::make_and(reqs));
            }
        }
        bail!("Unable to parse requirement: {}", req_json);
    }

    pub fn load_rooms(&mut self, room_pattern: &str) -> Result<()> {
        let mut room_json_map: HashMap<usize, JsonValue> = HashMap::new();
        for entry in glob::glob(room_pattern).unwrap() {
            if let Ok(path) = entry {
                let path_str = path.to_str().with_context(|| {
                    format!("Unable to convert path to string: {}", path.display())
                })?;
                if path_str.contains("ceres") || path_str.contains("roomDiagrams") {
                    continue;
                }

                let room_json = read_json(&path)?;
                room_json_map.insert(room_json["id"].as_usize().unwrap(), room_json);
            } else {
                bail!("Error processing region path: {}", entry.err().unwrap());
            }
        }

        let mut room_id_vec: Vec<usize> = room_json_map.keys().cloned().collect();
        room_id_vec.sort();
        for room_id in room_id_vec {
            let room_json = &room_json_map[&room_id];
            let room_name = room_json["name"].clone();
            let preprocessed_room_json = self
                .preprocess_room(room_json)
                .with_context(|| format!("Preprocessing room {room_name}"))?;
            self.process_room(&preprocessed_room_json)
                .with_context(|| format!("Processing room {room_name}"))?;
        }

        self.populate_target_locations()?;

        Ok(())
    }

    fn override_morph_ball_room(&mut self, room_json: &mut JsonValue) {
        // Override the Careful Jump strat to get out from Morph Ball location:
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
                        "i_canEscapeMorphLocation",
                    ]
                }],
            )
            .unwrap();
    }

    fn override_spore_spawn_room(&mut self, room_json: &mut JsonValue) {
        // Add lock on bottom door:
        let mut found = false;
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 2 {
                found = true;
                node_json["nodeSubType"] = "grey".into();
                node_json["locks"] = json::array![
                  {
                    "name": "Spore Spawn Gray Lock",
                    "lockType": "bossFight",
                    "unlockStrats": [
                      {
                        "name": "Base",
                        "notable": false,
                        "requires": ["f_DefeatedSporeSpawn"],
                        "flashSuitChecked": true,
                      }
                    ]
                  }
                ];
            }
        }
        assert!(found);
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
                    json::array!["f_ZebesAwake", "f_ClearedBabyKraidRoom"];
                node_json["locks"][0]["unlockStrats"][0]["requires"] = json::array![
                    {"or": [
                        {"obstaclesCleared": ["A"]},
                        "f_ClearedBabyKraidRoom"
                    ]}
                ];
            }
        }
    }

    fn override_plasma_room(&mut self, room_json: &mut JsonValue) {
        // Add yielded flag "f_ClearedPlasmaRoom" to gray door unlocks:
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 1 {
                node_json["locks"][0]["yields"] =
                    json::array!["f_ZebesAwake", "f_ClearedPlasmaRoom"];
                node_json["locks"][0]["unlockStrats"][0]["requires"] = json::array![
                    {"or": [
                        {"obstaclesCleared": ["A"]},
                        "f_ClearedPlasmaRoom"
                    ]}
                ];
            }
        }
    }

    fn override_metal_pirates_room(&mut self, room_json: &mut JsonValue) {
        // Add yielded flag "f_ClearedMetalPiratesRoom" to gray door unlock:
        for node_json in room_json["nodes"].members_mut() {
            if node_json["id"].as_i32().unwrap() == 1 {
                node_json["locks"][0]["yields"] =
                    json::array!["f_ZebesAwake", "f_ClearedMetalPiratesRoom"];
                node_json["locks"][0]["unlockStrats"][0]["requires"] = json::array![
                    {"or": [
                        {"obstaclesCleared": ["A"]},
                        "f_ClearedMetalPiratesRoom"
                    ]}
                ];
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
                        "requires": [
                            {"or": [
                                {"obstaclesCleared": ["A"]},
                                "f_ClearedMetalPiratesRoom"
                            ]}
                        ],
                        "flashSuitChecked": true,
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
        // or start the MB2 fight.
        // "id: 38": Destroy First Zebetite
        // node 2: right door node
        // node 4: MB2 fight node
        for x in room_json["strats"].members_mut() {
            if x["id"] == 38
                || (x["link"][0].as_i32().unwrap() == 2 && x["link"][1].as_i32().unwrap() != 2)
                || (x["link"][1].as_i32().unwrap() == 4)
            {
                x["requires"].push("i_MotherBrainBarrier1Clear").unwrap();
                x["requires"].push("i_MotherBrainBarrier2Clear").unwrap();
                x["requires"].push("i_MotherBrainBarrier3Clear").unwrap();
                x["requires"].push("i_MotherBrainBarrier4Clear").unwrap();
            }
        }

        for x in room_json["strats"].members_mut() {
            if x["id"].as_i32() == Some(34) {
                // Mother Brain 2 and 3 Fight: override requirements, removing rainbow beam damage requirement,
                // since we already handle this inside the enemyKill requirement
                x["requires"] = json::array![
                    {"enemyKill": {"enemies": [["Mother Brain 2"]]}}
                ];
            }
        }

        let mut new_strats: Vec<JsonValue> = vec![];
        for x in room_json["strats"].members_mut() {
            if x["id"].as_i32() == Some(8) {
                // Leave With Runway: shorten the runway length slightly: the objective barrier creates a closed end
                x["exitCondition"]["leaveWithRunway"]["openEnd"] = JsonValue::Number(0.into());

                let obj_conditions = [
                    "i_MotherBrainBarrier1Clear",
                    "i_MotherBrainBarrier2Clear",
                    "i_MotherBrainBarrier3Clear",
                    "i_MotherBrainBarrier4Clear",
                ];
                for num_objectives_complete in 1..=4 {
                    let mut strat = x.clone();
                    let runway_length = 5 + num_objectives_complete;
                    strat["exitCondition"]["leaveWithRunway"]["length"] =
                        JsonValue::Number(runway_length.into());
                    if num_objectives_complete == 4 {
                        strat["exitCondition"]["leaveWithRunway"]["openEnd"] =
                            JsonValue::Number(1.into());
                    }

                    strat["id"] = JsonValue::Number((10000 + num_objectives_complete).into());
                    if num_objectives_complete == 1 {
                        strat["name"] = JsonValue::String(format!(
                            "{}, 1 Barrier Cleared",
                            x["name"].as_str().unwrap()
                        ));
                    } else {
                        strat["name"] = JsonValue::String(format!(
                            "{}, {} Barriers Cleared",
                            x["name"].as_str().unwrap(),
                            num_objectives_complete
                        ));
                    }
                    for cond in obj_conditions.iter().take(num_objectives_complete) {
                        strat["requires"]
                            .push(JsonValue::String(cond.to_string()))
                            .unwrap();
                    }
                    new_strats.push(strat);
                }
            }
        }
        for strat in new_strats {
            room_json["strats"].push(strat).unwrap();
        }
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
                        "requires": [],
                        "flashSuitChecked": true,
                    }],
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

    fn get_default_unlocks_door(
        &self,
        room_json: &JsonValue,
        node_id: usize,
        to_node_id: usize,
    ) -> Result<JsonValue> {
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

    fn replace_obstacle_flag(req_json: &mut JsonValue, obstacle_flag: &str) {
        if req_json.is_string() {
            let s = req_json.as_str().unwrap();
            if obstacle_flag == s {
                *req_json = json::object! {"or": [s, {"obstaclesCleared": [s]}]};
            }
        } else if req_json.is_array() {
            for x in req_json.members_mut() {
                Self::replace_obstacle_flag(x, obstacle_flag);
            }
        } else if req_json.has_key("and") {
            Self::replace_obstacle_flag(&mut req_json["and"], obstacle_flag);
        } else if req_json.has_key("or") {
            Self::replace_obstacle_flag(&mut req_json["or"], obstacle_flag);
        }
    }

    fn preprocess_room(&mut self, room_json: &JsonValue) -> Result<JsonValue> {
        // We apply some changes to the sm-json-data specific to Map Rando.

        let mut new_room_json = room_json.clone();
        ensure!(room_json["nodes"].is_array());
        let mut extra_strats: Vec<JsonValue> = Vec::new();
        let room_id = room_json["id"].as_usize().unwrap();

        if room_json["name"].as_str().unwrap() == "Upper Tourian Save Room" {
            new_room_json["name"] = JsonValue::String("Tourian Map Room".to_string());
        }

        for strat_json in new_room_json["strats"].members_mut() {
            if strat_json["id"].as_usize().is_none() {
                let from_node_id = strat_json["link"][0].as_usize().unwrap();
                let to_node_id = strat_json["link"][0].as_usize().unwrap();
                warn!(
                    "Skipping strat without ID: {}:{}:{}:{}",
                    room_json["name"],
                    from_node_id,
                    to_node_id,
                    strat_json["name"].as_str().unwrap()
                );
            }
        }

        // Flags for which we want to add an obstacle in the room, to allow progression through (or back out of) the room
        // after setting the flag on the same graph traversal step (which cannot take into account the new flag).
        let obstacle_flag_map: HashMap<RoomId, String> = vec![
            (84, "f_DefeatedKraid"),
            (193, "f_DefeatedDraygon"),
            (142, "f_DefeatedRidley"),
            (150, "f_DefeatedGoldenTorizo"),
            (122, "f_DefeatedCrocomire"),
            (57, "f_DefeatedSporeSpawn"),
            (185, "f_DefeatedBotwoon"),
            (170, "f_MaridiaTubeBroken"),
            (222, "f_ShaktoolDoneDigging"),
            (149, "f_UsedAcidChozoStatue"),
            (226, "f_KilledMetroidRoom1"),
            (227, "f_KilledMetroidRoom2"),
            (228, "f_KilledMetroidRoom3"),
            (229, "f_KilledMetroidRoom4"),
        ]
        .into_iter()
        .map(|(x, y)| (x, y.to_string()))
        .collect();

        match room_id {
            38 => self.override_morph_ball_room(&mut new_room_json),
            225 => self.override_tourian_save_room(&mut new_room_json),
            238 => self.override_mother_brain_room(&mut new_room_json),
            161 => self.override_bowling_alley(&mut new_room_json),
            12 => self.override_pit_room(&mut new_room_json),
            82 => self.override_baby_kraid_room(&mut new_room_json),
            139 => self.override_metal_pirates_room(&mut new_room_json),
            219 => self.override_plasma_room(&mut new_room_json),
            57 => self.override_spore_spawn_room(&mut new_room_json),
            226 => self.override_metroid_room_1(&mut new_room_json),
            227 => self.override_metroid_room_2(&mut new_room_json),
            228 => self.override_metroid_room_3(&mut new_room_json),
            229 => self.override_metroid_room_4(&mut new_room_json),
            _ => {}
        }

        let logical_gray_door_node_ids: Vec<(RoomId, NodeId)> = get_logical_gray_door_node_ids();
        let flagged_gray_door_node_ids: Vec<(RoomId, NodeId)> = get_flagged_gray_door_node_ids();
        let mut extra_obstacles: Vec<String> = vec![];

        for node_json in new_room_json["nodes"].members_mut() {
            let node_id = node_json["id"].as_usize().unwrap();
            let node_type = node_json["nodeType"].as_str().unwrap().to_string();
            if (node_type == "door" || node_type == "exit")
                && node_json["useImplicitDoorUnlocks"].as_bool() != Some(false)
            {
                extra_strats.push(json::object! {
                    "link": [node_id, node_id],
                    "name": "Base (Unlock Door)",
                    "requires": [],
                    "unlocksDoors": self.get_default_unlocks_door(room_json, node_id, node_id)?,
                    "flashSuitChecked": true,
                });
            }
            if (node_type == "door" || node_type == "entrance")
                && node_json["useImplicitComeInNormally"].as_bool() != Some(false)
            {
                let spawn_node_id = node_json["spawnAt"].as_usize().unwrap_or(node_id);
                extra_strats.push(json::object! {
                    "link": [node_id, spawn_node_id],
                    "name": "Base (Come In Normally)",
                    "entranceCondition": {
                        "comeInNormally": {}
                    },
                    "requires": [],
                    "flashSuitChecked": true,
                });
            }

            if !node_json.has_key("spawnAt")
                && node_type == "door"
                && (node_json["doorOrientation"] == "left"
                    || node_json["doorOrientation"] == "right")
                && node_json["useImplicitComeInWithMockball"].as_bool() != Some(false)
            {
                let heated = self.get_room_heated(room_json, node_id)?;
                let req = if heated {
                    json::array![{"heatFrames": 10}]
                } else {
                    json::array![]
                };
                extra_strats.push(json::object! {
                    "link": [node_id, node_id],
                    "name": "Base (Come In With Mockball)",
                    "entranceCondition": {
                        "comeInWithMockball": {
                            "adjacentMinTiles": 0,
                            "remoteAndLandingMinTiles": [[0, 0]],
                            "speedBooster": "any"
                        }
                    },
                    "requires": req,
                    "flashSuitChecked": true,
                });
            }

            if !node_json.has_key("spawnAt")
                && node_type == "door"
                && node_json["doorOrientation"] == "down"
                && node_json["useImplicitComeInWithGrappleJump"].as_bool() != Some(false)
            {
                extra_strats.push(json::object! {
                    "link": [node_id, node_id],
                    "name": "Base (Come In With Grapple Jump)",
                    "entranceCondition": {
                        "comeInWithGrappleJump": {
                            "position": "any"
                        }
                    },
                    "requires": [],
                    "flashSuitChecked": true
                });
            }

            if node_type == "item" && !node_json.has_key("locks") {
                node_json["locks"] = json::array![
                    {
                      "name": "Dummy Item Lock",
                      "lockType": "gameFlag",
                      "unlockStrats": [
                        {
                          "name": "Base (Collect Item)",
                          "notable": false,
                          "requires": [],
                          "flashSuitChecked": true
                        }
                      ]
                    }
                ];
            }

            if logical_gray_door_node_ids.contains(&(room_id, node_id)) {
                let obstacle_name = format!("door_{node_id}");
                extra_obstacles.push(obstacle_name);
            }

            if node_json.has_key("locks")
                && (!["door", "entrance"].contains(&node_json["nodeType"].as_str().unwrap())
                    || flagged_gray_door_node_ids.contains(&(room_id, node_id)))
            {
                ensure!(node_json["locks"].len() == 1);
                let lock = node_json["locks"][0].clone();
                let mut unlock_strats = lock["unlockStrats"].clone();
                let yields = if lock["yields"] != JsonValue::Null {
                    lock["yields"].clone()
                } else {
                    node_json["yields"].clone()
                };

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
                            ],
                            "flashSuitChecked": true
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
                        ("door", _) => {
                            if flagged_gray_door_node_ids.contains(&(room_id, node_id)) {
                                new_strat["setsFlags"] = lock["yields"].clone();
                            } else {
                                continue;
                            }
                        }
                        ("event", "flag" | "boss") | ("junction", _) => {
                            new_strat["setsFlags"] = yields.clone();
                        }
                        ("item", _) => {
                            new_strat["collectsItems"] = json::array![node_id];
                            if (room_id, node_id) == (219, 2) {
                                // Plasma Room: collecting the item closes the door.
                                new_strat["resetsObstacles"] = json::array!["door_1"];
                            } else if (room_id, node_id) == (150, 3)
                                || (room_id, node_id) == (150, 4)
                            {
                                // Golden Torizo's Room: collecting either item closes the right door (if open).
                                new_strat["resetsObstacles"] = json::array!["door_2"];
                            } else if (room_id, node_id) == (122, 3) {
                                // Crocomire's Room: collecting the item closes the top door (if open).
                                // There's no strat yet where this could happen, but we handle it anyway just in case.
                                new_strat["resetsObstacles"] = json::array!["door_2"];
                            }
                        }
                        _ => {
                            continue;
                        }
                    }
                    extra_strats.push(new_strat);
                }
            }
        }

        let existing_strats: Vec<JsonValue> = new_room_json["strats"].members().cloned().collect();
        new_room_json["strats"].clear();
        // Put the extra strats at the beginning, to prioritize them during traversal.
        for strat in extra_strats {
            new_room_json["strats"].push(strat).unwrap();
        }
        for strat in existing_strats {
            new_room_json["strats"].push(strat).unwrap();
        }

        if let Some(obstacle_flag) = obstacle_flag_map.get(&room_id) {
            extra_obstacles.push(obstacle_flag.clone());
            ensure!(new_room_json["strats"].is_array());

            // For each strat requiring one of the "obstacle flags" listed above, modify the strat to include
            // a possibility depending on the obstacle instead:
            // e.g., "f_DefeatedKraid" becomes {"or": ["f_DefeatedKraid", {"obstaclesCleared": ["f_DefeatedKraid"]}]}
            for strat in new_room_json["strats"].members_mut() {
                Self::replace_obstacle_flag(&mut strat["requires"], obstacle_flag);
                if strat.has_key("unlocksDoors") {
                    for unlock in strat["unlocksDoors"].members_mut() {
                        Self::replace_obstacle_flag(&mut unlock["requires"], obstacle_flag);
                    }
                }
                let has_flag = strat["setsFlags"]
                    .members()
                    .any(|x| x.as_str().unwrap() == obstacle_flag);
                if has_flag {
                    if !strat.has_key("clearsObstacles") {
                        strat["clearsObstacles"] = json::array![];
                    }
                    strat["clearsObstacles"].push(obstacle_flag.clone())?;
                }
            }

            for node_json in new_room_json["nodes"].members_mut() {
                if node_json.has_key("locks") {
                    for lock_json in node_json["locks"].members_mut() {
                        for strat_json in lock_json["unlockStrats"].members_mut() {
                            Self::replace_obstacle_flag(&mut strat_json["requires"], obstacle_flag);
                        }
                    }
                }
            }
        }

        for strat_json in new_room_json["strats"].members_mut() {
            let from_node_id = strat_json["link"][0].as_usize().unwrap();

            if strat_json.has_key("entranceCondition")
                && logical_gray_door_node_ids.contains(&(room_id, from_node_id))
            {
                if !strat_json.has_key("clearsObstacles") {
                    strat_json["clearsObstacles"] = json::array![];
                }
                strat_json["clearsObstacles"].push(format!("door_{from_node_id}"))?;
            }
        }

        if !extra_obstacles.is_empty() {
            if !new_room_json.has_key("obstacles") {
                new_room_json["obstacles"] = json::array![];
            }
            for obstacle_name in &extra_obstacles {
                new_room_json["obstacles"]
                    .push(json::object! {
                        "id": obstacle_name.clone(),
                        "name": obstacle_name.clone(),
                    })
                    .unwrap();
            }
        }

        Ok(new_room_json)
    }

    pub fn get_obstacle_data(
        &self,
        strat_json: &JsonValue,
        from_obstacles_bitmask: ObstacleMask,
        obstacles_idx_map: &HashMap<String, usize>,
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
        Ok(node_json["doorEnvironments"][0]["physics"]
            .as_str()
            .unwrap()
            .to_string())
    }

    pub fn get_room_heated(&self, room_json: &JsonValue, node_id: NodeId) -> Result<bool> {
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
            return env["heated"]
                .as_bool()
                .context("Expecting 'heated' to be a bool");
        }
        bail!("No match for node {} in roomEnvironments", node_id);
    }

    pub fn all_links(&self) -> Vec<Link> {
        let mut links = self.links.clone();
        for (link, _) in self.node_gmode_regain_mobility.values().flatten() {
            let mut new_link = link.clone();
            new_link.requirement = Requirement::make_and(vec![
                link.requirement.clone(),
                Requirement::Tech(self.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE_IMMOBILE]),
            ]);
            links.push(new_link);
        }
        links
    }

    fn parse_grapple_swing_block(&self, block_json: &JsonValue) -> Result<GrappleSwingBlock> {
        let mut obstructions: Vec<(i32, i32)> = vec![];
        for ob in block_json["obstructions"].members() {
            obstructions.push((ob[0].as_i32().unwrap(), ob[1].as_i32().unwrap()));
        }
        Ok(GrappleSwingBlock {
            position: (
                Float::new(block_json["position"][0].as_f32().unwrap()),
                Float::new(block_json["position"][1].as_f32().unwrap()),
            ),
            environment: match block_json["environment"].as_str().unwrap_or("air") {
                "air" => GrappleSwingBlockEnvironment::Air,
                "water" => GrappleSwingBlockEnvironment::Water,
                _ => bail!(
                    "unexpected grapple swing block environment: {}",
                    block_json["environment"]
                ),
            },
            obstructions,
        })
    }

    fn parse_exit_condition(
        &self,
        exit_json: &JsonValue,
        room_id: RoomId,
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
        let req = Requirement::Free; // This was used by "leaveShinecharged" before but is currently unused.
        let exit_condition = match key {
            "leaveNormally" => ExitCondition::LeaveNormally {},
            "leaveWithRunway" => {
                let runway_geometry = parse_runway_geometry(value)?;
                let runway_effective_length = compute_runway_effective_length(&runway_geometry);
                let runway_heated = value["heated"].as_bool().unwrap_or(heated);
                ExitCondition::LeaveWithRunway {
                    effective_length: Float::new(runway_effective_length),
                    min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                    heated: runway_heated,
                    physics,
                    from_exit_node: from_node_id == to_node_id,
                }
            }
            "leaveShinecharged" => ExitCondition::LeaveShinecharged { physics },
            "leaveWithTemporaryBlue" => ExitCondition::LeaveWithTemporaryBlue {
                direction: parse_temporary_blue_direction(value["direction"].as_str())?,
            },
            "leaveWithSpark" => {
                let node_json = &self.node_json_map[&(room_id, to_node_id)];
                let door_orientation =
                    parse_door_orientation(node_json["doorOrientation"].as_str().unwrap())?;
                ExitCondition::LeaveWithSpark {
                    position: parse_spark_position(value["position"].as_str())?,
                    door_orientation,
                }
            }
            "leaveSpinning" => {
                let remote_runway_geometry = parse_runway_geometry(&value["remoteRunway"])?;
                let remote_runway_effective_length =
                    compute_runway_effective_length(&remote_runway_geometry);
                ExitCondition::LeaveSpinning {
                    remote_runway_length: Float::new(remote_runway_effective_length),
                    blue: parse_blue_option(value["blue"].as_str())?,
                    heated,
                    min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                    max_extra_run_speed: Float::new(parse_hex(&value["maxExtraRunSpeed"], 7.0)?),
                }
            }
            "leaveWithMockball" => {
                let remote_runway_geometry = parse_runway_geometry(&value["remoteRunway"])?;
                let remote_runway_effective_length =
                    compute_runway_effective_length(&remote_runway_geometry);
                let landing_runway_geometry = parse_runway_geometry(&value["landingRunway"])?;
                let landing_runway_effective_length =
                    compute_runway_effective_length(&landing_runway_geometry);
                ExitCondition::LeaveWithMockball {
                    remote_runway_length: Float::new(remote_runway_effective_length),
                    landing_runway_length: Float::new(landing_runway_effective_length),
                    blue: parse_blue_option(value["blue"].as_str())?,
                    heated,
                    min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                    max_extra_run_speed: Float::new(parse_hex(&value["maxExtraRunSpeed"], 7.0)?),
                }
            }
            "leaveWithSpringBallBounce" => {
                let remote_runway_geometry = parse_runway_geometry(&value["remoteRunway"])?;
                let remote_runway_effective_length =
                    compute_runway_effective_length(&remote_runway_geometry);
                let landing_runway_geometry = parse_runway_geometry(&value["landingRunway"])?;
                let landing_runway_effective_length =
                    compute_runway_effective_length(&landing_runway_geometry);
                ExitCondition::LeaveWithSpringBallBounce {
                    remote_runway_length: Float::new(remote_runway_effective_length),
                    landing_runway_length: Float::new(landing_runway_effective_length),
                    blue: parse_blue_option(value["blue"].as_str())?,
                    heated,
                    movement_type: parse_bounce_movement_type(
                        value["movementType"].as_str().unwrap(),
                    )?,
                    min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                    max_extra_run_speed: Float::new(parse_hex(&value["maxExtraRunSpeed"], 7.0)?),
                }
            }
            "leaveSpaceJumping" => {
                let remote_runway_geometry = parse_runway_geometry(&value["remoteRunway"])?;
                let remote_runway_effective_length =
                    compute_runway_effective_length(&remote_runway_geometry);
                ExitCondition::LeaveSpaceJumping {
                    remote_runway_length: Float::new(remote_runway_effective_length),
                    blue: parse_blue_option(value["blue"].as_str())?,
                    heated,
                    min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                    max_extra_run_speed: Float::new(parse_hex(&value["maxExtraRunSpeed"], 7.0)?),
                }
            }
            "leaveWithGModeSetup" => ExitCondition::LeaveWithGModeSetup {
                knockback: value["knockback"].as_bool().unwrap_or(true),
                heated,
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
                height: Float::new(
                    value["height"]
                        .as_f32()
                        .context("Expecting number 'height'")?,
                ),
                heated,
            },
            "leaveWithPlatformBelow" => ExitCondition::LeaveWithPlatformBelow {
                height: Float::new(
                    value["height"]
                        .as_f32()
                        .context("Expecting number 'height'")?,
                ),
                left_position: Float::new(
                    value["leftPosition"]
                        .as_f32()
                        .context("Expecting number 'leftPosition'")?,
                ),
                right_position: Float::new(
                    value["rightPosition"]
                        .as_f32()
                        .context("Expecting number 'rightPosition'")?,
                ),
            },
            "leaveWithSidePlatform" => {
                let runway_geometry = parse_runway_geometry(&value["runway"])?;
                let runway_effective_length = compute_runway_effective_length(&runway_geometry);

                ExitCondition::LeaveWithSidePlatform {
                    effective_length: Float::new(runway_effective_length),
                    height: Float::new(
                        value["height"]
                            .as_f32()
                            .context("Expecting number 'height'")?,
                    ),
                    obstruction: (
                        value["obstruction"][0].as_u16().unwrap(),
                        value["obstruction"][1].as_u16().unwrap(),
                    ),
                    environment: match physics {
                        Some(Physics::Water) => SidePlatformEnvironment::Water,
                        Some(Physics::Air) => SidePlatformEnvironment::Air,
                        _ => bail!(
                            "unexpected door physics in leaveWithSidePlatform: {:?}",
                            physics
                        ),
                    },
                }
            }
            "leaveWithGrappleSwing" => {
                let mut blocks: Vec<GrappleSwingBlock> = vec![];
                for b in value["blocks"].members() {
                    blocks.push(self.parse_grapple_swing_block(b)?);
                }
                ExitCondition::LeaveWithGrappleSwing { blocks }
            }
            "leaveWithGrappleJump" => ExitCondition::LeaveWithGrappleJump {
                position: parse_grapple_jump_position(value["position"].as_str())?,
            },
            "leaveWithGrappleTeleport" => ExitCondition::LeaveWithGrappleTeleport {
                block_positions: value["blockPositions"]
                    .members()
                    .map(|x| (x[0].as_u16().unwrap(), x[1].as_u16().unwrap()))
                    .collect(),
            },
            "leaveWithSamusEaterTeleport" => ExitCondition::LeaveWithSamusEaterTeleport {
                floor_positions: value["floorPositions"]
                    .members()
                    .map(|x| (x[0].as_u16().unwrap(), x[1].as_u16().unwrap()))
                    .collect(),
                ceiling_positions: value["ceilingPositions"]
                    .members()
                    .map(|x| (x[0].as_u16().unwrap(), x[1].as_u16().unwrap()))
                    .collect(),
            },
            "leaveWithSuperSink" => ExitCondition::LeaveWithSuperSink {},
            _ => {
                bail!(format!("Unrecognized exit condition: {}", key));
            }
        };
        Ok((exit_condition, req))
    }

    fn parse_entrance_condition(
        &mut self,
        entrance_json: &JsonValue,
        room_id: RoomId,
        from_node_id: NodeId,
        heated: bool,
    ) -> Result<(EntranceCondition, Requirement)> {
        ensure!(entrance_json.is_object());
        // TODO: implement check on comesInHeated:
        let _entrance_maybe_heated = entrance_json["comesInHeated"].as_bool().unwrap_or(true);
        let through_toilet = if entrance_json.has_key("comesThroughToilet") {
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
            ToiletCondition::Any // to cover implicit strats, e.g. "comeInNormally"
        };
        // FIXME: We rely on the fact that comesInHeated and comesThroughToilet are last in the schema,
        // ensuring that the first entry will be the main entrance condition. The auto-formatting script
        // does enforce the correct ordering, but still we should handle this in a more robust way.
        let (key, value) = entrance_json.entries().next().unwrap();
        ensure!(value.is_object());
        let req = Requirement::Free;
        let main = match key {
            "comeInNormally" => MainEntranceCondition::ComeInNormally {},
            "comeInRunning" => MainEntranceCondition::ComeInRunning {
                speed_booster: value["speedBooster"].as_bool(),
                min_tiles: Float::new(
                    value["minTiles"]
                        .as_f32()
                        .context("Expecting number 'minTiles'")?,
                ),
                max_tiles: Float::new(value["maxTiles"].as_f32().unwrap_or(255.0)),
            },
            "comeInJumping" => MainEntranceCondition::ComeInJumping {
                speed_booster: value["speedBooster"].as_bool(),
                min_tiles: Float::new(
                    value["minTiles"]
                        .as_f32()
                        .context("Expecting number 'minTiles'")?,
                ),
                max_tiles: Float::new(value["maxTiles"].as_f32().unwrap_or(255.0)),
            },
            "comeInSpaceJumping" => MainEntranceCondition::ComeInSpaceJumping {
                speed_booster: value["speedBooster"].as_bool(),
                min_tiles: Float::new(
                    value["minTiles"]
                        .as_f32()
                        .context("Expecting number 'minTiles'")?,
                ),
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
                    min_tiles: Float::new(value["minTiles"].as_f32().unwrap_or(0.0)),
                    heated,
                }
            }
            "comeInGettingBlueSpeed" => {
                let runway_geometry = parse_runway_geometry(value)?;
                // Subtract 0.25 tiles since the door transition skips over approximately that much distance beyond the door shell tile,
                // Subtract another 1 tile for leniency since taps are harder to time across a door transition:
                let runway_effective_length =
                    (compute_runway_effective_length(&runway_geometry) - 1.25).max(0.0);
                MainEntranceCondition::ComeInGettingBlueSpeed {
                    effective_length: Float::new(runway_effective_length),
                    min_tiles: Float::new(value["minTiles"].as_f32().unwrap_or(0.0)),
                    heated,
                    min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                    max_extra_run_speed: Float::new(parse_hex(&value["maxExtraRunSpeed"], 7.0)?),
                }
            }
            "comeInShinecharged" => MainEntranceCondition::ComeInShinecharged {},
            "comeInShinechargedJumping" => MainEntranceCondition::ComeInShinechargedJumping {},
            "comeInWithSpark" => {
                let node_json = &self.node_json_map[&(room_id, from_node_id)];
                let door_orientation =
                    parse_door_orientation(node_json["doorOrientation"].as_str().unwrap())?;
                MainEntranceCondition::ComeInWithSpark {
                    position: parse_spark_position(value["position"].as_str())?,
                    door_orientation,
                }
            }
            "comeInStutterShinecharging" => MainEntranceCondition::ComeInStutterShinecharging {
                min_tiles: Float::new(
                    value["minTiles"]
                        .as_f32()
                        .context("Expecting number 'minTiles'")?,
                ),
            },
            "comeInStutterGettingBlueSpeed" => {
                MainEntranceCondition::ComeInStutterGettingBlueSpeed {
                    min_tiles: Float::new(
                        value["minTiles"]
                            .as_f32()
                            .context("Expecting number 'minTiles'")?,
                    ),
                }
            }
            "comeInWithBombBoost" => MainEntranceCondition::ComeInWithBombBoost {},
            "comeInWithDoorStuckSetup" => {
                let node_json = &self.node_json_map[&(room_id, from_node_id)];
                let door_orientation =
                    parse_door_orientation(node_json["doorOrientation"].as_str().unwrap())?;
                MainEntranceCondition::ComeInWithDoorStuckSetup {
                    heated,
                    door_orientation,
                }
            }
            "comeInSpeedballing" => {
                let runway_geometry = parse_runway_geometry(&value["runway"])?;
                // Subtract 0.25 tiles since the door transition skips over approximately that much distance beyond the door shell tile,
                // Subtract another 1 tile for leniency since taps and/or speedball are harder to time across a door transition:
                let effective_runway_length =
                    (compute_runway_effective_length(&runway_geometry) - 1.25).max(0.0);
                MainEntranceCondition::ComeInSpeedballing {
                    effective_runway_length: Float::new(effective_runway_length),
                    min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                    max_extra_run_speed: Float::new(parse_hex(&value["maxExtraRunSpeed"], 7.0)?),
                    heated,
                }
            }
            "comeInWithTemporaryBlue" => MainEntranceCondition::ComeInWithTemporaryBlue {
                direction: parse_temporary_blue_direction(value["direction"].as_str())?,
            },
            "comeInSpinning" => MainEntranceCondition::ComeInSpinning {
                unusable_tiles: Float::new(value["unusableTiles"].as_f32().unwrap_or(0.0)),
                min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                max_extra_run_speed: Float::new(parse_hex(&value["maxExtraRunSpeed"], 7.0)?),
            },
            "comeInBlueSpinning" => MainEntranceCondition::ComeInBlueSpinning {
                unusable_tiles: Float::new(value["unusableTiles"].as_f32().unwrap_or(0.0)),
                min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                max_extra_run_speed: Float::new(parse_hex(&value["maxExtraRunSpeed"], 7.0)?),
            },
            "comeInBlueSpaceJumping" => MainEntranceCondition::ComeInBlueSpaceJumping {
                min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                max_extra_run_speed: Float::new(parse_hex(&value["maxExtraRunSpeed"], 7.0)?),
            },
            "comeInWithMockball" => MainEntranceCondition::ComeInWithMockball {
                speed_booster: value["speedBooster"].as_bool(),
                adjacent_min_tiles: Float::new(value["adjacentMinTiles"].as_f32().unwrap_or(255.0)),
                remote_and_landing_min_tiles: value["remoteAndLandingMinTiles"]
                    .members()
                    .map(|x| {
                        (
                            Float::new(x[0].as_f32().unwrap()),
                            Float::new(x[1].as_f32().unwrap()),
                        )
                    })
                    .collect(),
            },
            "comeInWithSpringBallBounce" => MainEntranceCondition::ComeInWithSpringBallBounce {
                speed_booster: value["speedBooster"].as_bool(),
                adjacent_min_tiles: Float::new(value["adjacentMinTiles"].as_f32().unwrap_or(255.0)),
                remote_and_landing_min_tiles: value["remoteAndLandingMinTiles"]
                    .members()
                    .map(|x| {
                        (
                            Float::new(x[0].as_f32().unwrap()),
                            Float::new(x[1].as_f32().unwrap()),
                        )
                    })
                    .collect(),
                movement_type: parse_bounce_movement_type(value["movementType"].as_str().unwrap())?,
            },
            "comeInWithBlueSpringBallBounce" => {
                MainEntranceCondition::ComeInWithBlueSpringBallBounce {
                    min_extra_run_speed: Float::new(parse_hex(&value["minExtraRunSpeed"], 0.0)?),
                    max_extra_run_speed: Float::new(parse_hex(&value["maxExtraRunSpeed"], 7.0)?),
                    min_landing_tiles: Float::new(value["minLandingTiles"].as_f32().unwrap_or(0.0)),
                    movement_type: parse_bounce_movement_type(
                        value["movementType"].as_str().unwrap(),
                    )?,
                }
            }
            "comeInWithRMode" => MainEntranceCondition::ComeInWithRMode { heated },
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
                    heated,
                }
            }
            "comeInWithStoredFallSpeed" => MainEntranceCondition::ComeInWithStoredFallSpeed {
                fall_speed_in_tiles: value["fallSpeedInTiles"]
                    .as_i32()
                    .context("Expecting integer 'fallSpeedInTiles")?,
            },
            "comeInWithWallJumpBelow" => MainEntranceCondition::ComeInWithWallJumpBelow {
                min_height: Float::new(
                    value["minHeight"]
                        .as_f32()
                        .context("Expecting number 'minHeight'")?,
                ),
            },
            "comeInWithSpaceJumpBelow" => MainEntranceCondition::ComeInWithSpaceJumpBelow {},
            "comeInWithPlatformBelow" => MainEntranceCondition::ComeInWithPlatformBelow {
                min_height: Float::new(value["minHeight"].as_f32().unwrap_or(0.0)),
                max_height: Float::new(value["maxHeight"].as_f32().unwrap_or(f32::INFINITY)),
                max_left_position: Float::new(
                    value["maxLeftPosition"].as_f32().unwrap_or(f32::INFINITY),
                ),
                min_right_position: Float::new(
                    value["minRightPosition"]
                        .as_f32()
                        .unwrap_or(f32::NEG_INFINITY),
                ),
            },
            "comeInWithSidePlatform" => {
                let mut platforms: Vec<SidePlatformEntrance> = vec![];

                for p in value["platforms"].members() {
                    platforms.push(SidePlatformEntrance {
                        min_height: Float::new(
                            p["minHeight"]
                                .as_f32()
                                .context("Expecting number 'minHeight'")?,
                        ),
                        max_height: Float::new(
                            p["maxHeight"]
                                .as_f32()
                                .context("Expecting number 'maxHeight'")?,
                        ),
                        min_tiles: Float::new(
                            p["minTiles"]
                                .as_f32()
                                .context("Expecting number 'minHeight'")?,
                        ),
                        speed_booster: p["speedBooster"].as_bool(),
                        obstructions: p["obstructions"]
                            .members()
                            .map(|x| (x[0].as_u16().unwrap(), x[1].as_u16().unwrap()))
                            .collect(),
                        environment: match p["environment"].as_str().unwrap_or("any") {
                            "air" => SidePlatformEnvironment::Air,
                            "water" => SidePlatformEnvironment::Water,
                            "any" => SidePlatformEnvironment::Any,
                            _ => {
                                bail!("unexpected side platform environment: {}", p["environment"])
                            }
                        },
                        requirement: if p.has_key("requires") {
                            let reqs_json: Vec<JsonValue> =
                                value["requires"].members().cloned().collect();
                            Requirement::make_and(
                                self.parse_requires_list(
                                    &reqs_json,
                                    &RequirementContext::default(),
                                )?,
                            )
                        } else {
                            Requirement::Free
                        },
                    });
                }
                MainEntranceCondition::ComeInWithSidePlatform { platforms }
            }
            "comeInWithGrappleSwing" => {
                let mut blocks: Vec<GrappleSwingBlock> = vec![];
                for b in value["blocks"].members() {
                    blocks.push(self.parse_grapple_swing_block(b)?);
                }
                MainEntranceCondition::ComeInWithGrappleSwing { blocks }
            }
            "comeInWithGrappleJump" => MainEntranceCondition::ComeInWithGrappleJump {
                position: parse_grapple_jump_position(value["position"].as_str())?,
            },
            "comeInWithGrappleTeleport" => MainEntranceCondition::ComeInWithGrappleTeleport {
                block_positions: value["blockPositions"]
                    .members()
                    .map(|x| (x[0].as_u16().unwrap(), x[1].as_u16().unwrap()))
                    .collect(),
            },
            "comeInWithSamusEaterTeleport" => MainEntranceCondition::ComeInWithSamusEaterTeleport {
                floor_positions: value["floorPositions"]
                    .members()
                    .map(|x| (x[0].as_u16().unwrap(), x[1].as_u16().unwrap()))
                    .collect(),
                ceiling_positions: value["ceilingPositions"]
                    .members()
                    .map(|x| (x[0].as_u16().unwrap(), x[1].as_u16().unwrap()))
                    .collect(),
            },
            "comeInWithSuperSink" => MainEntranceCondition::ComeInWithSuperSink {},
            _ => {
                bail!(format!("Unrecognized entrance condition: {}", key));
            }
        };
        Ok((
            EntranceCondition {
                through_toilet,
                main,
            },
            req,
        ))
    }

    pub fn does_come_in_shinecharged(&self, entrance_condition: &EntranceCondition) -> bool {
        matches!(
            entrance_condition.main,
            MainEntranceCondition::ComeInShinecharging { .. }
                | MainEntranceCondition::ComeInShinecharged { .. }
                | MainEntranceCondition::ComeInShinechargedJumping { .. }
                | MainEntranceCondition::ComeInStutterShinecharging { .. }
        )
    }

    pub fn does_leave_shinecharged(&self, exit_condition: &ExitCondition) -> bool {
        matches!(exit_condition, ExitCondition::LeaveShinecharged { .. })
    }

    fn process_strat(
        &mut self,
        strat_json: &JsonValue,
        room_json: &JsonValue,
        obstacles_idx_map: &HashMap<String, usize>,
        notable_map: &HashMap<String, NotableIdx>,
        node_implicit_door_unlocks: &HashMap<NodeId, bool>,
    ) -> Result<()> {
        let room_id = room_json["id"].as_usize().unwrap();
        let num_obstacles = obstacles_idx_map.len();
        let from_node_id = strat_json["link"][0].as_usize().unwrap();
        let to_node_id = strat_json["link"][1].as_usize().unwrap();
        let strat_id = strat_json["id"].as_usize();

        // TODO: deal with heated room more explicitly for Volcano Room, instead of guessing based on node ID:
        let from_heated = self.get_room_heated(room_json, from_node_id)?;
        let to_node_json = self.node_json_map[&(room_id, to_node_id)].clone();

        let to_heated = self.get_room_heated(room_json, to_node_id)?;
        let physics_res = self.get_node_physics(&self.node_json_map[&(room_id, to_node_id)]);
        let physics: Option<Physics> = if let Ok(physics_str) = &physics_res {
            Some(parse_physics(physics_str)?)
        } else {
            None
        };

        let (entrance_condition, entrance_req) = if strat_json.has_key("entranceCondition") {
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
        let bypasses_door_shell = strat_json["bypassesDoorShell"].as_bool().unwrap_or(false);
        let (exit_condition, exit_req) = if strat_json.has_key("exitCondition") {
            ensure!(strat_json["exitCondition"].is_object());
            let (e, r) = self.parse_exit_condition(
                &strat_json["exitCondition"],
                room_id,
                strat_json,
                to_heated,
                physics,
            )?;
            (Some(e), Some(r))
        } else if bypasses_door_shell {
            (
                Some(ExitCondition::LeaveNormally {}),
                Some(Requirement::Free),
            )
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
            if (entrance_condition.is_some() || gmode_regain_mobility.is_some())
                && from_obstacles_bitmask != 0
            {
                continue;
            }
            ensure!(strat_json["requires"].is_array());
            let requires_json: Vec<JsonValue> = strat_json["requires"].members().cloned().collect();

            let to_obstacles_bitmask =
                self.get_obstacle_data(strat_json, from_obstacles_bitmask, obstacles_idx_map)?;
            let ctx = RequirementContext {
                room_id,
                strat_name: strat_json["name"].as_str().unwrap(),
                to_node_id,
                room_heated: from_heated || to_heated,
                from_obstacles_bitmask,
                obstacles_idx_map: Some(obstacles_idx_map),
                unlocks_doors_json: if strat_json.has_key("unlocksDoors") {
                    Some(&strat_json["unlocksDoors"])
                } else {
                    None
                },
                node_implicit_door_unlocks: Some(node_implicit_door_unlocks),
                notable_map: Some(notable_map),
            };
            let mut requires_vec = vec![];
            if let Some(r) = &entrance_req {
                requires_vec.push(r.clone());
            }
            requires_vec.extend(self.parse_requires_list(&requires_json, &ctx)?);
            let strat_name = strat_json["name"].as_str().unwrap().to_string();
            let strat_notes = self.parse_note(&strat_json["note"]);

            if bypasses_door_shell {
                requires_vec.push(Requirement::Tech(
                    self.tech_isv.index_by_key[&TECH_ID_CAN_SKIP_DOOR_LOCK],
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

            if let Some(e) = &exit_condition {
                to_actions.push(VertexAction::Exit(e.clone()));
                requires_vec.push(exit_req.clone().unwrap());
            } else if ["door", "exit"].contains(&to_node_json["nodeType"].as_str().unwrap())
                && strat_json.has_key("unlocksDoors")
                && to_node_json["useImplicitLeaveNormally"].as_bool() != Some(false)
                && let Ok(unlock_to_door_req) = self.get_unlocks_doors_req(to_node_id, &ctx)
            {
                let maybe_exit_req = Some(unlock_to_door_req);
                to_actions.push(VertexAction::MaybeExit(
                    ExitCondition::LeaveNormally {},
                    maybe_exit_req.clone().unwrap(),
                ));
            }

            if !bypasses_door_shell && exit_condition.is_some() {
                let unlock_to_door_req = self.get_unlocks_doors_req(to_node_id, &ctx)?;
                requires_vec.push(unlock_to_door_req);
            }

            match strat_json["flashSuitChecked"].as_bool() {
                None => {
                    error!("Missing flashSuitChecked: {}", strat_json);
                }
                Some(false) => {
                    requires_vec.push(Requirement::NoFlashSuit);
                }
                Some(true) => {}
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
                obstacle_mask: if exit_condition.is_some() {
                    0
                } else {
                    to_obstacles_bitmask
                },
                actions: to_actions.clone(),
            });
            let start_with_shinecharge = if let Some(e) = &entrance_condition {
                self.does_come_in_shinecharged(e)
            } else {
                strat_json["startsWithShineCharge"].as_bool() == Some(true)
            };
            let end_with_shinecharge = if let Some(e) = &exit_condition {
                self.does_leave_shinecharged(e)
            } else {
                strat_json["endsWithShineCharge"].as_bool() == Some(true)
            };
            let link = Link {
                from_vertex_id,
                to_vertex_id,
                requirement: requirement.clone(),
                start_with_shinecharge,
                end_with_shinecharge,
                difficulty: 0,
                length: 1,
                strat_id,
                strat_name: strat_name.clone(),
                strat_notes,
            };
            if gmode_regain_mobility.is_some() {
                if entrance_condition.is_some() || exit_condition.is_some() {
                    bail!(
                        "gModeRegainMobility combined with entranceCondition or exitCondition is not allowed."
                    );
                }
                if from_node_id != to_node_id {
                    bail!("gModeRegainMobility `from` and `to` node must be equal.");
                }
                self.node_gmode_regain_mobility
                    .entry((room_id, to_node_id))
                    .or_default()
                    .push((link, gmode_regain_mobility.clone().unwrap()))
            } else if strat_json.has_key("farmCycleDrops") {
                let enemy_drops = self.parse_enemy_drops(&strat_json["farmCycleDrops"], false);
                let enemy_drops_buffed =
                    self.parse_enemy_drops(&strat_json["farmCycleDrops"], true);
                let mut has_high_energy: bool = false;
                let mut has_high_missiles: bool = false;
                let mut has_high_supers: bool = false;
                let mut has_high_power_bombs: bool = false;
                for drop in &enemy_drops {
                    if drop.power_bomb_weight.get() > 0.25 {
                        has_high_power_bombs = true;
                    }
                    if drop.super_weight.get() > 0.25 {
                        has_high_supers = true;
                    }
                    if (drop.large_energy_weight.get() + drop.small_energy_weight.get()) > 0.25
                        && drop.missile_weight.get() > 0.0
                    {
                        has_high_energy = true;
                    }
                    if drop.missile_weight.get() > 0.25
                        && drop.large_energy_weight.get() + drop.small_energy_weight.get() > 0.0
                    {
                        has_high_missiles = true;
                    }
                }
                for full_energy in [false, true] {
                    for full_missiles in [false, true] {
                        for full_supers in [false, true] {
                            for full_power_bombs in [false, true] {
                                if full_energy && !has_high_energy {
                                    continue;
                                }
                                if full_missiles && !has_high_missiles {
                                    continue;
                                }
                                if full_supers && !has_high_supers {
                                    continue;
                                }
                                if full_power_bombs && !has_high_power_bombs {
                                    continue;
                                }
                                if full_missiles && full_energy {
                                    continue;
                                }

                                let mut farm_link = link.clone();
                                if full_energy {
                                    farm_link.strat_name.push_str("- Full Energy");
                                }
                                if full_missiles {
                                    farm_link.strat_name.push_str("- Full Missiles");
                                }
                                if full_supers {
                                    farm_link.strat_name.push_str("- Full Supers");
                                }
                                if full_power_bombs {
                                    farm_link.strat_name.push_str("- Full Power Bombs");
                                }
                                farm_link.requirement = Requirement::Farm {
                                    requirement: Box::new(requirement.clone()),
                                    enemy_drops: enemy_drops.clone(),
                                    enemy_drops_buffed: enemy_drops_buffed.clone(),
                                    full_energy,
                                    full_missiles,
                                    full_supers,
                                    full_power_bombs,
                                };
                                self.links.push(farm_link.clone());
                            }
                        }
                    }
                }
            } else {
                self.links.push(link.clone());
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
                    start_with_shinecharge: false,
                    end_with_shinecharge: false,
                    difficulty: 0,
                    length: 1,
                    strat_id: None,
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
                    if unlock_node_id == to_node_id && exit_condition.is_some() {
                        continue;
                    }
                    let unlock_vertex_id = self.vertex_isv.add(&VertexKey {
                        room_id,
                        node_id: to_node_id,
                        obstacle_mask: to_obstacles_bitmask,
                        actions: vec![VertexAction::DoorUnlock(unlock_node_id, to_vertex_id)],
                    });
                    let unlock_req = self.get_unlocks_doors_req(unlock_node_id, &ctx)?;
                    self.links.push(Link {
                        from_vertex_id: to_vertex_id,
                        to_vertex_id: unlock_vertex_id,
                        requirement: unlock_req,
                        start_with_shinecharge: end_with_shinecharge,
                        end_with_shinecharge,
                        difficulty: 0,
                        length: 1,
                        strat_id: None,
                        strat_name: "Base (Unlock)".to_string(),
                        strat_notes: vec![],
                    });
                    self.links.push(Link {
                        from_vertex_id: unlock_vertex_id,
                        to_vertex_id,
                        requirement: Requirement::Free,
                        start_with_shinecharge: end_with_shinecharge,
                        end_with_shinecharge,
                        difficulty: 0,
                        length: 1,
                        strat_id: None,
                        strat_name: "Base (Return from Unlock)".to_string(),
                        strat_notes: vec![],
                    });
                }
            }
        }
        Ok(())
    }

    fn get_full_area(&self, room_json: &JsonValue) -> String {
        let area = room_json["area"].as_str().unwrap().to_string();
        let sub_area = room_json["subarea"].as_str().unwrap_or("").to_string();
        let sub_sub_area = room_json["subsubarea"].as_str().unwrap_or("").to_string();
        if !sub_sub_area.is_empty() {
            format!("{sub_sub_area} {sub_area} {area}")
        } else if !sub_area.is_empty() && sub_area != "Main" {
            format!("{sub_area} {area}")
        } else {
            area
        }
    }

    fn process_room(&mut self, room_json: &JsonValue) -> Result<()> {
        let room_id = room_json["id"].as_usize().unwrap();
        self.room_json_map.insert(room_id, room_json.clone());
        let mut room_ptr =
            parse_int::parse::<usize>(room_json["roomAddress"].as_str().unwrap()).unwrap();
        self.room_ptrs.push(room_ptr);
        self.raw_room_id_by_ptr.insert(room_ptr, room_id);
        if room_ptr == 0x7D69A {
            room_ptr = 0x7D646; // Treat East Pants Room as part of Pants Room
        } else if room_ptr == 0x7968F {
            room_ptr = 0x793FE; // Treat Homing Geemer Room as part of West Ocean
        } else {
            self.room_id_by_ptr.insert(room_ptr, room_id);
        }
        self.room_ptr_by_id.insert(room_id, room_ptr);
        self.room_full_area
            .insert(room_id, self.get_full_area(room_json));

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
        let logical_gray_door_node_ids = get_logical_gray_door_node_ids();
        for node_json in room_json["nodes"].members() {
            let node_id = node_json["id"].as_usize().unwrap();
            self.node_json_map
                .insert((room_id, node_id), node_json.clone());

            for obstacle_mask in 0..(1 << num_obstacles) {
                self.vertex_isv.add(&VertexKey {
                    room_id,
                    node_id,
                    obstacle_mask,
                    actions: vec![],
                });
            }

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

            if logical_gray_door_node_ids.contains(&(room_id, node_id)) {
                ensure!(node_json["locks"].is_array());
                let lock = &node_json["locks"][0];
                let mut req_list = vec![];
                ensure!(lock["unlockStrats"].is_array());
                for unlock_strat in lock["unlockStrats"].members() {
                    req_list.push(json::object! {"and": unlock_strat["requires"].clone()});
                }
                req_list.push(json::object! {"obstaclesCleared": [format!("door_{}", node_id)]});
                let req = json::object! {"or": JsonValue::Array(req_list)};
                self.grey_lock_map.insert((room_id, node_id), req);
            }

            let height = node_json["mapTileMask"].len();
            let width = node_json["mapTileMask"][0].len();
            let mut coords: Vec<(usize, usize)> = vec![];
            for y in 0..height {
                for x in 0..width {
                    if node_json["mapTileMask"][y][x] == 2 {
                        coords.push((x, y));
                    }
                }
            }
            self.node_tile_coords.insert((room_id, node_id), coords);
        }
        let mut node_implicit_door_unlocks: HashMap<NodeId, bool> = HashMap::new();
        for node_json in room_json["nodes"].members() {
            let node_id = node_json["id"].as_usize().unwrap();

            node_implicit_door_unlocks.insert(
                node_id,
                node_json["useImplicitDoorUnlocks"]
                    .as_bool()
                    .unwrap_or(true),
            );

            // Implicit leaveWithGMode:
            if !node_json.has_key("spawnAt")
                && node_json["nodeType"].as_str().unwrap() == "door"
                && node_json["isDoorImmediatelyClosed"].as_bool() != Some(true)
            {
                for morphed in [false, true] {
                    if !morphed
                        && node_json["useImplicitCarryGModeBackThrough"].as_bool() == Some(false)
                    {
                        continue;
                    }
                    if morphed
                        && (node_json["doorOrientation"] == "up"
                            || node_json["useImplicitCarryGModeMorphBackThrough"].as_bool()
                                == Some(false))
                    {
                        continue;
                    }
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
                                heated: self.get_room_heated(room_json, node_id)?,
                            },
                        })],
                    });
                    let to_vertex_id = self.vertex_isv.add(&VertexKey {
                        room_id,
                        node_id,
                        obstacle_mask: 0,
                        actions: vec![VertexAction::Exit(ExitCondition::LeaveWithGMode {
                            morphed,
                        })],
                    });
                    let link = Link {
                        from_vertex_id,
                        to_vertex_id,
                        requirement: Requirement::Free,
                        start_with_shinecharge: false,
                        end_with_shinecharge: false,
                        difficulty: 0,
                        length: 1,
                        strat_id: None,
                        strat_name: if morphed {
                            "Carry G-Mode Morph Back Through".to_string()
                        } else {
                            "Carry G-Mode Back Through".to_string()
                        },
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

        // Process notables:
        let mut notable_map: HashMap<String, NotableIdx> = HashMap::new();
        for notable in room_json["notables"].members() {
            let notable_name = notable["name"].as_str().unwrap().to_string();
            let notable_id = notable["id"].as_usize().unwrap() as NotableId;
            let notable_data = NotableInfo {
                room_id,
                notable_id,
                name: notable_name.clone(),
                note: self.parse_note(&notable["note"]).join(" "),
            };
            let notable_idx = self.notable_info.len();
            let notable_idx2 = self.notable_isv.add(&(room_id, notable_id));
            assert_eq!(notable_idx, notable_idx2);
            self.notable_info.push(notable_data);
            self.notable_id_by_name
                .insert((room_id, notable_name.clone()), notable_id);
            // TODO: the room-local `notable_map` could probably be eliminated, in favor of just using the global
            // one (`notable_id_by_name``)
            notable_map.insert(notable_name, notable_idx);
        }

        // Process strats:
        ensure!(room_json["strats"].is_array());
        for strat_json in room_json["strats"].members() {
            self.process_strat(
                strat_json,
                room_json,
                &obstacles_idx_map,
                &notable_map,
                &node_implicit_door_unlocks,
            )
            .context(format!(
                "Processing {} strat '{}'",
                strat_json["link"], strat_json["name"]
            ))?;
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

    fn add_connection(&mut self, src: (RoomId, NodeId), dst: (RoomId, NodeId), _conn: &JsonValue) {
        let src_ptr = self.node_ptr_map.get(&src).copied();
        let dst_ptr = self.node_ptr_map.get(&dst).copied();
        let is_west_ocean_bridge = src == (32, 7) || src == (32, 8);
        if (src_ptr.is_some() || dst_ptr.is_some()) && !is_west_ocean_bridge {
            self.door_ptr_pair_map.insert((src_ptr, dst_ptr), src);
            self.reverse_door_ptr_pair_map
                .insert(src, (src_ptr, dst_ptr));
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
        ]
        .into_iter()
        .map(|x| self.flag_isv.index_by_key[x])
        .collect();

        let mut node_pair_vec: Vec<(RoomId, NodeId)> = self.node_json_map.keys().cloned().collect();
        node_pair_vec.sort();
        for (room_id, node_id) in node_pair_vec {
            let node_json = &self.node_json_map[&(room_id, node_id)];
            if node_json["nodeType"] == "item" {
                self.item_locations.push((room_id, node_id));
            }
            if node_json.has_key("utility")
                && node_json["utility"].members().any(|x| x == "save")
                && room_id != 304
            {
                // room_id: 304 is the broken save room, which is not a logical save for the purposes of
                // guaranteed early save station, which is all this is currently used for.
                self.save_locations.push((room_id, node_id));
            }
        }

        let mut item_location_vertex_map: HashMap<(RoomId, NodeId), Vec<VertexId>> = HashMap::new();
        let mut flag_location_vertex_map: HashMap<FlagId, Vec<VertexId>> = HashMap::new();
        for (vertex_id, vertex_key) in self.vertex_isv.keys.iter().enumerate() {
            for action in &vertex_key.actions {
                match action {
                    VertexAction::Nothing => panic!("Unexpected VertexAction::Nothing"),
                    VertexAction::MaybeExit(exit_condition, exit_req) => self
                        .node_exit_conditions
                        .entry((vertex_key.room_id, vertex_key.node_id))
                        .or_default()
                        .push(ExitInfo {
                            vertex_id,
                            exit_condition: exit_condition.clone(),
                            exit_req: exit_req.clone(),
                        }),
                    VertexAction::Exit(exit_condition) => self
                        .node_exit_conditions
                        .entry((vertex_key.room_id, vertex_key.node_id))
                        .or_default()
                        .push(ExitInfo {
                            vertex_id,
                            exit_condition: exit_condition.clone(),
                            exit_req: Requirement::Free,
                        }),
                    VertexAction::Enter(entrance_condition) => self
                        .node_entrance_conditions
                        .entry((vertex_key.room_id, vertex_key.node_id))
                        .or_default()
                        .push((vertex_id, entrance_condition.clone())),
                    VertexAction::DoorUnlock(door_node_id, _) => self
                        .node_door_unlock
                        .entry((vertex_key.room_id, *door_node_id))
                        .or_default()
                        .push(vertex_id),
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

    pub fn get_weapon_mask(&self, items: &[bool], tech: &[bool]) -> WeaponMask {
        let mut weapon_mask = 0;
        let implicit_requires: HashSet<String> = vec!["PowerBeam"]
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        // TODO: possibly make this more efficient. We could avoid dealing with strings
        // and just use a pre-computed item bitmask per weapon. But not sure yet if it matters.
        'weapon: for (i, weapon_name) in self.weapon_isv.keys.iter().enumerate() {
            let weapon = &self.weapon_json_map[weapon_name];
            assert!(weapon["useRequires"].is_array());
            for req_json in weapon["useRequires"].members() {
                let req_name = req_json.as_str().unwrap();
                if implicit_requires.contains(req_name) {
                    continue;
                }
                if self.item_isv.index_by_key.contains_key(req_name) {
                    let item_idx = self.item_isv.index_by_key[req_name];
                    if !items[item_idx] {
                        continue 'weapon;
                    }
                } else if self.tech_id_by_name.contains_key(req_name) {
                    let tech_idx = self.tech_isv.index_by_key[&self.tech_id_by_name[req_name]];
                    if !tech[tech_idx] {
                        continue 'weapon;
                    }
                } else {
                    panic!("Unrecognized weapon requirement: {req_name}");
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
        let mut start_location_id_map: HashMap<(usize, usize), usize> = HashMap::new();
        for (i, loc) in start_locations.iter_mut().enumerate() {
            if start_location_id_map.contains_key(&(loc.room_id, loc.node_id)) {
                bail!(
                    "Non-unique (room_id, node_id) for start location: {:?}",
                    loc
                );
            }

            let door_node_id = loc.door_load_node_id.unwrap_or(loc.node_id);
            if !self
                .reverse_door_ptr_pair_map
                .contains_key(&(loc.room_id, door_node_id))
            {
                bail!("Invalid door node for start location: {:?}", loc);
            }

            start_location_id_map.insert((loc.room_id, loc.node_id), i);
            if loc.requires.is_none() {
                loc.requires_parsed = Some(Requirement::Free);
            } else {
                let mut req_json_list: Vec<JsonValue> = vec![];
                for req in loc.requires.as_ref().unwrap() {
                    let req_str = req.to_string();
                    let req_json = json::parse(&req_str)
                        .with_context(|| format!("Error parsing requires in {loc:?}"))?;
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
                panic!("Bad starting location: {loc:?}");
            }
        }
        self.start_locations = start_locations;
        self.start_location_id_map = start_location_id_map;
        Ok(())
    }

    fn has_energy_fill(req: &Requirement) -> bool {
        match req {
            Requirement::Farm { .. } => true,
            Requirement::EnergyRefill(_) => true,
            Requirement::RegularEnergyRefill(_) => true,
            Requirement::EnergyStationRefill => true,
            Requirement::And(requirements) => requirements.iter().any(Self::has_energy_fill),
            Requirement::Or(requirements) => requirements.iter().any(Self::has_energy_fill),
            _ => false,
        }
    }

    fn load_hub_locations(&mut self) -> Result<()> {
        // Hub locations used to be hand-curated to be reasonable initial farms.
        // Now we determine them automatically, based on whether energy can be farmed
        // at a node. The aim is for `sm-json-data` to handle all the farm logic
        // in a consistent way.

        let mut hub_farms: Vec<(VertexId, Requirement)> = vec![];
        for link in &self.links {
            if link.from_vertex_id != link.to_vertex_id {
                continue;
            }
            let vertex_key = &self.vertex_isv.keys[link.from_vertex_id];
            if vertex_key.obstacle_mask != 0 {
                continue;
            }
            if Self::has_energy_fill(&link.requirement) {
                hub_farms.push((link.from_vertex_id, link.requirement.clone()));
            }
        }

        self.hub_farms = hub_farms;
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
            // Make sure room IDs in room geometry match the sm-json-data (via the room address)
            let room_id = self.room_id_by_ptr[&room.rom_address];
            assert_eq!(room_id, room.room_id);
            self.room_idx_by_ptr.insert(room.rom_address, room_idx);
            self.room_idx_by_id.insert(room_id, room_idx);
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
            self.room_shape
                .insert(room_id, (room.map[0].len(), room.map.len()));
        }
        self.room_geometry = room_geometry;
        Ok(())
    }

    fn extract_all_tech_dependencies(&mut self) -> Result<()> {
        let tech_vec = self.tech_isv.keys.clone();
        for tech_id in &tech_vec {
            let tech_name = self.tech_json_map[tech_id]["name"]
                .as_str()
                .unwrap()
                .to_string();
            let req = self.get_tech_requirement(&tech_name, false)?;
            let deps: Vec<TechId> = self
                .extract_tech_dependencies(&req)
                .into_iter()
                .filter(|x| x != tech_id)
                .collect();
            self.tech_dependencies.insert(*tech_id, deps);
        }
        Ok(())
    }

    fn extract_all_strat_dependencies(&mut self) -> Result<()> {
        // TODO: get rid of this, or replace it.
        Ok(())
    }

    pub fn make_links_data(
        &mut self,
        link_difficulty_length_fn: &dyn Fn(&Link, &GameData) -> (u8, LinkLength),
    ) {
        let mut new_links = Vec::new();
        for link in &self.links {
            let mut link = link.clone();
            let (difficulty, length) = link_difficulty_length_fn(&link, self);
            link.difficulty = difficulty;
            link.length = length;
            new_links.push(link);
        }
        self.links = new_links.clone();
        self.base_links_data = LinksDataGroup::new(new_links, self.vertex_isv.keys.len(), 0);
        self.make_reset_room_requirements();
    }

    fn process_reset_room_req(req: &Requirement, free_unlock_id: (RoomId, NodeId)) -> Requirement {
        match req {
            &Requirement::DoorUnlocked { room_id, node_id }
                if (room_id, node_id) == free_unlock_id =>
            {
                Requirement::Free
            }
            &Requirement::UnlockDoor {
                room_id, node_id, ..
            } if (room_id, node_id) == free_unlock_id => Requirement::Free,
            Requirement::ReserveTrigger { .. } => Requirement::Never,
            Requirement::RegularEnergyDrain(_) => Requirement::Never,
            Requirement::ResetRoom { .. } => Requirement::Never,
            Requirement::And(and_reqs) => Requirement::make_and(
                and_reqs
                    .iter()
                    .map(|r| Self::process_reset_room_req(r, free_unlock_id))
                    .collect(),
            ),
            Requirement::Or(or_reqs) => Requirement::make_or(
                or_reqs
                    .iter()
                    .map(|r| Self::process_reset_room_req(r, free_unlock_id))
                    .collect(),
            ),
            _ => req.clone(),
        }
    }

    fn make_reset_room_requirements(&mut self) {
        let mut nodes: Vec<(RoomId, NodeId)> =
            self.node_entrance_conditions.keys().copied().collect();
        nodes.sort();
        for (room_id, node_id) in nodes {
            let heated = self
                .get_room_heated(&self.room_json_map[&room_id], node_id)
                .unwrap_or(true);
            let entrances = &self.node_entrance_conditions[&(room_id, node_id)];
            let mut entrance_vertex_ids: HashSet<VertexId> = HashSet::new();
            for (v, e) in entrances {
                if let MainEntranceCondition::ComeInNormally {} = e.main {
                    entrance_vertex_ids.insert(*v);
                }
            }

            let exits = &self
                .node_exit_conditions
                .get(&(room_id, node_id))
                .cloned()
                .unwrap_or_default();
            let mut exit_vertex_ids: HashSet<VertexId> = HashSet::new();
            for exit_info in exits {
                if let ExitCondition::LeaveNormally {} = exit_info.exit_condition {
                    exit_vertex_ids.insert(exit_info.vertex_id);
                }
            }

            let mut req_or: Vec<Requirement> = vec![];
            let mut entrance_reqs_map: HashMap<VertexId, Vec<Requirement>> = HashMap::new();
            for &entrance_id in &entrance_vertex_ids {
                for (_, link) in &self.base_links_data.links_by_src[entrance_id] {
                    if exit_vertex_ids.contains(&link.to_vertex_id) {
                        // Handle links that directly connect a comeInNormally vertex to a leaveNormally vertex:
                        // This is for the case where comeInNormally+leaveNormally is in the same strat.
                        // Here we assume that any heat frame requirements are included explicitly in the strat.
                        req_or.push(link.requirement.clone());
                    } else {
                        // Handle links from a comeInNormally vertex to an intermediate vertex
                        entrance_reqs_map
                            .entry(link.to_vertex_id)
                            .or_default()
                            .push(link.requirement.clone());
                    }
                }
            }

            let mut exit_reqs_map: HashMap<VertexId, Vec<Requirement>> = HashMap::new();
            for &exit_id in &exit_vertex_ids {
                for (_, link) in &self.base_links_data.links_by_dst[exit_id] {
                    // Note: links in `links_by_dst` have their `from_vertex_id`` and `to_vertex_id`` swapped.
                    exit_reqs_map
                        .entry(link.to_vertex_id)
                        .or_default()
                        .push(link.requirement.clone());
                }
            }

            for &v in entrance_reqs_map.keys() {
                if !exit_reqs_map.contains_key(&v) {
                    continue;
                }
                let entrance_or = Requirement::make_or(entrance_reqs_map[&v].clone());
                let exit_or = Requirement::make_or(exit_reqs_map[&v].clone());
                let mut req_and = vec![entrance_or, exit_or];
                let vertex_key = &self.vertex_isv.keys[v];
                if heated && vertex_key.node_id == node_id {
                    // If the comeInNormally and leaveNormally strats are at the door node,
                    // assume an implicit requirement of 40 heat frames to open the door:
                    req_and.push(Requirement::HeatFrames(40));
                }
                req_or.push(Requirement::make_and(req_and));
            }

            let reset_room_req =
                Self::process_reset_room_req(&Requirement::make_or(req_or), (room_id, node_id));

            self.node_reset_room_requirement
                .insert((room_id, node_id), reset_room_req);
        }
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

    pub fn load_room_name_font(&mut self, path: &Path) -> Result<()> {
        let img = read_image(path)?;
        let dim = img.dim();
        let char_map = [
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "abcdefghijklmnopqrstuvwxyz",
            "0123456789-'.",
        ];
        assert!(dim.1 % 8 == 0);
        assert!(dim.0 % 8 == 0);
        assert!(dim.1 / 8 == char_map[0].len());
        assert!(dim.0 / 8 == char_map.len());
        let mut gfx: Vec<GfxTile1Bpp> = vec![];
        let mut widths: Vec<u8> = vec![];
        let mut char_isv: IndexedVec<char> = IndexedVec::default();

        // Add a space character:
        gfx.push([0; 8]);
        widths.push(4);
        char_isv.add(&' ');

        for cy in 0..dim.0 / 8 {
            let char_row = &char_map[cy];
            for cx in 0..char_row.len() {
                let mut tile: GfxTile1Bpp = [0u8; 8];
                let mut max_x = 0;
                for py in 0..8 {
                    for px in 0..8 {
                        let y = cy * 8 + py;
                        let x = cx * 8 + px;
                        let r = img[(y, x, 0)];
                        let g = img[(y, x, 1)];
                        let b = img[(y, x, 2)];
                        if (r, g, b) == (255, 255, 255) {
                            tile[py] |= 1 << (7 - px);
                            max_x = max_x.max(px);
                        }
                    }
                }
                let pos = char_isv.add(&(char_row.as_bytes()[cx] as char));
                assert_eq!(pos, gfx.len());
                gfx.push(tile);
                widths.push(max_x as u8 + 2);
            }
        }

        self.room_name_font = VariableWidthFont {
            gfx,
            widths,
            char_isv,
        };
        Ok(())
    }

    fn load_reduced_flashing_patch(&mut self, path: &Path) -> Result<()> {
        let reduced_flashing_str = std::fs::read_to_string(path).with_context(|| {
            format!(
                "Unable to load reduced flashing patch at {}",
                path.display()
            )
        })?;
        self.reduced_flashing_patch = serde_json::from_str(&reduced_flashing_str)?;
        Ok(())
    }

    fn load_strat_videos(&mut self, path: &Path) -> Result<()> {
        let strat_videos_str = std::fs::read_to_string(path)
            .with_context(|| format!("Unable to load strat videos at {}", path.display()))?;
        let strat_videos: Vec<StratVideo> = serde_json::from_str(&strat_videos_str)?;
        for video in strat_videos {
            self.strat_videos
                .entry((video.room_id, video.strat_id))
                .or_default()
                .push(video);
        }
        Ok(())
    }

    fn load_map_tile_data(&mut self, path: &Path) -> Result<()> {
        let map_tile_data_str = std::fs::read_to_string(path)
            .with_context(|| format!("Unable to load map tile data at {}", path.display()))?;
        let map_tile_data_file: MapTileDataFile = serde_json::from_str(&map_tile_data_str)?;
        self.map_tile_data = map_tile_data_file.rooms;
        for room in &mut self.map_tile_data {
            for tile in &mut room.map_tiles {
                tile.heated = room.heated;
                if let Some(liquid_level) = room.liquid_level {
                    if (tile.coords.1 as f32) <= liquid_level - 1.0 {
                        tile.liquid_level = None;
                    } else if (tile.coords.1 as f32) >= liquid_level {
                        tile.liquid_level = Some(0.0);
                    } else {
                        tile.liquid_level = Some(liquid_level.fract());
                    }
                }
                if tile.liquid_level.is_some() {
                    tile.liquid_type = room.liquid_type;
                }
            }
        }
        Ok(())
    }

    pub fn patch_helpers(&mut self) {
        let must_patch_categories = ["Randomizer Dependent".to_string(), "Leniency".to_string()];
        let helper_substitutions: Vec<(&'static str, JsonValue)> = vec![
            ("h_heatProof", json::array!["Varia"]),
            ("h_heatResistant", json::array!["Varia"]),
            ("h_lavaProof", json::array!["Varia", "Gravity"]),
            (
                "h_fullEnemyDamageReduction",
                json::array!["Varia", "Gravity"],
            ),
            (
                "h_blueGateGlitchLeniency",
                json::array!["i_blueGateGlitchLeniency"],
            ),
            (
                "h_greenGateGlitchLeniency",
                json::array!["i_greenGateGlitchLeniency"],
            ),
            (
                "h_heatedBlueGateGlitchLeniency",
                json::array!["i_heatedBlueGateGlitchLeniency"],
            ),
            (
                "h_heatedGreenGateGlitchLeniency",
                json::array!["i_heatedGreenGateGlitchLeniency"],
            ),
            (
                "h_bombIntoCrystalFlashClipLeniency",
                json::array!["i_bombIntoCrystalFlashClipLeniency"],
            ),
            (
                "h_jumpIntoCrystalFlashClipLeniency",
                json::array!["i_jumpIntoCrystalFlashClipLeniency"],
            ),
            (
                "h_spikeSuitSpikeHitLeniency",
                json::array!["i_spikeSuitSpikeHitLeniency"],
            ),
            (
                "h_spikeSuitThornHitLeniency",
                json::array!["i_spikeSuitThornHitLeniency"],
            ),
            (
                "h_spikeSuitSamusEaterLeniency",
                json::array!["i_spikeSuitSamusEaterLeniency"],
            ),
            (
                "h_spikeSuitPowerBombLeniency",
                json::array!["i_spikeSuitPowerBombLeniency"],
            ),
            (
                "h_XModeSpikeHitLeniency",
                json::array!["i_XModeSpikeHitLeniency"],
            ),
            (
                "h_XModeThornHitLeniency",
                json::array!["i_XModeThornHitLeniency"],
            ),
            (
                "h_thornXModeFramePerfectExtraLeniency",
                json::array!["i_FramePerfectXModeThornHitLeniency"],
            ),
            (
                "h_thornDoubleXModeFramePerfectExtraLeniency",
                json::array!["i_FramePerfectDoubleXModeThornHitLeniency"],
            ),
            (
                "h_speedKeepSpikeHitLeniency",
                json::array!["i_speedKeepSpikeHitLeniency"],
            ),
            ("h_allItemsSpawned", json::array!["f_AllItemsSpawn"]),
            ("h_EverestMorphTunnelExpanded", json::array![]),
            ("h_activateBombTorizo", json::array![]),
            (
                "h_activateAcidChozo",
                json::array![{
                    "or": ["SpaceJump", "f_AcidChozoWithoutSpaceJump"]
                }],
            ),
            ("h_ShaktoolVanillaFlag", json::array!["never"]),
            ("h_ShaktoolCameraFix", json::array![]),
            ("h_KraidCameraFix", json::array![]),
            ("h_CrocomireCameraFix", json::array![]),
            ("h_ShaktoolSymmetricFlag", json::array![]),
            ("h_ClimbWithoutLava", json::array!["i_ClimbWithoutLava"]),
            (
                "h_SupersDoubleDamageMotherBrain",
                json::array!["i_SupersDoubleDamageMotherBrain"],
            ),
            ("h_useMissileRefillStation", json::array!["i_ammoRefill"]),
            (
                "h_MissileRefillStationAllAmmo",
                json::array!["i_ammoRefillAll"],
            ),
            (
                "h_useEnergyRefillStation",
                json::array!["i_energyStationRefill"],
            ),
            ("h_openTourianEscape1RightDoor", json::array![]),
            (
                "h_LowerNorfairElevatorDownwardFrames",
                json::array!["i_LowerNorfairElevatorDownwardFrames"],
            ),
            (
                "h_LowerNorfairElevatorUpwardFrames",
                json::array!["i_LowerNorfairElevatorUpwardFrames"],
            ),
            (
                "h_MainHallElevatorFrames",
                json::array!["i_MainHallElevatorFrames"],
            ),
            (
                "h_equipmentScreenCycleFrames",
                json::array!["i_equipmentScreenCycleFrames"],
            ),
            (
                "h_ShinesparksCostEnergy",
                json::array!["i_ShinesparksCostEnergy"],
            ),
            (
                "h_ElevatorCrystalFlashLeniency",
                json::array!["i_elevatorCrystalFlashLeniency"],
            ),
            (
                "h_heatedCrystalFlashRefill",
                json::array![
                    {"or": [
                        {"partialRefill": {"type": "Energy", "limit": 1440}},
                        {"and": [
                            "Varia",
                            {"partialRefill": {"type": "Energy", "limit": 1500}}
                        ]}
                    ]}
                ],
            ),
            (
                "h_acidCrystalFlashRefill",
                json::array![
                    {"or": [
                        {"partialRefill": {"type": "Energy", "limit": 1120}},
                        {"and": [
                            {"or": [
                                "Varia",
                                "Gravity"
                            ]},
                            {"partialRefill": {"type": "Energy", "limit": 1310}}
                        ]},
                        {"and": [
                            "Varia",
                            "Gravity",
                            {"partialRefill": {"type": "Energy", "limit": 1410}}
                        ]}
                    ]}
                ],
            ),
            (
                "h_heatedLavaCrystalFlashRefill",
                json::array![
                    {"or": [
                        {"partialRefill": {"type": "Energy", "limit": 1330}},
                        {"and": [
                            "Gravity",
                            {"partialRefill": {"type": "Energy", "limit": 1385}}
                        ]},
                        {"and": [
                            "Varia",
                            {"partialRefill": {"type": "Energy", "limit": 1440}}
                        ]},
                        {"and": [
                            "Varia",
                            "Gravity",
                            {"partialRefill": {"type": "Energy", "limit": 1500}}
                        ]}
                    ]}
                ],
            ),
            (
                "h_heatedAcidCrystalFlashRefill",
                json::array![
                    {"or": [
                        {"partialRefill": {"type": "Energy", "limit": 1075}},
                        {"and": [
                            "Varia",
                            {"partialRefill": {"type": "Energy", "limit": 1310}}
                        ]},
                        {"and": [
                            "Gravity",
                            {"partialRefill": {"type": "Energy", "limit": 1365}}
                        ]}
                    ]}
                ],
            ),
        ];
        let skip_helpers = vec![
            "h_SpringwallOverSpikes",
            "h_heatedSpringwall",
            "h_heatedIBJFromSpikes",
            "h_equipmentScreenFix",
            "h_bypassMotherBrainRoom",
            "h_partialEnemyDamageReduction",
            "h_doorImmediatelyClosedFix",
            "h_openZebetitesLeniency",
            "h_CrystalSparkLeniency",
            "h_IBJFromThorns",
            "h_IBJFromSpikes",
            "h_doubleEquipmentScreenCycleFrames",
            "h_extendedMoondanceBeetomLeniency",
            "h_RefillStationAllAmmo10PowerBombCrystalFlash",
        ];

        // Construct the set of helpers that need to be patched, and make sure they all are.
        let mut must_patch_helper_set: HashSet<String> = HashSet::new();
        for helper_json in self.helper_json_map.values_mut() {
            let helper_name = helper_json["name"].as_str().unwrap().to_string();
            let helper_category = &self.helper_category_map[&helper_name];
            if must_patch_categories.contains(helper_category) {
                must_patch_helper_set.insert(helper_name.clone());
            }
        }
        for (helper_name, new_requires) in helper_substitutions {
            self.helper_json_map
                .get_mut(helper_name)
                .with_context(|| format!("Helper {} not found", helper_name))
                .unwrap()["requires"] = new_requires;
            if !must_patch_helper_set.remove(helper_name) {
                panic!("Helper {} already patched.", helper_name)
            }
        }
        for helper_name in skip_helpers {
            if !must_patch_helper_set.remove(helper_name) {
                panic!(
                    "Helper {} either does not exist or was already patched or skipped.",
                    helper_name
                )
            }
        }
        for helper_name in must_patch_helper_set.iter() {
            error!("Randomizer-dependent helper {} is not patched", helper_name);
        }
        if !must_patch_helper_set.is_empty() {
            panic!("Randomizer-dependent helpers are not all patched");
        }
    }

    pub fn load_minimal(base_path: &Path) -> Result<GameData> {
        let sm_json_data_path = base_path.join("../sm-json-data");
        let mut game_data = GameData {
            sm_json_data_path,
            ..GameData::default()
        };

        game_data.load_items_and_flags()?;
        game_data.load_tech()?;
        game_data.load_helpers()?;
        game_data.patch_helpers();
        game_data.load_weapons()?;
        game_data.load_enemies()?;

        Ok(game_data)
    }

    pub fn load(base_path: &Path) -> Result<GameData> {
        let mut game_data = Self::load_minimal(base_path)?;

        let room_geometry_path = base_path.join("../room_geometry.json");
        let escape_timings_path = base_path.join("data/escape_timings.json");
        let start_locations_path = base_path.join("data/start_locations.json");
        let title_screen_path = base_path.join("../TitleScreen/Images");
        let room_name_font_path = base_path.join("data/room_name_font.png");
        let reduced_flashing_path = base_path.join("data/reduced_flashing.json");
        let strat_videos_path = base_path.join("data/strat_videos.json");
        let map_tile_path = base_path.join("data/map_tiles.json");

        game_data.load_reduced_flashing_patch(&reduced_flashing_path)?;
        game_data.load_strat_videos(&strat_videos_path)?;

        game_data.area_order = vec![
            "Central Crateria",
            "West Crateria",
            "East Crateria",
            "Blue Brinstar",
            "Green Brinstar",
            "Pink Brinstar",
            "Red Brinstar",
            "Kraid Brinstar",
            "East Upper Norfair",
            "West Upper Norfair",
            "Crocomire Upper Norfair",
            "West Lower Norfair",
            "East Lower Norfair",
            "Wrecked Ship",
            "Outer Maridia",
            "Pink Inner Maridia",
            "Yellow Inner Maridia",
            "Green Inner Maridia",
            "Tourian",
        ]
        .into_iter()
        .map(|x| x.to_string())
        .collect();

        let room_pattern =
            game_data.sm_json_data_path.to_str().unwrap().to_string() + "/region/**/*.json";
        game_data.load_rooms(&room_pattern)?;
        game_data.load_connections()?;
        game_data.extract_all_tech_dependencies()?;
        game_data.extract_all_strat_dependencies()?;

        game_data
            .load_room_geometry(&room_geometry_path)
            .context("Unable to load room geometry")?;
        game_data.load_escape_timings(&escape_timings_path)?;
        game_data.load_start_locations(&start_locations_path)?;
        game_data.load_hub_locations()?;
        game_data.load_map_tile_data(&map_tile_path)?;
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
        game_data.load_title_screens(&title_screen_path)?;
        game_data.load_room_name_font(&room_name_font_path)?;

        // for link in &game_data.links {
        //     let from_vertex_id = link.from_vertex_id;
        //     let from_vertex_key = &game_data.vertex_isv.keys[from_vertex_id];
        //     let to_vertex_id = link.to_vertex_id;
        //     let to_vertex_key = &game_data.vertex_isv.keys[to_vertex_id];
        //     if (from_vertex_key.room_id, from_vertex_key.node_id) == (44, 12)
        //         && (to_vertex_key.room_id, to_vertex_key.node_id) == (44, 12)
        //     {
        //         println!(
        //             "From: {:?}\nTo: {:?}\nLink: {:?}\n",
        //             from_vertex_key, to_vertex_key, link
        //         );
        //     }
        // }

        // List the longest node names:
        // let mut node_names: Vec<String> = vec![];
        // for node in game_data.node_json_map.values() {
        //     node_names.push(node["name"].as_str().unwrap().to_string());
        // }
        // node_names.sort_by_key(|x| -(x.len() as isize));
        // for name in node_names.iter().take(100) {
        //     println!("{}", name);
        // }

        Ok(game_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex() {
        let v = JsonValue::String("0.5".to_string());
        let h = parse_hex(&v, 0.0);
        assert!(h.is_err());

        let v = JsonValue::Null;
        let h = parse_hex(&v, 0.0);
        assert!(h.is_ok());
        assert!(h.unwrap() == 0.0);

        let v = JsonValue::String("$2.8".to_string());
        let h = parse_hex(&v, 0.0);
        assert!(h.is_ok());
        assert_eq!(h.unwrap(), 2.5);
    }
}
