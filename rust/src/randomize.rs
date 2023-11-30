pub mod escape_timer;

use crate::{
    game_data::{
        get_effective_runway_length, Capacity, DoorPosition, DoorPtrPair, EntranceCondition,
        ExitCondition, FlagId, GModeMobility, GModeMode, HubLocation, Item, ItemId, ItemLocationId,
        Link, LinkIdx, LinksDataGroup, Map, NodeId, Physics, Requirement, RoomGeometryRoomIdx,
        RoomId, StartLocation, VertexId,
    },
    traverse::{
        apply_requirement, get_bireachable_idxs, get_spoiler_route, traverse, GlobalState,
        LocalState, TraverseResult, IMPOSSIBLE_LOCAL_STATE, NUM_COST_METRICS,
    },
    web::logic::strip_name,
};
use anyhow::{bail, Context, Result};
use by_address::ByAddress;
use hashbrown::{HashMap, HashSet};
use log::info;
use rand::SeedableRng;
use rand::{seq::SliceRandom, Rng};
use serde_derive::{Deserialize, Serialize};
use std::{
    cmp::{max, min},
    convert::TryFrom,
    iter,
};
use strum::VariantNames;

use crate::game_data::GameData;

use self::escape_timer::SpoilerEscape;

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ProgressionRate {
    Slow,
    Uniform,
    Fast,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ItemPlacementStyle {
    Neutral,
    Forced,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ItemMarkers {
    Simple,
    Majors,
    Uniques,
    ThreeTiered,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ItemDotChange {
    Fade,
    Disappear,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum Objectives {
    Bosses,
    Minibosses,
    Metroids,
    Chozos,
    Pirates,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum DoorsMode {
    Blue,
    Ammo,
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
    Disabled,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum SaveAnimals {
    No,
    Maybe,
    Yes,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum MotherBrainFight {
    Vanilla,
    Short,
    Skip,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct DebugOptions {
    pub new_game_extra: bool,
    pub extended_spoiler: bool,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ItemPriorityGroup {
    pub name: String,
    pub items: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct DifficultyConfig {
    pub tech: Vec<String>,
    pub notable_strats: Vec<String>,
    // pub notable_strats: Vec<String>,
    pub shine_charge_tiles: f32,
    pub progression_rate: ProgressionRate,
    pub random_tank: bool,
    pub item_placement_style: ItemPlacementStyle,
    pub item_priorities: Vec<ItemPriorityGroup>,
    pub semi_filler_items: Vec<Item>,
    pub filler_items: Vec<Item>,
    pub early_filler_items: Vec<Item>,
    pub resource_multiplier: f32,
    pub gate_glitch_leniency: i32,
    pub door_stuck_leniency: i32,
    pub escape_timer_multiplier: f32,
    pub phantoon_proficiency: f32,
    pub draygon_proficiency: f32,
    pub ridley_proficiency: f32,
    pub botwoon_proficiency: f32,
    // Quality-of-life options:
    pub supers_double: bool,
    pub mother_brain_fight: MotherBrainFight,
    pub escape_movement_items: bool,
    pub escape_refill: bool,
    pub escape_enemies_cleared: bool,
    pub mark_map_stations: bool,
    pub transition_letters: bool,
    pub item_markers: ItemMarkers,
    pub item_dot_change: ItemDotChange,
    pub all_items_spawn: bool,
    pub acid_chozo: bool,
    pub buffed_drops: bool,
    pub fast_elevators: bool,
    pub fast_doors: bool,
    pub fast_pause_menu: bool,
    pub respin: bool,
    pub infinite_space_jump: bool,
    pub momentum_conservation: bool,
    // Game variations:
    pub objectives: Objectives,
    pub doors_mode: DoorsMode,
    pub randomized_start: bool,
    pub save_animals: SaveAnimals,
    pub early_save: bool,
    pub area_assignment: AreaAssignment,
    pub wall_jump: WallJump,
    pub maps_revealed: bool,
    pub vanilla_map: bool,
    pub ultra_low_qol: bool,
    // Presets:
    pub skill_assumptions_preset: Option<String>,
    pub item_progression_preset: Option<String>,
    pub quality_of_life_preset: Option<String>,
    // Debug:
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_options: Option<DebugOptions>,
}

// Includes preprocessing specific to the map:
pub struct Randomizer<'a> {
    pub map: &'a Map,
    pub locked_doors: &'a [LockedDoor], // Locked doors (not including gray doors)
    pub game_data: &'a GameData,
    pub difficulty_tiers: &'a [DifficultyConfig],
    pub base_links_data: &'a LinksDataGroup,
    pub seed_links_data: LinksDataGroup,
    pub initial_items_remaining: Vec<usize>, // Corresponds to GameData.items_isv (one count per distinct item name)
}

#[derive(Clone)]
struct ItemLocationState {
    pub placed_item: Option<Item>,
    pub collected: bool,
    pub reachable: bool,
    pub bireachable: bool,
    pub bireachable_vertex_id: Option<VertexId>,
}

#[derive(Clone)]
struct FlagLocationState {
    pub bireachable: bool,
    pub bireachable_vertex_id: Option<VertexId>,
}

#[derive(Clone)]
struct SaveLocationState {
    pub bireachable: bool,
}

#[derive(Clone)]
struct DebugData {
    global_state: GlobalState,
    forward: TraverseResult,
    reverse: TraverseResult,
}

#[derive(Clone, Copy, Debug)]
pub enum DoorType {
    Red,
    Green,
    Yellow,
}

#[derive(Clone, Copy)]
pub struct LockedDoor {
    pub src_ptr_pair: DoorPtrPair,
    pub dst_ptr_pair: DoorPtrPair,
    pub door_type: DoorType,
    pub bidirectional: bool, // if true, the door is locked on both sides, with a shared state
}

// State that changes over the course of item placement attempts
struct RandomizationState {
    step_num: usize,
    start_location: StartLocation,
    hub_location: HubLocation,
    item_precedence: Vec<Item>, // An ordering of the 21 distinct item names. The game will prioritize placing key items earlier in the list.
    save_location_state: Vec<SaveLocationState>, // Corresponds to GameData.item_locations (one record for each of 100 item locations)
    item_location_state: Vec<ItemLocationState>, // Corresponds to GameData.item_locations (one record for each of 100 item locations)
    flag_location_state: Vec<FlagLocationState>, // Corresponds to GameData.flag_locations
    items_remaining: Vec<usize>, // Corresponds to GameData.items_isv (one count for each of 21 distinct item names)
    global_state: GlobalState,
    debug_data: Option<DebugData>,
    previous_debug_data: Option<DebugData>,
    key_visited_vertices: HashSet<usize>,
}

pub struct Randomization {
    pub difficulty: DifficultyConfig,
    pub map: Map,
    pub locked_doors: Vec<LockedDoor>,
    pub item_placement: Vec<Item>,
    pub start_location: StartLocation,
    pub spoiler_log: SpoilerLog,
    pub seed: usize,
    pub display_seed: usize,
}

struct SelectItemsOutput {
    key_items: Vec<Item>,
    other_items: Vec<Item>,
    new_items_remaining: Vec<usize>,
}

struct VertexInfo {
    area_name: String,
    room_name: String,
    room_coords: (usize, usize),
    node_name: String,
    node_id: usize,
}

pub fn randomize_map_areas(map: &mut Map, seed: usize) {
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let mut area_mapping: Vec<usize> = (0..6).collect();
    area_mapping.shuffle(&mut rng);

    let mut subarea_mapping: Vec<Vec<usize>> = vec![(0..2).collect(); 6];
    for i in 0..6 {
        subarea_mapping[i].shuffle(&mut rng);
    }

    for i in 0..map.area.len() {
        map.area[i] = area_mapping[map.area[i]];
        map.subarea[i] = subarea_mapping[map.area[i]][map.subarea[i]];
    }
}

fn get_door_requirement(
    locked_door_idx: Option<usize>,
    locked_doors: &[LockedDoor],
    game_data: &GameData,
) -> Requirement {
    if let Some(idx) = locked_door_idx {
        let locked_door = &locked_doors[idx];
        let ptr_pair = locked_door.src_ptr_pair;
        let (room_idx, _) = game_data.room_and_door_idxs_by_door_ptr_pair[&ptr_pair];
        let heated = game_data.room_geometry[room_idx].heated;
        match locked_door.door_type {
            DoorType::Red => {
                if heated {
                    Requirement::Or(vec![
                        Requirement::And(vec![
                            Requirement::Missiles(5),
                            Requirement::HeatFrames(50),
                        ]),
                        Requirement::Supers(1),
                    ])
                } else {
                    Requirement::Or(vec![Requirement::Missiles(5), Requirement::Supers(1)])
                }
            }
            DoorType::Green => Requirement::Supers(1),
            DoorType::Yellow => {
                if heated {
                    Requirement::And(vec![
                        Requirement::Item(Item::Morph as ItemId),
                        Requirement::PowerBombs(1),
                        Requirement::HeatFrames(110),
                    ])
                } else {
                    Requirement::And(vec![
                        Requirement::Item(Item::Morph as ItemId),
                        Requirement::PowerBombs(1),
                    ])
                }
            }
        }
    } else {
        Requirement::Free
    }
}

fn add_door_links(
    src_room_id: RoomId,
    src_node_id: NodeId,
    dst_room_id: RoomId,
    dst_node_id: NodeId,
    locked_door_idx: Option<usize>,
    game_data: &GameData,
    links: &mut Vec<Link>,
    locked_doors: &[LockedDoor],
) {
    for obstacle_bitmask in 0..(1 << game_data.room_num_obstacles[&src_room_id]) {
        let from_vertex_id =
            game_data.vertex_isv.index_by_key[&(src_room_id, src_node_id, obstacle_bitmask)];
        let dst_node_id_spawn = *game_data
            .node_spawn_at_map
            .get(&(dst_room_id, dst_node_id))
            .unwrap_or(&dst_node_id);
        let to_vertex_id = game_data.vertex_isv.index_by_key[&(dst_room_id, dst_node_id_spawn, 0)];

        links.push(Link {
            from_vertex_id,
            to_vertex_id,
            requirement: get_door_requirement(locked_door_idx, locked_doors, game_data),
            entrance_condition: None,
            bypasses_door_shell: false,
            notable_strat_name: None,
            strat_name: "(Door transition)".to_string(),
            strat_notes: vec![],
            sublinks: vec![],
        });
    }
}

fn compute_run_frames(tiles: f32) -> i32 {
    let frames = if tiles <= 7.0 {
        9.0 + 4.0 * tiles
    } else if tiles <= 16.0 {
        15.0 + 3.0 * tiles
    } else if tiles <= 42.0 {
        32.0 + 2.0 * tiles
    } else {
        47.0 + 64.0 / 39.0 * tiles
    };
    frames.ceil() as i32
}

struct Preprocessor<'a> {
    game_data: &'a GameData,
    door_map: HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
    locked_doors: &'a [LockedDoor],
    locked_node_map: HashMap<(RoomId, NodeId), usize>,
    // Cache of previously-processed or currently-processing inputs. This is used to avoid infinite
    // recursion in cases of circular dependencies (e.g. cycles of leaveWithGMode)
    // This is the old-style cross-room logical requirements (in process of being deprecated):
    preprocessed_output: HashMap<ByAddress<&'a Requirement>, Option<Requirement>>,
    // Similar cache for the new style of cross-room strats. The keys are references to links with
    // entrance conditions that need to be resolved: the values are lists of possible resolutions of the link
    // (which have no entrance conditions), where `from_vertex_id` has been replaced by one in a connecting
    // room with a matching exit condition, and any necessary requirements from the cross-room strat have
    // been added):
    preprocessed_links: HashMap<ByAddress<&'a Link>, Vec<Link>>,
}

// // TODO: Remove this if heatFrames are removed from runways in sm-json-data.
// // (but we might want to keep them for canComeInCharged?)
// fn strip_heat_frames(req: &Requirement) -> Requirement {
//     match req {
//         Requirement::HeatFrames(_) => Requirement::Free,
//         Requirement::And(sub_reqs) => {
//             Requirement::make_and(sub_reqs.iter().map(strip_heat_frames).collect())
//         }
//         Requirement::Or(sub_reqs) => {
//             Requirement::make_or(sub_reqs.iter().map(strip_heat_frames).collect())
//         }
//         _ => req.clone(),
//     }
// }

fn compute_shinecharge_frames(other_runway_length: f32, runway_length: f32) -> (i32, i32) {
    let combined_length = other_runway_length + runway_length;
    if combined_length > 31.3 {
        // Dash can be held the whole time:
        let total_time = compute_run_frames(combined_length);
        let other_time = compute_run_frames(other_runway_length);
        return (other_time, total_time - other_time);
    }
    // Combined runway is too short to hold dash the whole time. A shortcharge is needed:
    let total_time = 85.0; // 85 frames to charge a shinespark (assuming a good enough 1-tap)
    let initial_speed = 0.125;
    let acceleration =
        2.0 * (combined_length - initial_speed * total_time) / (total_time * total_time);
    let other_time =
        (f32::sqrt(initial_speed * initial_speed + 2.0 * acceleration * other_runway_length)
            - initial_speed)
            / acceleration;
    let other_time = other_time.ceil() as i32;
    (other_time, total_time as i32 - other_time)
}

impl<'a> Preprocessor<'a> {
    pub fn new(
        game_data: &'a GameData,
        map: &'a Map,
        locked_doors: &'a [LockedDoor],
        locked_door_map: &'a HashMap<DoorPtrPair, usize>,
    ) -> Self {
        let mut door_map: HashMap<(RoomId, NodeId), (RoomId, NodeId)> = HashMap::new();
        let mut locked_node_map: HashMap<(RoomId, NodeId), usize> = HashMap::new();
        for &((src_exit_ptr, src_entrance_ptr), (dst_exit_ptr, dst_entrance_ptr), bidirectional) in
            &map.doors
        {
            if !bidirectional {
                // For now we omit sand connections from cross-room strats, because the fact that you can't
                // go back up would make the strats unsound (with the current way cross-room strats are interpreted)
                continue;
            }
            let (src_room_id, src_node_id) =
                game_data.door_ptr_pair_map[&(src_exit_ptr, src_entrance_ptr)];
            let (_, unlocked_src_node_id) =
                game_data.unlocked_door_ptr_pair_map[&(src_exit_ptr, src_entrance_ptr)];
            let (dst_room_id, dst_node_id) =
                game_data.door_ptr_pair_map[&(dst_exit_ptr, dst_entrance_ptr)];
            let (_, unlocked_dst_node_id) =
                game_data.unlocked_door_ptr_pair_map[&(dst_exit_ptr, dst_entrance_ptr)];
            // println!("({}, {}) <-> ({}, {})", src_room_id, src_node_id, dst_room_id, dst_node_id);
            door_map.insert(
                (src_room_id, unlocked_src_node_id),
                (dst_room_id, dst_node_id),
            );
            door_map.insert(
                (dst_room_id, unlocked_dst_node_id),
                (src_room_id, src_node_id),
            );

            if let Some(&idx) = locked_door_map.get(&(src_exit_ptr, src_entrance_ptr)) {
                locked_node_map.insert((src_room_id, unlocked_src_node_id), idx);
            }
            if let Some(&idx) = locked_door_map.get(&(dst_exit_ptr, dst_entrance_ptr)) {
                locked_node_map.insert((dst_room_id, unlocked_dst_node_id), idx);
            }
        }
        Preprocessor {
            game_data,
            door_map,
            locked_doors,
            locked_node_map,
            preprocessed_output: HashMap::new(),
            preprocessed_links: HashMap::new(),
        }
    }

    fn get_come_in_running_reqs(
        &self,
        exit_link: &Link,
        exit_condition: &ExitCondition,
        speed_booster: Option<bool>,
        min_tiles: f32,
        max_tiles: f32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                effective_length,
                heated,
                physics,
            } => {
                if *effective_length < min_tiles {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![];
                if speed_booster == Some(true) {
                    reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
                }
                if speed_booster == Some(false) {
                    reqs.push(Requirement::Tech(
                        self.game_data.tech_isv.index_by_key["canDisableEquipment"],
                    ));
                }
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                    // TODO: in sm-json-data, add physics property to leaveWithRunway schema (for door nodes with multiple possible physics)
                }
                if *heated {
                    let heat_frames = if exit_link.from_vertex_id == exit_link.to_vertex_id {
                        compute_run_frames(min_tiles) * 2 + 20
                    } else {
                        if *effective_length > max_tiles {
                            // 10 heat frames to position after stopping on a dime, before resuming running
                            compute_run_frames(*effective_length - max_tiles)
                                + compute_run_frames(max_tiles)
                                + 10
                        } else {
                            compute_run_frames(*effective_length)
                        }
                    };
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_shinecharging_reqs(
        &self,
        exit_link: &Link,
        exit_condition: &ExitCondition,
        runway_length: f32,
        runway_heated: bool,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                effective_length,
                heated,
                physics,
            } => {
                let mut reqs: Vec<Requirement> = vec![];
                let combined_runway_length = *effective_length + runway_length;
                reqs.push(Requirement::make_shinecharge(combined_runway_length));
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if exit_link.from_vertex_id == exit_link.to_vertex_id {
                    // Runway in the other room starts and ends at the door so we need to run both directions:
                    if runway_heated && *heated {
                        // Both rooms are heated. Heat frames are optimized by minimizing runway usage in the source room.
                        // But since the shortcharge difficulty is not known here, we conservatively assume up to 33 tiles
                        // of the combined runway may need to be used. (TODO: Instead add a Requirement enum case to handle this more accurately.)
                        let other_runway_length =
                            f32::max(0.0, f32::min(*effective_length, 33.0 - runway_length));
                        let heat_frames_1 = compute_run_frames(other_runway_length) + 20;
                        let heat_frames_2 =
                            i32::max(85, compute_run_frames(other_runway_length + runway_length));
                        // Add 5 lenience frames (partly to account for the possibility of some inexactness in our calculations)
                        reqs.push(Requirement::HeatFrames(heat_frames_1 + heat_frames_2 + 5));
                    } else if !runway_heated && *heated {
                        // Only the destination room is heated. Heat frames are optimized by using the full runway in
                        // the source room.
                        let (_, heat_frames) =
                            compute_shinecharge_frames(*effective_length, runway_length);
                        reqs.push(Requirement::HeatFrames(heat_frames + 5));
                    } else if runway_heated && !*heated {
                        // Only the source room is heated. As in the first case above, heat frames are optimized by
                        // minimizing runway usage in the source room. (TODO: Use new Requirement enum case.)
                        let other_runway_length =
                            f32::max(0.0, f32::min(*effective_length, 33.0 - runway_length));
                        let heat_frames_1 = compute_run_frames(other_runway_length) + 20;
                        let (heat_frames_2, _) =
                            compute_shinecharge_frames(other_runway_length, runway_length);
                        reqs.push(Requirement::HeatFrames(heat_frames_1 + heat_frames_2 + 5));
                    }
                } else if runway_heated || *heated {
                    // Runway in the other room starts at a different node and runs toward the door. The full combined
                    // runway is used.
                    let (frames_1, frames_2) =
                        compute_shinecharge_frames(*effective_length, runway_length);
                    let mut heat_frames = 5;
                    if *heated {
                        // Heat frames for source room
                        heat_frames += frames_1;
                    }
                    if runway_heated {
                        // Heat frames for destination room
                        heat_frames += frames_2;
                    }
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_shinecharged_reqs(
        &self,
        exit_link: &Link,
        exit_condition: &ExitCondition,
        frames_required: i32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveShinecharged { frames_remaining } => {
                if *frames_remaining < frames_required {
                    None
                } else {
                    Some(Requirement::Free)
                }
            }
            ExitCondition::LeaveWithRunway {
                effective_length,
                heated,
                physics,
            } => {
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::make_shinecharge(*effective_length));
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated {
                    if exit_link.from_vertex_id == exit_link.to_vertex_id {
                        let runway_length = f32::min(33.0, *effective_length);
                        let run_frames = compute_run_frames(runway_length);
                        let heat_frames_1 = run_frames + 20;
                        let heat_frames_2 = i32::max(85, run_frames);
                        reqs.push(Requirement::HeatFrames(heat_frames_1 + heat_frames_2 + 5));
                    } else {
                        let heat_frames = i32::max(85, compute_run_frames(*effective_length));
                        reqs.push(Requirement::HeatFrames(heat_frames + 5));
                    }
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_stutter_shinecharging_reqs(
        &self,
        exit_link: &Link,
        exit_condition: &ExitCondition,
        min_tiles: f32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                effective_length,
                heated,
                physics,
            } => {
                if *physics != Some(Physics::Air) {
                    return None;
                }
                if *effective_length < min_tiles {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key["canStutterWaterShineCharge"],
                ));
                reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
                if *heated {
                    let heat_frames = if exit_link.from_vertex_id == exit_link.to_vertex_id {
                        compute_run_frames(min_tiles) * 2 + 20
                    } else {
                        compute_run_frames(*effective_length)
                    };
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_spark_reqs(
        &self,
        exit_link: &Link,
        exit_condition: &ExitCondition,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithSpark {} => Some(Requirement::Free),
            ExitCondition::LeaveShinecharged { .. } => Some(Requirement::Free),
            ExitCondition::LeaveWithRunway {
                effective_length,
                heated,
                physics,
            } => {
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::make_shinecharge(*effective_length));
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated {
                    if exit_link.from_vertex_id == exit_link.to_vertex_id {
                        let runway_length = f32::min(33.0, *effective_length);
                        let run_frames = compute_run_frames(runway_length);
                        let heat_frames_1 = run_frames + 20;
                        let heat_frames_2 = i32::max(85, run_frames);
                        reqs.push(Requirement::HeatFrames(heat_frames_1 + heat_frames_2 + 5));
                    } else {
                        let heat_frames = i32::max(85, compute_run_frames(*effective_length));
                        reqs.push(Requirement::HeatFrames(heat_frames + 5));
                    }
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_bomb_boost_reqs(
        &self,
        exit_link: &Link,
        exit_condition: &ExitCondition,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                effective_length,
                heated,
                physics,
            } => {
                let mut reqs: Vec<Requirement> = vec![];
                if *physics != Some(Physics::Air) {
                    return None;
                }
                reqs.push(Requirement::And(vec![
                    Requirement::Item(Item::Morph as ItemId),
                    Requirement::Or(vec![
                        Requirement::Item(Item::Bombs as ItemId),
                        Requirement::PowerBombs(1),
                    ]),
                ]));
                if *heated {
                    let mut heat_frames = 100;
                    if exit_link.from_vertex_id != exit_link.to_vertex_id {
                        heat_frames += compute_run_frames(*effective_length);
                    }
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_door_stuck_setup_reqs(
        &self,
        exit_link: &Link,
        exit_condition: &ExitCondition,
        entrance_heated: bool,
    ) -> Option<Requirement> {
        let (room_id, node_id, _) = self.game_data.vertex_isv.keys[exit_link.to_vertex_id];
        let door_position = *self
            .game_data
            .door_position
            .get(&(room_id, node_id))
            .expect(&format!(
                "door_position not found for ({}, {})",
                room_id, node_id
            ));
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                heated, physics, ..
            } => {
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key["canStationarySpinJump"],
                ));
                if door_position == DoorPosition::Right {
                    reqs.push(Requirement::Tech(
                        self.game_data.tech_isv.index_by_key["canRightSideDoorStuck"],
                    ));
                    if *physics != Some(Physics::Air) {
                        reqs.push(Requirement::Or(vec![
                            Requirement::Item(Item::Gravity as ItemId),
                            Requirement::Tech(
                                self.game_data.tech_isv.index_by_key
                                    ["canRightSideDoorStuckFromWater"],
                            ),
                        ]));
                    }
                }
                let mut heat_frames_per_attempt = 0;
                if *heated {
                    heat_frames_per_attempt += 100;
                }
                if entrance_heated {
                    heat_frames_per_attempt += 50;
                }
                if heat_frames_per_attempt > 0 {
                    reqs.push(Requirement::HeatFrames(heat_frames_per_attempt));
                    reqs.push(Requirement::HeatedDoorStuckLeniency {
                        heat_frames: heat_frames_per_attempt,
                    })
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_r_mode_reqs(&self, exit_condition: &ExitCondition) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithGModeSetup { .. } => {
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key["canEnterRMode"],
                ));
                reqs.push(Requirement::Item(Item::XRayScope as ItemId));
                reqs.push(Requirement::ReserveTrigger {
                    min_reserve_energy: 1,
                    max_reserve_energy: 400,
                });
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_g_mode_reqs(
        &self,
        exit_condition: &ExitCondition,
        entrance_link: &Link,
        mode: GModeMode,
        entrance_morphed: bool,
        mobility: GModeMobility,
    ) -> Option<Requirement> {
        let (room_id, node_id, _) = self.game_data.vertex_isv.keys[entrance_link.from_vertex_id];
        let empty_vec = vec![];
        let regain_mobility_vec = self
            .game_data
            .node_gmode_regain_mobility
            .get(&(room_id, node_id))
            .unwrap_or(&empty_vec);
        match exit_condition {
            ExitCondition::LeaveWithGModeSetup { knockback } => {
                if mode == GModeMode::Indirect {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key["canEnterGMode"],
                ));
                if entrance_morphed {
                    reqs.push(Requirement::Or(vec![
                        Requirement::Tech(
                            self.game_data.tech_isv.index_by_key["canArtificialMorph"],
                        ),
                        Requirement::Item(Item::Morph as ItemId),
                    ]));
                }
                reqs.push(Requirement::Item(Item::XRayScope as ItemId));

                let mobile_req = if *knockback {
                    Requirement::ReserveTrigger {
                        min_reserve_energy: 1,
                        max_reserve_energy: 4,
                    }
                } else {
                    Requirement::Never
                };
                let immobile_req = if regain_mobility_vec.len() > 0 {
                    let mut immobile_req_or_vec: Vec<Requirement> = Vec::new();
                    for (regain_mobility_link, _) in regain_mobility_vec {
                        immobile_req_or_vec.push(Requirement::make_and(vec![
                            Requirement::Tech(
                                self.game_data.tech_isv.index_by_key["canEnterGModeImmobile"],
                            ),
                            Requirement::ReserveTrigger {
                                min_reserve_energy: 1,
                                max_reserve_energy: 400,
                            },
                            regain_mobility_link.requirement.clone(),
                        ]));
                    }
                    Requirement::make_or(immobile_req_or_vec)
                } else {
                    Requirement::Never
                };

                match mobility {
                    GModeMobility::Any => {
                        reqs.push(Requirement::make_or(vec![mobile_req, immobile_req]));
                    }
                    GModeMobility::Mobile => {
                        reqs.push(mobile_req);
                    }
                    GModeMobility::Immobile => {
                        reqs.push(immobile_req);
                    }
                }

                Some(Requirement::make_and(reqs))
            }
            ExitCondition::LeaveWithGMode { morphed } => {
                if mode == GModeMode::Direct {
                    return None;
                }
                if !morphed && entrance_morphed {
                    Some(Requirement::Item(Item::Morph as ItemId))
                } else {
                    Some(Requirement::Free)
                }
            }
            _ => None,
        }
    }

    fn get_come_in_with_stored_fall_speed_reqs(
        &self,
        exit_condition: &ExitCondition,
        fall_speed_in_tiles_needed: i32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithStoredFallSpeed {
                fall_speed_in_tiles,
            } => {
                if *fall_speed_in_tiles != fall_speed_in_tiles_needed {
                    return None;
                }
                return Some(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key["canMoonfall"],
                ));
            }
            _ => None,
        }
    }

    fn get_cross_room_reqs(
        &self,
        exit_link: &Link,
        exit_condition: &ExitCondition,
        entrance_link: &Link,
        entrance_condition: &EntranceCondition,
    ) -> Option<Requirement> {
        match entrance_condition {
            EntranceCondition::ComeInRunning {
                speed_booster,
                min_tiles,
                max_tiles,
            } => self.get_come_in_running_reqs(
                exit_link,
                exit_condition,
                *speed_booster,
                *min_tiles,
                *max_tiles,
            ),
            EntranceCondition::ComeInJumping {
                speed_booster,
                min_tiles,
                max_tiles,
            } => self.get_come_in_running_reqs(
                exit_link,
                exit_condition,
                *speed_booster,
                *min_tiles,
                *max_tiles,
            ),
            EntranceCondition::ComeInShinecharging {
                effective_length,
                heated,
            } => self.get_come_in_shinecharging_reqs(
                exit_link,
                exit_condition,
                *effective_length,
                *heated,
            ),
            EntranceCondition::ComeInShinecharged { frames_required } => {
                self.get_come_in_shinecharged_reqs(exit_link, exit_condition, *frames_required)
            }
            EntranceCondition::ComeInWithSpark {} => {
                self.get_come_in_with_spark_reqs(exit_link, exit_condition)
            }
            EntranceCondition::ComeInStutterShinecharging { min_tiles } => {
                self.get_come_in_stutter_shinecharging_reqs(exit_link, exit_condition, *min_tiles)
            }
            EntranceCondition::ComeInWithBombBoost {} => {
                self.get_come_in_with_bomb_boost_reqs(exit_link, exit_condition)
            }
            EntranceCondition::ComeInWithDoorStuckSetup { heated } => {
                self.get_come_in_with_door_stuck_setup_reqs(exit_link, exit_condition, *heated)
            }
            EntranceCondition::ComeInSpeedballing {
                effective_runway_length,
            } => {
                let mut req_or: Vec<Requirement> = vec![];
                if let Some(req) = self.get_come_in_shinecharging_reqs(
                    exit_link,
                    exit_condition,
                    *effective_runway_length - 5.0,
                    false,
                ) {
                    req_or.push(Requirement::make_and(vec![
                        Requirement::Tech(
                            self.game_data.tech_isv.index_by_key["canSlowShortCharge"],
                        ),
                        req,
                    ]));
                }
                if let Some(req) = self.get_come_in_shinecharging_reqs(
                    exit_link,
                    exit_condition,
                    *effective_runway_length - 14.0,
                    false,
                ) {
                    req_or.push(req);
                }
                if req_or.is_empty() {
                    None
                } else {
                    Some(Requirement::make_or(req_or))
                }
            }
            EntranceCondition::ComeInWithRMode {} => {
                self.get_come_in_with_r_mode_reqs(exit_condition)
            }
            EntranceCondition::ComeInWithGMode {
                mode,
                morphed,
                mobility,
            } => self.get_come_in_with_g_mode_reqs(
                exit_condition,
                entrance_link,
                *mode,
                *morphed,
                *mobility,
            ),
            EntranceCondition::ComeInWithStoredFallSpeed {
                fall_speed_in_tiles,
            } => self.get_come_in_with_stored_fall_speed_reqs(exit_condition, *fall_speed_in_tiles),
        }
    }

    fn preprocess_link(&mut self, link: &'a Link) -> Vec<Link> {
        let to_vertex_id = if link.bypasses_door_shell {
            let (room_id, node_id, _) = self.game_data.vertex_isv.keys[link.to_vertex_id];
            let mut unlocked_node_id = node_id;
            if self
                .game_data
                .unlocked_node_map
                .contains_key(&(room_id, node_id))
            {
                unlocked_node_id = self.game_data.unlocked_node_map[&(room_id, node_id)];
            }
            if let Some(&(other_room_id, other_node_id)) =
                self.door_map.get(&(room_id, unlocked_node_id))
            {
                let node_id_spawn = *self
                    .game_data
                    .node_spawn_at_map
                    .get(&(other_room_id, other_node_id))
                    .unwrap_or(&other_node_id);
                // println!("Connecting door bypass strat: ({}, {}) to ({}, {}): {}", room_id, node_id, other_room_id, node_id_spawn, link.strat_name);
                self.game_data.vertex_isv.index_by_key[&(other_room_id, node_id_spawn, 0)]
            } else {
                panic!(
                    "bypassesDoorShell strat with no door: room_id={}, node_id={}, strat: {}",
                    room_id, node_id, link.strat_name
                );
            }
        } else {
            link.to_vertex_id
        };
        if let Some(entrance_condition) = &link.entrance_condition {
            // Process new-style cross room strats:
            let key = ByAddress(link);
            if self.preprocessed_links.contains_key(&key) {
                let val = &self.preprocessed_links[&key];
                val.clone()
            } else {
                // Create an initially empty preprocessed output, to avoid infinite recursion in case of cycles of
                // matching entrance/exit conditions.
                self.preprocessed_links.insert(key, vec![]);

                let mut new_links: Vec<Link> = vec![];
                let (room_id, node_id, _) = self.game_data.vertex_isv.keys[link.from_vertex_id];
                let mut unlocked_node_id = node_id;
                if self
                    .game_data
                    .unlocked_node_map
                    .contains_key(&(room_id, node_id))
                {
                    unlocked_node_id = self.game_data.unlocked_node_map[&(room_id, node_id)];
                }
                if let Some(&(other_room_id, other_node_id)) =
                    self.door_map.get(&(room_id, unlocked_node_id))
                {
                    if let Some(exits) = self
                        .game_data
                        .node_exits
                        .get(&(other_room_id, other_node_id))
                    {
                        let locked_door_idx = self
                            .locked_node_map
                            .get(&(other_room_id, other_node_id))
                            .map(|x| *x);
                        let door_req = get_door_requirement(
                            locked_door_idx,
                            self.locked_doors,
                            self.game_data,
                        );

                        for (raw_exit_link, exit_condition) in exits {
                            let exit_links = self.preprocess_link(raw_exit_link);
                            for exit_link in &exit_links {
                                let cross_req_opt = self.get_cross_room_reqs(
                                    exit_link,
                                    exit_condition,
                                    link,
                                    entrance_condition,
                                );
                                if let Some(cross_req) = cross_req_opt {
                                    if let Requirement::Never = cross_req {
                                        continue;
                                    }
                                    let mut req_and_list = vec![];
                                    req_and_list.push(exit_link.requirement.clone());
                                    req_and_list.push(cross_req);
                                    if !exit_link.bypasses_door_shell {
                                        req_and_list.push(door_req.clone());
                                    }
                                    req_and_list.push(
                                        self.preprocess_requirement(&link.requirement, &link),
                                    );
                                    let req = Requirement::make_and(req_and_list);
                                    // println!("{:?}", door_req);
                                    let mut strat_notes = exit_link.strat_notes.clone();
                                    strat_notes.extend(link.strat_notes.clone());
                                    let mut sublinks = exit_link.sublinks.clone();
                                    if sublinks.is_empty() {
                                        sublinks.push(exit_link.clone());
                                    }
                                    sublinks.push(link.clone());
                                    new_links.push(Link {
                                        from_vertex_id: exit_link.from_vertex_id,
                                        to_vertex_id,
                                        requirement: req,
                                        entrance_condition: None,
                                        bypasses_door_shell: false, // any door shell bypass has already been processed by replacing to_vertex_id with other side of door
                                        notable_strat_name: None, // TODO: Replace with list of notable strats and use them
                                        strat_name: format!(
                                            "{}; {}",
                                            exit_link.strat_name, link.strat_name
                                        ),
                                        strat_notes: strat_notes,
                                        sublinks,
                                    });
                                    // println!("Other room, node: {}, {}: {:?}, {:?}", other_room_id, other_node_id, exit_condition, new_links.last().unwrap());
                                }
                            }
                        }
                    }
                }
                *self.preprocessed_links.get_mut(&key).unwrap() = new_links.clone();
                new_links
            }
        } else {
            // Process old-style cross-room logical requirements:
            vec![Link {
                from_vertex_id: link.from_vertex_id,
                to_vertex_id,
                requirement: self.preprocess_requirement(&link.requirement, link),
                entrance_condition: None,
                bypasses_door_shell: false, // any door shell bypass has already been processed by replacing to_vertex_id with other side of door
                notable_strat_name: link.notable_strat_name.clone(),
                strat_name: link.strat_name.clone(),
                strat_notes: link.strat_notes.clone(),
                sublinks: vec![],
            }]
        }
    }

    fn preprocess_requirement(&mut self, req: &'a Requirement, link: &Link) -> Requirement {
        let key = ByAddress(req);
        if self.preprocessed_output.contains_key(&key) {
            if let Some(val) = &self.preprocessed_output[&key] {
                return val.clone();
            } else {
                // Circular dependency detected, which cannot be satisfied.
                // println!("Circular requirement: {:?}", req);
                return Requirement::Never;
            }
        }
        self.preprocessed_output.insert(key, None);

        let out = match req {
            Requirement::AdjacentRunway {
                room_id,
                node_id,
                used_tiles,
                use_frames,
                physics,
                override_runway_requirements,
            } => self.preprocess_adjacent_runway(
                *room_id,
                *node_id,
                *used_tiles,
                *use_frames,
                physics,
                *override_runway_requirements,
                link,
            ),
            Requirement::AdjacentJumpway {
                room_id,
                node_id,
                jumpway_type,
                min_height,
                max_height,
                max_left_position,
                min_right_position,
            } => self.preprocess_adjacent_jumpway(
                *room_id,
                *node_id,
                jumpway_type,
                *min_height,
                *max_height,
                *max_left_position,
                *min_right_position,
                link,
            ),
            Requirement::CanComeInCharged {
                room_id,
                node_id,
                frames_remaining,
                unusable_tiles,
            } => self.preprocess_can_come_in_charged(
                *room_id,
                *node_id,
                *frames_remaining,
                *unusable_tiles,
                link,
            ),
            Requirement::ComeInWithRMode { room_id, node_ids } => {
                self.preprocess_come_in_with_rmode(*room_id, node_ids, link)
            }
            Requirement::ComeInWithGMode {
                room_id,
                node_ids,
                mode,
                artificial_morph,
                mobility,
            } => self.preprocess_come_in_with_gmode(
                *room_id,
                node_ids,
                mode,
                *artificial_morph,
                mobility,
                link,
            ),
            Requirement::DoorUnlocked { room_id, node_id } => {
                self.preprocess_door_unlocked(*room_id, *node_id)
            }
            Requirement::And(sub_reqs) => Requirement::make_and(
                sub_reqs
                    .iter()
                    .map(|r| self.preprocess_requirement(r, link))
                    .collect(),
            ),
            Requirement::Or(sub_reqs) => Requirement::make_or(
                sub_reqs
                    .iter()
                    .map(|r| self.preprocess_requirement(r, link))
                    .collect(),
            ),
            _ => req.clone(),
        };
        self.preprocessed_output.insert(key, Some(out.clone()));
        out
    }

    fn preprocess_can_come_in_charged(
        &mut self,
        room_id: RoomId,
        node_id: NodeId,
        frames_remaining: i32,
        unusable_tiles: i32,
        _link: &Link,
    ) -> Requirement {
        let mut unlocked_node_id = node_id;
        if self
            .game_data
            .unlocked_node_map
            .contains_key(&(room_id, node_id))
        {
            unlocked_node_id = self.game_data.unlocked_node_map[&(room_id, node_id)];
        }
        if let Some(&(other_room_id, other_node_id)) =
            self.door_map.get(&(room_id, unlocked_node_id))
        {
            let runways = &self.game_data.node_runways_map[&(room_id, node_id)];
            let other_runways = &self.game_data.node_runways_map[&(other_room_id, other_node_id)];
            let can_leave_charged_vec =
                &self.game_data.node_can_leave_charged_map[&(other_room_id, other_node_id)];
            let locked_door_idx = self
                .locked_node_map
                .get(&(room_id, unlocked_node_id))
                .map(|x| *x);
            let door_req = get_door_requirement(locked_door_idx, self.locked_doors, self.game_data);
            let mut req_vec: Vec<Requirement> = vec![];

            // let from_triple = self.game_data.vertex_isv.keys[_link.from_vertex_id];
            // let to_triple = self.game_data.vertex_isv.keys[_link.to_vertex_id];
            // println!(
            //     "Link: from={:?}, to={:?}, strat={}",
            //     from_triple, to_triple, _link.strat_name
            // );
            // println!("frames_remaining={frames_remaining}, shinespark_frames={shinespark_frames}");
            // println!("In-room runways:");
            // for runway in runways {
            //     println!("{:?}", runway);
            // }
            // println!("Other-room runways:");
            // for runway in other_runways {
            //     println!("{:?}", runway);
            // }
            // println!("canLeaveCharged:");
            // for can_leave_charged in can_leave_charged_vec {
            //     println!("{:?}", can_leave_charged);
            // }

            // Strats for in-room runways:
            for runway in runways {
                let effective_length =
                    get_effective_runway_length(runway.length as f32, runway.open_end as f32)
                        - unusable_tiles as f32;
                let req = Requirement::make_shinecharge(effective_length);
                req_vec.push(Requirement::make_and(vec![
                    req,
                    self.preprocess_requirement(&runway.requirement, _link),
                ]));
            }

            // Strats for other-room runways:
            for runway in other_runways {
                let effective_length =
                    get_effective_runway_length(runway.length as f32, runway.open_end as f32)
                        - unusable_tiles as f32;
                let req = Requirement::make_shinecharge(effective_length);
                req_vec.push(Requirement::make_and(vec![
                    door_req.clone(),
                    req,
                    self.preprocess_requirement(&runway.requirement, _link),
                ]));
            }

            // Strats for cross-room combined runways:
            for runway in runways {
                if !runway.usable_coming_in {
                    continue;
                }
                for other_runway in other_runways {
                    let in_room_effective_length =
                        get_effective_runway_length(runway.length as f32, runway.open_end as f32);
                    let other_room_effective_length = get_effective_runway_length(
                        other_runway.length as f32,
                        other_runway.open_end as f32,
                    );
                    let total_effective_length = in_room_effective_length
                        + other_room_effective_length
                        - 1.0
                        - unusable_tiles as f32;
                    let req = Requirement::make_shinecharge(total_effective_length);
                    req_vec.push(Requirement::make_and(vec![
                        door_req.clone(),
                        req,
                        self.preprocess_requirement(&runway.requirement, _link),
                        self.preprocess_requirement(&other_runway.requirement, _link),
                    ]));
                    // println!("{} - {}: {:?}", runway.name, other_runway.name, req_vec.last().unwrap());
                }
            }

            // Strats for canLeaveCharged from other room:
            for can_leave_charged in can_leave_charged_vec {
                if can_leave_charged.frames_remaining < frames_remaining {
                    continue;
                }
                let effective_length = get_effective_runway_length(
                    can_leave_charged.used_tiles as f32,
                    can_leave_charged.open_end as f32,
                );
                let req = Requirement::ShineCharge {
                    used_tiles: effective_length,
                };
                req_vec.push(Requirement::make_and(vec![
                    door_req.clone(),
                    req,
                    self.preprocess_requirement(&can_leave_charged.requirement, _link),
                ]));
            }

            // println!("Strats: {:?} {:?}\n", _link.strat_notes, req_vec);
            let out = Requirement::make_or(req_vec);
            out
        } else {
            // println!(
            //     "In canComeInCharged, ({}, {}) is not door node?",
            //     room_id, node_id
            // );
            Requirement::Never
        }
    }

    fn preprocess_adjacent_runway(
        &mut self,
        room_id: RoomId,
        node_id: NodeId,
        used_tiles: f32,
        use_frames: Option<i32>,
        physics: &Option<String>,
        override_runway_requirements: bool,
        _link: &Link,
    ) -> Requirement {
        // println!("{} {} {}", room_id, node_id, _link.strat_name);
        let mut unlocked_node_id = node_id;
        if self
            .game_data
            .unlocked_node_map
            .contains_key(&(room_id, node_id))
        {
            unlocked_node_id = self.game_data.unlocked_node_map[&(room_id, node_id)];
        }
        // if !self.door_map.contains_key(&(room_id, unlocked_node_id)) {
        //     info!("Ignoring adjacent runway with unrecognized node: room_id={}, node_id={}", room_id, node_id);
        // }

        let (other_room_id, other_node_id) = *self
            .door_map
            .get(&(room_id, unlocked_node_id))
            .with_context(|| {
                format!(
                    "No door_map entry for ({}, {}): {:?}",
                    room_id, unlocked_node_id, _link
                )
            })
            .unwrap();
        let runways = &self.game_data.node_runways_map[&(other_room_id, other_node_id)];
        let locked_door_idx = self
            .locked_node_map
            .get(&(room_id, unlocked_node_id))
            .map(|x| *x);
        let door_req = get_door_requirement(locked_door_idx, self.locked_doors, self.game_data);
        let mut req_vec: Vec<Requirement> = vec![];
        for runway in runways {
            let effective_length =
                get_effective_runway_length(runway.length as f32, runway.open_end as f32);
            // println!(
            //     "  {}: length={}, open_end={}, physics={}, heated={}, req={:?}",
            //     runway.name, runway.length, runway.open_end, runway.physics, runway.heated, runway.requirement
            // );
            if effective_length < used_tiles {
                continue;
            }
            let mut reqs: Vec<Requirement> = vec![door_req.clone()];
            if let Some(physics_str) = physics.as_ref() {
                if physics_str == "normal" {
                    if runway.physics == "water" {
                        reqs.push(Requirement::Item(Item::Gravity as usize));
                    } else if runway.physics != "air" {
                        continue;
                    }
                } else if &runway.physics != physics_str {
                    continue;
                }
            }
            if override_runway_requirements {
                if runway.heated {
                    if let Some(frames) = use_frames {
                        reqs.push(Requirement::HeatFrames(frames));
                    } else {
                        // TODO: Use a more accurate estimate (and take into account if we have SpeedBooster):
                        let frames = used_tiles * 10.0 + 20.0;
                        reqs.push(Requirement::HeatFrames(frames as i32));
                    }
                }
            } else {
                reqs.push(self.preprocess_requirement(&runway.requirement, _link));
            }
            req_vec.push(Requirement::make_and(reqs));
        }
        let out = Requirement::make_or(req_vec);
        // println!(
        //     "{}: used_tiles={}, use_frames={:?}, physics={:?}, {:?}",
        //     _link.strat_name, used_tiles, use_frames, physics, out
        // );
        out
    }

    fn preprocess_adjacent_jumpway(
        &mut self,
        room_id: RoomId,
        node_id: NodeId,
        jumpway_type: &str,
        min_height: Option<f32>,
        max_height: Option<f32>,
        max_left_position: Option<f32>,
        min_right_position: Option<f32>,
        _link: &Link,
    ) -> Requirement {
        // println!("{} {} {}", room_id, node_id, _link.strat_name);
        let mut unlocked_node_id = node_id;
        if self
            .game_data
            .unlocked_node_map
            .contains_key(&(room_id, node_id))
        {
            unlocked_node_id = self.game_data.unlocked_node_map[&(room_id, node_id)];
        }
        let (other_room_id, other_node_id) = self.door_map[&(room_id, unlocked_node_id)];
        // Commenting this out for now: we can't safely skip over the Toilet, because it centers Samus horizontally
        // which could interfere with the strat (particularly jumps up into Pseudo Plasma Spark Room):
        // if (other_room_id, other_node_id) == (321, 1) {
        //     // Check jumpways below Toilet
        //     (other_room_id, other_node_id) = self.door_map[&(321, 2)];
        // }
        let jumpways = &self.game_data.node_jumpways_map[&(other_room_id, other_node_id)];
        let locked_door_idx = self
            .locked_node_map
            .get(&(room_id, unlocked_node_id))
            .map(|x| *x);
        let door_req = get_door_requirement(locked_door_idx, self.locked_doors, self.game_data);
        let mut req_vec: Vec<Requirement> = vec![];
        for jumpway in jumpways {
            if jumpway.jumpway_type != jumpway_type {
                continue;
            }
            if let Some(x) = min_height {
                if jumpway.height < x {
                    continue;
                }
            }
            if let Some(x) = max_height {
                if jumpway.height > x {
                    continue;
                }
            }
            if let Some(x) = max_left_position {
                if jumpway.left_position.unwrap() > x {
                    continue;
                }
            }
            if let Some(x) = min_right_position {
                if jumpway.right_position.unwrap() < x {
                    continue;
                }
            }
            // println!("{}", jumpway.name);
            let jumpway_req = self.preprocess_requirement(&jumpway.requirement, _link);
            req_vec.push(Requirement::make_and(vec![door_req.clone(), jumpway_req]));
        }
        let out = Requirement::make_or(req_vec);
        // println!(
        //     "{}, {}, {}: {:?}",
        //     self.game_data.room_json_map[&room_id]["name"],
        //     self.game_data.node_json_map[&(room_id, node_id)]["name"],
        //     _link.strat_name, out
        // );
        out
    }

    fn preprocess_come_in_with_rmode(
        &mut self,
        room_id: RoomId,
        node_ids: &[NodeId],
        link: &Link,
    ) -> Requirement {
        let rmode_tech_id = self.game_data.tech_isv.index_by_key["canEnterRMode"];
        let xray_item_id = self.game_data.item_isv.index_by_key["XRayScope"];
        let mut req_or_list: Vec<Requirement> = Vec::new();
        for &node_id in node_ids {
            let mut unlocked_node_id = node_id;
            if self
                .game_data
                .unlocked_node_map
                .contains_key(&(room_id, node_id))
            {
                unlocked_node_id = self.game_data.unlocked_node_map[&(room_id, node_id)];
            }
            if let Some(&(other_room_id, other_node_id)) =
                self.door_map.get(&(room_id, unlocked_node_id))
            {
                let locked_door_idx = self
                    .locked_node_map
                    .get(&(room_id, unlocked_node_id))
                    .map(|x| *x);
                let door_req =
                    get_door_requirement(locked_door_idx, self.locked_doors, self.game_data);
                let leave_with_gmode_setup_vec = &self.game_data.node_leave_with_gmode_setup_map
                    [&(other_room_id, other_node_id)];
                for leave_with_gmode_setup in leave_with_gmode_setup_vec {
                    let mut req_and_list: Vec<Requirement> = Vec::new();
                    req_and_list.push(door_req.clone());
                    req_and_list.push(
                        self.preprocess_requirement(&leave_with_gmode_setup.requirement, link),
                    );
                    req_and_list.push(Requirement::Tech(rmode_tech_id));
                    req_and_list.push(Requirement::Item(xray_item_id));
                    req_and_list.push(Requirement::ReserveTrigger {
                        min_reserve_energy: 1,
                        max_reserve_energy: 400,
                    });
                    req_or_list.push(Requirement::make_and(req_and_list));
                }
            }
        }

        let out = Requirement::make_or(req_or_list);
        // println!(
        //     "{} ({}) {:?} {}: {:?}",
        //     self.game_data.room_json_map[&room_id]["name"], room_id, node_ids, link.strat_name, out
        // );
        out
    }

    fn preprocess_come_in_with_gmode(
        &mut self,
        room_id: RoomId,
        node_ids: &[NodeId],
        mode: &str,
        artificial_morph: bool,
        mobility: &str,
        link: &Link,
    ) -> Requirement {
        let gmode_tech_id = self.game_data.tech_isv.index_by_key["canEnterGMode"];
        let gmode_immobile_tech_id = self.game_data.tech_isv.index_by_key["canEnterGModeImmobile"];
        let artificial_morph_tech_id = self.game_data.tech_isv.index_by_key["canArtificialMorph"];
        let morph_item_id = self.game_data.item_isv.index_by_key["Morph"];
        let xray_item_id = self.game_data.item_isv.index_by_key["XRayScope"];
        let mut req_or_list: Vec<Requirement> = Vec::new();
        for &node_id in node_ids {
            let mut unlocked_node_id = node_id;
            if self
                .game_data
                .unlocked_node_map
                .contains_key(&(room_id, node_id))
            {
                unlocked_node_id = self.game_data.unlocked_node_map[&(room_id, node_id)];
            }
            if let Some(&(other_room_id, other_node_id)) =
                self.door_map.get(&(room_id, unlocked_node_id))
            {
                let locked_door_idx = self
                    .locked_node_map
                    .get(&(room_id, unlocked_node_id))
                    .map(|x| *x);
                let door_req =
                    get_door_requirement(locked_door_idx, self.locked_doors, self.game_data);
                let gmode_immobile_opt = self
                    .game_data
                    .node_gmode_immobile_map
                    .get(&(room_id, node_id));
                if mode == "direct" || mode == "any" {
                    let leave_with_gmode_setup_vec = &self
                        .game_data
                        .node_leave_with_gmode_setup_map[&(other_room_id, other_node_id)];
                    for leave_with_gmode_setup in leave_with_gmode_setup_vec {
                        let mut req_and_list: Vec<Requirement> = Vec::new();
                        req_and_list.push(door_req.clone());
                        req_and_list.push(
                            self.preprocess_requirement(&leave_with_gmode_setup.requirement, link),
                        );
                        req_and_list.push(Requirement::Tech(gmode_tech_id));
                        if artificial_morph {
                            req_and_list.push(Requirement::Or(vec![
                                Requirement::Tech(artificial_morph_tech_id),
                                Requirement::Item(morph_item_id),
                            ]));
                        }
                        req_and_list.push(Requirement::Item(xray_item_id));

                        let mobile_req = if leave_with_gmode_setup.knockback {
                            Requirement::ReserveTrigger {
                                min_reserve_energy: 1,
                                max_reserve_energy: 4,
                            }
                        } else {
                            Requirement::Never
                        };
                        let immobile_req = if let Some(gmode_immobile) = gmode_immobile_opt {
                            let mut immobile_req_vec: Vec<Requirement> = Vec::new();
                            immobile_req_vec.push(Requirement::Tech(gmode_immobile_tech_id));
                            immobile_req_vec.push(Requirement::ReserveTrigger {
                                min_reserve_energy: 1,
                                max_reserve_energy: 400,
                            });
                            immobile_req_vec.push(gmode_immobile.requirement.clone());
                            Requirement::make_and(immobile_req_vec)
                        } else {
                            Requirement::Never
                        };

                        if mobility == "any" {
                            req_and_list.push(Requirement::make_or(vec![mobile_req, immobile_req]));
                        } else if mobility == "mobile" {
                            req_and_list.push(mobile_req);
                        } else if mobility == "immobile" {
                            req_and_list.push(immobile_req);
                        } else {
                            panic!("Invalid mobility {}", mobility);
                        }

                        req_or_list.push(Requirement::make_and(req_and_list));
                    }
                }

                if mode == "indirect" || mode == "any" {
                    let leave_with_gmode_vec =
                        &self.game_data.node_leave_with_gmode_map[&(other_room_id, other_node_id)];
                    for leave_with_gmode in leave_with_gmode_vec {
                        // TODO: fix to handle case where Morph item is collected
                        if !artificial_morph || leave_with_gmode.artificial_morph {
                            let mut req_and_list: Vec<Requirement> = Vec::new();
                            req_and_list.push(door_req.clone());
                            req_and_list.push(
                                self.preprocess_requirement(&leave_with_gmode.requirement, link),
                            );
                            req_or_list.push(Requirement::make_and(req_and_list));
                        }
                    }
                }
            }
        }

        let out = Requirement::make_or(req_or_list);
        // println!(
        //     "{} ({}) {:?} {}: {:?}",
        //     self.game_data.room_json_map[&room_id]["name"], room_id, node_ids, link.strat_name, out
        // );
        out
    }

    fn preprocess_door_unlocked(&mut self, room_id: RoomId, node_id: NodeId) -> Requirement {
        let mut unlocked_node_id = node_id;
        if self
            .game_data
            .unlocked_node_map
            .contains_key(&(room_id, node_id))
        {
            unlocked_node_id = self.game_data.unlocked_node_map[&(room_id, node_id)];
        }
        let locked_door_idx = self
            .locked_node_map
            .get(&(room_id, unlocked_node_id))
            .map(|x| *x);
        let door_req = get_door_requirement(locked_door_idx, self.locked_doors, self.game_data);
        return door_req;
    }
}

fn get_randomizable_doors(
    game_data: &GameData,
    difficulty: &DifficultyConfig,
) -> HashSet<DoorPtrPair> {
    // Doors which we do not want to randomize:
    let mut non_randomizable_doors: HashSet<DoorPtrPair> = vec![
        // Gray doors - Pirate rooms:
        (0x18B7A, 0x18B62), // Pit Room left
        (0x18B86, 0x18B92), // Pit Room right
        (0x19192, 0x1917A), // Baby Kraid left
        (0x1919E, 0x191AA), // Baby Kraid right
        (0x1A558, 0x1A54C), // Plasma Room
        (0x19A32, 0x19966), // Metal Pirates left
        (0x19A3E, 0x19A1A), // Metal Pirates right
        // Gray doors - Bosses:
        (0x191CE, 0x191B6), // Kraid left
        (0x191DA, 0x19252), // Kraid right
        (0x1A2C4, 0x1A2AC), // Phantoon
        (0x1A978, 0x1A924), // Draygon left
        (0x1A96C, 0x1A840), // Draygon right
        (0x198B2, 0x19A62), // Ridley left
        (0x198BE, 0x198CA), // Ridley right
        (0x1AA8C, 0x1AAE0), // Mother Brain left
        (0x1AA80, 0x1AAC8), // Mother Brain right
        // Gray doors - Minibosses:
        (0x18BAA, 0x18BC2), // Bomb Torizo
        (0x18E56, 0x18E3E), // Spore Spawn bottom
        (0x193EA, 0x193D2), // Crocomire top
        (0x1A90C, 0x1A774), // Botwoon left
        (0x19882, 0x19A86), // Golden Torizo right
        // Save stations:
        (0x189BE, 0x1899A), // Crateria Save Room
        (0x19006, 0x18D12), // Green Brinstar Main Shaft Save Room
        (0x19012, 0x18F52), // Etecoon Save Room
        (0x18FD6, 0x18DF6), // Big Pink Save Room
        (0x1926A, 0x190D2), // Caterpillar Save Room
        (0x1925E, 0x19186), // Warehouse Save Room
        (0x1A828, 0x1A744), // Aqueduct Save Room
        (0x1A888, 0x1A7EC), // Draygon Save Room left
        (0x1A87C, 0x1A930), // Draygon Save Room right
        (0x1A5F4, 0x1A588), // Forgotten Highway Save Room
        (0x1A324, 0x1A354), // Glass Tunnel Save Room
        (0x19822, 0x193BA), // Crocomire Save Room
        (0x19462, 0x19456), // Post Crocomire Save Room
        (0x1982E, 0x19702), // Lower Norfair Elevator Save Room
        (0x19816, 0x192FA), // Frog Savestation left
        (0x1980A, 0x197DA), // Frog Savestation right
        (0x197CE, 0x1959A), // Bubble Mountain Save Room
        (0x19AB6, 0x19A0E), // Red Kihunter Shaft Save Room
        (0x1A318, 0x1A240), // Wrecked Ship Save Room
        (0x1AAD4, 0x1AABC), // Lower Tourian Save Room
        // Map stations:
        (0x18C2E, 0x18BDA), // Crateria Map Room
        (0x18D72, 0x18D36), // Brinstar Map Room
        (0x197C2, 0x19306), // Norfair Map Room
        (0x1A5E8, 0x1A51C), // Maridia Map Room
        (0x1A2B8, 0x1A2A0), // Wrecked Ship Map Room
        (0x1AB40, 0x1A99C), // Tourian Map Room (Upper Tourian Save Room)
        // Refill stations:
        (0x18D96, 0x18D7E), // Green Brinstar Missile Refill Room
        (0x18F6A, 0x18DBA), // Dachora Energy Refill Room
        (0x191FE, 0x1904E), // Sloaters Refill
        (0x1A894, 0x1A8F4), // Maridia Missile Refill Room
        (0x1A930, 0x1A87C), // Maridia Health Refill Room
        (0x19786, 0x19756), // Nutella Refill left
        (0x19792, 0x1976E), // Nutella Refill right
        (0x1920A, 0x191C2), // Kraid Recharge Station
        (0x198A6, 0x19A7A), // Golden Torizo Energy Recharge
        (0x1AA74, 0x1AA68), // Tourian Recharge Room
        // Pants room interior door
        (0x1A7A4, 0x1A78C), // Left door
        (0x1A78C, 0x1A7A4), // Right door
        // Bad doors that logic would not be able to properly account for (yet):
        (0x19996, 0x1997E), // Amphitheatre left door
        (0x1AA14, 0x1AA20), // Tourian Blue Hopper Room left door
        // Items: (to avoid an interaction in map tiles between doors disappearing and items disappearing)
        (0x18FA6, 0x18EDA), // First Missile Room
        (0x18FFA, 0x18FEE), // Billy Mays Room
        (0x18D66, 0x18D5A), // Brinstar Reserve Tank Room
        (0x18F3A, 0x18F5E), // Etecoon Energy Tank Room (top left door)
        (0x18F5E, 0x18F3A), // Etecoon Supers Room
        (0x18E02, 0x18E62), // Big Pink (top door to Pink Brinstar Power Bomb Room)
        (0x18FCA, 0x18FBE), // Hopper Energy Tank Room
        (0x19132, 0x19126), // Spazer Room
        (0x19162, 0x1914A), // Warehouse Energy Tank Room
        (0x19252, 0x191DA), // Varia Suit Room
        (0x18ADE, 0x18A36), // The Moat (left door)
        (0x18C9A, 0x18C82), // The Final Missile
        (0x18BE6, 0x18C3A), // Terminator Room (left door)
        (0x18B0E, 0x18952), // Gauntlet Energy Tank Room (right door)
        (0x1A924, 0x1A978), // Space Jump Room
        (0x19A62, 0x198B2), // Ridley Tank Room
        (0x199D2, 0x19A9E), // Lower Norfair Escape Power Bomb Room (left door)
        (0x199DE, 0x199C6), // Lower Norfair Escape Power Bomb Room (top door)
        (0x19876, 0x1983A), // Golden Torizo's Room (left door)
        (0x19A86, 0x19882), // Screw Attack Room (left door)
        (0x1941A, 0x192D6), // Hi Jump Energy Tank Room (right door)
        (0x193F6, 0x19426), // Hi Jump Boots Room
        (0x1929A, 0x19732), // Cathedral (right door)
        (0x1953A, 0x19552), // Green Bubbles Missile Room
        (0x195B2, 0x195BE), // Speed Booster Hall
        (0x195BE, 0x195B2), // Speed Booster Room
        (0x1962A, 0x1961E), // Wave Beam Room
        (0x1935A, 0x1937E), // Ice Beam Room
        (0x1938A, 0x19336), // Crumble Shaft (top right door)
        (0x19402, 0x192E2), // Crocomire Escape (left door)
        (0x1946E, 0x1943E), // Post Crocomire Power Bomb Room
        (0x19516, 0x194DA), // Grapple Beam Room (bottom right door)
        (0x1A2E8, 0x1A210), // Wrecked Ship West Super Room
        (0x1A300, 0x18A06), // Gravity Suit Room (left door)
        (0x1A30C, 0x1A1A4), // Gravity Suit Room (right door)
    ]
    .into_iter()
    .map(|(x, y)| (Some(x), Some(y)))
    .collect();

    // Avoid placing an ammo door on a tile with an objective "X", as it looks bad.
    match difficulty.objectives {
        Objectives::Bosses => {
            // The boss doors are all gray and were already excluded above.
        }
        Objectives::Minibosses => {
            // Spore Spawn Room right door:
            non_randomizable_doors.insert((Some(0x18E4A), Some(0x18D2A)));
            // Crocomire left door:
            non_randomizable_doors.insert((Some(0x193DE), Some(0x19432)));
            // Botwoon right door:
            non_randomizable_doors.insert((Some(0x1A918), Some(0x1A84C)));
            // Golden Torizo left door:
            non_randomizable_doors.insert((Some(0x19876), Some(0x1983A)));
        }
        Objectives::Metroids => {
            // Metroid Room 1 left door:
            non_randomizable_doors.insert((Some(0x1A9B4), Some(0x1A9C0)));
            // Metroid Room 1 right door:
            non_randomizable_doors.insert((Some(0x1A9A8), Some(0x1A984)));
            // Metroid Room 2 top right door:
            non_randomizable_doors.insert((Some(0x1A9C0), Some(0x1A9B4)));
            // Metroid Room 2 bottom right door:
            non_randomizable_doors.insert((Some(0x1A9CC), Some(0x1A9D8)));
            // Metroid Room 3 left door:
            non_randomizable_doors.insert((Some(0x1A9D8), Some(0x1A9CC)));
            // Metroid Room 3 right door:
            non_randomizable_doors.insert((Some(0x1A9E4), Some(0x1A9F0)));
            // Metroid Room 4 left door:
            non_randomizable_doors.insert((Some(0x1A9F0), Some(0x1A9E4)));
            // Metroid Room 4 bottom door:
            non_randomizable_doors.insert((Some(0x1A9FC), Some(0x1AA08)));
        }
        Objectives::Chozos => {
            // All the door tiles with X's have a gray door, so are covered above.
        }
        Objectives::Pirates => {
            // These doors are all gray, so are covered above.
        }
    }

    let mut out: Vec<DoorPtrPair> = vec![];
    for room in &game_data.room_geometry {
        for door in &room.doors {
            let pair = (door.exit_ptr, door.entrance_ptr);
            let has_door_cap = door.offset.is_some();
            if has_door_cap && !non_randomizable_doors.contains(&pair) {
                out.push(pair);
            }
        }
    }
    out.into_iter().collect()
}

fn get_randomizable_door_connections(
    game_data: &GameData,
    map: &Map,
    difficulty: &DifficultyConfig,
) -> Vec<(DoorPtrPair, DoorPtrPair)> {
    let doors = get_randomizable_doors(game_data, difficulty);
    let mut out: Vec<(DoorPtrPair, DoorPtrPair)> = vec![];
    for (src_door_ptr_pair, dst_door_ptr_pair, _bidirectional) in &map.doors {
        if doors.contains(src_door_ptr_pair) && doors.contains(dst_door_ptr_pair) {
            out.push((*src_door_ptr_pair, *dst_door_ptr_pair));
        }
    }
    out
}

pub fn randomize_doors(
    game_data: &GameData,
    map: &Map,
    difficulty: &DifficultyConfig,
    seed: usize,
) -> Vec<LockedDoor> {
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let get_loc = |ptr_pair: DoorPtrPair| -> (RoomGeometryRoomIdx, usize, usize) {
        let (room_idx, door_idx) = game_data.room_and_door_idxs_by_door_ptr_pair[&ptr_pair];
        let room = &game_data.room_geometry[room_idx];
        let door = &room.doors[door_idx];
        (room_idx, door.x, door.y)
    };
    let mut used_locs: HashSet<(RoomGeometryRoomIdx, usize, usize)> = HashSet::new();

    match difficulty.doors_mode {
        DoorsMode::Blue => {
            vec![]
        }
        DoorsMode::Ammo => {
            let red_doors_cnt = 30;
            let green_doors_cnt = 15;
            let yellow_doors_cnt = 10;
            let total_cnt = red_doors_cnt + green_doors_cnt + yellow_doors_cnt;
            let mut door_types = vec![];
            door_types.extend(vec![DoorType::Red; red_doors_cnt]);
            door_types.extend(vec![DoorType::Green; green_doors_cnt]);
            door_types.extend(vec![DoorType::Yellow; yellow_doors_cnt]);

            let door_conns = get_randomizable_door_connections(game_data, map, difficulty);
            let mut out: Vec<LockedDoor> = vec![];
            let idxs = rand::seq::index::sample(&mut rng, door_conns.len(), total_cnt);
            for (i, idx) in idxs.into_iter().enumerate() {
                let conn = &door_conns[idx];
                let door = LockedDoor {
                    src_ptr_pair: conn.0,
                    dst_ptr_pair: conn.1,
                    door_type: door_types[i],
                    bidirectional: true,
                };

                // Make sure we don't put two ammo doors in the same tile (since that would interfere
                // with the mechanism for making the doors disappear from the map).
                let src_loc = get_loc(door.src_ptr_pair);
                let dst_loc = get_loc(door.dst_ptr_pair);
                if !used_locs.contains(&src_loc) && !used_locs.contains(&dst_loc) {
                    used_locs.insert(src_loc);
                    used_locs.insert(dst_loc);
                    out.push(door);
                }
            }
            out
        }
    }
}

fn is_req_possible(req: &Requirement, tech_active: &[bool], strats_active: &[bool]) -> bool {
    match req {
        Requirement::Tech(tech_id) => tech_active[*tech_id],
        Requirement::Strat(strat_id) => strats_active[*strat_id],
        Requirement::And(reqs) => reqs
            .iter()
            .all(|x| is_req_possible(x, tech_active, strats_active)),
        Requirement::Or(reqs) => reqs
            .iter()
            .any(|x| is_req_possible(x, tech_active, strats_active)),
        _ => true,
    }
}

pub fn filter_links(
    links: &[Link],
    game_data: &GameData,
    difficulty: &DifficultyConfig,
) -> Vec<Link> {
    let mut out = vec![];
    let tech_vec = get_tech_vec(game_data, difficulty);
    let strat_vec = get_strat_vec(game_data, difficulty);
    for link in links {
        if is_req_possible(&link.requirement, &tech_vec, &strat_vec) {
            out.push(link.clone())
        }
    }
    out
}

fn get_tech_vec(game_data: &GameData, difficulty: &DifficultyConfig) -> Vec<bool> {
    let tech_set: HashSet<String> = difficulty.tech.iter().map(|x| x.clone()).collect();
    game_data
        .tech_isv
        .keys
        .iter()
        .map(|x| tech_set.contains(x))
        .collect()
}

fn get_strat_vec(game_data: &GameData, difficulty: &DifficultyConfig) -> Vec<bool> {
    let strat_set: HashSet<String> = difficulty
        .notable_strats
        .iter()
        .map(|x| x.clone())
        .collect();
    game_data
        .notable_strat_isv
        .keys
        .iter()
        .map(|x| strat_set.contains(x))
        .collect()
}

impl<'r> Randomizer<'r> {
    pub fn new(
        map: &'r Map,
        locked_doors: &'r [LockedDoor],
        difficulty_tiers: &'r [DifficultyConfig],
        game_data: &'r GameData,
        base_links_data: &'r LinksDataGroup,
        seed_links: &'r [Link],
    ) -> Randomizer<'r> {
        let mut locked_door_map: HashMap<DoorPtrPair, usize> = HashMap::new();
        for (i, door) in locked_doors.iter().enumerate() {
            locked_door_map.insert(door.src_ptr_pair, i);
            if door.bidirectional {
                locked_door_map.insert(door.dst_ptr_pair, i);
            }
        }

        let mut preprocessor = Preprocessor::new(game_data, map, locked_doors, &locked_door_map);
        let mut preprocessed_seed_links: Vec<Link> = seed_links
            .iter()
            .map(|x| preprocessor.preprocess_link(x))
            .flatten()
            .collect();
        for door in &map.doors {
            let src_exit_ptr = door.0 .0;
            let src_entrance_ptr = door.0 .1;
            let dst_exit_ptr = door.1 .0;
            let dst_entrance_ptr = door.1 .1;
            let bidirectional = door.2;
            let (src_room_id, src_node_id) =
                game_data.door_ptr_pair_map[&(src_exit_ptr, src_entrance_ptr)];
            let (_, unlocked_src_node_id) =
                game_data.unlocked_door_ptr_pair_map[&(src_exit_ptr, src_entrance_ptr)];
            let src_locked_door_idx = locked_door_map
                .get(&(src_exit_ptr, src_entrance_ptr))
                .map(|x| *x);
            let (dst_room_id, dst_node_id) =
                game_data.door_ptr_pair_map[&(dst_exit_ptr, dst_entrance_ptr)];
            let (_, unlocked_dst_node_id) =
                game_data.unlocked_door_ptr_pair_map[&(dst_exit_ptr, dst_entrance_ptr)];
            let dst_locked_door_idx = locked_door_map
                .get(&(dst_exit_ptr, dst_entrance_ptr))
                .map(|x| *x);

            add_door_links(
                src_room_id,
                unlocked_src_node_id,
                dst_room_id,
                dst_node_id,
                src_locked_door_idx,
                game_data,
                &mut preprocessed_seed_links,
                locked_doors,
            );
            // if (src_room_id, unlocked_src_node_id) == (220, 2) {
            //     println!("pants");
            // }
            // if src_room_id == 220 || dst_room_id == 220 || src_room_id == 322 || dst_room_id == 322 {
            //     println!("({:x}, {:x}) ({:x}, {:x}) ({}, {})  ({}, {}) {}",
            //     src_exit_ptr.unwrap(), src_entrance_ptr.unwrap(),
            //     dst_exit_ptr.unwrap(), dst_entrance_ptr.unwrap(),
            //     src_room_id, src_node_id, dst_room_id, dst_node_id, bidirectional);
            // }
            if (src_room_id, unlocked_src_node_id) == (322, 2) {
                // For East Pants Room right door, add a corresponding (one-way) link for Pants Room too:
                add_door_links(
                    220,
                    2,
                    dst_room_id,
                    dst_node_id,
                    src_locked_door_idx,
                    game_data,
                    &mut preprocessed_seed_links,
                    locked_doors,
                );
            }
            if bidirectional {
                add_door_links(
                    dst_room_id,
                    unlocked_dst_node_id,
                    src_room_id,
                    src_node_id,
                    dst_locked_door_idx,
                    game_data,
                    &mut preprocessed_seed_links,
                    locked_doors,
                );
            }
        }

        let mut initial_items_remaining: Vec<usize> = vec![1; game_data.item_isv.keys.len()];
        initial_items_remaining[Item::WallJump as usize] = 0;
        initial_items_remaining[Item::Missile as usize] = 46;
        initial_items_remaining[Item::Super as usize] = 10;
        initial_items_remaining[Item::PowerBomb as usize] = 10;
        initial_items_remaining[Item::ETank as usize] = 14;
        initial_items_remaining[Item::ReserveTank as usize] = 4;
        if difficulty_tiers[0].wall_jump == WallJump::Collectible {
            initial_items_remaining[Item::Missile as usize] -= 1;
            initial_items_remaining[Item::WallJump as usize] = 1;
        }
        assert!(initial_items_remaining.iter().sum::<usize>() == game_data.item_locations.len());

        Randomizer {
            map,
            locked_doors,
            initial_items_remaining,
            game_data,
            base_links_data,
            seed_links_data: LinksDataGroup::new(
                preprocessed_seed_links,
                game_data.vertex_isv.keys.len(),
                base_links_data.links.len(),
            ),
            difficulty_tiers,
        }
    }

    pub fn get_link(&self, idx: usize) -> &Link {
        let base_links_len = self.base_links_data.links.len();
        if idx < base_links_len {
            &self.base_links_data.links[idx]
        } else {
            &self.seed_links_data.links[idx - base_links_len]
        }
    }

    fn get_initial_flag_vec(&self) -> Vec<bool> {
        let mut flag_vec = vec![false; self.game_data.flag_isv.keys.len()];
        let tourian_open_idx = self.game_data.flag_isv.index_by_key["f_TourianOpen"];
        flag_vec[tourian_open_idx] = true;
        if self.difficulty_tiers[0].all_items_spawn {
            let all_items_spawn_idx = self.game_data.flag_isv.index_by_key["f_AllItemsSpawn"];
            flag_vec[all_items_spawn_idx] = true;
        }
        if self.difficulty_tiers[0].acid_chozo {
            let acid_chozo_without_space_jump_idx =
                self.game_data.flag_isv.index_by_key["f_AcidChozoWithoutSpaceJump"];
            flag_vec[acid_chozo_without_space_jump_idx] = true;
        }
        flag_vec
    }

    fn update_reachability(&self, state: &mut RandomizationState) {
        let num_vertices = self.game_data.vertex_isv.keys.len();
        // let start_vertex_id = self.game_data.vertex_isv.index_by_key[&(8, 5, 0)]; // Landing site
        let start_vertex_id = self.game_data.vertex_isv.index_by_key
            [&(state.hub_location.room_id, state.hub_location.node_id, 0)];
        let mut forward = traverse(
            &self.base_links_data,
            &self.seed_links_data,
            None,
            &state.global_state,
            LocalState::new(),
            num_vertices,
            start_vertex_id,
            false,
            &self.difficulty_tiers[0],
            self.game_data,
        );
        let mut reverse = traverse(
            &self.base_links_data,
            &self.seed_links_data,
            None,
            &state.global_state,
            LocalState::new(),
            num_vertices,
            start_vertex_id,
            true,
            &self.difficulty_tiers[0],
            self.game_data,
        );
        for (i, vertex_ids) in self.game_data.item_vertex_ids.iter().enumerate() {
            // Clear out any previous bireachable markers (because in rare cases a previously bireachable
            // vertex can become no longer "bireachable" due to the imperfect cost heuristic used for
            // resource management.)
            state.item_location_state[i].bireachable = false;
            state.item_location_state[i].bireachable_vertex_id = None;

            for &v in vertex_ids {
                if forward.cost[v].iter().any(|&x| f32::is_finite(x)) {
                    state.item_location_state[i].reachable = true;
                    if !state.item_location_state[i].bireachable
                        && get_bireachable_idxs(&state.global_state, v, &mut forward, &mut reverse)
                            .is_some()
                    {
                        state.item_location_state[i].bireachable = true;
                        state.item_location_state[i].bireachable_vertex_id = Some(v);
                    }
                }
            }
        }
        for (i, vertex_ids) in self.game_data.flag_vertex_ids.iter().enumerate() {
            // Clear out any previous bireachable markers (because in rare cases a previously bireachable
            // vertex can become no longer "bireachable" due to the imperfect cost heuristic used for
            // resource management.)
            state.flag_location_state[i].bireachable = false;
            state.flag_location_state[i].bireachable_vertex_id = None;

            for &v in vertex_ids {
                if !state.flag_location_state[i].bireachable
                    && get_bireachable_idxs(&state.global_state, v, &mut forward, &mut reverse)
                        .is_some()
                {
                    state.flag_location_state[i].bireachable = true;
                    state.flag_location_state[i].bireachable_vertex_id = Some(v);
                }
            }
        }

        for (i, (room_id, node_id)) in self.game_data.save_locations.iter().enumerate() {
            state.save_location_state[i].bireachable = false;
            let vertex_id = self.game_data.vertex_isv.index_by_key[&(*room_id, *node_id, 0)];
            if get_bireachable_idxs(&state.global_state, vertex_id, &mut forward, &mut reverse)
                .is_some()
            {
                state.save_location_state[i].bireachable = true;
            }
        }

        // Store TraverseResults to use for constructing spoiler log
        state.debug_data = Some(DebugData {
            global_state: state.global_state.clone(),
            forward,
            reverse,
        });
    }

    fn select_items<R: Rng>(
        &self,
        state: &RandomizationState,
        num_bireachable: usize,
        num_oneway_reachable: usize,
        attempt_num: usize,
        rng: &mut R,
    ) -> Option<SelectItemsOutput> {
        let num_items_to_place = match self.difficulty_tiers[0].progression_rate {
            ProgressionRate::Slow => num_bireachable + num_oneway_reachable,
            ProgressionRate::Uniform => num_bireachable,
            ProgressionRate::Fast => num_bireachable,
        };
        let filtered_item_precedence: Vec<Item> = state
            .item_precedence
            .iter()
            .copied()
            .filter(|&item| state.items_remaining[item as usize] > 0)
            .collect();
        let num_key_items_remaining = filtered_item_precedence.len();
        let num_items_remaining: usize = state.items_remaining.iter().sum();
        let mut num_key_items_to_place = match self.difficulty_tiers[0].progression_rate {
            ProgressionRate::Slow => 1,
            ProgressionRate::Uniform => f32::ceil(
                (num_key_items_remaining as f32) / (num_items_remaining as f32)
                    * (num_items_to_place as f32),
            ) as usize,
            ProgressionRate::Fast => f32::ceil(
                2.0 * (num_key_items_remaining as f32) / (num_items_remaining as f32)
                    * (num_items_to_place as f32),
            ) as usize,
        };
        if num_items_remaining < num_items_to_place + 20 {
            num_key_items_to_place = num_key_items_remaining;
        }
        num_key_items_to_place = min(
            num_key_items_to_place,
            min(num_bireachable, num_key_items_remaining),
        );
        let mut key_items_to_place: Vec<Item> = vec![];

        if num_key_items_to_place >= 1 {
            if num_key_items_to_place - 1 + attempt_num >= num_key_items_remaining {
                return None;
            }

            // If we will be placing `k` key items, we let the first `k - 1` items to place remain fixed based on the
            // item precedence order, while we vary the last key item across attempts (to try to find some choice that
            // will expand the set of bireachable item locations).
            key_items_to_place
                .extend(filtered_item_precedence[0..(num_key_items_to_place - 1)].iter());
            key_items_to_place
                .push(filtered_item_precedence[num_key_items_to_place - 1 + attempt_num]);
            assert!(key_items_to_place.len() == num_key_items_to_place);
        } else {
            if attempt_num > 0 {
                return None;
            }
        }

        // println!("key items to place: {:?}", key_items_to_place);

        let mut new_items_remaining = state.items_remaining.clone();
        for &item in &key_items_to_place {
            new_items_remaining[item as usize] -= 1;
        }

        let num_other_items_to_place = num_items_to_place - num_key_items_to_place;

        let expansion_item_set: HashSet<Item> =
            [Item::ETank, Item::ReserveTank, Item::Super, Item::PowerBomb]
                .into_iter()
                .collect();
        let mut item_types_to_prioritize: Vec<Item> = vec![];
        let mut item_types_to_mix: Vec<Item> = vec![Item::Missile];
        let mut item_types_to_delay: Vec<Item> = vec![];
        let mut item_types_to_extra_delay: Vec<Item> = vec![];

        for &item in &state.item_precedence {
            if item == Item::Missile || new_items_remaining[item as usize] == 0 {
                continue;
            }
            if self.difficulty_tiers[0].early_filler_items.contains(&item)
                && new_items_remaining[item as usize] == self.initial_items_remaining[item as usize]
            {
                item_types_to_prioritize.push(item);
                item_types_to_mix.push(item);
            } else if self.difficulty_tiers[0].filler_items.contains(&item)
                || (self.difficulty_tiers[0].semi_filler_items.contains(&item)
                    && state.items_remaining[item as usize]
                        < self.initial_items_remaining[item as usize])
            {
                item_types_to_mix.push(item);
            } else if expansion_item_set.contains(&item) {
                item_types_to_delay.push(item);
            } else {
                item_types_to_extra_delay.push(item);
            }
        }
        // println!("prioritize: {:?}, mix: {:?}, delay: {:?}, extra: {:?}",
        //     item_types_to_prioritize, item_types_to_mix, item_types_to_delay, item_types_to_extra_delay);

        let mut items_to_mix: Vec<Item> = Vec::new();
        for &item in &item_types_to_mix {
            let mut cnt = new_items_remaining[item as usize];
            if item_types_to_prioritize.contains(&item) {
                // println!("{:?}: {} {}", item, new_items_remaining[item as usize], self.initial_items_remaining[item as usize]);
                cnt -= 1;
            }
            for _ in 0..cnt {
                items_to_mix.push(item);
            }
        }
        let mut items_to_delay: Vec<Item> = Vec::new();
        for &item in &item_types_to_delay {
            for _ in 0..new_items_remaining[item as usize] {
                items_to_delay.push(item);
            }
        }
        let mut items_to_extra_delay: Vec<Item> = Vec::new();
        for &item in &item_types_to_extra_delay {
            for _ in 0..new_items_remaining[item as usize] {
                items_to_extra_delay.push(item);
            }
        }
        items_to_mix.shuffle(rng);
        let mut other_items_to_place: Vec<Item> = item_types_to_prioritize;
        other_items_to_place.extend(items_to_mix);
        other_items_to_place.extend(items_to_delay);
        other_items_to_place.extend(items_to_extra_delay);
        other_items_to_place = other_items_to_place[0..num_other_items_to_place].to_vec();

        // println!("other items to place: {:?}", other_items_to_place);

        for &item in &other_items_to_place {
            new_items_remaining[item as usize] -= 1;
        }
        Some(SelectItemsOutput {
            key_items: key_items_to_place,
            other_items: other_items_to_place,
            new_items_remaining,
        })
    }

    fn get_init_traverse(
        &self,
        state: &RandomizationState,
        init_traverse: Option<&TraverseResult>,
    ) -> Option<TraverseResult> {
        if let Some(init) = init_traverse {
            let mut out = init.clone();
            for v in 0..init.local_states.len() {
                if !state.key_visited_vertices.contains(&v) {
                    out.local_states[v] = [IMPOSSIBLE_LOCAL_STATE; NUM_COST_METRICS];
                    out.cost[v] = [f32::INFINITY; NUM_COST_METRICS];
                    out.start_trail_ids[v] = [-1; NUM_COST_METRICS];
                }
            }
            Some(out)
        } else {
            None
        }
    }

    fn find_hard_location(
        &self,
        state: &RandomizationState,
        bireachable_locations: &[ItemLocationId],
        init_traverse: Option<&TraverseResult>,
    ) -> (usize, usize) {
        // For forced mode, we prioritize placing a key item at a location that is inaccessible at
        // lower difficulty tiers. This function returns an index into `bireachable_locations`, identifying
        // a location with the hardest possible difficulty to reach.
        let num_vertices = self.game_data.vertex_isv.keys.len();
        // let start_vertex_id = self.game_data.vertex_isv.index_by_key[&(8, 5, 0)]; // Landing site
        let start_vertex_id = self.game_data.vertex_isv.index_by_key
            [&(state.hub_location.room_id, state.hub_location.node_id, 0)];

        // for &v in &state.key_visited_vertices {
        //     println!("key visited: {:?}", self.game_data.vertex_isv.keys[v]);
        // }
        // println!("items: {:?}", state.global_state.items);

        // println!("global_state: {:?}", state.global_state);
        // print!("Flags: ");
        // for (i, flag) in self.game_data.flag_isv.keys.iter().enumerate() {
        //     if state.global_state.flags[i] {
        //         print!("{} ", flag);
        //     }
        // }
        // println!("");

        for tier in 1..self.difficulty_tiers.len() {
            let difficulty = &self.difficulty_tiers[tier];
            let mut tmp_global = state.global_state.clone();
            tmp_global.tech = get_tech_vec(&self.game_data, difficulty);
            tmp_global.notable_strats = get_strat_vec(&self.game_data, difficulty);
            tmp_global.shine_charge_tiles = difficulty.shine_charge_tiles;
            // print!("tier:{} tech:", tier);
            // for (i, tech) in self.game_data.tech_isv.keys.iter().enumerate() {
            //     if tmp_global.tech[i] {
            //         print!("{} ", tech);
            //     }
            // }
            // println!("");
            let traverse_result = traverse(
                &self.base_links_data,
                &self.seed_links_data,
                self.get_init_traverse(state, init_traverse),
                &tmp_global,
                LocalState::new(),
                num_vertices,
                start_vertex_id,
                false,
                difficulty,
                self.game_data,
            );

            for (i, &item_location_id) in bireachable_locations.iter().enumerate() {
                let mut is_reachable = false;
                for &v in &self.game_data.item_vertex_ids[item_location_id] {
                    if traverse_result.cost[v].iter().any(|&x| f32::is_finite(x)) {
                        is_reachable = true;
                    }
                }
                if !is_reachable {
                    return (i, tier);
                }
            }
        }
        return (0, self.difficulty_tiers.len());
    }

    fn place_items(
        &self,
        attempt_num_rando: usize,
        state: &RandomizationState,
        new_state: &mut RandomizationState,
        bireachable_locations: &[ItemLocationId],
        other_locations: &[ItemLocationId],
        key_items_to_place: &[Item],
        other_items_to_place: &[Item],
    ) {
        info!(
            "[attempt {attempt_num_rando}] Placing {:?}, {:?}",
            key_items_to_place, other_items_to_place
        );
        // println!("[attempt {attempt_num_rando}] # bireachable = {}", bireachable_locations.len());
        let mut new_bireachable_locations: Vec<ItemLocationId> = bireachable_locations.to_vec();
        if self.difficulty_tiers.len() > 1 {
            let traverse_result = match state.previous_debug_data.as_ref() {
                Some(x) => Some(&x.forward),
                None => None,
            };
            for i in 0..key_items_to_place.len() {
                let (hard_idx, tier) = if key_items_to_place.len() > 1 {
                    // We're placing more than one key item in this step. Obtaining some of them could help make
                    // others easier to obtain. So we use "new_state" to try to find locations that are still hard to
                    // reach even with the new items.
                    self.find_hard_location(
                        new_state,
                        &new_bireachable_locations[i..],
                        traverse_result,
                    )
                } else {
                    // We're only placing one key item in this step. Try to find a location that is hard to reach
                    // without already having the new item.
                    self.find_hard_location(state, &new_bireachable_locations[i..], traverse_result)
                };
                info!(
                    "[attempt {attempt_num_rando}] {:?} in tier {} (of {})",
                    key_items_to_place[i],
                    tier,
                    self.difficulty_tiers.len()
                );

                let hard_loc = new_bireachable_locations[i + hard_idx];
                new_bireachable_locations.swap(i, i + hard_idx);

                // Mark the vertices along the path to the newly chosen hard location. Vertices that are
                // easily accessible from along this path are then discouraged from being chosen later
                // as hard locations (since the point of forced mode is to ensure unique hard strats
                // are required; we don't want it to be the same hard strat over and over again).
                let hard_vertex_id = state.item_location_state[hard_loc]
                    .bireachable_vertex_id
                    .unwrap();
                let forward = &state.debug_data.as_ref().unwrap().forward;
                let reverse = &state.debug_data.as_ref().unwrap().reverse;
                let (forward_cost_idx, _) =
                    get_bireachable_idxs(&state.global_state, hard_vertex_id, forward, reverse)
                        .unwrap();
                let route = get_spoiler_route(
                    &state.debug_data.as_ref().unwrap().forward,
                    hard_vertex_id,
                    forward_cost_idx,
                );
                for &link_idx in &route {
                    let vertex_id = self.get_link(link_idx as usize).to_vertex_id;
                    new_state.key_visited_vertices.insert(vertex_id);
                }
            }
        }

        let mut all_locations: Vec<ItemLocationId> = Vec::new();
        all_locations.extend(new_bireachable_locations);
        all_locations.extend(other_locations);
        let mut all_items_to_place: Vec<Item> = Vec::new();
        all_items_to_place.extend(key_items_to_place);
        all_items_to_place.extend(other_items_to_place);
        assert!(all_locations.len() == all_items_to_place.len());
        for (&loc, &item) in iter::zip(&all_locations, &all_items_to_place) {
            new_state.item_location_state[loc].placed_item = Some(item);
        }
    }

    fn finish(&self, attempt_num_rando: usize, state: &mut RandomizationState) {
        let mut remaining_items: Vec<Item> = Vec::new();
        for item_id in 0..self.game_data.item_isv.keys.len() {
            for _ in 0..state.items_remaining[item_id] {
                remaining_items.push(Item::try_from(item_id).unwrap());
            }
        }
        info!(
            "[attempt {attempt_num_rando}] Finishing with {:?}",
            remaining_items
        );
        let mut idx = 0;
        for item_loc_state in &mut state.item_location_state {
            if item_loc_state.placed_item.is_none() {
                item_loc_state.placed_item = Some(remaining_items[idx]);
                idx += 1;
            }
        }
        assert!(idx == remaining_items.len());
    }

    fn provides_progression(
        &self,
        old_state: &RandomizationState,
        new_state: &mut RandomizationState,
        select: &SelectItemsOutput,
        placed_uncollected_bireachable_items: &[Item],
        num_unplaced_bireachable: usize,
    ) -> bool {
        // Collect all the items that would be collectible in this scenario:
        // 1) Items that were already placed on an earlier step; this is only applicable to filler items
        // (normally Missiles) on Slow progression, which became one-way reachable on an earlier step but are now
        // bireachable.
        // 2) Key items,
        // 3) Other items
        for &item in placed_uncollected_bireachable_items.iter().chain(
            select
                .key_items
                .iter()
                .chain(select.other_items.iter())
                .take(num_unplaced_bireachable),
        ) {
            new_state.global_state.collect(item, self.game_data);
        }

        // info!("Trying placing {:?}", key_items_to_place);

        self.update_reachability(new_state);
        let num_bireachable = new_state
            .item_location_state
            .iter()
            .filter(|x| x.bireachable)
            .count();
        let num_reachable = new_state
            .item_location_state
            .iter()
            .filter(|x| x.reachable)
            .count();
        let num_one_way_reachable = num_reachable - num_bireachable;

        // Maximum acceptable number of one-way-reachable items. This is to try to avoid extreme
        // cases where the player would gain access to very large areas that they cannot return from:
        let one_way_reachable_limit = 20;
        // let one_way_reachable_limit = 100;

        // Check if all items are already bireachable. It isn't necessary for correctness to check this case,
        // but it speeds up the last step, where no further progress is possible (meaning there is no point
        // trying a bunch of possible key items to place to try to make more progress.
        let all_items_bireachable = num_bireachable == new_state.item_location_state.len();

        let gives_expansion = if all_items_bireachable {
            true
        } else if self.difficulty_tiers[0].progression_rate == ProgressionRate::Slow {
            iter::zip(
                &new_state.item_location_state,
                &old_state.item_location_state,
            )
            .any(|(n, o)| n.bireachable && !o.reachable)
        } else {
            iter::zip(
                &new_state.item_location_state,
                &old_state.item_location_state,
            )
            .any(|(n, o)| n.bireachable && !o.bireachable)
        };

        num_one_way_reachable < one_way_reachable_limit && gives_expansion
    }

    fn multi_attempt_select_items<R: Rng + Clone>(
        &self,
        attempt_num_rando: usize,
        state: &RandomizationState,
        placed_uncollected_bireachable_items: &[Item],
        num_unplaced_bireachable: usize,
        num_unplaced_oneway_reachable: usize,
        rng: &mut R,
    ) -> (SelectItemsOutput, RandomizationState) {
        let mut attempt_num = 0;
        let mut selection = self
            .select_items(
                state,
                num_unplaced_bireachable,
                num_unplaced_oneway_reachable,
                attempt_num,
                rng,
            )
            .unwrap();

        loop {
            let mut new_state: RandomizationState = RandomizationState {
                step_num: state.step_num,
                start_location: state.start_location.clone(),
                hub_location: state.hub_location.clone(),
                item_precedence: state.item_precedence.clone(),
                item_location_state: state.item_location_state.clone(),
                flag_location_state: state.flag_location_state.clone(),
                save_location_state: state.save_location_state.clone(),
                items_remaining: selection.new_items_remaining.clone(),
                global_state: state.global_state.clone(),
                debug_data: None,
                previous_debug_data: None,
                key_visited_vertices: HashSet::new(),
            };

            if self.provides_progression(
                &state,
                &mut new_state,
                &selection,
                &placed_uncollected_bireachable_items,
                num_unplaced_bireachable,
            ) {
                return (selection, new_state);
            }

            let new_selection = self.select_items(
                state,
                num_unplaced_bireachable,
                num_unplaced_oneway_reachable,
                attempt_num,
                rng,
            );
            if let Some(s) = new_selection {
                selection = s;
            } else {
                info!("[attempt {attempt_num_rando}] Exhausted key item placement attempts");
                return (selection, new_state);
            }
            attempt_num += 1;
        }
    }

    fn step<R: Rng + Clone>(
        &self,
        attempt_num_rando: usize,
        state: &mut RandomizationState,
        rng: &mut R,
    ) -> (SpoilerSummary, SpoilerDetails) {
        let orig_global_state = state.global_state.clone();
        let mut spoiler_flag_summaries: Vec<SpoilerFlagSummary> = Vec::new();
        let mut spoiler_flag_details: Vec<SpoilerFlagDetails> = Vec::new();
        loop {
            let mut any_new_flag = false;
            for i in 0..self.game_data.flag_locations.len() {
                let flag_id = self.game_data.flag_locations[i].2;
                if state.global_state.flags[flag_id] {
                    continue;
                }
                if state.flag_location_state[i].bireachable {
                    any_new_flag = true;
                    let flag_vertex_id =
                        state.flag_location_state[i].bireachable_vertex_id.unwrap();
                    spoiler_flag_summaries.push(self.get_spoiler_flag_summary(
                        &state,
                        flag_vertex_id,
                        flag_id,
                    ));
                    spoiler_flag_details.push(self.get_spoiler_flag_details(
                        &state,
                        flag_vertex_id,
                        flag_id,
                    ));
                    state.global_state.flags[flag_id] = true;
                }
            }
            if any_new_flag {
                self.update_reachability(state);
            } else {
                break;
            }
        }

        let mut placed_uncollected_bireachable_loc: Vec<ItemLocationId> = Vec::new();
        let mut placed_uncollected_bireachable_items: Vec<Item> = Vec::new();
        let mut unplaced_bireachable: Vec<ItemLocationId> = Vec::new();
        let mut unplaced_oneway_reachable: Vec<ItemLocationId> = Vec::new();
        for (i, item_location_state) in state.item_location_state.iter().enumerate() {
            if let Some(item) = item_location_state.placed_item {
                if !item_location_state.collected && item_location_state.bireachable {
                    placed_uncollected_bireachable_loc.push(i);
                    placed_uncollected_bireachable_items.push(item);
                }
            } else {
                if item_location_state.bireachable {
                    unplaced_bireachable.push(i);
                } else if item_location_state.reachable {
                    unplaced_oneway_reachable.push(i);
                }
            }
        }
        unplaced_bireachable.shuffle(rng);
        unplaced_oneway_reachable.shuffle(rng);
        let (selection, mut new_state) = self.multi_attempt_select_items(
            attempt_num_rando,
            &state,
            &placed_uncollected_bireachable_items,
            unplaced_bireachable.len(),
            unplaced_oneway_reachable.len(),
            rng,
        );
        new_state.previous_debug_data = state.debug_data.clone();
        new_state.key_visited_vertices = state.key_visited_vertices.clone();

        // Mark the newly collected items that were placed on earlier steps:
        for &loc in &placed_uncollected_bireachable_loc {
            new_state.item_location_state[loc].collected = true;
        }

        // Place the new items:
        if self.difficulty_tiers[0].progression_rate == ProgressionRate::Slow {
            // With Slow progression, place items in all newly reachable locations (bireachable as
            // well as one-way-reachable locations). One-way-reachable locations are filled only
            // with non-key items, to minimize the possibility of them being usable to break from the
            // intended sequence.
            self.place_items(
                attempt_num_rando,
                &state,
                &mut new_state,
                &unplaced_bireachable,
                &unplaced_oneway_reachable,
                &selection.key_items,
                &selection.other_items,
            );
        } else {
            // In Uniform and Fast progression, only place items at bireachable locations. We defer placing items at
            // one-way-reachable locations so that they may get key items placed there later after
            // becoming bireachable.
            self.place_items(
                attempt_num_rando,
                &state,
                &mut new_state,
                &unplaced_bireachable,
                &[],
                &selection.key_items,
                &selection.other_items,
            );
        }

        // Mark the newly placed bireachable items as collected:
        for &loc in &unplaced_bireachable {
            new_state.item_location_state[loc].collected = true;
        }

        let spoiler_summary = self.get_spoiler_summary(
            &orig_global_state,
            state,
            &new_state,
            spoiler_flag_summaries,
        );
        let spoiler_details =
            self.get_spoiler_details(&orig_global_state, state, &new_state, spoiler_flag_details);
        *state = new_state;
        (spoiler_summary, spoiler_details)
    }

    fn get_randomization(
        &self,
        state: &RandomizationState,
        spoiler_summaries: Vec<SpoilerSummary>,
        spoiler_details: Vec<SpoilerDetails>,
        mut debug_data_vec: Vec<DebugData>,
        seed: usize,
        display_seed: usize,
    ) -> Result<Randomization> {
        // Compute the first step on which each node becomes reachable/bireachable:
        let mut node_reachable_step: HashMap<(RoomId, NodeId), usize> = HashMap::new();
        let mut node_bireachable_step: HashMap<(RoomId, NodeId), usize> = HashMap::new();
        let mut map_tile_reachable_step: HashMap<(RoomId, (usize, usize)), usize> = HashMap::new();
        let mut map_tile_bireachable_step: HashMap<(RoomId, (usize, usize)), usize> =
            HashMap::new();

        for (step, debug_data) in debug_data_vec.iter_mut().enumerate() {
            // println!("step={}, global_state={:?}", step, debug_data.global_state);
            for (v, (room_id, node_id, _obstacle_bitmask)) in
                self.game_data.vertex_isv.keys.iter().enumerate()
            {
                if node_bireachable_step.contains_key(&(*room_id, *node_id)) {
                    continue;
                }
                if get_bireachable_idxs(
                    &debug_data.global_state,
                    v,
                    &mut debug_data.forward,
                    &mut debug_data.reverse,
                )
                .is_some()
                {
                    node_bireachable_step.insert((*room_id, *node_id), step);
                    let room_ptr = self.game_data.room_ptr_by_id[room_id];
                    let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
                    if let Some(coords) = self.game_data.node_tile_coords.get(&(*room_id, *node_id))
                    {
                        for (x, y) in coords.iter().copied() {
                            let key = (room_idx, (x, y));
                            if !map_tile_bireachable_step.contains_key(&key) {
                                map_tile_bireachable_step.insert(key, step);
                            }
                        }
                    }
                }

                if node_reachable_step.contains_key(&(*room_id, *node_id)) {
                    continue;
                }
                if debug_data.forward.cost[v]
                    .iter()
                    .any(|&x| f32::is_finite(x))
                {
                    node_reachable_step.insert((*room_id, *node_id), step);
                    let room_ptr = self.game_data.room_ptr_by_id[room_id];
                    let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
                    if let Some(coords) = self.game_data.node_tile_coords.get(&(*room_id, *node_id))
                    {
                        for (x, y) in coords.iter().copied() {
                            let key = (room_idx, (x, y));
                            if !map_tile_reachable_step.contains_key(&key) {
                                map_tile_reachable_step.insert(key, step);
                            }
                        }
                    }
                }
            }
        }

        let item_placement: Vec<Item> = state
            .item_location_state
            .iter()
            .map(|x| x.placed_item.unwrap())
            .collect();
        let spoiler_all_items = state
            .item_location_state
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let (r, n) = self.game_data.item_locations[i];
                let item_vertex_info = self.get_vertex_info_by_id(r, n);
                let location = SpoilerLocation {
                    area: item_vertex_info.area_name,
                    room: item_vertex_info.room_name,
                    node: item_vertex_info.node_name,
                    coords: item_vertex_info.room_coords,
                };
                let item = x.placed_item.unwrap();
                SpoilerItemLoc {
                    item: Item::VARIANTS[item as usize].to_string(),
                    location,
                }
            })
            .collect();
        let spoiler_all_rooms = self
            .map
            .rooms
            .iter()
            .enumerate()
            .zip(self.game_data.room_geometry.iter())
            .map(|((room_idx, c), g)| {
                let room = self.game_data.room_id_by_ptr[&g.rom_address];
                // let room = self.game_data.room_json_map[&room]["name"]
                //     .as_str()
                //     .unwrap()
                //     .to_string();
                let room = g.name.clone();
                let short_name = strip_name(&room);
                let height = g.map.len();
                let width = g.map[0].len();
                let mut map_reachable_step: Vec<Vec<u8>> = vec![vec![255; width]; height];
                let mut map_bireachable_step: Vec<Vec<u8>> = vec![vec![255; width]; height];
                for y in 0..height {
                    for x in 0..width {
                        if g.map[y][x] != 0 {
                            let key = (room_idx, (x, y));
                            if let Some(step) = map_tile_reachable_step.get(&key) {
                                map_reachable_step[y][x] = *step as u8;
                            }
                            if let Some(step) = map_tile_bireachable_step.get(&key) {
                                map_bireachable_step[y][x] = *step as u8;
                            }
                        }
                    }
                }
                SpoilerRoomLoc {
                    room,
                    short_name,
                    map: g.map.clone(),
                    map_reachable_step,
                    map_bireachable_step,
                    coords: *c,
                }
            })
            .collect();
        let spoiler_escape =
            escape_timer::compute_escape_data(self.game_data, self.map, &self.difficulty_tiers[0])?;
        let spoiler_log = SpoilerLog {
            summary: spoiler_summaries,
            escape: spoiler_escape,
            details: spoiler_details,
            all_items: spoiler_all_items,
            all_rooms: spoiler_all_rooms,
        };

        // // Messing around with starting location. TODO: remove this
        // let start_locations: Vec<StartLocation> =
        //     serde_json::from_str(&std::fs::read_to_string(&"data/start_locations.json").unwrap()).unwrap();
        // let loc = start_locations.last().unwrap();

        Ok(Randomization {
            difficulty: self.difficulty_tiers[0].clone(),
            map: self.map.clone(),
            locked_doors: self.locked_doors.to_vec(),
            item_placement,
            spoiler_log,
            seed,
            display_seed,
            // start_location: loc.clone(),
            start_location: state.start_location.clone(),
        })
    }

    fn get_item_precedence<R: Rng>(
        &self,
        item_priorities: &[ItemPriorityGroup],
        rng: &mut R,
    ) -> Vec<Item> {
        let mut item_precedence: Vec<Item> = Vec::new();
        if self.difficulty_tiers[0].progression_rate == ProgressionRate::Slow {
            // With slow progression, prioritize placing a missile over other key items.
            item_precedence.push(Item::Missile);
        }
        for priority_group in item_priorities {
            let mut items = priority_group.items.clone();
            items.shuffle(rng);
            for item_name in &items {
                let item_idx = self.game_data.item_isv.index_by_key[item_name];
                item_precedence.push(Item::try_from(item_idx).unwrap());
            }
        }
        item_precedence
    }

    fn rerandomize_tank_precedence<R: Rng>(
        &self,
        item_precedence: &mut [Item],
        rng: &mut R,
    ) {
        if rng.gen_bool(0.5) {
            return;
        }
        let etank_idx = item_precedence.iter().position(|&x| x == Item::ETank).unwrap();
        let reserve_idx = item_precedence.iter().position(|&x| x == Item::ReserveTank).unwrap();
        item_precedence[etank_idx] = Item::ReserveTank;
        item_precedence[reserve_idx] = Item::ETank;
    }

    pub fn determine_start_location<R: Rng>(
        &self,
        attempt_num_rando: usize,
        num_attempts: usize,
        rng: &mut R,
    ) -> Result<(StartLocation, HubLocation)> {
        if !self.difficulty_tiers[0].randomized_start {
            let mut ship_start = StartLocation::default();
            ship_start.name = "Ship".to_string();
            ship_start.room_id = 8;
            ship_start.node_id = 5;
            ship_start.door_load_node_id = Some(2);
            ship_start.x = 72.0;
            ship_start.y = 69.5;

            let mut ship_hub = HubLocation::default();
            ship_hub.name = "Ship".to_string();
            ship_hub.room_id = 8;
            ship_hub.node_id = 5;

            return Ok((ship_start, ship_hub));
        }
        'attempt: for i in 0..num_attempts {
            info!("[attempt {attempt_num_rando}] start location attempt {}", i);
            let start_loc_idx = rng.gen_range(0..self.game_data.start_locations.len());
            let start_loc = self.game_data.start_locations[start_loc_idx].clone();

            info!("[attempt {attempt_num_rando}] start: {:?}", start_loc);
            let num_vertices = self.game_data.vertex_isv.keys.len();
            let start_vertex_id =
                self.game_data.vertex_isv.index_by_key[&(start_loc.room_id, start_loc.node_id, 0)];
            let global = self.get_initial_global_state();
            let local = apply_requirement(
                &start_loc.requires_parsed.as_ref().unwrap(),
                &global,
                LocalState::new(),
                false,
                &self.difficulty_tiers[0],
                self.game_data,
            );
            if local.is_none() {
                continue;
            }
            let forward = traverse(
                &self.base_links_data,
                &self.seed_links_data,
                None,
                &global,
                local.unwrap(),
                num_vertices,
                start_vertex_id,
                false,
                &self.difficulty_tiers[0],
                self.game_data,
            );
            let forward0 = traverse(
                &self.base_links_data,
                &self.seed_links_data,
                None,
                &global,
                LocalState::new(),
                num_vertices,
                start_vertex_id,
                false,
                &self.difficulty_tiers[0],
                self.game_data,
            );
            let reverse = traverse(
                &self.base_links_data,
                &self.seed_links_data,
                None,
                &global,
                LocalState::new(),
                num_vertices,
                start_vertex_id,
                true,
                &self.difficulty_tiers[0],
                self.game_data,
            );

            // We require several conditions for a start location to be valid with a given hub location:
            // 1) The hub location must be one-way reachable from the start location, including initial start location
            // requirements (e.g. including requirements to reach the starting node from the actual start location, which
            // may not be at a node)
            // 2) The starting node (not the actual start location) must be bireachable from the hub location
            // (ie. there must be a logical round-trip path from the hub to the starting node and back)
            // 3) Any logical requirements on the hub must be satisfied.
            // 4) The Ship must not be bireachable from the hub.
            for hub in &self.game_data.hub_locations {
                let hub_vertex_id =
                    self.game_data.vertex_isv.index_by_key[&(hub.room_id, hub.node_id, 0)];
                if forward.cost[hub_vertex_id]
                    .iter()
                    .any(|&x| f32::is_finite(x))
                    && get_bireachable_idxs(&global, hub_vertex_id, &forward0, &reverse).is_some()
                {
                    if hub.room_id == 8 {
                        // Reject starting location if the Ship is initially bireachable from it.
                        // (Note: The Ship is first in hub_locations.json, so this check happens before other hubs are considered.)
                        continue 'attempt;
                    }

                    let local = apply_requirement(
                        &hub.requires_parsed.as_ref().unwrap(),
                        &global,
                        LocalState::new(),
                        false,
                        &self.difficulty_tiers[0],
                        self.game_data,
                    );
                    if local.is_some() {
                        return Ok((start_loc, hub.clone()));
                    }
                }
            }
        }
        bail!("[attempt {attempt_num_rando}] Failed to find start location.")
    }

    fn get_initial_global_state(&self) -> GlobalState {
        let items = vec![false; self.game_data.item_isv.keys.len()];
        let weapon_mask = self.game_data.get_weapon_mask(&items);
        GlobalState {
            tech: get_tech_vec(&self.game_data, &self.difficulty_tiers[0]),
            notable_strats: get_strat_vec(&self.game_data, &self.difficulty_tiers[0]),
            items: items,
            flags: self.get_initial_flag_vec(),
            max_energy: 99,
            max_reserves: 0,
            max_missiles: 0,
            max_supers: 0,
            max_power_bombs: 0,
            weapon_mask: weapon_mask,
            shine_charge_tiles: self.difficulty_tiers[0].shine_charge_tiles,
        }
    }

    pub fn randomize(
        &self,
        attempt_num_rando: usize,
        seed: usize,
        display_seed: usize,
    ) -> Result<Randomization> {
        let mut rng_seed = [0u8; 32];
        rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
        let mut rng = rand::rngs::StdRng::from_seed(rng_seed);
        let initial_global_state = self.get_initial_global_state();
        let initial_item_location_state = ItemLocationState {
            placed_item: None,
            collected: false,
            reachable: false,
            bireachable: false,
            bireachable_vertex_id: None,
        };
        let initial_flag_location_state = FlagLocationState {
            bireachable: false,
            bireachable_vertex_id: None,
        };
        let initial_save_location_state = SaveLocationState { bireachable: false };
        let num_attempts_start_location = 10;
        let (start_location, hub_location) = self.determine_start_location(
            attempt_num_rando,
            num_attempts_start_location,
            &mut rng,
        )?;
        let item_precedence: Vec<Item> =
            self.get_item_precedence(&self.difficulty_tiers[0].item_priorities, &mut rng);
        info!(
            "[attempt {attempt_num_rando}] Item precedence: {:?}",
            item_precedence
        );
        let mut state = RandomizationState {
            step_num: 1,
            item_precedence,
            start_location,
            hub_location,
            item_location_state: vec![
                initial_item_location_state;
                self.game_data.item_locations.len()
            ],
            flag_location_state: vec![
                initial_flag_location_state;
                self.game_data.flag_locations.len()
            ],
            save_location_state: vec![
                initial_save_location_state;
                self.game_data.save_locations.len()
            ],
            items_remaining: self.initial_items_remaining.clone(),
            global_state: initial_global_state,
            debug_data: None,
            previous_debug_data: None,
            key_visited_vertices: HashSet::new(),
        };
        self.update_reachability(&mut state);
        if !state.item_location_state.iter().any(|x| x.bireachable) {
            bail!("[attempt {attempt_num_rando}] No initially bireachable item locations");
        }
        let mut spoiler_summary_vec: Vec<SpoilerSummary> = Vec::new();
        let mut spoiler_details_vec: Vec<SpoilerDetails> = Vec::new();
        let mut debug_data_vec: Vec<DebugData> = Vec::new();
        loop {
            if self.difficulty_tiers[0].random_tank {
                self.rerandomize_tank_precedence(&mut state.item_precedence, &mut rng);
            }
            let (spoiler_summary, spoiler_details) =
                self.step(attempt_num_rando, &mut state, &mut rng);
            let cnt_collected = state
                .item_location_state
                .iter()
                .filter(|x| x.collected)
                .count();
            let cnt_placed = state
                .item_location_state
                .iter()
                .filter(|x| x.placed_item.is_some())
                .count();
            let cnt_reachable = state
                .item_location_state
                .iter()
                .filter(|x| x.reachable)
                .count();
            let cnt_bireachable = state
                .item_location_state
                .iter()
                .filter(|x| x.bireachable)
                .count();
            info!("[attempt {attempt_num_rando}] step={0}, bireachable={cnt_bireachable}, reachable={cnt_reachable}, placed={cnt_placed}, collected={cnt_collected}", state.step_num);

            if spoiler_summary.items.len() > 0 || spoiler_summary.flags.len() > 0 {
                // If we gained anything on this step (an item or a flag), generate a spoiler step:
                spoiler_summary_vec.push(spoiler_summary);
                spoiler_details_vec.push(spoiler_details);
                // Append `debug_data`
                debug_data_vec.push(state.previous_debug_data.as_ref().unwrap().clone());
            } else {
                // No further progress was made on the last step. So we are done with this attempt: either we have
                // succeeded or we have failed.

                // Check that at least one instance of each item can be collected.
                for i in 0..self.initial_items_remaining.len() {
                    if self.initial_items_remaining[i] > 0 && !state.global_state.items[i] {
                        bail!("[attempt {attempt_num_rando}] Attempt failed: Key items not all collectible");
                    }
                }

                // Check that Phantoon can be defeated. This is to rule out the possibility that Phantoon may be locked
                // behind Bowling Alley.
                let phantoon_flag_id = self.game_data.flag_isv.index_by_key["f_DefeatedPhantoon"];
                let mut phantoon_defeated = false;
                for (i, (_, _, flag_id)) in self.game_data.flag_locations.iter().enumerate() {
                    if *flag_id == phantoon_flag_id && state.flag_location_state[i].bireachable {
                        phantoon_defeated = true;
                    }
                }
                if !phantoon_defeated {
                    bail!("[attempt {attempt_num_rando}] Attempt failed: Phantoon not defeated");
                }

                // Success:
                break;
            }

            if state.step_num == 1 && self.difficulty_tiers[0].early_save {
                if !state.save_location_state.iter().any(|x| x.bireachable) {
                    bail!(
                        "[attempt {attempt_num_rando}] Attempt failed: no accessible save location"
                    );
                }
            }
            state.step_num += 1;
        }
        self.finish(attempt_num_rando, &mut state);
        self.get_randomization(
            &state,
            spoiler_summary_vec,
            spoiler_details_vec,
            debug_data_vec,
            seed,
            display_seed,
        )
    }
}

// Spoiler log ---------------------------------------------------------

#[derive(Serialize, Deserialize, Debug)]
pub struct SpoilerRouteEntry {
    area: String,
    room: String,
    node: String,
    short_room: String,
    from_node_id: usize,
    to_node_id: usize,
    obstacles_bitmask: usize,
    coords: (usize, usize),
    strat_name: String,
    short_strat_name: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    strat_notes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    energy_remaining: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reserves_remaining: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    missiles_remaining: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    supers_remaining: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    power_bombs_remaining: Option<Capacity>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerLocation {
    pub area: String,
    pub room: String,
    pub node: String,
    pub coords: (usize, usize),
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerStartState {
    max_energy: Capacity,
    max_reserves: Capacity,
    max_missiles: Capacity,
    max_supers: Capacity,
    max_power_bombs: Capacity,
    items: Vec<String>,
    flags: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerItemDetails {
    item: String,
    location: SpoilerLocation,
    obtain_route: Vec<SpoilerRouteEntry>,
    return_route: Vec<SpoilerRouteEntry>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerFlagDetails {
    flag: String,
    location: SpoilerLocation,
    obtain_route: Vec<SpoilerRouteEntry>,
    return_route: Vec<SpoilerRouteEntry>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerDetails {
    step: usize,
    start_state: SpoilerStartState,
    flags: Vec<SpoilerFlagDetails>,
    items: Vec<SpoilerItemDetails>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerItemLoc {
    item: String,
    location: SpoilerLocation,
}
#[derive(Serialize, Deserialize)]
pub struct SpoilerRoomLoc {
    // here temporarily, most likely, since these can be baked into the web UI
    room: String,
    short_name: String,
    map: Vec<Vec<u8>>,
    map_reachable_step: Vec<Vec<u8>>,
    map_bireachable_step: Vec<Vec<u8>>,
    coords: (usize, usize),
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerItemSummary {
    pub item: String,
    pub location: SpoilerLocation,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerFlagSummary {
    flag: String,
    // area: String,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerSummary {
    pub step: usize,
    pub flags: Vec<SpoilerFlagSummary>,
    pub items: Vec<SpoilerItemSummary>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerLog {
    pub summary: Vec<SpoilerSummary>,
    pub escape: SpoilerEscape,
    pub details: Vec<SpoilerDetails>,
    pub all_items: Vec<SpoilerItemLoc>,
    pub all_rooms: Vec<SpoilerRoomLoc>,
}

struct ResourcesRemaining {
    energy: Option<Capacity>,
    reserves: Option<Capacity>,
    missiles: Option<Capacity>,
    supers: Option<Capacity>,
    power_bombs: Option<Capacity>,
}

impl<'a> Randomizer<'a> {
    fn get_vertex_info(&self, vertex_id: usize) -> VertexInfo {
        let (room_id, node_id, _obstacle_bitmask) = self.game_data.vertex_isv.keys[vertex_id];
        self.get_vertex_info_by_id(room_id, node_id)
    }
    fn get_vertex_info_by_id(&self, room_id: RoomId, node_id: NodeId) -> VertexInfo {
        let room_ptr = self.game_data.room_ptr_by_id[&room_id];
        let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
        let area = self.map.area[room_idx];
        let room_coords = self.map.rooms[room_idx];
        VertexInfo {
            area_name: self.game_data.area_names[area].clone(),
            room_name: self.game_data.room_json_map[&room_id]["name"]
                .as_str()
                .unwrap()
                .to_string(),
            room_coords,
            node_name: self.game_data.node_json_map[&(room_id, node_id)]["name"]
                .as_str()
                .unwrap()
                .to_string(),
            node_id,
        }
    }

    fn get_spoiler_start_state(&self, global_state: &GlobalState) -> SpoilerStartState {
        let mut items: Vec<String> = Vec::new();
        for i in 0..self.game_data.item_isv.keys.len() {
            if global_state.items[i] {
                items.push(self.game_data.item_isv.keys[i].to_string());
            }
        }
        let mut flags: Vec<String> = Vec::new();
        for i in 0..self.game_data.flag_isv.keys.len() {
            if global_state.flags[i] {
                flags.push(self.game_data.flag_isv.keys[i].to_string());
            }
        }
        SpoilerStartState {
            max_energy: global_state.max_energy,
            max_reserves: global_state.max_reserves,
            max_missiles: global_state.max_missiles,
            max_supers: global_state.max_supers,
            max_power_bombs: global_state.max_power_bombs,
            items: items,
            flags: flags,
        }
    }

    fn get_resources_remaining(
        &self,
        global_state: &GlobalState,
        local_state: LocalState,
        new_local_state: LocalState,
    ) -> ResourcesRemaining {
        let energy_remaining: Option<Capacity> =
            if new_local_state.energy_used != local_state.energy_used {
                Some(global_state.max_energy - new_local_state.energy_used)
            } else {
                None
            };
        let reserves_remaining: Option<Capacity> =
            if new_local_state.reserves_used != local_state.reserves_used {
                Some(global_state.max_reserves - new_local_state.reserves_used)
            } else {
                None
            };
        let missiles_remaining: Option<Capacity> =
            if new_local_state.missiles_used != local_state.missiles_used {
                Some(global_state.max_missiles - new_local_state.missiles_used)
            } else {
                None
            };
        let supers_remaining: Option<Capacity> =
            if new_local_state.supers_used != local_state.supers_used {
                Some(global_state.max_supers - new_local_state.supers_used)
            } else {
                None
            };
        let power_bombs_remaining: Option<Capacity> =
            if new_local_state.power_bombs_used != local_state.power_bombs_used {
                Some(global_state.max_power_bombs - new_local_state.power_bombs_used)
            } else {
                None
            };
        ResourcesRemaining {
            energy: energy_remaining,
            reserves: reserves_remaining,
            missiles: missiles_remaining,
            supers: supers_remaining,
            power_bombs: power_bombs_remaining,
        }
    }

    fn get_spoiler_route(
        &self,
        global_state: &GlobalState,
        local_state: &mut LocalState,
        link_idxs: &[LinkIdx],
        difficulty: &DifficultyConfig,
    ) -> Vec<SpoilerRouteEntry> {
        let mut route: Vec<SpoilerRouteEntry> = Vec::new();
        // info!("global: {:?}", global_state);
        // global_state.print_debug(self.game_data);
        for &link_idx in link_idxs {
            let link = self.get_link(link_idx as usize);
            let raw_link = self.get_link(link_idx as usize);
            let sublinks = if raw_link.sublinks.len() > 0 {
                raw_link.sublinks.clone()
            } else {
                vec![raw_link.clone()]
            };

            let new_local_state_opt = apply_requirement(
                &link.requirement,
                &global_state,
                *local_state,
                false,
                difficulty,
                self.game_data,
            );
            let new_local_state = new_local_state_opt.unwrap();
            let new_resources =
                self.get_resources_remaining(&global_state, *local_state, new_local_state);
            for (i, link) in sublinks.iter().enumerate() {
                let last = i == sublinks.len() - 1;
                let from_vertex_info = self.get_vertex_info(link.from_vertex_id);
                let to_vertex_info = self.get_vertex_info(link.to_vertex_id);
                let (_, _, to_obstacles_mask) = self.game_data.vertex_isv.keys[link.to_vertex_id];
                // info!("local: {:?}", local_state);
                // info!("{:?}", link);

                let spoiler_entry = SpoilerRouteEntry {
                    area: to_vertex_info.area_name,
                    short_room: strip_name(&to_vertex_info.room_name),
                    room: to_vertex_info.room_name,
                    node: to_vertex_info.node_name,
                    from_node_id: from_vertex_info.node_id,
                    to_node_id: to_vertex_info.node_id,
                    obstacles_bitmask: to_obstacles_mask,
                    coords: to_vertex_info.room_coords,
                    strat_name: link.strat_name.clone(),
                    short_strat_name: strip_name(&link.strat_name),
                    strat_notes: link.strat_notes.clone(),
                    energy_remaining: if last { new_resources.energy } else { None },
                    reserves_remaining: if last { new_resources.reserves } else { None },
                    missiles_remaining: if last { new_resources.missiles } else { None },
                    supers_remaining: if last { new_resources.supers } else { None },
                    power_bombs_remaining: if last {
                        new_resources.power_bombs
                    } else {
                        None
                    },
                };
                // info!("spoiler: {:?}", spoiler_entry);
                route.push(spoiler_entry);
            }
            *local_state = new_local_state;
        }
        // info!("local: {:?}", local_state);
        route
    }

    fn get_spoiler_route_reverse(
        &self,
        global_state: &GlobalState,
        local_state_end_forward: LocalState,
        link_idxs: &[LinkIdx],
        difficulty: &DifficultyConfig,
    ) -> Vec<SpoilerRouteEntry> {
        let mut route: Vec<SpoilerRouteEntry> = Vec::new();
        let mut consumption_vec: Vec<LocalState> = Vec::new();

        let mut local_state = LocalState::new();
        for &link_idx in link_idxs {
            let link = self.get_link(link_idx as usize);
            let new_local_state = apply_requirement(
                &link.requirement,
                &global_state,
                local_state,
                true,
                difficulty,
                self.game_data,
            )
            .unwrap();
            // Here we make an assumption that any negative change in resource usage represents
            // a full refill; this is true for now though we may want to change this later, e.g.
            // if we want to represent non-farmable enemy drops.
            let energy_used = if new_local_state.energy_used < local_state.energy_used {
                -Capacity::MAX
            } else {
                new_local_state.energy_used - local_state.energy_used
            };
            let reserves_used = if new_local_state.reserves_used < local_state.reserves_used {
                -Capacity::MAX
            } else {
                new_local_state.reserves_used - local_state.reserves_used
            };
            let missiles_used = if new_local_state.missiles_used < local_state.missiles_used {
                -Capacity::MAX
            } else {
                new_local_state.missiles_used - local_state.missiles_used
            };
            let supers_used = if new_local_state.supers_used < local_state.supers_used {
                -Capacity::MAX
            } else {
                new_local_state.supers_used - local_state.supers_used
            };
            let power_bombs_used =
                if new_local_state.power_bombs_used < local_state.power_bombs_used {
                    -Capacity::MAX
                } else {
                    new_local_state.power_bombs_used - local_state.power_bombs_used
                };
            consumption_vec.push(LocalState {
                energy_used,
                reserves_used,
                missiles_used,
                supers_used,
                power_bombs_used,
            });
            local_state = new_local_state;
        }

        local_state = local_state_end_forward;
        for i in (0..link_idxs.len()).rev() {
            let link_idx = link_idxs[i];
            let raw_link = self.get_link(link_idx as usize);
            let sublinks = if raw_link.sublinks.len() > 0 {
                raw_link.sublinks.clone()
            } else {
                vec![raw_link.clone()]
            };
            let consumption = consumption_vec[i];
            let mut new_local_state = LocalState {
                energy_used: max(0, local_state.energy_used + consumption.energy_used),
                reserves_used: max(0, local_state.reserves_used + consumption.reserves_used),
                missiles_used: max(0, local_state.missiles_used + consumption.missiles_used),
                supers_used: max(0, local_state.supers_used + consumption.supers_used),
                power_bombs_used: max(
                    0,
                    local_state.power_bombs_used + consumption.power_bombs_used,
                ),
            };
            if new_local_state.energy_used >= global_state.max_energy {
                new_local_state.reserves_used +=
                    new_local_state.energy_used - (global_state.max_energy - 1);
                new_local_state.energy_used = global_state.max_energy - 1;
            }
            assert!(new_local_state.reserves_used <= global_state.max_reserves);
            assert!(new_local_state.missiles_used <= global_state.max_missiles);
            assert!(new_local_state.supers_used <= global_state.max_supers);
            assert!(new_local_state.power_bombs_used <= global_state.max_power_bombs);
            let new_resources =
                self.get_resources_remaining(&global_state, local_state, new_local_state);

            for (i, link) in sublinks.iter().enumerate() {
                let last = i == sublinks.len() - 1;
                let from_vertex_info = self.get_vertex_info(link.from_vertex_id);
                let to_vertex_info = self.get_vertex_info(link.to_vertex_id);
                let (_, _, to_obstacles_mask) = self.game_data.vertex_isv.keys[link.to_vertex_id];
                let spoiler_entry = SpoilerRouteEntry {
                    area: to_vertex_info.area_name,
                    short_room: strip_name(&to_vertex_info.room_name),
                    room: to_vertex_info.room_name,
                    from_node_id: from_vertex_info.node_id,
                    to_node_id: to_vertex_info.node_id,
                    node: to_vertex_info.node_name,
                    obstacles_bitmask: to_obstacles_mask,
                    coords: to_vertex_info.room_coords,
                    strat_name: link.strat_name.clone(),
                    short_strat_name: strip_name(&link.strat_name),
                    strat_notes: link.strat_notes.clone(),
                    energy_remaining: if last { new_resources.energy } else { None },
                    reserves_remaining: if last { new_resources.reserves } else { None },
                    missiles_remaining: if last { new_resources.missiles } else { None },
                    supers_remaining: if last { new_resources.supers } else { None },
                    power_bombs_remaining: if last {
                        new_resources.power_bombs
                    } else {
                        None
                    },
                };
                route.push(spoiler_entry);
                local_state = new_local_state;
            }
        }
        route
    }

    fn get_spoiler_route_birectional(
        &self,
        state: &RandomizationState,
        vertex_id: usize,
    ) -> (Vec<SpoilerRouteEntry>, Vec<SpoilerRouteEntry>) {
        // info!("vertex_id: {}", vertex_id);
        // info!("forward: {:?}", state.debug_data.as_ref().unwrap().forward.local_states[vertex_id]);
        // info!("reverse: {:?}", state.debug_data.as_ref().unwrap().reverse.local_states[vertex_id]);
        let forward = &state.debug_data.as_ref().unwrap().forward;
        let reverse = &state.debug_data.as_ref().unwrap().reverse;
        let (forward_cost_idx, reverse_cost_idx) =
            get_bireachable_idxs(&state.global_state, vertex_id, forward, reverse).unwrap();
        let forward_link_idxs: Vec<LinkIdx> =
            get_spoiler_route(forward, vertex_id, forward_cost_idx);
        let reverse_link_idxs: Vec<LinkIdx> =
            get_spoiler_route(reverse, vertex_id, reverse_cost_idx);
        let mut local_state = LocalState::new();
        // info!("obtain");
        let obtain_route = self.get_spoiler_route(
            &state.global_state,
            &mut local_state,
            &forward_link_idxs,
            &self.difficulty_tiers[0],
        );
        // info!("return");
        let return_route = self.get_spoiler_route_reverse(
            &state.global_state,
            local_state,
            &reverse_link_idxs,
            &self.difficulty_tiers[0],
        );
        (obtain_route, return_route)
    }

    fn get_spoiler_item_details(
        &self,
        state: &RandomizationState,
        item_vertex_id: usize,
        item: Item,
    ) -> SpoilerItemDetails {
        let (obtain_route, return_route) =
            self.get_spoiler_route_birectional(state, item_vertex_id);
        let item_vertex_info = self.get_vertex_info(item_vertex_id);
        SpoilerItemDetails {
            item: Item::VARIANTS[item as usize].to_string(),
            location: SpoilerLocation {
                area: item_vertex_info.area_name,
                room: item_vertex_info.room_name,
                node: item_vertex_info.node_name,
                coords: item_vertex_info.room_coords,
            },
            obtain_route: obtain_route,
            return_route: return_route,
        }
    }

    fn get_spoiler_item_summary(
        &self,
        _state: &RandomizationState,
        item_vertex_id: usize,
        item: Item,
    ) -> SpoilerItemSummary {
        let item_vertex_info = self.get_vertex_info(item_vertex_id);
        SpoilerItemSummary {
            item: Item::VARIANTS[item as usize].to_string(),
            location: SpoilerLocation {
                area: item_vertex_info.area_name,
                room: item_vertex_info.room_name,
                node: item_vertex_info.node_name,
                coords: item_vertex_info.room_coords,
            },
        }
    }

    fn get_spoiler_flag_details(
        &self,
        state: &RandomizationState,
        flag_vertex_id: usize,
        flag_id: FlagId,
    ) -> SpoilerFlagDetails {
        let (obtain_route, return_route) =
            self.get_spoiler_route_birectional(state, flag_vertex_id);
        let flag_vertex_info = self.get_vertex_info(flag_vertex_id);
        SpoilerFlagDetails {
            flag: self.game_data.flag_isv.keys[flag_id].to_string(),
            location: SpoilerLocation {
                area: flag_vertex_info.area_name,
                room: flag_vertex_info.room_name,
                node: flag_vertex_info.node_name,
                coords: flag_vertex_info.room_coords,
            },
            obtain_route: obtain_route,
            return_route: return_route,
        }
    }

    fn get_spoiler_flag_summary(
        &self,
        _state: &RandomizationState,
        _flag_vertex_id: usize,
        flag_id: FlagId,
    ) -> SpoilerFlagSummary {
        // let flag_vertex_info = self.get_vertex_info(flag_vertex_id);
        SpoilerFlagSummary {
            flag: self.game_data.flag_isv.keys[flag_id].to_string(),
        }
    }

    fn get_spoiler_details(
        &self,
        orig_global_state: &GlobalState, // Global state before acquiring new flags
        state: &RandomizationState,      // State after acquiring new flags but not new items
        new_state: &RandomizationState,  // State after acquiring new flags and new items
        spoiler_flag_details: Vec<SpoilerFlagDetails>,
    ) -> SpoilerDetails {
        let mut items: Vec<SpoilerItemDetails> = Vec::new();
        for i in 0..self.game_data.item_locations.len() {
            if let Some(item) = new_state.item_location_state[i].placed_item {
                if !state.item_location_state[i].collected
                    && new_state.item_location_state[i].collected
                {
                    // info!("Item: {item:?}");
                    let item_vertex_id =
                        state.item_location_state[i].bireachable_vertex_id.unwrap();
                    items.push(self.get_spoiler_item_details(state, item_vertex_id, item));
                }
            }
        }
        SpoilerDetails {
            step: state.step_num,
            start_state: self.get_spoiler_start_state(orig_global_state),
            items,
            flags: spoiler_flag_details,
        }
    }

    fn get_spoiler_summary(
        &self,
        _orig_global_state: &GlobalState, // Global state before acquiring new flags
        state: &RandomizationState,       // State after acquiring new flags but not new items
        new_state: &RandomizationState,   // State after acquiring new flags and new items
        spoiler_flag_summaries: Vec<SpoilerFlagSummary>,
    ) -> SpoilerSummary {
        let mut items: Vec<SpoilerItemSummary> = Vec::new();
        for i in 0..self.game_data.item_locations.len() {
            if let Some(item) = new_state.item_location_state[i].placed_item {
                if !state.item_location_state[i].collected
                    && new_state.item_location_state[i].collected
                {
                    let item_vertex_id =
                        state.item_location_state[i].bireachable_vertex_id.unwrap();
                    items.push(self.get_spoiler_item_summary(state, item_vertex_id, item));
                }
            }
        }
        SpoilerSummary {
            step: state.step_num,
            items,
            flags: spoiler_flag_summaries,
        }
    }
}
