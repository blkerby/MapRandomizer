pub mod escape_timer;
mod run_speed;

use crate::helpers::get_item_priorities;
use crate::patch::NUM_AREAS;
use crate::patch::map_tiles::get_objective_tiles;
use crate::settings::{
    DoorsMode, FillerItemPriority, ItemPlacementStyle, ItemPriorityStrength, KeyItemPriority,
    MotherBrainFight, Objective, ObjectiveSetting, ProgressionRate, RandomizerSettings,
    SaveAnimals, SkillAssumptionSettings, StartLocationMode, WallJump, SpikeSuits,
};
use crate::spoiler_log::{
    SpoilerLocalState, SpoilerLog, SpoilerRoomLoc, SpoilerRouteEntry, SpoilerStartLocation,
    SpoilerTraversal, get_spoiler_game_data, get_spoiler_log, get_spoiler_route,
};
use crate::traverse::{
    LocalStateReducer, LockedDoorData, TraversalUpdate, Traverser, apply_requirement,
    get_bireachable_idxs, get_spoiler_trail_ids_by_idx, simple_cost_config,
};
use anyhow::{Context, Result, bail};
use hashbrown::{HashMap, HashSet};
use log::{debug, info};
use maprando_game::{
    self, AreaIdx, BeamType, BlueOption, BounceMovementType, Capacity, DoorOrientation,
    DoorPtrPair, DoorType, EntranceCondition, ExitCondition, Float, GModeMobility, GModeMode,
    GameData, GrappleJumpPosition, GrappleSwingBlock, HubLocation, Item, ItemId, ItemLocationId,
    Link, LinksDataGroup, MainEntranceCondition, Map, NodeId, NotableId, Physics, Requirement,
    RoomGeometryRoomIdx, RoomId, SidePlatformEntrance, SidePlatformEnvironment, SparkPosition,
    StartLocation, TECH_ID_CAN_ARTIFICIAL_MORPH, TECH_ID_CAN_CARRY_FLASH_SUIT,
    TECH_ID_CAN_DISABLE_EQUIPMENT, TECH_ID_CAN_ENTER_G_MODE, TECH_ID_CAN_ENTER_G_MODE_IMMOBILE,
    TECH_ID_CAN_ENTER_R_MODE, TECH_ID_CAN_GRAPPLE_JUMP, TECH_ID_CAN_GRAPPLE_TELEPORT,
    TECH_ID_CAN_HEATED_G_MODE, TECH_ID_CAN_HORIZONTAL_SHINESPARK, TECH_ID_CAN_MIDAIR_SHINESPARK,
    TECH_ID_CAN_MOCKBALL, TECH_ID_CAN_MOONFALL, TECH_ID_CAN_PRECISE_GRAPPLE,
    TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK, TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK_FROM_WATER,
    TECH_ID_CAN_SAMUS_EATER_TELEPORT, TECH_ID_CAN_SHINECHARGE_MOVEMENT,
    TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP, TECH_ID_CAN_SPEEDBALL,
    TECH_ID_CAN_SPRING_BALL_BOUNCE, TECH_ID_CAN_STATIONARY_SPIN_JUMP,
    TECH_ID_CAN_STUTTER_WATER_SHINECHARGE, TECH_ID_CAN_SUPER_SINK, TECH_ID_CAN_TEMPORARY_BLUE,
    TECH_ID_CAN_TRICKY_CARRY_FLASH_SUIT, TECH_ID_CAN_SPIKE_SUIT, TECH_ID_CAN_SLOPE_SPARK, 
    TECH_ID_CAN_RMODE_KNOCKBACK_SPARK, TechId, TemporaryBlueDirection, TraversalId, VertexId, VertexKey,
};
use maprando_logic::{GlobalState, Inventory, LocalState};
use rand::SeedableRng;
use rand::{Rng, seq::SliceRandom};
use run_speed::{
    get_extra_run_speed_tiles, get_max_extra_run_speed, get_shortcharge_max_extra_run_speed,
    get_shortcharge_min_extra_run_speed,
};
use serde_derive::{Deserialize, Serialize};
use std::cell::RefCell;
use std::{cmp::min, convert::TryFrom, hash::Hash, iter, time::SystemTime};
use strum::VariantNames;

// Once there are fewer than 20 item locations remaining to be filled, key items will be
// placed as quickly as possible. This helps prevent generation failures particularly on lower
// difficulty settings where some item locations may never be accessible (e.g. Main Street Missile).
const KEY_ITEM_FINISH_THRESHOLD: usize = 20;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ItemPriorityGroup {
    pub priority: KeyItemPriority,
    pub items: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct DifficultyConfig {
    pub name: String,
    pub tech: Vec<bool>,
    pub notables: Vec<bool>,
    pub shine_charge_tiles: f32,
    pub heated_shine_charge_tiles: f32,
    pub shinecharge_leniency_frames: Capacity,
    pub speed_ball_tiles: f32,
    pub resource_multiplier: f32,
    pub farm_time_limit: f32,
    pub gate_glitch_leniency: Capacity,
    pub door_stuck_leniency: Capacity,
    pub elevator_cf_leniency: Capacity,
    pub bomb_into_cf_leniency: Capacity,
    pub jump_into_cf_leniency: Capacity,
    pub spike_xmode_leniency: Capacity,
    pub spike_speed_keep_leniency: Capacity,
    pub escape_timer_multiplier: f32,
    pub phantoon_proficiency: f32,
    pub draygon_proficiency: f32,
    pub ridley_proficiency: f32,
    pub botwoon_proficiency: f32,
    pub mother_brain_proficiency: f32,
}

impl DifficultyConfig {
    pub fn new(
        skill: &SkillAssumptionSettings,
        game_data: &GameData,
        implicit_tech: &[TechId],
        implicit_notables: &[(RoomId, NotableId)],
    ) -> Self {
        let mut tech: Vec<bool> = vec![false; game_data.tech_isv.keys.len()];
        let mut notables: Vec<bool> = vec![false; game_data.notable_isv.keys.len()];
        
        for &tech_id in implicit_tech {
            let tech_idx = game_data.tech_isv.index_by_key[&tech_id];
            tech[tech_idx] = true;
        }
        for tech_setting in &skill.tech_settings {
            if tech_setting.enabled {
                let tech_idx = game_data.tech_isv.index_by_key[&tech_setting.id];
                tech[tech_idx] = true;
            }
        }

        for &(room_id, notable_id) in implicit_notables {
            let notable_idx = game_data.notable_isv.index_by_key[&(room_id, notable_id)];
            notables[notable_idx] = true;
        }
        for notable_setting in &skill.notable_settings {
            if notable_setting.enabled {
                let notable_idx = game_data.notable_isv.index_by_key
                    [&(notable_setting.room_id, notable_setting.notable_id)];
                notables[notable_idx] = true;
            }
        }

        Self {
            name: skill.preset.clone().unwrap_or("Beyond".to_string()),
            tech,
            notables,
            shine_charge_tiles: skill.shinespark_tiles,
            heated_shine_charge_tiles: skill.heated_shinespark_tiles,
            shinecharge_leniency_frames: skill.shinecharge_leniency_frames as Capacity,
            speed_ball_tiles: skill.speed_ball_tiles,
            resource_multiplier: skill.resource_multiplier,
            farm_time_limit: skill.farm_time_limit,
            gate_glitch_leniency: skill.gate_glitch_leniency as Capacity,
            bomb_into_cf_leniency: skill.bomb_into_cf_leniency as Capacity,
            jump_into_cf_leniency: skill.jump_into_cf_leniency as Capacity,
            spike_xmode_leniency: skill.spike_xmode_leniency as Capacity,
            spike_speed_keep_leniency: skill.spike_speed_keep_leniency as Capacity,
            elevator_cf_leniency: skill.elevator_cf_leniency as Capacity,
            door_stuck_leniency: skill.door_stuck_leniency as Capacity,
            escape_timer_multiplier: skill.escape_timer_multiplier,
            phantoon_proficiency: skill.phantoon_proficiency,
            draygon_proficiency: skill.draygon_proficiency,
            ridley_proficiency: skill.ridley_proficiency,
            botwoon_proficiency: skill.botwoon_proficiency,
            mother_brain_proficiency: skill.mother_brain_proficiency,
        }
    }

    pub fn intersect(&self, other: &DifficultyConfig) -> DifficultyConfig {
        let tech: Vec<bool> = self
            .tech
            .iter()
            .zip(other.tech.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        let notables: Vec<bool> = self
            .notables
            .iter()
            .zip(other.notables.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        DifficultyConfig {
            name: self.name.clone(),
            tech,
            notables,
            shine_charge_tiles: f32::max(self.shine_charge_tiles, other.shine_charge_tiles),
            heated_shine_charge_tiles: f32::max(
                self.heated_shine_charge_tiles,
                other.heated_shine_charge_tiles,
            ),
            speed_ball_tiles: f32::max(self.speed_ball_tiles, other.speed_ball_tiles),
            shinecharge_leniency_frames: Capacity::max(
                self.shinecharge_leniency_frames,
                other.shinecharge_leniency_frames,
            ),
            resource_multiplier: f32::max(self.resource_multiplier, other.resource_multiplier),
            farm_time_limit: f32::min(self.farm_time_limit, other.farm_time_limit),
            gate_glitch_leniency: Capacity::max(
                self.gate_glitch_leniency,
                other.gate_glitch_leniency as Capacity,
            ),
            door_stuck_leniency: Capacity::max(
                self.door_stuck_leniency,
                other.door_stuck_leniency as Capacity,
            ),
            elevator_cf_leniency: Capacity::max(
                self.elevator_cf_leniency,
                other.elevator_cf_leniency as Capacity,
            ),
            bomb_into_cf_leniency: Capacity::max(
                self.bomb_into_cf_leniency,
                other.bomb_into_cf_leniency as Capacity,
            ),
            jump_into_cf_leniency: Capacity::max(
                self.jump_into_cf_leniency,
                other.jump_into_cf_leniency as Capacity,
            ),
            spike_xmode_leniency: Capacity::max(
                self.spike_xmode_leniency,
                other.spike_xmode_leniency as Capacity,
            ),
            spike_speed_keep_leniency: Capacity::max(
                self.spike_speed_keep_leniency,
                other.spike_speed_keep_leniency as Capacity,
            ),
            escape_timer_multiplier: f32::max(
                self.escape_timer_multiplier,
                other.escape_timer_multiplier,
            ),
            phantoon_proficiency: f32::min(self.phantoon_proficiency, other.phantoon_proficiency),
            draygon_proficiency: f32::min(self.draygon_proficiency, other.draygon_proficiency),
            ridley_proficiency: f32::min(self.ridley_proficiency, other.ridley_proficiency),
            botwoon_proficiency: f32::min(self.botwoon_proficiency, other.botwoon_proficiency),
            mother_brain_proficiency: f32::min(
                self.mother_brain_proficiency,
                other.mother_brain_proficiency,
            ),
        }
    }
}

// Includes preprocessing specific to the map:
pub struct Randomizer<'a> {
    pub map: &'a Map,
    pub door_map: HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
    pub item_areas: Vec<AreaIdx>, // assigned area of each item, in order by game_data.item_locations
    pub toilet_intersections: Vec<RoomGeometryRoomIdx>,
    pub locked_door_data: &'a LockedDoorData,
    pub game_data: &'a GameData,
    pub settings: &'a RandomizerSettings,
    pub objectives: Vec<Objective>,
    pub filler_priority_map: HashMap<Item, FillerItemPriority>,
    pub item_priority_groups: Vec<ItemPriorityGroup>,
    pub difficulty_tiers: &'a [DifficultyConfig],
    pub base_links_data: &'a LinksDataGroup,
    pub seed_links_data: LinksDataGroup,
    pub initial_items_remaining: Vec<usize>, // Corresponds to GameData.items_isv (one count per distinct item name)
    pub next_traversal_number: RefCell<usize>,
}

#[derive(Clone)]
pub struct ItemLocationState {
    pub placed_item: Option<Item>,
    pub placed_tier: Option<usize>,
    pub collected: bool,
    pub reachable_traversal: Option<TraversalId>,
    pub bireachable_traversal: Option<TraversalId>,
    pub bireachable_vertex_id: Option<VertexId>,
    pub difficulty_tier: Option<usize>,
}

#[derive(Clone)]
pub struct FlagLocationState {
    pub reachable_traversal: Option<TraversalId>,
    pub reachable_vertex_id: Option<VertexId>,
    pub bireachable_traversal: Option<TraversalId>,
    pub bireachable_vertex_id: Option<VertexId>,
}

#[derive(Clone)]
pub struct DoorState {
    pub bireachable_traversal: Option<TraversalId>,
    pub bireachable_vertex_id: Option<VertexId>,
}

#[derive(Clone)]
pub struct SaveLocationState {
    pub bireachable_traversal: Option<TraversalId>,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct LockedDoor {
    pub src_ptr_pair: DoorPtrPair,
    pub dst_ptr_pair: DoorPtrPair,
    pub door_type: DoorType,
    pub bidirectional: bool, // if true, the door is locked on both sides, with a shared state
}

// The core graph traversal state, which is large so we want to avoid cloning it where possible.
#[derive(Clone)]
pub struct TraverserPair {
    pub forward: Traverser,
    pub reverse: Traverser,
}

// Other randomization state that changes during or across item placement attempts,
// small enough that cloning it is fine.
#[derive(Clone)]
pub struct RandomizationState {
    pub step_num: usize,
    pub start_location: StartLocation,
    pub hub_location: HubLocation,
    pub item_precedence: Vec<Item>, // An ordering of the 21 distinct item names. The game will prioritize placing key items earlier in the list.
    pub save_location_state: Vec<SaveLocationState>,
    pub item_location_state: Vec<ItemLocationState>, // Corresponds to GameData.item_locations (one record for each of 100 item locations)
    pub flag_location_state: Vec<FlagLocationState>, // Corresponds to GameData.flag_locations
    pub door_state: Vec<DoorState>,                  // Corresponds to LockedDoorData.locked_doors
    pub items_remaining: Vec<usize>, // Corresponds to GameData.items_isv (one count for each of 21 distinct item names)
    pub global_state: GlobalState,
    pub starting_local_state: LocalState, // Initial local state at the hub location
    pub last_key_areas: Vec<AreaIdx>,
}

// Info about an item used during ROM patching, to show info in the credits
#[derive(Serialize, Deserialize)]
pub struct EssentialItemSpoilerInfo {
    pub item: Item,
    pub step: Option<usize>,
    pub area: Option<String>,
}
// Spoiler data that is used during ROM patching (e.g. to show info in the credits)
#[derive(Serialize, Deserialize)]
pub struct EssentialSpoilerData {
    pub item_spoiler_info: Vec<EssentialItemSpoilerInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct Randomization {
    pub objectives: Vec<Objective>,
    pub save_animals: SaveAnimals,
    pub map: Map,
    pub toilet_intersections: Vec<RoomGeometryRoomIdx>,
    pub locked_doors: Vec<LockedDoor>,
    pub item_placement: Vec<Item>,
    pub start_location: StartLocation,
    pub escape_time_seconds: f32,
    pub essential_spoiler_data: EssentialSpoilerData,
    pub seed: usize,
    pub display_seed: usize,
    pub seed_name: String,
}

struct SelectItemsOutput {
    key_items: Vec<Item>,
    other_items: Vec<Item>,
}

pub struct StartLocationData {
    pub start_location: StartLocation,
    pub hub_location: HubLocation,
    pub hub_obtain_route: Vec<SpoilerRouteEntry>,
    pub hub_return_route: Vec<SpoilerRouteEntry>,
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

    let mut subsubarea_mapping: Vec<Vec<Vec<usize>>> = vec![vec![(0..2).collect(); 2]; 6];
    for i in 0..6 {
        for j in 0..2 {
            subsubarea_mapping[i][j].shuffle(&mut rng);
        }
    }

    for i in 0..map.area.len() {
        map.area[i] = area_mapping[map.area[i]];
        map.subarea[i] = subarea_mapping[map.area[i]][map.subarea[i]];
        map.subsubarea[i] = subsubarea_mapping[map.area[i]][map.subarea[i]][map.subsubarea[i]];
    }
}

pub fn order_map_areas(map: &mut Map, seed: usize, game_data: &GameData) {
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let mut area_tile_cnt: [isize; NUM_AREAS] = [0; NUM_AREAS];
    for (room_idx, area) in map.area.iter().copied().enumerate() {
        for row in &game_data.room_geometry[room_idx].map {
            for cell in row {
                if *cell == 1 {
                    area_tile_cnt[area] += 1;
                }
            }
        }
    }
    let mut area_rank: Vec<AreaIdx> = (0..NUM_AREAS).collect();
    area_rank.sort_by_key(|&i| area_tile_cnt[i]);

    let mut area_mapping: Vec<usize> = vec![0; NUM_AREAS];
    area_mapping[area_rank[5]] = 2; // Norfair (largest area)
    area_mapping[area_rank[4]] = 1; // Brinstar
    area_mapping[area_rank[3]] = 4; // Maridia
    area_mapping[area_rank[2]] = 0; // Crateria
    area_mapping[area_rank[1]] = 5; // Tourian
    area_mapping[area_rank[0]] = 3; // Wrecked Ship

    let mut subarea_mapping: Vec<Vec<usize>> = vec![(0..2).collect(); 6];
    for i in 0..6 {
        subarea_mapping[i].shuffle(&mut rng);
    }

    let mut subsubarea_mapping: Vec<Vec<Vec<usize>>> = vec![vec![(0..2).collect(); 2]; 6];
    for i in 0..6 {
        for j in 0..2 {
            subsubarea_mapping[i][j].shuffle(&mut rng);
        }
    }

    for i in 0..map.area.len() {
        map.area[i] = area_mapping[map.area[i]];
        map.subarea[i] = subarea_mapping[map.area[i]][map.subarea[i]];
        map.subsubarea[i] = subsubarea_mapping[map.area[i]][map.subarea[i]][map.subsubarea[i]];
    }
}

fn compute_run_frames(tiles: f32) -> Capacity {
    assert!(tiles >= 0.0);
    let frames = if tiles <= 7.0 {
        9.0 + 4.0 * tiles
    } else if tiles <= 16.0 {
        15.0 + 3.0 * tiles
    } else if tiles <= 42.0 {
        32.0 + 2.0 * tiles
    } else {
        47.0 + 64.0 / 39.0 * tiles
    };
    frames.ceil() as Capacity
}

fn remove_some_duplicates<T: Clone + PartialEq + Eq + Hash>(
    x: &[T],
    dup_set: &HashSet<T>,
) -> Vec<T> {
    let mut out: Vec<T> = vec![];
    let mut seen_set: HashSet<T> = HashSet::new();
    for e in x {
        if seen_set.contains(e) {
            continue;
        }
        if dup_set.contains(e) {
            seen_set.insert(e.clone());
        }
        out.push(e.clone());
    }
    out
}

pub struct Preprocessor<'a> {
    pub game_data: &'a GameData,
    pub door_map: HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
    pub difficulty: &'a DifficultyConfig,
}

fn compute_shinecharge_frames(
    other_runway_length: f32,
    runway_length: f32,
) -> (Capacity, Capacity) {
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
    let other_time = other_time.ceil() as Capacity;
    (other_time, total_time as Capacity - other_time)
}

impl<'a> Preprocessor<'a> {
    pub fn new(game_data: &'a GameData, map: &'a Map, difficulty: &'a DifficultyConfig) -> Self {
        let mut door_map: HashMap<(RoomId, NodeId), (RoomId, NodeId)> = HashMap::new();
        for &((src_exit_ptr, src_entrance_ptr), (dst_exit_ptr, dst_entrance_ptr), _bidirectional) in
            &map.doors
        {
            let (src_room_id, src_node_id) =
                game_data.door_ptr_pair_map[&(src_exit_ptr, src_entrance_ptr)];
            let (dst_room_id, dst_node_id) =
                game_data.door_ptr_pair_map[&(dst_exit_ptr, dst_entrance_ptr)];
            door_map.insert((src_room_id, src_node_id), (dst_room_id, dst_node_id));
            door_map.insert((dst_room_id, dst_node_id), (src_room_id, src_node_id));

            if (dst_room_id, dst_node_id) == (32, 1) {
                // West Ocean bottom left door, West Ocean Bridge left door
                door_map.insert((32, 7), (src_room_id, src_node_id));
            }
            if (src_room_id, src_node_id) == (32, 5) {
                // West Ocean bottom right door, West Ocean Bridge right door
                door_map.insert((32, 8), (dst_room_id, dst_node_id));
            }
        }
        Preprocessor {
            game_data,
            door_map,
            difficulty,
        }
    }

    fn add_door_links(
        &self,
        src_room_id: usize,
        src_node_id: usize,
        dst_room_id: usize,
        dst_node_id: usize,
        is_toilet: bool,
        door_links: &mut Vec<Link>,
    ) {
        let empty_vec_exits = vec![];
        let empty_vec_entrances = vec![];
        for exit_info in self
            .game_data
            .node_exit_conditions
            .get(&(src_room_id, src_node_id))
            .unwrap_or(&empty_vec_exits)
        {
            let src_vertex_id = exit_info.vertex_id;
            let exit_condition = &exit_info.exit_condition;
            let exit_req = &exit_info.exit_req;
            for &(dst_vertex_id, ref entrance_condition) in self
                .game_data
                .node_entrance_conditions
                .get(&(dst_room_id, dst_node_id))
                .unwrap_or(&empty_vec_entrances)
            {
                if entrance_condition.through_toilet == maprando_game::ToiletCondition::Yes
                    && !is_toilet
                {
                    // The strat requires passing through the Toilet, which is not the case here.
                    continue;
                } else if entrance_condition.through_toilet == maprando_game::ToiletCondition::No
                    && is_toilet
                {
                    // The strat requires not passing through the Toilet, but here it does.
                    continue;
                }
                let mut req_opt = self.get_cross_room_reqs(
                    exit_condition,
                    src_room_id,
                    src_node_id,
                    entrance_condition,
                    dst_room_id,
                    dst_node_id,
                    is_toilet,
                );
                req_opt = req_opt
                    .map(|req| Requirement::make_and(vec![req, Requirement::DoorTransition]));
                let exit_with_shinecharge = self.game_data.does_leave_shinecharged(exit_condition);
                let enter_with_shinecharge =
                    self.game_data.does_come_in_shinecharged(entrance_condition);

                if let Some(req) = req_opt {
                    // if (src_room_id, src_node_id) == (44, 4) {
                    //     println!(
                    //         "{} ({}, {}, {:?}) -> {} ({}, {}, {:?}): {:?}",
                    //         src_vertex_id,
                    //         src_room_id,
                    //         src_node_id,
                    //         exit_condition,
                    //         dst_vertex_id,
                    //         dst_room_id,
                    //         dst_node_id,
                    //         entrance_condition,
                    //         req
                    //     );
                    // }
                    door_links.push(Link {
                        from_vertex_id: src_vertex_id,
                        to_vertex_id: dst_vertex_id,
                        requirement: Requirement::make_and(vec![req, exit_req.clone()]),
                        start_with_shinecharge: exit_with_shinecharge,
                        end_with_shinecharge: enter_with_shinecharge,
                        difficulty: 0,
                        length: 1,
                        strat_id: None,
                        strat_name: "Base (Cross Room)".to_string(),
                        strat_notes: vec![],
                    });
                }
            }
        }
    }

    pub fn get_all_door_links(&self) -> Vec<Link> {
        let mut door_links = vec![];
        for (&(src_room_id, src_node_id), &(dst_room_id, dst_node_id)) in self.door_map.iter() {
            self.add_door_links(
                src_room_id,
                src_node_id,
                dst_room_id,
                dst_node_id,
                src_room_id == 321,
                &mut door_links,
            );
            if src_room_id == 321 {
                // Create links that skip over the Toilet:
                let src_node_id = if src_node_id == 1 { 2 } else { 1 };
                let (src_room_id, src_node_id) = *self.door_map.get(&(321, src_node_id)).unwrap();
                self.add_door_links(
                    src_room_id,
                    src_node_id,
                    dst_room_id,
                    dst_node_id,
                    true,
                    &mut door_links,
                );
            }
        }
        let extra_door_links: Vec<((usize, usize), (usize, usize))> = vec![
            ((220, 2), (322, 2)), // East Pants Room right door, Pants Room right door
            ((32, 7), (32, 1)),   // West Ocean bottom left door, West Ocean Bridge left door
            ((32, 8), (32, 5)),   // West Ocean bottom right door, West Ocean Bridge right door
        ];
        for ((src_room_id, src_node_id), (dst_other_room_id, dst_other_node_id)) in extra_door_links
        {
            if let Some(&(dst_room_id, dst_node_id)) =
                self.door_map.get(&(dst_other_room_id, dst_other_node_id))
            {
                self.add_door_links(
                    src_room_id,
                    src_node_id,
                    dst_room_id,
                    dst_node_id,
                    false,
                    &mut door_links,
                )
            }
        }

        // for link in &door_links {
        //     let from_vertex_id = link.from_vertex_id;
        //     let from_vertex_key = &self.game_data.vertex_isv.keys[from_vertex_id];
        //     let to_vertex_id = link.to_vertex_id;
        //     let to_vertex_key = &self.game_data.vertex_isv.keys[to_vertex_id];
        //     if (to_vertex_key.room_id, to_vertex_key.node_id) == (66, 3) && from_vertex_key.room_id != 66 {
        //         println!("From: {:?}\nTo: {:?}\nLink: {:?}\n", from_vertex_key, to_vertex_key, link);
        //     }
        // }

        door_links
    }

    fn get_cross_room_reqs(
        &self,
        exit_condition: &ExitCondition,
        _exit_room_id: RoomId,
        _exit_node_id: NodeId,
        entrance_condition: &EntranceCondition,
        entrance_room_id: RoomId,
        entrance_node_id: NodeId,
        is_toilet: bool,
    ) -> Option<Requirement> {
        match &entrance_condition.main {
            MainEntranceCondition::ComeInNormally {} => {
                self.get_come_in_normally_reqs(exit_condition)
            }
            MainEntranceCondition::ComeInRunning {
                speed_booster,
                min_tiles,
                max_tiles,
            } => self.get_come_in_running_reqs(
                exit_condition,
                *speed_booster,
                min_tiles.get(),
                max_tiles.get(),
            ),
            MainEntranceCondition::ComeInJumping {
                speed_booster,
                min_tiles,
                max_tiles,
            } => self.get_come_in_running_reqs(
                exit_condition,
                *speed_booster,
                min_tiles.get(),
                max_tiles.get(),
            ),
            MainEntranceCondition::ComeInSpaceJumping {
                speed_booster,
                min_tiles,
                max_tiles,
            } => self.get_come_in_space_jumping_reqs(
                exit_condition,
                *speed_booster,
                min_tiles.get(),
                max_tiles.get(),
            ),
            MainEntranceCondition::ComeInShinecharging {
                effective_length,
                min_tiles,
                heated,
            } => self.get_come_in_shinecharging_reqs(
                exit_condition,
                effective_length.get(),
                min_tiles.get(),
                *heated,
            ),
            MainEntranceCondition::ComeInGettingBlueSpeed {
                effective_length,
                min_tiles,
                heated,
                min_extra_run_speed,
                max_extra_run_speed,
            } => self.get_come_in_getting_blue_speed_reqs(
                exit_condition,
                effective_length.get(),
                min_tiles.get(),
                *heated,
                min_extra_run_speed.get(),
                max_extra_run_speed.get(),
            ),
            MainEntranceCondition::ComeInShinecharged {} => {
                self.get_come_in_shinecharged_reqs(exit_condition)
            }
            MainEntranceCondition::ComeInShinechargedJumping {} => {
                self.get_come_in_shinecharged_jumping_reqs(exit_condition)
            }
            MainEntranceCondition::ComeInWithSpark {
                position,
                door_orientation,
            } => self.get_come_in_with_spark_reqs(exit_condition, *position, *door_orientation),
            MainEntranceCondition::ComeInStutterShinecharging { min_tiles } => {
                self.get_come_in_stutter_shinecharging_reqs(exit_condition, min_tiles.get(), true)
            }
            MainEntranceCondition::ComeInStutterGettingBlueSpeed { min_tiles } => {
                self.get_come_in_stutter_shinecharging_reqs(exit_condition, min_tiles.get(), false)
            }
            MainEntranceCondition::ComeInWithBombBoost {} => {
                self.get_come_in_with_bomb_boost_reqs(exit_condition)
            }
            MainEntranceCondition::ComeInWithDoorStuckSetup {
                heated,
                door_orientation,
            } => self.get_come_in_with_door_stuck_setup_reqs(
                exit_condition,
                *heated,
                *door_orientation,
            ),
            MainEntranceCondition::ComeInSpeedballing {
                effective_runway_length,
                min_extra_run_speed,
                max_extra_run_speed,
                heated,
            } => self.get_come_in_speedballing_reqs(
                exit_condition,
                effective_runway_length.get(),
                min_extra_run_speed.get(),
                max_extra_run_speed.get(),
                *heated,
            ),
            MainEntranceCondition::ComeInWithTemporaryBlue { direction } => {
                self.get_come_in_with_temporary_blue_reqs(exit_condition, *direction)
            }
            MainEntranceCondition::ComeInSpinning {
                unusable_tiles,
                min_extra_run_speed,
                max_extra_run_speed,
            } => self.get_come_in_spinning_reqs(
                exit_condition,
                unusable_tiles.get(),
                min_extra_run_speed.get(),
                max_extra_run_speed.get(),
            ),
            MainEntranceCondition::ComeInBlueSpinning {
                unusable_tiles,
                min_extra_run_speed,
                max_extra_run_speed,
            } => self.get_come_in_blue_spinning_reqs(
                exit_condition,
                unusable_tiles.get(),
                min_extra_run_speed.get(),
                max_extra_run_speed.get(),
            ),
            MainEntranceCondition::ComeInBlueSpaceJumping {
                min_extra_run_speed,
                max_extra_run_speed,
            } => self.get_come_in_blue_space_jumping_reqs(
                exit_condition,
                min_extra_run_speed.get(),
                max_extra_run_speed.get(),
            ),
            MainEntranceCondition::ComeInWithMockball {
                speed_booster,
                adjacent_min_tiles,
                remote_and_landing_min_tiles,
            } => self.get_come_in_with_mockball_reqs(
                exit_condition,
                *speed_booster,
                adjacent_min_tiles.get(),
                remote_and_landing_min_tiles
                    .iter()
                    .map(|(a, b)| (a.get(), b.get()))
                    .collect(),
            ),
            MainEntranceCondition::ComeInWithSpringBallBounce {
                speed_booster,
                adjacent_min_tiles,
                remote_and_landing_min_tiles,
                movement_type,
            } => self.get_come_in_with_spring_ball_bounce_reqs(
                exit_condition,
                *speed_booster,
                adjacent_min_tiles.get(),
                remote_and_landing_min_tiles
                    .iter()
                    .map(|(a, b)| (a.get(), b.get()))
                    .collect(),
                *movement_type,
            ),
            MainEntranceCondition::ComeInWithBlueSpringBallBounce {
                min_extra_run_speed,
                max_extra_run_speed,
                min_landing_tiles,
                movement_type,
            } => self.get_come_in_with_blue_spring_ball_bounce_reqs(
                exit_condition,
                min_extra_run_speed.get(),
                max_extra_run_speed.get(),
                min_landing_tiles.get(),
                *movement_type,
            ),
            MainEntranceCondition::ComeInWithRMode { heated } => {
                self.get_come_in_with_r_mode_reqs(exit_condition, *heated)
            }
            MainEntranceCondition::ComeInWithGMode {
                mode,
                morphed,
                mobility,
                heated,
            } => self.get_come_in_with_g_mode_reqs(
                exit_condition,
                entrance_room_id,
                entrance_node_id,
                *mode,
                *morphed,
                *mobility,
                *heated,
                is_toilet,
            ),
            MainEntranceCondition::ComeInWithStoredFallSpeed {
                fall_speed_in_tiles,
            } => self.get_come_in_with_stored_fall_speed_reqs(exit_condition, *fall_speed_in_tiles),
            MainEntranceCondition::ComeInWithWallJumpBelow { min_height } => {
                self.get_come_in_with_wall_jump_below_reqs(exit_condition, min_height.get())
            }
            MainEntranceCondition::ComeInWithSpaceJumpBelow {} => {
                self.get_come_in_with_space_jump_below_reqs(exit_condition)
            }
            MainEntranceCondition::ComeInWithPlatformBelow {
                min_height,
                max_height,
                max_left_position,
                min_right_position,
            } => self.get_come_in_with_platform_below_reqs(
                exit_condition,
                min_height.get(),
                max_height.get(),
                max_left_position.get(),
                min_right_position.get(),
            ),
            MainEntranceCondition::ComeInWithSidePlatform { platforms } => {
                self.get_come_in_with_side_platform_reqs(exit_condition, platforms)
            }
            MainEntranceCondition::ComeInWithGrappleSwing { blocks } => {
                self.get_come_in_with_grapple_swing_reqs(exit_condition, blocks)
            }
            MainEntranceCondition::ComeInWithGrappleJump { position } => {
                self.get_come_in_with_grapple_jump_reqs(exit_condition, *position)
            }
            MainEntranceCondition::ComeInWithGrappleTeleport { block_positions } => {
                self.get_come_in_with_grapple_teleport_reqs(exit_condition, block_positions)
            }
            MainEntranceCondition::ComeInWithSamusEaterTeleport {
                floor_positions,
                ceiling_positions,
            } => self.get_come_in_with_samus_eater_teleport_reqs(
                exit_condition,
                floor_positions,
                ceiling_positions,
            ),
            MainEntranceCondition::ComeInWithSuperSink {} => {
                self.get_come_in_with_super_sink_reqs(exit_condition)
            }
        }
    }

    fn get_come_in_normally_reqs(&self, exit_condition: &ExitCondition) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveNormally {} => Some(Requirement::Free),
            _ => None,
        }
    }

    fn get_come_in_running_reqs(
        &self,
        exit_condition: &ExitCondition,
        speed_booster: Option<bool>,
        min_tiles: f32,
        max_tiles: f32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                if effective_length < min_tiles {
                    return None;
                }
                if get_extra_run_speed_tiles(min_extra_run_speed.get()) > max_tiles {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![];
                if speed_booster == Some(true) {
                    reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
                }
                if speed_booster == Some(false) {
                    reqs.push(Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_DISABLE_EQUIPMENT],
                    ));
                }
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                    // TODO: in sm-json-data, add physics property to leaveWithRunway schema (for door nodes with multiple possible physics)
                }
                if *heated {
                    let heat_frames = if *from_exit_node {
                        compute_run_frames(min_tiles) * 2 + 20
                    } else if effective_length > max_tiles {
                        // 10 heat frames to position after stopping on a dime, before resuming running
                        compute_run_frames(effective_length - max_tiles)
                            + compute_run_frames(max_tiles)
                            + 10
                    } else {
                        compute_run_frames(effective_length)
                    };
                    reqs.push(Requirement::HeatFrames(heat_frames as Capacity));
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_space_jumping_reqs(
        &self,
        exit_condition: &ExitCondition,
        speed_booster: Option<bool>,
        min_tiles: f32,
        _max_tiles: f32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveSpaceJumping {
                remote_runway_length,
                blue,
                ..
            } => {
                // TODO: Take into account any exit constraints on min_extra_run_speed and max_extra_run_speed.
                // Currently there might not be any scenarios where this matters, but that could change?
                // It is awkward because for a non-blue entrance strat like this, the constraints are measured in tiles rather
                // than run speed, though we could convert between the two.
                let remote_runway_length = remote_runway_length.get();
                if *blue == BlueOption::Yes {
                    return None;
                }
                if remote_runway_length < min_tiles {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![Requirement::Item(Item::SpaceJump as ItemId)];
                if speed_booster == Some(true) {
                    reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
                }
                if speed_booster == Some(false) {
                    reqs.push(Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_DISABLE_EQUIPMENT],
                    ));
                }
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key
                        [&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                ));
                reqs.push(Requirement::Or(vec![
                    Requirement::NoFlashSuit,
                    Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_TRICKY_CARRY_FLASH_SUIT],
                    ),
                ]));
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_cross_room_shortcharge_heat_frames(
        &self,
        from_exit_node: bool,
        entrance_length: f32,
        exit_length: f32,
        entrance_heated: bool,
        exit_heated: bool,
    ) -> Capacity {
        let mut total_heat_frames = 0;
        if from_exit_node {
            // Runway in the exiting room starts and ends at the door so we need to run both directions:
            if entrance_heated && exit_heated {
                // Both rooms are heated. Heat frames are optimized by minimizing runway usage in the source room.
                // But since the shortcharge difficulty is not known here, we conservatively assume up to 33 tiles
                // of the combined runway may need to be used. (TODO: Instead add a Requirement enum case to handle this more accurately.)
                let other_runway_length =
                    f32::max(0.0, f32::min(exit_length, 33.0 - entrance_length));
                let heat_frames_1 = compute_run_frames(other_runway_length) + 20;
                let heat_frames_2 = Capacity::max(
                    85,
                    compute_run_frames(other_runway_length + entrance_length),
                );
                // Add 5 lenience frames (partly to account for the possibility of some inexactness in our calculations)
                total_heat_frames += heat_frames_1 + heat_frames_2 + 5;
            } else if !entrance_heated && exit_heated {
                // Only the destination room is heated. Heat frames are optimized by using the full runway in
                // the source room.
                let (_, heat_frames) = compute_shinecharge_frames(exit_length, entrance_length);
                total_heat_frames += heat_frames + 5;
            } else if entrance_heated && !exit_heated {
                // Only the source room is heated. As in the first case above, heat frames are optimized by
                // minimizing runway usage in the source room. (TODO: Use new Requirement enum case.)
                let other_runway_length =
                    f32::max(0.0, f32::min(exit_length, 33.0 - entrance_length));
                let heat_frames_1 = compute_run_frames(other_runway_length) + 20;
                let (heat_frames_2, _) =
                    compute_shinecharge_frames(other_runway_length, entrance_length);
                total_heat_frames += heat_frames_1 + heat_frames_2 + 5;
            }
        } else if entrance_heated || exit_heated {
            // Runway in the other room starts at a different node and runs toward the door. The full combined
            // runway is used.
            let (frames_1, frames_2) = compute_shinecharge_frames(exit_length, entrance_length);
            total_heat_frames += 5;
            if exit_heated {
                // Heat frames for source room
                total_heat_frames += frames_1;
            }
            if entrance_heated {
                // Heat frames for destination room
                total_heat_frames += frames_2;
            }
        }
        total_heat_frames
    }

    fn add_run_speed_reqs(
        &self,
        exit_runway_length: f32,
        exit_min_extra_run_speed: f32,
        exit_max_extra_run_speed: f32,
        exit_heated: bool,
        entrance_min_extra_run_speed: f32,
        entrance_max_extra_run_speed: f32,
        reqs: &mut Vec<Requirement>,
    ) -> bool {
        let shortcharge_min_speed =
            get_shortcharge_min_extra_run_speed(self.difficulty.shine_charge_tiles);
        let shortcharge_max_speed_opt = get_shortcharge_max_extra_run_speed(
            self.difficulty.shine_charge_tiles,
            exit_runway_length,
        );
        let exit_min_speed = f32::max(entrance_min_extra_run_speed, shortcharge_min_speed);
        let exit_max_speed = f32::min(
            entrance_max_extra_run_speed,
            shortcharge_max_speed_opt.unwrap_or(-1.0),
        );
        let overall_min_speed = f32::max(exit_min_speed, exit_min_extra_run_speed);
        let overall_max_speed = f32::min(exit_max_speed, exit_max_extra_run_speed);
        if overall_min_speed > overall_max_speed {
            return false;
        }

        if exit_heated {
            let exit_min_speed = f32::max(
                entrance_min_extra_run_speed,
                get_shortcharge_min_extra_run_speed(self.difficulty.heated_shine_charge_tiles),
            );
            let exit_max_speed = f32::min(
                entrance_max_extra_run_speed,
                get_shortcharge_max_extra_run_speed(
                    self.difficulty.heated_shine_charge_tiles,
                    exit_runway_length,
                )
                .unwrap_or(-1.0),
            );
            let overall_min_speed = f32::max(exit_min_speed, exit_min_extra_run_speed);
            let overall_max_speed = f32::min(exit_max_speed, exit_max_extra_run_speed);
            if overall_min_speed > overall_max_speed {
                reqs.push(Requirement::Item(Item::Varia as usize));
            }
        }
        true
    }

    fn get_come_in_getting_blue_speed_reqs(
        &self,
        exit_condition: &ExitCondition,
        mut runway_length: f32,
        min_tiles: f32,
        runway_heated: bool,
        entrance_min_extra_run_speed: f32,
        entrance_max_extra_run_speed: f32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed,
                heated,
                physics,
                from_exit_node,
            } => {
                let mut effective_length = effective_length.get();
                if effective_length < min_tiles {
                    return None;
                }
                if runway_length < 0.0 {
                    // TODO: remove this hack: strats with negative runway length here coming in should use comeInBlueSpinning instead.
                    // add a test on the sm-json-data side to enforce this.
                    effective_length += runway_length;
                    runway_length = 0.0;
                }

                let mut reqs: Vec<Requirement> = vec![];
                let combined_runway_length = effective_length + runway_length;

                if !self.add_run_speed_reqs(
                    combined_runway_length,
                    min_extra_run_speed.get(),
                    7.0,
                    *heated || runway_heated,
                    entrance_min_extra_run_speed,
                    entrance_max_extra_run_speed,
                    &mut reqs,
                ) {
                    return None;
                }

                reqs.push(Requirement::make_blue_speed(
                    combined_runway_length,
                    runway_heated || *heated,
                ));
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated || runway_heated {
                    let heat_frames = self.get_cross_room_shortcharge_heat_frames(
                        *from_exit_node,
                        runway_length,
                        effective_length,
                        runway_heated,
                        *heated,
                    );
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_shinecharging_reqs(
        &self,
        exit_condition: &ExitCondition,
        mut runway_length: f32,
        min_tiles: f32,
        runway_heated: bool,
    ) -> Option<Requirement> {
        // TODO: Remove min_tiles here, after strats have been correctly split off using "comeInGettingBlueSpeed"?
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                effective_length,
                heated,
                physics,
                from_exit_node,
                ..
            } => {
                let mut effective_length = effective_length.get();
                if effective_length < min_tiles {
                    return None;
                }
                if runway_length < 0.0 {
                    // TODO: remove this hack: strats with negative runway length here coming in should use comeInBlueSpinning instead.
                    // add a test on the sm-json-data side to enforce this.
                    effective_length += runway_length;
                    runway_length = 0.0;
                }

                let mut reqs: Vec<Requirement> = vec![];
                let combined_runway_length = effective_length + runway_length;
                reqs.push(Requirement::make_shinecharge(
                    combined_runway_length,
                    runway_heated || *heated,
                ));
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated || runway_heated {
                    let heat_frames = self.get_cross_room_shortcharge_heat_frames(
                        *from_exit_node,
                        runway_length,
                        effective_length,
                        runway_heated,
                        *heated,
                    );
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_speedballing_reqs(
        &self,
        exit_condition: &ExitCondition,
        mut runway_length: f32,
        final_min_extra_run_speed: f32,
        final_max_extra_run_speed: f32,
        runway_heated: bool,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed,
                heated,
                physics,
                from_exit_node,
            } => {
                let mut effective_length = effective_length.get();
                if runway_length < 0.0 {
                    // TODO: remove this hack: strats with negative runway length here coming in should use comeInBlueSpinning instead.
                    // add a test on the sm-json-data side to enforce this.
                    effective_length += runway_length;
                    runway_length = 0.0;
                }

                let mut reqs: Vec<Requirement> = vec![Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPEEDBALL],
                )];
                let combined_runway_length = effective_length + runway_length;
                reqs.push(Requirement::SpeedBall {
                    used_tiles: Float::new(combined_runway_length),
                    heated: *heated || runway_heated,
                });

                let midair_length = f32::max(
                    0.0,
                    self.difficulty.speed_ball_tiles - self.difficulty.shine_charge_tiles,
                );
                let shortcharge_length = combined_runway_length - midair_length;
                if !self.add_run_speed_reqs(
                    shortcharge_length,
                    min_extra_run_speed.get(),
                    7.0,
                    *heated,
                    final_min_extra_run_speed,
                    final_max_extra_run_speed,
                    &mut reqs,
                ) {
                    return None;
                }

                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated || runway_heated {
                    // Speedball would technically have slightly different heat frames (compared to a shortcharge) since you no longer
                    // gaining run speed while in the air, but this is a small enough difference to neglect for now. There should be
                    // enough lenience in the heat frame calculation already to account for it.
                    let heat_frames = self.get_cross_room_shortcharge_heat_frames(
                        *from_exit_node,
                        runway_length,
                        effective_length,
                        runway_heated,
                        *heated,
                    );
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                Some(Requirement::make_and(reqs))
            }
            ExitCondition::LeaveWithMockball {
                remote_runway_length,
                landing_runway_length: _,
                blue,
                heated,
                min_extra_run_speed,
                max_extra_run_speed,
            } => {
                let remote_runway_length = remote_runway_length.get();
                if *blue == BlueOption::Yes {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPEEDBALL],
                )];
                reqs.push(Requirement::Item(
                    self.game_data.item_isv.index_by_key["Morph"],
                ));
                reqs.push(Requirement::Item(
                    self.game_data.item_isv.index_by_key["SpeedBooster"],
                ));
                if !self.add_run_speed_reqs(
                    remote_runway_length,
                    min_extra_run_speed.get(),
                    max_extra_run_speed.get(),
                    *heated,
                    final_min_extra_run_speed,
                    final_max_extra_run_speed,
                    &mut reqs,
                ) {
                    return None;
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_spinning_reqs(
        &self,
        exit_condition: &ExitCondition,
        unusable_tiles: f32,
        entrance_min_extra_run_speed: f32,
        entrance_max_extra_run_speed: f32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveSpinning {
                remote_runway_length,
                blue,
                heated: _,
                min_extra_run_speed,
                max_extra_run_speed,
            } => {
                let remote_runway_length = remote_runway_length.get();
                let min_extra_run_speed = min_extra_run_speed.get();
                let max_extra_run_speed = max_extra_run_speed.get();
                let runway_max_speed = get_max_extra_run_speed(remote_runway_length);

                let overall_max_extra_run_speed = f32::min(
                    max_extra_run_speed,
                    f32::min(entrance_max_extra_run_speed, runway_max_speed),
                );
                let overall_min_extra_run_speed =
                    f32::max(min_extra_run_speed, entrance_min_extra_run_speed);

                if overall_min_extra_run_speed > overall_max_extra_run_speed {
                    return None;
                }
                if *blue == BlueOption::Yes {
                    return None;
                }
                Some(Requirement::Free)
            }
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                let mut reqs: Vec<Requirement> = vec![];

                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }

                let min_tiles = get_extra_run_speed_tiles(entrance_min_extra_run_speed);
                let max_tiles = get_extra_run_speed_tiles(entrance_max_extra_run_speed);

                if min_tiles > effective_length - unusable_tiles {
                    return None;
                }
                if min_extra_run_speed.get() > entrance_max_extra_run_speed {
                    return None;
                }

                if *heated {
                    let heat_frames = if *from_exit_node {
                        compute_run_frames(min_tiles + unusable_tiles) * 2 + 20
                    } else if max_tiles < effective_length - unusable_tiles {
                        // 10 heat frames to position after stopping on a dime, before resuming running
                        compute_run_frames(effective_length - unusable_tiles - max_tiles)
                            + compute_run_frames(max_tiles + unusable_tiles)
                            + 10
                    } else {
                        compute_run_frames(effective_length)
                    };
                    reqs.push(Requirement::HeatFrames(heat_frames as Capacity));
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_blue_spinning_reqs(
        &self,
        exit_condition: &ExitCondition,
        unusable_tiles: f32,
        entrance_min_extra_run_speed: f32,
        entrance_max_extra_run_speed: f32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveSpinning {
                remote_runway_length,
                blue,
                heated,
                min_extra_run_speed,
                max_extra_run_speed,
            } => {
                let mut reqs: Vec<Requirement> = vec![];

                if !self.add_run_speed_reqs(
                    remote_runway_length.get(),
                    min_extra_run_speed.get(),
                    max_extra_run_speed.get(),
                    *heated,
                    entrance_min_extra_run_speed,
                    entrance_max_extra_run_speed,
                    &mut reqs,
                ) {
                    return None;
                }
                if *blue == BlueOption::No {
                    return None;
                }
                Some(Requirement::make_shinecharge(
                    remote_runway_length.get(),
                    *heated,
                ))
            }
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                let mut reqs: Vec<Requirement> = vec![];

                if !self.add_run_speed_reqs(
                    effective_length,
                    min_extra_run_speed.get(),
                    7.0,
                    *heated,
                    entrance_min_extra_run_speed,
                    entrance_max_extra_run_speed,
                    &mut reqs,
                ) {
                    return None;
                }

                reqs.push(Requirement::make_shinecharge(
                    effective_length - unusable_tiles,
                    *heated,
                ));
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *from_exit_node {
                    // Runway in the other room starts and ends at the door so we need to run both directions:
                    if *heated {
                        // Shortcharge difficulty is not known here, so we conservatively assume up to 33 tiles
                        // of runway may need to be used. (TODO: Instead add a Requirement enum case to handle this more accurately.)
                        let other_runway_length = f32::min(effective_length, 33.0 + unusable_tiles);
                        let heat_frames_1 = compute_run_frames(other_runway_length) + 20;
                        let (heat_frames_2, _) =
                            compute_shinecharge_frames(other_runway_length, 0.0);
                        reqs.push(Requirement::HeatFrames(heat_frames_1 + heat_frames_2 + 5));
                    }
                } else if *heated {
                    // Runway in the other room starts at a different node and runs toward the door. The full combined
                    // runway is used.
                    let (frames_1, _) = compute_shinecharge_frames(effective_length, 0.0);
                    let heat_frames = frames_1 + 5;
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_blue_space_jumping_reqs(
        &self,
        exit_condition: &ExitCondition,
        entrance_min_extra_run_speed: f32,
        entrance_max_extra_run_speed: f32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveSpaceJumping {
                remote_runway_length,
                blue,
                heated,
                min_extra_run_speed,
                max_extra_run_speed,
            } => {
                let mut reqs: Vec<Requirement> = vec![];

                if !self.add_run_speed_reqs(
                    remote_runway_length.get(),
                    min_extra_run_speed.get(),
                    max_extra_run_speed.get(),
                    *heated,
                    entrance_min_extra_run_speed,
                    entrance_max_extra_run_speed,
                    &mut reqs,
                ) {
                    return None;
                }
                if *blue == BlueOption::No {
                    return None;
                }
                reqs.push(Requirement::make_blue_speed(
                    remote_runway_length.get(),
                    *heated,
                ));
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key
                        [&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                ));
                reqs.push(Requirement::Or(vec![
                    Requirement::NoFlashSuit,
                    Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_TRICKY_CARRY_FLASH_SUIT],
                    ),
                ]));
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_mockball_reqs(
        &self,
        exit_condition: &ExitCondition,
        speed_booster: Option<bool>,
        adjacent_min_tiles: f32,
        remote_and_landing_min_tiles: Vec<(f32, f32)>,
    ) -> Option<Requirement> {
        let mut reqs: Vec<Requirement> = vec![];
        if speed_booster == Some(true) {
            reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
        }
        if speed_booster == Some(false) {
            reqs.push(Requirement::Tech(
                self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_DISABLE_EQUIPMENT],
            ));
        }
        match exit_condition {
            ExitCondition::LeaveWithMockball {
                remote_runway_length,
                landing_runway_length,
                blue,
                ..
            } => {
                // TODO: Take into account any exit constraints on min_extra_run_speed and max_extra_run_speed.
                let remote_runway_length = remote_runway_length.get();
                let landing_runway_length = landing_runway_length.get();

                if *blue == BlueOption::Yes {
                    return None;
                }
                if !remote_and_landing_min_tiles
                    .iter()
                    .any(|(r, d)| *r <= remote_runway_length && *d <= landing_runway_length)
                {
                    return None;
                }
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                ));
                reqs.push(Requirement::Item(Item::Morph as ItemId));
                Some(Requirement::make_and(reqs))
            }
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed: _,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                if effective_length < adjacent_min_tiles {
                    return None;
                }
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                    // TODO: in sm-json-data, add physics property to leaveWithRunway schema (for door nodes with multiple possible physics)
                }
                if *heated {
                    let heat_frames = if *from_exit_node {
                        compute_run_frames(adjacent_min_tiles) * 2 + 20
                    } else {
                        compute_run_frames(effective_length)
                    };
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                ));
                reqs.push(Requirement::Item(Item::Morph as ItemId));
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_spring_ball_bounce_reqs(
        &self,
        exit_condition: &ExitCondition,
        speed_booster: Option<bool>,
        adjacent_min_tiles: f32,
        remote_and_landing_min_tiles: Vec<(f32, f32)>,
        exit_movement_type: BounceMovementType,
    ) -> Option<Requirement> {
        let mut reqs: Vec<Requirement> = vec![];
        if speed_booster == Some(true) {
            reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
        }
        if speed_booster == Some(false) {
            reqs.push(Requirement::Tech(
                self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_DISABLE_EQUIPMENT],
            ));
        }
        match exit_condition {
            ExitCondition::LeaveWithMockball {
                remote_runway_length,
                landing_runway_length,
                blue,
                ..
            } => {
                // TODO: Take into account any exit constraints on min_extra_run_speed and max_extra_run_speed.
                // Currently there might not be any scenarios where this matters, but that could change?
                // It is awkward because for a non-blue entrance strat like this, the constraints are measured in tiles rather
                // than run speed, though we could convert between the two.
                let remote_runway_length = remote_runway_length.get();
                let landing_runway_length = landing_runway_length.get();
                if *blue == BlueOption::Yes {
                    return None;
                }
                if !remote_and_landing_min_tiles
                    .iter()
                    .any(|(r, d)| *r <= remote_runway_length && *d <= landing_runway_length)
                {
                    return None;
                }
                if exit_movement_type == BounceMovementType::Uncontrolled {
                    return None;
                }
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                ));
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                ));
                reqs.push(Requirement::Item(Item::Morph as ItemId));
                reqs.push(Requirement::Item(Item::SpringBall as ItemId));
                Some(Requirement::make_and(reqs))
            }
            ExitCondition::LeaveWithSpringBallBounce {
                remote_runway_length,
                landing_runway_length,
                blue,
                movement_type,
                ..
            } => {
                // TODO: Take into account any exit constraints on min_extra_run_speed and max_extra_run_speed.
                let remote_runway_length = remote_runway_length.get();
                let landing_runway_length = landing_runway_length.get();
                if *blue == BlueOption::Yes {
                    return None;
                }
                if !remote_and_landing_min_tiles
                    .iter()
                    .any(|(r, d)| *r <= remote_runway_length && *d <= landing_runway_length)
                {
                    return None;
                }
                if *movement_type != exit_movement_type
                    && *movement_type != BounceMovementType::Any
                    && exit_movement_type != BounceMovementType::Any
                {
                    return None;
                }
                if *movement_type == BounceMovementType::Controlled
                    || exit_movement_type == BounceMovementType::Controlled
                {
                    reqs.push(Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                    ));
                }
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                ));
                reqs.push(Requirement::Item(Item::Morph as ItemId));
                reqs.push(Requirement::Item(Item::SpringBall as ItemId));
                Some(Requirement::make_and(reqs))
            }
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed: _,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                if effective_length < adjacent_min_tiles {
                    return None;
                }
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated {
                    let heat_frames = if *from_exit_node {
                        compute_run_frames(adjacent_min_tiles) * 2 + 20
                    } else {
                        compute_run_frames(effective_length)
                    };
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                if exit_movement_type == BounceMovementType::Controlled {
                    reqs.push(Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                    ));
                }
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                ));
                reqs.push(Requirement::Item(Item::Morph as ItemId));
                reqs.push(Requirement::Item(Item::SpringBall as ItemId));
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_blue_spring_ball_bounce_reqs(
        &self,
        exit_condition: &ExitCondition,
        entrance_min_extra_run_speed: f32,
        entrance_max_extra_run_speed: f32,
        min_landing_tiles: f32,
        entrance_movement_type: BounceMovementType,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithMockball {
                remote_runway_length,
                landing_runway_length,
                blue,
                heated,
                min_extra_run_speed,
                max_extra_run_speed,
            } => {
                let remote_runway_length = remote_runway_length.get();
                let landing_runway_length = landing_runway_length.get();
                if *blue == BlueOption::Yes {
                    return None;
                }
                if entrance_movement_type == BounceMovementType::Uncontrolled {
                    return None;
                }
                if landing_runway_length < min_landing_tiles {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![];

                if !self.add_run_speed_reqs(
                    remote_runway_length,
                    min_extra_run_speed.get(),
                    max_extra_run_speed.get(),
                    *heated,
                    entrance_min_extra_run_speed,
                    entrance_max_extra_run_speed,
                    &mut reqs,
                ) {
                    return None;
                }

                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                ));
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                ));
                reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
                reqs.push(Requirement::Item(Item::Morph as ItemId));
                reqs.push(Requirement::Item(Item::SpringBall as ItemId));
                Some(Requirement::make_and(reqs))
            }
            ExitCondition::LeaveWithSpringBallBounce {
                remote_runway_length,
                landing_runway_length,
                blue,
                heated,
                movement_type,
                min_extra_run_speed,
                max_extra_run_speed,
            } => {
                let remote_runway_length = remote_runway_length.get();
                let landing_runway_length = landing_runway_length.get();
                if *blue == BlueOption::Yes {
                    return None;
                }
                if landing_runway_length < min_landing_tiles {
                    return None;
                }
                if *movement_type != entrance_movement_type
                    && *movement_type != BounceMovementType::Any
                    && entrance_movement_type != BounceMovementType::Any
                {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![];

                if !self.add_run_speed_reqs(
                    remote_runway_length,
                    min_extra_run_speed.get(),
                    max_extra_run_speed.get(),
                    *heated,
                    entrance_min_extra_run_speed,
                    entrance_max_extra_run_speed,
                    &mut reqs,
                ) {
                    return None;
                }

                if *movement_type == BounceMovementType::Controlled
                    || entrance_movement_type == BounceMovementType::Controlled
                {
                    reqs.push(Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                    ));
                }
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                ));
                reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
                reqs.push(Requirement::Item(Item::Morph as ItemId));
                reqs.push(Requirement::Item(Item::SpringBall as ItemId));
                Some(Requirement::make_and(reqs))
            }
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                let mut reqs: Vec<Requirement> = vec![];

                if !self.add_run_speed_reqs(
                    effective_length,
                    min_extra_run_speed.get(),
                    7.0,
                    *heated,
                    entrance_min_extra_run_speed,
                    entrance_max_extra_run_speed,
                    &mut reqs,
                ) {
                    return None;
                }

                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated {
                    let heat_frames = if *from_exit_node {
                        // For now, be conservative by assuming we use the whole runway. This could be refined later:
                        compute_run_frames(effective_length) * 2 + 20
                    } else {
                        compute_run_frames(effective_length)
                    };
                    reqs.push(Requirement::HeatFrames(heat_frames));
                }
                if entrance_movement_type == BounceMovementType::Controlled {
                    reqs.push(Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                    ));
                }
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                ));
                reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
                reqs.push(Requirement::Item(Item::Morph as ItemId));
                reqs.push(Requirement::Item(Item::SpringBall as ItemId));
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_shinecharged_reqs(&self, exit_condition: &ExitCondition) -> Option<Requirement> {
        let mut reqs = vec![];
        reqs.push(Requirement::Tech(
            self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SHINECHARGE_MOVEMENT],
        ));
        match exit_condition {
            ExitCondition::LeaveShinecharged { .. } => Some(Requirement::Free),
            ExitCondition::LeaveNormally {} => Some(Requirement::UseFlashSuit {
                carry_flash_suit_tech_idx: self.game_data.tech_isv.index_by_key
                    [&TECH_ID_CAN_CARRY_FLASH_SUIT],
            }),
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed: _,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::make_shinecharge(effective_length, *heated));
                reqs.push(Requirement::ShineChargeFrames(10)); // Assume shinecharge is obtained 10 frames before going through door.
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated {
                    if *from_exit_node {
                        let runway_length = f32::min(33.0, effective_length);
                        let run_frames = compute_run_frames(runway_length);
                        let heat_frames_1 = run_frames + 20;
                        let heat_frames_2 = Capacity::max(85, run_frames);
                        reqs.push(Requirement::HeatFrames(heat_frames_1 + heat_frames_2 + 15));
                    } else {
                        let heat_frames = Capacity::max(85, compute_run_frames(effective_length));
                        reqs.push(Requirement::HeatFrames(heat_frames + 5));
                    }
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_shinecharged_jumping_reqs(
        &self,
        exit_condition: &ExitCondition,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveNormally {} => Some(Requirement::UseFlashSuit {
                carry_flash_suit_tech_idx: self.game_data.tech_isv.index_by_key
                    [&TECH_ID_CAN_CARRY_FLASH_SUIT],
            }),
            ExitCondition::LeaveShinecharged { physics } => {
                if *physics != Some(Physics::Air) {
                    return None;
                }
                Some(Requirement::Free)
            }
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed: _,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::make_shinecharge(effective_length, *heated));
                reqs.push(Requirement::ShineChargeFrames(10)); // Assume shinecharge is obtained 10 frames before going through door.
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated {
                    if *from_exit_node {
                        let runway_length = f32::min(33.0, effective_length);
                        let run_frames = compute_run_frames(runway_length);
                        let heat_frames_1 = run_frames + 20;
                        let heat_frames_2 = Capacity::max(85, run_frames);
                        reqs.push(Requirement::HeatFrames(heat_frames_1 + heat_frames_2 + 15));
                    } else {
                        let heat_frames = Capacity::max(85, compute_run_frames(effective_length));
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
        exit_condition: &ExitCondition,
        min_tiles: f32,
        include_shinecharge: bool,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed: _,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                if *physics != Some(Physics::Air) {
                    return None;
                }
                if effective_length < min_tiles {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_STUTTER_WATER_SHINECHARGE],
                ));
                reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
                if include_shinecharge {
                    reqs.push(Requirement::ShineCharge {
                        used_tiles: Float::new(45.0),
                        heated: false,
                    });
                }
                if *heated {
                    let heat_frames = if *from_exit_node {
                        compute_run_frames(min_tiles) * 2 + 20
                    } else {
                        compute_run_frames(effective_length)
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
        exit_condition: &ExitCondition,
        come_in_position: SparkPosition,
        door_orientation: DoorOrientation,
    ) -> Option<Requirement> {
        let mut reqs = vec![];
        if door_orientation == DoorOrientation::Left || door_orientation == DoorOrientation::Right {
            reqs.push(Requirement::Tech(
                self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_HORIZONTAL_SHINESPARK],
            ));
        }
        if come_in_position == SparkPosition::Top {
            reqs.push(Requirement::Tech(
                self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MIDAIR_SHINESPARK],
            ));
        }
        match exit_condition {
            ExitCondition::LeaveNormally {} => Some(Requirement::UseFlashSuit {
                carry_flash_suit_tech_idx: self.game_data.tech_isv.index_by_key
                    [&TECH_ID_CAN_CARRY_FLASH_SUIT],
            }),
            ExitCondition::LeaveWithSpark { position, .. } => {
                if *position == come_in_position
                    || *position == SparkPosition::Any
                    || come_in_position == SparkPosition::Any
                {
                    if *position == SparkPosition::Top && come_in_position == SparkPosition::Any {
                        reqs.push(Requirement::Tech(
                            self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MIDAIR_SHINESPARK],
                        ));
                    }
                    Some(Requirement::make_and(reqs))
                } else {
                    None
                }
            }
            ExitCondition::LeaveShinecharged { .. } => {
                // Shinecharge frames are handled through Requirement::ShineChargeFrames
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SHINECHARGE_MOVEMENT],
                ));
                Some(Requirement::make_and(reqs))
            }
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed: _,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                reqs.push(Requirement::make_shinecharge(effective_length, *heated));
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated {
                    if *from_exit_node {
                        let runway_length = f32::min(33.0, effective_length);
                        let run_frames = compute_run_frames(runway_length);
                        let heat_frames_1 = run_frames + 20;
                        let heat_frames_2 = Capacity::max(85, run_frames);
                        reqs.push(Requirement::HeatFrames(heat_frames_1 + heat_frames_2 + 5));
                    } else {
                        let heat_frames = Capacity::max(85, compute_run_frames(effective_length));
                        reqs.push(Requirement::HeatFrames(heat_frames + 5));
                    }
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_temporary_blue_reqs(
        &self,
        exit_condition: &ExitCondition,
        exit_direction: TemporaryBlueDirection,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithTemporaryBlue { direction } => {
                if *direction != exit_direction
                    && *direction != TemporaryBlueDirection::Any
                    && exit_direction != TemporaryBlueDirection::Any
                {
                    return None;
                }
                Some(Requirement::make_and(vec![
                    Requirement::NoFlashSuit,
                    Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_TEMPORARY_BLUE],
                    ),
                ]))
            }
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed: _,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_TEMPORARY_BLUE],
                ));
                reqs.push(Requirement::make_shinecharge(effective_length, *heated));
                if *physics != Some(Physics::Air) {
                    reqs.push(Requirement::Item(Item::Gravity as ItemId));
                }
                if *heated {
                    let heat_frames_temp_blue = 200;
                    if *from_exit_node {
                        let runway_length = f32::min(33.0, effective_length);
                        let run_frames = compute_run_frames(runway_length);
                        let heat_frames_1 = run_frames + 20;
                        let heat_frames_2 = Capacity::max(85, run_frames);
                        reqs.push(Requirement::HeatFrames(
                            heat_frames_1 + heat_frames_2 + heat_frames_temp_blue + 15,
                        ));
                    } else {
                        let heat_frames = Capacity::max(85, compute_run_frames(effective_length));
                        reqs.push(Requirement::HeatFrames(
                            heat_frames + heat_frames_temp_blue + 5,
                        ));
                    }
                }
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_bomb_boost_reqs(
        &self,
        exit_condition: &ExitCondition,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                effective_length,
                min_extra_run_speed: _,
                heated,
                physics,
                from_exit_node,
            } => {
                let effective_length = effective_length.get();
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
                    if *from_exit_node {
                        heat_frames += compute_run_frames(effective_length);
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
        exit_condition: &ExitCondition,
        entrance_heated: bool,
        door_orientation: DoorOrientation,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithRunway {
                heated,
                physics,
                from_exit_node,
                ..
            } => {
                if !from_exit_node {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_STATIONARY_SPIN_JUMP],
                ));
                if door_orientation == DoorOrientation::Right {
                    reqs.push(Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK],
                    ));
                    if *physics != Some(Physics::Air) {
                        reqs.push(Requirement::Or(vec![
                            Requirement::Item(Item::Gravity as ItemId),
                            Requirement::Tech(
                                self.game_data.tech_isv.index_by_key
                                    [&TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK_FROM_WATER],
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

    fn get_come_in_with_r_mode_reqs(
        &self,
        exit_condition: &ExitCondition,
        heated: bool,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithGModeSetup { .. } => {
                let reqs: Vec<Requirement> = vec![
                    Requirement::Tech(
                        self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_R_MODE],
                    ),
                    Requirement::Item(Item::XRayScope as ItemId),
                    Requirement::NoFlashSuit,
                    Requirement::ReserveTrigger {
                        min_reserve_energy: 1,
                        max_reserve_energy: 400,
                        heated,
                    },
                ];
                Some(Requirement::make_and(reqs))
            }
            _ => None,
        }
    }

    fn get_come_in_with_g_mode_reqs(
        &self,
        exit_condition: &ExitCondition,
        entrance_room_id: RoomId,
        entrance_node_id: NodeId,
        mut mode: GModeMode,
        entrance_morphed: bool,
        mobility: GModeMobility,
        entrance_heated: bool,
        is_toilet: bool,
    ) -> Option<Requirement> {
        if is_toilet {
            // Take into account that obtaining direct G-mode in the Toilet is not possible.
            match mode {
                GModeMode::Any => {
                    mode = GModeMode::Indirect;
                }
                GModeMode::Direct => {
                    return None;
                }
                GModeMode::Indirect => {}
            }
        }

        let empty_vec = vec![];
        let regain_mobility_vec = self
            .game_data
            .node_gmode_regain_mobility
            .get(&(entrance_room_id, entrance_node_id))
            .unwrap_or(&empty_vec);
        match exit_condition {
            ExitCondition::LeaveWithGModeSetup { knockback, heated } => {
                if mode == GModeMode::Indirect {
                    return None;
                }
                let mut reqs: Vec<Requirement> = vec![];
                reqs.push(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE],
                ));
                if entrance_morphed {
                    reqs.push(Requirement::Or(vec![
                        Requirement::Tech(
                            self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_ARTIFICIAL_MORPH],
                        ),
                        Requirement::Item(Item::Morph as ItemId),
                    ]));
                }
                if *heated || entrance_heated {
                    reqs.push(Requirement::make_or(vec![
                        Requirement::Tech(
                            self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_HEATED_G_MODE],
                        ),
                        Requirement::Item(Item::Varia as ItemId),
                    ]));
                }
                reqs.push(Requirement::NoFlashSuit);
                reqs.push(Requirement::Item(Item::XRayScope as ItemId));

                let mobile_req = if *knockback {
                    Requirement::ReserveTrigger {
                        min_reserve_energy: 1,
                        max_reserve_energy: 4,
                        heated: false,
                    }
                } else {
                    Requirement::Never
                };
                let immobile_req = if !regain_mobility_vec.is_empty() {
                    let mut immobile_req_or_vec: Vec<Requirement> = Vec::new();
                    for (regain_mobility_link, _) in regain_mobility_vec {
                        immobile_req_or_vec.push(Requirement::make_and(vec![
                            Requirement::Tech(
                                self.game_data.tech_isv.index_by_key
                                    [&TECH_ID_CAN_ENTER_G_MODE_IMMOBILE],
                            ),
                            Requirement::ReserveTrigger {
                                min_reserve_energy: 1,
                                max_reserve_energy: 400,
                                heated: false,
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
                Some(Requirement::Tech(
                    self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOONFALL],
                ))
            }
            _ => None,
        }
    }

    fn get_come_in_with_wall_jump_below_reqs(
        &self,
        exit_condition: &ExitCondition,
        min_height: f32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithDoorFrameBelow { height, .. } => {
                let height = height.get();
                if height < min_height {
                    return None;
                }
                Some(Requirement::Walljump)
            }
            _ => None,
        }
    }

    fn get_come_in_with_space_jump_below_reqs(
        &self,
        exit_condition: &ExitCondition,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithDoorFrameBelow { heated, .. } => {
                let mut reqs_and_vec = vec![];

                reqs_and_vec.push(Requirement::Item(
                    self.game_data.item_isv.index_by_key["SpaceJump"],
                ));
                if *heated {
                    reqs_and_vec.push(Requirement::HeatFrames(30));
                }
                Some(Requirement::make_and(reqs_and_vec))
            }
            _ => None,
        }
    }

    fn get_come_in_with_side_platform_reqs(
        &self,
        exit_condition: &ExitCondition,
        platforms: &[SidePlatformEntrance],
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithSidePlatform {
                effective_length,
                height,
                obstruction,
                environment,
            } => {
                let effective_length = effective_length.get();
                let height = height.get();
                let mut reqs_or_vec = vec![];
                for p in platforms {
                    let mut reqs = vec![];
                    if &p.environment != environment
                        && p.environment != SidePlatformEnvironment::Any
                        && environment != &SidePlatformEnvironment::Any
                    {
                        continue;
                    }
                    if effective_length < p.min_tiles.get() {
                        continue;
                    }
                    if height < p.min_height.get() || height > p.max_height.get() {
                        continue;
                    }
                    if !p.obstructions.contains(obstruction) {
                        continue;
                    }
                    if p.speed_booster == Some(true) {
                        reqs.push(Requirement::Item(Item::SpeedBooster as ItemId));
                    }
                    if p.speed_booster == Some(false) {
                        reqs.push(Requirement::Tech(
                            self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_DISABLE_EQUIPMENT],
                        ));
                    }
                    reqs.push(p.requirement.clone());
                    reqs_or_vec.push(Requirement::make_and(reqs));
                }
                Some(Requirement::make_and(vec![
                    Requirement::Tech(
                        self.game_data.tech_isv.index_by_key
                            [&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                    ),
                    Requirement::make_or(reqs_or_vec),
                ]))
            }
            _ => None,
        }
    }

    fn get_come_in_with_grapple_swing_reqs(
        &self,
        exit_condition: &ExitCondition,
        entrance_blocks: &[GrappleSwingBlock],
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithGrappleSwing { blocks } => {
                let entrance_blocks_set: HashSet<GrappleSwingBlock> =
                    entrance_blocks.iter().cloned().collect();
                if blocks.iter().any(|x| entrance_blocks_set.contains(x)) {
                    Some(Requirement::make_and(vec![
                        Requirement::Tech(
                            self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_PRECISE_GRAPPLE],
                        ),
                        Requirement::Item(self.game_data.item_isv.index_by_key["Grapple"]),
                    ]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn get_come_in_with_grapple_jump_reqs(
        &self,
        exit_condition: &ExitCondition,
        entrance_position: GrappleJumpPosition,
    ) -> Option<Requirement> {
        let grapple_jump_req = self.game_data.tech_requirement[&(TECH_ID_CAN_GRAPPLE_JUMP, true)]
            .clone()
            .unwrap();

        match exit_condition {
            ExitCondition::LeaveWithGrappleJump { position } => {
                if position == &GrappleJumpPosition::Any
                    || entrance_position == GrappleJumpPosition::Any
                    || &entrance_position == position
                {
                    Some(Requirement::make_and(vec![
                        grapple_jump_req,
                        Requirement::Item(self.game_data.item_isv.index_by_key["Grapple"]),
                        Requirement::Item(self.game_data.item_isv.index_by_key["Morph"]),
                    ]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn get_come_in_with_grapple_teleport_reqs(
        &self,
        exit_condition: &ExitCondition,
        entrance_block_positions: &[(u16, u16)],
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithGrappleTeleport { block_positions } => {
                let entrance_block_positions_set: HashSet<(u16, u16)> =
                    entrance_block_positions.iter().copied().collect();
                if block_positions
                    .iter()
                    .any(|x| entrance_block_positions_set.contains(x))
                {
                    Some(Requirement::make_and(vec![
                        Requirement::Tech(
                            self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_TELEPORT],
                        ),
                        Requirement::Item(self.game_data.item_isv.index_by_key["Grapple"]),
                    ]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn get_come_in_with_samus_eater_teleport_reqs(
        &self,
        exit_condition: &ExitCondition,
        entrance_floor_positions: &[(u16, u16)],
        entrance_ceiling_positions: &[(u16, u16)],
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithSamusEaterTeleport {
                floor_positions,
                ceiling_positions,
            } => {
                let entrance_floor_positions_set: HashSet<(u16, u16)> =
                    entrance_floor_positions.iter().copied().collect();
                let entrance_ceiling_positions_set: HashSet<(u16, u16)> =
                    entrance_ceiling_positions.iter().copied().collect();
                if floor_positions
                    .iter()
                    .any(|x| entrance_floor_positions_set.contains(x))
                    || ceiling_positions
                        .iter()
                        .any(|x| entrance_ceiling_positions_set.contains(x))
                {
                    Some(Requirement::make_and(vec![
                        Requirement::NoFlashSuit,
                        Requirement::Tech(
                            self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SAMUS_EATER_TELEPORT],
                        ),
                    ]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn get_come_in_with_super_sink_reqs(
        &self,
        exit_condition: &ExitCondition,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithSuperSink {} => Some(Requirement::Tech(
                self.game_data.tech_isv.index_by_key[&TECH_ID_CAN_SUPER_SINK],
            )),
            _ => None,
        }
    }

    fn get_come_in_with_platform_below_reqs(
        &self,
        exit_condition: &ExitCondition,
        min_height: f32,
        max_height: f32,
        max_left_position: f32,
        min_right_position: f32,
    ) -> Option<Requirement> {
        match exit_condition {
            ExitCondition::LeaveWithPlatformBelow {
                height,
                left_position,
                right_position,
            } => {
                let height = height.get();
                let left_position = left_position.get();
                let right_position = right_position.get();
                if height < min_height || height > max_height {
                    return None;
                }
                if left_position > max_left_position {
                    return None;
                }
                if right_position < min_right_position {
                    return None;
                }
                Some(Requirement::Free)
            }
            _ => None,
        }
    }
}

fn get_randomizable_doors(
    game_data: &GameData,
    walls: &[DoorPtrPair],
    objectives: &[Objective],
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

    non_randomizable_doors.extend(walls);

    // Avoid placing an ammo door on a tile with an objective "X", as it looks bad.
    for i in objectives.iter() {
        use Objective::*;
        match i {
            SporeSpawn => {
                non_randomizable_doors.insert((Some(0x18E4A), Some(0x18D2A)));
            }
            Crocomire => {
                non_randomizable_doors.insert((Some(0x193DE), Some(0x19432)));
            }
            Botwoon => {
                non_randomizable_doors.insert((Some(0x1A918), Some(0x1A84C)));
            }
            GoldenTorizo => {
                non_randomizable_doors.insert((Some(0x19876), Some(0x1983A)));
            }
            MetroidRoom1 => {
                non_randomizable_doors.insert((Some(0x1A9B4), Some(0x1A9C0))); // left
                non_randomizable_doors.insert((Some(0x1A9A8), Some(0x1A984))); // right
            }
            MetroidRoom2 => {
                non_randomizable_doors.insert((Some(0x1A9C0), Some(0x1A9B4))); // top right
                non_randomizable_doors.insert((Some(0x1A9CC), Some(0x1A9D8))); // bottom right
            }
            MetroidRoom3 => {
                non_randomizable_doors.insert((Some(0x1A9D8), Some(0x1A9CC))); // left
                non_randomizable_doors.insert((Some(0x1A9E4), Some(0x1A9F0))); // right
            }
            MetroidRoom4 => {
                non_randomizable_doors.insert((Some(0x1A9F0), Some(0x1A9E4))); // left
                non_randomizable_doors.insert((Some(0x1A9FC), Some(0x1AA08))); // bottom
            }
            _ => {} // All other tiles have gray doors and are excluded above.
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
    walls: &[DoorPtrPair],
    objectives: &[Objective],
) -> Vec<(DoorPtrPair, DoorPtrPair)> {
    let doors = get_randomizable_doors(game_data, walls, objectives);
    let mut out: Vec<(DoorPtrPair, DoorPtrPair)> = vec![];
    for (src_door_ptr_pair, dst_door_ptr_pair, _bidirectional) in &map.doors {
        if doors.contains(src_door_ptr_pair) && doors.contains(dst_door_ptr_pair) {
            out.push((*src_door_ptr_pair, *dst_door_ptr_pair));
        }
    }
    out
}

fn get_walls(map: &Map, game_data: &GameData) -> Vec<DoorPtrPair> {
    let mut out = vec![];
    let mut door_set: HashSet<DoorPtrPair> = HashSet::new();
    for door in &map.doors {
        door_set.insert(door.0);
        door_set.insert(door.1);
    }
    for (room_idx, room) in game_data.room_geometry.iter().enumerate() {
        if !map.room_mask[room_idx] {
            continue;
        }
        for door in &room.doors {
            let pair = (door.exit_ptr, door.entrance_ptr);
            if !door_set.contains(&pair) {
                out.push(pair);
            }
        }
    }
    out
}

pub fn randomize_doors(
    game_data: &GameData,
    map: &Map,
    settings: &RandomizerSettings,
    objectives: &[Objective],
    seed: usize,
) -> LockedDoorData {
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
    let mut used_beam_rooms: HashSet<RoomGeometryRoomIdx> = HashSet::new();
    let mut door_types = vec![];

    match settings.doors_mode {
        DoorsMode::Blue => {}
        DoorsMode::Ammo => {
            let red_doors_cnt = 30;
            let green_doors_cnt = 15;
            let yellow_doors_cnt = 10;
            door_types.extend(vec![DoorType::Red; red_doors_cnt]);
            door_types.extend(vec![DoorType::Green; green_doors_cnt]);
            door_types.extend(vec![DoorType::Yellow; yellow_doors_cnt]);
        }
        DoorsMode::Beam => {
            let red_doors_cnt = 18;
            let green_doors_cnt = 10;
            let yellow_doors_cnt = 7;
            let beam_door_each_cnt = 4;
            door_types.extend(vec![DoorType::Red; red_doors_cnt]);
            door_types.extend(vec![DoorType::Green; green_doors_cnt]);
            door_types.extend(vec![DoorType::Yellow; yellow_doors_cnt]);
            door_types.extend(vec![DoorType::Beam(BeamType::Charge); beam_door_each_cnt]);
            door_types.extend(vec![DoorType::Beam(BeamType::Ice); beam_door_each_cnt]);
            door_types.extend(vec![DoorType::Beam(BeamType::Wave); beam_door_each_cnt]);
            door_types.extend(vec![DoorType::Beam(BeamType::Spazer); beam_door_each_cnt]);
            door_types.extend(vec![DoorType::Beam(BeamType::Plasma); beam_door_each_cnt]);
        }
    };
    let walls = get_walls(map, game_data);
    let door_conns = get_randomizable_door_connections(game_data, map, &walls, objectives);
    let mut locked_doors: Vec<LockedDoor> = vec![];
    let total_cnt = door_types.len();
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
        if used_locs.contains(&src_loc) || used_locs.contains(&dst_loc) {
            continue;
        }
        if let DoorType::Beam(_) = door_types[i] {
            let src_room_idx = src_loc.0;
            let dst_room_idx = dst_loc.0;
            if used_beam_rooms.contains(&src_room_idx) || used_beam_rooms.contains(&dst_room_idx) {
                continue;
            }
            used_beam_rooms.insert(src_room_idx);
            used_beam_rooms.insert(dst_room_idx);
        }
        used_locs.insert(src_loc);
        used_locs.insert(dst_loc);
        locked_doors.push(door);
    }

    for &ptr_pair in &walls {
        locked_doors.push(LockedDoor {
            src_ptr_pair: ptr_pair,
            dst_ptr_pair: (None, None),
            door_type: DoorType::Wall,
            bidirectional: false,
        });
    }

    let mut locked_door_node_map: HashMap<(RoomId, NodeId), usize> = HashMap::new();
    for (i, door) in locked_doors.iter().enumerate() {
        let (src_room_id, src_node_id) = game_data.door_ptr_pair_map[&door.src_ptr_pair];
        locked_door_node_map.insert((src_room_id, src_node_id), i);
        if door.bidirectional {
            let (dst_room_id, dst_node_id) = game_data.door_ptr_pair_map[&door.dst_ptr_pair];
            locked_door_node_map.insert((dst_room_id, dst_node_id), i);
        }
    }

    // Homing Geemer Room left door -> West Ocean Bridge left door
    if let Some(&idx) = locked_door_node_map.get(&(313, 1)) {
        locked_door_node_map.insert((32, 7), idx);
    }

    // Homing Geemer Room right door -> West Ocean Bridge right door
    if let Some(&idx) = locked_door_node_map.get(&(313, 2)) {
        locked_door_node_map.insert((32, 8), idx);
    }

    // Pants Room right door -> East Pants Room right door
    if let Some(&idx) = locked_door_node_map.get(&(322, 2)) {
        locked_door_node_map.insert((220, 2), idx);
    }

    let mut locked_door_vertex_ids = vec![vec![]; locked_doors.len()];
    for (&(room_id, node_id), vertex_ids) in &game_data.node_door_unlock {
        if let Some(&locked_door_idx) = locked_door_node_map.get(&(room_id, node_id)) {
            locked_door_vertex_ids[locked_door_idx].extend(vertex_ids);
        }
    }

    LockedDoorData {
        locked_doors,
        locked_door_node_map,
        locked_door_vertex_ids,
    }
}

// An inexpensive way to evaluate whether a requirement may be possible to satisfy,
// given the tech and notables available. A more precise evaluation is computed
// by crate::difficulty::is_req_possible.
fn is_req_maybe_possible(
    req: &Requirement,
    tech_active: &[bool],
    notables_active: &[bool],
) -> bool {
    match req {
        Requirement::Tech(tech_idx) => tech_active[*tech_idx],
        Requirement::Notable(notable_idx) => notables_active[*notable_idx],
        Requirement::And(reqs) => reqs
            .iter()
            .all(|x| is_req_maybe_possible(x, tech_active, notables_active)),
        Requirement::Or(reqs) => reqs
            .iter()
            .any(|x| is_req_maybe_possible(x, tech_active, notables_active)),
        _ => true,
    }
}

pub fn filter_links(
    links: &[Link],
    _game_data: &GameData,
    difficulty: &DifficultyConfig,
) -> Vec<Link> {
    let mut out = vec![];
    for link in links {
        if is_req_maybe_possible(&link.requirement, &difficulty.tech, &difficulty.notables) {
            out.push(link.clone())
        }
    }
    out
}

fn get_minimal_tank_count(difficulty: &DifficultyConfig) -> usize {
    if difficulty.ridley_proficiency < 0.3 {
        12
    } else if difficulty.ridley_proficiency < 0.8 {
        9
    } else if difficulty.ridley_proficiency < 0.9 {
        7
    } else {
        3
    }
}

pub fn strip_name(s: &str) -> String {
    let mut out = String::new();
    for word in s.split_inclusive(|x: char| !x.is_ascii_alphabetic()) {
        let capitalized_word = word[0..1].to_ascii_uppercase() + &word[1..];
        let stripped_word: String = capitalized_word
            .chars()
            .filter(|x| x.is_ascii_alphanumeric())
            .collect();
        out += &stripped_word;
    }
    out
}

pub fn is_equivalent_difficulty(a: &DifficultyConfig, b: &DifficultyConfig) -> bool {
    let mut a1 = a.clone();
    let mut b1 = b.clone();
    a1.name = "".to_string();
    b1.name = "".to_string();
    a1 == b1
}

pub fn get_difficulty_tiers(
    settings: &RandomizerSettings,
    tier_settings: &[DifficultyConfig],
    game_data: &GameData,
    implicit_tech: &[TechId],
    implicit_notables: &[(RoomId, NotableId)],
) -> Vec<DifficultyConfig> {
    let main_tier = DifficultyConfig::new(
        &settings.skill_assumption_settings,
        game_data,
        implicit_tech,
        implicit_notables,
    );
    let mut difficulty_tiers = vec![];

    difficulty_tiers.push(main_tier.clone());
    if [ItemPlacementStyle::Forced, ItemPlacementStyle::Local]
        .contains(&settings.item_progression_settings.item_placement_style)
    {
        for ref_tier in tier_settings {
            let new_tier = DifficultyConfig::intersect(ref_tier, &main_tier);
            if is_equivalent_difficulty(&new_tier, difficulty_tiers.last().unwrap()) {
                difficulty_tiers.pop();
            }
            difficulty_tiers.push(new_tier);
        }
    }
    difficulty_tiers
}

pub fn get_objectives<R: Rng>(
    settings: &RandomizerSettings,
    map: Option<&Map>,
    game_data: &GameData,
    rng: &mut R,
) -> Vec<Objective> {
    let obj_settings = &settings.objective_settings;
    let num_objectives =
        rng.gen_range(obj_settings.min_objectives..=obj_settings.max_objectives) as usize;
    let mut random_options: Vec<Objective> = vec![];
    let mut out = vec![];

    'obj: for obj_option in &obj_settings.objective_options {
        // Skip objective option if it does not exist on the map.
        let obj_tiles = get_objective_tiles(&[obj_option.objective]);
        if let Some(map) = map {
            for (room_id, _, _) in obj_tiles {
                let room_ptr = game_data.room_ptr_by_id[&room_id];
                let room_idx = game_data.room_idx_by_ptr[&room_ptr];
                if !map.room_mask[room_idx] {
                    continue 'obj;
                }
            }
        }

        match obj_option.setting {
            ObjectiveSetting::No => {}
            ObjectiveSetting::Maybe => {
                random_options.push(obj_option.objective);
            }
            ObjectiveSetting::Yes => {
                out.push(obj_option.objective);
            }
        }
    }

    if out.len() + random_options.len() < num_objectives {
        out.extend(random_options);
    } else {
        out.extend(random_options.choose_multiple(rng, num_objectives - out.len()));
    }
    out
}

impl<'r> Randomizer<'r> {
    pub fn new<R: Rng>(
        map: &'r Map,
        locked_door_data: &'r LockedDoorData,
        objectives: Vec<Objective>,
        settings: &'r RandomizerSettings,
        difficulty_tiers: &'r [DifficultyConfig],
        game_data: &'r GameData,
        base_links_data: &'r LinksDataGroup,
        _rng: &mut R,
    ) -> Randomizer<'r> {
        let mut available_items: usize = 0;
        for (room_id, _) in &game_data.item_locations {
            let room_idx = game_data.room_idx_by_id[room_id];
            if map.room_mask[room_idx] {
                available_items += 1;
            }
        }

        let preprocessor = Preprocessor::new(game_data, map, &difficulty_tiers[0]);
        let preprocessed_seed_links: Vec<Link> = preprocessor.get_all_door_links();
        info!(
            "{} base links, {} door links",
            base_links_data.links.len(),
            preprocessed_seed_links.len()
        );

        let mut initial_items_remaining: Vec<usize> = vec![1; game_data.item_isv.keys.len()];
        initial_items_remaining[Item::Nothing as usize] = 0;
        for x in &settings.item_progression_settings.item_pool {
            initial_items_remaining[x.item as usize] = x.count;
        }
        if settings.other_settings.wall_jump == WallJump::Vanilla {
            initial_items_remaining[Item::WallJump as usize] = 0;
        }

        let mut minimal_tank_count = get_minimal_tank_count(&difficulty_tiers[0]);
        for x in &settings.item_progression_settings.starting_items {
            initial_items_remaining[x.item as usize] -=
                usize::min(x.count, initial_items_remaining[x.item as usize]);
            if x.item == Item::ETank || x.item == Item::ReserveTank {
                minimal_tank_count = minimal_tank_count.saturating_sub(x.count);
            }
        }

        while initial_items_remaining[Item::ETank as usize]
            + initial_items_remaining[Item::ReserveTank as usize]
            < minimal_tank_count
        {
            initial_items_remaining[Item::ETank as usize] += 1;
        }

        let target_initial_items = initial_items_remaining.clone();
        let ammo_shortage_weight: Vec<(Item, f32)> = vec![
            (Item::Missile, 0.12),
            (Item::PowerBomb, 1.0),
            (Item::Super, 1.0),
        ];
        let tank_shortage_weight: Vec<(Item, f32)> =
            vec![(Item::ETank, 0.7), (Item::ReserveTank, 2.0)];
        for _ in 0..initial_items_remaining
            .iter()
            .sum::<usize>()
            .saturating_sub(available_items)
        {
            let tank_count = initial_items_remaining[Item::ETank as usize]
                + initial_items_remaining[Item::ReserveTank as usize];
            let mut removal_options = ammo_shortage_weight.clone();
            if tank_count > minimal_tank_count {
                removal_options.extend(tank_shortage_weight.clone());
            }
            let mut best_removal_item: Option<Item> = None;
            let mut best_removal_cost: f32 = f32::MAX;
            for (item, weight) in removal_options {
                if initial_items_remaining[item as usize] == 0 {
                    continue;
                }
                let gap =
                    target_initial_items[item as usize] - initial_items_remaining[item as usize];
                let cost = (gap + 1) as f32 * weight;
                if cost < best_removal_cost {
                    best_removal_cost = cost;
                    best_removal_item = Some(item);
                }
            }
            if let Some(item) = best_removal_item {
                initial_items_remaining[item as usize] -= 1;
            } else {
                break;
            }
        }

        if initial_items_remaining.iter().sum::<usize>() > available_items {
            // TODO: fail more gracefully
            panic!("Not enough available item locations: {available_items}");
        }

        initial_items_remaining[Item::Nothing as usize] =
            available_items - initial_items_remaining.iter().sum::<usize>();

        let toilet_intersections = Self::get_toilet_intersections(map, game_data);

        let filler_priority_map: HashMap<Item, FillerItemPriority> = settings
            .item_progression_settings
            .filler_items
            .iter()
            .map(|x| (x.item, x.priority))
            .collect();

        let mut item_areas: Vec<AreaIdx> = Vec::new();
        for &(room_id, _) in &game_data.item_locations {
            let room_ptr = game_data.room_ptr_by_id[&room_id];
            let room_idx = game_data.room_idx_by_ptr[&room_ptr];
            let area = map.area[room_idx];
            item_areas.push(area);
        }

        Randomizer {
            map,
            door_map: preprocessor.door_map,
            item_areas,
            toilet_intersections,
            locked_door_data,
            initial_items_remaining,
            game_data,
            settings,
            objectives,
            filler_priority_map,
            item_priority_groups: get_item_priorities(
                &settings.item_progression_settings.key_item_priority,
            ),
            base_links_data,
            seed_links_data: LinksDataGroup::new(
                preprocessed_seed_links,
                game_data.vertex_isv.keys.len(),
                base_links_data.links.len(),
            ),
            difficulty_tiers,
            next_traversal_number: RefCell::new(0),
        }
    }

    pub fn get_toilet_intersections(map: &Map, game_data: &GameData) -> Vec<RoomGeometryRoomIdx> {
        let mut out = vec![];
        if !map.room_mask[game_data.toilet_room_idx] {
            return out;
        }
        let toilet_pos = map.rooms[game_data.toilet_room_idx];
        for room_idx in 0..map.rooms.len() {
            if !map.room_mask[room_idx] {
                continue;
            }
            if room_idx == game_data.toilet_room_idx {
                continue;
            }
            let room_map = &game_data.room_geometry[room_idx].map;
            let room_pos = map.rooms[room_idx];
            let room_height = room_map.len() as isize;
            let room_width = room_map[0].len() as isize;
            let rel_pos_x = (toilet_pos.0 as isize) - (room_pos.0 as isize);
            let rel_pos_y = (toilet_pos.1 as isize) - (room_pos.1 as isize);

            if rel_pos_x >= 0 && rel_pos_x < room_width {
                for y in 2..8 {
                    let y1 = rel_pos_y + y;
                    if y1 >= 0 && y1 < room_height && room_map[y1 as usize][rel_pos_x as usize] == 1
                    {
                        out.push(room_idx);
                        break;
                    }
                }
            }
        }
        out
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
        if self.settings.quality_of_life_settings.all_items_spawn {
            let all_items_spawn_idx = self.game_data.flag_isv.index_by_key["f_AllItemsSpawn"];
            flag_vec[all_items_spawn_idx] = true;
        }
        if self.settings.quality_of_life_settings.acid_chozo {
            let acid_chozo_without_space_jump_idx =
                self.game_data.flag_isv.index_by_key["f_AcidChozoWithoutSpaceJump"];
            flag_vec[acid_chozo_without_space_jump_idx] = true;
        }
        flag_vec
    }

    pub fn update_reachability(
        &self,
        state: &mut RandomizationState,
        traverser_pair: &mut TraverserPair,
    ) {
        traverser_pair.forward.traverse(
            self.base_links_data,
            &self.seed_links_data,
            &state.global_state,
            self.settings,
            &self.difficulty_tiers[0],
            self.game_data,
            &self.door_map,
            self.locked_door_data,
            &self.objectives,
            state.step_num,
        );
        traverser_pair.reverse.traverse(
            self.base_links_data,
            &self.seed_links_data,
            &state.global_state,
            self.settings,
            &self.difficulty_tiers[0],
            self.game_data,
            &self.door_map,
            self.locked_door_data,
            &self.objectives,
            state.step_num,
        );
        let forward = &traverser_pair.forward;
        let reverse = &traverser_pair.reverse;
        let traversal_num = forward.past_steps.len() - 1;
        for (i, vertex_ids) in self.game_data.item_vertex_ids.iter().enumerate() {
            for &v in vertex_ids {
                if !forward.lsr[v].local.is_empty() {
                    if state.item_location_state[i].reachable_traversal.is_none() {
                        state.item_location_state[i].reachable_traversal = Some(traversal_num);
                    }
                    if state.item_location_state[i].bireachable_traversal.is_none()
                        && get_bireachable_idxs(&state.global_state, v, forward, reverse).is_some()
                    {
                        state.item_location_state[i].bireachable_traversal = Some(traversal_num);
                        state.item_location_state[i].bireachable_vertex_id = Some(v);
                    }
                }
            }
        }
        for (i, vertex_ids) in self.game_data.flag_vertex_ids.iter().enumerate() {
            for &v in vertex_ids {
                if !forward.lsr[v].local.is_empty() {
                    if state.flag_location_state[i].reachable_traversal.is_none() {
                        state.flag_location_state[i].reachable_traversal = Some(traversal_num);
                        state.flag_location_state[i].reachable_vertex_id = Some(v);
                    }
                    if state.flag_location_state[i].bireachable_traversal.is_none()
                        && get_bireachable_idxs(&state.global_state, v, forward, reverse).is_some()
                    {
                        state.flag_location_state[i].bireachable_traversal = Some(traversal_num);
                        state.flag_location_state[i].bireachable_vertex_id = Some(v);
                    }
                }
            }
        }
        for (i, vertex_ids) in self
            .locked_door_data
            .locked_door_vertex_ids
            .iter()
            .enumerate()
        {
            for &v in vertex_ids {
                if !forward.lsr[v].local.is_empty()
                    && state.door_state[i].bireachable_traversal.is_none()
                    && get_bireachable_idxs(&state.global_state, v, forward, reverse).is_some()
                {
                    state.door_state[i].bireachable_traversal = Some(traversal_num);
                    state.door_state[i].bireachable_vertex_id = Some(v);
                }
            }
        }
        for (i, (room_id, node_id)) in self.game_data.save_locations.iter().enumerate() {
            let vertex_id = self.game_data.vertex_isv.index_by_key[&VertexKey {
                room_id: *room_id,
                node_id: *node_id,
                obstacle_mask: 0,
                actions: vec![],
            }];
            if state.save_location_state[i].bireachable_traversal.is_none()
                && get_bireachable_idxs(&state.global_state, vertex_id, forward, reverse).is_some()
            {
                state.save_location_state[i].bireachable_traversal = Some(traversal_num);
            }
        }

        // TODO: Update the starting local state for later iterations, based on available farming:
        // state.starting_local_state = self.get_initial_local_state(state);
    }

    // Determine how many key items vs. filler items to place on this step.
    fn determine_item_split(
        &self,
        state: &RandomizationState,
        num_bireachable: usize,
        num_oneway_reachable: usize,
    ) -> (usize, usize) {
        let num_items_to_place = num_bireachable + num_oneway_reachable;
        let filtered_item_precedence: Vec<Item> = state
            .item_precedence
            .iter()
            .copied()
            .filter(|&item| {
                state.items_remaining[item as usize] == self.initial_items_remaining[item as usize]
            })
            .collect();
        let num_key_items_remaining = filtered_item_precedence.len();
        let num_items_remaining: usize = state.items_remaining.iter().sum();
        let mut num_key_items_to_place =
            match self.settings.item_progression_settings.progression_rate {
                ProgressionRate::Slow => 1,
                ProgressionRate::Uniform => usize::max(
                    1,
                    f32::round(
                        (num_key_items_remaining as f32) / (num_items_remaining as f32)
                            * (num_items_to_place as f32),
                    ) as usize,
                ),
                ProgressionRate::Fast => usize::max(
                    1,
                    f32::round(
                        2.0 * (num_key_items_remaining as f32) / (num_items_remaining as f32)
                            * (num_items_to_place as f32),
                    ) as usize,
                ),
            };

        // If we're at the end, dump as many key items as possible:
        if !self
            .settings
            .item_progression_settings
            .stop_item_placement_early
            && num_items_remaining < num_items_to_place + KEY_ITEM_FINISH_THRESHOLD
        {
            num_key_items_to_place = num_key_items_remaining;
        }

        // But we can't place more key items than we have unfilled bireachable item locations:
        num_key_items_to_place = min(
            num_key_items_to_place,
            min(num_bireachable, num_key_items_remaining),
        );

        let num_filler_items_to_place = num_items_to_place - num_key_items_to_place;

        (num_key_items_to_place, num_filler_items_to_place)
    }

    fn select_filler_items<R: Rng>(
        &self,
        state: &RandomizationState,
        num_bireachable_filler_items_to_select: usize,
        num_one_way_reachable_filler_items_to_select: usize,
        rng: &mut R,
    ) -> Vec<Item> {
        // In the future we might do something different with how bireachable locations are filled vs. one-way,
        // but for now they are just lumped together:
        let num_filler_items_to_select =
            num_bireachable_filler_items_to_select + num_one_way_reachable_filler_items_to_select;
        let expansion_item_set: HashSet<Item> =
            [Item::ETank, Item::ReserveTank, Item::Super, Item::PowerBomb]
                .into_iter()
                .collect();
        let mut item_types_to_prioritize: Vec<Item> = vec![];
        let mut item_types_to_mix: Vec<Item> = vec![Item::Missile, Item::Nothing];
        let mut item_types_to_delay: Vec<Item> = vec![];
        let mut item_types_to_extra_delay: Vec<Item> = vec![];
        let mut early_filler_slots_remaining = num_bireachable_filler_items_to_select;

        for &item in &state.item_precedence {
            if item == Item::Missile
                || item == Item::Nothing
                || state.items_remaining[item as usize] == 0
            {
                continue;
            }
            let filler_type = self.filler_priority_map[&item];
            if filler_type == FillerItemPriority::Early
                && !state.global_state.inventory.items[item as usize]
                && early_filler_slots_remaining > 0
            {
                item_types_to_prioritize.push(item);
                item_types_to_mix.push(item);
                early_filler_slots_remaining -= 1;
            } else if filler_type == FillerItemPriority::Early
                || filler_type == FillerItemPriority::Yes
                || (filler_type == FillerItemPriority::Semi
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

        let mut items_to_mix: Vec<Item> = Vec::new();
        for &item in &item_types_to_mix {
            let mut cnt = state.items_remaining[item as usize];
            if item_types_to_prioritize.contains(&item) {
                cnt -= 1;
            }
            for _ in 0..cnt {
                items_to_mix.push(item);
            }
        }
        let mut items_to_delay: Vec<Item> = Vec::new();
        for &item in &item_types_to_delay {
            for _ in 0..state.items_remaining[item as usize] {
                items_to_delay.push(item);
            }
        }
        let mut items_to_extra_delay: Vec<Item> = Vec::new();
        for &item in &item_types_to_extra_delay {
            for _ in 0..state.items_remaining[item as usize] {
                if self
                    .settings
                    .item_progression_settings
                    .stop_item_placement_early
                {
                    // When using "Stop item placement early", place extra Nothing items rather than dumping key items.
                    // It could sometimes result in failure due to not leaving enough places to put needed key items,
                    // but this is an acceptable risk and shouldn't happen too often.
                    items_to_extra_delay.push(Item::Nothing);
                } else {
                    items_to_extra_delay.push(item);
                }
            }
        }
        items_to_mix.shuffle(rng);
        let mut items_to_place: Vec<Item> = item_types_to_prioritize;
        items_to_place.extend(items_to_mix);
        items_to_place.extend(items_to_delay);
        items_to_place.extend(items_to_extra_delay);
        if self.settings.item_progression_settings.spazer_before_plasma {
            self.apply_spazer_plasma_priority(&mut items_to_place);
        }
        items_to_place = items_to_place[0..num_filler_items_to_select].to_vec();
        items_to_place
    }

    fn select_key_items(
        &self,
        state: &RandomizationState,
        num_key_items_to_select: usize,
        attempt_num: usize,
    ) -> Option<Vec<Item>> {
        if num_key_items_to_select >= 1 {
            let mut unplaced_items: Vec<Item> = vec![];
            let mut placed_items: Vec<Item> = vec![];
            let mut additional_items: Vec<Item> = vec![];

            for &item in &state.item_precedence {
                if state.items_remaining[item as usize] > 0
                    || (self
                        .settings
                        .item_progression_settings
                        .stop_item_placement_early
                        && item == Item::Nothing)
                {
                    if self.settings.item_progression_settings.progression_rate
                        == ProgressionRate::Slow
                    {
                        // With Slow progression, items that have been placed before (e.g. an ETank) are treated like any other
                        // item, still keeping their same position in the key item priority
                        unplaced_items.push(item);
                    } else {
                        // With Uniform and Fast progression, items that have been placed before get put in last priority:
                        if state.items_remaining[item as usize]
                            == self.initial_items_remaining[item as usize]
                        {
                            unplaced_items.push(item);
                        } else {
                            placed_items.push(item);
                        }
                    }

                    if state.items_remaining[item as usize] >= 2 {
                        let cnt = state.items_remaining[item as usize] - 1;
                        for _ in 0..cnt {
                            additional_items.push(item);
                        }
                    }
                }
            }

            let cnt_different_items_remaining = unplaced_items.len() + placed_items.len();
            let mut remaining_items: Vec<Item> = vec![];
            remaining_items.extend(unplaced_items);
            remaining_items.extend(placed_items);
            remaining_items.extend(additional_items);

            if attempt_num > 0
                && num_key_items_to_select - 1 + attempt_num >= cnt_different_items_remaining
            {
                return None;
            }

            // If we will be placing `k` key items, we let the first `k - 1` items to place remain fixed based on the
            // item precedence order, while we vary the last key item across attempts (to try to find some choice that
            // will expand the set of bireachable item locations).
            let mut key_items_to_place: Vec<Item> = vec![];
            key_items_to_place.extend(remaining_items[0..(num_key_items_to_select - 1)].iter());
            key_items_to_place.push(remaining_items[num_key_items_to_select - 1 + attempt_num]);
            assert!(key_items_to_place.len() == num_key_items_to_select);
            Some(key_items_to_place)
        } else if attempt_num > 0 {
            None
        } else {
            Some(vec![])
        }
    }

    fn find_hard_location(
        &self,
        state: &RandomizationState,
        bireachable_locations: &[ItemLocationId],
        traverser: &mut Traverser,
        preferred_areas: &[AreaIdx],
    ) -> (usize, usize) {
        // For forced mode, we prioritize placing a key item at a location that is inaccessible at
        // lower difficulty tiers. This function returns an index into `bireachable_locations`, identifying
        // a location with the hardest possible difficulty to reach.

        for tier in 1..self.difficulty_tiers.len() {
            let difficulty = &self.difficulty_tiers[tier];

            traverser.traverse(
                self.base_links_data,
                &self.seed_links_data,
                &state.global_state,
                self.settings,
                difficulty,
                self.game_data,
                &self.door_map,
                self.locked_door_data,
                &self.objectives,
                0,
            );

            let mut preferred_locs: Vec<usize> = Vec::new();
            let mut other_locs: Vec<usize> = Vec::new();

            for (i, &item_location_id) in bireachable_locations.iter().enumerate() {
                let mut is_reachable = false;
                for &v in &self.game_data.item_vertex_ids[item_location_id] {
                    if !traverser.lsr[v].local.is_empty() {
                        is_reachable = true;
                    }
                }
                if !is_reachable {
                    if self.settings.item_progression_settings.item_placement_style
                        == ItemPlacementStyle::Local
                        && preferred_areas.contains(&self.item_areas[item_location_id])
                    {
                        preferred_locs.push(i);
                    } else {
                        other_locs.push(i);
                    }
                }
            }

            traverser.pop_step();
            if !preferred_locs.is_empty() {
                return (preferred_locs[0], tier - 1);
            } else if !other_locs.is_empty() {
                return (other_locs[0], tier - 1);
            }
        }
        (0, self.difficulty_tiers.len() - 1)
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
        traverser_pair: &TraverserPair,
    ) {
        info!(
            "[attempt {attempt_num_rando}] Placing {key_items_to_place:?}, {other_items_to_place:?}"
        );

        let num_items_remaining: usize = state.items_remaining.iter().sum();
        let num_items_to_place: usize = key_items_to_place.len() + other_items_to_place.len();
        let skip_hard_placement = !self
            .settings
            .item_progression_settings
            .stop_item_placement_early
            && num_items_remaining < num_items_to_place + KEY_ITEM_FINISH_THRESHOLD;

        let mut new_bireachable_locations: Vec<ItemLocationId> = bireachable_locations.to_vec();
        let mut tier_vec: Vec<usize> = vec![];
        if self.difficulty_tiers.len() > 1 && !skip_hard_placement {
            // TODO: avoid doing this clone:
            let mut past_traverser = traverser_pair.forward.clone();
            let step = past_traverser.past_steps.last().unwrap().step_num;
            while past_traverser.past_steps.len() > 1
                && past_traverser.past_steps.last().unwrap().step_num >= step - 1
            {
                past_traverser.pop_step();
            }
            let mut new_key_areas: Vec<AreaIdx> = Vec::new();
            for i in 0..key_items_to_place.len() {
                let (hard_idx, tier) = if key_items_to_place.len() > 1 {
                    // We're placing more than one key item in this step. Obtaining some of them could help make
                    // others easier to obtain. So we use "new_state" to try to find locations that are still hard to
                    // reach even with the new items.
                    self.find_hard_location(
                        new_state,
                        &new_bireachable_locations[i..],
                        &mut past_traverser,
                        &state.last_key_areas,
                    )
                } else {
                    // We're only placing one key item in this step. Try to find a location that is hard to reach
                    // without already having the new item.
                    self.find_hard_location(
                        state,
                        &new_bireachable_locations[i..],
                        &mut past_traverser,
                        &state.last_key_areas,
                    )
                };
                info!(
                    "[attempt {attempt_num_rando}] {:?} in tier {} (of {})",
                    key_items_to_place[i],
                    tier,
                    self.difficulty_tiers.len(),
                );

                let hard_loc = new_bireachable_locations[i + hard_idx];
                new_bireachable_locations.swap(i, i + hard_idx);
                tier_vec.push(tier);

                new_key_areas.push(self.item_areas[hard_loc]);
            }
            new_state.last_key_areas = new_key_areas;
        }

        let mut all_locations: Vec<ItemLocationId> = Vec::new();
        all_locations.extend(new_bireachable_locations);
        all_locations.extend(other_locations);
        let mut all_items_to_place: Vec<Item> = Vec::new();
        all_items_to_place.extend(key_items_to_place);
        all_items_to_place.extend(other_items_to_place);
        assert!(all_locations.len() == all_items_to_place.len());
        for i in 0..all_locations.len() {
            let loc = all_locations[i];
            let item = all_items_to_place[i];
            let tier = if i < tier_vec.len() {
                Some(tier_vec[i])
            } else {
                None
            };
            new_state.item_location_state[loc].placed_item = Some(item);
            new_state.item_location_state[loc].placed_tier = tier;
        }
    }

    fn finish<R: Rng>(
        &self,
        attempt_num_rando: usize,
        state: &mut RandomizationState,
        rng: &mut R,
    ) {
        let mut remaining_items: Vec<Item> = Vec::new();
        for item_id in 0..self.game_data.item_isv.keys.len() {
            for _ in 0..state.items_remaining[item_id] {
                remaining_items.push(Item::try_from(item_id).unwrap());
            }
        }
        if self
            .settings
            .item_progression_settings
            .stop_item_placement_early
        {
            info!("[attempt {attempt_num_rando}] Finishing without {remaining_items:?}");
            for item_loc_state in &mut state.item_location_state {
                if item_loc_state.placed_item.is_none() {
                    item_loc_state.placed_item = Some(Item::Nothing);
                }
            }
        } else {
            info!("[attempt {attempt_num_rando}] Finishing with {remaining_items:?}");
            remaining_items.shuffle(rng);
            let mut idx = 0;
            for (i, item_loc_state) in state.item_location_state.iter_mut().enumerate() {
                let room_id = self.game_data.item_locations[i].0;
                let room_idx = self.game_data.room_idx_by_id[&room_id];
                if !self.map.room_mask[room_idx] {
                    item_loc_state.placed_item = Some(Item::Nothing);
                }
                if item_loc_state.placed_item.is_none() {
                    item_loc_state.placed_item = Some(remaining_items[idx]);
                    idx += 1;
                }
            }
        }
    }

    fn collect_items(
        &self,
        state: &mut RandomizationState,
        key_items: &[Item],
        filler_items: &[Item],
        placed_uncollected_bireachable_items: &[Item],
        num_unplaced_bireachable: usize,
    ) {
        // Collect all the items that would be collectible in this scenario:
        // 1) Items that were already placed on an earlier step; this is only applicable to filler items,
        // which became one-way reachable on an earlier step but are now bireachable.
        // 2) Key items,
        // 3) Other items
        for &item in placed_uncollected_bireachable_items.iter().chain(
            key_items
                .iter()
                .chain(filler_items.iter())
                .take(num_unplaced_bireachable),
        ) {
            state.global_state.collect(
                item,
                self.game_data,
                self.settings
                    .item_progression_settings
                    .ammo_collect_fraction,
                &self.difficulty_tiers[0].tech,
            );
        }
        // Update `items_remaining` for key items only.
        // This is assumed to have already been done for filler items.
        for &item in key_items {
            if state.items_remaining[item as usize] > 0 {
                state.items_remaining[item as usize] -= 1;
            }
        }
    }

    fn provides_progression(
        &self,
        old_state: &RandomizationState,
        new_state: &RandomizationState,
    ) -> bool {
        let num_bireachable = new_state
            .item_location_state
            .iter()
            .filter(|x| x.bireachable_traversal.is_some())
            .count();
        let num_reachable = new_state
            .item_location_state
            .iter()
            .filter(|x| x.reachable_traversal.is_some())
            .count();
        let num_one_way_reachable = num_reachable - num_bireachable;

        // Maximum acceptable number of one-way-reachable items. This is to try to avoid extreme
        // cases where the player would gain access to very large areas that they cannot return from:
        let one_way_reachable_limit = 20;

        // Check if an instance of each item is already collected.
        let all_items_bireachable = new_state
            .items_remaining
            .iter()
            .zip(self.initial_items_remaining.iter())
            .all(|(n, i)| n < i || *i == 0);

        let gives_expansion = if all_items_bireachable {
            true
        } else {
            iter::zip(
                &new_state.item_location_state,
                &old_state.item_location_state,
            )
            .any(|(n, o)| n.bireachable_traversal.is_some() && o.reachable_traversal.is_none())
        };

        let is_done = self
            .settings
            .item_progression_settings
            .stop_item_placement_early
            && self.is_game_beatable(new_state);

        (num_one_way_reachable < one_way_reachable_limit && gives_expansion) || is_done
    }

    fn get_initial_local_state(
        &self,
        state: &RandomizationState,
        traverser_pair: &TraverserPair,
    ) -> LocalState {
        let start_vertex_id = self.game_data.vertex_isv.index_by_key[&VertexKey {
            room_id: state.hub_location.room_id,
            node_id: state.hub_location.node_id,
            obstacle_mask: 0,
            actions: vec![],
        }];
        let cost_metric_idx = 0; // use energy-sensitive cost metric
        let i = traverser_pair.forward.lsr[start_vertex_id].best_cost_idxs[cost_metric_idx];
        traverser_pair.forward.lsr[start_vertex_id].local[i as usize]
    }

    fn multi_attempt_select_items<R: Rng + Clone>(
        &self,
        attempt_num_rando: usize,
        state: &RandomizationState,
        placed_uncollected_bireachable_items: &[Item],
        num_unplaced_bireachable: usize,
        num_unplaced_oneway_reachable: usize,
        rng: &mut R,
        traverser_pair: &mut TraverserPair,
    ) -> Result<(SelectItemsOutput, RandomizationState)> {
        let (num_key_items_to_select, num_filler_items_to_select) = self.determine_item_split(
            state,
            num_unplaced_bireachable,
            num_unplaced_oneway_reachable,
        );
        let num_bireachable_filler_items_to_select =
            num_unplaced_bireachable - num_key_items_to_select;
        let num_one_way_reachable_filler_items_to_select =
            num_filler_items_to_select - num_bireachable_filler_items_to_select;
        let selected_filler_items = self.select_filler_items(
            state,
            num_bireachable_filler_items_to_select,
            num_one_way_reachable_filler_items_to_select,
            rng,
        );

        let mut new_state_filler: RandomizationState = RandomizationState {
            step_num: state.step_num + 1,
            start_location: state.start_location.clone(),
            hub_location: state.hub_location.clone(),
            item_precedence: state.item_precedence.clone(),
            item_location_state: state.item_location_state.clone(),
            flag_location_state: state.flag_location_state.clone(),
            save_location_state: state.save_location_state.clone(),
            door_state: state.door_state.clone(),
            items_remaining: state.items_remaining.clone(),
            global_state: state.global_state.clone(),
            starting_local_state: self.get_initial_local_state(state, traverser_pair),
            last_key_areas: Vec::new(),
        };
        for &item in &selected_filler_items {
            // We check if items_remaining is positive, only because with "Stop item placement early" there
            // could be extra (unplanned) Nothing items placed.
            if new_state_filler.items_remaining[item as usize] > 0 {
                new_state_filler.items_remaining[item as usize] -= 1;
            }
        }

        let mut attempt_num = 0;
        let mut selected_key_items = self
            .select_key_items(&new_state_filler, num_key_items_to_select, attempt_num)
            .unwrap();
        let num_traversal_steps = traverser_pair.forward.past_steps.len();

        loop {
            assert_eq!(num_traversal_steps, traverser_pair.forward.past_steps.len());
            assert_eq!(num_traversal_steps, traverser_pair.reverse.past_steps.len());
            let mut new_state: RandomizationState = new_state_filler.clone();

            self.collect_items(
                &mut new_state,
                &selected_key_items,
                &selected_filler_items,
                placed_uncollected_bireachable_items,
                num_unplaced_bireachable,
            );
            self.update_reachability(&mut new_state, traverser_pair);

            let provides_progression = self.provides_progression(state, &new_state);
            debug!(
                "[attempt {attempt_num_rando}] items {selected_key_items:?}, {selected_filler_items:?}, provides_progression = {}",
                provides_progression
            );
            if provides_progression {
                let selection = SelectItemsOutput {
                    key_items: selected_key_items,
                    other_items: selected_filler_items,
                };
                return Ok((selection, new_state));
            }
            traverser_pair.forward.pop_step();
            traverser_pair.reverse.pop_step();

            attempt_num += 1;
            if let Some(new_selected_key_items) =
                self.select_key_items(&new_state_filler, num_key_items_to_select, attempt_num)
            {
                selected_key_items = new_selected_key_items;
            } else {
                if self.settings.item_progression_settings.progression_rate == ProgressionRate::Slow
                {
                    info!(
                        "[attempt {attempt_num_rando}] Continuing with last-ditch effort after exhausting key item placement attempts"
                    );
                } else {
                    bail!(
                        "[attempt {attempt_num_rando}] Failing after exhausting key item placement attempts"
                    );
                }
                if self
                    .settings
                    .item_progression_settings
                    .stop_item_placement_early
                {
                    selected_key_items.fill(Item::Nothing);
                    new_state = new_state_filler;
                    self.collect_items(
                        &mut new_state,
                        &selected_key_items,
                        &selected_filler_items,
                        placed_uncollected_bireachable_items,
                        num_unplaced_bireachable,
                    );
                    self.update_reachability(&mut new_state, traverser_pair);
                }
                let selection = SelectItemsOutput {
                    key_items: selected_key_items,
                    other_items: selected_filler_items,
                };
                return Ok((selection, new_state));
            }
        }
    }

    fn step<R: Rng + Clone>(
        &self,
        attempt_num_rando: usize,
        state: &mut RandomizationState,
        traverser_pair: &mut TraverserPair,
        rng: &mut R,
    ) -> Result<bool> {
        loop {
            let mut any_update = false;
            for (i, &flag_id) in self.game_data.flag_ids.iter().enumerate() {
                if state.global_state.flags[flag_id] {
                    continue;
                }
                if state.flag_location_state[i].reachable_traversal.is_some()
                    && flag_id == self.game_data.mother_brain_defeated_flag_id
                {
                    // f_DefeatedMotherBrain flag is special in that we only require one-way reachability for it:
                    any_update = true;
                    state.global_state.flags[flag_id] = true;
                } else if state.flag_location_state[i].bireachable_traversal.is_some() {
                    any_update = true;
                    state.global_state.flags[flag_id] = true;
                }
            }
            for i in 0..self.locked_door_data.locked_doors.len() {
                if state.global_state.doors_unlocked[i] {
                    continue;
                }
                if state.door_state[i].bireachable_traversal.is_some() {
                    any_update = true;
                    state.global_state.doors_unlocked[i] = true;
                }
            }
            if any_update {
                self.update_reachability(state, traverser_pair);
            } else {
                break;
            }
        }

        if self
            .settings
            .item_progression_settings
            .stop_item_placement_early
            && self.is_game_beatable(state)
        {
            info!("Stopping early");
            self.update_reachability(state, traverser_pair);
            return Ok(true);
        }

        let mut placed_uncollected_bireachable_loc: Vec<ItemLocationId> = Vec::new();
        let mut placed_uncollected_bireachable_items: Vec<Item> = Vec::new();
        let mut unplaced_bireachable: Vec<ItemLocationId> = Vec::new();
        let mut unplaced_oneway_reachable: Vec<ItemLocationId> = Vec::new();
        for (i, item_location_state) in state.item_location_state.iter().enumerate() {
            if let Some(item) = item_location_state.placed_item {
                if !item_location_state.collected
                    && item_location_state.bireachable_traversal.is_some()
                {
                    placed_uncollected_bireachable_loc.push(i);
                    placed_uncollected_bireachable_items.push(item);
                }
            } else if item_location_state.bireachable_traversal.is_some() {
                unplaced_bireachable.push(i);
            } else if item_location_state.reachable_traversal.is_some() {
                unplaced_oneway_reachable.push(i);
            }
        }
        unplaced_bireachable.shuffle(rng);
        unplaced_oneway_reachable.shuffle(rng);
        let (selection, mut new_state) = self.multi_attempt_select_items(
            attempt_num_rando,
            state,
            &placed_uncollected_bireachable_items,
            unplaced_bireachable.len(),
            unplaced_oneway_reachable.len(),
            rng,
            traverser_pair,
        )?;

        // Mark the newly collected items that were placed on earlier steps:
        for &loc in &placed_uncollected_bireachable_loc {
            new_state.item_location_state[loc].collected = true;
        }

        // Place the new items:
        // We place items in all newly reachable locations (bireachable as
        // well as one-way-reachable locations). One-way-reachable locations are filled only
        // with filler items, to reduce the possibility of them being usable to break from the
        // intended logical sequence.
        self.place_items(
            attempt_num_rando,
            state,
            &mut new_state,
            &unplaced_bireachable,
            &unplaced_oneway_reachable,
            &selection.key_items,
            &selection.other_items,
            traverser_pair,
        );

        // Mark the newly placed bireachable items as collected:
        for &loc in &unplaced_bireachable {
            new_state.item_location_state[loc].collected = true;
        }

        *state = new_state;
        Ok(false)
    }

    fn get_seed_name(&self, seed: usize) -> String {
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut rng_seed = [0u8; 32];
        rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
        rng_seed[8..24].copy_from_slice(&t.to_le_bytes());
        let mut rng = rand::rngs::StdRng::from_seed(rng_seed);
        // Leave out vowels and characters that could read like vowels, to minimize the chance
        // of forming words.
        let alphabet = "256789BCDFGHJKLMNPQRSTVWXYZbcdfghjkmnpqrstvwxyz";
        let mut out: String = String::new();
        let num_chars = 9;
        for _ in 0..num_chars {
            let i = rng.gen_range(0..alphabet.len());
            let c = alphabet.as_bytes()[i] as char;
            out.push(c);
        }
        out
    }

    fn get_essential_spoiler_data(
        &self,
        settings: &RandomizerSettings,
        spoiler_log: &SpoilerLog,
    ) -> EssentialSpoilerData {
        let mut item_spoiler_info: Vec<EssentialItemSpoilerInfo> = vec![];
        let mut items_set: HashSet<Item> = HashSet::new();

        // Include starting items first, as "step 0":
        for x in &settings.item_progression_settings.starting_items {
            if x.count > 0 {
                // Skip WallJump if it appears in the starting items without Collectible wall jump
                if x.item == Item::WallJump
                    && settings.other_settings.wall_jump == WallJump::Vanilla
                {
                    continue;
                }
                item_spoiler_info.push(EssentialItemSpoilerInfo {
                    item: x.item,
                    step: Some(0),
                    area: None,
                });
                items_set.insert(x.item);
            }
        }

        // Include collectible items in the middle:
        for (step, step_summary) in spoiler_log.summary.iter().enumerate() {
            for item_info in step_summary.items.iter() {
                let item = Item::try_from(item_info.item.as_str()).unwrap();
                if !items_set.contains(&item) {
                    item_spoiler_info.push(EssentialItemSpoilerInfo {
                        item,
                        step: Some(step + 1),
                        area: Some(item_info.location.area.clone()),
                    });
                    items_set.insert(item);
                }
            }
        }

        // Include logically uncollectible items:
        for loc in &spoiler_log.all_items {
            if loc.item == "Nothing" {
                continue;
            }
            let item = Item::try_from(loc.item.as_str()).unwrap();
            if !items_set.contains(&item) {
                item_spoiler_info.push(EssentialItemSpoilerInfo {
                    item,
                    step: None,
                    area: Some(loc.location.area.clone()),
                });
                items_set.insert(item);
            }
        }

        // Include unplaced items at the end:
        for &name in Item::VARIANTS {
            if name == "Nothing" {
                continue;
            }
            if settings.other_settings.wall_jump != WallJump::Collectible && name == "WallJump" {
                // Don't show "WallJump" item unless using Collectible mode.
                continue;
            }
            let item = Item::try_from(name).unwrap();
            if !items_set.contains(&item) {
                item_spoiler_info.push(EssentialItemSpoilerInfo {
                    item,
                    step: None,
                    area: None,
                });
                items_set.insert(item);
            }
        }

        EssentialSpoilerData { item_spoiler_info }
    }

    pub fn rebuild_step(&self, state: &RandomizationState, traverser: &mut Traverser) {
        let num_reachable0 = traverser.lsr.iter().filter(|x| !x.local.is_empty()).count();
        traverser.unfinish_step();

        for (i, lsr) in &mut traverser.lsr.iter_mut().enumerate() {
            traverser.step.updates.push(TraversalUpdate {
                vertex_id: i,
                old_lsr: lsr.clone(),
            });
            *lsr = LocalStateReducer::default();
        }

        let start_vertex_id = self.game_data.vertex_isv.index_by_key[&VertexKey {
            room_id: state.hub_location.room_id,
            node_id: state.hub_location.node_id,
            obstacle_mask: 0,
            actions: vec![],
        }];
        let global = traverser.step.global_state.clone();
        traverser.traverse_dijkstra(
            self.base_links_data,
            &self.seed_links_data,
            &global,
            self.settings,
            &self.difficulty_tiers[0],
            self.game_data,
            &self.door_map,
            self.locked_door_data,
            &self.objectives,
            start_vertex_id,
            traverser.step.step_num,
        );
        let num_reachable1 = traverser.lsr.iter().filter(|x| !x.local.is_empty()).count();
        debug!(
            "Rebuilt step {}, traversal {} (reverse={}), reachable {} -> {}",
            traverser.past_steps.last().unwrap().step_num,
            traverser.past_steps.len() - 1,
            traverser.reverse,
            num_reachable0,
            num_reachable1,
        );
    }

    pub fn get_randomization<R: Rng>(
        &self,
        state: &RandomizationState,
        seed: usize,
        display_seed: usize,
        rng: &mut R,
        traverser_pair: &mut TraverserPair,
        start_location_data: &StartLocationData,
    ) -> Result<(Randomization, SpoilerLog)> {
        let save_animals = if self.settings.save_animals == SaveAnimals::Random {
            if rng.gen_bool(0.5) {
                SaveAnimals::Yes
            } else {
                SaveAnimals::No
            }
        } else {
            self.settings.save_animals
        };

        let spoiler_log = get_spoiler_log(
            self,
            state,
            traverser_pair,
            save_animals,
            start_location_data,
        )?;

        let item_placement: Vec<Item> = state
            .item_location_state
            .iter()
            .map(|x| x.placed_item.unwrap())
            .collect();

        let randomization = Randomization {
            objectives: self.objectives.clone(),
            save_animals,
            map: self.map.clone(),
            toilet_intersections: self.toilet_intersections.clone(),
            locked_doors: self.locked_door_data.locked_doors.clone(),
            item_placement,
            escape_time_seconds: spoiler_log.escape.final_time_seconds,
            essential_spoiler_data: self.get_essential_spoiler_data(self.settings, &spoiler_log),
            seed,
            display_seed,
            seed_name: self.get_seed_name(seed),
            start_location: state.start_location.clone(),
        };
        Ok((randomization, spoiler_log))
    }

    fn get_item_precedence<R: Rng>(
        &self,
        item_priorities: &[ItemPriorityGroup],
        item_priority_strength: ItemPriorityStrength,
        rng: &mut R,
    ) -> Vec<Item> {
        let mut item_precedence: Vec<Item> = Vec::new();
        if self.settings.item_progression_settings.progression_rate == ProgressionRate::Slow {
            // With slow progression, prioritize placing nothing and missiles over other key items.
            item_precedence.push(Item::Nothing);
            item_precedence.push(Item::Missile);
        }
        match item_priority_strength {
            ItemPriorityStrength::Moderate => {
                assert!(item_priorities.len() == 3);
                let mut items = vec![];
                for (i, priority_group) in item_priorities.iter().enumerate() {
                    for item_name in &priority_group.items {
                        items.push(item_name.clone());
                        if i != 1 {
                            // Include a second copy of Early and Late items:
                            items.push(item_name.clone());
                        }
                    }
                }
                items.shuffle(rng);

                // Remove the later copy of each "Early" item
                items = remove_some_duplicates(
                    &items,
                    &item_priorities[0].items.iter().cloned().collect(),
                );

                // Remove the earlier copy of each "Late" item
                items.reverse();
                items = remove_some_duplicates(
                    &items,
                    &item_priorities[2].items.iter().cloned().collect(),
                );
                items.reverse();

                for item_name in &items {
                    let item_idx = self.game_data.item_isv.index_by_key[item_name];
                    item_precedence.push(Item::try_from(item_idx).unwrap());
                }
            }
            ItemPriorityStrength::Heavy => {
                for priority_group in item_priorities {
                    let mut items = priority_group.items.clone();
                    items.shuffle(rng);
                    for item_name in &items {
                        let item_idx = self.game_data.item_isv.index_by_key[item_name];
                        item_precedence.push(Item::try_from(item_idx).unwrap());
                    }
                }
            }
        }
        if self.settings.item_progression_settings.progression_rate != ProgressionRate::Slow {
            // With Normal and Uniform progression, prioritize all other key items over missiles
            // and nothing.
            item_precedence.push(Item::Missile);
            item_precedence.push(Item::Nothing);
        }
        item_precedence
    }

    fn rerandomize_tank_precedence<R: Rng>(&self, item_precedence: &mut [Item], rng: &mut R) {
        if rng.gen_bool(0.5) {
            return;
        }
        let Some(etank_idx) = item_precedence.iter().position(|&x| x == Item::ETank) else {
            return;
        };
        let Some(reserve_idx) = item_precedence.iter().position(|&x| x == Item::ReserveTank) else {
            return;
        };
        item_precedence[etank_idx] = Item::ReserveTank;
        item_precedence[reserve_idx] = Item::ETank;
    }

    fn apply_spazer_plasma_priority(&self, item_precedence: &mut [Item]) {
        let spazer_idx_opt = item_precedence.iter().position(|&x| x == Item::Spazer);
        let plasma_idx_opt = item_precedence.iter().position(|&x| x == Item::Plasma);
        if spazer_idx_opt.is_none() || plasma_idx_opt.is_none() {
            return;
        }
        let spazer_idx = spazer_idx_opt.unwrap();
        let plasma_idx = plasma_idx_opt.unwrap();
        if plasma_idx < spazer_idx {
            item_precedence[plasma_idx] = Item::Spazer;
            item_precedence[spazer_idx] = Item::Plasma;
        }
    }

    pub fn determine_start_location<R: Rng>(
        &self,
        attempt_num_rando: usize,
        num_attempts: usize,
        rng: &mut R,
        traverser_pair: &mut TraverserPair,
    ) -> Result<StartLocationData> {
        let cost_config = simple_cost_config();

        if self.settings.start_location_settings.mode == StartLocationMode::Ship {
            let ship_start = StartLocation {
                name: "Ship".to_string(),
                room_id: 8,
                node_id: 5,
                door_load_node_id: Some(2),
                x: 72.0,
                y: 69.5,
                ..StartLocation::default()
            };

            let ship_hub = HubLocation {
                room_id: 8,
                node_id: 5,
                ..HubLocation::default()
            };

            return Ok(StartLocationData {
                start_location: ship_start,
                hub_location: ship_hub,
                hub_obtain_route: vec![],
                hub_return_route: vec![],
            });
        }

        for i in 0..num_attempts {
            info!("[attempt {attempt_num_rando}] start location attempt {i}");
            let start_loc_idx = match self.settings.start_location_settings.mode {
                StartLocationMode::Random => rng.gen_range(0..self.game_data.start_locations.len()),
                StartLocationMode::Custom => {
                    let mut idx: Option<usize> = None;
                    let room_id = self
                        .settings
                        .start_location_settings
                        .room_id
                        .context("expected room_id")?;
                    let node_id = self
                        .settings
                        .start_location_settings
                        .node_id
                        .context("expected node_id")?;
                    for (j, loc) in self.game_data.start_locations.iter().enumerate() {
                        if loc.room_id == room_id && loc.node_id == node_id {
                            idx = Some(j);
                            break;
                        }
                    }
                    if idx.is_none() {
                        bail!("Unknown start location ({}, {})", room_id, node_id);
                    }
                    idx.unwrap()
                }
                _ => panic!(
                    "Unexpected start location mode: {:?}",
                    self.settings.start_location_settings.mode
                ),
            };
            let start_loc = self.game_data.start_locations[start_loc_idx].clone();

            info!("[attempt {attempt_num_rando}] start: {start_loc:?}");
            let start_vertex_id = self.game_data.vertex_isv.index_by_key[&VertexKey {
                room_id: start_loc.room_id,
                node_id: start_loc.node_id,
                obstacle_mask: 0,
                actions: vec![],
            }];
            let (global, _) = self.get_initial_states();
            let local = apply_requirement(
                start_loc.requires_parsed.as_ref().unwrap(),
                &global,
                LocalState::full(false),
                false,
                self.settings,
                &self.difficulty_tiers[0],
                self.game_data,
                &self.door_map,
                self.locked_door_data,
                &self.objectives,
                &cost_config,
            );
            let Some(local) = local else {
                continue;
            };

            traverser_pair
                .forward
                .add_origin(local, &global.inventory, start_vertex_id);
            traverser_pair.forward.traverse(
                self.base_links_data,
                &self.seed_links_data,
                &global,
                self.settings,
                &self.difficulty_tiers[0],
                self.game_data,
                &self.door_map,
                self.locked_door_data,
                &self.objectives,
                0,
            );
            let forward = &traverser_pair.forward;

            let mut has_reachable_item = false;
            for &v in self.game_data.item_vertex_ids.iter().flatten() {
                if !forward.lsr[v].local.is_empty() {
                    has_reachable_item = true;
                }
            }
            if !has_reachable_item {
                traverser_pair.forward.pop_step();
                continue;
            }

            traverser_pair.reverse.add_origin(
                LocalState::full(true),
                &global.inventory,
                start_vertex_id,
            );
            traverser_pair.reverse.traverse(
                self.base_links_data,
                &self.seed_links_data,
                &global,
                self.settings,
                &self.difficulty_tiers[0],
                self.game_data,
                &self.door_map,
                self.locked_door_data,
                &self.objectives,
                0,
            );
            let reverse = &traverser_pair.reverse;

            // For a hub location to be valid for a given start location, there must be a path from the
            // start location to the hub location and back to the starting node. Note that the forward path
            // from start location to the hub location includes initial start location requirements
            // (e.g. including requirements to reach the starting node from the actual start location, which
            // may not be at a node), while the reverse path only needs to go back to the starting node,
            // which is in the same room as the start location but is not necessarily exactly the same.
            // Among the valid hubs, we select one with the best energy farm.
            let mut best_hub_vertex_id: VertexId = start_vertex_id;
            let mut best_hub_cost: Capacity =
                global.inventory.max_energy - 1 + global.inventory.max_reserves;
            for &(hub_vertex_id, ref hub_req) in [(start_vertex_id, Requirement::Free)]
                .iter()
                .chain(self.game_data.hub_farms.iter())
            {
                if get_bireachable_idxs(&global, hub_vertex_id, forward, reverse).is_none() {
                    continue;
                };

                let new_local = apply_requirement(
                    hub_req,
                    &global,
                    LocalState::empty(),
                    false,
                    self.settings,
                    &self.difficulty_tiers[0],
                    self.game_data,
                    &self.door_map,
                    self.locked_door_data,
                    &self.objectives,
                    &cost_config,
                );
                let hub_cost = if let Some(loc) = new_local {
                    loc.energy_missing(&global.inventory, true)
                } else {
                    Capacity::MAX
                };
                if hub_cost < best_hub_cost {
                    best_hub_cost = hub_cost;
                    best_hub_vertex_id = hub_vertex_id;
                }
            }

            let Some((forward_cost_idx, reverse_cost_idx)) =
                get_bireachable_idxs(&global, best_hub_vertex_id, forward, reverse)
            else {
                panic!("inconsistent result from get_bireachable_idxs");
            };

            let vertex_key = self.game_data.vertex_isv.keys[best_hub_vertex_id].clone();
            let hub_location = HubLocation {
                room_id: vertex_key.room_id,
                node_id: vertex_key.node_id,
                vertex_id: best_hub_vertex_id,
            };

            let hub_obtain_trail_ids =
                get_spoiler_trail_ids_by_idx(forward, best_hub_vertex_id, forward_cost_idx);
            let hub_return_trail_ids =
                get_spoiler_trail_ids_by_idx(reverse, best_hub_vertex_id, reverse_cost_idx);

            let hub_obtain_route =
                get_spoiler_route(self, &global, &hub_obtain_trail_ids, forward, false);
            let hub_return_route =
                get_spoiler_route(self, &global, &hub_return_trail_ids, reverse, true);

            traverser_pair.forward.pop_step();
            traverser_pair.reverse.pop_step();

            return Ok(StartLocationData {
                start_location: start_loc,
                hub_location,
                hub_obtain_route,
                hub_return_route,
            });
        }
        bail!("[attempt {attempt_num_rando}] Failed to find start location.")
    }

    fn get_pool_inventory(&self) -> Inventory {
        let acf = self
            .settings
            .item_progression_settings
            .ammo_collect_fraction;
        let collectible_missile_packs = self.initial_items_remaining[Item::Missile as usize];
        let collectible_super_packs = self.initial_items_remaining[Item::Super as usize];
        let collectible_pb_packs = self.initial_items_remaining[Item::PowerBomb as usize];
        let collectible_etanks = self.initial_items_remaining[Item::ETank as usize];
        let collectible_reserve_tanks = self.initial_items_remaining[Item::ReserveTank as usize];
        Inventory {
            items: self
                .initial_items_remaining
                .iter()
                .map(|&x| x > 0)
                .collect(),
            max_energy: (99 + collectible_etanks * 100) as Capacity,
            max_reserves: (collectible_reserve_tanks * 100) as Capacity,
            max_missiles: (acf * collectible_missile_packs as f32).round() as Capacity * 5,
            max_supers: (acf * collectible_super_packs as f32).round() as Capacity * 5,
            max_power_bombs: (acf * collectible_pb_packs as f32).round() as Capacity * 5,
            collectible_missile_packs: collectible_missile_packs as Capacity,
            collectible_super_packs: collectible_super_packs as Capacity,
            collectible_power_bomb_packs: collectible_pb_packs as Capacity,
            collectible_reserve_tanks: collectible_reserve_tanks as Capacity,
        }
    }

    fn get_initial_states(&self) -> (GlobalState, LocalState) {
        let items = vec![false; self.game_data.item_isv.keys.len()];
        let weapon_mask = self
            .game_data
            .get_weapon_mask(&items, &self.difficulty_tiers[0].tech);
        let mut global = GlobalState {
            inventory: Inventory {
                items,
                max_energy: 99,
                max_reserves: 0,
                max_missiles: 0,
                max_supers: 0,
                max_power_bombs: 0,
                collectible_missile_packs: 0,
                collectible_super_packs: 0,
                collectible_power_bomb_packs: 0,
                collectible_reserve_tanks: 0,
            },
            pool_inventory: self.get_pool_inventory(),
            flags: self.get_initial_flag_vec(),
            doors_unlocked: vec![false; self.locked_door_data.locked_doors.len()],
            weapon_mask,
        };
        for x in &self.settings.item_progression_settings.starting_items {
            if x.item == Item::WallJump
                && self.settings.other_settings.wall_jump == WallJump::Vanilla
            {
                continue;
            }
            for _ in 0..x.count {
                global.collect(
                    x.item,
                    self.game_data,
                    self.settings
                        .item_progression_settings
                        .ammo_collect_fraction,
                    &self.difficulty_tiers[0].tech,
                );
            }
        }
        let local = LocalState::empty();
        (global, local)
    }

    pub fn dummy_randomize<R: Rng>(
        &self,
        seed: usize,
        display_seed: usize,
        rng: &mut R,
    ) -> Result<(Randomization, SpoilerLog)> {
        // For the "Escape" start location mode, item placement is irrelevant since you start
        // with all items collected.

        let save_animals = if self.settings.save_animals == SaveAnimals::Random {
            if rng.gen_bool(0.5) {
                SaveAnimals::Yes
            } else {
                SaveAnimals::No
            }
        } else {
            self.settings.save_animals
        };

        let spoiler_escape = escape_timer::compute_escape_data(
            self.game_data,
            self.map,
            self.settings,
            save_animals != SaveAnimals::No,
            &self.difficulty_tiers[0],
        )?;
        let mut spoiler_all_rooms: Vec<SpoilerRoomLoc> = Vec::new();
        for (room_idx, room_coords) in self.map.rooms.iter().enumerate() {
            if !self.map.room_mask[room_idx] {
                continue;
            }
            let room_geom = &self.game_data.room_geometry[room_idx];
            let room_id = self.game_data.room_id_by_ptr[&room_geom.rom_address];
            let room = self.game_data.room_json_map[&room_id]["name"]
                .as_str()
                .unwrap()
                .to_string();
            let height = room_geom.map.len();
            let width = room_geom.map[0].len();
            let map_reachable_step: Vec<Vec<u8>> = vec![vec![255; width]; height];
            let map_bireachable_step: Vec<Vec<u8>> = vec![vec![255; width]; height];
            spoiler_all_rooms.push(SpoilerRoomLoc {
                room_id,
                room,
                map: room_geom.map.clone(),
                map_reachable_step,
                map_bireachable_step,
                coords: *room_coords,
            });
        }

        let spoiler_log = SpoilerLog {
            item_priority: vec![],
            summary: vec![],
            objectives: vec![],
            start_location: SpoilerStartLocation {
                room_id: StartLocation::default().room_id,
                name: StartLocation::default().name,
                node_id: StartLocation::default().node_id,
                x: StartLocation::default().x,
                y: StartLocation::default().y,
            },
            hub_location_name: String::new(),
            hub_obtain_route: vec![],
            hub_return_route: vec![],
            escape: spoiler_escape,
            details: vec![],
            all_items: vec![],
            all_rooms: spoiler_all_rooms,
            game_data: get_spoiler_game_data(self),
            forward_traversal: SpoilerTraversal {
                initial_local_state: SpoilerLocalState::new(
                    LocalState::empty(),
                    LocalState::empty(),
                    true,
                ),
                steps: vec![],
                prev_trail_ids: vec![],
                link_idxs: vec![],
                local_states: vec![],
            },
            reverse_traversal: SpoilerTraversal {
                initial_local_state: SpoilerLocalState::new(
                    LocalState::empty(),
                    LocalState::empty(),
                    true,
                ),
                steps: vec![],
                prev_trail_ids: vec![],
                link_idxs: vec![],
                local_states: vec![],
            },
        };

        let randomization = Randomization {
            objectives: self.objectives.clone(),
            save_animals,
            map: self.map.clone(),
            toilet_intersections: self.toilet_intersections.clone(),
            locked_doors: self.locked_door_data.locked_doors.clone(),
            item_placement: vec![Item::Nothing; 100],
            escape_time_seconds: spoiler_log.escape.final_time_seconds,
            essential_spoiler_data: self.get_essential_spoiler_data(self.settings, &spoiler_log),
            seed,
            seed_name: self.get_seed_name(seed),
            display_seed,
            start_location: StartLocation::default(),
        };
        Ok((randomization, spoiler_log))
    }

    fn is_game_beatable(&self, state: &RandomizationState) -> bool {
        for (i, &flag_id) in self.game_data.flag_ids.iter().enumerate() {
            if flag_id == self.game_data.mother_brain_defeated_flag_id
                && state.flag_location_state[i].reachable_traversal.is_some()
            {
                return true;
            }
        }
        false
    }

    pub fn randomize(
        &self,
        attempt_num_rando: usize,
        seed: usize,
        display_seed: usize,
    ) -> Result<(Randomization, SpoilerLog)> {
        let mut rng_seed = [0u8; 32];
        rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
        let mut rng = rand::rngs::StdRng::from_seed(rng_seed);
        if self.settings.start_location_settings.mode == StartLocationMode::Escape {
            return self.dummy_randomize(seed, display_seed, &mut rng);
        }
        let (initial_global_state, initial_local_state) = self.get_initial_states();
        let initial_item_location_state = ItemLocationState {
            placed_item: None,
            placed_tier: None,
            collected: false,
            reachable_traversal: None,
            bireachable_traversal: None,
            bireachable_vertex_id: None,
            difficulty_tier: None,
        };
        let initial_flag_location_state = FlagLocationState {
            reachable_traversal: None,
            reachable_vertex_id: None,
            bireachable_traversal: None,
            bireachable_vertex_id: None,
        };
        let initial_save_location_state = SaveLocationState {
            bireachable_traversal: None,
        };
        let initial_door_state = DoorState {
            bireachable_traversal: None,
            bireachable_vertex_id: None,
        };
        let num_attempts_start_location = if self.game_data.start_locations.len() > 1
            && self.settings.start_location_settings.mode != StartLocationMode::Custom
        {
            10
        } else {
            1
        };
        let num_vertices = self.game_data.vertex_isv.keys.len();
        let mut traverser_pair = TraverserPair {
            forward: Traverser::new(
                num_vertices,
                false,
                initial_local_state,
                &initial_global_state,
            ),
            reverse: Traverser::new(
                num_vertices,
                true,
                initial_local_state,
                &initial_global_state,
            ),
        };
        let start_location_data = self.determine_start_location(
            attempt_num_rando,
            num_attempts_start_location,
            &mut rng,
            &mut traverser_pair,
        )?;
        let mut item_precedence: Vec<Item> = self.get_item_precedence(
            &self.item_priority_groups,
            self.settings
                .item_progression_settings
                .item_priority_strength,
            &mut rng,
        );
        if self.settings.item_progression_settings.spazer_before_plasma {
            self.apply_spazer_plasma_priority(&mut item_precedence);
        }
        info!("[attempt {attempt_num_rando}] Item precedence: {item_precedence:?}");
        let mut state = RandomizationState {
            step_num: 1,
            item_precedence,
            start_location: start_location_data.start_location.clone(),
            hub_location: start_location_data.hub_location.clone(),
            item_location_state: vec![
                initial_item_location_state;
                self.game_data.item_locations.len()
            ],
            flag_location_state: vec![initial_flag_location_state; self.game_data.flag_ids.len()],
            save_location_state: vec![
                initial_save_location_state;
                self.game_data.save_locations.len()
            ],
            door_state: vec![initial_door_state; self.locked_door_data.locked_doors.len()],
            items_remaining: self.initial_items_remaining.clone(),
            starting_local_state: initial_local_state,
            global_state: initial_global_state,
            last_key_areas: Vec::new(),
        };
        let start_vertex_id = self.game_data.vertex_isv.index_by_key[&VertexKey {
            room_id: state.hub_location.room_id,
            node_id: state.hub_location.node_id,
            obstacle_mask: 0,
            actions: vec![],
        }];
        traverser_pair.forward.add_origin(
            initial_local_state,
            &state.global_state.inventory,
            start_vertex_id,
        );
        traverser_pair.forward.finish_step(1);
        traverser_pair.reverse.add_origin(
            LocalState::full(true),
            &state.global_state.inventory,
            start_vertex_id,
        );
        traverser_pair.reverse.finish_step(1);
        self.update_reachability(&mut state, &mut traverser_pair);
        if !state
            .item_location_state
            .iter()
            .any(|x| x.bireachable_traversal.is_some())
        {
            bail!("[attempt {attempt_num_rando}] No initially bireachable item locations");
        }
        loop {
            if self.settings.item_progression_settings.random_tank {
                self.rerandomize_tank_precedence(&mut state.item_precedence, &mut rng);
            }
            let last_cnt_bireachable = state
                .item_location_state
                .iter()
                .filter(|x| x.bireachable_traversal.is_some())
                .count();
            let last_cnt_flag_bireachable = state
                .flag_location_state
                .iter()
                .filter(|x| x.bireachable_traversal.is_some())
                .count();
            let is_early_stop =
                self.step(attempt_num_rando, &mut state, &mut traverser_pair, &mut rng)?;
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
                .filter(|x| x.reachable_traversal.is_some())
                .count();
            let cnt_bireachable = state
                .item_location_state
                .iter()
                .filter(|x| x.bireachable_traversal.is_some())
                .count();
            let cnt_flag_bireachable = state
                .flag_location_state
                .iter()
                .filter(|x| x.bireachable_traversal.is_some())
                .count();
            info!(
                "[attempt {attempt_num_rando}] step={0}, bireachable={cnt_bireachable}, reachable={cnt_reachable}, placed={cnt_placed}, collected={cnt_collected}",
                state.step_num
            );

            let any_progress = cnt_bireachable > last_cnt_bireachable
                || cnt_flag_bireachable > last_cnt_flag_bireachable;

            if is_early_stop {
                break;
            }

            if !any_progress {
                // No further progress was made on the last step. So we are done with this attempt: either we have
                // succeeded or we have failed.

                if !self.is_game_beatable(&state) {
                    bail!("[attempt {attempt_num_rando}] Attempt failed: Game not beatable");
                }

                if !self
                    .settings
                    .item_progression_settings
                    .stop_item_placement_early
                {
                    // Check that at least one instance of each item can be collected.
                    for i in 0..self.initial_items_remaining.len() {
                        if self.initial_items_remaining[i] > 0
                            && !state.global_state.inventory.items[i]
                        {
                            bail!(
                                "[attempt {attempt_num_rando}] Attempt failed: Key items not all collectible, missing {:?}",
                                Item::try_from(i).unwrap()
                            );
                        }
                    }

                    if self.settings.map_layout != "Small" {
                        // Check that Phantoon can be defeated. This is to rule out the possibility that Phantoon may be locked
                        // behind Bowling Alley. On Small maps we relax this, since Phantoon may not exist; the game is still
                        // verified to be logically beatable, but possibly some part of the map could be inaccessible due
                        // to Bowling Alley.
                        let phantoon_flag_id =
                            self.game_data.flag_isv.index_by_key["f_DefeatedPhantoon"];
                        let mut phantoon_defeated = false;
                        for (i, flag_id) in self.game_data.flag_ids.iter().enumerate() {
                            if *flag_id == phantoon_flag_id
                                && state.flag_location_state[i].bireachable_traversal.is_some()
                            {
                                phantoon_defeated = true;
                            }
                        }

                        if !phantoon_defeated {
                            bail!(
                                "[attempt {attempt_num_rando}] Attempt failed: Phantoon not defeated"
                            );
                        }
                    }
                }

                // Success:
                break;
            }

            if state.step_num == 2
                && self.settings.quality_of_life_settings.early_save
                && !state
                    .save_location_state
                    .iter()
                    .any(|x| x.bireachable_traversal.is_some())
            {
                bail!("[attempt {attempt_num_rando}] Attempt failed: no accessible save location");
            }
        }
        self.finish(attempt_num_rando, &mut state, &mut rng);
        self.get_randomization(
            &state,
            seed,
            display_seed,
            &mut rng,
            &mut traverser_pair,
            &start_location_data,
        )
    }
}
