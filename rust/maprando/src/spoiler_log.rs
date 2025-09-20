use anyhow::Result;
use hashbrown::HashMap;
use maprando_game::{
    BeamType, Capacity, DoorType, FlagId, Item, LinkIdx, NodeId, Requirement, RoomId, StepTrailId,
    TraversalId, VertexId, VertexKey,
};
use maprando_logic::{GlobalState, LocalState};
use serde::{Deserialize, Serialize};
use strum::VariantNames;

use crate::{
    randomize::{
        RandomizationState, Randomizer, StartLocationData, TraverserPair,
        escape_timer::{self, SpoilerEscape},
        strip_name,
    },
    settings::SaveAnimals,
    traverse::{
        LocalStateReducer, NUM_COST_METRICS, Traverser, get_bireachable_idxs,
        get_one_way_reachable_idx, get_spoiler_trail_ids,
    },
};

fn get_start_trail_ids(lsr: &LocalStateReducer<StepTrailId>) -> [StepTrailId; NUM_COST_METRICS] {
    let mut out = [-1; NUM_COST_METRICS];
    for i in 0..NUM_COST_METRICS {
        let idx = lsr.best_cost_idxs[i];
        out[i] = lsr.trail_ids[idx as usize];
    }
    out
}

pub fn get_spoiler_traversal(tr: &Traverser) -> SpoilerTraversal {
    let mut steps: Vec<SpoilerTraversalStep> = vec![];
    let num_traversals = tr.past_steps.len();
    let mut first_updates_by_vertex: HashMap<(VertexId, TraversalId), usize> = HashMap::new();
    let mut last_updates_by_vertex: HashMap<(VertexId, TraversalId), usize> = HashMap::new();
    for (t, s) in tr.past_steps.iter().enumerate() {
        for (i, u) in s.updates.iter().enumerate() {
            first_updates_by_vertex.entry((u.vertex_id, t)).or_insert(i);
            last_updates_by_vertex.insert((u.vertex_id, t), i);
        }
    }

    for (t, s) in tr.past_steps.iter().enumerate() {
        let mut updated_vertex_ids: Vec<VertexId> = vec![];
        let mut updated_start_trail_ids: Vec<[StepTrailId; NUM_COST_METRICS]> = vec![];
        for (i, u) in s.updates.iter().enumerate() {
            if last_updates_by_vertex[&(u.vertex_id, t)] != i {
                continue;
            }

            // TODO: fix this.
            let mut new_start_trail_id = get_start_trail_ids(&tr.lsr[u.vertex_id]);
            for t1 in (t + 1)..num_traversals {
                if let Some(&j) = first_updates_by_vertex.get(&(u.vertex_id, t1)) {
                    new_start_trail_id = get_start_trail_ids(&tr.past_steps[t1].updates[j].old_lsr);
                    break;
                }
            }
            updated_vertex_ids.push(u.vertex_id);
            updated_start_trail_ids.push(new_start_trail_id);
        }
        steps.push(SpoilerTraversalStep {
            updated_vertex_ids,
            updated_start_trail_ids,
            step_num: s.step_num,
        });
    }

    let mut prev_trail_ids: Vec<StepTrailId> = vec![];
    let mut link_idxs: Vec<LinkIdx> = vec![];
    let mut local_states: Vec<SpoilerLocalState> = vec![];
    for t in &tr.step_trails {
        let old_state = if t.local_state.prev_trail_id >= 0 {
            tr.step_trails[t.local_state.prev_trail_id as usize].local_state
        } else {
            LocalState::full()
        };
        let spoiler_local_state = SpoilerLocalState::new(t.local_state, old_state);
        prev_trail_ids.push(t.local_state.prev_trail_id);
        link_idxs.push(t.link_idx);
        local_states.push(spoiler_local_state);
    }

    SpoilerTraversal {
        prev_trail_ids,
        link_idxs,
        local_states,
        steps,
    }
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerDetails {
    pub step: usize,
    pub start_state: SpoilerStartState,
    pub flags: Vec<SpoilerFlagDetails>,
    pub doors: Vec<SpoilerDoorDetails>,
    pub items: Vec<SpoilerItemDetails>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerItemLoc {
    pub item: String,
    pub location: SpoilerLocation,
}
#[derive(Serialize, Deserialize)]
pub struct SpoilerRoomLoc {
    // here temporarily, most likely, since these can be baked into the web UI
    pub room_id: usize,
    pub room: String,
    pub map: Vec<Vec<u8>>,
    pub map_reachable_step: Vec<Vec<u8>>,
    pub map_bireachable_step: Vec<Vec<u8>>,
    pub coords: (usize, usize),
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerItemSummary {
    pub item: String,
    pub location: SpoilerLocation,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerFlagSummary {
    flag: String,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerDoorSummary {
    door_type: String,
    location: SpoilerLocation,
    direction: String,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerSummary {
    pub step: usize,
    pub flags: Vec<SpoilerFlagSummary>,
    pub doors: Vec<SpoilerDoorSummary>,
    pub items: Vec<SpoilerItemSummary>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerLink {
    pub from_vertex_id: VertexId,
    pub to_vertex_id: VertexId,
    pub strat_id: Option<usize>,
    pub strat_name: String,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerRoom {
    pub room_id: usize,
    pub name: String,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerNode {
    pub room_id: usize,
    pub node_id: usize,
    pub name: String,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerGameData {
    rooms: Vec<SpoilerRoom>,
    nodes: Vec<SpoilerNode>,
    vertices: Vec<VertexKey>,
    links: Vec<SpoilerLink>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerTraversalStep {
    pub step_num: usize,
    pub updated_vertex_ids: Vec<VertexId>,
    pub updated_start_trail_ids: Vec<[StepTrailId; NUM_COST_METRICS]>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerTraversal {
    pub prev_trail_ids: Vec<StepTrailId>,
    pub link_idxs: Vec<LinkIdx>,
    pub local_states: Vec<SpoilerLocalState>,
    pub steps: Vec<SpoilerTraversalStep>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerLog {
    pub item_priority: Vec<String>,
    pub summary: Vec<SpoilerSummary>,
    pub objectives: Vec<String>,
    pub escape: SpoilerEscape,
    pub start_location: SpoilerStartLocation,
    pub hub_location_name: String,
    pub hub_obtain_route: Vec<SpoilerRouteEntry>,
    pub hub_return_route: Vec<SpoilerRouteEntry>,
    pub details: Vec<SpoilerDetails>,
    pub all_items: Vec<SpoilerItemLoc>,
    pub all_rooms: Vec<SpoilerRoomLoc>,
    pub game_data: SpoilerGameData,
    pub forward_traversal: SpoilerTraversal,
    pub reverse_traversal: SpoilerTraversal,
}

// Spoiler log ---------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpoilerRouteEntry {
    pub area: String,
    pub room: String,
    pub node: String,
    pub room_id: usize,
    pub short_room: String,
    pub from_node_id: usize,
    pub to_node_id: usize,
    pub obstacles_bitmask: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coords: Option<(usize, usize)>,
    pub strat_name: String,
    pub strat_id: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub strat_notes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reserves_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub missiles_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supers_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_bombs_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub relevant_flags: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerLocation {
    pub area: String,
    pub room_id: usize,
    pub room: String,
    pub node_id: usize,
    pub node: String,
    pub coords: (usize, usize),
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerStartLocation {
    pub name: String,
    pub room_id: usize,
    pub node_id: usize,
    pub x: f32,
    pub y: f32,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerStartState {
    max_energy: Capacity,
    max_reserves: Capacity,
    max_missiles: Capacity,
    max_supers: Capacity,
    max_power_bombs: Capacity,
    collectible_missiles: Capacity,
    collectible_supers: Capacity,
    collectible_power_bombs: Capacity,
    items: Vec<String>,
    flags: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerItemDetails {
    pub item: String,
    pub location: SpoilerLocation,
    pub reachable_step: usize,
    pub difficulty: Option<String>,
    pub obtain_route: Vec<SpoilerRouteEntry>,
    pub return_route: Vec<SpoilerRouteEntry>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerFlagDetails {
    pub flag: String,
    pub location: SpoilerLocation,
    pub reachable_step: usize,
    pub obtain_route: Vec<SpoilerRouteEntry>,
    pub return_route: Vec<SpoilerRouteEntry>,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerDoorDetails {
    door_type: String,
    direction: String,
    location: SpoilerLocation,
    obtain_route: Vec<SpoilerRouteEntry>,
    return_route: Vec<SpoilerRouteEntry>,
}

#[derive(Serialize, Deserialize, Copy, Clone, Default)]
pub struct SpoilerLocalState {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reserves_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub missiles_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supers_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_bombs_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shinecharge_frames_remaining: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycle_frames: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub farm_baseline_energy_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub farm_baseline_reserves_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub farm_baseline_missiles_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub farm_baseline_supers_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub farm_baseline_power_bombs_used: Option<Capacity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_suit: Option<Capacity>,
}

struct VertexInfo {
    area_name: String,
    room_id: usize,
    room_name: String,
    room_coords: (usize, usize),
    node_name: String,
    node_id: usize,
}

impl SpoilerLocalState {
    fn new(local: LocalState, ref_local: LocalState) -> Self {
        Self {
            energy_used: if local.energy == ref_local.energy {
                None
            } else {
                Some(local.energy)
            },
            reserves_used: if local.reserves_used == ref_local.reserves_used {
                None
            } else {
                Some(local.reserves_used)
            },
            missiles_used: if local.missiles_used == ref_local.missiles_used {
                None
            } else {
                Some(local.missiles_used)
            },
            supers_used: if local.supers_used == ref_local.supers_used {
                None
            } else {
                Some(local.supers_used)
            },
            power_bombs_used: if local.power_bombs_used == ref_local.power_bombs_used {
                None
            } else {
                Some(local.power_bombs_used)
            },
            shinecharge_frames_remaining: if local.shinecharge_frames_remaining
                == ref_local.shinecharge_frames_remaining
            {
                None
            } else {
                Some(local.shinecharge_frames_remaining)
            },
            cycle_frames: if local.cycle_frames == ref_local.cycle_frames {
                None
            } else {
                Some(local.cycle_frames)
            },
            farm_baseline_energy_used: if local.farm_baseline_energy_used
                == ref_local.farm_baseline_energy_used
            {
                None
            } else {
                Some(local.farm_baseline_energy_used)
            },
            farm_baseline_reserves_used: if local.farm_baseline_reserves_used
                == ref_local.farm_baseline_reserves_used
            {
                None
            } else {
                Some(local.farm_baseline_reserves_used)
            },
            farm_baseline_missiles_used: if local.farm_baseline_missiles_used
                == ref_local.farm_baseline_missiles_used
            {
                None
            } else {
                Some(local.farm_baseline_missiles_used)
            },
            farm_baseline_supers_used: if local.farm_baseline_supers_used
                == ref_local.farm_baseline_supers_used
            {
                None
            } else {
                Some(local.farm_baseline_supers_used)
            },
            farm_baseline_power_bombs_used: if local.farm_baseline_power_bombs_used
                == ref_local.farm_baseline_power_bombs_used
            {
                None
            } else {
                Some(local.farm_baseline_power_bombs_used)
            },
            flash_suit: if local.flash_suit == ref_local.flash_suit {
                None
            } else {
                Some(if local.flash_suit { 1 } else { 0 })
            },
        }
    }
}

fn extract_relevant_flags(req: &Requirement, out: &mut Vec<usize>) {
    match req {
        Requirement::Flag(flag_id) => {
            out.push(*flag_id);
        }
        Requirement::And(reqs) => {
            for r in reqs {
                extract_relevant_flags(r, out);
            }
        }
        Requirement::Or(reqs) => {
            for r in reqs {
                extract_relevant_flags(r, out);
            }
        }
        _ => {}
    }
}

fn get_vertex_info(randomizer: &Randomizer, vertex_id: usize) -> VertexInfo {
    let VertexKey {
        room_id, node_id, ..
    } = randomizer.game_data.vertex_isv.keys[vertex_id];
    get_vertex_info_by_id(randomizer, room_id, node_id)
}

fn get_vertex_info_by_id(randomizer: &Randomizer, room_id: RoomId, node_id: NodeId) -> VertexInfo {
    let room_ptr = randomizer.game_data.room_ptr_by_id[&room_id];
    let room_idx = randomizer.game_data.room_idx_by_ptr[&room_ptr];
    let area = randomizer.map.area[room_idx];
    let room_coords = randomizer.map.rooms[room_idx];
    VertexInfo {
        area_name: randomizer.game_data.area_names[area].clone(),
        room_name: randomizer.game_data.room_json_map[&room_id]["name"]
            .as_str()
            .unwrap()
            .to_string(),
        room_id,
        room_coords,
        node_name: randomizer.game_data.node_json_map[&(room_id, node_id)]["name"]
            .as_str()
            .unwrap()
            .to_string(),
        node_id,
    }
}

fn get_spoiler_start_state(
    randomizer: &Randomizer,
    global_state: &GlobalState,
) -> SpoilerStartState {
    let mut items: Vec<String> = Vec::new();
    for i in 0..randomizer.game_data.item_isv.keys.len() {
        if global_state.inventory.items[i] {
            items.push(randomizer.game_data.item_isv.keys[i].to_string());
        }
    }
    let mut flags: Vec<String> = Vec::new();
    for i in 0..randomizer.game_data.flag_isv.keys.len() {
        if global_state.flags[i] {
            flags.push(randomizer.game_data.flag_isv.keys[i].to_string());
        }
    }
    SpoilerStartState {
        max_energy: global_state.inventory.max_energy,
        max_reserves: global_state.inventory.max_reserves,
        max_missiles: global_state.inventory.max_missiles,
        max_supers: global_state.inventory.max_supers,
        max_power_bombs: global_state.inventory.max_power_bombs,
        collectible_missiles: global_state.inventory.collectible_missile_packs * 5,
        collectible_supers: global_state.inventory.collectible_super_packs * 5,
        collectible_power_bombs: global_state.inventory.collectible_power_bomb_packs * 5,
        items,
        flags,
    }
}

pub fn get_spoiler_route(
    randomizer: &Randomizer,
    global_state: &GlobalState,
    trail_ids: &[StepTrailId],
    traverser: &Traverser,
    reverse: bool,
) -> Vec<SpoilerRouteEntry> {
    let mut route: Vec<SpoilerRouteEntry> = Vec::new();

    if trail_ids.is_empty() {
        return route;
    }
    for &trail_id in trail_ids {
        let trail = &traverser.step_trails[trail_id as usize];
        let link_idx = trail.link_idx;
        let link = randomizer.get_link(link_idx as usize);
        let new_local_state = trail.local_state;
        let from_vertex_info = get_vertex_info(randomizer, link.from_vertex_id);
        let to_vertex_info = get_vertex_info(randomizer, link.to_vertex_id);
        let VertexKey {
            obstacle_mask: to_obstacles_mask,
            ..
        } = randomizer.game_data.vertex_isv.keys[link.to_vertex_id];
        let door_coords = randomizer
            .game_data
            .node_coords
            .get(&(to_vertex_info.room_id, to_vertex_info.node_id))
            .copied();
        let coords = door_coords.map(|(x, y)| {
            (
                x + to_vertex_info.room_coords.0,
                y + to_vertex_info.room_coords.1,
            )
        });

        let mut relevant_flag_idxs = vec![];
        extract_relevant_flags(&link.requirement, &mut relevant_flag_idxs);
        relevant_flag_idxs.sort();
        relevant_flag_idxs.dedup();
        let mut relevant_flags = vec![];
        for flag_idx in relevant_flag_idxs {
            let flag_name = randomizer.game_data.flag_isv.keys[flag_idx].clone();
            if global_state.flags[flag_idx] {
                relevant_flags.push(flag_name);
            }
        }

        let spoiler_entry = SpoilerRouteEntry {
            area: to_vertex_info.area_name,
            short_room: strip_name(&to_vertex_info.room_name),
            room: to_vertex_info.room_name,
            node: to_vertex_info.node_name,
            room_id: to_vertex_info.room_id,
            from_node_id: from_vertex_info.node_id,
            to_node_id: to_vertex_info.node_id,
            strat_id: link.strat_id,
            obstacles_bitmask: to_obstacles_mask,
            coords,
            strat_name: link.strat_name.clone(),
            strat_notes: link.strat_notes.clone(),
            energy_used: Some(new_local_state.energy),
            reserves_used: Some(new_local_state.reserves_used),
            missiles_used: Some(new_local_state.missiles_used),
            supers_used: Some(new_local_state.supers_used),
            power_bombs_used: Some(new_local_state.power_bombs_used),
            relevant_flags,
        };
        route.push(spoiler_entry);
    }

    if reverse {
        route.reverse();
    }

    // Remove repeated resource values, to reduce clutter in the spoiler view:
    for i in (0..(route.len() - 1)).rev() {
        if route[i + 1].energy_used == route[i].energy_used {
            route[i + 1].energy_used = None;
        }
        if route[i + 1].reserves_used == route[i].reserves_used {
            route[i + 1].reserves_used = None;
        }
        if route[i + 1].missiles_used == route[i].missiles_used {
            route[i + 1].missiles_used = None;
        }
        if route[i + 1].supers_used == route[i].supers_used {
            route[i + 1].supers_used = None;
        }
        if route[i + 1].power_bombs_used == route[i].power_bombs_used {
            route[i + 1].power_bombs_used = None;
        }
    }
    route
}

fn get_spoiler_route_birectional(
    randomizer: &Randomizer,
    state: &RandomizationState,
    vertex_id: usize,
    traverser_pair: &TraverserPair,
) -> (Vec<SpoilerRouteEntry>, Vec<SpoilerRouteEntry>) {
    let forward = &traverser_pair.forward;
    let reverse = &traverser_pair.reverse;
    let global_state = &state.global_state;
    let (forward_cost_idx, reverse_cost_idx) =
        get_bireachable_idxs(global_state, vertex_id, forward, reverse).unwrap();
    let forward_trail_ids: Vec<StepTrailId> =
        get_spoiler_trail_ids(forward, vertex_id, forward_cost_idx);
    let reverse_trail_ids: Vec<StepTrailId> =
        get_spoiler_trail_ids(reverse, vertex_id, reverse_cost_idx);
    let obtain_route =
        get_spoiler_route(randomizer, global_state, &forward_trail_ids, forward, false);
    let return_route =
        get_spoiler_route(randomizer, global_state, &reverse_trail_ids, reverse, true);
    (obtain_route, return_route)
}

fn get_spoiler_route_one_way(
    randomizer: &Randomizer,
    state: &RandomizationState,
    vertex_id: usize,
    forward: &Traverser,
) -> Vec<SpoilerRouteEntry> {
    let global_state = &state.global_state;
    let forward_cost_idx = get_one_way_reachable_idx(vertex_id, forward).unwrap();
    let forward_trail_ids: Vec<StepTrailId> =
        get_spoiler_trail_ids(forward, vertex_id, forward_cost_idx);
    get_spoiler_route(randomizer, global_state, &forward_trail_ids, forward, false)
}

fn get_spoiler_item_details(
    randomizer: &Randomizer,
    state: &RandomizationState,
    item_vertex_id: usize,
    item: Item,
    tier: Option<usize>,
    item_location_idx: usize,
    traverser_pair: &TraverserPair,
) -> SpoilerItemDetails {
    let (obtain_route, return_route) =
        get_spoiler_route_birectional(randomizer, state, item_vertex_id, traverser_pair);
    let (room_id, node_id) = randomizer.game_data.item_locations[item_location_idx];
    let item_vertex_info = get_vertex_info_by_id(randomizer, room_id, node_id);
    let reachable_traversal = state.item_location_state[item_location_idx]
        .reachable_traversal
        .unwrap();
    SpoilerItemDetails {
        item: Item::VARIANTS[item as usize].to_string(),
        location: SpoilerLocation {
            area: item_vertex_info.area_name,
            room_id: item_vertex_info.room_id,
            room: item_vertex_info.room_name,
            node_id: item_vertex_info.node_id,
            node: item_vertex_info.node_name,
            coords: item_vertex_info.room_coords,
        },
        reachable_step: traverser_pair.forward.past_steps[reachable_traversal].step_num,
        difficulty: tier.map(|x| randomizer.difficulty_tiers[x].name.clone()),
        obtain_route,
        return_route,
    }
}

fn get_spoiler_item_summary(
    randomizer: &Randomizer,
    _state: &RandomizationState,
    _item_vertex_id: usize,
    item: Item,
    item_location_idx: usize,
) -> SpoilerItemSummary {
    let (room_id, node_id) = randomizer.game_data.item_locations[item_location_idx];
    let item_vertex_info = get_vertex_info_by_id(randomizer, room_id, node_id);
    SpoilerItemSummary {
        item: Item::VARIANTS[item as usize].to_string(),
        location: SpoilerLocation {
            area: item_vertex_info.area_name,
            room_id: item_vertex_info.room_id,
            room: item_vertex_info.room_name,
            node_id: item_vertex_info.node_id,
            node: item_vertex_info.node_name,
            coords: item_vertex_info.room_coords,
        },
    }
}

pub fn get_spoiler_flag_details(
    randomizer: &Randomizer,
    state: &RandomizationState,
    flag_vertex_id: usize,
    flag_id: FlagId,
    flag_idx: usize,
    traverser_pair: &TraverserPair,
) -> SpoilerFlagDetails {
    let (obtain_route, return_route) =
        get_spoiler_route_birectional(randomizer, state, flag_vertex_id, traverser_pair);
    let flag_vertex_info = get_vertex_info(randomizer, flag_vertex_id);
    let reachable_traversal = state.flag_location_state[flag_idx]
        .reachable_traversal
        .unwrap();
    SpoilerFlagDetails {
        flag: randomizer.game_data.flag_isv.keys[flag_id].to_string(),
        location: SpoilerLocation {
            area: flag_vertex_info.area_name,
            room_id: flag_vertex_info.room_id,
            room: flag_vertex_info.room_name,
            node_id: flag_vertex_info.node_id,
            node: flag_vertex_info.node_name,
            coords: flag_vertex_info.room_coords,
        },
        reachable_step: traverser_pair.forward.past_steps[reachable_traversal].step_num,
        obtain_route,
        return_route,
    }
}

pub fn get_spoiler_flag_details_one_way(
    randomizer: &Randomizer,
    state: &RandomizationState,
    flag_vertex_id: usize,
    flag_id: FlagId,
    flag_idx: usize,
    forward: &Traverser,
) -> SpoilerFlagDetails {
    // This is for a one-way reachable flag, used for f_DefeatedMotherBrain:
    let obtain_route = get_spoiler_route_one_way(randomizer, state, flag_vertex_id, forward);
    let flag_vertex_info = get_vertex_info(randomizer, flag_vertex_id);
    let reachable_traversal = state.flag_location_state[flag_idx]
        .reachable_traversal
        .unwrap();
    SpoilerFlagDetails {
        flag: randomizer.game_data.flag_isv.keys[flag_id].to_string(),
        location: SpoilerLocation {
            area: flag_vertex_info.area_name,
            room_id: flag_vertex_info.room_id,
            room: flag_vertex_info.room_name,
            node_id: flag_vertex_info.node_id,
            node: flag_vertex_info.node_name,
            coords: flag_vertex_info.room_coords,
        },
        reachable_step: forward.past_steps[reachable_traversal].step_num,
        obtain_route,
        return_route: vec![],
    }
}

fn get_door_type_name(door_type: DoorType) -> String {
    match door_type {
        DoorType::Blue => "blue",
        DoorType::Red => "red",
        DoorType::Green => "green",
        DoorType::Yellow => "yellow",
        DoorType::Gray => "gray",
        DoorType::Wall => "wall",
        DoorType::Beam(beam) => match beam {
            BeamType::Charge => "charge",
            BeamType::Ice => "ice",
            BeamType::Wave => "wave",
            BeamType::Spazer => "spazer",
            BeamType::Plasma => "plasma",
        },
    }
    .to_string()
}

pub fn get_spoiler_door_details(
    randomizer: &Randomizer,
    state: &RandomizationState,
    unlock_vertex_id: usize,
    locked_door_idx: usize,
    traverser_pair: &TraverserPair,
) -> SpoilerDoorDetails {
    let (obtain_route, return_route) =
        get_spoiler_route_birectional(randomizer, state, unlock_vertex_id, traverser_pair);
    let summary = get_spoiler_door_summary(randomizer, unlock_vertex_id, locked_door_idx);
    SpoilerDoorDetails {
        door_type: summary.door_type,
        location: summary.location,
        direction: summary.direction,
        obtain_route,
        return_route,
    }
}

pub fn get_spoiler_flag_summary(
    randomizer: &Randomizer,
    _state: &RandomizationState,
    _flag_vertex_id: usize,
    flag_id: FlagId,
) -> SpoilerFlagSummary {
    SpoilerFlagSummary {
        flag: randomizer.game_data.flag_isv.keys[flag_id].to_string(),
    }
}

pub fn get_spoiler_door_summary(
    randomizer: &Randomizer,
    _unlock_vertex_id: usize,
    locked_door_idx: usize,
) -> SpoilerDoorSummary {
    let locked_door = &randomizer.locked_door_data.locked_doors[locked_door_idx];
    let (room_id, node_id) = randomizer.game_data.door_ptr_pair_map[&locked_door.src_ptr_pair];
    let door_vertex_id = randomizer.game_data.vertex_isv.index_by_key[&VertexKey {
        room_id,
        node_id,
        obstacle_mask: 0,
        actions: vec![],
    }];
    let door_vertex_info = get_vertex_info(randomizer, door_vertex_id);
    let mut direction = "none".to_string();
    let mut coords = door_vertex_info.room_coords;
    let ptr_pairs = vec![locked_door.src_ptr_pair];
    for ptr_pair in ptr_pairs {
        let (room_idx, door_idx) =
            randomizer.game_data.room_and_door_idxs_by_door_ptr_pair[&ptr_pair];
        if !randomizer.map.room_mask[room_idx] {
            continue;
        }
        let room_geom = &randomizer.game_data.room_geometry[room_idx];
        let door = &room_geom.doors[door_idx];
        direction = door.direction.clone();
        coords.0 += door.x;
        coords.1 += door.y;
    }
    SpoilerDoorSummary {
        door_type: get_door_type_name(
            randomizer.locked_door_data.locked_doors[locked_door_idx].door_type,
        ),
        location: SpoilerLocation {
            area: door_vertex_info.area_name,
            room_id: door_vertex_info.room_id,
            room: door_vertex_info.room_name,
            node_id: door_vertex_info.node_id,
            node: door_vertex_info.node_name,
            coords,
        },
        direction,
    }
}

pub fn get_spoiler_game_data(randomizer: &Randomizer) -> SpoilerGameData {
    let mut rooms: Vec<SpoilerRoom> = vec![];
    let mut nodes: Vec<SpoilerNode> = vec![];
    for &room_ptr in &randomizer.game_data.room_ptrs {
        let room_id = randomizer.game_data.raw_room_id_by_ptr[&room_ptr];
        let room = &randomizer.game_data.room_json_map[&room_id];
        rooms.push(SpoilerRoom {
            room_id,
            name: room["name"].as_str().unwrap().to_string(),
        });

        for node in room["nodes"].members() {
            let node_id = node["id"].as_usize().unwrap();
            let node_name = node["name"].as_str().unwrap().to_string();
            nodes.push(SpoilerNode {
                room_id,
                node_id,
                name: node_name,
            });
        }
    }

    let mut links: Vec<SpoilerLink> = vec![];
    for link in randomizer
        .base_links_data
        .links
        .iter()
        .chain(randomizer.seed_links_data.links.iter())
    {
        links.push(SpoilerLink {
            from_vertex_id: link.from_vertex_id,
            to_vertex_id: link.to_vertex_id,
            strat_id: link.strat_id,
            strat_name: link.strat_name.clone(),
        });
    }
    SpoilerGameData {
        rooms,
        nodes,
        vertices: randomizer.game_data.vertex_isv.keys.clone(),
        links,
    }
}

pub fn get_spoiler_log(
    randomizer: &Randomizer,
    state: &RandomizationState,
    traverser_pair: &mut TraverserPair,
    save_animals: SaveAnimals,
    start_location_data: &StartLocationData,
) -> Result<SpoilerLog> {
    let forward_traversal = get_spoiler_traversal(&traverser_pair.forward);
    let reverse_traversal = get_spoiler_traversal(&traverser_pair.reverse);

    // Compute the first step on which each node becomes reachable/bireachable:
    let mut node_reachable_step: HashMap<(RoomId, NodeId), usize> = HashMap::new();
    let mut node_bireachable_step: HashMap<(RoomId, NodeId), usize> = HashMap::new();
    let mut map_tile_reachable_step: HashMap<(RoomId, (usize, usize)), usize> = HashMap::new();
    let mut map_tile_bireachable_step: HashMap<(RoomId, (usize, usize)), usize> = HashMap::new();

    let mut traversal_num = traverser_pair.forward.past_steps.len() - 1;
    let mut spoiler_summaries: Vec<SpoilerSummary> = vec![];
    let mut spoiler_details: Vec<SpoilerDetails> = vec![];
    let mut done: bool = false;

    while !done {
        let step_num = traverser_pair.forward.past_steps[traversal_num].step_num;
        let mut spoiler_item_summaries: Vec<SpoilerItemSummary> = vec![];
        let mut spoiler_flag_summaries: Vec<SpoilerFlagSummary> = vec![];
        let mut spoiler_door_summaries: Vec<SpoilerDoorSummary> = vec![];
        let mut spoiler_item_details: Vec<SpoilerItemDetails> = vec![];
        let mut spoiler_flag_details: Vec<SpoilerFlagDetails> = vec![];
        let mut spoiler_door_details: Vec<SpoilerDoorDetails> = vec![];
        let global_state = traverser_pair.forward.past_steps[traversal_num]
            .global_state
            .clone();

        while !done && traverser_pair.forward.past_steps[traversal_num].step_num == step_num {
            assert_eq!(traverser_pair.forward.past_steps.len(), traversal_num + 1);
            assert_eq!(traverser_pair.reverse.past_steps.len(), traversal_num + 1);
            for (i, item_state) in state.item_location_state.iter().enumerate() {
                if item_state.bireachable_traversal != Some(traversal_num) {
                    continue;
                }
                let Some(item) = item_state.placed_item else {
                    continue;
                };
                if item == Item::Nothing {
                    continue;
                }
                let item_vertex_id = item_state.bireachable_vertex_id.unwrap();
                let item_summary =
                    get_spoiler_item_summary(randomizer, state, item_vertex_id, item, i);
                spoiler_item_summaries.push(item_summary);
                let item_details = get_spoiler_item_details(
                    randomizer,
                    state,
                    item_vertex_id,
                    item,
                    item_state.placed_tier,
                    i,
                    traverser_pair,
                );
                spoiler_item_details.push(item_details);
            }

            for (i, flag_state) in state.flag_location_state.iter().enumerate() {
                let flag_id = randomizer.game_data.flag_ids[i];
                if flag_id == randomizer.game_data.mother_brain_defeated_flag_id {
                    if flag_state.reachable_traversal != Some(traversal_num) {
                        continue;
                    }
                    let flag_vertex_id = flag_state.reachable_vertex_id.unwrap();
                    let flag_summary =
                        get_spoiler_flag_summary(randomizer, state, flag_vertex_id, flag_id);
                    spoiler_flag_summaries.push(flag_summary);
                    let flag_details = get_spoiler_flag_details_one_way(
                        randomizer,
                        state,
                        flag_vertex_id,
                        flag_id,
                        i,
                        &traverser_pair.forward,
                    );
                    spoiler_flag_details.push(flag_details);
                } else {
                    if flag_state.bireachable_traversal != Some(traversal_num) {
                        continue;
                    }
                    let flag_vertex_id = flag_state.bireachable_vertex_id.unwrap();
                    let flag_summary =
                        get_spoiler_flag_summary(randomizer, state, flag_vertex_id, flag_id);
                    spoiler_flag_summaries.push(flag_summary);
                    let flag_details = get_spoiler_flag_details(
                        randomizer,
                        state,
                        flag_vertex_id,
                        flag_id,
                        i,
                        traverser_pair,
                    );
                    spoiler_flag_details.push(flag_details);
                }
            }

            for (i, door_state) in state.door_state.iter().enumerate() {
                if door_state.bireachable_traversal != Some(traversal_num) {
                    continue;
                }
                let unlock_vertex_id = door_state.bireachable_vertex_id.unwrap();
                let door_summary = get_spoiler_door_summary(randomizer, unlock_vertex_id, i);
                spoiler_door_summaries.push(door_summary);
                let door_details = get_spoiler_door_details(
                    randomizer,
                    state,
                    unlock_vertex_id,
                    i,
                    traverser_pair,
                );
                spoiler_door_details.push(door_details);
            }

            let forward_step = &traverser_pair.forward.past_steps[traversal_num];
            for (
                v,
                VertexKey {
                    room_id, node_id, ..
                },
            ) in randomizer.game_data.vertex_isv.keys.iter().enumerate()
            {
                if get_bireachable_idxs(
                    &forward_step.global_state,
                    v,
                    &traverser_pair.forward,
                    &traverser_pair.reverse,
                )
                .is_some()
                {
                    node_bireachable_step.insert((*room_id, *node_id), step_num.saturating_sub(1));
                    let room_ptr = randomizer.game_data.room_ptr_by_id[room_id];
                    let room_idx = randomizer.game_data.room_idx_by_ptr[&room_ptr];
                    if let Some(coords) = randomizer
                        .game_data
                        .node_tile_coords
                        .get(&(*room_id, *node_id))
                    {
                        for (x, y) in coords.iter().copied() {
                            let key = if *room_id == 322 {
                                // Adjust for East Pants Room being offset by one screen right and down from Pants Room
                                (room_idx, (x + 1, y + 1))
                            } else if *room_id == 313 {
                                // Adjust Homing Geemer Room being offset from West Ocean:
                                (room_idx, (x + 5, y + 2))
                            } else {
                                (room_idx, (x, y))
                            };
                            map_tile_bireachable_step.insert(key, step_num.saturating_sub(1));
                        }
                    }
                }

                if !traverser_pair.forward.lsr[v].local.is_empty() {
                    node_reachable_step.insert((*room_id, *node_id), step_num.saturating_sub(1));
                    let room_ptr = randomizer.game_data.room_ptr_by_id[room_id];
                    let room_idx = randomizer.game_data.room_idx_by_ptr[&room_ptr];
                    if let Some(coords) = randomizer
                        .game_data
                        .node_tile_coords
                        .get(&(*room_id, *node_id))
                    {
                        for (x, y) in coords.iter().copied() {
                            let key = if *room_id == 322 {
                                // Adjust for East Pants Room being offset by one screen right and down from Pants Room
                                (room_idx, (x + 1, y + 1))
                            } else if *room_id == 313 {
                                // Adjust Homing Geemer Room being offset from West Ocean:
                                (room_idx, (x + 5, y + 2))
                            } else {
                                (room_idx, (x, y))
                            };
                            map_tile_reachable_step.insert(key, step_num.saturating_sub(1));
                        }
                    }
                }
            }

            traverser_pair.forward.pop_step();
            traverser_pair.reverse.pop_step();
            if traverser_pair.forward.past_steps.is_empty() {
                done = true;
            } else {
                traversal_num -= 1;
            }
        }

        spoiler_item_summaries.reverse();
        spoiler_flag_summaries.reverse();
        spoiler_door_summaries.reverse();
        spoiler_summaries.push(SpoilerSummary {
            step: step_num,
            items: spoiler_item_summaries,
            flags: spoiler_flag_summaries,
            doors: spoiler_door_summaries,
        });

        spoiler_item_details.reverse();
        spoiler_flag_details.reverse();
        spoiler_door_details.reverse();
        spoiler_details.push(SpoilerDetails {
            step: step_num,
            start_state: get_spoiler_start_state(randomizer, &global_state),
            items: spoiler_item_details,
            flags: spoiler_flag_details,
            doors: spoiler_door_details,
        });
    }
    spoiler_summaries.reverse();
    spoiler_details.reverse();

    let spoiler_all_items = state
        .item_location_state
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let (r, n) = randomizer.game_data.item_locations[i];
            let item_vertex_info = get_vertex_info_by_id(randomizer, r, n);
            let room_id = item_vertex_info.room_id;
            let node_id = item_vertex_info.node_id;
            let node_coords = randomizer.game_data.node_coords[&(room_id, node_id)];
            let coords = (
                item_vertex_info.room_coords.0 + node_coords.0,
                item_vertex_info.room_coords.1 + node_coords.1,
            );
            let location = SpoilerLocation {
                area: item_vertex_info.area_name,
                room_id,
                room: item_vertex_info.room_name,
                node_id,
                node: item_vertex_info.node_name,
                coords,
            };
            let item = x.placed_item.unwrap();
            SpoilerItemLoc {
                item: Item::VARIANTS[item as usize].to_string(),
                location,
            }
        })
        .collect();

    let mut spoiler_all_rooms: Vec<SpoilerRoomLoc> = Vec::new();
    for (room_idx, room_coords) in randomizer.map.rooms.iter().enumerate() {
        if !randomizer.map.room_mask[room_idx] {
            continue;
        }
        let room_geom = &randomizer.game_data.room_geometry[room_idx];
        let room_id = randomizer.game_data.room_id_by_ptr[&room_geom.rom_address];
        let room = randomizer.game_data.room_json_map[&room_id]["name"]
            .as_str()
            .unwrap()
            .to_string();
        let map = if room_idx == randomizer.game_data.toilet_room_idx {
            vec![vec![1; 1]; 10]
        } else {
            room_geom.map.clone()
        };
        let height = map.len();
        let width = map[0].len();
        let mut map_reachable_step: Vec<Vec<u8>> = vec![vec![255; width]; height];
        let mut map_bireachable_step: Vec<Vec<u8>> = vec![vec![255; width]; height];
        for y in 0..height {
            for x in 0..width {
                if map[y][x] != 0 {
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
        spoiler_all_rooms.push(SpoilerRoomLoc {
            room_id,
            room,
            map,
            map_reachable_step,
            map_bireachable_step,
            coords: *room_coords,
        });
    }

    let spoiler_escape = escape_timer::compute_escape_data(
        randomizer.game_data,
        randomizer.map,
        randomizer.settings,
        save_animals != SaveAnimals::No,
        &randomizer.difficulty_tiers[0],
    )?;

    let spoiler_objectives: Vec<String> = randomizer
        .objectives
        .iter()
        .map(|x| x.get_flag_name().to_owned())
        .collect();

    let hub_room_id = state.hub_location.room_id;
    let hub_room_name = randomizer.game_data.room_json_map[&hub_room_id]["name"]
        .as_str()
        .unwrap()
        .to_string();
    Ok(SpoilerLog {
        item_priority: state
            .item_precedence
            .iter()
            .map(|x| format!("{x:?}"))
            .collect(),
        summary: spoiler_summaries,
        objectives: spoiler_objectives,
        start_location: SpoilerStartLocation {
            name: state.start_location.name.clone(),
            room_id: state.start_location.room_id,
            node_id: state.start_location.node_id,
            x: state.start_location.x,
            y: state.start_location.y,
        },
        hub_location_name: hub_room_name,
        hub_obtain_route: start_location_data.hub_obtain_route.clone(),
        hub_return_route: start_location_data.hub_return_route.clone(),
        escape: spoiler_escape,
        details: spoiler_details,
        all_items: spoiler_all_items,
        all_rooms: spoiler_all_rooms,
        game_data: get_spoiler_game_data(randomizer),
        forward_traversal,
        reverse_traversal,
    })
}
