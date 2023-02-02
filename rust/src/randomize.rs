pub mod escape_timer;

use crate::{
    game_data::{self, Capacity, FlagId, Item, ItemLocationId, Link, Map, VertexId},
    traverse::{
        apply_requirement, get_spoiler_route, is_bireachable, traverse, GlobalState, LinkIdx,
        LocalState, TraverseResult,
    },
};
use hashbrown::HashSet;
use log::info;
use rand::SeedableRng;
use rand::{seq::SliceRandom, Rng};
use serde_derive::{Deserialize, Serialize};
use std::{cmp::{min, max}, convert::TryFrom, iter};
use strum::VariantNames;

use crate::game_data::GameData;

use self::escape_timer::SpoilerEscape;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum ItemPlacementStrategy {
    Open,
    Semiclosed,
    Closed,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct DebugOptions {
    pub new_game_extra: bool,
    pub extended_spoiler: bool,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct DifficultyConfig {
    pub tech: Vec<String>,
    pub shine_charge_tiles: i32,
    pub item_placement_strategy: ItemPlacementStrategy,
    pub resource_multiplier: f32,
    pub escape_timer_multiplier: f32,
    pub save_animals: bool,
    pub ridley_proficiency: f32,
    // Quality-of-life options:
    pub supers_double: bool,
    pub streamlined_escape: bool,
    pub mark_map_stations: bool,
    pub mark_majors: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_options: Option<DebugOptions>,
}

// Includes preprocessing specific to the map:
pub struct Randomizer<'a> {
    pub map: &'a Map,
    pub game_data: &'a GameData,
    pub difficulty: &'a DifficultyConfig,
    pub links: Vec<Link>,
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

struct DebugData {
    forward: TraverseResult,
    reverse: TraverseResult,
}

// State that changes over the course of item placement attempts
struct RandomizationState {
    step_num: usize,
    item_precedence: Vec<Item>, // An ordering of the 21 distinct item names. The game will prioritize placing key items earlier in the list.
    item_location_state: Vec<ItemLocationState>, // Corresponds to GameData.item_locations (one record for each of 100 item locations)
    flag_location_state: Vec<FlagLocationState>, // Corresponds to GameData.flag_locations
    items_remaining: Vec<usize>, // Corresponds to GameData.items_isv (one count for each of 21 distinct item names)
    global_state: GlobalState,
    done: bool, // Have all key items been placed?
    debug_data: Option<DebugData>,
}

pub struct Randomization {
    pub difficulty: DifficultyConfig,
    pub map: Map,
    pub item_placement: Vec<Item>,
    pub spoiler_log: SpoilerLog,
}

struct SelectItemsOutput {
    key_items: Vec<Item>,
    other_items: Vec<Item>,
    new_items_remaining: Vec<usize>,
}

struct VertexInfo {
    area_name: String,
    room_name: String,
    node_name: String,
}

impl<'r> Randomizer<'r> {
    pub fn new(
        map: &'r Map,
        difficulty: &'r DifficultyConfig,
        game_data: &'r GameData,
    ) -> Randomizer<'r> {
        let mut door_edges: Vec<(VertexId, VertexId)> = Vec::new();
        for door in &map.doors {
            let src_exit_ptr = door.0 .0;
            let src_entrance_ptr = door.0 .1;
            let dst_exit_ptr = door.1 .0;
            let dst_entrance_ptr = door.1 .1;
            let bidirectional = door.2;
            let (src_room_id, src_node_id) =
                game_data.door_ptr_pair_map[&(src_exit_ptr, src_entrance_ptr)];
            let (dst_room_id, dst_node_id) =
                game_data.door_ptr_pair_map[&(dst_exit_ptr, dst_entrance_ptr)];
            for obstacle_bitmask in 0..(1 << game_data.room_num_obstacles[&src_room_id]) {
                let src_vertex_id = game_data.vertex_isv.index_by_key
                    [&(src_room_id, src_node_id, obstacle_bitmask)];
                let dst_vertex_id =
                    game_data.vertex_isv.index_by_key[&(dst_room_id, dst_node_id, 0)];
                door_edges.push((src_vertex_id, dst_vertex_id));
            }
            if bidirectional {
                for obstacle_bitmask in 0..(1 << game_data.room_num_obstacles[&dst_room_id]) {
                    let src_vertex_id =
                        game_data.vertex_isv.index_by_key[&(src_room_id, src_node_id, 0)];
                    let dst_vertex_id = game_data.vertex_isv.index_by_key
                        [&(dst_room_id, dst_node_id, obstacle_bitmask)];
                    door_edges.push((dst_vertex_id, src_vertex_id));
                }
            }
        }
        let mut links = game_data.links.clone();
        for &(from_vertex_id, to_vertex_id) in &door_edges {
            links.push(Link {
                from_vertex_id,
                to_vertex_id,
                requirement: game_data::Requirement::Free,
                strat_name: "(Door transition)".to_string(),
            })
        }

        let mut initial_items_remaining: Vec<usize> = vec![1; game_data.item_isv.keys.len()];
        initial_items_remaining[Item::Missile as usize] = 46;
        initial_items_remaining[Item::Super as usize] = 10;
        initial_items_remaining[Item::PowerBomb as usize] = 10;
        initial_items_remaining[Item::ETank as usize] = 14;
        initial_items_remaining[Item::ReserveTank as usize] = 4;
        assert!(initial_items_remaining.iter().sum::<usize>() == game_data.item_locations.len());

        Randomizer {
            map,
            initial_items_remaining,
            game_data,
            links,
            difficulty,
        }
    }

    fn get_tech_vec(&self) -> Vec<bool> {
        let tech_set: HashSet<String> = self.difficulty.tech.iter().map(|x| x.clone()).collect();
        self.game_data
            .tech_isv
            .keys
            .iter()
            .map(|x| tech_set.contains(x))
            .collect()
    }

    fn get_initial_flag_vec(&self) -> Vec<bool> {
        let mut flag_vec = vec![false; self.game_data.flag_isv.keys.len()];
        let tourian_open_idx = self.game_data.flag_isv.index_by_key["f_TourianOpen"];
        flag_vec[tourian_open_idx] = true;
        flag_vec
    }

    fn update_reachability(&self, state: &mut RandomizationState) {
        let num_vertices = self.game_data.vertex_isv.keys.len();
        let start_vertex_id = self.game_data.vertex_isv.index_by_key[&(8, 5, 0)]; // Landing site
        let forward = traverse(
            &self.links,
            &state.global_state,
            num_vertices,
            start_vertex_id,
            false,
            self.difficulty,
            self.game_data,
        );
        let reverse = traverse(
            &self.links,
            &state.global_state,
            num_vertices,
            start_vertex_id,
            true,
            self.difficulty,
            self.game_data,
        );
        for (i, vertex_ids) in self.game_data.item_vertex_ids.iter().enumerate() {
            // Clear out any previous bireachable markers (because in rare cases a previously bireachable
            // vertex can become no longer "bireachable" due to the imperfect cost heuristic used for
            // resource management.)
            state.item_location_state[i].bireachable = false;
            state.item_location_state[i].bireachable_vertex_id = None;

            for &v in vertex_ids {
                if forward.local_states[v].is_some() {
                    state.item_location_state[i].reachable = true;
                    if !state.item_location_state[i].bireachable
                        && is_bireachable(
                            &state.global_state,
                            &forward.local_states[v],
                            &reverse.local_states[v],
                        )
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
                    && is_bireachable(
                        &state.global_state,
                        &forward.local_states[v],
                        &reverse.local_states[v],
                    )
                {
                    state.flag_location_state[i].bireachable = true;
                    state.flag_location_state[i].bireachable_vertex_id = Some(v);
                }
            }
        }
        // Store TraverseResults to use for constructing spoiler log
        state.debug_data = Some(DebugData { forward, reverse });
    }

    fn select_items<R: Rng>(
        &self,
        state: &RandomizationState,
        num_bireachable: usize,
        num_oneway_reachable: usize,
        attempt_num: usize,
        rng: &mut R,
    ) -> Option<SelectItemsOutput> {
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
        let mut num_key_items_to_place = match self.difficulty.item_placement_strategy {
            ItemPlacementStrategy::Semiclosed | ItemPlacementStrategy::Closed => 1,
            ItemPlacementStrategy::Open => f32::ceil(
                (num_key_items_remaining as f32) / (num_items_remaining as f32)
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
        assert!(num_key_items_to_place >= 1);
        if num_key_items_to_place - 1 + attempt_num >= num_key_items_remaining {
            return None;
        }

        // If we will be placing `k` key items, we let the first `k - 1` items to place remain fixed based on the
        // item precedence order, while we vary the last key item across attempts (to try to find some choice that
        // will expand the set of bireachable item locations).
        let mut key_items_to_place: Vec<Item> =
            filtered_item_precedence[0..(num_key_items_to_place - 1)].to_vec();
        key_items_to_place.push(filtered_item_precedence[num_key_items_to_place - 1 + attempt_num]);
        assert!(key_items_to_place.len() == num_key_items_to_place);

        let mut new_items_remaining = state.items_remaining.clone();
        for &item in &key_items_to_place {
            new_items_remaining[item as usize] -= 1;
        }

        let num_other_items_to_place = num_items_to_place - num_key_items_to_place;
        let mut item_types_to_mix: Vec<Item>;
        let mut item_types_to_delay: Vec<Item>;
        match self.difficulty.item_placement_strategy {
            ItemPlacementStrategy::Open => {
                item_types_to_mix = vec![
                    Item::Missile,
                    Item::ETank,
                    Item::ReserveTank,
                    Item::Super,
                    Item::PowerBomb,
                ];
                item_types_to_delay = vec![];
            }
            ItemPlacementStrategy::Semiclosed => {
                item_types_to_mix = vec![Item::Missile, Item::ETank, Item::ReserveTank];
                item_types_to_delay = vec![];
                if state.items_remaining[Item::Super as usize]
                    < self.initial_items_remaining[Item::Super as usize]
                {
                    item_types_to_mix.push(Item::Super);
                } else {
                    item_types_to_delay.push(Item::Super);
                }
                if state.items_remaining[Item::PowerBomb as usize]
                    < self.initial_items_remaining[Item::PowerBomb as usize]
                {
                    item_types_to_mix.push(Item::PowerBomb);
                } else {
                    item_types_to_delay.push(Item::PowerBomb);
                }
            }
            ItemPlacementStrategy::Closed => {
                item_types_to_mix = vec![Item::Missile];
                if state.items_remaining[Item::PowerBomb as usize]
                    < self.initial_items_remaining[Item::PowerBomb as usize]
                    && state.items_remaining[Item::Super as usize]
                        == self.initial_items_remaining[Item::Super as usize]
                {
                    item_types_to_delay =
                        vec![Item::ETank, Item::ReserveTank, Item::PowerBomb, Item::Super];
                } else {
                    item_types_to_delay =
                        vec![Item::ETank, Item::ReserveTank, Item::Super, Item::PowerBomb];
                }
            }
        }

        let mut items_to_mix: Vec<Item> = Vec::new();
        for &item in &item_types_to_mix {
            for _ in 0..new_items_remaining[item as usize] {
                items_to_mix.push(item);
            }
        }
        let mut expansion_items_to_delay: Vec<Item> = Vec::new();
        for &item in &item_types_to_delay {
            for _ in 0..new_items_remaining[item as usize] {
                expansion_items_to_delay.push(item);
            }
        }
        let mut key_items_to_delay: Vec<Item> = Vec::new();
        for item_id in 0..self.game_data.item_isv.keys.len() {
            let item = Item::try_from(item_id).unwrap();
            if ![
                Item::Missile,
                Item::Super,
                Item::PowerBomb,
                Item::ETank,
                Item::ReserveTank,
            ]
            .contains(&item)
            {
                key_items_to_delay.push(item);
            }
        }

        let mut other_items_to_place: Vec<Item> = items_to_mix;
        other_items_to_place.shuffle(rng);
        other_items_to_place.extend(expansion_items_to_delay);
        other_items_to_place.extend(key_items_to_delay);
        other_items_to_place = other_items_to_place[0..num_other_items_to_place].to_vec();
        for &item in &other_items_to_place {
            new_items_remaining[item as usize] -= 1;
        }
        Some(SelectItemsOutput {
            key_items: key_items_to_place,
            other_items: other_items_to_place,
            new_items_remaining,
        })
    }

    fn place_items(
        &self,
        state: &mut RandomizationState,
        bireachable_locations: &[ItemLocationId],
        other_locations: &[ItemLocationId],
        key_items_to_place: &[Item],
        other_items_to_place: &[Item],
    ) {
        let mut all_locations: Vec<ItemLocationId> = Vec::new();
        all_locations.extend(bireachable_locations);
        all_locations.extend(other_locations);
        let mut all_items_to_place: Vec<Item> = Vec::new();
        all_items_to_place.extend(key_items_to_place);
        all_items_to_place.extend(other_items_to_place);
        for (&loc, &item) in iter::zip(&all_locations, &all_items_to_place) {
            state.item_location_state[loc].placed_item = Some(item);
        }
    }

    fn collect_items(&self, state: &mut RandomizationState) {
        for item_loc_state in &mut state.item_location_state {
            if !item_loc_state.collected && item_loc_state.bireachable {
                if let Some(item) = item_loc_state.placed_item {
                    state.global_state.collect(item, self.game_data);
                    item_loc_state.collected = true;
                }
            }
        }
    }

    fn finish(&self, state: &mut RandomizationState) {
        let mut remaining_items: Vec<Item> = Vec::new();
        for item_id in 0..self.game_data.item_isv.keys.len() {
            for _ in 0..state.items_remaining[item_id] {
                remaining_items.push(Item::try_from(item_id).unwrap());
            }
        }
        let mut idx = 0;
        for item_loc_state in &mut state.item_location_state {
            if item_loc_state.placed_item.is_none() {
                item_loc_state.placed_item = Some(remaining_items[idx]);
                idx += 1;
            }
        }
        assert!(idx == remaining_items.len());
    }

    fn step<R: Rng>(
        &self,
        state: &mut RandomizationState,
        rng: &mut R,
    ) -> Option<(SpoilerSummary, SpoilerDetails)> {
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

        let mut unplaced_bireachable: Vec<ItemLocationId> = Vec::new();
        let mut unplaced_oneway_reachable: Vec<ItemLocationId> = Vec::new();
        for (i, item_location_state) in state.item_location_state.iter().enumerate() {
            if item_location_state.placed_item.is_some() {
                continue;
            }
            if item_location_state.bireachable {
                unplaced_bireachable.push(i);
            } else if item_location_state.reachable {
                unplaced_oneway_reachable.push(i);
            }
        }
        unplaced_bireachable.shuffle(rng);
        unplaced_oneway_reachable.shuffle(rng);
        let mut attempt_num = 0;
        let mut new_state: RandomizationState;
        loop {
            let select_result = self.select_items(
                state,
                unplaced_bireachable.len(),
                unplaced_oneway_reachable.len(),
                attempt_num,
                rng,
            );
            if let Some(select_res) = select_result {
                new_state = RandomizationState {
                    step_num: state.step_num,
                    item_precedence: state.item_precedence.clone(),
                    item_location_state: state.item_location_state.clone(),
                    flag_location_state: state.flag_location_state.clone(),
                    items_remaining: select_res.new_items_remaining,
                    global_state: state.global_state.clone(),
                    done: false,
                    debug_data: None,
                };
                self.place_items(
                    &mut new_state,
                    &unplaced_bireachable,
                    &unplaced_oneway_reachable,
                    &select_res.key_items,
                    &select_res.other_items,
                );
                self.collect_items(&mut new_state);
                if iter::zip(&new_state.items_remaining, &self.initial_items_remaining)
                    .all(|(x, y)| x < y)
                {
                    // At least one instance of each item have been placed. This should be enough
                    // to ensure the game is beatable, so we are done.
                    new_state.done = true;
                    break;
                } else {
                    // println!("not all collected:");
                    // for (i, (x, y)) in iter::zip(&new_state.items_remaining, &self.initial_items_remaining).enumerate() {
                    //     if x >= y {
                    //         println!("item={}, remaining={x} ,initial={y}", self.game_data.item_isv.keys[i]);
                    //     }
                    // }
                    // println!("");
                }

                self.update_reachability(&mut new_state);
                if iter::zip(&new_state.item_location_state, &state.item_location_state)
                    .any(|(n, o)| n.bireachable && !o.reachable)
                {
                    // Progress: the new items unlock at least one bireachable item location that wasn't reachable before.
                    break;
                }
            } else {
                info!("Exhausted key item placement attempts");
                return None;
            }
            // println!("attempt failed");
            attempt_num += 1;
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
        Some((spoiler_summary, spoiler_details))
    }

    fn get_randomization(
        &self,
        state: &RandomizationState,
        spoiler_summaries: Vec<SpoilerSummary>,
        spoiler_details: Vec<SpoilerDetails>,
    ) -> Randomization {
        let item_placement = state
            .item_location_state
            .iter()
            .map(|x| x.placed_item.unwrap())
            .collect();
        let spoiler_escape =
            escape_timer::compute_escape_data(self.game_data, self.map, self.difficulty);
        let spoiler_log = SpoilerLog {
            summary: spoiler_summaries,
            escape: spoiler_escape,
            details: spoiler_details,
        };
        Randomization {
            difficulty: self.difficulty.clone(),
            map: self.map.clone(),
            item_placement,
            spoiler_log,
        }
    }

    pub fn randomize(&self, seed: usize) -> Option<Randomization> {
        let mut rng_seed = [0u8; 32];
        rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
        let mut rng = rand::rngs::StdRng::from_seed(rng_seed);
        let initial_global_state = {
            let items = vec![false; self.game_data.item_isv.keys.len()];
            let weapon_mask = self.game_data.get_weapon_mask(&items);
            GlobalState {
                tech: self.get_tech_vec(),
                items: items,
                flags: self.get_initial_flag_vec(),
                max_energy: 99,
                max_reserves: 0,
                max_missiles: 0,
                max_supers: 0,
                max_power_bombs: 0,
                weapon_mask: weapon_mask,
                shine_charge_tiles: self.difficulty.shine_charge_tiles,
            }
        };

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
        let mut item_precedence: Vec<Item> = (0..self.game_data.item_isv.keys.len())
            .map(|i| Item::try_from(i).unwrap())
            .collect();
        item_precedence.shuffle(&mut rng);
        let mut state = RandomizationState {
            step_num: 1,
            item_precedence,
            item_location_state: vec![
                initial_item_location_state;
                self.game_data.item_locations.len()
            ],
            flag_location_state: vec![
                initial_flag_location_state;
                self.game_data.flag_locations.len()
            ],
            items_remaining: self.initial_items_remaining.clone(),
            global_state: initial_global_state,
            done: false,
            debug_data: None,
        };
        self.update_reachability(&mut state);
        if !state.item_location_state.iter().any(|x| x.bireachable) {
            info!("No initially bireachable item locations");
            return None;
        }
        let mut spoiler_summary_vec: Vec<SpoilerSummary> = Vec::new();
        let mut spoiler_details_vec: Vec<SpoilerDetails> = Vec::new();
        while !state.done {
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
            info!("step={0}, bireachable={cnt_bireachable}, reachable={cnt_reachable}, placed={cnt_placed}, collected={cnt_collected}", state.step_num);
            match self.step(&mut state, &mut rng) {
                Some((spoiler_summary, spoiler_details)) => {
                    spoiler_summary_vec.push(spoiler_summary);
                    spoiler_details_vec.push(spoiler_details);
                }
                None => return None,
            }
            state.step_num += 1;
        }
        self.finish(&mut state);
        Some(self.get_randomization(&state, spoiler_summary_vec, spoiler_details_vec))
    }
}

// Spoiler log ---------------------------------------------------------

#[derive(Serialize, Deserialize, Debug)]
pub struct SpoilerRouteEntry {
    area: String,
    room: String,
    node: String,
    strat_name: String,
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
    area: String,
    room: String,
    node: String,
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
pub struct SpoilerItemSummary {
    item: String,
    location: SpoilerLocation,
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
}

impl<'a> Randomizer<'a> {
    fn get_vertex_info(&self, vertex_id: usize) -> VertexInfo {
        let (room_id, node_id, _obstacle_bitmask) = self.game_data.vertex_isv.keys[vertex_id];
        let room_ptr = self.game_data.room_ptr_by_id[&room_id];
        let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
        let area = self.map.area[room_idx];
        VertexInfo {
            area_name: self.game_data.area_names[area].clone(),
            room_name: self.game_data.room_json_map[&room_id]["name"]
                .as_str()
                .unwrap()
                .to_string(),
            node_name: self.game_data.node_json_map[&(room_id, node_id)]["name"]
                .as_str()
                .unwrap()
                .to_string(),
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
            let link = &self.links[link_idx as usize];
            let to_vertex_info = self.get_vertex_info(link.to_vertex_id);
            // info!("local: {:?}", local_state);
            // info!("{:?}", link);
            let new_local_state = apply_requirement(
                &link.requirement,
                &global_state,
                *local_state,
                false,
                difficulty,
            )
            .unwrap();
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

            let spoiler_entry = SpoilerRouteEntry {
                area: to_vertex_info.area_name,
                room: to_vertex_info.room_name,
                node: to_vertex_info.node_name,
                strat_name: link.strat_name.clone(),
                energy_remaining,
                reserves_remaining,
                missiles_remaining,
                supers_remaining,
                power_bombs_remaining,
            };
            // info!("spoiler: {:?}", spoiler_entry);
            route.push(spoiler_entry);
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
            let link = &self.links[link_idx as usize];
            let new_local_state = apply_requirement(
                &link.requirement,
                &global_state,
                local_state,
                true,
                difficulty,
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
            let power_bombs_used = if new_local_state.power_bombs_used < local_state.power_bombs_used {
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
            let link = &self.links[link_idx as usize];
            let to_vertex_info = self.get_vertex_info(link.to_vertex_id);
            let consumption = consumption_vec[i];
            let mut new_local_state = LocalState {
                energy_used: max(0, local_state.energy_used + consumption.energy_used),
                reserves_used: max(0, local_state.reserves_used + consumption.reserves_used),
                missiles_used: max(0, local_state.missiles_used + consumption.missiles_used),
                supers_used: max(0, local_state.supers_used + consumption.supers_used),
                power_bombs_used: max(0, local_state.power_bombs_used + consumption.power_bombs_used),
            };
            if new_local_state.energy_used >= global_state.max_energy {
                new_local_state.reserves_used += new_local_state.energy_used - (global_state.max_energy - 1);
                new_local_state.energy_used = global_state.max_energy - 1;
            }
            assert!(new_local_state.reserves_used <= global_state.max_reserves);
            assert!(new_local_state.missiles_used <= global_state.max_missiles);
            assert!(new_local_state.supers_used <= global_state.max_supers);
            assert!(new_local_state.power_bombs_used <= global_state.max_power_bombs);
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
            let spoiler_entry = SpoilerRouteEntry {
                area: to_vertex_info.area_name,
                room: to_vertex_info.room_name,
                node: to_vertex_info.node_name,
                strat_name: link.strat_name.clone(),
                energy_remaining,
                reserves_remaining,
                missiles_remaining,
                supers_remaining,
                power_bombs_remaining,
            };
            route.push(spoiler_entry);
            local_state = new_local_state;
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
        let forward_link_idxs: Vec<LinkIdx> =
            get_spoiler_route(&state.debug_data.as_ref().unwrap().forward, vertex_id);
        let reverse_link_idxs: Vec<LinkIdx> =
            get_spoiler_route(&state.debug_data.as_ref().unwrap().reverse, vertex_id);
        let mut local_state = LocalState::new();
        // info!("obtain");
        let obtain_route = self.get_spoiler_route(
            &state.global_state,
            &mut local_state,
            &forward_link_idxs,
            self.difficulty,
        );
        // info!("return");
        let return_route = self.get_spoiler_route_reverse(
            &state.global_state,
            local_state,
            &reverse_link_idxs,
            self.difficulty,
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
                let mut first_item = state.items_remaining[item as usize]
                    == self.initial_items_remaining[item as usize];
                if let Some(debug_options) = &self.difficulty.debug_options {
                    if debug_options.extended_spoiler {
                        first_item = true;
                    }
                }
                if first_item
                    && !state.item_location_state[i].collected
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
                let mut first_item = state.items_remaining[item as usize]
                    == self.initial_items_remaining[item as usize];
                if let Some(debug_options) = &self.difficulty.debug_options {
                    if debug_options.extended_spoiler {
                        first_item = true;
                    }
                }
                if first_item
                    && !state.item_location_state[i].collected
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
