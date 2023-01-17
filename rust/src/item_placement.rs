use crate::{
    game_data::{self, Item, ItemLocationId, Link, NodePtr, VertexId},
    traverse::{is_bireachable, traverse, GlobalState},
};
use hashbrown::HashSet;
use json::JsonValue;
use rand::{seq::SliceRandom, Rng};
use std::{cmp::min, convert::TryFrom, iter};

use crate::game_data::GameData;

#[derive(Clone)]
pub enum ItemPlacementStrategy {
    Open,
    Semiclosed,
    Closed,
}

#[derive(Clone)]
pub struct DifficultyConfig {
    pub tech: Vec<String>,
    pub shine_charge_tiles: i32,
    pub item_placement_strategy: ItemPlacementStrategy,
}

// Includes preprocessing specific to the map:
pub struct Randomizer<'a> {
    pub game_data: &'a GameData,
    pub difficulty: &'a DifficultyConfig,
    pub links: Vec<Link>,
    pub initial_items_remaining: Vec<usize>, // Corresponds to GameData.items_isv (one count per distinct item name)
}

enum StepResult {
    Progress,
    Done,
    Fail,
}

#[derive(Clone)]
struct ItemLocationState {
    pub placed_item: Option<Item>,
    pub collected: bool,
    pub reachable: bool,
    pub bireachable: bool,
}

// State that changes over the course of item placement attempts
#[derive(Clone)]
struct RandomizationState {
    item_precedence: Vec<Item>, // An ordering of the 21 distinct item names. The game will prioritize placing key items earlier in the list.
    item_location_state: Vec<ItemLocationState>, // Corresponds to GameData.item_locations (one record for each of 100 item locations)
    flag_bireachable: Vec<bool>,  // Corresponds to GameData.flag_locations
    items_remaining: Vec<usize>, // Corresponds to GameData.items_isv (one count for each of 21 distinct item names)
    global_state: GlobalState,
}

pub struct Randomization {
    pub item_placement: Vec<Item>,
    // TODO: add spoiler log
}

fn parse_door_ptr(x: &JsonValue) -> Option<NodePtr> {
    if x.is_number() {
        Some(x.as_usize().unwrap())
    } else if x.is_null() {
        None
    } else {
        panic!("Unexpected door pointer: {}", x);
    }
}

struct SelectItemsOutput {
    key_items: Vec<Item>,
    other_items: Vec<Item>,
    new_items_remaining: Vec<usize>,
}

impl<'r> Randomizer<'r> {
    pub fn new(
        map: &'r JsonValue,
        difficulty: &'r DifficultyConfig,
        game_data: &'r GameData,
    ) -> Randomizer<'r> {
        assert!(map["doors"].is_array());
        let mut door_edges: Vec<(VertexId, VertexId)> = Vec::new();
        for door_json in map["doors"].members() {
            let src_exit_ptr = parse_door_ptr(&door_json[0][0]);
            let src_entrance_ptr = parse_door_ptr(&door_json[0][1]);
            let dst_exit_ptr = parse_door_ptr(&door_json[1][0]);
            let dst_entrance_ptr = parse_door_ptr(&door_json[1][1]);
            let bidirectional = door_json[2].as_bool().unwrap();
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

    fn get_flag_vec(&self) -> Vec<bool> {
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
            self.game_data,
        );

        // for (i, &(room_id, node_id, obstacle_bitmask)) in self.game_data.vertex_isv.keys.iter().enumerate() {
        //     if forward.local_states[i].is_some() {
        //         let room_name = &self.game_data.room_json_map[&room_id]["name"];
        //         let node_name = &self.game_data.node_json_map[&(room_id, node_id)]["name"];
        //         println!("{room_name}: {node_name} ({obstacle_bitmask})");
        //     }
        // }

        let reverse = traverse(
            &self.links,
            &state.global_state,
            num_vertices,
            start_vertex_id,
            true,
            self.game_data,
        );
        for (i, vertex_ids) in self.game_data.item_vertex_ids.iter().enumerate() {
            for &v in vertex_ids {
                if forward.local_states[v].is_some() {
                    state.item_location_state[i].reachable = true;
                    if is_bireachable(
                        &state.global_state,
                        &forward.local_states[v],
                        &reverse.local_states[v],
                    ) {
                        state.item_location_state[i].bireachable = true;
                    }
                }
            }
        }
        for (i, vertex_ids) in self.game_data.flag_vertex_ids.iter().enumerate() {
            for &v in vertex_ids {
                if is_bireachable(
                    &state.global_state,
                    &forward.local_states[v],
                    &reverse.local_states[v],
                ) {
                    state.flag_bireachable[i] = true;
                }
            }
        }
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
            // println!("dumping key items");
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
                if new_items_remaining[Item::Super as usize]
                    < self.initial_items_remaining[Item::Super as usize]
                {
                    item_types_to_mix.push(Item::Super);
                } else {
                    item_types_to_delay.push(Item::Super);
                }
                if new_items_remaining[Item::PowerBomb as usize]
                    < self.initial_items_remaining[Item::PowerBomb as usize]
                {
                    item_types_to_mix.push(Item::PowerBomb);
                } else {
                    item_types_to_delay.push(Item::PowerBomb);
                }
            }
            ItemPlacementStrategy::Closed => {
                item_types_to_mix = vec![Item::Missile];
                if new_items_remaining[Item::PowerBomb as usize]
                    < self.initial_items_remaining[Item::PowerBomb as usize]
                    && new_items_remaining[Item::Super as usize]
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

    fn step<R: Rng>(&self, state: &mut RandomizationState, rng: &mut R) -> StepResult {
        loop {
            let mut any_new_flag = false;
            for i in 0..self.game_data.flag_locations.len() {
                let flag_id = self.game_data.flag_locations[i].2;
                if state.global_state.flags[flag_id] {
                    continue;
                }
                if state.flag_bireachable[i] {
                    any_new_flag = true;
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
        loop {
            let select_result = self.select_items(
                state,
                unplaced_bireachable.len(),
                unplaced_oneway_reachable.len(),
                attempt_num,
                rng,
            );
            if let Some(select_res) = select_result {
                let mut new_state = state.clone();
                new_state.items_remaining = select_res.new_items_remaining;
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
                    new_state.clone_into(state);
                    self.finish(state);
                    return StepResult::Done;
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
                    new_state.clone_into(state);
                    return StepResult::Progress;
                }
            } else {
                println!("Exhausted key item placement attempts");
                return StepResult::Fail;
            }
            // println!("attempt failed");
            attempt_num += 1;
        }
    }

    fn get_randomization(&self, state: &RandomizationState) -> Randomization {
        let item_placement = state
            .item_location_state
            .iter()
            .map(|x| x.placed_item.unwrap())
            .collect();
        Randomization { item_placement }
    }

    pub fn randomize<R: Rng>(&self, rng: &mut R) -> Option<Randomization> {
        let initial_global_state = {
            let items = vec![false; self.game_data.item_isv.keys.len()];
            let weapon_mask = self.game_data.get_weapon_mask(&items);
            GlobalState {
                tech: self.get_tech_vec(),
                items: items,
                flags: self.get_flag_vec(),
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
        };
        let mut item_precedence: Vec<Item> = (0..self.game_data.item_isv.keys.len())
            .map(|i| Item::try_from(i).unwrap())
            .collect();
        item_precedence.shuffle(rng);
        let mut state = RandomizationState {
            item_precedence,
            item_location_state: vec![
                initial_item_location_state;
                self.game_data.item_locations.len()
            ],
            flag_bireachable: vec![false; self.game_data.flag_locations.len()],
            items_remaining: self.initial_items_remaining.clone(),
            global_state: initial_global_state,
        };
        self.update_reachability(&mut state);
        if !state.item_location_state.iter().any(|x| x.bireachable) {
            println!("No initially bireachable item locations");
            return None;
        }
        let mut step_num = 1;
        loop {
            let cnt_collected = state.item_location_state.iter().filter(|x| x.collected).count();
            let cnt_placed = state.item_location_state.iter().filter(|x| x.placed_item.is_some()).count();
            let cnt_reachable = state.item_location_state.iter().filter(|x| x.reachable).count();
            let cnt_bireachable = state.item_location_state.iter().filter(|x| x.bireachable).count();
            println!("step={step_num}, bireachable={cnt_bireachable}, reachable={cnt_reachable}, placed={cnt_placed}, collected={cnt_collected}");
            match self.step(&mut state, rng) {
                StepResult::Progress => {}
                StepResult::Done => break,
                StepResult::Fail => return None,
            }
            step_num += 1;
        }
        Some(self.get_randomization(&state))
    }
}
