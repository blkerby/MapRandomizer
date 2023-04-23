pub mod escape_timer;

use crate::{
    game_data::{
        self, Capacity, FlagId, Item, ItemLocationId, Link, Map, NodeId, Requirement, RoomId,
        TechId, VertexId,
    },
    traverse::{
        apply_requirement, get_spoiler_route, is_bireachable, traverse, GlobalState, LinkIdx,
        LocalState, TraverseResult,
    },
};
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
    Normal,
    Fast,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ItemPlacementStyle {
    Neutral,
    Forced,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ItemMarkers {
    Basic,
    Majors,
    Uniques,
    ThreeTiered,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum Objectives {
    Bosses,
    Minibosses,
    Metroids,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum MotherBrainFight {
    Vanilla,
    Short,
    Skipped,
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
    pub item_placement_style: ItemPlacementStyle,
    pub item_priorities: Vec<ItemPriorityGroup>,
    pub filler_items: Vec<Item>,
    pub resource_multiplier: f32,
    pub escape_timer_multiplier: f32,
    pub save_animals: bool,
    pub phantoon_proficiency: f32,
    pub draygon_proficiency: f32,
    pub ridley_proficiency: f32,
    pub botwoon_proficiency: f32,
    // Quality-of-life options:
    pub supers_double: bool,
    pub mother_brain_fight: MotherBrainFight,
    pub escape_movement_items: bool,
    pub escape_enemies_cleared: bool,
    pub mark_map_stations: bool,
    pub item_markers: ItemMarkers,
    pub all_items_spawn: bool,
    pub fast_elevators: bool,
    pub fast_doors: bool,
    // Objectives:
    pub objectives: Objectives,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_options: Option<DebugOptions>,
}

// Includes preprocessing specific to the map:
pub struct Randomizer<'a> {
    pub map: &'a Map,
    pub game_data: &'a GameData,
    pub difficulty_tiers: &'a [DifficultyConfig],
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

#[derive(Clone)]
struct DebugData {
    global_state: GlobalState,
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
    previous_debug_data: Option<DebugData>,
    key_visited_vertices: HashSet<usize>,
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
    room_coords: (usize, usize),
    node_name: String,
}

fn add_door_links(
    src_room_id: RoomId,
    src_node_id: NodeId,
    dst_room_id: RoomId,
    dst_node_id: NodeId,
    game_data: &GameData,
    links: &mut Vec<Link>,
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
            requirement: game_data::Requirement::Free,
            notable_strat_name: None,
            strat_name: "(Door transition)".to_string(),
            strat_notes: vec![],
        });
    }
}

struct Preprocessor<'a> {
    game_data: &'a GameData,
    door_map: HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
    // Cache of previously-processed or currently-processing inputs. This is used to avoid infinite
    // recursion in cases of circular dependencies (e.g. cycles of leaveWithGMode)
    preprocessed_output: HashMap<ByAddress<&'a Requirement>, Option<Requirement>>,
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

impl<'a> Preprocessor<'a> {
    pub fn new(game_data: &'a GameData, map: &'a Map) -> Self {
        let mut door_map: HashMap<(RoomId, NodeId), (RoomId, NodeId)> = HashMap::new();
        for &((src_exit_ptr, src_entrance_ptr), (dst_exit_ptr, dst_entrance_ptr), _) in &map.doors {
            let (src_room_id, src_node_id) =
                game_data.door_ptr_pair_map[&(src_exit_ptr, src_entrance_ptr)];
            let (dst_room_id, dst_node_id) =
                game_data.door_ptr_pair_map[&(dst_exit_ptr, dst_entrance_ptr)];
            // println!("({}, {}) <-> ({}, {})", src_room_id, src_node_id, dst_room_id, dst_node_id);
            door_map.insert((src_room_id, src_node_id), (dst_room_id, dst_node_id));
            door_map.insert((dst_room_id, dst_node_id), (src_room_id, src_node_id));
        }
        Preprocessor {
            game_data,
            door_map,
            preprocessed_output: HashMap::new(),
        }
    }

    fn preprocess_link(&mut self, link: &'a Link) -> Link {
        Link {
            from_vertex_id: link.from_vertex_id,
            to_vertex_id: link.to_vertex_id,
            requirement: self.preprocess_requirement(&link.requirement, link),
            notable_strat_name: link.notable_strat_name.clone(),
            strat_name: link.strat_name.clone(),
            strat_notes: link.strat_notes.clone(),
        }
    }

    fn preprocess_requirement(&mut self, req: &'a Requirement, link: &Link) -> Requirement {
        let key = ByAddress(req);
        if self.preprocessed_output.contains_key(&key) {
            if let Some(val) = &self.preprocessed_output[&key] {
                return val.clone();
            } else {
                // Circular dependency detected, which cannot be satisfied.
                println!("Circular requirement: {:?}", req);
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
            Requirement::CanComeInCharged {
                shinespark_tech_id,
                room_id,
                node_id,
                frames_remaining,
                shinespark_frames,
                excess_shinespark_frames,
            } => self.preprocess_can_come_in_charged(
                *shinespark_tech_id,
                *room_id,
                *node_id,
                *frames_remaining,
                *shinespark_frames,
                *excess_shinespark_frames,
                link,
            ),
            Requirement::ComeInWithGMode {
                room_id,
                node_ids,
                mode,
                artificial_morph,
            } => self.preprocess_come_in_with_gmode(
                *room_id,
                node_ids,
                mode,
                *artificial_morph,
                link,
            ),
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
        shinespark_tech_id: TechId,
        room_id: RoomId,
        node_id: NodeId,
        frames_remaining: i32,
        shinespark_frames: i32,
        excess_shinespark_frames: i32,
        _link: &Link,
    ) -> Requirement {
        if let Some(&(other_room_id, other_node_id)) = self.door_map.get(&(room_id, node_id)) {
            let runways = &self.game_data.node_runways_map[&(room_id, node_id)];
            let other_runways = &self.game_data.node_runways_map[&(other_room_id, other_node_id)];
            let can_leave_charged_vec =
                &self.game_data.node_can_leave_charged_map[&(other_room_id, other_node_id)];
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
                if runway.length < 15 {
                    continue;
                }
                let req = Requirement::ShineCharge {
                    shinespark_tech_id,
                    used_tiles: runway.length as f32,
                    shinespark_frames,
                    excess_shinespark_frames,
                };
                req_vec.push(Requirement::make_and(vec![req, runway.requirement.clone()]));
            }

            // Strats for other-room runways:
            for runway in other_runways {
                if runway.length < 15 {
                    continue;
                }
                let req = Requirement::ShineCharge {
                    shinespark_tech_id,
                    used_tiles: runway.length as f32,
                    shinespark_frames,
                    excess_shinespark_frames,
                };
                req_vec.push(Requirement::make_and(vec![req, runway.requirement.clone()]));
            }

            // Strats for cross-room combined runways:
            for runway in runways {
                if !runway.usable_coming_in {
                    continue;
                }
                for other_runway in other_runways {
                    let used_tiles = runway.length + other_runway.length - 1;
                    if used_tiles < 15 {
                        continue;
                    }
                    let req = Requirement::ShineCharge {
                        shinespark_tech_id,
                        used_tiles: used_tiles as f32,
                        shinespark_frames,
                        excess_shinespark_frames,
                    };
                    req_vec.push(Requirement::make_and(vec![
                        req,
                        runway.requirement.clone(),
                        other_runway.requirement.clone(),
                    ]));
                }
            }

            // Strats for canLeaveCharged from other room:
            for can_leave_charged in can_leave_charged_vec {
                if can_leave_charged.frames_remaining < frames_remaining {
                    continue;
                }
                let req = Requirement::ShineCharge {
                    shinespark_tech_id,
                    used_tiles: can_leave_charged.used_tiles as f32,
                    shinespark_frames: shinespark_frames
                        + can_leave_charged.shinespark_frames.unwrap_or(0),
                    excess_shinespark_frames,
                };
                req_vec.push(Requirement::make_and(vec![
                    req,
                    can_leave_charged.requirement.clone(),
                ]));
            }

            // println!("Strats: {:?}\n", req_vec);
            let out = Requirement::make_or(req_vec);
            out
        } else {
            println!(
                "In canComeInCharged, ({}, {}) is not door node?",
                room_id, node_id
            );
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
        let (other_room_id, other_node_id) = self.door_map[&(room_id, node_id)];
        let runways = &self.game_data.node_runways_map[&(other_room_id, other_node_id)];
        let mut req_vec: Vec<Requirement> = vec![];
        for runway in runways {
            let effective_length = runway.length as f32 + runway.open_end as f32 * 0.5;
            // println!(
            //     "  {}: length={}, open_end={}, physics={}, heated={}, req={:?}",
            //     runway.name, runway.length, runway.open_end, runway.physics, runway.heated, runway.requirement
            // );
            if effective_length < used_tiles {
                continue;
            }
            let mut reqs: Vec<Requirement> = vec![];
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
                reqs.push(runway.requirement.clone());
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

    fn preprocess_come_in_with_gmode(
        &mut self,
        room_id: RoomId,
        node_ids: &[NodeId],
        mode: &str,
        artificial_morph: bool,
        link: &Link,
    ) -> Requirement {
        let gmode_tech_id = self.game_data.tech_isv.index_by_key["canEnterGMode"];
        let gmode_immobile_tech_id = self.game_data.tech_isv.index_by_key["canEnterGModeImmobile"];
        let artificial_morph_tech_id = self.game_data.tech_isv.index_by_key["canArtificialMorph"];
        let morph_item_id = self.game_data.item_isv.index_by_key["Morph"];
        let xray_item_id = self.game_data.item_isv.index_by_key["XRayScope"];
        let mut req_or_list: Vec<Requirement> = Vec::new();
        for &node_id in node_ids {
            if let Some(&(other_room_id, other_node_id)) = self.door_map.get(&(room_id, node_id)) {
                let gmode_immobile_opt = self
                    .game_data
                    .node_gmode_immobile_map
                    .get(&(other_room_id, other_node_id));
                if mode == "direct" || mode == "any" {
                    let leave_with_gmode_setup_vec = &self
                        .game_data
                        .node_leave_with_gmode_setup_map[&(other_room_id, other_node_id)];
                    for leave_with_gmode_setup in leave_with_gmode_setup_vec {
                        let mut req_and_list: Vec<Requirement> = Vec::new();
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
                            immobile_req_vec.push(gmode_immobile.requirement.clone());
                            immobile_req_vec.push(Requirement::Tech(gmode_immobile_tech_id));
                            immobile_req_vec.push(Requirement::ReserveTrigger { 
                                min_reserve_energy: 1, 
                                max_reserve_energy: 400,
                            });
                            Requirement::make_and(immobile_req_vec)
                        } else {
                            Requirement::Never
                        };
                        req_and_list.push(Requirement::make_or(vec![mobile_req, immobile_req]));

                        req_or_list.push(Requirement::make_and(req_and_list));
                    }
                }

                if mode == "indirect" || mode == "any" {
                    let leave_with_gmode_vec =
                        &self.game_data.node_leave_with_gmode_map[&(other_room_id, other_node_id)];
                    for leave_with_gmode in leave_with_gmode_vec {
                        if !artificial_morph || leave_with_gmode.artificial_morph {
                            req_or_list.push(
                                self.preprocess_requirement(&leave_with_gmode.requirement, link),
                            );
                        }
                    }
                }
            }
        }

        let out = Requirement::make_or(req_or_list);
        println!(
            "{} ({}) {:?} {}: {:?}",
            self.game_data.room_json_map[&room_id]["name"], room_id, node_ids, link.strat_name, out
        );
        out
    }
}

impl<'r> Randomizer<'r> {
    pub fn new(
        map: &'r Map,
        difficulty_tiers: &'r [DifficultyConfig],
        game_data: &'r GameData,
    ) -> Randomizer<'r> {
        let mut preprocessor = Preprocessor::new(game_data, map);
        let mut links: Vec<Link> = game_data
            .links
            .iter()
            .map(|x| preprocessor.preprocess_link(x))
            .collect();
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

            add_door_links(
                src_room_id,
                src_node_id,
                dst_room_id,
                dst_node_id,
                game_data,
                &mut links,
            );
            if bidirectional {
                add_door_links(
                    dst_room_id,
                    dst_node_id,
                    src_room_id,
                    src_node_id,
                    game_data,
                    &mut links,
                );
            }
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
            difficulty_tiers,
        }
    }

    fn get_tech_vec(&self, tier: usize) -> Vec<bool> {
        let tech_set: HashSet<String> = self.difficulty_tiers[tier]
            .tech
            .iter()
            .map(|x| x.clone())
            .collect();
        self.game_data
            .tech_isv
            .keys
            .iter()
            .map(|x| tech_set.contains(x))
            .collect()
    }

    fn get_strat_vec(&self, tier: usize) -> Vec<bool> {
        let strat_set: HashSet<String> = self.difficulty_tiers[tier]
            .notable_strats
            .iter()
            .map(|x| x.clone())
            .collect();
        self.game_data
            .notable_strat_isv
            .keys
            .iter()
            .map(|x| strat_set.contains(x))
            .collect()
    }

    fn get_initial_flag_vec(&self) -> Vec<bool> {
        let mut flag_vec = vec![false; self.game_data.flag_isv.keys.len()];
        let tourian_open_idx = self.game_data.flag_isv.index_by_key["f_TourianOpen"];
        flag_vec[tourian_open_idx] = true;
        if self.difficulty_tiers[0].all_items_spawn {
            let all_items_spawn_idx = self.game_data.flag_isv.index_by_key["f_AllItemsSpawn"];
            flag_vec[all_items_spawn_idx] = true;
        }
        flag_vec
    }

    fn update_reachability(&self, state: &mut RandomizationState) {
        let num_vertices = self.game_data.vertex_isv.keys.len();
        let start_vertex_id = self.game_data.vertex_isv.index_by_key[&(8, 5, 0)]; // Landing site
        let forward = traverse(
            &self.links,
            None,
            &state.global_state,
            num_vertices,
            start_vertex_id,
            false,
            &self.difficulty_tiers[0],
            self.game_data,
        );
        let reverse = traverse(
            &self.links,
            None,
            &state.global_state,
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
            ProgressionRate::Normal => num_bireachable,
            ProgressionRate::Fast => num_bireachable,
        };
        let filtered_item_precedence: Vec<Item> = state
            .item_precedence
            .iter()
            .copied()
            .filter(|&item| {
                state.items_remaining[item as usize] == self.initial_items_remaining[item as usize]
                    || (item == Item::Missile && state.items_remaining[item as usize] > 0)
            })
            .collect();
        let num_key_items_remaining = filtered_item_precedence.len();
        let num_items_remaining: usize = state.items_remaining.iter().sum();
        let mut num_key_items_to_place = match self.difficulty_tiers[0].progression_rate {
            ProgressionRate::Slow => 1,
            ProgressionRate::Normal => f32::ceil(
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

        let expansion_item_set: HashSet<Item> =
            [Item::ETank, Item::ReserveTank, Item::Super, Item::PowerBomb]
                .into_iter()
                .collect();
        let mut item_types_to_mix: Vec<Item> = vec![Item::Missile];
        let mut item_types_to_delay: Vec<Item> = vec![];

        for &item in &state.item_precedence {
            if !expansion_item_set.contains(&item) || item == Item::Missile {
                continue;
            }
            if self.difficulty_tiers[0].filler_items.contains(&item)
                || state.items_remaining[item as usize]
                    < self.initial_items_remaining[item as usize]
            {
                item_types_to_mix.push(item);
            } else {
                item_types_to_delay.push(item);
            }
        }

        for &item in &state.item_precedence {
            if expansion_item_set.contains(&item) || item == Item::Missile {
                continue;
            }
            if self.difficulty_tiers[0].filler_items.contains(&item) {
                item_types_to_mix.push(item);
            } else {
                item_types_to_delay.push(item);
            }
        }

        // println!("mix: {:?}, delay: {:?}", item_types_to_mix, item_types_to_delay);

        let mut items_to_mix: Vec<Item> = Vec::new();
        for &item in &item_types_to_mix {
            for _ in 0..new_items_remaining[item as usize] {
                items_to_mix.push(item);
            }
        }
        let mut items_to_delay: Vec<Item> = Vec::new();
        for &item in &item_types_to_delay {
            for _ in 0..new_items_remaining[item as usize] {
                items_to_delay.push(item);
            }
        }
        let mut other_items_to_place: Vec<Item> = items_to_mix;
        other_items_to_place.shuffle(rng);
        other_items_to_place.extend(items_to_delay);
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

    fn get_init_traverse(
        &self,
        state: &RandomizationState,
        init_traverse: Option<&TraverseResult>,
    ) -> Option<TraverseResult> {
        if let Some(init) = init_traverse {
            let mut out = init.clone();
            for v in 0..init.local_states.len() {
                if !state.key_visited_vertices.contains(&v) {
                    out.local_states[v] = None;
                    out.cost[v] = f32::INFINITY;
                    out.start_trail_ids[v] = None;
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
        let start_vertex_id = self.game_data.vertex_isv.index_by_key[&(8, 5, 0)]; // Landing site

        for tier in 1..self.difficulty_tiers.len() {
            let difficulty = &self.difficulty_tiers[tier];
            let mut tmp_global = state.global_state.clone();
            tmp_global.tech = self.get_tech_vec(tier);
            tmp_global.notable_strats = self.get_strat_vec(tier);
            tmp_global.shine_charge_tiles = difficulty.shine_charge_tiles;
            let traverse_result = traverse(
                &self.links,
                self.get_init_traverse(state, init_traverse),
                &tmp_global,
                num_vertices,
                start_vertex_id,
                false,
                difficulty,
                self.game_data,
            );

            for (i, &item_location_id) in bireachable_locations.iter().enumerate() {
                let mut is_reachable = false;
                for &v in &self.game_data.item_vertex_ids[item_location_id] {
                    if traverse_result.local_states[v].is_some() {
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
        state: &RandomizationState,
        new_state: &mut RandomizationState,
        bireachable_locations: &[ItemLocationId],
        other_locations: &[ItemLocationId],
        key_items_to_place: &[Item],
        other_items_to_place: &[Item],
    ) {
        info!(
            "Placing {:?}, {:?}",
            key_items_to_place, other_items_to_place
        );
        // println!("# bireachable = {}", bireachable_locations.len());
        let mut new_bireachable_locations: Vec<ItemLocationId> = bireachable_locations.to_vec();
        if self.difficulty_tiers.len() > 1 {
            let traverse_result = match state.previous_debug_data.as_ref() {
                Some(x) => Some(&x.forward),
                None => None,
            };
            for i in 0..key_items_to_place.len() {
                let (hard_idx, tier) = self.find_hard_location(
                    new_state,
                    &new_bireachable_locations[i..],
                    traverse_result,
                );
                info!(
                    "{:?} in tier {} (of {})",
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
                let route =
                    get_spoiler_route(&state.debug_data.as_ref().unwrap().forward, hard_vertex_id);
                for &link_idx in &route {
                    let vertex_id = self.links[link_idx as usize].to_vertex_id;
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

    fn finish(&self, state: &mut RandomizationState) {
        let mut remaining_items: Vec<Item> = Vec::new();
        for item_id in 0..self.game_data.item_isv.keys.len() {
            for _ in 0..state.items_remaining[item_id] {
                remaining_items.push(Item::try_from(item_id).unwrap());
            }
        }
        info!("Finishing with {:?}", remaining_items);
        let mut idx = 0;
        for item_loc_state in &mut state.item_location_state {
            if item_loc_state.placed_item.is_none() {
                item_loc_state.placed_item = Some(remaining_items[idx]);
                idx += 1;
            }
        }
        assert!(idx == remaining_items.len());
    }

    fn step<R: Rng + Clone>(
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
        let mut attempt_num = 0;
        let mut new_state: RandomizationState;
        let mut key_items_to_place: Vec<Item>;
        let mut other_items_to_place: Vec<Item>;
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
                    previous_debug_data: None,
                    key_visited_vertices: HashSet::new(),
                };
                key_items_to_place = select_res.key_items;
                other_items_to_place = select_res.other_items;

                // info!("Trying placing {:?}", key_items_to_place);
                for &item in placed_uncollected_bireachable_items.iter().chain(
                    key_items_to_place
                        .iter()
                        .chain(other_items_to_place.iter())
                        .take(unplaced_bireachable.len()),
                ) {
                    new_state.global_state.collect(item, self.game_data);
                }

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

                let gives_expansion =
                    if self.difficulty_tiers[0].progression_rate == ProgressionRate::Slow {
                        iter::zip(&new_state.item_location_state, &state.item_location_state)
                            .any(|(n, o)| n.bireachable && !o.reachable)
                    } else {
                        iter::zip(&new_state.item_location_state, &state.item_location_state)
                            .any(|(n, o)| n.bireachable && !o.bireachable)
                    };

                if num_one_way_reachable < one_way_reachable_limit && gives_expansion {
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
                &state,
                &mut new_state,
                &unplaced_bireachable,
                &unplaced_oneway_reachable,
                &key_items_to_place,
                &other_items_to_place,
            );
        } else {
            // In Normal and Fast progression, only place items at bireachable locations. We defer placing items at
            // one-way-reachable locations so that they may get key items placed there later after
            // becoming bireachable.
            self.place_items(
                &state,
                &mut new_state,
                &unplaced_bireachable,
                &[],
                &key_items_to_place,
                &other_items_to_place,
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
        Some((spoiler_summary, spoiler_details))
    }

    fn get_randomization(
        &self,
        state: &RandomizationState,
        spoiler_summaries: Vec<SpoilerSummary>,
        spoiler_details: Vec<SpoilerDetails>,
        debug_data_vec: Vec<DebugData>,
    ) -> Randomization {
        // Compute the first step on which each node becomes reachable/bireachable:
        let mut node_reachable_step: HashMap<(RoomId, NodeId), usize> = HashMap::new();
        let mut node_bireachable_step: HashMap<(RoomId, NodeId), usize> = HashMap::new();
        let mut map_tile_reachable_step: HashMap<(RoomId, (usize, usize)), usize> = HashMap::new();
        let mut map_tile_bireachable_step: HashMap<(RoomId, (usize, usize)), usize> =
            HashMap::new();

        for (step, debug_data) in debug_data_vec.iter().enumerate() {
            for (v, (room_id, node_id, _obstacle_bitmask)) in
                self.game_data.vertex_isv.keys.iter().enumerate()
            {
                if node_bireachable_step.contains_key(&(*room_id, *node_id)) {
                    continue;
                }
                if is_bireachable(
                    &debug_data.global_state,
                    &debug_data.forward.local_states[v],
                    &debug_data.reverse.local_states[v],
                ) {
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
                if debug_data.forward.local_states[v].is_some() {
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

        let item_placement = state
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
                let room = self.game_data.room_json_map[&room]["name"]
                    .as_str()
                    .unwrap()
                    .to_string();
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
                    map: g.map.clone(),
                    map_reachable_step,
                    map_bireachable_step,
                    coords: *c,
                }
            })
            .collect();
        let spoiler_escape =
            escape_timer::compute_escape_data(self.game_data, self.map, &self.difficulty_tiers[0]);
        let spoiler_log = SpoilerLog {
            summary: spoiler_summaries,
            escape: spoiler_escape,
            details: spoiler_details,
            all_items: spoiler_all_items,
            all_rooms: spoiler_all_rooms,
        };
        Randomization {
            difficulty: self.difficulty_tiers[0].clone(),
            map: self.map.clone(),
            item_placement,
            spoiler_log,
        }
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

    pub fn randomize(&self, seed: usize) -> Option<Randomization> {
        let mut rng_seed = [0u8; 32];
        rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
        let mut rng = rand::rngs::StdRng::from_seed(rng_seed);
        let initial_global_state = {
            let items = vec![false; self.game_data.item_isv.keys.len()];
            let weapon_mask = self.game_data.get_weapon_mask(&items);
            GlobalState {
                tech: self.get_tech_vec(0),
                notable_strats: self.get_strat_vec(0),
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
        let item_precedence: Vec<Item> =
            self.get_item_precedence(&self.difficulty_tiers[0].item_priorities, &mut rng);
        info!("Item precedence: {:?}", item_precedence);
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
            previous_debug_data: None,
            key_visited_vertices: HashSet::new(),
        };
        self.update_reachability(&mut state);
        if !state.item_location_state.iter().any(|x| x.bireachable) {
            info!("No initially bireachable item locations");
            return None;
        }
        let mut spoiler_summary_vec: Vec<SpoilerSummary> = Vec::new();
        let mut spoiler_details_vec: Vec<SpoilerDetails> = Vec::new();
        let mut debug_data_vec: Vec<DebugData> = Vec::new();
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
                    // Append `debug_data` is present (which it always should be except after the final step)
                    if let Some(debug_data) = &state.previous_debug_data {
                        debug_data_vec.push(debug_data.clone());
                    }
                }
                None => return None,
            }
            state.step_num += 1;
        }
        self.finish(&mut state);
        Some(self.get_randomization(
            &state,
            spoiler_summary_vec,
            spoiler_details_vec,
            debug_data_vec,
        ))
    }
}

// Spoiler log ---------------------------------------------------------

#[derive(Serialize, Deserialize, Debug)]
pub struct SpoilerRouteEntry {
    area: String,
    room: String,
    node: String,
    obstacles_bitmask: usize,
    coords: (usize, usize),
    strat_name: String,
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
    area: String,
    room: String,
    node: String,
    coords: (usize, usize),
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
    map: Vec<Vec<u8>>,
    map_reachable_step: Vec<Vec<u8>>,
    map_bireachable_step: Vec<Vec<u8>>,
    coords: (usize, usize),
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
    pub all_items: Vec<SpoilerItemLoc>,
    pub all_rooms: Vec<SpoilerRoomLoc>,
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
            let (_, _, to_obstacles_mask) = self.game_data.vertex_isv.keys[link.to_vertex_id];
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
                obstacles_bitmask: to_obstacles_mask,
                coords: to_vertex_info.room_coords,
                strat_name: link.strat_name.clone(),
                strat_notes: link.strat_notes.clone(),
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
            let link = &self.links[link_idx as usize];
            let to_vertex_info = self.get_vertex_info(link.to_vertex_id);
            let (_, _, to_obstacles_mask) = self.game_data.vertex_isv.keys[link.to_vertex_id];
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
                obstacles_bitmask: to_obstacles_mask,
                coords: to_vertex_info.room_coords,
                strat_name: link.strat_name.clone(),
                strat_notes: link.strat_notes.clone(),
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
