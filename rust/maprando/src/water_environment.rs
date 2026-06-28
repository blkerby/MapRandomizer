use std::collections::{HashSet, VecDeque};

use anyhow::{Result, ensure};
use hashbrown::HashMap;
use json;
use log::info;
use maprando_game::{GameData, Item, ItemId, Map, Requirement, RoomId};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::settings::{ExperimentalSettings, RandomizerSettings};

/// Default donor room for water FX: Fish Tank (fully underwater).
pub const DEFAULT_WATER_DONOR_ROOM_PTR: usize = 0x7D017;
pub const DEFAULT_WATER_DONOR_ROOM_ID: RoomId = 173;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct WaterAssignment {
    /// Pause-map liquid surface in room screen-row coordinates (same as `map_tiles.json`).
    /// `0.0` means fully flooded; higher values leave upper rows dry.
    pub liquid_level: f32,
    /// ROM address of the vanilla room whose water FX template is copied.
    pub donor_room_ptr: usize,
}

const EXCLUDED_ROOM_NAMES: &[&str] = &[
    "Rising Tide",
    "Volcano Room",
    "Amphitheatre",
    "Acid Statue Room",
    "Climb",
    "Speed Booster Hall",
    "Mother Brain Room",
    "Tourian Escape Room 1",
    "Tourian Escape Room 2",
    "Tourian Escape Room 3",
    "Tourian Escape Room 4",
    "Pants Room",
    "East Pants Room",
    "Homing Geemer Room",
    "West Ocean",
    "Toilet Bowl",
    "Ceres Elevator Room",
    "Ceres Jump Tutorial Room",
    "Ceres Stairwell Room",
    "Ceres Crateria Landing Room",
    "Ceres Back Door Room",
    "Ceres Dead End Room",
    "Ceres Small Energy Refill Room",
    "Ceres Small Missile Refill Room",
    "Metroid Quarters",
    "Metroid Room 1",
    "Metroid Room 2",
    "Metroid Room 3",
    "Metroid Room 4",
    "Metroid Room 5",
    "Metroid Room 6",
    "Fish Tank",
    "Maridia Tube",
    "The Beach",
    "Mama Turtle Room",
];

fn room_name(game_data: &GameData, room_id: RoomId) -> String {
    game_data.room_json_map[&room_id]["name"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

pub fn room_has_water(game_data: &GameData, room_id: RoomId) -> bool {
    for ((rid, _), node_json) in &game_data.node_json_map {
        if *rid != room_id {
            continue;
        }
        if node_json["nodeType"].as_str() != Some("door") {
            continue;
        }
        if let Some(physics) = node_json["doorEnvironments"][0]["physics"].as_str() {
            if physics == "water" {
                return true;
            }
        }
    }
    false
}

fn is_room_eligible(game_data: &GameData, room_id: RoomId, room_idx: usize, map: &Map) -> bool {
    if !map.room_mask[room_idx] {
        return false;
    }
    let name = room_name(game_data, room_id);
    if EXCLUDED_ROOM_NAMES.contains(&name.as_str()) {
        return false;
    }
    if room_has_water(game_data, room_id) {
        return false;
    }
    // Single-tile save/refill/map rooms are poor candidates for flooding.
    let geometry = &game_data.room_geometry[room_idx];
    if geometry.map.len() == 1 && geometry.map[0].len() == 1 {
        return false;
    }
    true
}

const NEAR_SPAWN_PREFERRED_ROOMS: &[&str] = &[
    "Pre-Map Flyway",
    "Gauntlet Entrance",
    "Terminator Room",
    "Parlor And Alcatraz",
    "Construction Zone",
    "Crateria Tube",
    "Red Brinstar Elevator Room",
    "Alcatraz Tunnel",
    "Cathedral Entrance",
    "Blue Brinstar Energy Tank Room",
    "Early Supers Room",
];

const NEAR_SPAWN_MAX_DEPTH: usize = 4;

fn build_room_adjacency(game_data: &GameData) -> HashMap<RoomId, Vec<RoomId>> {
    let mut adj: HashMap<RoomId, HashSet<RoomId>> = HashMap::new();
    for link in &game_data.links {
        let from_room = game_data.vertex_isv.keys[link.from_vertex_id].room_id;
        let to_room = game_data.vertex_isv.keys[link.to_vertex_id].room_id;
        if from_room != to_room {
            adj.entry(from_room).or_default().insert(to_room);
            adj.entry(to_room).or_default().insert(from_room);
        }
    }
    adj.into_iter()
        .map(|(room_id, neighbors)| {
            let mut neighbors: Vec<_> = neighbors.into_iter().collect();
            neighbors.sort_unstable();
            (room_id, neighbors)
        })
        .collect()
}

pub fn get_eligible_rooms_near_spawn(
    game_data: &GameData,
    map: &Map,
    max_count: usize,
) -> Vec<(RoomId, usize)> {
    if max_count == 0 {
        return vec![];
    }

    let mut result = vec![];
    let mut chosen = HashSet::new();
    for &preferred_name in NEAR_SPAWN_PREFERRED_ROOMS {
        if result.len() >= max_count {
            break;
        }
        for (room_idx, room) in game_data.room_geometry.iter().enumerate() {
            if room.name != preferred_name {
                continue;
            }
            let room_id = room.room_id;
            if chosen.insert(room_id)
                && is_room_eligible(game_data, room_id, room_idx, map)
            {
                result.push((room_id, room_idx));
            }
            break;
        }
    }

    if result.len() >= max_count {
        return result;
    }

    let spawn_room_id = game_data.room_geometry[game_data.ship_room_idx].room_id;
    let adj = build_room_adjacency(game_data);
    let mut visited = HashSet::new();
    let mut queue = VecDeque::from([(spawn_room_id, 0usize)]);
    visited.insert(spawn_room_id);

    while let Some((room_id, depth)) = queue.pop_front() {
        let room_idx = game_data.room_idx_by_id[&room_id];
        if depth > 0
            && chosen.insert(room_id)
            && is_room_eligible(game_data, room_id, room_idx, map)
        {
            result.push((room_id, room_idx));
            if result.len() >= max_count {
                break;
            }
        }
        if depth < NEAR_SPAWN_MAX_DEPTH {
            for neighbor in adj.get(&room_id).into_iter().flat_map(|v| v.iter()) {
                if visited.insert(*neighbor) {
                    queue.push_back((*neighbor, depth + 1));
                }
            }
        }
    }
    result
}

pub fn get_eligible_water_rooms(game_data: &GameData, map: &Map) -> Vec<(RoomId, usize)> {
    let mut eligible = vec![];
    for (room_idx, room) in game_data.room_geometry.iter().enumerate() {
        let room_id = room.room_id;
        if is_room_eligible(game_data, room_id, room_idx, map) {
            eligible.push((room_id, room_idx));
        }
    }
    eligible
}

pub fn generate_water_assignments(
    map: &Map,
    game_data: &GameData,
    settings: &ExperimentalSettings,
    seed: usize,
) -> HashMap<RoomId, WaterAssignment> {
    if !settings.randomize_water_environments {
        return HashMap::new();
    }

    let eligible = if settings.water_flood_near_spawn {
        let count = settings.water_room_count as usize;
        get_eligible_rooms_near_spawn(game_data, map, count)
    } else {
        let mut eligible = get_eligible_water_rooms(game_data, map);
        if eligible.is_empty() {
            return HashMap::new();
        }

        let count = settings
            .water_room_count
            .min(eligible.len() as u32) as usize;
        if count == 0 {
            return HashMap::new();
        }

        let mut rng_seed = [0u8; 32];
        rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
        let mut rng = StdRng::from_seed(rng_seed);
        eligible.shuffle(&mut rng);
        eligible.truncate(count);
        eligible
    };

    if eligible.is_empty() {
        return HashMap::new();
    }

    if settings.water_flood_near_spawn {
        let names: Vec<_> = eligible
            .iter()
            .map(|(room_id, _)| room_name(game_data, *room_id))
            .collect();
        info!(
            "Flooding {} dry rooms near spawn for testing: {}",
            eligible.len(),
            names.join(", ")
        );
    }

    let mut assignments = HashMap::new();
    for (room_id, _room_idx) in eligible {
        assignments.insert(
            room_id,
            WaterAssignment {
                liquid_level: FULL_FLOOD_MAP_LIQUID_LEVEL,
                donor_room_ptr: DEFAULT_WATER_DONOR_ROOM_PTR,
            },
        );
    }
    assignments
}

/// Pause-map liquid surface Y in room screen-row coordinates (`0.0` = fully flooded).
pub const FULL_FLOOD_MAP_LIQUID_LEVEL: f32 = 0.0;

pub fn tile_liquid_level_for_room(room_liquid_level: f32, tile_y: f32) -> Option<f32> {
    if tile_y <= room_liquid_level - 1.0 {
        None
    } else if tile_y >= room_liquid_level {
        Some(0.0)
    } else {
        Some(room_liquid_level.fract())
    }
}

fn set_door_physics_water(node_json: &mut json::JsonValue) {
    if node_json["nodeType"].as_str() != Some("door") {
        return;
    }
    if !node_json.has_key("doorEnvironments") {
        node_json["doorEnvironments"] = json::array![];
    }
    if node_json["doorEnvironments"].is_empty() {
        node_json["doorEnvironments"]
            .push(json::object! { "physics" => "water" })
            .unwrap();
    } else {
        node_json["doorEnvironments"][0]["physics"] = "water".into();
    }
}

pub fn prepare_game_data_with_water(
    base: &GameData,
    assignments: &HashMap<RoomId, WaterAssignment>,
) -> Result<GameData> {
    if assignments.is_empty() {
        return Ok(base.clone());
    }

    let mut game_data = base.clone();
    let flooded: HashSet<RoomId> = assignments.keys().copied().collect();

    for &room_id in &flooded {
        ensure!(
            game_data.room_json_map.contains_key(&room_id),
            "Unknown room id {room_id} in water assignment"
        );

        let room_json = game_data.room_json_map.get_mut(&room_id).unwrap();
        for node_json in room_json["nodes"].members_mut() {
            set_door_physics_water(node_json);
            let node_id = node_json["id"].as_usize().unwrap();
            if let Some(stored) = game_data.node_json_map.get_mut(&(room_id, node_id)) {
                set_door_physics_water(stored);
            }
        }
    }

    patch_links_for_water(&mut game_data, &flooded);
    Ok(game_data)
}

fn patch_links_for_water(game_data: &mut GameData, flooded: &HashSet<RoomId>) {
    let gravity_req = Requirement::Item(Item::Gravity as ItemId);
    for link in &mut game_data.links {
        let from_key = &game_data.vertex_isv.keys[link.from_vertex_id];
        let to_key = &game_data.vertex_isv.keys[link.to_vertex_id];
        if flooded.contains(&from_key.room_id) || flooded.contains(&to_key.room_id) {
            if !requirement_implies_gravity(&link.requirement) {
                link.requirement = Requirement::make_and(vec![
                    link.requirement.clone(),
                    gravity_req.clone(),
                ]);
            }
        }
    }
}

fn requirement_implies_gravity(req: &Requirement) -> bool {
    match req {
        Requirement::Item(item) => *item == Item::Gravity as ItemId,
        Requirement::And(reqs) => reqs.iter().any(requirement_implies_gravity),
        Requirement::Or(reqs) => reqs.iter().all(requirement_implies_gravity),
        _ => false,
    }
}

pub fn water_enabled(settings: &RandomizerSettings) -> bool {
    settings.experimental_settings.randomize_water_environments
}
