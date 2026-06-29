use anyhow::{Result, ensure};
use hashbrown::{HashMap, HashSet};
use json;
use log::info;
use maprando_game::{GameData, Map, RoomId};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::environment_logic::effective_environment_pair_count;
use crate::settings::{ExperimentalSettings, RandomizerSettings};
use crate::water_environment::{
    get_eligible_rooms_near_spawn, room_has_water, room_name,
};

/// Default donor room for heat FX: Volcano Room.
pub const DEFAULT_HEAT_DONOR_ROOM_PTR: usize = 0x7AE32;
pub const DEFAULT_HEAT_DONOR_ROOM_ID: RoomId = 116;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HeatAssignment {
    /// ROM address of the vanilla room whose heat FX template is copied.
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

fn is_room_eligible(
    game_data: &GameData,
    room_id: RoomId,
    room_idx: usize,
    map: &Map,
    exclude: &HashSet<RoomId>,
    blocked: &HashSet<RoomId>,
) -> bool {
    if exclude.contains(&room_id) || blocked.contains(&room_id) {
        return false;
    }
    if !map.room_mask[room_idx] {
        return false;
    }
    let name = room_name(game_data, room_id);
    if EXCLUDED_ROOM_NAMES.contains(&name.as_str()) {
        return false;
    }
    if game_data.room_geometry[room_idx].heated {
        return false;
    }
    if room_has_water(game_data, room_id) {
        return false;
    }
    let geometry = &game_data.room_geometry[room_idx];
    if geometry.map.len() == 1 && geometry.map[0].len() == 1 {
        return false;
    }
    true
}

pub fn get_eligible_heat_rooms(
    game_data: &GameData,
    map: &Map,
    exclude: &HashSet<RoomId>,
    blocked: &HashSet<RoomId>,
) -> Vec<(RoomId, usize)> {
    let mut eligible = vec![];
    for (room_idx, room) in game_data.room_geometry.iter().enumerate() {
        let room_id = room.room_id;
        if is_room_eligible(game_data, room_id, room_idx, map, exclude, blocked) {
            eligible.push((room_id, room_idx));
        }
    }
    eligible
}

pub fn generate_heat_assignments(
    map: &Map,
    game_data: &GameData,
    settings: &ExperimentalSettings,
    seed: usize,
    exclude: &HashSet<RoomId>,
    blocked: &HashSet<RoomId>,
) -> HashMap<RoomId, HeatAssignment> {
    if !settings.randomize_heat_environments {
        return HashMap::new();
    }

    let eligible = if settings.heat_flood_near_spawn {
        let count = settings.heat_room_count as usize;
        get_eligible_rooms_near_spawn(game_data, map, count)
            .into_iter()
            .filter(|(room_id, room_idx)| {
                is_room_eligible(game_data, *room_id, *room_idx, map, exclude, blocked)
            })
            .collect()
    } else {
        let mut eligible = get_eligible_heat_rooms(game_data, map, exclude, blocked);
        if eligible.is_empty() {
            return HashMap::new();
        }

        let count = settings
            .heat_room_count
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

    if settings.heat_flood_near_spawn {
        let names: Vec<_> = eligible
            .iter()
            .map(|(room_id, _)| room_name(game_data, *room_id))
            .collect();
        info!(
            "Heating {} dry rooms near spawn for testing: {}",
            eligible.len(),
            names.join(", ")
        );
    } else if !eligible.is_empty() && !settings.heat_flood_near_spawn {
        info!("Heating {} rooms across the map", eligible.len());
    }

    let mut assignments = HashMap::new();
    for (room_id, _room_idx) in eligible {
        assignments.insert(
            room_id,
            HeatAssignment {
                donor_room_ptr: DEFAULT_HEAT_DONOR_ROOM_PTR,
            },
        );
    }
    assignments
}

/// Heat random dry rooms and cool the same number of vanilla heated rooms (1:1 balance).
pub fn generate_balanced_heat_environment_assignments(
    map: &Map,
    game_data: &GameData,
    settings: &ExperimentalSettings,
    seed: usize,
    exclude: &HashSet<RoomId>,
    blocked: &HashSet<RoomId>,
) -> (
    HashMap<RoomId, HeatAssignment>,
    HashMap<RoomId, DryHeatAssignment>,
) {
    if !settings.randomize_heat_environments {
        return (HashMap::new(), HashMap::new());
    }
    if settings.heat_flood_near_spawn {
        return (
            generate_heat_assignments(map, game_data, settings, seed, exclude, blocked),
            HashMap::new(),
        );
    }

    let mut heat_eligible = get_eligible_heat_rooms(game_data, map, exclude, blocked);
    let mut cool_eligible = get_eligible_dry_heat_rooms(game_data, map, blocked);
    let pair_count = effective_environment_pair_count(settings, settings.heat_room_count)
        .min(heat_eligible.len() as u32)
        .min(cool_eligible.len() as u32) as usize;
    if pair_count == 0 {
        return (HashMap::new(), HashMap::new());
    }

    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
    let mut heat_rng = StdRng::from_seed(rng_seed);
    heat_eligible.shuffle(&mut heat_rng);
    heat_eligible.truncate(pair_count);

    rng_seed[..8].copy_from_slice(&seed.wrapping_add(7919).to_le_bytes());
    let mut cool_rng = StdRng::from_seed(rng_seed);
    cool_eligible.shuffle(&mut cool_rng);
    cool_eligible.truncate(pair_count);

    info!(
        "Balanced heat: heating {} dry rooms and cooling {} vanilla heated rooms",
        pair_count, pair_count
    );

    let heat_assignments = heat_eligible
        .into_iter()
        .map(|(room_id, _room_idx)| {
            (
                room_id,
                HeatAssignment {
                    donor_room_ptr: DEFAULT_HEAT_DONOR_ROOM_PTR,
                },
            )
        })
        .collect();

    let dry_heat_assignments = cool_eligible
        .into_iter()
        .map(|(room_id, _room_idx)| {
            (
                room_id,
                DryHeatAssignment {
                    donor_room_ptr: crate::water_environment::DEFAULT_DRY_DONOR_ROOM_PTR,
                },
            )
        })
        .collect();

    (heat_assignments, dry_heat_assignments)
}

fn set_room_heated(room_json: &mut json::JsonValue) {
    if !room_json.has_key("roomEnvironments") {
        room_json["roomEnvironments"] = json::array![];
    }
    if room_json["roomEnvironments"].is_empty() {
        room_json["roomEnvironments"]
            .push(json::object! { "heated" => true })
            .unwrap();
    } else {
        for env in room_json["roomEnvironments"].members_mut() {
            env["heated"] = true.into();
        }
    }
}

pub fn apply_heat_overlay(
    game_data: &mut GameData,
    assignments: &HashMap<RoomId, HeatAssignment>,
) -> Result<()> {
    if assignments.is_empty() {
        return Ok(());
    }

    for &room_id in assignments.keys() {
        ensure!(
            game_data.room_json_map.contains_key(&room_id),
            "Unknown room id {room_id} in heat assignment"
        );
        let room_idx = game_data.room_idx_by_id[&room_id];
        game_data.room_geometry[room_idx].heated = true;
        let room_json = game_data.room_json_map.get_mut(&room_id).unwrap();
        set_room_heated(room_json);
    }
    Ok(())
}

pub fn prepare_game_data_with_heat(
    base: &GameData,
    assignments: &HashMap<RoomId, HeatAssignment>,
) -> Result<GameData> {
    if assignments.is_empty() {
        return Ok(base.clone());
    }
    let mut game_data = base.clone();
    apply_heat_overlay(&mut game_data, assignments)?;
    Ok(game_data)
}

pub fn heat_enabled(settings: &RandomizerSettings) -> bool {
    settings.experimental_settings.randomize_heat_environments
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DryHeatAssignment {
    pub donor_room_ptr: usize,
}

fn is_dry_heat_eligible(
    game_data: &GameData,
    room_id: RoomId,
    room_idx: usize,
    map: &Map,
    blocked: &HashSet<RoomId>,
) -> bool {
    if blocked.contains(&room_id) {
        return false;
    }
    if !map.room_mask[room_idx] {
        return false;
    }
    let name = room_name(game_data, room_id);
    if EXCLUDED_ROOM_NAMES.contains(&name.as_str()) {
        return false;
    }
    if !game_data.room_geometry[room_idx].heated {
        return false;
    }
    if room_has_water(game_data, room_id) {
        return false;
    }
    let geometry = &game_data.room_geometry[room_idx];
    if geometry.map.len() == 1 && geometry.map[0].len() == 1 {
        return false;
    }
    true
}

pub fn get_eligible_dry_heat_rooms(
    game_data: &GameData,
    map: &Map,
    blocked: &HashSet<RoomId>,
) -> Vec<(RoomId, usize)> {
    let mut eligible = vec![];
    for (room_idx, room) in game_data.room_geometry.iter().enumerate() {
        let room_id = room.room_id;
        if is_dry_heat_eligible(game_data, room_id, room_idx, map, blocked) {
            eligible.push((room_id, room_idx));
        }
    }
    eligible
}

fn clear_room_heated(room_json: &mut json::JsonValue) {
    if !room_json.has_key("roomEnvironments") {
        return;
    }
    for env in room_json["roomEnvironments"].members_mut() {
        env["heated"] = false.into();
    }
}

pub fn apply_dry_heat_overlay(
    game_data: &mut GameData,
    assignments: &HashMap<RoomId, DryHeatAssignment>,
) -> Result<()> {
    if assignments.is_empty() {
        return Ok(());
    }

    for &room_id in assignments.keys() {
        ensure!(
            game_data.room_json_map.contains_key(&room_id),
            "Unknown room id {room_id} in dry heat assignment"
        );
        let room_idx = game_data.room_idx_by_id[&room_id];
        game_data.room_geometry[room_idx].heated = false;
        let room_json = game_data.room_json_map.get_mut(&room_id).unwrap();
        clear_room_heated(room_json);
    }
    Ok(())
}
