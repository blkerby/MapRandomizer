use hashbrown::HashMap;
use maprando::settings::ExperimentalSettings;
use maprando::water_environment::{
    WaterAssignment, FULL_FLOOD_MAP_LIQUID_LEVEL, generate_water_assignments,
    prepare_game_data_with_water, room_has_water, tile_liquid_level_for_room,
};
use maprando_game::{GameData, Item, ItemId, Map, Requirement};

fn minimal_map_with_room(game_data: &GameData, room_id: usize) -> Map {
    let room_idx = game_data.room_idx_by_id[&room_id];
    let mut room_mask = vec![false; game_data.room_geometry.len()];
    room_mask[room_idx] = true;
    Map {
        area: vec![0; game_data.room_geometry.len()],
        subarea: vec![0; game_data.room_geometry.len()],
        subsubarea: vec![0; game_data.room_geometry.len()],
        rooms: vec![(0, 0); game_data.room_geometry.len()],
        room_mask,
        doors: vec![],
    }
}

#[test]
fn generate_water_assignments_respects_eligibility() -> anyhow::Result<()> {
    let game_data = GameData::load(std::path::Path::new(".."))?;
    let map = minimal_map_with_room(&game_data, 7); // The Moat (dry crateria room)
    let settings = ExperimentalSettings {
        randomize_water_environments: true,
        water_room_count: 1,
        ..Default::default()
    };
    let assignments = generate_water_assignments(&map, &game_data, &settings, 12345, &hashbrown::HashSet::new());
    assert_eq!(assignments.len(), 1);
    assert!(assignments.contains_key(&7));
    assert!(!room_has_water(&game_data, 7));
    Ok(())
}

#[test]
fn generate_water_assignments_skips_existing_water_rooms() -> anyhow::Result<()> {
    let game_data = GameData::load(std::path::Path::new(".."))?;
    let map = minimal_map_with_room(&game_data, 173); // Fish Tank
    let settings = ExperimentalSettings {
        randomize_water_environments: true,
        water_room_count: 1,
        ..Default::default()
    };
    let assignments = generate_water_assignments(&map, &game_data, &settings, 999, &hashbrown::HashSet::new());
    assert!(assignments.is_empty());
    assert!(room_has_water(&game_data, 173));
    Ok(())
}

#[test]
fn generate_water_assignments_near_spawn() -> anyhow::Result<()> {
    let game_data = GameData::load(std::path::Path::new(".."))?;
    let map = Map {
        area: vec![0; game_data.room_geometry.len()],
        subarea: vec![0; game_data.room_geometry.len()],
        subsubarea: vec![0; game_data.room_geometry.len()],
        rooms: vec![(0, 0); game_data.room_geometry.len()],
        room_mask: vec![true; game_data.room_geometry.len()],
        doors: vec![],
    };
    let settings = ExperimentalSettings {
        randomize_water_environments: true,
        water_flood_near_spawn: true,
        water_room_count: 5,
        ..Default::default()
    };
    let assignments = generate_water_assignments(&map, &game_data, &settings, 0, &hashbrown::HashSet::new());
    assert!(!assignments.is_empty());
    let names: Vec<_> = assignments
        .keys()
        .map(|room_id| game_data.room_json_map[room_id]["name"].as_str().unwrap())
        .collect();
    assert!(
        names.iter().any(|n| {
            *n == "Pre-Map Flyway"
                || *n == "Gauntlet Entrance"
                || *n == "Terminator Room"
        }),
        "expected early Crateria rooms near spawn, got {names:?}"
    );
    Ok(())
}

#[test]
fn tile_liquid_level_for_room_full_flood() {
    assert_eq!(
        tile_liquid_level_for_room(FULL_FLOOD_MAP_LIQUID_LEVEL, 0.0),
        Some(0.0)
    );
    assert_eq!(
        tile_liquid_level_for_room(FULL_FLOOD_MAP_LIQUID_LEVEL, 4.0),
        Some(0.0)
    );
}

#[test]
fn prepare_game_data_with_water_updates_door_physics_and_links() -> anyhow::Result<()> {
    let game_data = GameData::load(std::path::Path::new(".."))?;
    let room_id = 7;
    let mut assignments = HashMap::new();
    assignments.insert(
        room_id,
        WaterAssignment {
            liquid_level: 0.0,
            donor_room_ptr: maprando::water_environment::DEFAULT_WATER_DONOR_ROOM_PTR,
        },
    );
    let prepared = prepare_game_data_with_water(&game_data, &assignments)?;
    for ((rid, _), node) in &prepared.node_json_map {
        if *rid != room_id || node["nodeType"].as_str() != Some("door") {
            continue;
        }
        assert_eq!(
            node["doorEnvironments"][0]["physics"].as_str(),
            Some("water")
        );
    }

    let base_links_touching_room: usize = game_data
        .links
        .iter()
        .filter(|link| {
            let from = &game_data.vertex_isv.keys[link.from_vertex_id];
            let to = &game_data.vertex_isv.keys[link.to_vertex_id];
            from.room_id == room_id || to.room_id == room_id
        })
        .count();
    assert!(base_links_touching_room > 0);

    let gravity_req = Requirement::Item(Item::Gravity as ItemId);
    for link in &prepared.links {
        let from = &prepared.vertex_isv.keys[link.from_vertex_id];
        let to = &prepared.vertex_isv.keys[link.to_vertex_id];
        if from.room_id == room_id || to.room_id == room_id {
            if let Requirement::And(reqs) = &link.requirement {
                assert!(reqs.iter().any(|r| r == &gravity_req));
            }
        }
    }
    Ok(())
}

#[test]
fn generate_balanced_water_assignments_pairs_flood_and_dry() -> anyhow::Result<()> {
    use maprando::water_environment::generate_balanced_water_environment_assignments;

    let game_data = GameData::load(std::path::Path::new(".."))?;
    let map = Map {
        area: vec![0; game_data.room_geometry.len()],
        subarea: vec![0; game_data.room_geometry.len()],
        subsubarea: vec![0; game_data.room_geometry.len()],
        rooms: vec![(0, 0); game_data.room_geometry.len()],
        room_mask: vec![true; game_data.room_geometry.len()],
        doors: vec![],
    };
    let settings = ExperimentalSettings {
        randomize_water_environments: true,
        ..Default::default()
    };
    let (flood, dry) = generate_balanced_water_environment_assignments(
        &map,
        &game_data,
        &settings,
        4242,
        &hashbrown::HashSet::new(),
    );
    assert!(!flood.is_empty());
    assert_eq!(flood.len(), dry.len());
    for room_id in flood.keys() {
        assert!(!room_has_water(&game_data, *room_id));
    }
    for room_id in dry.keys() {
        assert!(room_has_water(&game_data, *room_id));
    }
    Ok(())
}
