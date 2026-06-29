use hashbrown::HashMap;
use maprando::settings::ExperimentalSettings;
use maprando::heat_environment::{
    HeatAssignment, generate_heat_assignments, prepare_game_data_with_heat,
};
use maprando::water_environment::room_has_water;
use maprando_game::{GameData, Map};

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
fn generate_heat_assignments_respects_eligibility() -> anyhow::Result<()> {
    let game_data = GameData::load(std::path::Path::new(".."))?;
    let map = minimal_map_with_room(&game_data, 7); // The Moat (dry crateria room)
    let settings = ExperimentalSettings {
        randomize_heat_environments: true,
        heat_room_count: 1,
        ..Default::default()
    };
    let assignments = generate_heat_assignments(&map, &game_data, &settings, 12345, &Default::default(), &hashbrown::HashSet::new());
    assert_eq!(assignments.len(), 1);
    assert!(assignments.contains_key(&7));
    Ok(())
}

#[test]
fn generate_heat_assignments_skips_existing_heated_rooms() -> anyhow::Result<()> {
    let game_data = GameData::load(std::path::Path::new(".."))?;
    let map = minimal_map_with_room(&game_data, 116); // Volcano Room
    let settings = ExperimentalSettings {
        randomize_heat_environments: true,
        heat_room_count: 1,
        ..Default::default()
    };
    let assignments = generate_heat_assignments(&map, &game_data, &settings, 999, &Default::default(), &hashbrown::HashSet::new());
    assert!(assignments.is_empty());
    Ok(())
}

#[test]
fn generate_heat_assignments_skips_water_rooms() -> anyhow::Result<()> {
    let game_data = GameData::load(std::path::Path::new(".."))?;
    let map = minimal_map_with_room(&game_data, 173); // Fish Tank
    let settings = ExperimentalSettings {
        randomize_heat_environments: true,
        heat_room_count: 1,
        ..Default::default()
    };
    let assignments = generate_heat_assignments(&map, &game_data, &settings, 999, &Default::default(), &hashbrown::HashSet::new());
    assert!(assignments.is_empty());
    assert!(room_has_water(&game_data, 173));
    Ok(())
}

#[test]
fn generate_heat_assignments_near_spawn() -> anyhow::Result<()> {
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
        randomize_heat_environments: true,
        heat_flood_near_spawn: true,
        heat_room_count: 5,
        ..Default::default()
    };
    let assignments = generate_heat_assignments(&map, &game_data, &settings, 0, &Default::default(), &hashbrown::HashSet::new());
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
fn prepare_game_data_with_heat_updates_room_environments() -> anyhow::Result<()> {
    let game_data = GameData::load(std::path::Path::new(".."))?;
    let room_id = 7;
    let mut assignments = HashMap::new();
    assignments.insert(
        room_id,
        HeatAssignment {
            donor_room_ptr: maprando::heat_environment::DEFAULT_HEAT_DONOR_ROOM_PTR,
        },
    );
    let prepared = prepare_game_data_with_heat(&game_data, &assignments)?;
    let room_json = &prepared.room_json_map[&room_id];
    assert!(room_json["roomEnvironments"]
        .members()
        .any(|env| env["heated"].as_bool() == Some(true)));
    Ok(())
}

#[test]
fn generate_balanced_heat_assignments_pairs_heat_and_cool() -> anyhow::Result<()> {
    use maprando::heat_environment::generate_balanced_heat_environment_assignments;

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
        randomize_heat_environments: true,
        ..Default::default()
    };
    let (heat, cool) = generate_balanced_heat_environment_assignments(
        &map,
        &game_data,
        &settings,
        4242,
        &hashbrown::HashSet::new(),
        &hashbrown::HashSet::new(),
    );
    assert!(!heat.is_empty());
    assert_eq!(heat.len(), cool.len());
    for room_id in heat.keys() {
        let room_idx = game_data.room_idx_by_id[room_id];
        assert!(!game_data.room_geometry[room_idx].heated);
    }
    for room_id in cool.keys() {
        let room_idx = game_data.room_idx_by_id[room_id];
        assert!(game_data.room_geometry[room_idx].heated);
    }
    Ok(())
}
