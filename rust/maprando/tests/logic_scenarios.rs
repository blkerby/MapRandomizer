use std::path::Path;

use anyhow::{Context, Result, bail};
use hashbrown::HashMap;
use maprando::{
    randomize::{DifficultyConfig, Preprocessor},
    settings::{
        InitialMapRevealSettings, ItemProgressionSettings, Objective, ObjectiveOption,
        ObjectiveSetting, ObjectiveSettings, OtherSettings, QualityOfLifeSettings,
        RandomizerSettings, SkillAssumptionSettings, StartLocationSettings,
    },
    traverse::{LockedDoorData, Traverser},
};
use maprando_game::{
    Capacity, GameData, LinksDataGroup, NodeId, ObstacleMask, RoomId, VertexId, VertexKey,
};
use maprando_logic::{GlobalState, Inventory, LocalState, ResourceLevel};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ConnectionsList {
    connections: Vec<Connection>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Connection {
    from_room_id: RoomId,
    from_node_id: NodeId,
    to_room_id: RoomId,
    to_node_id: NodeId,
    bidirectional: bool,
}

#[derive(Debug, Deserialize)]
struct ScenariosList {
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Scenario {
    #[serde(rename = "name")]
    _name: Option<String>,
    #[serde(default)]
    settings: ScenarioSettings,
    global_state: Option<ScenarioGlobalState>,
    start_room_id: usize,
    start_node_id: usize,
    #[serde(default)]
    start_obstacles_cleared: Vec<String>,
    start_state: Option<ScenarioState>,
    end_room_id: usize,
    end_node_id: usize,
    #[serde(default)]
    end_obstacles_cleared: Vec<String>,
    end_state: Option<ScenarioState>,
    #[serde(default)]
    fail: bool,
}

#[derive(Default, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScenarioSettings {
    disableable_etanks: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScenarioGlobalState {
    #[serde(default)]
    items: Vec<String>,
    #[serde(default)]
    flags: Vec<String>,
    #[serde(default)]
    disabled_tech: Vec<String>,
    max_energy: Option<Capacity>,
    max_reserves: Option<Capacity>,
    max_missiles: Option<Capacity>,
    max_supers: Option<Capacity>,
    max_power_bombs: Option<Capacity>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScenarioState {
    energy: Option<ResourceLevel>,
    reserves: Option<ResourceLevel>,
    missiles: Option<ResourceLevel>,
    supers: Option<ResourceLevel>,
    power_bombs: Option<ResourceLevel>,
    shinecharge_frames_remaining: Option<Capacity>,
    flash_suit: Option<bool>,
}

fn get_settings(scenario: &Scenario) -> Result<RandomizerSettings> {
    let settings = &scenario.settings;
    // Many settings are irrelevant to these tests, e.g. item progression settings.
    // We generally use settings as close to vanilla as possible (e.g. QoL off),
    // to avoid impact to the tests as QoL evolves.
    Ok(RandomizerSettings {
        version: 0,
        name: None,
        skill_assumption_settings: SkillAssumptionSettings {
            preset: None,
            shinespark_tiles: 12.0,
            heated_shinespark_tiles: 13.0,
            speed_ball_tiles: 14.0,
            shinecharge_leniency_frames: 0,
            resource_multiplier: 1.0,
            farm_time_limit: 0.0,
            gate_glitch_leniency: 0,
            door_stuck_leniency: 0,
            bomb_into_cf_leniency: 0,
            jump_into_cf_leniency: 0,
            spike_xmode_leniency: 0,
            phantoon_proficiency: 1.0,
            draygon_proficiency: 1.0,
            ridley_proficiency: 1.0,
            botwoon_proficiency: 1.0,
            mother_brain_proficiency: 1.0,
            escape_timer_multiplier: 1.0,
            // Omitted as we fill the tech in DifficultyConfig directly:
            tech_settings: vec![],
            notable_settings: vec![],
        },
        item_progression_settings: ItemProgressionSettings {
            preset: None,
            progression_rate: maprando::settings::ProgressionRate::Fast,
            item_placement_style: maprando::settings::ItemPlacementStyle::Neutral,
            item_priority_strength: maprando::settings::ItemPriorityStrength::Moderate,
            random_tank: true,
            spazer_before_plasma: true,
            item_pool_preset: None,
            stop_item_placement_early: false,
            ammo_collect_fraction: 1.0,
            item_pool: vec![],
            starting_items_preset: None,
            starting_items: vec![],
            key_item_priority: vec![],
            filler_items: vec![],
        },
        quality_of_life_settings: QualityOfLifeSettings {
            preset: None,
            initial_map_reveal_settings: InitialMapRevealSettings {
                preset: None,
                map_stations: maprando::settings::MapRevealLevel::No,
                save_stations: maprando::settings::MapRevealLevel::No,
                refill_stations: maprando::settings::MapRevealLevel::No,
                ship: maprando::settings::MapRevealLevel::No,
                objectives: maprando::settings::MapRevealLevel::No,
                area_transitions: maprando::settings::MapRevealLevel::No,
                items1: maprando::settings::MapRevealLevel::No,
                items2: maprando::settings::MapRevealLevel::No,
                items3: maprando::settings::MapRevealLevel::No,
                items4: maprando::settings::MapRevealLevel::No,
                other: maprando::settings::MapRevealLevel::No,
                all_areas: false,
            },
            item_markers: maprando::settings::ItemMarkers::Simple,
            room_outline_revealed: false,
            opposite_area_revealed: false,
            mother_brain_fight: maprando::settings::MotherBrainFight::Vanilla,
            supers_double: false,
            escape_movement_items: false,
            escape_refill: false,
            escape_enemies_cleared: false,
            fast_elevators: false,
            fast_doors: false,
            fast_pause_menu: false,
            fanfares: maprando::settings::Fanfares::Vanilla,
            respin: false,
            infinite_space_jump: false,
            momentum_conservation: false,
            all_items_spawn: false,
            acid_chozo: false,
            remove_climb_lava: false,
            etank_refill: maprando::settings::ETankRefill::Vanilla,
            energy_station_reserves: false,
            disableable_etanks: settings.disableable_etanks.unwrap_or(false),
            reserve_backward_transfer: false,
            buffed_drops: false,
            early_save: false,
            persist_flash_suit: false,
            persist_blue_suit: false,
        },
        objective_settings: ObjectiveSettings {
            // These settings are unused. Bosses are used in `test_scenario``.
            preset: None,
            objective_options: vec![],
            min_objectives: 0,
            max_objectives: 0,
            objective_screen: maprando::settings::ObjectiveScreen::Disabled,
        },
        map_layout: String::new(),
        doors_mode: maprando::settings::DoorsMode::Blue,
        start_location_settings: StartLocationSettings {
            mode: maprando::settings::StartLocationMode::Ship,
            room_id: None,
            node_id: None,
        },
        save_animals: maprando::settings::SaveAnimals::No,
        other_settings: OtherSettings {
            wall_jump: maprando::settings::WallJump::Vanilla,
            area_assignment: maprando::settings::AreaAssignment::Standard,
            door_locks_size: maprando::settings::DoorLocksSize::Large,
            map_station_reveal: maprando::settings::MapStationReveal::Full,
            energy_free_shinesparks: false,
            ultra_low_qol: false,
            race_mode: false,
            random_seed: None,
        },
        debug: false,
    })
}

fn get_difficulty(
    game_data: &GameData,
    settings: &RandomizerSettings,
    scenario: &Scenario,
) -> Result<DifficultyConfig> {
    let mut difficulty =
        DifficultyConfig::new(&settings.skill_assumption_settings, game_data, &[], &[]);
    difficulty.tech.fill(true);
    if let Some(ref scenario_global) = scenario.global_state {
        for tech_str in &scenario_global.disabled_tech {
            let tech_id = *game_data
                .tech_id_by_name
                .get(tech_str)
                .context(format!("Unknown tech '{}'", tech_str))?;
            let tech_idx = game_data.tech_isv.index_by_key[&tech_id];
            difficulty.tech[tech_idx] = false;
        }
    }
    Ok(difficulty)
}

fn get_preprocessor<'a>(
    game_data: &'a GameData,
    connections: &[Connection],
    difficulty: &'a DifficultyConfig,
) -> Result<Preprocessor<'a>> {
    let mut door_map = HashMap::new();
    for conn in connections {
        match door_map.entry((conn.from_room_id, conn.from_node_id)) {
            hashbrown::hash_map::Entry::Occupied(occupied_entry) => {
                bail!("Conflicting connection: {:?}", conn);
            }
            hashbrown::hash_map::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert((conn.to_room_id, conn.to_node_id))
            }
        };
        if conn.bidirectional {
            match door_map.entry((conn.to_room_id, conn.to_node_id)) {
                hashbrown::hash_map::Entry::Occupied(occupied_entry) => {
                    bail!("Conflicting backward connection: {:?}", conn);
                }
                hashbrown::hash_map::Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert((conn.from_room_id, conn.from_node_id))
                }
            };
        }
    }
    Ok(Preprocessor {
        game_data,
        door_map,
        difficulty,
    })
}

fn get_global_state(
    game_data: &GameData,
    difficulty: &DifficultyConfig,
    scenario: &Scenario,
) -> Result<GlobalState> {
    let mut flags = vec![false; game_data.flag_isv.keys.len()];

    let mut inventory = Inventory {
        items: vec![false; game_data.item_isv.keys.len()],
        max_energy: 99,
        max_reserves: 0,
        max_missiles: 0,
        max_supers: 0,
        max_power_bombs: 0,
        collectible_missile_packs: 0,
        collectible_super_packs: 0,
        collectible_power_bomb_packs: 0,
        collectible_reserve_tanks: 0,
    };

    if let Some(ref scenario_global) = scenario.global_state {
        for item_str in &scenario_global.items {
            let item_idx = *game_data
                .item_isv
                .index_by_key
                .get(item_str)
                .context(format!("Unknown item '{}'", item_str))?;
            inventory.items[item_idx] = true;
        }

        for flag_str in &scenario_global.flags {
            let flag_idx = *game_data
                .flag_isv
                .index_by_key
                .get(flag_str)
                .context(format!("Unknown flag '{}'", flag_str))?;
            flags[flag_idx] = true;
        }

        if let Some(max_energy) = scenario_global.max_energy {
            inventory.max_energy = max_energy;
        }

        if let Some(max_reserves) = scenario_global.max_reserves {
            inventory.max_reserves = max_reserves;
        }

        if let Some(max_missiles) = scenario_global.max_missiles {
            inventory.max_missiles = max_missiles;
        }

        if let Some(max_supers) = scenario_global.max_supers {
            inventory.max_supers = max_supers;
        }

        if let Some(max_power_bombs) = scenario_global.max_power_bombs {
            inventory.max_power_bombs = max_power_bombs;
        }
    }

    let weapon_mask = game_data.get_weapon_mask(&inventory.items, &difficulty.tech);
    Ok(GlobalState {
        inventory: inventory.clone(),
        pool_inventory: inventory,
        flags,
        doors_unlocked: vec![],
        weapon_mask,
    })
}

fn get_local_state(state_opt: &Option<ScenarioState>) -> LocalState {
    let mut local = LocalState::empty();
    let Some(state) = state_opt else {
        return local;
    };
    if let Some(level) = state.energy {
        local.energy = level.into();
    }
    if let Some(level) = state.reserves {
        local.reserves = level.into();
    }
    if let Some(level) = state.missiles {
        local.missiles = level.into();
    }
    if let Some(level) = state.supers {
        local.supers = level.into();
    }
    if let Some(level) = state.power_bombs {
        local.power_bombs = level.into();
    }
    if let Some(frames) = state.shinecharge_frames_remaining {
        local.shinecharge_frames_remaining = frames;
    }
    if let Some(flash_suit) = state.flash_suit {
        local.flash_suit = flash_suit;
    }
    local
}

fn get_obstacle_mask(
    game_data: &GameData,
    room_id: RoomId,
    obstacles: &[String],
) -> Result<ObstacleMask> {
    let obstacle_idx_map = &game_data.room_obstacle_idx_map[&room_id];
    let mut obstacle_mask = 0;
    for obs in obstacles {
        let idx = obstacle_idx_map
            .get(obs)
            .with_context(|| format!("Obstacle not found: '{}'", obs))?;
        obstacle_mask |= 1 << idx;
    }
    Ok(obstacle_mask)
}

fn test_scenario(
    game_data: &GameData,
    connections: &[Connection],
    scenario: &Scenario,
) -> Result<()> {
    let settings = get_settings(scenario)?;
    let difficulty = get_difficulty(game_data, &settings, scenario)?;
    let preprocessor = get_preprocessor(game_data, connections, &difficulty)?;
    let cross_links = preprocessor.get_all_door_links();
    let cross_links_data = LinksDataGroup::new(
        cross_links,
        game_data.vertex_isv.keys.len(),
        game_data.base_links_data.links.len(),
    );
    let global_state = get_global_state(game_data, &difficulty, scenario)?;
    let start_local_state = get_local_state(&scenario.start_state);
    let end_local_state = get_local_state(&scenario.end_state);
    let objectives = vec![
        Objective::Kraid,
        Objective::Phantoon,
        Objective::Draygon,
        Objective::Ridley,
    ];

    let start_vertex_key = VertexKey {
        room_id: scenario.start_room_id,
        node_id: scenario.start_node_id,
        obstacle_mask: get_obstacle_mask(
            game_data,
            scenario.start_room_id,
            &scenario.start_obstacles_cleared,
        )?,
        actions: vec![],
    };
    let start_vertex_id = *game_data
        .vertex_isv
        .index_by_key
        .get(&start_vertex_key)
        .context("Start vertex not found")?;

    let end_vertex_key = VertexKey {
        room_id: scenario.end_room_id,
        node_id: scenario.end_node_id,
        obstacle_mask: get_obstacle_mask(
            game_data,
            scenario.end_room_id,
            &scenario.end_obstacles_cleared,
        )?,
        actions: vec![],
    };
    let end_vertex_id = *game_data
        .vertex_isv
        .index_by_key
        .get(&end_vertex_key)
        .context("End vertex not found")?;

    let num_vertices = game_data.vertex_isv.keys.len();
    let locked_door_data = LockedDoorData {
        locked_doors: vec![],
        locked_door_node_map: HashMap::new(),
        locked_door_vertex_ids: vec![],
    };
    let inventory = &global_state.inventory;

    for reverse in [false, true] {
        println!("reverse: {}", reverse);
        let initial_vertex_id: VertexId;
        let initial_local_state: LocalState;
        let final_vertex_id: VertexId;
        let final_local_state: LocalState;

        if reverse {
            initial_vertex_id = end_vertex_id;
            initial_local_state = end_local_state;
            final_vertex_id = start_vertex_id;
            final_local_state = start_local_state;
        } else {
            initial_vertex_id = start_vertex_id;
            initial_local_state = start_local_state;
            final_vertex_id = end_vertex_id;
            final_local_state = end_local_state;
        };

        let mut traverser =
            Traverser::new(num_vertices, reverse, initial_local_state, &global_state);

        traverser.add_origin(
            initial_local_state,
            &global_state.inventory,
            initial_vertex_id,
        );
        traverser.traverse(
            &game_data.base_links_data,
            &cross_links_data,
            &global_state,
            &settings,
            &difficulty,
            game_data,
            &preprocessor.door_map,
            &locked_door_data,
            &objectives,
            0,
        );

        let mut exact_success: bool = false;
        let mut success: bool = false;
        for &local in &traverser.lsr[final_vertex_id].local {
            let energy_pass = local.energy_available(inventory, true, reverse)
                >= final_local_state.energy_available(inventory, true, reverse);
            let energy_exact = local.energy_available(inventory, true, reverse)
                == final_local_state.energy_available(inventory, true, reverse);
            let reserves_pass = local.reserves_available(inventory, reverse)
                >= final_local_state.reserves_available(inventory, reverse);
            let reserves_exact = local.reserves_available(inventory, reverse)
                == final_local_state.reserves_available(inventory, reverse);
            let missiles_pass = local.missiles_available(inventory, reverse)
                >= final_local_state.missiles_available(inventory, reverse);
            let missiles_exact = local.missiles_available(inventory, reverse)
                == final_local_state.missiles_available(inventory, reverse);
            let supers_pass = local.supers_available(inventory, reverse)
                >= final_local_state.supers_available(inventory, reverse);
            let supers_exact = local.supers_available(inventory, reverse)
                == final_local_state.supers_available(inventory, reverse);
            let power_bombs_pass = local.power_bombs_available(inventory, reverse)
                >= final_local_state.power_bombs_available(inventory, reverse);
            let power_bombs_exact = local.power_bombs_available(inventory, reverse)
                == final_local_state.power_bombs_available(inventory, reverse);
            let shinecharge_frames_pass = local.shinecharge_frames_available(reverse)
                >= final_local_state.shinecharge_frames_available(reverse);
            let shinecharge_frames_exact = local.shinecharge_frames_available(reverse)
                == final_local_state.shinecharge_frames_available(reverse);
            let flash_suit_pass = local.flash_suit_available(reverse)
                >= final_local_state.flash_suit_available(reverse);
            let flash_suit_exact = local.flash_suit_available(reverse)
                == final_local_state.flash_suit_available(reverse);
            if energy_pass
                && reserves_pass
                && missiles_pass
                && supers_pass
                && power_bombs_pass
                && shinecharge_frames_pass
                && flash_suit_pass
            {
                success = true;
            }
            if energy_exact
                && reserves_exact
                && missiles_exact
                && supers_exact
                && power_bombs_exact
                && shinecharge_frames_exact
                && flash_suit_exact
            {
                exact_success = true;
            }
        }
        if scenario.fail {
            if success {
                bail!(
                    "Failure expected, but traversal succeeds, with final local state(s): {:?}",
                    traverser.lsr[final_vertex_id].local
                );
            } else {
                continue;
            }
        } else if !success {
            bail!(
                "Traversal fails. Final local state(s): {:?}",
                traverser.lsr[final_vertex_id].local
            );
        } else if !exact_success {
            bail!(
                "Traversal result is not exact. Final local state(s): {:?}",
                traverser.lsr[final_vertex_id].local
            );
        }
    }
    Ok(())
}

#[test]
fn test_logic_scenarios() -> Result<()> {
    std::env::set_current_dir(Path::new(".."))?;
    let base_game_data = GameData::load_minimal()?;
    for entry in std::fs::read_dir("maprando/tests/scenarios")? {
        let entry = entry?;
        println!("{}", entry.file_name().display());

        let room_pattern = entry.path().to_str().unwrap().to_owned() + "/room*.json";
        let mut game_data = base_game_data.clone();
        let num_rooms = game_data.room_ptrs.len();
        game_data.load_rooms(&room_pattern)?;

        let connections_path = entry.path().join("connections.json");
        let connections_list: ConnectionsList = if connections_path.exists() || num_rooms > 1 {
            let connections_str = std::fs::read_to_string(connections_path.clone())
                .context(format!("loading {}", connections_path.display()))?;
            serde_json::from_str(&connections_str)
                .context(format!("parsing {}", connections_path.display()))?
        } else {
            ConnectionsList {
                connections: vec![],
            }
        };

        let scenarios_path = entry.path().join("scenarios.json");
        let scenarios_str = std::fs::read_to_string(scenarios_path.clone())
            .context(format!("loading {}", scenarios_path.display()))?;
        let scenarios_list: ScenariosList = serde_json::from_str(&scenarios_str)
            .context(format!("parsing {}", scenarios_path.display()))?;
        for scenario in &scenarios_list.scenarios {
            println!("Scenario: {:?}", scenario);
            test_scenario(&game_data, &connections_list.connections, scenario)?;
        }
    }
    Ok(())
}
