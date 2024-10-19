use anyhow::Result;
use hashbrown::HashMap;
use maprando::{
    randomize::{
        DifficultyConfig, ItemPriorityGroup
    }, settings::{AreaAssignment, DoorLocksSize, DoorsMode, ETankRefill, ItemDotChange, ItemMarkers, ItemPlacementStyle, ItemPriorityStrength, KeyItemPriority, MapStationReveal, MapsRevealed, MotherBrainFight, ProgressionRate, SaveAnimals, StartLocationMode, WallJump}, traverse::{apply_requirement, LockedDoorData}
};
use maprando_game::{Capacity, GameData, Item, Requirement, TECH_ID_CAN_BE_VERY_PATIENT};
use maprando_logic::{GlobalState, Inventory, LocalState};
use std::path::Path;

fn run_scenario(
    proficiency: f32,
    missile_cnt: Capacity,
    super_cnt: Capacity,
    item_loadout: &[&'static str],
    patience: bool,
    game_data: &GameData,
) {
    let mut items = vec![false; game_data.item_isv.keys.len()];
    for &item in item_loadout {
        match item {
            "V" => {
                items[Item::Varia as usize] = true;
            }
            "G" => {
                items[Item::Gravity as usize] = true;
            }
            "C" => {
                items[Item::Charge as usize] = true;
            }
            "I" => {
                items[Item::Ice as usize] = true;
            }
            "W" => {
                items[Item::Wave as usize] = true;
            }
            "S" => {
                items[Item::Spazer as usize] = true;
            }
            "P" => {
                items[Item::Plasma as usize] = true;
            }
            "M" => {
                items[Item::Morph as usize] = true;
            }
            "R" => {
                items[Item::ScrewAttack as usize] = true;
            }
            _ => panic!("unrecognized beam {}", item),
        }
    }

    let weapon_mask = game_data.get_weapon_mask(&items);
    let global_state = GlobalState {
        tech: vec![patience; game_data.tech_isv.keys.len()],
        notables: vec![true; game_data.notable_isv.keys.len()],
        inventory: Inventory {
            items: items,
            max_energy: 1899,
            max_missiles: missile_cnt,
            max_reserves: 0,
            max_supers: super_cnt,
            max_power_bombs: 0,
        },
        flags: vec![false; game_data.flag_isv.keys.len()],
        doors_unlocked: vec![],
        weapon_mask,
    };
    let local_state = LocalState {
        energy_used: 0,
        reserves_used: 0,
        missiles_used: 0,
        supers_used: 0,
        power_bombs_used: 0,
        shinecharge_frames_remaining: 0,
    };
    let difficulty = DifficultyConfig {
        name: None,
        tech: vec![],
        notables: vec![],
        shine_charge_tiles: 16.0,
        heated_shine_charge_tiles: 16.0,
        speed_ball_tiles: 24.0,
        shinecharge_leniency_frames: 15,
        progression_rate: ProgressionRate::Uniform,
        random_tank: true,
        spazer_before_plasma: true,
        stop_item_placement_early: false,
        item_pool: vec![],
        starting_items: vec![],
        semi_filler_items: vec![],
        filler_items: vec![Item::Missile],
        early_filler_items: vec![],
        item_placement_style: ItemPlacementStyle::Neutral,
        item_priority_strength: ItemPriorityStrength::Moderate,
        item_priorities: vec![ItemPriorityGroup {
            priority: KeyItemPriority::Default,
            items: game_data.item_isv.keys.clone(),
        }],
        resource_multiplier: 1.0,
        escape_timer_multiplier: 1.0,
        gate_glitch_leniency: 0,
        door_stuck_leniency: 0,
        phantoon_proficiency: proficiency,
        draygon_proficiency: proficiency,
        ridley_proficiency: proficiency,
        botwoon_proficiency: proficiency,
        mother_brain_proficiency: proficiency,
        supers_double: true,
        mother_brain_fight: MotherBrainFight::Short,
        escape_enemies_cleared: true,
        escape_refill: true,
        escape_movement_items: true,
        mark_map_stations: true,
        room_outline_revealed: true,
        opposite_area_revealed: true,
        transition_letters: false,
        door_locks_size: DoorLocksSize::Small,
        item_markers: ItemMarkers::ThreeTiered,
        item_dot_change: ItemDotChange::Fade,
        all_items_spawn: true,
        acid_chozo: true,
        remove_climb_lava: true,
        buffed_drops: true,
        fast_elevators: true,
        fast_doors: true,
        fast_pause_menu: true,
        respin: false,
        infinite_space_jump: false,
        momentum_conservation: false,
        objectives: vec![],
        doors_mode: DoorsMode::Ammo,
        start_location_mode: StartLocationMode::Ship,
        save_animals: SaveAnimals::No,
        early_save: false,
        area_assignment: AreaAssignment::Standard,
        wall_jump: WallJump::Vanilla,
        etank_refill: ETankRefill::Vanilla,
        maps_revealed: MapsRevealed::Full,
        map_station_reveal: MapStationReveal::Full,
        energy_free_shinesparks: false,
        vanilla_map: false,
        ultra_low_qol: false,
        skill_assumptions_preset: None,
        item_progression_preset: None,
        quality_of_life_preset: None,
        debug: false,
    };

    let locked_door_data = LockedDoorData {
        locked_doors: vec![],
        locked_door_node_map: HashMap::new(),
        locked_door_vertex_ids: vec![],
    };

    let new_local_state_opt = apply_requirement(
        &Requirement::DraygonFight {
            can_be_very_patient_tech_idx: game_data.tech_isv.index_by_key
                [&TECH_ID_CAN_BE_VERY_PATIENT],
        },
        &global_state,
        local_state,
        false,
        &difficulty,
        game_data,
        &locked_door_data,
    );

    let outcome = new_local_state_opt
        .map(|x| format!("{}", x.energy_used))
        .unwrap_or("n/a".to_string());
    println!(
        "proficiency={}, items={:?}, missiles={}, patience={}: {}",
        proficiency, item_loadout, missile_cnt, patience, outcome
    );
}

fn main() -> Result<()> {
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let escape_timings_path = Path::new("data/escape_timings.json");
    let start_locations_path = Path::new("data/start_locations.json");
    let hub_locations_path = Path::new("data/hub_locations.json");
    let reduced_flashing_path = Path::new("data/reduced_flashing.json");
    let strat_videos_path = Path::new("data/strat_videos.json");
    let title_screen_path = Path::new("../TitleScreen/Images");

    let game_data = GameData::load(
        sm_json_data_path,
        room_geometry_path,
        escape_timings_path,
        start_locations_path,
        hub_locations_path,
        title_screen_path,
        reduced_flashing_path,
        strat_videos_path,
    )?;

    let proficiencies = vec![0.0, 0.3, 0.5, 0.7, 0.8, 0.825, 0.85, 0.9, 0.95, 1.0];
    let missile_counts = vec![20];
    let super_counts = vec![0];
    let item_loadouts = vec![vec!["M"]];

    for &proficiency in &proficiencies {
        for &missile_cnt in &missile_counts {
            for &super_cnt in &super_counts {
                for beam_loadout in &item_loadouts {
                    for patience in [true, false] {
                        run_scenario(
                            proficiency,
                            missile_cnt,
                            super_cnt,
                            beam_loadout,
                            patience,
                            &game_data,
                        );
                    }
                }
            }
        }
    }

    Ok(())
}
