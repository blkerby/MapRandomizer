use anyhow::Result;
// use hashbrown::HashSet;
use maprando::{
    game_data::{GameData, Item, Requirement},
    randomize::{
        AreaAssignment, DifficultyConfig, DoorsMode, ItemDotChange, ItemMarkers,
        ItemPlacementStyle, ItemPriorityGroup, MotherBrainFight, Objectives, ProgressionRate,
        SaveAnimals, WallJump, MapsRevealed,
    },
    traverse::{apply_requirement, GlobalState, LocalState},
};
use std::path::Path;

// fn strip_cross_room_reqs(req: &Requirement) -> Requirement {
//     match req {
//         Requirement::AdjacentRunway { .. } => Requirement::Never,
//         Requirement::CanComeInCharged { .. } => Requirement::Never,
//         Requirement::And(sub_reqs) => {
//             Requirement::make_and(sub_reqs.iter().map(strip_cross_room_reqs).collect())
//         }
//         Requirement::Or(sub_reqs) => {
//             Requirement::make_or(sub_reqs.iter().map(strip_cross_room_reqs).collect())
//         }
//         _ => req.clone(),
//     }
// }

fn run_scenario(
    proficiency: f32,
    missile_cnt: i32,
    super_cnt: i32,
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
        notable_strats: vec![true; game_data.notable_strat_isv.keys.len()],
        flags: vec![false; game_data.flag_isv.keys.len()],
        items: items,
        max_energy: 1899,
        max_missiles: missile_cnt,
        max_reserves: 0,
        max_supers: super_cnt,
        max_power_bombs: 0,
        shine_charge_tiles: 16.0,
        heated_shine_charge_tiles: 16.0,
        weapon_mask,
    };
    let local_state = LocalState {
        energy_used: 0,
        reserves_used: 0,
        missiles_used: 0,
        supers_used: 0,
        power_bombs_used: 0,
    };
    let difficulty = DifficultyConfig {
        name: None,
        tech: vec![],
        notable_strats: vec![],
        shine_charge_tiles: 16.0,
        heated_shine_charge_tiles: 16.0,
        shinecharge_leniency_frames: 15,
        progression_rate: ProgressionRate::Uniform,
        random_tank: true,
        spazer_before_plasma: true,
        starting_items: vec![],
        semi_filler_items: vec![],
        filler_items: vec![Item::Missile],
        early_filler_items: vec![],
        item_placement_style: ItemPlacementStyle::Neutral,
        item_priorities: vec![ItemPriorityGroup {
            name: "Default".to_string(),
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
        transition_letters: false,
        item_markers: ItemMarkers::ThreeTiered,
        item_dot_change: ItemDotChange::Fade,
        all_items_spawn: true,
        acid_chozo: true,
        buffed_drops: true,
        fast_elevators: true,
        fast_doors: true,
        fast_pause_menu: true,
        respin: false,
        infinite_space_jump: false,
        momentum_conservation: false,
        objectives: Objectives::Bosses,
        doors_mode: DoorsMode::Ammo,
        start_location_mode: maprando::randomize::StartLocationMode::Ship,
        save_animals: SaveAnimals::No,
        early_save: false,
        area_assignment: AreaAssignment::Standard,
        wall_jump: WallJump::Vanilla,
        etank_refill: maprando::randomize::EtankRefill::Vanilla,
        maps_revealed: MapsRevealed::Yes,
        energy_free_shinesparks: false,
        vanilla_map: false,
        ultra_low_qol: false,
        skill_assumptions_preset: None,
        item_progression_preset: None,
        quality_of_life_preset: None,
        debug_options: None,
    };

    // println!(
    //     "{:?}",
    //     apply_requirement(&Requirement::PhantoonFight {  }, &global_state, local_state, false, &difficulty, &game_data)
    // );

    // let new_local_state_opt = apply_requirement(
    //     &Requirement::DraygonFight {
    //         can_be_very_patient_tech_id: game_data.tech_isv.index_by_key["canBeVeryPatient"],
    //     },
    //     &global_state,
    //     local_state,
    //     false,
    //     &difficulty,
    //     game_data,
    // );

    let new_local_state_opt = apply_requirement(
            &Requirement::RidleyFight {
                can_be_very_patient_tech_id: game_data.tech_isv.index_by_key["canBeVeryPatient"]
            },
            &global_state,
            local_state,
            false,
            &difficulty,
            game_data
    );

    let outcome = new_local_state_opt.map(|x| format!("{}", x.energy_used)).unwrap_or("n/a".to_string());
    println!(
        "proficiency={}, items={:?}, missiles={}, patience={}: {}",
        proficiency, item_loadout, missile_cnt, patience, outcome
    );

}

fn main() -> Result<()> {
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let palette_theme_path = Path::new("../palette_smart_exports");
    let escape_timings_path = Path::new("data/escape_timings.json");
    let start_locations_path = Path::new("data/start_locations.json");
    let hub_locations_path = Path::new("data/hub_locations.json");
    let mosaic_path = Path::new("../Mosaic");
    let title_screen_path = Path::new("../TitleScreen/Images");

    let game_data = GameData::load(
        sm_json_data_path,
        room_geometry_path,
        palette_theme_path,
        escape_timings_path,
        start_locations_path,
        hub_locations_path,
        mosaic_path,
        title_screen_path,
    )?;

    // let proficiencies = vec![0.0, 0.3, 0.5, 0.7, 1.0];
    // let missile_counts = vec![0, 5, 30, 60, 120];
    // let super_counts = vec![0, 5, 30];
    // let item_loadouts = vec![
    //     vec![],
    //     vec!["C"],
    //     vec!["C", "I"],
    //     vec!["C", "I", "W", "S"],
    //     vec!["C", "P"],
    //     vec!["C", "I", "W", "P"],
    //     vec!["G"],
    //     vec!["G", "C"], 
    //     vec!["G", "C", "I", "W", "P"],
    //     vec!["V", "G"],
    //     vec!["V", "G", "C"],
    //     vec!["V", "G", "C", "I", "W", "P"],
    // ];

    let proficiencies = vec![0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0];
    let missile_counts = vec![0];
    let super_counts = vec![0];
    let item_loadouts = vec![
        vec!["M", "R", "C", "I", "W", "P"],
    ];


    for &proficiency in &proficiencies {
        for &missile_cnt in &missile_counts {
            for &super_cnt in &super_counts {
                for beam_loadout in &item_loadouts {
                    for patience in [true, false] {
                        run_scenario(proficiency, missile_cnt, super_cnt, beam_loadout, patience, &game_data);
                    }
                }
            }    
        }
    }

    Ok(())
}
