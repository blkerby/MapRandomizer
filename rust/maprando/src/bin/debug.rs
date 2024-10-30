use anyhow::Result;
use hashbrown::HashMap;
use maprando::{
    preset::PresetData,
    randomize::{get_objectives, DifficultyConfig},
    settings::RandomizerSettings,
    traverse::{apply_requirement, LockedDoorData},
};
use maprando_game::{Capacity, GameData, Item, Requirement, TECH_ID_CAN_BE_VERY_PATIENT};
use maprando_logic::{GlobalState, Inventory, LocalState};
use rand::SeedableRng;
use std::path::Path;

fn run_scenario(
    proficiency: f32,
    missile_cnt: Capacity,
    super_cnt: Capacity,
    item_loadout: &[&'static str],
    patience: bool,
    settings: &RandomizerSettings,
    mut difficulty: DifficultyConfig,
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
    let locked_door_data = LockedDoorData {
        locked_doors: vec![],
        locked_door_node_map: HashMap::new(),
        locked_door_vertex_ids: vec![],
    };

    let rng_seed = [0u8; 32];
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let objectives = get_objectives(&settings, &mut rng);
    difficulty.draygon_proficiency = proficiency;
    let new_local_state_opt = apply_requirement(
        &Requirement::DraygonFight {
            can_be_very_patient_tech_idx: game_data.tech_isv.index_by_key
                [&TECH_ID_CAN_BE_VERY_PATIENT],
        },
        &global_state,
        local_state,
        false,
        settings,
        &difficulty,
        game_data,
        &locked_door_data,
        &objectives,
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
    let tech_path = Path::new("data/tech_data.json");
    let notable_path = Path::new("data/notable_data.json");
    let presets_path = Path::new("data/presets");

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

    let preset_data = PresetData::load(tech_path, notable_path, presets_path, &game_data)?;
    let mut settings = preset_data.default_preset.clone();
    settings.skill_assumption_settings = preset_data.skill_presets.last().unwrap().clone();
    let difficulty = preset_data.difficulty_tiers.last().unwrap();

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
                            &settings,
                            difficulty.clone(),
                            &game_data,
                        );
                    }
                }
            }
        }
    }

    Ok(())
}
