// TODO: consider removing this later. It's not a bad lint but I don't want to deal with it now.
#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use hashbrown::HashMap;
use maprando::{
    preset::PresetData,
    randomize::{DifficultyConfig, get_objectives},
    settings::RandomizerSettings,
    traverse::{LockedDoorData, apply_requirement, simple_cost_config},
};
use maprando_game::{
    Capacity, GameData, Item, NodeId, Requirement, RidleyStuck, RoomId,
    TECH_ID_CAN_BE_EXTREMELY_PATIENT, TECH_ID_CAN_BE_PATIENT, TECH_ID_CAN_BE_VERY_PATIENT,
};
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
            _ => panic!("unrecognized beam {item}"),
        }
    }

    let weapon_mask = game_data.get_weapon_mask(&items, &difficulty.tech);
    let inventory = Inventory {
        items,
        max_energy: 1899,
        max_missiles: missile_cnt,
        max_reserves: 0,
        max_supers: super_cnt,
        max_power_bombs: 0,
        collectible_missile_packs: 0,
        collectible_super_packs: 0,
        collectible_power_bomb_packs: 0,
        collectible_reserve_tanks: 0,
    };
    let global_state = GlobalState {
        pool_inventory: inventory.clone(),
        inventory,
        flags: vec![false; game_data.flag_isv.keys.len()],
        doors_unlocked: vec![],
        weapon_mask,
    };
    let local_state = LocalState::full(false);
    let locked_door_data = LockedDoorData {
        locked_doors: vec![],
        locked_door_node_map: HashMap::new(),
        locked_door_vertex_ids: vec![],
    };
    let door_map: HashMap<(RoomId, NodeId), (RoomId, NodeId)> = HashMap::new();

    let rng_seed = [0u8; 32];
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let objectives = get_objectives(settings, None, game_data, &mut rng);
    difficulty.draygon_proficiency = proficiency;
    difficulty.ridley_proficiency = proficiency;
    difficulty.tech[game_data.tech_isv.index_by_key[&TECH_ID_CAN_BE_VERY_PATIENT]] = patience;
    difficulty.tech[game_data.tech_isv.index_by_key[&TECH_ID_CAN_BE_EXTREMELY_PATIENT]] = patience;
    // let new_local_state_opt = apply_requirement(
    //     &Requirement::DraygonFight {
    //         can_be_very_patient_tech_idx: game_data.tech_isv.index_by_key
    //             [&TECH_ID_CAN_BE_VERY_PATIENT],
    //     },
    //     &global_state,
    //     local_state,
    //     false,
    //     settings,
    //     &difficulty,
    //     game_data,
    //     &locked_door_data,
    //     &objectives,
    // );
    let cost_config = simple_cost_config();
    let new_local_state_opt = apply_requirement(
        &Requirement::RidleyFight {
            can_be_patient_tech_idx: game_data.tech_isv.index_by_key[&TECH_ID_CAN_BE_PATIENT],
            can_be_very_patient_tech_idx: game_data.tech_isv.index_by_key
                [&TECH_ID_CAN_BE_VERY_PATIENT],
            can_be_extremely_patient_tech_idx: game_data.tech_isv.index_by_key
                [&TECH_ID_CAN_BE_EXTREMELY_PATIENT],
            power_bombs: true,
            g_mode: false,
            stuck: RidleyStuck::None,
        },
        &global_state,
        local_state,
        false,
        settings,
        &difficulty,
        game_data,
        &door_map,
        &locked_door_data,
        &objectives,
        &cost_config,
    );

    let outcome = new_local_state_opt
        .map(|x| format!("{:?}", x.energy))
        .unwrap_or("n/a".to_string());
    println!(
        "proficiency={proficiency}, items={item_loadout:?}, missiles={missile_cnt}, patience={patience}: {outcome}"
    );
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let tech_path = Path::new("data/tech_data.json");
    let notable_path = Path::new("data/notable_data.json");
    let presets_path = Path::new("data/presets");

    let mut game_data = GameData::load(Path::new("."))?;
    game_data.make_links_data(&|_, _| (0, 1));

    let preset_data = PresetData::load(tech_path, notable_path, presets_path, &game_data)?;
    let mut settings = preset_data.default_preset.clone();
    settings.skill_assumption_settings = preset_data.skill_presets.last().unwrap().clone();
    let difficulty = preset_data.difficulty_tiers.last().unwrap();

    let proficiencies = vec![0.0, 0.3, 0.5, 0.7, 0.8, 0.825, 0.85, 0.9, 0.95, 1.0];
    let missile_counts = vec![60];
    let super_counts = vec![0];
    let item_loadouts = vec![vec!["M", "V", "C"]];

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
