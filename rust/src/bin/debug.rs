use anyhow::Result;
// use hashbrown::HashSet;
use maprando::{
    game_data::{GameData, Item, Requirement},
    randomize::{
        DifficultyConfig, ItemMarkers, ItemPlacementStyle, ItemPriorityGroup, ProgressionRate, Objectives, MotherBrainFight,
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

fn main() -> Result<()> {
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let palette_path = Path::new("../palettes.json");
    let game_data = GameData::load(sm_json_data_path, room_geometry_path, palette_path)?;

    // let vertex_id_src = game_data.vertex_isv.index_by_key[&(77, 1, 0)];
    // let vertex_id_dst = game_data.vertex_isv.index_by_key[&(77, 3, 3)];
    // let mut links: Vec<Link> = Vec::new();
    // for link in &game_data.links {
    //     if link.from_vertex_id == vertex_id_src && link.to_vertex_id == vertex_id_dst {
    //         println!("{}: {:?}", link.strat_name, link.requirement);
    //         links.push(link.clone());
    //         break;
    //     }
    // }

    let mut items = vec![false; game_data.item_isv.keys.len()];
    // items[Item::Missile as usize] = true;
    // items[Item::SpaceJump as usize] = true;
    // items[Item::Super as usize] = true;
    // items[Item::Morph as usize] = true;
    items[Item::ScrewAttack as usize] = true;
    items[Item::Charge as usize] = true;
    // items[Item::Wave as usize] = true;
    // items[Item::Ice as usize] = true;
    // items[Item::Spazer as usize] = true;
    // items[Item::Plasma as usize] = true;
    items[Item::Varia as usize] = true;
    // items[Item::Gravity as usize] = true;

    let weapon_mask = game_data.get_weapon_mask(&items);
    let global_state = GlobalState {
        tech: vec![true; game_data.tech_isv.keys.len()],
        notable_strats: vec![true; game_data.notable_strat_isv.keys.len()],
        flags: vec![false; game_data.flag_isv.keys.len()],
        items: items,
        max_energy: 20000,
        max_missiles: 0,
        max_reserves: 0,
        max_supers: 0,
        max_power_bombs: 0,
        shine_charge_tiles: 16.0,
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
        tech: vec![],
        notable_strats: vec![],
        shine_charge_tiles: 16.0,
        progression_rate: ProgressionRate::Normal,
        filler_items: vec![Item::Missile],
        item_placement_style: ItemPlacementStyle::Neutral,
        item_priorities: vec![ItemPriorityGroup {
            name: "Default".to_string(),
            items: game_data.item_isv.keys.clone(),
        }],
        resource_multiplier: 1.0,
        escape_timer_multiplier: 1.0,
        phantoon_proficiency: 1.0,
        draygon_proficiency: 1.0,
        ridley_proficiency: 0.8,
        botwoon_proficiency: 1.0,
        save_animals: false,
        supers_double: true,
        mother_brain_fight: MotherBrainFight::Short,
        escape_enemies_cleared: true,
        escape_movement_items: true,
        mark_map_stations: true,
        item_markers: ItemMarkers::ThreeTiered,
        all_items_spawn: true,
        fast_elevators: true,
        fast_doors: true,
        objectives: Objectives::Bosses,
        debug_options: None,
    };

    // println!("{:?}", game_data.helpers["h_heatResistant"]);

    // println!(
    //     "{:?}",
    //     apply_requirement(&Requirement::PhantoonFight {  }, &global_state, local_state, false, &difficulty)
    // );
    // println!("{} links", game_data.links.len());

    // println!(
    //     "{:?}",
    //     apply_requirement(&Requirement::DraygonFight {
    //         can_be_patient_tech_id: game_data.tech_isv.index_by_key["canBePatient"]
    //     }, &global_state, local_state, false, &difficulty)
    // );

    println!(
        "{:?}",
        apply_requirement(&Requirement::RidleyFight {
            can_be_patient_tech_id: game_data.tech_isv.index_by_key["canBePatient"]
        }, &global_state, local_state, false, &difficulty)
    );

    // let mut name_set: HashSet<String> = HashSet::new();
    // for link in &game_data.links {
    //     if let Some(name) = link.notable_strat_name.clone() {
    //         if !name_set.contains(&name) {
    //             println!("{}", name);
    //             name_set.insert(name.clone());
    //         }
    //         // println!("{}", link.requirement);
    //     }
    // }

    // // let get_link_count = |global: &GlobalState| {
    // //     let mut cnt = 0;
    // //     for link in &game_data.links {
    // //         let req = strip_cross_room_reqs(&link.requirement);
    // //         if apply_requirement(&req, &global, local_state, false, &difficulty).is_some() {
    // //             cnt += 1;
    // //         }
    // //     }
    // //     cnt
    // // };
    // let get_link_set = |global: &GlobalState| {
    //     let mut out: HashSet<usize> = HashSet::new();
    //     for (i, link) in game_data.links.iter().enumerate() {
    //         let req = strip_cross_room_reqs(&link.requirement);
    //         if apply_requirement(&req, &global, local_state, false, &difficulty).is_some() {
    //             out.insert(i);
    //         }
    //     }
    //     out
    // };

    // let baseline_set = get_link_set(&global_state);
    // let mut global_state_item = global_state.clone();
    // let item = Item::Bombs;
    // global_state_item.collect(item, &game_data);
    // let bombs_set = get_link_set(&global_state_item);
    // println!("{:?}: {} {}", item, baseline_set.len(), bombs_set.len());

    // for i in bombs_set.difference(&baseline_set) {
    //     let link = &game_data.links[*i];
    //     let (from_room_id, _, _) = game_data.vertex_isv.keys[link.from_vertex_id];
    //     let from_room_name = &game_data.room_json_map[&from_room_id]["name"];
    //     println!("{}: {}", from_room_name, link.strat_name);
    // }

    // let baseline_cnt = get_link_count(&global_state);
    // println!("Total links: {}", game_data.links.len());
    // println!("Baseline: {}", baseline_cnt);
    // for item_idx in 0..global_state.items.len() {
    //     let mut global_state_item = global_state.clone();
    //     let item = Item::try_from(item_idx).unwrap();
    //     global_state_item.collect(item, &game_data);
    //     let cnt = get_link_count(&global_state_item);
    //     println!("{:?}: {}", item, cnt - baseline_cnt);
    // }
    Ok(())
}
