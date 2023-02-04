use anyhow::Result;
use maprando::{
    game_data::{GameData, Item, Requirement},
    randomize::{DifficultyConfig, ItemPlacementStrategy},
    traverse::{apply_requirement, GlobalState, LocalState},
};
use std::path::Path;

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
    items[Item::Missile as usize] = true;
    items[Item::SpaceJump as usize] = true;
    items[Item::Super as usize] = true;
    items[Item::Morph as usize] = true;
    items[Item::ScrewAttack as usize] = true;
    items[Item::Charge as usize] = true;
    items[Item::Wave as usize] = true;
    items[Item::Ice as usize] = true;
    items[Item::Spazer as usize] = true;
    items[Item::Plasma as usize] = true;
    // items[Item::Varia as usize] = true;
    // items[Item::Gravity as usize] = true;

    let weapon_mask = game_data.get_weapon_mask(&items);
    let global_state = GlobalState {
        tech: vec![true; game_data.tech_isv.keys.len()],
        flags: vec![false; game_data.flag_isv.keys.len()],
        items: items,
        max_energy: 1800,
        max_missiles: 230,
        max_reserves: 0,
        max_supers: 50,
        max_power_bombs: 0,
        shine_charge_tiles: 32,
        weapon_mask,
    };
    let local_state = LocalState {
        energy_used: 0,
        reserves_used: 0,
        missiles_used: 0,
        supers_used: 0,
        power_bombs_used: 0,
    };
    let reverse = false;
    let difficulty = DifficultyConfig {
        tech: vec![],
        shine_charge_tiles: 32,
        item_placement_strategy: ItemPlacementStrategy::Open,
        resource_multiplier: 1.0,
        escape_timer_multiplier: 1.0,
        ridley_proficiency: 0.5,
        save_animals: false,
        supers_double: true,
        streamlined_escape: true,
        mark_map_stations: true,
        mark_majors: true,
        debug_options: None,
    };

    let res = apply_requirement(
        &Requirement::RidleyFight {
            can_be_patient_tech_id: game_data.tech_isv.index_by_key["canBePatient"],
        },
        &global_state,
        local_state,
        reverse,
        &difficulty,
    );
    // let res = apply_requirement(&links[0].requirement, &global_state, local_state, reverse, &difficulty);
    println!("{:?}", res);
    Ok(())
}
