use anyhow::Result;
use maprando::{game_data::{GameData, Link, Item}, traverse::{GlobalState, LocalState, apply_requirement}, randomize::{DifficultyConfig, ItemPlacementStrategy}};
use std::path::Path;

fn main() -> Result<()> {
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let game_data = GameData::load(sm_json_data_path, room_geometry_path)?;

    let vertex_id_src = game_data.vertex_isv.index_by_key[&(77, 1, 0)];
    let vertex_id_dst = game_data.vertex_isv.index_by_key[&(77, 3, 3)];
    let mut links: Vec<Link> = Vec::new();
    for link in &game_data.links {
        if link.from_vertex_id == vertex_id_src && link.to_vertex_id == vertex_id_dst {
            println!("{}: {:?}", link.strat_name, link.requirement);
            links.push(link.clone());
            break;
        }
    }

    let mut items = vec![false; game_data.item_isv.keys.len()];
    items[Item::Missile as usize] = true;
    items[Item::SpaceJump as usize] = true;
    items[Item::Morph as usize] = true;
    let weapon_mask = game_data.get_weapon_mask(&items);
    let global_state = GlobalState {
        tech: vec![true; game_data.tech_isv.keys.len()],
        flags: vec![false; game_data.flag_isv.keys.len()],
        items: items,
        max_energy: 99,
        max_missiles: 5,
        max_reserves: 0,
        max_supers: 0,
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
        save_animals: false,
        debug_options: None,
    };

    let res = apply_requirement(&links[0].requirement, &global_state, local_state, reverse, &difficulty);
    println!("{:?}", res);
    Ok(())
}
