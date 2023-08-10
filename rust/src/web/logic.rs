use glob::glob;
use hashbrown::{HashMap, HashSet};
use json::JsonValue;
use sailfish::TemplateOnce;
use urlencoding;

use crate::game_data::{GameData, Link, NodeId, Requirement, RoomId};
use crate::randomize::{DebugOptions, DifficultyConfig};
use crate::traverse::{apply_requirement, GlobalState, LocalState};
use crate::web::VERSION;

use super::PresetData;

#[derive(Clone)]
struct RoomStrat {
    room_name: String,
    room_name_stripped: String,
    area: String,
    strat_name: String,
    strat_name_stripped: String,
    notable: bool,
    from_node_id: usize,
    from_node_name: String,
    to_node_id: usize,
    to_node_name: String,
    note: String,
    requires: String,                         // new-line separated requirements
    obstacles: Vec<(String, String, String)>, // list of (obstacle name, obstacle requires, additional obstacles)
    clears_obstacles: Vec<String>,
    difficulty_idx: usize,
    difficulty_name: String,
}

#[derive(TemplateOnce, Clone)]
#[template(path = "logic/room.stpl")]
struct RoomTemplate {
    version: usize,
    difficulty_names: Vec<String>,
    room_id: usize,
    room_name: String,
    room_name_stripped: String,
    room_name_url_encoded: String,
    area: String,
    room_diagram_path: String,
    nodes: Vec<(usize, String)>,
    strats: Vec<RoomStrat>,
    room_json: String,
}

#[derive(TemplateOnce, Clone)]
#[template(path = "logic/tech.stpl")]
struct TechTemplate {
    version: usize,
    difficulty_names: Vec<String>,
    tech_name: String,
    tech_note: String,
    tech_dependencies: String,
    tech_difficulty_idx: usize,
    tech_difficulty_name: String,
    strats: Vec<RoomStrat>,
    tech_gif_listing: HashSet<String>,
    area_order: Vec<String>,
}

#[derive(TemplateOnce, Clone)]
#[template(path = "logic/strat_page.stpl")]
struct StratTemplate {
    version: usize,
    room_id: usize,
    room_name: String,
    room_name_stripped: String,
    room_name_url_encoded: String,
    area: String,
    room_diagram_path: String,
    strat_name: String,
    strat: RoomStrat,
}

#[derive(TemplateOnce)]
#[template(path = "logic/logic.stpl")]
struct LogicIndexTemplate<'a> {
    version: usize,
    rooms: &'a [RoomTemplate],
    tech: &'a [TechTemplate],
    area_order: &'a [String],
    tech_difficulties: Vec<String>,
}

#[derive(Default)]
pub struct LogicData {
    pub index_html: String,                            // Logic index page
    pub room_html: HashMap<String, String>, // Map from room name (alphanumeric characters only) to rendered HTML.
    pub tech_html: HashMap<String, String>, // Map from tech name to rendered HTML.
    pub tech_strat_counts: HashMap<String, usize>, // Map from tech name to strat count using that tech.
    pub strat_html: HashMap<(String, usize, usize, String), String>, // Map from (room name, from node ID, to node ID, strat name) to rendered HTML.
}

fn list_room_diagram_files() -> HashMap<usize, String> {
    let mut out: HashMap<usize, String> = HashMap::new();
    for entry in glob("static/sm-json-data/region/*/roomDiagrams/*.png").unwrap() {
        match entry {
            Ok(path) => {
                let path_string = path.to_str().unwrap().to_string();
                let segments: Vec<&str> = path_string.split("_").collect();
                let subregion = segments[0];
                if subregion == "ceres" {
                    continue;
                }
                let room_id: usize = str::parse(segments[1]).unwrap();
                // let img = image::open(path).unwrap();
                // println!("{:?}", img.dimensions());
                out.insert(room_id, path_string);
            }
            Err(e) => panic!("Failure reading room diagrams: {:?}", e),
        }
    }
    out
}

fn make_requires(requires_json: &JsonValue) -> String {
    let mut out: Vec<String> = vec![];
    for req in requires_json.members() {
        out.push(req.pretty(2));
    }
    out.join("\n")
}

fn extract_tech_rec(req: &JsonValue, tech: &mut HashSet<usize>, game_data: &GameData) {
    if req.is_string() {
        let value = req.as_str().unwrap();
        if let Some(idx) = game_data.tech_isv.index_by_key.get(value) {
            // Skipping tech dependencies, so that only techs that explicitly appear in a strat (or via a helper)
            // will show up under the corresponding tech page.
            tech.insert(*idx);
        } else if let Some(helper_json) = game_data.helper_json_map.get(value) {
            for r in helper_json["requires"].members() {
                extract_tech_rec(r, tech, game_data);
            }
        }
    } else if req.is_object() && req.len() == 1 {
        let (key, value) = req.entries().next().unwrap();
        if key == "and" || key == "or" {
            for x in value.members() {
                extract_tech_rec(x, tech, game_data);
            }
        } else if key == "canShineCharge" && value["shinesparkFrames"].as_i32().unwrap() > 0 {
            tech.insert(game_data.tech_isv.index_by_key["canShinespark"]);
        } else if key == "canComeInCharged" && value["shinesparkFrames"].as_i32().unwrap() > 0 {
            tech.insert(game_data.tech_isv.index_by_key["canShinespark"]);
        } else if key == "comeInWithRMode" {
            tech.insert(game_data.tech_isv.index_by_key["canEnterRMode"]);
        } else if key == "comeInWithGMode" {
            tech.insert(game_data.tech_isv.index_by_key["canEnterGMode"]);
            if value["artificialMorph"].as_bool().unwrap() {
                tech.insert(game_data.tech_isv.index_by_key["canArtificialMorph"]);
            }
        }
    }
}

fn make_tech_templates<'a>(
    game_data: &GameData,
    room_templates: &[RoomTemplate],
    tech_gif_listing: &'a HashSet<String>,
    presets: &[PresetData],
    global_states: &[GlobalState],
    area_order: &[String],
) -> Vec<TechTemplate> {
    let mut tech_strat_ids: Vec<HashSet<(RoomId, NodeId, NodeId, String)>> =
        vec![HashSet::new(); game_data.tech_isv.keys.len()];
    for room_json in game_data.room_json_map.values() {
        let room_id = room_json["id"].as_usize().unwrap();
        for link_json in room_json["links"].members() {
            for link_to_json in link_json["to"].members() {
                for strat_json in link_to_json["strats"].members() {
                    let from_node_id = link_json["from"].as_usize().unwrap();
                    let to_node_id = link_to_json["id"].as_usize().unwrap();
                    let strat_name = strat_json["name"].as_str().unwrap().to_string();
                    let ids = (room_id, from_node_id, to_node_id, strat_name);
                    let mut tech_set: HashSet<usize> = HashSet::new();
                    for req in strat_json["requires"].members() {
                        extract_tech_rec(req, &mut tech_set, game_data);
                    }
                    for tech_idx in tech_set {
                        tech_strat_ids[tech_idx].insert(ids.clone());
                    }
                }
            }
        }
    }

    let mut room_strat_map: HashMap<(RoomId, NodeId, NodeId, String), &RoomStrat> = HashMap::new();
    for template in room_templates {
        for strat in &template.strats {
            room_strat_map.insert(
                (
                    template.room_id,
                    strat.from_node_id,
                    strat.to_node_id,
                    strat.strat_name.to_string(),
                ),
                strat,
            );
        }
    }

    let mut tech_templates: Vec<TechTemplate> = vec![];
    for (tech_idx, tech_ids) in tech_strat_ids.iter().enumerate() {
        let tech_name = game_data.tech_isv.keys[tech_idx].clone();
        let tech_note = game_data.tech_description[&tech_name].clone();
        let tech_dependencies = game_data.tech_dependencies[&tech_name].join(", ");
        let mut strats: Vec<RoomStrat> = vec![];
        let mut difficulty_idx = global_states.len();

        for (i, global) in global_states.iter().enumerate() {
            if global.tech[tech_idx] {
                difficulty_idx = i;
                break;
            }
        }
        let difficulty_name = if difficulty_idx == global_states.len() {
            "Beyond".to_string()
        } else {
            presets[difficulty_idx].preset.name.clone()
        };

        for strat_ids in tech_ids {
            // Infinitely-spawning farm strats aren't included (TODO: fix that?)
            if room_strat_map.contains_key(strat_ids) {
                strats.push(room_strat_map[strat_ids].clone());
            }
        }
        strats.sort_by_key(|s| {
            (
                area_order.iter().position(|a| a == &s.area).unwrap(),
                s.room_name.clone(),
                s.from_node_id,
                s.to_node_id,
                s.strat_name.clone(),
            )
        });
        let mut difficulty_names: Vec<String> =
            presets.iter().map(|x| x.preset.name.clone()).collect();
        difficulty_names.push("Beyond".to_string());
        let template = TechTemplate {
            version: VERSION,
            difficulty_names,
            tech_name: tech_name.clone(),
            tech_note,
            tech_dependencies,
            tech_difficulty_idx: difficulty_idx,
            tech_difficulty_name: difficulty_name,
            strats,
            tech_gif_listing: tech_gif_listing.clone(),
            area_order: area_order.to_vec(),
        };
        tech_templates.push(template);
    }
    tech_templates
}

fn strip_name(s: &str) -> String {
    let mut out = String::new();
    for word in s.split_inclusive(|x: char| !x.is_ascii_alphabetic()) {
        let capitalized_word = word[0..1].to_ascii_uppercase() + &word[1..];
        let stripped_word: String = capitalized_word.chars().filter(|x| x.is_ascii_alphanumeric()).collect();
        out += &stripped_word;
    }
    out
    // s.chars().filter(|x| x.is_ascii_alphanumeric()).collect()
}

fn get_difficulty_config(preset: &PresetData) -> DifficultyConfig {
    let mut tech_vec: Vec<String> = vec![];
    for (tech_name, enabled) in &preset.tech_setting {
        if *enabled {
            tech_vec.push(tech_name.clone());
        }
    }
    let mut strat_vec: Vec<String> = vec![];
    for (strat_name, enabled) in &preset.notable_strat_setting {
        if *enabled {
            strat_vec.push(strat_name.clone());
        }
    }
    // It's annoying how much irrelevant stuff we have to fill in here. TODO: restructure to make things cleaner
    DifficultyConfig {
        tech: tech_vec,
        notable_strats: strat_vec,
        shine_charge_tiles: preset.preset.shinespark_tiles as f32,
        progression_rate: crate::randomize::ProgressionRate::Fast,
        item_placement_style: crate::randomize::ItemPlacementStyle::Forced,
        item_priorities: vec![],
        filler_items: vec![],
        early_filler_items: vec![],
        resource_multiplier: preset.preset.resource_multiplier,
        escape_timer_multiplier: preset.preset.escape_timer_multiplier,
        gate_glitch_leniency: preset.preset.gate_glitch_leniency as i32,
        phantoon_proficiency: preset.preset.phantoon_proficiency,
        draygon_proficiency: preset.preset.draygon_proficiency,
        ridley_proficiency: preset.preset.ridley_proficiency,
        botwoon_proficiency: preset.preset.botwoon_proficiency,
        supers_double: true,
        mother_brain_fight: crate::randomize::MotherBrainFight::Short,
        escape_movement_items: true,
        escape_refill: true,
        escape_enemies_cleared: true,
        mark_map_stations: true,
        transition_letters: false,
        item_markers: crate::randomize::ItemMarkers::ThreeTiered,
        all_items_spawn: true,
        acid_chozo: true,
        fast_elevators: true,
        fast_doors: true,
        fast_pause_menu: true,
        respin: false,
        infinite_space_jump: false,
        momentum_conservation: false,
        objectives: crate::randomize::Objectives::Bosses,
        save_animals: false,
        randomized_start: false,
        disable_walljump: false,
        maps_revealed: false,
        vanilla_map: false,
        ultra_low_qol: false,
        skill_assumptions_preset: None,
        item_progression_preset: None,
        quality_of_life_preset: None,
        debug_options: Some(DebugOptions {
            new_game_extra: false,
            extended_spoiler: false,
        }),
    }
}

fn strip_cross_room_reqs(req: Requirement, game_data: &GameData) -> Requirement {
    match req {
        Requirement::And(subreqs) => Requirement::And(
            subreqs
                .into_iter()
                .map(|x| strip_cross_room_reqs(x, game_data))
                .collect(),
        ),
        Requirement::Or(subreqs) => Requirement::Or(
            subreqs
                .into_iter()
                .map(|x| strip_cross_room_reqs(x, game_data))
                .collect(),
        ),
        Requirement::AdjacentJumpway { .. } => Requirement::Free,
        Requirement::AdjacentRunway { .. } => Requirement::Free,
        Requirement::CanComeInCharged { .. } => {
            Requirement::Tech(game_data.tech_isv.index_by_key["canShinespark"])
        }
        Requirement::ComeInWithRMode { .. } => {
            Requirement::Tech(game_data.tech_isv.index_by_key["canEnterRMode"])
        }
        Requirement::ComeInWithGMode { .. } => {
            Requirement::Tech(game_data.tech_isv.index_by_key["canEnterGMode"])
        }
        Requirement::NotFlag(_) => Requirement::Free,
        _ => req,
    }
}

fn get_strat_difficulty(
    room_id: usize,
    from_node_id: usize,
    to_node_id: usize,
    strat_name: String,
    game_data: &GameData,
    difficulty_configs: &[DifficultyConfig],
    global_states: &[GlobalState],
    links_by_ids: &HashMap<(RoomId, NodeId, NodeId, String), Vec<Link>>,
) -> usize {
    for (i, difficulty) in difficulty_configs.iter().enumerate() {
        let global = &global_states[i];

        let local = LocalState {
            energy_used: 0,
            reserves_used: 0,
            missiles_used: 0,
            supers_used: 0,
            power_bombs_used: 0,
        };

        for link in &links_by_ids[&(room_id, from_node_id, to_node_id, strat_name.clone())] {
            let req = strip_cross_room_reqs(link.requirement.clone(), game_data);
            let new_local = apply_requirement(&req, &global, local, false, difficulty);
            if new_local.is_some() {
                return i;
            }
        }
    }
    difficulty_configs.len()
}

fn make_room_template(
    room_json: &JsonValue,
    room_diagram_listing: &HashMap<usize, String>,
    game_data: &GameData,
    presets: &[PresetData],
    difficulty_configs: &[DifficultyConfig],
    global_states: &[GlobalState],
    links_by_ids: &HashMap<(RoomId, NodeId, NodeId, String), Vec<Link>>,
) -> RoomTemplate {
    let mut room_strats: Vec<RoomStrat> = vec![];
    let room_id = room_json["id"].as_usize().unwrap();
    let room_name = room_json["name"].as_str().unwrap().to_string();
    let room_name_stripped = strip_name(&room_name);
    let mut node_name_map: HashMap<usize, String> = HashMap::new();
    let mut nodes: Vec<(usize, String)> = vec![];
    for node_json in room_json["nodes"].members() {
        let node_id = node_json["id"].as_usize().unwrap();
        let node_name = node_json["name"].as_str().unwrap();
        node_name_map.insert(node_id, node_name.to_string());
        nodes.push((node_id, node_name.to_string()));
    }
    let area = room_json["area"].as_str().unwrap().to_string();
    let sub_area = room_json["subarea"].as_str().unwrap_or("").to_string();
    let sub_sub_area = room_json["subsubarea"].as_str().unwrap_or("").to_string();
    let full_area = if sub_sub_area != "" {
        format!("{} {} {}", sub_sub_area, sub_area, area)
    } else if sub_area != "" && sub_area != "Main" {
        format!("{} {}", sub_area, area)
    } else {
        area
    };

    for link_json in room_json["links"].members() {
        for link_to_json in link_json["to"].members() {
            for strat_json in link_to_json["strats"].members() {
                let from_node_id = link_json["from"].as_usize().unwrap();
                let to_node_id = link_to_json["id"].as_usize().unwrap();
                let mut obstacles: Vec<(String, String, String)> = vec![];
                for obstacle_json in strat_json["obstacles"].members() {
                    let obstacle_id = obstacle_json["id"].as_str().unwrap().to_string();
                    let obstacle_requires = make_requires(&obstacle_json["requires"]);
                    let mut additional: Vec<String> = vec![];
                    for x in obstacle_json["additionalObstacles"].members() {
                        additional.push(x.as_str().unwrap().to_string());
                    }
                    obstacles.push((obstacle_id, obstacle_requires, additional.join(", ")));
                }
                let strat_name = strat_json["name"].as_str().unwrap().to_string();
                let difficulty_idx = get_strat_difficulty(
                    room_id,
                    from_node_id,
                    to_node_id,
                    strat_name,
                    game_data,
                    difficulty_configs,
                    global_states,
                    links_by_ids,
                );
                let difficulty_name = if difficulty_idx == difficulty_configs.len() {
                    "Beyond".to_string()
                } else {
                    presets[difficulty_idx].preset.name.clone()
                };
                let clears_obstacles: Vec<String> = if strat_json.has_key("clearsObstacles") {
                    strat_json["clearsObstacles"]
                        .members()
                        .map(|x| x.as_str().unwrap().to_string())
                        .collect()
                } else {
                    vec![]
                };
                let strat_name = strat_json["name"].as_str().unwrap().to_string();
                let strat = RoomStrat {
                    room_name: room_name.clone(),
                    room_name_stripped: room_name_stripped.clone(),
                    area: full_area.clone(),
                    strat_name: strat_name.clone(),
                    strat_name_stripped: strip_name(&strat_name),
                    notable: strat_json["notable"].as_bool().unwrap_or(false),
                    from_node_id,
                    from_node_name: node_name_map[&from_node_id].clone(),
                    to_node_id,
                    to_node_name: node_name_map[&to_node_id].clone(),
                    note: game_data.parse_note(&strat_json["note"]).join(" "),
                    requires: make_requires(&strat_json["requires"]),
                    obstacles,
                    clears_obstacles,
                    difficulty_idx,
                    difficulty_name,
                };
                room_strats.push(strat);
            }
        }
    }
    // let shape = *game_data.room_shape.get(&room_id).unwrap_or(&(1, 1));
    let mut difficulty_names: Vec<String> = presets.iter().map(|x| x.preset.name.clone()).collect();
    difficulty_names.push("Beyond".to_string());

    RoomTemplate {
        version: VERSION,
        difficulty_names,
        room_id,
        room_name_url_encoded: urlencoding::encode(&room_name).into_owned(),
        room_name,
        room_name_stripped,
        area: full_area,
        room_diagram_path: room_diagram_listing[&room_id].clone(),
        nodes,
        strats: room_strats,
        room_json: room_json.pretty(2),
    }
}

fn make_strat_template(
    room: &RoomTemplate,
    strat: &RoomStrat,
) -> StratTemplate {
    StratTemplate {
        version: VERSION,
        room_id: room.room_id,
        room_name: room.room_name.clone(),
        room_name_stripped: room.room_name_stripped.clone(),
        room_name_url_encoded: room.room_name_url_encoded.clone(),
        area: room.area.clone(),
        room_diagram_path: room.room_diagram_path.clone(),
        strat_name: strat.strat_name.clone(),
        strat: strat.clone(),
    }
}

impl LogicData {
    pub fn new(
        game_data: &GameData,
        tech_gif_listing: &HashSet<String>,
        presets: &[PresetData],
    ) -> LogicData {
        let mut out = LogicData::default();
        let room_diagram_listing = list_room_diagram_files();
        let mut room_templates: Vec<RoomTemplate> = vec![];
        let difficulty_configs: Vec<DifficultyConfig> =
            presets.iter().map(get_difficulty_config).collect();

        let area_order: Vec<String> = vec![
            "Central Crateria",
            "West Crateria",
            "East Crateria",
            "Blue Brinstar",
            "Green Brinstar",
            "Pink Brinstar",
            "Red Brinstar",
            "Kraid Brinstar",
            "East Upper Norfair",
            "West Upper Norfair",
            "Crocomire Upper Norfair",
            "West Lower Norfair",
            "East Lower Norfair",
            "Outer Maridia",
            "Pink Inner Maridia",
            "Yellow Inner Maridia",
            "Green Inner Maridia",
            "Wrecked Ship",
            "Tourian",
        ]
        .into_iter()
        .map(|x| x.to_string())
        .collect();

        let mut global_states: Vec<GlobalState> = vec![];
        for difficulty in &difficulty_configs {
            let items = vec![true; game_data.item_isv.keys.len()];
            let weapon_mask = game_data.get_weapon_mask(&items);

            let mut tech = vec![false; game_data.tech_isv.keys.len()];
            for tech_name in &difficulty.tech {
                tech[game_data.tech_isv.index_by_key[tech_name]] = true;
            }

            let mut notable_strats = vec![false; game_data.notable_strat_isv.keys.len()];
            for strat_name in &difficulty.notable_strats {
                notable_strats[game_data.notable_strat_isv.index_by_key[strat_name]] = true;
            }

            let global = GlobalState {
                tech,
                notable_strats,
                items: items,
                flags: vec![true; game_data.flag_isv.keys.len()],
                max_energy: 1499,
                max_reserves: 400,
                max_missiles: 230,
                max_supers: 50,
                max_power_bombs: 50,
                weapon_mask: weapon_mask,
                shine_charge_tiles: difficulty.shine_charge_tiles,
            };

            global_states.push(global);
        }

        let mut links_by_ids: HashMap<(RoomId, NodeId, NodeId, String), Vec<Link>> = HashMap::new();
        for link in &game_data.links {
            let (link_room_id, link_from_node_id, _) =
                game_data.vertex_isv.keys[link.from_vertex_id];
            let (_, link_to_node_id, _) = game_data.vertex_isv.keys[link.to_vertex_id];
            let link_ids = (
                link_room_id,
                link_from_node_id,
                link_to_node_id,
                link.strat_name.clone(),
            );
            links_by_ids
                .entry(link_ids)
                .or_insert(vec![])
                .push(link.clone());
        }

        for (_, room_json) in game_data.room_json_map.iter() {
            let template = make_room_template(
                room_json,
                &room_diagram_listing,
                &game_data,
                presets,
                &difficulty_configs,
                &global_states,
                &links_by_ids,
            );
            let html = template.clone().render_once().unwrap();
            let stripped_room_name = strip_name(&template.room_name);
            out.room_html.insert(stripped_room_name.clone(), html);
            room_templates.push(template.clone());

            for strat in &template.strats {
                let strat_template = make_strat_template(&template, &strat);
                let strat_html = strat_template.render_once().unwrap();
                let stripped_strat_name = strip_name(&strat.strat_name);
                out.strat_html
                    .insert((stripped_room_name.clone(), strat.from_node_id, strat.to_node_id, stripped_strat_name), strat_html);
            }
        }
        room_templates.sort_by_key(|x| (x.area.clone(), x.room_name.clone()));

        let tech_templates = make_tech_templates(
            game_data,
            &room_templates,
            tech_gif_listing,
            presets,
            &global_states,
            &area_order,
        );
        for template in &tech_templates {
            let html = template.clone().render_once().unwrap();
            let strat_count = template
                .strats
                .iter()
                .filter(|x| x.difficulty_idx <= template.tech_difficulty_idx)
                .count();
            out.tech_strat_counts
                .insert(template.tech_name.clone(), strat_count);
            out.tech_html.insert(template.tech_name.clone(), html);
        }

        let index_template = LogicIndexTemplate {
            version: VERSION,
            rooms: &room_templates,
            tech: &tech_templates,
            area_order: &area_order,
            tech_difficulties: presets.iter().map(|x| x.preset.name.clone()).collect(),
        };
        out.index_html = index_template.render_once().unwrap();
        out
    }
}
