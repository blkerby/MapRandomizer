use askama::Template;
use glob::glob;
use hashbrown::{HashMap, HashSet};
use json::JsonValue;
use log::warn;
use maprando::{
    randomize::DifficultyConfig,
    traverse::{apply_requirement, LockedDoorData},
};
use maprando_game::{
    Capacity, ExitCondition, GameData, Link, MainEntranceCondition, NodeId, NotableId, NotableIdx,
    Requirement, RoomId, StratId, StratVideo, TechId, VertexAction, VertexKey,
    TECH_ID_CAN_ARTIFICIAL_MORPH, TECH_ID_CAN_BOMB_HORIZONTALLY, TECH_ID_CAN_ENEMY_STUCK_MOONFALL,
    TECH_ID_CAN_ENTER_G_MODE, TECH_ID_CAN_ENTER_R_MODE, TECH_ID_CAN_GRAPPLE_TELEPORT,
    TECH_ID_CAN_MOONDANCE, TECH_ID_CAN_SHINESPARK, TECH_ID_CAN_SKIP_DOOR_LOCK,
    TECH_ID_CAN_SPEEDBALL, TECH_ID_CAN_STUTTER_WATER_SHINECHARGE, TECH_ID_CAN_TEMPORARY_BLUE,
};
use maprando_logic::{GlobalState, Inventory, LocalState};
use std::path::PathBuf;
use urlencoding;

use super::{PresetData, VersionInfo};

#[derive(Clone)]
struct RoomStrat {
    room_id: usize,
    room_name: String,
    area: String,
    strat_id: usize,
    strat_name: String,
    bypasses_door_shell: bool,
    from_node_id: usize,
    from_node_name: String,
    to_node_id: usize,
    to_node_name: String,
    note: String,
    entrance_condition: Option<String>,
    requires: String, // new-line separated requirements
    exit_condition: Option<String>,
    clears_obstacles: Vec<String>,
    resets_obstacles: Vec<String>,
    unlocks_doors: Option<String>,
    difficulty_idx: usize,
    difficulty_name: String,
}

#[derive(Template, Clone)]
#[template(path = "logic/room.html")]
struct RoomTemplate<'a> {
    version_info: VersionInfo,
    difficulty_names: Vec<String>,
    room_id: usize,
    room_name: String,
    room_name_url_encoded: String,
    area: String,
    room_diagram_path: String,
    nodes: Vec<(usize, String)>,
    strats: Vec<RoomStrat>,
    strat_videos: &'a HashMap<(RoomId, StratId), Vec<StratVideo>>,
    room_json: String,
    video_storage_url: String,
}

#[derive(Template, Clone)]
#[template(path = "logic/tech.html")]
struct TechTemplate<'a> {
    version_info: VersionInfo,
    difficulty_names: Vec<String>,
    tech_id: TechId,
    tech_name: String,
    tech_note: String,
    tech_dependencies: String,
    tech_difficulty_idx: usize,
    tech_difficulty_name: String,
    strats: Vec<RoomStrat>,
    strat_videos: &'a HashMap<(RoomId, StratId), Vec<StratVideo>>,
    tech_video_id: Option<usize>,
    video_storage_url: String,
}

#[derive(Template, Clone)]
#[template(path = "logic/notable.html")]
struct NotableTemplate<'a> {
    version_info: VersionInfo,
    difficulty_names: Vec<String>,
    room_id: RoomId,
    room_name: String,
    notable_id: NotableId,
    notable_name: String,
    notable_note: String,
    notable_difficulty_idx: usize,
    notable_difficulty_name: String,
    strats: Vec<RoomStrat>,
    strat_videos: &'a HashMap<(RoomId, StratId), Vec<StratVideo>>,
    notable_video_id: Option<usize>,
    video_storage_url: String,
}

#[derive(Template, Clone)]
#[template(path = "logic/strat_page.html")]
struct StratTemplate<'a> {
    version_info: VersionInfo,
    room_id: usize,
    room_name: String,
    room_name_url_encoded: String,
    room_diagram_path: String,
    strat_name: String,
    strat: RoomStrat,
    strat_videos: &'a HashMap<(RoomId, StratId), Vec<StratVideo>>,
    video_storage_url: String,
}

#[derive(Template)]
#[template(path = "logic/logic.html")]
struct LogicIndexTemplate<'a> {
    version_info: VersionInfo,
    rooms: &'a [RoomTemplate<'a>],
    tech: &'a [TechTemplate<'a>],
    _notables: &'a [NotableTemplate<'a>],
    area_order: &'a [String],
    tech_difficulties: Vec<String>,
}

#[derive(Default)]
pub struct LogicData {
    pub index_html: String,                                 // Logic index page
    pub room_html: HashMap<RoomId, String>,                 // Map from room ID to rendered HTML.
    pub tech_html: HashMap<TechId, String>,                 // Map from tech ID to rendered HTML.
    pub tech_strat_counts: HashMap<TechId, usize>, // Map from tech ID to strat count using that tech.
    pub notable_html: HashMap<(RoomId, NotableId), String>, // Map from room/notable ID to rendered HTML.
    pub notable_strat_counts: HashMap<(RoomId, NotableId), usize>, // Map from tech ID to strat count using that tech.
    pub strat_html: HashMap<(RoomId, NodeId, NodeId, StratId), String>, // Map from (room ID, from node ID, to node ID, strat ID) to rendered HTML.
}

fn list_room_diagram_files() -> HashMap<usize, String> {
    let mut out: HashMap<usize, String> = HashMap::new();
    for entry in glob("../sm-json-data/region/*/roomDiagrams/*.png").unwrap() {
        match entry {
            Ok(path) => {
                let mut new_path = PathBuf::new();
                new_path = new_path.join("static");
                for c in path.components().skip(1) {
                    new_path = new_path.join(c);
                }

                let path_string = new_path.to_str().unwrap().to_string();
                let segments: Vec<&str> = path_string.split(|c| c == '_' || c == '.').collect();
                let subregion = segments[0];
                if subregion == "ceres" {
                    continue;
                }
                let room_id: usize = str::parse(segments[2]).unwrap();
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
        if let Some(tech_id) = game_data.tech_id_by_name.get(value) {
            // Skipping tech dependencies, so that only techs that explicitly appear in a strat (or via a helper)
            // will show up under the corresponding tech page.
            let tech_idx = game_data.tech_isv.index_by_key[tech_id];
            tech.insert(tech_idx);
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
        } else if key == "tech" {
            extract_tech_rec(value, tech, game_data);
        } else if key == "shinespark" {
            tech.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_SHINESPARK]);
        } else if key == "comeInWithRMode" {
            tech.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_R_MODE]);
        } else if key == "comeInWithGMode" {
            tech.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE]);
            if value["artificialMorph"].as_bool().unwrap() {
                tech.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_ARTIFICIAL_MORPH]);
            }
        }
    }
}

fn extract_notable_rec(
    req: &JsonValue,
    room_id: RoomId,
    notables: &mut HashSet<NotableIdx>,
    game_data: &GameData,
) {
    if req.is_object() && req.len() == 1 {
        let (key, value) = req.entries().next().unwrap();
        if key == "notable" {
            let notable_id =
                game_data.notable_id_by_name[&(room_id, value.as_str().unwrap().to_string())];
            let notable_idx = game_data.notable_isv.index_by_key[&(room_id, notable_id)];
            notables.insert(notable_idx);
        } else if key == "and" || key == "or" {
            for x in value.members() {
                extract_notable_rec(x, room_id, notables, game_data);
            }
        }
    }
}

fn make_tech_templates<'a>(
    game_data: &'a GameData,
    room_templates: &[RoomTemplate<'a>],
    presets: &[PresetData],
    global_states: &[GlobalState],
    area_order: &[String],
    video_storage_url: &str,
    version_info: &VersionInfo,
) -> Vec<TechTemplate<'a>> {
    let mut tech_strat_ids: Vec<HashSet<(RoomId, NodeId, NodeId, String)>> =
        vec![HashSet::new(); game_data.tech_isv.keys.len()];
    for room_json in game_data.room_json_map.values() {
        let room_id = room_json["id"].as_usize().unwrap();
        for strat_json in room_json["strats"].members() {
            let from_node_id = strat_json["link"][0].as_usize().unwrap();
            let to_node_id = strat_json["link"][1].as_usize().unwrap();
            let strat_name = strat_json["name"].as_str().unwrap().to_string();
            let ids = (room_id, from_node_id, to_node_id, strat_name);
            let mut tech_set: HashSet<usize> = HashSet::new();
            for req in strat_json["requires"].members() {
                extract_tech_rec(req, &mut tech_set, game_data);
            }
            if strat_json["bypassesDoorShell"].as_bool() == Some(true) {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_SKIP_DOOR_LOCK]);
            }
            if strat_json["entranceCondition"].has_key("comeInWithGMode") {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE]);
            }
            if strat_json["entranceCondition"].has_key("comeInWithRMode") {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_R_MODE]);
            }
            if strat_json["entranceCondition"].has_key("comeInSpeedballing") {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPEEDBALL]);
            }
            if strat_json["entranceCondition"].has_key("comeInStutterShinecharging") {
                tech_set.insert(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_STUTTER_WATER_SHINECHARGE],
                );
            }
            if strat_json["entranceCondition"].has_key("comeInWithTemporaryBlue") {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_TEMPORARY_BLUE]);
            }
            if strat_json["entranceCondition"].has_key("comeInWithBombBoost") {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_BOMB_HORIZONTALLY]);
            }
            if strat_json["entranceCondition"].has_key("comeInWithGrappleTeleport") {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_TELEPORT]);
            }
            if strat_json["exitCondition"].has_key("leaveWithGModeSetup") {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE]);
            }
            if strat_json["exitCondition"].has_key("leaveWithGMode") {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE]);
            }
            if strat_json["exitCondition"].has_key("leaveWithGrappleTeleport") {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_TELEPORT]);
            }

            for tech_idx in tech_set {
                tech_strat_ids[tech_idx].insert(ids.clone());
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

    let mut tech_templates: Vec<TechTemplate<'a>> = vec![];
    for (tech_idx, tech_ids) in tech_strat_ids.iter().enumerate() {
        let tech_id = game_data.tech_isv.keys[tech_idx].clone();
        let tech_note = game_data.tech_description[&tech_id].clone();
        let tech_dependency_names: Vec<String> = game_data.tech_dependencies[&tech_id]
            .iter()
            .map(|tech_id| {
                game_data.tech_json_map[tech_id]["name"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        let tech_dependencies = tech_dependency_names.join(", ");
        let mut strats: Vec<RoomStrat> = vec![];
        let mut difficulty_idx = global_states.len();

        for (i, global) in global_states.iter().enumerate() {
            if global.tech[tech_idx] {
                difficulty_idx = i;
                break;
            }
        }
        let difficulty_name = if difficulty_idx == global_states.len() {
            "Ignored".to_string()
        } else {
            presets[difficulty_idx].preset.name.clone()
        };

        for strat_ids in tech_ids {
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
        let difficulty_names: Vec<String> = presets.iter().map(|x| x.preset.name.clone()).collect();
        let template = TechTemplate {
            version_info: version_info.clone(),
            difficulty_names,
            tech_id,
            tech_name: game_data.tech_json_map[&tech_id]["name"]
                .as_str()
                .unwrap()
                .to_string(),
            tech_note,
            tech_dependencies,
            tech_difficulty_idx: difficulty_idx,
            tech_difficulty_name: difficulty_name,
            strats,
            strat_videos: &game_data.strat_videos,
            tech_video_id: presets.last().unwrap().tech_setting[tech_idx].0.video_id,
            video_storage_url: video_storage_url.to_string(),
        };
        tech_templates.push(template);
    }
    tech_templates
}

fn make_notable_templates<'a>(
    game_data: &'a GameData,
    room_templates: &[RoomTemplate<'a>],
    presets: &[PresetData],
    global_states: &[GlobalState],
    area_order: &[String],
    video_storage_url: &str,
    version_info: &VersionInfo,
) -> Vec<NotableTemplate<'a>> {
    let mut notable_strat_ids: Vec<HashSet<(RoomId, NodeId, NodeId, String)>> =
        vec![HashSet::new(); game_data.notable_isv.keys.len()];
    for room_json in game_data.room_json_map.values() {
        let room_id = room_json["id"].as_usize().unwrap();
        for strat_json in room_json["strats"].members() {
            let from_node_id = strat_json["link"][0].as_usize().unwrap();
            let to_node_id = strat_json["link"][1].as_usize().unwrap();
            let strat_name = strat_json["name"].as_str().unwrap().to_string();
            let ids = (room_id, from_node_id, to_node_id, strat_name);
            let mut notable_set: HashSet<NotableIdx> = HashSet::new();
            for req in strat_json["requires"].members() {
                extract_notable_rec(req, room_id, &mut notable_set, game_data);
            }

            for notable_idx in notable_set {
                notable_strat_ids[notable_idx].insert(ids.clone());
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

    let mut notable_templates: Vec<NotableTemplate<'a>> = vec![];
    for (notable_idx, ids_set) in notable_strat_ids.iter().enumerate() {
        let room_id = game_data.notable_data[notable_idx].room_id.clone();
        let notable_id = game_data.notable_data[notable_idx].notable_id.clone();
        let notable_name = game_data.notable_data[notable_idx].name.clone();
        let notable_note = game_data.notable_data[notable_idx].note.clone();
        let mut strats: Vec<RoomStrat> = vec![];
        let mut difficulty_idx = global_states.len();

        for (i, global) in global_states.iter().enumerate() {
            if global.notables[notable_idx] {
                difficulty_idx = i;
                break;
            }
        }
        let difficulty_name = if difficulty_idx == global_states.len() {
            "Ignored".to_string()
        } else {
            presets[difficulty_idx].preset.name.clone()
        };

        for ids in ids_set {
            if room_strat_map.contains_key(ids) {
                strats.push(room_strat_map[ids].clone());
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
        let difficulty_names: Vec<String> = presets.iter().map(|x| x.preset.name.clone()).collect();
        let template = NotableTemplate {
            version_info: version_info.clone(),
            difficulty_names,
            room_id,
            room_name: game_data.room_json_map[&room_id]["name"]
                .as_str()
                .unwrap()
                .to_string(),
            notable_id,
            notable_name,
            notable_note,
            notable_difficulty_idx: difficulty_idx,
            notable_difficulty_name: difficulty_name,
            strats,
            strat_videos: &game_data.strat_videos,
            notable_video_id: presets.last().unwrap().notable_setting[notable_idx]
                .0
                .video_id,
            video_storage_url: video_storage_url.to_string(),
        };
        notable_templates.push(template);
    }
    notable_templates
}

fn get_difficulty_config(preset: &PresetData, _game_data: &GameData) -> DifficultyConfig {
    let mut tech_vec: Vec<TechId> = vec![];
    for (tech_setting, enabled) in &preset.tech_setting {
        if *enabled {
            tech_vec.push(tech_setting.tech_id);
        }
    }
    let mut notable_vec: Vec<(RoomId, NotableId)> = vec![];
    for (notable_setting, enabled) in preset.notable_setting.iter() {
        if *enabled {
            notable_vec.push((notable_setting.room_id, notable_setting.notable_id));
        }
    }
    // It's annoying how much irrelevant stuff we have to fill in here. TODO: restructure to make things cleaner
    DifficultyConfig {
        name: None,
        tech: tech_vec,
        notables: notable_vec,
        shine_charge_tiles: preset.preset.shinespark_tiles as f32,
        heated_shine_charge_tiles: preset.preset.heated_shinespark_tiles as f32,
        speed_ball_tiles: preset.preset.speed_ball_tiles as f32,
        shinecharge_leniency_frames: preset.preset.shinecharge_leniency_frames as Capacity,
        progression_rate: maprando::settings::ProgressionRate::Fast,
        random_tank: true,
        spazer_before_plasma: true,
        stop_item_placement_early: false,
        item_pool: vec![],
        starting_items: vec![],
        item_placement_style: maprando::settings::ItemPlacementStyle::Forced,
        item_priority_strength: maprando::settings::ItemPriorityStrength::Moderate,
        item_priorities: vec![],
        filler_items: vec![],
        semi_filler_items: vec![],
        early_filler_items: vec![],
        resource_multiplier: preset.preset.resource_multiplier,
        escape_timer_multiplier: preset.preset.escape_timer_multiplier,
        gate_glitch_leniency: preset.preset.gate_glitch_leniency as Capacity,
        door_stuck_leniency: preset.preset.door_stuck_leniency as Capacity,
        phantoon_proficiency: preset.preset.phantoon_proficiency,
        draygon_proficiency: preset.preset.draygon_proficiency,
        ridley_proficiency: preset.preset.ridley_proficiency,
        botwoon_proficiency: preset.preset.botwoon_proficiency,
        mother_brain_proficiency: preset.preset.mother_brain_proficiency,
        supers_double: true,
        mother_brain_fight: maprando::settings::MotherBrainFight::Short,
        escape_movement_items: true,
        escape_refill: true,
        escape_enemies_cleared: true,
        mark_map_stations: true,
        room_outline_revealed: true,
        opposite_area_revealed: true,
        transition_letters: false,
        door_locks_size: maprando::settings::DoorLocksSize::Small,
        item_markers: maprando::settings::ItemMarkers::ThreeTiered,
        item_dot_change: maprando::settings::ItemDotChange::Fade,
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
        doors_mode: maprando::settings::DoorsMode::Ammo,
        save_animals: maprando::settings::SaveAnimals::No,
        start_location_mode: maprando::settings::StartLocationMode::Ship,
        early_save: false,
        area_assignment: maprando::settings::AreaAssignment::Standard,
        wall_jump: maprando::settings::WallJump::Vanilla,
        etank_refill: maprando::settings::ETankRefill::Vanilla,
        maps_revealed: maprando::settings::MapsRevealed::Full,
        map_station_reveal: maprando::settings::MapStationReveal::Full,
        vanilla_map: false,
        ultra_low_qol: false,
        energy_free_shinesparks: false,
        skill_assumptions_preset: None,
        item_progression_preset: None,
        quality_of_life_preset: None,
        debug: false,
    }
}

fn get_cross_room_reqs(link: &Link, game_data: &GameData) -> Requirement {
    let mut reqs: Vec<Requirement> = vec![];
    let from_vertex_key = &game_data.vertex_isv.keys[link.from_vertex_id];
    let to_vertex_key = &game_data.vertex_isv.keys[link.to_vertex_id];
    for action in &from_vertex_key.actions {
        if let VertexAction::Enter(entrance_condition) = action {
            let main = &entrance_condition.main;
            if let MainEntranceCondition::ComeInWithGMode { .. } = main {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE],
                ));
            }
            if let MainEntranceCondition::ComeInWithRMode { .. } = main {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_R_MODE],
                ));
            }
            if let MainEntranceCondition::ComeInSpeedballing { .. } = main {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPEEDBALL],
                ));
            }
            if let MainEntranceCondition::ComeInStutterShinecharging { .. } = main {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_STUTTER_WATER_SHINECHARGE],
                ));
            }
            if let MainEntranceCondition::ComeInWithTemporaryBlue { .. } = main {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_TEMPORARY_BLUE],
                ));
            }
            if let MainEntranceCondition::ComeInWithBombBoost {} = main {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_BOMB_HORIZONTALLY],
                ));
            }
            if let MainEntranceCondition::ComeInWithGrappleTeleport { .. } = main {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_TELEPORT],
                ));
            }
            if let MainEntranceCondition::ComeInWithStoredFallSpeed { .. } = main {
                reqs.push(Requirement::Or(vec![
                    Requirement::Tech(game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOONDANCE]),
                    Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENEMY_STUCK_MOONFALL],
                    ),
                ]));
            }
        }
    }
    for action in &to_vertex_key.actions {
        if let VertexAction::Exit(exit_condition) = action {
            if let ExitCondition::LeaveWithGMode { .. } = exit_condition {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE],
                ));
            }
            if let ExitCondition::LeaveWithGModeSetup { .. } = exit_condition {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE],
                ));
            }
            if let ExitCondition::LeaveWithGrappleTeleport { .. } = exit_condition {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_TELEPORT],
                ));
            }
        }
    }
    Requirement::make_and(reqs)
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
        Requirement::DoorUnlocked { .. } => Requirement::Free,
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
    let locked_door_data = LockedDoorData {
        locked_doors: vec![],
        locked_door_node_map: HashMap::new(),
        locked_door_vertex_ids: vec![],
    };
    for (i, difficulty) in difficulty_configs.iter().enumerate() {
        if i == 0 {
            // Skip the "Implicit" difficulty
            continue;
        }
        let global = &global_states[i];

        let local = LocalState {
            energy_used: 0,
            reserves_used: 0,
            missiles_used: 0,
            supers_used: 0,
            power_bombs_used: 0,
            shinecharge_frames_remaining: 180 - difficulty.shinecharge_leniency_frames,
        };

        let key = (room_id, from_node_id, to_node_id, strat_name.clone());
        if !links_by_ids.contains_key(&key) {
            return difficulty_configs.len();
        }
        for link in &links_by_ids[&key] {
            let extra_req = get_cross_room_reqs(link, game_data);
            let main_req = strip_cross_room_reqs(link.requirement.clone(), game_data);
            let combined_req = Requirement::make_and(vec![extra_req, main_req]);
            let new_local = apply_requirement(
                &combined_req,
                &global,
                local,
                false,
                difficulty,
                game_data,
                &locked_door_data,
            );
            if new_local.is_some() {
                return i;
            }
        }
    }
    difficulty_configs.len()
}

fn make_room_template<'a>(
    room_json: &JsonValue,
    room_diagram_listing: &HashMap<usize, String>,
    game_data: &'a GameData,
    presets: &[PresetData],
    difficulty_configs: &[DifficultyConfig],
    global_states: &[GlobalState],
    links_by_ids: &HashMap<(RoomId, NodeId, NodeId, String), Vec<Link>>,
    video_storage_url: &str,
    version_info: &VersionInfo,
) -> RoomTemplate<'a> {
    let mut room_strats: Vec<RoomStrat> = vec![];
    let room_id = room_json["id"].as_usize().unwrap();
    let room_name = room_json["name"].as_str().unwrap().to_string();
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

    for strat_json in room_json["strats"].members() {
        if !strat_json["id"].is_number() {
            continue;
        }
        let strat_id = strat_json["id"].as_usize().unwrap();
        let from_node_id = strat_json["link"][0].as_usize().unwrap();
        let to_node_id = strat_json["link"][1].as_usize().unwrap();
        let strat_name = strat_json["name"].as_str().unwrap().to_string();
        if strat_name.starts_with("Base (") {
            // Ignore internal strats for unlocking doors, etc.
            continue;
        }
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
            "Ignored".to_string()
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
        let resets_obstacles: Vec<String> = if strat_json.has_key("resetsObstacles") {
            strat_json["resetsObstacles"]
                .members()
                .map(|x| x.as_str().unwrap().to_string())
                .collect()
        } else {
            vec![]
        };
        let entrance_condition: Option<String> = if strat_json.has_key("entranceCondition") {
            Some(strat_json["entranceCondition"].pretty(2))
        } else {
            None
        };
        let exit_condition: Option<String> = if strat_json.has_key("exitCondition") {
            Some(strat_json["exitCondition"].pretty(2))
        } else {
            None
        };

        let unlocks_doors: Option<String> = if strat_json.has_key("unlocksDoors") {
            let mut unlocks_strs: Vec<String> = vec![];
            for unlock_json in strat_json["unlocksDoors"].members() {
                let raw_str = unlock_json.dump();
                if raw_str.len() < 120 {
                    unlocks_strs.push(raw_str);
                } else {
                    unlocks_strs.push(unlock_json.pretty(2));
                }
            }
            Some(unlocks_strs.join("\n"))
        } else {
            None
        };

        let strat_name = strat_json["name"].as_str().unwrap().to_string();
        let strat = RoomStrat {
            room_id,
            room_name: room_name.clone(),
            area: full_area.clone(),
            strat_id,
            strat_name: strat_name.clone(),
            bypasses_door_shell: strat_json["bypassesDoorShell"].as_bool() == Some(true),
            from_node_id,
            from_node_name: node_name_map[&from_node_id].clone(),
            to_node_id,
            to_node_name: node_name_map[&to_node_id].clone(),
            note: game_data.parse_note(&strat_json["note"]).join(" "),
            entrance_condition,
            requires: make_requires(&strat_json["requires"]),
            unlocks_doors,
            exit_condition,
            clears_obstacles,
            resets_obstacles,
            difficulty_idx,
            difficulty_name,
        };
        room_strats.push(strat);
    }
    let difficulty_names: Vec<String> = presets.iter().map(|x| x.preset.name.clone()).collect();

    RoomTemplate {
        version_info: version_info.clone(),
        difficulty_names,
        room_id,
        room_name_url_encoded: urlencoding::encode(&room_name).into_owned(),
        room_name,
        area: full_area,
        room_diagram_path: room_diagram_listing[&room_id].clone(),
        nodes,
        strats: room_strats,
        strat_videos: &game_data.strat_videos,
        room_json: room_json.pretty(2),
        video_storage_url: video_storage_url.to_string(),
    }
}

fn make_strat_template<'a>(
    room: &RoomTemplate<'a>,
    strat: &RoomStrat,
    video_storage_url: &str,
    version_info: &VersionInfo,
    game_data: &'a GameData,
) -> StratTemplate<'a> {
    StratTemplate {
        version_info: version_info.clone(),
        room_id: room.room_id,
        room_name: room.room_name.clone(),
        room_name_url_encoded: room.room_name_url_encoded.clone(),
        room_diagram_path: room.room_diagram_path.clone(),
        strat_name: strat.strat_name.clone(),
        strat: strat.clone(),
        strat_videos: &game_data.strat_videos,
        video_storage_url: video_storage_url.to_string(),
    }
}

impl LogicData {
    pub fn new(
        game_data: &GameData,
        presets: &[PresetData],
        version_info: &VersionInfo,
        video_storage_url: &str,
    ) -> LogicData {
        let mut out = LogicData::default();
        let room_diagram_listing = list_room_diagram_files();
        let mut room_templates: Vec<RoomTemplate> = vec![];
        let mut difficulty_configs: Vec<DifficultyConfig> = presets
            .iter()
            .map(|p| get_difficulty_config(p, game_data))
            .collect();

        // Remove the "Ignored" difficulty tier: everything above Beyond will be labeled as "Ignored" already.
        difficulty_configs.pop();

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
            "Wrecked Ship",
            "Outer Maridia",
            "Pink Inner Maridia",
            "Yellow Inner Maridia",
            "Green Inner Maridia",
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

            let mut notable_strats = vec![false; game_data.notable_isv.keys.len()];
            for strat_name in &difficulty.notables {
                notable_strats[game_data.notable_isv.index_by_key[strat_name]] = true;
            }

            let global = GlobalState {
                tech,
                notables: notable_strats,
                inventory: Inventory {
                    items: items,
                    max_energy: 1499,
                    max_reserves: 400,
                    max_missiles: 230,
                    max_supers: 50,
                    max_power_bombs: 50,
                },
                flags: vec![true; game_data.flag_isv.keys.len()],
                doors_unlocked: vec![],
                weapon_mask: weapon_mask,
            };

            global_states.push(global);
        }

        let mut links_by_ids: HashMap<(RoomId, NodeId, NodeId, String), Vec<Link>> = HashMap::new();
        for link in game_data.all_links() {
            let VertexKey {
                room_id: link_room_id,
                node_id: link_from_node_id,
                ..
            } = game_data.vertex_isv.keys[link.from_vertex_id];
            let VertexKey {
                node_id: link_to_node_id,
                ..
            } = game_data.vertex_isv.keys[link.to_vertex_id];
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
            let room_id = room_json["id"].as_usize().unwrap();
            let template = make_room_template(
                room_json,
                &room_diagram_listing,
                &game_data,
                presets,
                &difficulty_configs,
                &global_states,
                &links_by_ids,
                video_storage_url,
                version_info,
            );
            let html = template.clone().render().unwrap();
            out.room_html.insert(room_id, html);
            room_templates.push(template.clone());

            for strat in &template.strats {
                let strat_template = make_strat_template(
                    &template,
                    &strat,
                    video_storage_url,
                    version_info,
                    game_data,
                );
                let strat_html = strat_template.render().unwrap();
                out.strat_html.insert(
                    (
                        room_id,
                        strat.from_node_id,
                        strat.to_node_id,
                        strat.strat_id,
                    ),
                    strat_html,
                );
            }
        }
        room_templates.sort_by_key(|x| (x.area.clone(), x.room_name.clone()));

        let tech_templates = make_tech_templates(
            game_data,
            &room_templates,
            presets,
            &global_states,
            &area_order,
            video_storage_url,
            version_info,
        );
        for template in &tech_templates {
            let html = template.clone().render().unwrap();
            let strat_count = template
                .strats
                .iter()
                .filter(|x| x.difficulty_idx <= template.tech_difficulty_idx)
                .count();
            // if strat_count == 0 {
            //     warn!("Tech {} ({}) has no strats in its assigned difficulty {}",
            //         template.tech_id, template.tech_name, template.tech_difficulty_name);
            // }
            out.tech_strat_counts.insert(template.tech_id, strat_count);
            out.tech_html.insert(template.tech_id, html);
        }

        let notable_templates = make_notable_templates(
            game_data,
            &room_templates,
            presets,
            &global_states,
            &area_order,
            video_storage_url,
            version_info,
        );
        for template in &notable_templates {
            let html = template.clone().render().unwrap();
            let strat_count = template
                .strats
                .iter()
                .filter(|x| x.difficulty_idx <= template.notable_difficulty_idx)
                .count();
            if strat_count == 0 {
                warn!(
                    "Notable strat ({}, {}) {}: {} has no strats in its difficulty: {}",
                    template.room_id,
                    template.notable_id,
                    template.room_name,
                    template.notable_name,
                    template.notable_difficulty_name
                );
            }
            out.notable_strat_counts
                .insert((template.room_id, template.notable_id), strat_count);
            out.notable_html
                .insert((template.room_id, template.notable_id), html);
        }

        let index_template = LogicIndexTemplate {
            version_info: version_info.clone(),
            rooms: &room_templates,
            tech: &tech_templates,
            _notables: &notable_templates,
            area_order: &area_order,
            tech_difficulties: presets.iter().map(|x| x.preset.name.clone()).collect(),
        };
        out.index_html = index_template.render().unwrap();
        out
    }
}
