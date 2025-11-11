use anyhow::Result;
use askama::Template;
use glob::glob;
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use json::JsonValue;
use log::warn;
use maprando::{
    preset::PresetData,
    randomize::{EssentialSpoilerData, Randomization},
    settings::{Objective, RandomizerSettings},
    spoiler_map::{self, image::RgbaImage},
    traverse::{LockedDoorData, apply_requirement},
};
use maprando_game::{
    DoorOrientation, ExitCondition, GameData, Item, Link, MainEntranceCondition, Map, NodeId,
    NotableId, NotableIdx, Requirement, RoomId, SparkPosition, StartLocation, StratId, StratVideo,
    TECH_ID_CAN_ARTIFICIAL_MORPH, TECH_ID_CAN_BOMB_HORIZONTALLY, TECH_ID_CAN_CARRY_FLASH_SUIT,
    TECH_ID_CAN_DISABLE_EQUIPMENT, TECH_ID_CAN_ENEMY_STUCK_MOONFALL, TECH_ID_CAN_ENTER_G_MODE,
    TECH_ID_CAN_ENTER_G_MODE_IMMOBILE, TECH_ID_CAN_ENTER_R_MODE, TECH_ID_CAN_EXTENDED_MOONDANCE,
    TECH_ID_CAN_GRAPPLE_JUMP, TECH_ID_CAN_GRAPPLE_TELEPORT, TECH_ID_CAN_HEAT_RUN,
    TECH_ID_CAN_HEATED_G_MODE, TECH_ID_CAN_HORIZONTAL_SHINESPARK, TECH_ID_CAN_MIDAIR_SHINESPARK,
    TECH_ID_CAN_MOCKBALL, TECH_ID_CAN_MOONDANCE, TECH_ID_CAN_MOONFALL, TECH_ID_CAN_PRECISE_GRAPPLE,
    TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK, TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK_FROM_WATER,
    TECH_ID_CAN_SAMUS_EATER_TELEPORT, TECH_ID_CAN_SHINECHARGE_MOVEMENT, TECH_ID_CAN_SHINESPARK,
    TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP, TECH_ID_CAN_SKIP_DOOR_LOCK, TECH_ID_CAN_SPEEDBALL,
    TECH_ID_CAN_SPRING_BALL_BOUNCE, TECH_ID_CAN_STATIONARY_SPIN_JUMP,
    TECH_ID_CAN_STUTTER_WATER_SHINECHARGE, TECH_ID_CAN_SUPER_SINK, TECH_ID_CAN_TEMPORARY_BLUE,
    TECH_ID_CAN_WALLJUMP, TechId, VertexAction, VertexKey,
};
use maprando_logic::{GlobalState, Inventory, LocalState};
use std::{io::Cursor, path::PathBuf};

use super::VersionInfo;

#[derive(Clone)]
struct EnemyDrop {
    enemy: String,
    count: usize,
}

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
    detail_note: String,
    dev_note: String,
    entrance_condition: Option<String>,
    requires: String, // new-line separated requirements
    exit_condition: Option<String>,
    clears_obstacles: Vec<String>,
    resets_obstacles: Vec<String>,
    collects_items: Vec<String>,
    sets_flags: Vec<String>,
    unlocks_doors: Option<String>,
    farm_cycle_drops: Vec<EnemyDrop>,
    difficulty_idx: usize,
    difficulty_name: String,
}

#[derive(Template, Clone)]
#[template(path = "logic/room.html")]
struct RoomTemplate<'a> {
    version_info: VersionInfo,
    preset_data: &'a PresetData,
    room_id: usize,
    room_name: String,
    twin_room_id: Option<usize>,
    twin_room_name: Option<String>,
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
    preset_data: &'a PresetData,
    tech_id: TechId,
    tech_name: String,
    tech_note: String,
    tech_detail_note: String,
    tech_dev_note: String,
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
    preset_data: &'a PresetData,
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
    room_polygons: &'a [RoomPolygon],
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
    pub vanilla_map_png: Vec<u8>, // PNG of vanilla map, to show on logic index page
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
                let segments: Vec<&str> = path_string.split(['_', '.']).collect();
                let subregion = segments[0];
                if subregion == "ceres" {
                    continue;
                }
                let room_id: usize = str::parse(segments[2]).unwrap();
                out.insert(room_id, path_string);
            }
            Err(e) => panic!("Failure reading room diagrams: {e:?}"),
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
        match key {
            "and" | "or" => {
                for x in value.members() {
                    extract_tech_rec(x, tech, game_data);
                }
            }
            "tech" => {
                extract_tech_rec(value, tech, game_data);
            }
            "shinespark" => {
                tech.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_SHINESPARK]);
            }
            "heatFrames" | "heatFramesWithEnergyDrops" => {
                tech.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_HEAT_RUN]);
            }
            "speedBall" => {
                tech.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPEEDBALL]);
            }
            "disableEquipment" => {
                tech.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_DISABLE_EQUIPMENT]);
            }
            _ => {}
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
    preset_data: &'a PresetData,
    _global: &GlobalState,
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
            let from_node_json = &game_data.node_json_map[&(room_id, from_node_id)];
            let to_node_json = &game_data.node_json_map[&(room_id, to_node_id)];
            let mut tech_set: HashSet<usize> = HashSet::new();

            // TODO: think about if we could automate extracting tech requirements, as this
            // code is error-prone and awkward to maintain, as it can easily fall out of sync
            // with the requirements actually used in the randomizer logic. At the very least,
            // it could probably be extracted from the results from `get_cross_room_reqs`` below:
            for req in strat_json["requires"].members() {
                extract_tech_rec(req, &mut tech_set, game_data);
            }
            if strat_json["bypassesDoorShell"].as_bool() == Some(true) {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_SKIP_DOOR_LOCK]);
            }
            if strat_json.has_key("gModeRegainMobility") {
                tech_set
                    .insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE_IMMOBILE]);
                if game_data.get_room_heated(room_json, from_node_id).unwrap() {
                    tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_HEATED_G_MODE]);
                }
            }
            let entrance_condition_techs = vec![
                ("comeInShinecharged", vec![TECH_ID_CAN_SHINECHARGE_MOVEMENT]),
                (
                    "comeInShinechargedJumping",
                    vec![TECH_ID_CAN_SHINECHARGE_MOVEMENT],
                ),
                ("comeInWithBombBoost", vec![TECH_ID_CAN_BOMB_HORIZONTALLY]),
                (
                    "comeInStutterShinecharging",
                    vec![TECH_ID_CAN_STUTTER_WATER_SHINECHARGE],
                ),
                (
                    "comeInWithDoorStuckSetup",
                    vec![TECH_ID_CAN_STATIONARY_SPIN_JUMP],
                ),
                ("comeInSpeedballing", vec![TECH_ID_CAN_SPEEDBALL]),
                ("comeInWithTemporaryBlue", vec![TECH_ID_CAN_TEMPORARY_BLUE]),
                ("comeInWithMockball", vec![TECH_ID_CAN_MOCKBALL]),
                (
                    "comeInWithSpringBallBounce",
                    vec![TECH_ID_CAN_SPRING_BALL_BOUNCE],
                ),
                (
                    "comeInWithBlueSpringBallBounce",
                    vec![TECH_ID_CAN_SPRING_BALL_BOUNCE],
                ),
                ("comeInWithStoredFallSpeed", vec![TECH_ID_CAN_MOONFALL]),
                ("comeInWithGMode", vec![TECH_ID_CAN_ENTER_G_MODE]),
                ("comeInWithRMode", vec![TECH_ID_CAN_ENTER_R_MODE]),
                ("comeInWithGrappleSwing", vec![TECH_ID_CAN_PRECISE_GRAPPLE]),
                ("comeInWithGrappleJump", vec![TECH_ID_CAN_GRAPPLE_JUMP]),
                (
                    "comeInWithGrappleTeleport",
                    vec![TECH_ID_CAN_GRAPPLE_TELEPORT],
                ),
                (
                    "comeInWithSamusEaterTeleport",
                    vec![TECH_ID_CAN_SAMUS_EATER_TELEPORT],
                ),
                ("comeInWithWallJumpBelow", vec![TECH_ID_CAN_WALLJUMP]),
                (
                    "comeInWithSidePlatform",
                    vec![TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                ),
                ("comeInWithSuperSink", vec![TECH_ID_CAN_SUPER_SINK]),
            ];

            for (entrance_name, tech_ids) in entrance_condition_techs {
                if strat_json["entranceCondition"].has_key(entrance_name) {
                    for t in tech_ids {
                        tech_set.insert(game_data.tech_isv.index_by_key[&t]);
                    }
                }
            }

            let speedbooster_entrance_conditions = vec![
                "comeInRunning",
                "comeInJumping",
                "comeInSpaceJumping",
                "comeInWithMockball",
            ];
            for entrance_name in speedbooster_entrance_conditions {
                if strat_json["entranceCondition"].has_key(entrance_name)
                    && strat_json["entranceCondition"][entrance_name]["speedBooster"].as_bool()
                        == Some(false)
                {
                    tech_set
                        .insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_DISABLE_EQUIPMENT]);
                }
            }

            if strat_json["entranceCondition"].has_key("comeInWithDoorStuckSetup")
                && from_node_json["doorOrientation"].as_str() == Some("right")
            {
                tech_set
                    .insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK]);
                tech_set.insert(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK_FROM_WATER],
                );
            }
            if strat_json["entranceCondition"].has_key("comeInWithSpark") {
                let door_orientation = from_node_json["doorOrientation"].as_str().unwrap();
                if door_orientation == "right" || door_orientation == "left" {
                    tech_set.insert(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_HORIZONTAL_SHINESPARK],
                    );
                    if strat_json["entranceCondition"]["comeInWithSpark"]["position"].as_str()
                        == Some("top")
                    {
                        tech_set.insert(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_MIDAIR_SHINESPARK],
                        );
                    }
                }
            }
            if strat_json["entranceCondition"].has_key("comeInWithSidePlatformJump") {
                for p in strat_json["entranceCondition"]["comeInWithSidePlatformJump"]["platforms"]
                    .members()
                {
                    if p.has_key("requires") {
                        for req in p["requires"].members() {
                            extract_tech_rec(req, &mut tech_set, game_data);
                        }
                    }
                }
            }

            if strat_json["entranceCondition"].has_key("comeInWithGMode") {
                if strat_json["entranceCondition"]["comeInWithGMode"]["morphed"].as_bool()
                    == Some(true)
                {
                    tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_ARTIFICIAL_MORPH]);
                }
                if game_data.get_room_heated(room_json, from_node_id).unwrap() {
                    tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_HEATED_G_MODE]);
                }
            }

            let exit_condition_techs = vec![
                ("leaveShinecharged", vec![TECH_ID_CAN_SHINECHARGE_MOVEMENT]),
                ("leaveWithTemporaryBlue", vec![TECH_ID_CAN_TEMPORARY_BLUE]),
                ("leaveWithMockball", vec![TECH_ID_CAN_MOCKBALL]),
                (
                    "leaveWithSpringBallBounce",
                    vec![TECH_ID_CAN_SPRING_BALL_BOUNCE],
                ),
                ("leaveWithGModeSetup", vec![TECH_ID_CAN_ENTER_G_MODE]),
                ("leaveWithGMode", vec![TECH_ID_CAN_ENTER_G_MODE]),
                (
                    "leaveWithGrappleTeleport",
                    vec![TECH_ID_CAN_GRAPPLE_TELEPORT],
                ),
                (
                    "leaveWithSamusEaterTeleport",
                    vec![TECH_ID_CAN_SAMUS_EATER_TELEPORT],
                ),
                (
                    "leaveWithSidePlatform",
                    vec![TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                ),
                ("leaveWithSuperSink", vec![TECH_ID_CAN_SUPER_SINK]),
            ];

            for (exit_name, tech_ids) in exit_condition_techs {
                if strat_json["exitCondition"].has_key(exit_name) {
                    for t in tech_ids {
                        tech_set.insert(game_data.tech_isv.index_by_key[&t]);
                    }
                }
            }

            if strat_json["exitCondition"].has_key("leaveWithSpark") {
                let door_orientation = to_node_json["doorOrientation"].as_str().unwrap();
                if door_orientation == "right" || door_orientation == "left" {
                    tech_set.insert(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_HORIZONTAL_SHINESPARK],
                    );
                    if strat_json["exitCondition"]["leaveWithSpark"]["position"].as_str()
                        == Some("top")
                    {
                        tech_set.insert(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_MIDAIR_SHINESPARK],
                        );
                    }
                }
            }
            if strat_json["exitCondition"].has_key("leaveWithGModeSetup")
                && game_data.get_room_heated(room_json, to_node_id).unwrap()
            {
                tech_set.insert(game_data.tech_isv.index_by_key[&TECH_ID_CAN_HEATED_G_MODE]);
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
        let tech_id = game_data.tech_isv.keys[tech_idx];
        let tech_json = &game_data.tech_json_map[&tech_id];
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
        let tech_data = &preset_data.tech_data_map[&tech_id];
        let difficulty_name = tech_data.difficulty.clone();
        let difficulty_idx = preset_data.difficulty_levels.index_by_key[&difficulty_name];

        for strat_ids in tech_ids {
            if room_strat_map.contains_key(strat_ids) {
                strats.push(room_strat_map[strat_ids].clone());
            }
        }
        strats.sort_by_key(|s| {
            (
                game_data
                    .area_order
                    .iter()
                    .position(|a| a == &s.area)
                    .unwrap(),
                s.room_name.clone(),
                s.from_node_id,
                s.to_node_id,
                s.strat_name.clone(),
            )
        });
        let template = TechTemplate {
            version_info: version_info.clone(),
            preset_data,
            tech_id,
            tech_name: tech_json["name"].as_str().unwrap().to_string(),
            tech_note: game_data.parse_note(&tech_json["note"]).join(" "),
            tech_detail_note: game_data.parse_note(&tech_json["detailNote"]).join(" "),
            tech_dev_note: game_data.parse_note(&tech_json["devNote"]).join(" "),
            tech_dependencies,
            tech_difficulty_idx: difficulty_idx,
            tech_difficulty_name: difficulty_name,
            strats,
            strat_videos: &game_data.strat_videos,
            tech_video_id: preset_data.tech_data_map[&tech_id].video_id,
            video_storage_url: video_storage_url.to_string(),
        };
        tech_templates.push(template);
    }
    tech_templates
}

fn make_notable_templates<'a>(
    game_data: &'a GameData,
    room_templates: &[RoomTemplate<'a>],
    preset_data: &'a PresetData,
    _global: &GlobalState,
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
        let room_id = game_data.notable_info[notable_idx].room_id;
        let notable_info = &game_data.notable_info[notable_idx];
        let notable_id = notable_info.notable_id;
        let notable_name = notable_info.name.clone();
        let notable_note = notable_info.note.clone();
        let notable_data = &preset_data.notable_data_map[&(room_id, notable_id)];
        let difficulty_name = &notable_data.difficulty;
        let mut strats: Vec<RoomStrat> = vec![];
        let difficulty_idx = preset_data.difficulty_levels.index_by_key[difficulty_name];

        for ids in ids_set {
            if room_strat_map.contains_key(ids) {
                strats.push(room_strat_map[ids].clone());
            }
        }
        strats.sort_by_key(|s| {
            (
                game_data
                    .area_order
                    .iter()
                    .position(|a| a == &s.area)
                    .unwrap(),
                s.room_name.clone(),
                s.from_node_id,
                s.to_node_id,
                s.strat_name.clone(),
            )
        });
        let template = NotableTemplate {
            version_info: version_info.clone(),
            preset_data,
            room_id,
            room_name: game_data.room_json_map[&room_id]["name"]
                .as_str()
                .unwrap()
                .to_string(),
            notable_id,
            notable_name,
            notable_note,
            notable_difficulty_idx: difficulty_idx,
            notable_difficulty_name: difficulty_name.to_string(),
            strats,
            strat_videos: &game_data.strat_videos,
            notable_video_id: preset_data.notable_data_map[&(room_id, notable_id)].video_id,
            video_storage_url: video_storage_url.to_string(),
        };
        notable_templates.push(template);
    }
    notable_templates
}

fn get_cross_room_reqs(link: &Link, game_data: &GameData) -> Requirement {
    let mut reqs: Vec<Requirement> = vec![];
    let from_vertex_key = &game_data.vertex_isv.keys[link.from_vertex_id];
    let to_vertex_key = &game_data.vertex_isv.keys[link.to_vertex_id];

    // TODO: handle gModeRegainMobility in some cleaner way:
    for (l, _) in game_data
        .node_gmode_regain_mobility
        .get(&(from_vertex_key.room_id, from_vertex_key.node_id))
        .unwrap_or(&vec![])
    {
        if l.strat_id.is_some() && l.strat_id == link.strat_id {
            reqs.push(Requirement::Tech(
                game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE_IMMOBILE],
            ));
            if game_data
                .get_room_heated(
                    &game_data.room_json_map[&from_vertex_key.room_id],
                    from_vertex_key.node_id,
                )
                .unwrap()
            {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_HEATED_G_MODE],
                ));
            }
        }
    }

    for action in &from_vertex_key.actions {
        if let VertexAction::Enter(entrance_condition) = action {
            match &entrance_condition.main {
                MainEntranceCondition::ComeInNormally { .. } => {}
                MainEntranceCondition::ComeInRunning { .. } => {}
                MainEntranceCondition::ComeInJumping { .. } => {}
                MainEntranceCondition::ComeInSpaceJumping { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                    ));
                }
                MainEntranceCondition::ComeInBlueSpaceJumping { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                    ));
                }
                MainEntranceCondition::ComeInGettingBlueSpeed { .. } => {}
                MainEntranceCondition::ComeInShinecharging { .. } => {}
                MainEntranceCondition::ComeInShinecharged { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SHINECHARGE_MOVEMENT],
                    ));
                }
                MainEntranceCondition::ComeInShinechargedJumping { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SHINECHARGE_MOVEMENT],
                    ));
                }
                MainEntranceCondition::ComeInWithSpark {
                    position,
                    door_orientation,
                } => {
                    if [DoorOrientation::Left, DoorOrientation::Right].contains(door_orientation) {
                        reqs.push(Requirement::Tech(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_HORIZONTAL_SHINESPARK],
                        ));
                        if position == &SparkPosition::Top {
                            reqs.push(Requirement::Tech(
                                game_data.tech_isv.index_by_key[&TECH_ID_CAN_MIDAIR_SHINESPARK],
                            ));
                        }
                    }
                }
                MainEntranceCondition::ComeInWithBombBoost {} => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_BOMB_HORIZONTALLY],
                    ));
                }
                MainEntranceCondition::ComeInStutterShinecharging { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_STUTTER_WATER_SHINECHARGE],
                    ));
                }
                MainEntranceCondition::ComeInStutterGettingBlueSpeed { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_STUTTER_WATER_SHINECHARGE],
                    ));
                }
                MainEntranceCondition::ComeInWithDoorStuckSetup {
                    door_orientation, ..
                } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_STATIONARY_SPIN_JUMP],
                    ));
                    if door_orientation == &DoorOrientation::Right {
                        reqs.push(Requirement::Tech(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK],
                        ));
                    }
                }
                MainEntranceCondition::ComeInSpeedballing { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPEEDBALL],
                    ));
                }
                MainEntranceCondition::ComeInWithTemporaryBlue { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_TEMPORARY_BLUE],
                    ));
                }
                MainEntranceCondition::ComeInWithMockball { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                    ));
                }
                MainEntranceCondition::ComeInWithSpringBallBounce { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                    ));
                }
                MainEntranceCondition::ComeInWithBlueSpringBallBounce { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                    ));
                }
                MainEntranceCondition::ComeInSpinning { .. } => {}
                MainEntranceCondition::ComeInBlueSpinning { .. } => {}
                MainEntranceCondition::ComeInWithStoredFallSpeed {
                    fall_speed_in_tiles,
                } => {
                    reqs.push(Requirement::Or(vec![
                        Requirement::Tech(game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOONDANCE]),
                        Requirement::Tech(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENEMY_STUCK_MOONFALL],
                        ),
                    ]));
                    if fall_speed_in_tiles == &2 {
                        reqs.push(Requirement::Tech(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_EXTENDED_MOONDANCE],
                        ));
                    }
                }
                MainEntranceCondition::ComeInWithRMode { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_R_MODE],
                    ));
                }
                MainEntranceCondition::ComeInWithGMode { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE],
                    ));
                }
                MainEntranceCondition::ComeInWithWallJumpBelow { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_WALLJUMP],
                    ));
                }
                MainEntranceCondition::ComeInWithSpaceJumpBelow { .. } => {}
                MainEntranceCondition::ComeInWithPlatformBelow { .. } => {}
                MainEntranceCondition::ComeInWithSidePlatform { platforms } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                    ));
                    for p in platforms {
                        reqs.push(p.requirement.clone());
                    }
                }
                MainEntranceCondition::ComeInWithGrappleSwing { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_PRECISE_GRAPPLE],
                    ));
                }
                MainEntranceCondition::ComeInWithGrappleJump { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_JUMP],
                    ));
                }
                MainEntranceCondition::ComeInWithGrappleTeleport { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_TELEPORT],
                    ));
                }
                MainEntranceCondition::ComeInWithSamusEaterTeleport { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SAMUS_EATER_TELEPORT],
                    ));
                }
                MainEntranceCondition::ComeInWithSuperSink { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SUPER_SINK],
                    ));
                }
            }
        }
    }
    for action in &to_vertex_key.actions {
        if let VertexAction::Exit(exit_condition) = action {
            match exit_condition {
                ExitCondition::LeaveNormally { .. } => {}
                ExitCondition::LeaveWithRunway { .. } => {}
                ExitCondition::LeaveShinecharged { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SHINECHARGE_MOVEMENT],
                    ));
                }
                ExitCondition::LeaveWithTemporaryBlue { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_TEMPORARY_BLUE],
                    ));
                }
                ExitCondition::LeaveWithSpark {
                    position,
                    door_orientation,
                } => {
                    if [DoorOrientation::Left, DoorOrientation::Right].contains(door_orientation) {
                        reqs.push(Requirement::Tech(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_HORIZONTAL_SHINESPARK],
                        ));
                        if position == &SparkPosition::Top {
                            reqs.push(Requirement::Tech(
                                game_data.tech_isv.index_by_key[&TECH_ID_CAN_MIDAIR_SHINESPARK],
                            ));
                        }
                    }
                }
                ExitCondition::LeaveSpinning { .. } => {}
                ExitCondition::LeaveWithMockball { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                    ));
                }
                ExitCondition::LeaveWithSpringBallBounce { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                    ));
                }
                ExitCondition::LeaveSpaceJumping { .. } => {}
                ExitCondition::LeaveWithStoredFallSpeed { .. } => {}
                ExitCondition::LeaveWithGModeSetup { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE],
                    ));
                }
                ExitCondition::LeaveWithGMode { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE],
                    ));
                }
                ExitCondition::LeaveWithDoorFrameBelow { .. } => {}
                ExitCondition::LeaveWithPlatformBelow { .. } => {}
                ExitCondition::LeaveWithSidePlatform { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                    ));
                }
                ExitCondition::LeaveWithGrappleSwing { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_PRECISE_GRAPPLE],
                    ));
                }
                ExitCondition::LeaveWithGrappleJump { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_JUMP],
                    ));
                }
                ExitCondition::LeaveWithGrappleTeleport { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_TELEPORT],
                    ));
                }
                ExitCondition::LeaveWithSamusEaterTeleport { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SAMUS_EATER_TELEPORT],
                    ));
                }
                ExitCondition::LeaveWithSuperSink { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SUPER_SINK],
                    ));
                }
            }
        }
    }
    Requirement::make_and(reqs)
}

fn strip_cross_room_reqs(req: Requirement) -> Requirement {
    match req {
        Requirement::And(subreqs) => {
            Requirement::make_and(subreqs.into_iter().map(strip_cross_room_reqs).collect())
        }
        Requirement::Or(subreqs) => {
            Requirement::make_or(subreqs.into_iter().map(strip_cross_room_reqs).collect())
        }
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
    preset_data: &PresetData,
    global: &GlobalState,
    links_by_ids: &HashMap<(RoomId, NodeId, NodeId, String), Vec<Link>>,
) -> usize {
    let locked_door_data = LockedDoorData {
        locked_doors: vec![],
        locked_door_node_map: HashMap::new(),
        locked_door_vertex_ids: vec![],
    };
    let door_map: HashMap<(RoomId, NodeId), (RoomId, NodeId)> = HashMap::new();
    for difficulty in preset_data.difficulty_tiers.iter().rev() {
        if difficulty.name == "Implicit" {
            // Skip the "Implicit" difficulty
            continue;
        }
        let difficulty_idx = preset_data.difficulty_levels.index_by_key[&difficulty.name];

        let flash_suit_obtainable =
            difficulty.tech[game_data.tech_isv.index_by_key[&TECH_ID_CAN_CARRY_FLASH_SUIT]];

        let local = LocalState {
            shinecharge_frames_remaining: 180 - difficulty.shinecharge_leniency_frames,
            flash_suit: flash_suit_obtainable,
            ..LocalState::full(false)
        };

        let key = (room_id, from_node_id, to_node_id, strat_name.clone());
        if !links_by_ids.contains_key(&key) {
            return preset_data.difficulty_levels.keys.len() - 1;
        }
        for link in &links_by_ids[&key] {
            let extra_req = get_cross_room_reqs(link, game_data);
            let main_req = strip_cross_room_reqs(link.requirement.clone());
            let combined_req = Requirement::make_and(vec![extra_req, main_req]);
            let new_local = apply_requirement(
                &combined_req,
                global,
                local,
                false,
                &preset_data.logic_page_preset,
                difficulty,
                game_data,
                &door_map,
                &locked_door_data,
                &[],
            );
            if new_local.is_some() {
                return difficulty_idx;
            }
        }
    }

    // Strat isn't logical on any settings, so it is Ignored:
    preset_data.difficulty_levels.keys.len() - 1
}

fn make_room_template<'a>(
    room_json: &JsonValue,
    room_diagram_listing: &HashMap<usize, String>,
    game_data: &'a GameData,
    preset_data: &'a PresetData,
    global: &GlobalState,
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
    let full_area = game_data.room_full_area[&room_id].clone();

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
            preset_data,
            global,
            links_by_ids,
        );
        let difficulty_name = preset_data.difficulty_levels.keys[difficulty_idx].clone();
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
        let collects_items: Vec<String> = if strat_json.has_key("collectsItems") {
            strat_json["collectsItems"]
                .members()
                .map(|x| x.as_usize().unwrap().to_string())
                .collect()
        } else {
            vec![]
        };
        let sets_flags: Vec<String> = if strat_json.has_key("setsFlags") {
            strat_json["setsFlags"]
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

        let farm_cycle_drops: Vec<EnemyDrop> = if strat_json.has_key("farmCycleDrops") {
            let mut drops: Vec<EnemyDrop> = vec![];
            for drop in strat_json["farmCycleDrops"].members() {
                drops.push(EnemyDrop {
                    enemy: drop["enemy"].as_str().unwrap().to_string(),
                    count: drop["count"].as_usize().unwrap(),
                });
            }
            drops
        } else {
            vec![]
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
            detail_note: game_data.parse_note(&strat_json["detailNote"]).join(" "),
            dev_note: game_data.parse_note(&strat_json["devNote"]).join(" "),
            entrance_condition,
            requires: make_requires(&strat_json["requires"]),
            unlocks_doors,
            exit_condition,
            clears_obstacles,
            resets_obstacles,
            collects_items,
            sets_flags,
            farm_cycle_drops,
            difficulty_idx,
            difficulty_name,
        };
        room_strats.push(strat);
    }

    let twin_room_id = match room_id {
        220 => Some(322),
        322 => Some(220),
        32 => Some(313),
        313 => Some(32),
        _ => None,
    };
    let twin_room_name = twin_room_id.map(|x| {
        game_data.room_json_map[&x]["name"]
            .as_str()
            .unwrap()
            .to_string()
    });
    RoomTemplate {
        version_info: version_info.clone(),
        preset_data,
        room_id,
        room_name_url_encoded: urlencoding::encode(&room_name).into_owned(),
        room_name,
        twin_room_id,
        twin_room_name,
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

fn get_vanilla_randomization(vanilla_map: &Map) -> Randomization {
    // For now we're not using the actual vanilla item placement,
    // since we're only using this to draw the map.
    Randomization {
        objectives: vec![
            Objective::Kraid,
            Objective::Phantoon,
            Objective::Draygon,
            Objective::Ridley,
        ],
        save_animals: maprando::settings::SaveAnimals::Optional,
        map: vanilla_map.clone(),
        toilet_intersections: vec![],
        locked_doors: vec![],
        item_placement: vec![Item::Missile; 100],
        start_location: StartLocation::default(),
        escape_time_seconds: 0.0,
        essential_spoiler_data: EssentialSpoilerData {
            item_spoiler_info: vec![],
        },
        seed: 0,
        display_seed: 0,
        seed_name: "".to_string(),
    }
}

struct RoomPolygon {
    room_id: usize,
    room_name: String,
    svg_path: String,
}

#[derive(Default)]
struct VanillaMapData {
    png: Vec<u8>,
    room_polygons: Vec<RoomPolygon>,
}

type Point = (i32, i32);

#[derive(Default)]
struct PolygonBuffer {
    edges: HashMap<(Point, Point), i32>,
}

impl PolygonBuffer {
    pub fn add_edge(&mut self, mut p1: Point, mut p2: Point, mut wt: i32) {
        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
            wt = -wt;
        }
        *self.edges.entry((p1, p2)).or_default() += wt;
        if self.edges[&(p1, p2)] == 0 {
            self.edges.remove(&(p1, p2));
        }
    }

    pub fn add_poly(&mut self, points: Vec<Point>) {
        for (&p1, &p2) in points.iter().circular_tuple_windows() {
            self.add_edge(p1, p2, 1);
        }
    }

    pub fn get_edge(&self, mut p1: Point, mut p2: Point) -> i32 {
        let mut sgn = 1;
        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
            sgn = -1;
        }
        match self.edges.get(&(p1, p2)) {
            Some(wt) => sgn * wt,
            None => 0,
        }
    }

    fn extract_path(&mut self) -> Option<Vec<Point>> {
        let e = self.edges.keys().next()?;
        let p0 = e.0;
        let mut p = p0;
        let mut out = vec![p0];
        'outer: loop {
            for p1 in self.neighbors(p) {
                let wt = self.get_edge(p, p1);
                if wt > 0 {
                    self.add_edge(p, p1, -1);
                    out.push(p1);
                    p = p1;
                    if p == p0 {
                        return Some(out);
                    }
                    continue 'outer;
                }
            }
            panic!("failed to continue path: {out:?}");
        }
    }

    fn extract_all_paths(&mut self) -> Vec<Vec<Point>> {
        let mut out = vec![];
        while let Some(path) = self.extract_path() {
            out.push(path);
        }
        out
    }

    fn neighbors(&self, p: Point) -> Vec<Point> {
        let (x, y) = p;
        vec![(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    }
}

fn get_vanilla_map_data(
    vanilla_map: &Map,
    game_data: &GameData,
    settings: &RandomizerSettings,
) -> Result<VanillaMapData> {
    let randomization = get_vanilla_randomization(vanilla_map);
    let mut settings = settings.clone();
    settings.map_layout = "Vanilla".to_string();
    let (img, _) = spoiler_map::get_spoiler_images(&randomization, game_data, &settings, true)?;

    let mut cropped_img = RgbaImage::new(68 * 8 + 1, 59 * 8 + 1);
    let offset_x = 3 * 8;
    let offset_y = 4 * 8 - 1;
    for y in 0..cropped_img.height() {
        for x in 0..cropped_img.width() {
            let p = img.get_pixel(x + offset_x, y + offset_y);
            cropped_img.put_pixel(x, y, *p);
        }
    }

    let mut png: Vec<u8> = Vec::new();
    cropped_img.write_to(
        &mut Cursor::new(&mut png),
        spoiler_map::image::ImageOutputFormat::Png,
    )?;

    let mut room_polygons: Vec<RoomPolygon> = vec![];
    for (room_idx, room) in game_data.room_geometry.iter().enumerate() {
        let room_id = game_data.room_id_by_ptr[&room.rom_address];

        let mut poly_buf = PolygonBuffer::default();
        for y in 0..room.map.len() {
            for x in 0..room.map[0].len() {
                if room.map[y][x] == 0 {
                    continue;
                }
                if room.room_id == 224 && y == 0 {
                    // Skip the top tile of Tourian First Room since it overlaps with Statues Room.
                    continue;
                }
                let x = x as i32;
                let y = y as i32;
                poly_buf.add_poly(vec![(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]);
            }
        }

        let room_json = &game_data.room_json_map[&room_id];
        let room_name = room_json["name"].as_str().unwrap().to_string();
        let room_x = vanilla_map.rooms[room_idx].0 as i32;
        let room_y = vanilla_map.rooms[room_idx].1 as i32;
        let mut svg_path: String = String::new();
        for path in poly_buf.extract_all_paths() {
            svg_path.push_str("M ");
            svg_path.push_str(
                &path
                    .iter()
                    .map(|(x, y)| {
                        let x1 = (room_x + x + 1) * 8 - offset_x as i32;
                        let y1 = (room_y + y + 1) * 8 - offset_y as i32;
                        format!("{x1},{y1}")
                    })
                    .join(" L "),
            );
            svg_path.push_str("Z ");
        }

        room_polygons.push(RoomPolygon {
            room_id,
            room_name,
            svg_path,
        });
    }

    Ok(VanillaMapData { png, room_polygons })
}

impl LogicData {
    pub fn new(
        game_data: &GameData,
        preset_data: &PresetData,
        version_info: &VersionInfo,
        video_storage_url: &str,
        vanilla_map: &Map,
    ) -> Result<LogicData> {
        let mut out = LogicData::default();
        let vanilla_map_data =
            get_vanilla_map_data(vanilla_map, game_data, &preset_data.default_preset)?;
        out.vanilla_map_png = vanilla_map_data.png;
        let room_diagram_listing = list_room_diagram_files();
        let mut room_templates: Vec<RoomTemplate> = vec![];

        let items = vec![true; game_data.item_isv.keys.len()];
        let tech = vec![true; game_data.tech_isv.keys.len()];
        let weapon_mask = game_data.get_weapon_mask(&items, &tech);
        let inventory = Inventory {
            items,
            max_energy: 1499,
            max_reserves: 400,
            max_missiles: 230,
            max_supers: 50,
            max_power_bombs: 50,
            collectible_missile_packs: 0,
            collectible_super_packs: 0,
            collectible_power_bomb_packs: 0,
            collectible_reserve_tanks: 0,
        };
        let global = GlobalState {
            pool_inventory: inventory.clone(),
            inventory,
            flags: vec![true; game_data.flag_isv.keys.len()],
            doors_unlocked: vec![],
            weapon_mask,
        };

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
                game_data,
                preset_data,
                &global,
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
                    strat,
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
            preset_data,
            &global,
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
            preset_data,
            &global,
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
            area_order: &game_data.area_order,
            tech_difficulties: preset_data.difficulty_levels.keys.clone(),
            room_polygons: &vanilla_map_data.room_polygons,
        };
        out.index_html = index_template.render().unwrap();
        Ok(out)
    }
}
