use crate::web::{AppData, PresetData, VersionInfo};
use actix_web::{HttpResponse, Responder, get, web};
use askama::Template;
use hashbrown::HashMap;
use maprando::settings::{ObjectiveGroup, get_objective_groups};
use maprando_game::{NotableId, RoomId, StartLocation, TechId};

#[derive(Template)]
#[template(path = "generate/main.html")]
struct GenerateTemplate<'a> {
    version_info: VersionInfo,
    progression_rates: Vec<&'static str>,
    item_placement_styles: Vec<&'static str>,
    objective_groups: Vec<ObjectiveGroup>,
    preset_data: &'a PresetData,
    full_presets_json: String,
    skill_presets_json: String,
    item_presets_json: String,
    qol_presets_json: String,
    objective_presets_json: String,
    item_priorities: Vec<String>,
    item_names_multiple: Vec<String>,
    item_names_single: Vec<String>,
    prioritizable_items: Vec<String>,
    tech_description: &'a HashMap<TechId, String>,
    tech_dependencies_str: &'a HashMap<TechId, String>,
    notable_description: &'a HashMap<(RoomId, NotableId), String>,
    tech_strat_counts: &'a HashMap<TechId, usize>,
    notable_strat_counts: &'a HashMap<(RoomId, NotableId), usize>,
    video_storage_url: &'a str,
    start_locations_by_area: Vec<(String, Vec<StartLocation>)>,
}

#[get("/generate")]
async fn generate(app_data: web::Data<AppData>) -> impl Responder {
    let item_names_multiple: Vec<String> = ["Missile", "ETank", "ReserveTank", "Super", "PowerBomb"]
        .into_iter()
        .map(|x| x.to_string())
        .collect();
        
    let item_names_single: Vec<String> = [
        "Charge",
        "Ice",
        "Wave",
        "Spazer",
        "Plasma",
        "XRayScope",
        "Morph",
        "Bombs",
        "Grapple",
        "HiJump",
        "SpeedBooster",
        "SpringBall",
        "SpaceJump",
        "ScrewAttack",
        "Varia",
        "Gravity",
    ]
    .into_iter()
    .map(|x| x.to_string())
    .collect();

    let prioritizable_items: Vec<String> = [
        "ETank",
        "ReserveTank",
        "Super",
        "PowerBomb",
        "Charge",
        "Ice",
        "Wave",
        "Spazer",
        "Plasma",
        "XRayScope",
        "Morph",
        "Bombs",
        "Grapple",
        "HiJump",
        "SpeedBooster",
        "SpringBall",
        "WallJump",
        "SpaceJump",
        "ScrewAttack",
        "Varia",
        "Gravity",
    ]
    .into_iter()
    .map(|x| x.to_string())
    .collect();

    let mut notable_description: HashMap<(RoomId, NotableId), String> = HashMap::new();
    for i in 0..app_data.game_data.notable_info.len() {
        let notable_data = &app_data.game_data.notable_info[i];
        notable_description.insert(
            (notable_data.room_id, notable_data.notable_id),
            notable_data.note.clone(),
        );
    }

    let mut tech_dependencies_strs: HashMap<TechId, String> = HashMap::new();
    for (tech_id, deps) in &app_data.game_data.tech_dependencies {
        let s: Vec<String> = deps
            .iter()
            .map(|t| app_data.game_data.tech_names[t].clone())
            .collect();
        tech_dependencies_strs.insert(*tech_id, s.join(", "));
    }

    let full_presets_json = serde_json::to_string(&app_data.preset_data.full_presets).unwrap();
    let skill_presets_json = serde_json::to_string(&app_data.preset_data.skill_presets).unwrap();
    let item_presets_json =
        serde_json::to_string(&app_data.preset_data.item_progression_presets).unwrap();
    let qol_presets_json =
        serde_json::to_string(&app_data.preset_data.quality_of_life_presets).unwrap();
    let objective_presets_json =
        serde_json::to_string(&app_data.preset_data.objective_presets).unwrap();

    let mut start_locations_by_area: Vec<(String, Vec<StartLocation>)> = app_data
        .game_data
        .area_order
        .iter()
        .map(|x| (x.clone(), vec![]))
        .collect();
    let area_order_idx: HashMap<String, usize> = app_data
        .game_data
        .area_order
        .iter()
        .enumerate()
        .map(|(i, x)| (x.clone(), i))
        .collect();
    for loc in &app_data.game_data.start_locations {
        let full_area = app_data.game_data.room_full_area[&loc.room_id].clone();
        let full_area_idx = area_order_idx[&full_area];
        start_locations_by_area[full_area_idx].1.push(loc.clone());
    }
    for s in &mut start_locations_by_area {
        s.1.sort_by_key(|x| {
            (
                app_data.game_data.room_json_map[&x.room_id]["name"]
                    .as_str()
                    .unwrap()
                    .to_string(),
                x.name.clone(),
            )
        });
    }

    let generate_template = GenerateTemplate {
        version_info: app_data.version_info.clone(),
        progression_rates: vec!["Fast", "Uniform", "Slow"],
        item_placement_styles: vec!["Neutral", "Forced", "Local"],
        objective_groups: get_objective_groups(),
        item_names_multiple,
        item_names_single,
        item_priorities: ["Early", "Default", "Late", "Never"]
            .iter()
            .map(|x| x.to_string())
            .collect(),
        prioritizable_items,
        preset_data: &app_data.preset_data,
        full_presets_json,
        skill_presets_json,
        item_presets_json,
        qol_presets_json,
        objective_presets_json,
        tech_description: &app_data.game_data.tech_description,
        tech_dependencies_str: &tech_dependencies_strs,
        notable_description: &notable_description,
        tech_strat_counts: &app_data.logic_data.tech_strat_counts,
        notable_strat_counts: &app_data.logic_data.notable_strat_counts,
        video_storage_url: &app_data.video_storage_url,
        start_locations_by_area,
    };
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(generate_template.render().unwrap())
}
