use crate::web::{AppData, PresetData, VersionInfo, HQ_VIDEO_URL_ROOT};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;
use hashbrown::{HashMap, HashSet};

#[derive(Template)]
#[template(path = "generate/main.html")]
struct GenerateTemplate<'a> {
    version_info: VersionInfo,
    progression_rates: Vec<&'static str>,
    item_placement_styles: Vec<&'static str>,
    objectives: Vec<&'static str>,
    preset_data: &'a [PresetData],
    item_priorities: Vec<String>,
    item_pool_multiple: Vec<String>,
    starting_items_multiple: Vec<String>,
    starting_items_single: Vec<String>,
    prioritizable_items: Vec<String>,
    tech_description: &'a HashMap<String, String>,
    tech_dependencies: &'a HashMap<String, Vec<String>>,
    tech_gif_listing: &'a HashSet<String>,
    notable_gif_listing: &'a HashSet<String>,
    tech_strat_counts: &'a HashMap<String, usize>,
    hq_video_url_root: &'a str,
}

#[get("/generate")]
async fn generate(app_data: web::Data<AppData>) -> impl Responder {
    let item_pool_multiple: Vec<String> = ["Missile", "ETank", "ReserveTank", "Super", "PowerBomb"]
        .into_iter()
        .map(|x| x.to_string())
        .collect();

    let starting_items_multiple: Vec<String> =
        ["Missile", "ETank", "ReserveTank", "Super", "PowerBomb"]
            .into_iter()
            .map(|x| x.to_string())
            .collect();

    let starting_items_single: Vec<String> = [
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
    let generate_template = GenerateTemplate {
        version_info: app_data.version_info.clone(),
        progression_rates: vec!["Fast", "Uniform", "Slow"],
        item_placement_styles: vec!["Neutral", "Forced"],
        objectives: vec![
            "None",
            "Bosses",
            "Minibosses",
            "Metroids",
            "Chozos",
            "Pirates",
            "Random",
        ],
        item_pool_multiple,
        starting_items_multiple,
        starting_items_single,
        item_priorities: vec!["Early", "Default", "Late"]
            .iter()
            .map(|x| x.to_string())
            .collect(),
        prioritizable_items,
        preset_data: &app_data.preset_data,
        tech_description: &app_data.game_data.tech_description,
        tech_dependencies: &app_data.game_data.tech_dependencies,
        tech_gif_listing: &app_data.tech_gif_listing,
        notable_gif_listing: &app_data.notable_gif_listing,
        tech_strat_counts: &app_data.logic_data.tech_strat_counts,
        hq_video_url_root: HQ_VIDEO_URL_ROOT,
    };
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(generate_template.render().unwrap())
}
