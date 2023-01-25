use std::path::Path;


use actix_easy_multipart::MultipartForm;
use actix_easy_multipart::tempfile::Tempfile;
use actix_easy_multipart::text::Text;
use actix_web::{middleware::Logger};

use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use hashbrown::{HashMap, HashSet};
use maprando::game_data::GameData;
use sailfish::TemplateOnce;
use serde_derive::{Deserialize, Serialize};

const VERSION: usize = 29;

#[derive(Serialize, Deserialize, Clone)]
struct Preset {
    name: String,
    shinespark_tiles: usize,
    resource_multiplier: f32,
    escape_timer_multiplier: f32,
    tech: Vec<String>,
}

struct PresetData {
    preset: Preset,
    tech_setting: Vec<(String, bool)>,
}

struct AppData {
    game_data: GameData,
    preset_data: Vec<PresetData>,
}

#[derive(TemplateOnce)]
#[template(path = "home/main.stpl")]
struct HomeTemplate<'a> {
    version: usize,
    item_placement_strategies: Vec<&'static str>,
    preset_data: &'a [PresetData],
    tech_description: &'a HashMap<String, String>,
}

#[get("/")]
async fn home(app_data: web::Data<AppData>) -> impl Responder {
    let home_template = HomeTemplate {
        version: VERSION,
        item_placement_strategies: vec!["Open", "Semiclosed", "Closed"],
        preset_data: &app_data.preset_data,
        tech_description: &app_data.game_data.tech_description,
    };
    HttpResponse::Ok().body(home_template.render_once().unwrap())
}

#[derive(MultipartForm)]
struct RandomizeRequest {
    rom: Tempfile,
    item_placement_strategy: Text<String>,
    preset: Text<String>,
    shinespark_tiles: Text<usize>,
    resource_multiplier: Text<f32>,
    escape_timer_multiplier: Text<f32>,
    save_animals: Text<String>,
    tech_json: Text<String>,
}

#[post("/randomize")]
async fn randomize(req: MultipartForm<RandomizeRequest>, app_data: web::Data<AppData>) -> impl Responder {
    println!("{:?}", req.item_placement_strategy);
    println!("{:?}", req.tech_json);
    HttpResponse::Ok().body("hello")
}

fn init_presets(presets: Vec<Preset>, game_data: &GameData) -> Vec<PresetData> {
    let mut out: Vec<PresetData> = Vec::new();
    let mut cumulative_tech: HashSet<String> = HashSet::new();

    // Tech which is currently not used by any strat in logic, so we avoid showing on the website:
    let ignored_tech: HashSet<String> = ["canWallIceClip", "canGrappleClip", "canUseSpeedEchoes"]
        .iter()
        .map(|x| x.to_string())
        .collect();
    for tech in &ignored_tech {
        if !game_data.tech_isv.index_by_key.contains_key(tech) {
            panic!("Unrecognized ignored tech \"{tech}\"");
        }
    }

    let visible_tech: Vec<String> = game_data
        .tech_isv
        .keys
        .iter()
        .filter(|&x| !ignored_tech.contains(x))
        .cloned()
        .collect();
    let visible_tech_set: HashSet<String> = visible_tech.iter().cloned().collect();
    
    for preset in presets {
        for tech in &preset.tech {
            if cumulative_tech.contains(tech) {
                panic!("Tech \"{tech}\" appears in presets more than once.");
            }
            if !visible_tech_set.contains(tech) {
                panic!("Unrecognized tech \"{tech}\" appears in preset {}", preset.name);
            }
            cumulative_tech.insert(tech.clone());
        }
        let mut tech_setting: Vec<(String, bool)> = Vec::new();
        for tech in &visible_tech {
            tech_setting.push((tech.clone(), cumulative_tech.contains(tech)));
        }
        out.push(PresetData {
            preset: preset,
            tech_setting: tech_setting,
        });
    }
    for tech in &visible_tech_set {
        if !cumulative_tech.contains(tech) {
            panic!("Tech \"{tech}\" not found in any preset.");
        }
    }
    out
}

#[actix_web::main]
async fn main() {
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let game_data = GameData::load(sm_json_data_path, room_geometry_path).unwrap();
    let presets: Vec<Preset> =
        serde_json::from_str(&std::fs::read_to_string(&"data/presets.json").unwrap()).unwrap();

    let preset_data = init_presets(presets, &game_data);
    let app_data = web::Data::new(AppData {
        game_data,
        preset_data,
    });

    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    HttpServer::new(move || {
        App::new()
            .app_data(app_data.clone())
            .wrap(Logger::default())
            .service(home)
            .service(randomize)
            .service(actix_files::Files::new("/static", "static"))
    })
    .bind("localhost:8080")
    .unwrap()
    .run()
    .await
    .unwrap();
}
