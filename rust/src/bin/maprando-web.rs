use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};

use actix_easy_multipart::bytes::Bytes;
use actix_easy_multipart::text::Text;
use actix_easy_multipart::{MultipartForm, MultipartFormConfig};
use actix_web::http::header::{ContentDisposition, DispositionParam, DispositionType};
use actix_web::middleware::Logger;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use anyhow::{Context, Result};
use hashbrown::{HashMap, HashSet};
use log::info;
use maprando::game_data::{GameData, Map};
use maprando::patch::{make_rom, Rom};
use maprando::randomize::{DifficultyConfig, Randomization, Randomizer};
use maprando::spoiler_map;
use rand::{RngCore, SeedableRng};
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

struct MapRepository {
    base_path: PathBuf,
    filenames: Vec<String>,
}

struct AppData {
    game_data: GameData,
    preset_data: Vec<PresetData>,
    map_repository: MapRepository,
}

impl MapRepository {
    fn new(base_path: &Path) -> Result<Self> {
        let mut filenames: Vec<String> = Vec::new();
        for path in std::fs::read_dir(base_path)? {
            filenames.push(path?.file_name().into_string().unwrap());
        }
        info!("{} maps available", filenames.len());
        Ok(MapRepository {
            base_path: base_path.to_owned(),
            filenames,
        })
    }

    fn get_map(&self, seed: usize) -> Result<Map> {
        let idx = seed % self.filenames.len();
        let path = self.base_path.join(&self.filenames[idx]);
        let map_string = std::fs::read_to_string(&path)
            .with_context(|| format!("Unable to read map file at {}", path.display()))?;
        info!("Map: {}", path.display());
        let map: Map = serde_json::from_str(&map_string)
            .with_context(|| format!("Unable to parse map file at {}", path.display()))?;
        Ok(map)
    }
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
    rom: Bytes,
    item_placement_strategy: Text<String>,
    preset: Option<Text<String>>,
    shinespark_tiles: Text<usize>,
    resource_multiplier: Text<f32>,
    escape_timer_multiplier: Text<f32>,
    save_animals: Text<String>,
    tech_json: Text<String>,
    random_seed: Text<usize>,
}

#[derive(Serialize, Deserialize)]
struct Config {
    version: usize,
    random_seed: usize,
    map_seed: usize,
    item_placement_seed: usize,
    preset: Option<String>,
    difficulty: DifficultyConfig,
}

fn get_difficulty_hash(difficulty: &DifficultyConfig) -> usize {
    let difficulty_str = serde_json::to_string(&difficulty).unwrap();
    let mut state = hashers::fx_hash::FxHasher::default();
    difficulty_str.hash(&mut state);
    state.finish() as usize
}

#[post("/randomize")]
async fn randomize(
    req: MultipartForm<RandomizeRequest>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let rom = Rom {
        data: req.rom.data.to_vec(),
    };
    if rom.data.len() < 3145728 || rom.data.len() > 8388608 {
        return HttpResponse::BadRequest().body("Invalid input ROM");
    }
    let tech_json: serde_json::Value = serde_json::from_str(&req.tech_json).unwrap();
    let mut tech_vec: Vec<String> = Vec::new();
    for (tech, is_enabled) in tech_json.as_object().unwrap().iter() {
        if is_enabled.as_bool().unwrap() {
            tech_vec.push(tech.to_string());
        }
    }
    let difficulty = DifficultyConfig {
        tech: tech_vec,
        shine_charge_tiles: req.shinespark_tiles.0 as i32,
        item_placement_strategy: match req.item_placement_strategy.0.as_str() {
            "Open" => maprando::randomize::ItemPlacementStrategy::Open,
            "Semiclosed" => maprando::randomize::ItemPlacementStrategy::Semiclosed,
            "Closed" => maprando::randomize::ItemPlacementStrategy::Closed,
            _ => panic!(
                "Unrecognized item placement strategy {}",
                req.item_placement_strategy.0.as_str()
            ),
        },
        resource_multiplier: req.resource_multiplier.0,
        escape_timer_multiplier: req.escape_timer_multiplier.0,
        save_animals: req.save_animals.0 == "On",
        debug_options: None,
    };
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&req.random_seed.to_le_bytes());
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);
    let max_attempts = 100;
    let mut attempt_num = 0;
    let randomization: Randomization;
    let difficulty_hash = get_difficulty_hash(&difficulty);
    let base_filename: String = format!(
        "smmr-v{}-{}-{}",
        VERSION, req.random_seed.0, difficulty_hash
    );
    info!("Starting {base_filename}");
    let mut map_seed: usize;
    let mut item_placement_seed: usize;
    loop {
        attempt_num += 1;
        if attempt_num > max_attempts {
            return HttpResponse::InternalServerError()
                .body("Failed too many randomization attempts");
        }
        map_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        let map = app_data.map_repository.get_map(map_seed).unwrap();
        item_placement_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        info!("Map seed={map_seed}, item placement seed={item_placement_seed}");
        let randomizer = Randomizer::new(&map, &difficulty, &app_data.game_data);
        randomization = match randomizer.randomize(item_placement_seed) {
            Some(r) => r,
            None => continue,
        };
        break;
    }

    let config = Config {
        version: VERSION,
        random_seed: req.random_seed.0,
        map_seed,
        item_placement_seed,
        preset: req.preset.as_ref().map(|x| x.0.clone()),
        difficulty,
    };

    let output_rom = make_rom(&rom, &randomization, &app_data.game_data).unwrap();
    let mut zip_vec: Vec<u8> = Vec::new();
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut zip_vec));

    let mut write_file = |name: &str, data: &[u8]| -> Result<()> {
        zip.start_file(name, zip::write::FileOptions::default())?;
        zip.write_all(data)?;
        Ok(())
    };

    // Write the ROM (to the ZIP file)
    write_file(&(base_filename.to_string() + ".sfc"), &output_rom.data).unwrap();

    // Write the config JSON
    let config_str = serde_json::to_vec_pretty(&config).unwrap();
    write_file(&(base_filename.to_string() + "-config.json"), &config_str).unwrap();

    // Write the spoiler log
    let spoiler_bytes = serde_json::to_vec_pretty(&randomization.spoiler_log).unwrap();
    write_file(&(base_filename.to_string() + "-spoiler.json"), &spoiler_bytes).unwrap();

    // Write the spoiler maps
    let spoiler_map_assigned =
        spoiler_map::get_spoiler_map(&output_rom, &randomization.map, &app_data.game_data, false).unwrap();
    write_file(&(base_filename.to_string() + "-map.png"), &spoiler_map_assigned).unwrap();
    let spoiler_map_vanilla =
        spoiler_map::get_spoiler_map(&output_rom, &randomization.map, &app_data.game_data, true).unwrap();
    write_file(&(base_filename.to_string() + "-map-vanilla.png"), &spoiler_map_vanilla).unwrap();

    zip.finish().unwrap();
    drop(zip);

    HttpResponse::Ok()
        .content_type("application/octet-stream")
        .insert_header(ContentDisposition {
            disposition: DispositionType::Attachment,
            parameters: vec![DispositionParam::Filename(base_filename.to_string() + ".zip")],
        })
        .body(zip_vec)
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
                panic!(
                    "Unrecognized tech \"{tech}\" appears in preset {}",
                    preset.name
                );
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
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

        let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let maps_path =
        Path::new("../maps/session-2022-06-03T17:19:29.727911.pkl-bk30-subarea-balance");
    let game_data = GameData::load(sm_json_data_path, room_geometry_path).unwrap();
    let presets: Vec<Preset> =
        serde_json::from_str(&std::fs::read_to_string(&"data/presets.json").unwrap()).unwrap();

    let preset_data = init_presets(presets, &game_data);
    let app_data = web::Data::new(AppData {
        game_data,
        preset_data,
        map_repository: MapRepository::new(maps_path).unwrap(),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(app_data.clone())
            .app_data(
                MultipartFormConfig::default()
                    .memory_limit(16_000_000)
                    .total_limit(16_000_000),
            )
            .wrap(Logger::default())
            .service(home)
            .service(randomize)
            .service(actix_files::Files::new("/static", "static"))
    })
    .bind("0.0.0.0:8080")
    .unwrap()
    .run()
    .await
    .unwrap();
}
