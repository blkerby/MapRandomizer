// TODO: consider removing this later. It's not a bad lint but I don't want to deal with it now.
#![allow(clippy::too_many_arguments)]

mod randomize_helpers;
mod logic;
mod logic_helper;
mod seed;
mod web;

use actix_easy_multipart::{MultipartForm, MultipartFormConfig, text::Text};
use actix_files::NamedFile;
use actix_web::{
    App, HttpRequest, HttpResponse, HttpServer, Responder, get,
    middleware::{Compress, Logger},
    post,
};
use askama::Template;
use clap::Parser;
use hashbrown::HashMap;
use log::{error, info};
use rand::{RngCore, SeedableRng};
use serde_derive::{Deserialize, Serialize};
use serde_variant::to_variant_name;
use std::time::{Duration, SystemTime};
use std::{path::Path, time::Instant};

use crate::{
    logic_helper::LogicData,
    web::{AppData, VERSION, VersionInfo},
};
use randomize_helpers::*;
use maprando::settings::{ObjectiveGroup, get_objective_groups};
use maprando::{
    customize::{mosaic::MosaicTheme, samus_sprite::SamusSpriteCategory},
    difficulty::{get_full_global, get_link_difficulty_length},
    map_repository::MapRepository,
    preset::PresetData,
    randomize::{
        DifficultyConfig, Randomization, Randomizer, assign_map_areas, filter_links,
        get_difficulty_tiers, get_objectives, randomize_doors,
    },
    seed_repository::SeedRepository,
    settings::{RandomizerSettings, StartLocationMode, try_upgrade_settings},
    spoiler_log::SpoilerLog,
};
use maprando_game::GameData;
use maprando_game::{LinksDataGroup, Map};
use maprando_game::{NotableId, RoomId, StartLocation, TechId};

const VISUALIZER_PATH: &str = "../visualizer/";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    seed_repository_url: String,
    #[arg(long, default_value = "https://map-rando-videos.b-cdn.net")]
    video_storage_url: String,
    #[arg(long)]
    video_storage_path: Option<String>,
    #[arg(long, action)]
    debug: bool,
    #[arg(long, action)]
    static_visualizer: bool,
    #[arg(long, action)]
    dev: bool,
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

fn load_visualizer_files() -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = vec![];
    for entry_res in std::fs::read_dir(VISUALIZER_PATH).unwrap() {
        let entry = entry_res.unwrap();
        let name = entry.file_name().to_str().unwrap().to_string();
        let data = std::fs::read(entry.path()).unwrap();
        files.push((name, data));
    }
    files
}

fn build_app_data() -> AppData {
    let start_time = Instant::now();
    let args = Args::parse();
    let etank_colors_path = Path::new("data/etank_colors.json");
    let vanilla_map_path = Path::new("../maps/vanilla");
    let small_maps_path = Path::new("../maps/v119-small-avro");
    let standard_maps_path = Path::new("../maps/v119-standard-avro");
    let wild_maps_path = Path::new("../maps/v119-wild-avro");
    let samus_sprites_path = Path::new("../MapRandoSprites/samus_sprites/manifest.json");
    let tech_path = Path::new("data/tech_data.json");
    let notable_path = Path::new("data/notable_data.json");
    let presets_path = Path::new("data/presets");
    let mosaic_themes = vec![
        ("OuterCrateria", "Outer Crateria"),
        ("InnerCrateria", "Inner Crateria"),
        ("BlueBrinstar", "Blue Brinstar"),
        ("GreenBrinstar", "Green Brinstar"),
        ("PinkBrinstar", "Pink Brinstar"),
        ("RedBrinstar", "Red Brinstar"),
        ("WarehouseBrinstar", "Warehouse Brinstar"),
        ("UpperNorfair", "Upper Norfair"),
        ("LowerNorfair", "Lower Norfair"),
        ("WreckedShip", "Wrecked Ship"),
        ("WestMaridia", "West Maridia"),
        ("YellowMaridia", "Yellow Maridia"),
        ("Bedrock", "Bedrock"),
        ("MechaTourian", "Mecha Tourian"),
        ("MetroidHabitat", "Metroid Habitat"),
    ]
    .into_iter()
    .map(|(x, y)| MosaicTheme {
        name: x.to_string(),
        display_name: y.to_string(),
    })
    .collect();

    let mut game_data = GameData::load(Path::new(".")).unwrap();

    info!("Loading logic preset data");
    let etank_colors: Vec<Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string(etank_colors_path).unwrap()).unwrap();
    let version_info = VersionInfo {
        version: VERSION,
        dev: args.dev,
        commit_hash: std::env::var("GIT_COMMIT_HASH").unwrap_or_default(),
    };
    let video_storage_url = if args.video_storage_path.is_some() {
        "/videos".to_string()
    } else {
        args.video_storage_url.clone()
    };

    let preset_data = PresetData::load(tech_path, notable_path, presets_path, &game_data).unwrap();
    let global = get_full_global(&game_data);
    game_data.make_links_data(&|link, game_data| {
        get_link_difficulty_length(link, game_data, &preset_data, &global)
    });

    let map_repositories: HashMap<String, MapRepository> = vec![
        (
            "Vanilla".to_string(),
            MapRepository::new("Vanilla", vanilla_map_path).unwrap(),
        ),
        (
            "Small".to_string(),
            MapRepository::new("Small", small_maps_path).unwrap(),
        ),
        (
            "Standard".to_string(),
            MapRepository::new("Standard", standard_maps_path).unwrap(),
        ),
        (
            "Wild".to_string(),
            MapRepository::new("Wild", wild_maps_path).unwrap(),
        ),
    ]
    .into_iter()
    .collect();
    let vanilla_map = map_repositories["Vanilla"]
        .get_map_batch(0, &game_data)
        .unwrap()[0]
        .clone();

    let logic_data = LogicData::new(
        &game_data,
        &preset_data,
        &version_info,
        &video_storage_url,
        &vanilla_map,
    )
    .unwrap();
    let samus_sprite_categories: Vec<SamusSpriteCategory> =
        serde_json::from_str(&std::fs::read_to_string(samus_sprites_path).unwrap()).unwrap();

    let app_data = AppData {
        game_data,
        preset_data,
        map_repositories,
        seed_repository: SeedRepository::new(&args.seed_repository_url).unwrap(),
        visualizer_files: load_visualizer_files(),
        video_storage_url,
        video_storage_path: args.video_storage_path.clone(),
        logic_data,
        samus_sprite_categories,
        _debug: args.debug,
        port: args.port,
        version_info,
        static_visualizer: args.static_visualizer,
        etank_colors,
        mosaic_themes,
    };
    info!("Start-up time: {:.3}s", start_time.elapsed().as_secs_f32());
    app_data
}

pub async fn fav_icon() -> actix_web::Result<actix_files::NamedFile> {
    Ok(NamedFile::open("static/favicon.ico")?)
}

#[derive(Template)]
#[template(path = "home.html")]
struct HomeTemplate {
    version_info: VersionInfo,
}

#[get("/")]
async fn home(app_data: actix_web::web::Data<AppData>) -> impl Responder {
    let home_template = HomeTemplate {
        version_info: app_data.version_info.clone(),
    };
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(home_template.render().unwrap())
}

#[derive(Template)]
#[template(path = "releases.html")]
struct ReleasesTemplate {
    version_info: VersionInfo,
}

#[get("/releases")]
async fn releases(app_data: actix_web::web::Data<AppData>) -> impl Responder {
    let changes_template = ReleasesTemplate {
        version_info: app_data.version_info.clone(),
    };
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(changes_template.render().unwrap())
}

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
    doors_presets_json: String,
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
async fn generate(app_data: actix_web::web::Data<AppData>) -> impl Responder {
    let item_names_multiple: Vec<String> =
        ["Missile", "ETank", "ReserveTank", "Super", "PowerBomb"]
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
        "SparkBooster",
        "BlueBooster",
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
        "SparkBooster",
        "BlueBooster",
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
    let doors_presets_json = serde_json::to_string(&app_data.preset_data.doors_presets).unwrap();
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
        item_priorities: ["Early", "Default", "Late"]
            .iter()
            .map(|x| x.to_string())
            .collect(),
        prioritizable_items,
        preset_data: &app_data.preset_data,
        full_presets_json,
        skill_presets_json,
        item_presets_json,
        qol_presets_json,
        doors_presets_json,
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

#[derive(Serialize, Deserialize)]
struct SeedData {
    version: usize,
    timestamp: usize,
    peer_addr: String,
    http_headers: serde_json::Map<String, serde_json::Value>,
    random_seed: usize,
    map_seed: usize,
    door_randomization_seed: usize,
    item_placement_seed: usize,
    settings: RandomizerSettings,
    // TODO: get rid of all the redundant stuff below:
    race_mode: bool,
    preset: Option<String>,
    item_progression_preset: Option<String>,
    difficulty: DifficultyConfig,
    quality_of_life_preset: Option<String>,
    supers_double: bool,
    mother_brain_fight: String,
    escape_enemies_cleared: bool,
    escape_refill: bool,
    escape_movement_items: bool,
    item_markers: String,
    all_items_spawn: bool,
    acid_chozo: bool,
    remove_climb_lava: bool,
    buffed_drops: bool,
    fast_elevators: bool,
    fast_doors: bool,
    fast_pause_menu: bool,
    respin: bool,
    infinite_space_jump: bool,
    momentum_conservation: bool,
    fanfares: String,
    objectives: Vec<String>,
    doors_preset: Option<String>,
    red_doors_count: i32,
    green_doors_count: i32,
    yellow_doors_count: i32,
    charge_doors_count: i32,
    ice_doors_count: i32,
    wave_doors_count: i32,
    spazer_doors_count: i32,
    plasma_doors_count: i32,
    start_location_mode: String,
    map_layout: String,
    save_animals: String,
    early_save: bool,
    wall_jump: String,
    vanilla_map: bool,
    ultra_low_qol: bool,
}

#[derive(MultipartForm)]
struct RandomizeRequest {
    spoiler_token: Text<String>,
    settings: Text<String>,
}

#[derive(Serialize)]
struct RandomizeResponse {
    seed_url: String,
}

struct AttemptOutput {
    random_seed: usize,
    map_seed: usize,
    door_randomization_seed: usize,
    item_placement_seed: usize,
    randomization: Randomization,
    spoiler_log: SpoilerLog,
    difficulty_tiers: Vec<DifficultyConfig>,
}

#[derive(Debug)]
enum AttemptError {
    TooManyAttempts,
    TimedOut,
}

fn handle_randomize_request(
    settings: RandomizerSettings,
    app_data: actix_web::web::Data<AppData>,
) -> Result<AttemptOutput, AttemptError> {
    let race_mode = settings.other_settings.race_mode;
    let random_seed = match (settings.other_settings.random_seed, race_mode) {
        (_, true) | (None, _) => get_random_seed(),
        (Some(s), false) => s,
    };
    let display_seed = if race_mode {
        get_random_seed()
    } else {
        random_seed
    };
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&random_seed.to_le_bytes());
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let difficulty_tiers = get_difficulty_tiers(
        &settings,
        &app_data.preset_data.difficulty_tiers,
        &app_data.game_data,
        &app_data.preset_data.tech_by_difficulty["Implicit"],
        &app_data.preset_data.notables_by_difficulty["Implicit"],
    );

    let filtered_base_links = filter_links(
        &app_data.game_data.links,
        &app_data.game_data,
        &difficulty_tiers[0],
    );
    let filtered_base_links_data = LinksDataGroup::new(
        filtered_base_links,
        app_data.game_data.vertex_isv.keys.len(),
        0,
    );
    let map_layout = settings.map_layout.clone();
    let max_attempts = 2000;
    let attempts_timeout = Duration::from_secs(25);
    let max_attempts_per_map = if settings.start_location_settings.mode == StartLocationMode::Random
    {
        10
    } else {
        1
    };
    let max_map_attempts = max_attempts / max_attempts_per_map;
    info!(
        "Random seed={random_seed}, max_attempts_per_map={max_attempts_per_map}, max_map_attempts={max_map_attempts}, difficulty={:?}",
        difficulty_tiers[0]
    );

    let time_start_attempts = Instant::now();
    let mut attempt_num = 0;
    let mut map_batch: Vec<Map> = vec![];
    for _ in 0..max_map_attempts {
        let map_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        let door_randomization_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;

        if !app_data.map_repositories.contains_key(&map_layout) {
            // TODO: it doesn't make sense to panic on things like this.
            panic!("Unrecognized map layout option: {map_layout}");
        }

        if map_batch.is_empty() {
            map_batch = app_data.map_repositories[&map_layout]
                .get_map_batch(map_seed, &app_data.game_data)
                .unwrap();
        }

        let mut map = map_batch.pop().unwrap();
        if !assign_map_areas(&mut map, &settings, map_seed, &app_data.game_data) {
            info!("Area assignment failed for map seed={map_seed}");
            continue;
        }
        let objectives = get_objectives(&settings, Some(&map), &app_data.game_data, &mut rng);
        let locked_door_data = randomize_doors(
            &app_data.game_data,
            &map,
            &settings,
            &objectives,
            door_randomization_seed,
        );
        let randomizer = Randomizer::new(
            &map,
            &locked_door_data,
            objectives.clone(),
            &settings,
            &difficulty_tiers,
            &app_data.game_data,
            &filtered_base_links_data,
            &mut rng,
        );
        for _ in 0..max_attempts_per_map {
            let item_placement_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
            attempt_num += 1;

            info!(
                "Attempt {attempt_num}/{max_attempts}: Map seed={map_seed}, door randomization seed={door_randomization_seed}, item placement seed={item_placement_seed}"
            );
            let randomization_result =
                randomizer.randomize(attempt_num, item_placement_seed, display_seed, true);
            let (randomization, spoiler_log) = match randomization_result {
                Ok(x) => x,
                Err(e) => {
                    info!("Attempt {attempt_num}/{max_attempts}: Randomization failed: {e}");
                    if time_start_attempts.elapsed() > attempts_timeout {
                        return Err(AttemptError::TimedOut);
                    }
                    continue;
                }
            };
            info!(
                "Successful attempt {attempt_num}/{attempt_num}/{max_attempts}: display_seed={}, random_seed={random_seed}, map_seed={map_seed}, door_randomization_seed={door_randomization_seed}, item_placement_seed={item_placement_seed}",
                randomization.display_seed,
            );

            info!(
                "Wall-clock time for attempts: {:?} sec",
                time_start_attempts.elapsed().as_secs_f32()
            );
            let output_result = Ok(AttemptOutput {
                random_seed,
                map_seed,
                door_randomization_seed,
                item_placement_seed,
                randomization,
                spoiler_log,
                difficulty_tiers,
            });
            return output_result;
        }
    }
    Err(AttemptError::TooManyAttempts)
}

#[post("/randomize")]
async fn randomize(
    req: MultipartForm<RandomizeRequest>,
    http_req: HttpRequest,
    app_data: actix_web::web::Data<AppData>,
) -> impl Responder {
    let mut settings =
        match try_upgrade_settings(req.settings.0.to_string(), &app_data.preset_data, true) {
            Ok(s) => s.1,
            Err(e) => {
                return HttpResponse::BadRequest().body(e.to_string());
            }
        };

    let mut validated_preset = false;
    for s in &app_data.preset_data.full_presets {
        if s == &settings {
            validated_preset = true;
            break;
        }
    }
    if !validated_preset {
        settings.name = Some("Custom".to_string());
    }

    if settings.other_settings.random_seed == Some(0) {
        return HttpResponse::BadRequest().body("Invalid random seed: 0");
    }

    if settings.skill_assumption_settings.ridley_proficiency < 0.0
        || settings.skill_assumption_settings.ridley_proficiency > 1.0
    {
        return HttpResponse::BadRequest().body("Invalid Ridley proficiency");
    }

    let settings_copy = settings.clone();
    let app_data_copy = app_data.clone();
    let output_result = actix_web::rt::task::spawn_blocking(|| {
        handle_randomize_request(settings_copy, app_data_copy)
    })
    .await
    .unwrap();

    let output = match output_result {
        Ok(x) => x,
        Err(AttemptError::TimedOut) => {
            return HttpResponse::InternalServerError()
                .body("Failed too many randomization attempts (timeout reached)");
        }
        Err(AttemptError::TooManyAttempts) => {
            return HttpResponse::InternalServerError()
                .body("Failed too many randomization attempts (maximum attempt count reached)");
        }
    };

    let timestamp = match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(n) => n.as_millis() as usize,
        Err(_) => panic!("SystemTime before UNIX EPOCH!"),
    };

    let seed_data = SeedData {
        version: VERSION,
        timestamp,
        peer_addr: http_req
            .peer_addr()
            .map(|x| format!("{x:?}"))
            .unwrap_or_default(),
        http_headers: format_http_headers(&http_req),
        random_seed: output.random_seed,
        map_seed: output.map_seed,
        door_randomization_seed: output.door_randomization_seed,
        item_placement_seed: output.item_placement_seed,
        settings: settings.clone(),
        race_mode: settings.other_settings.race_mode,
        preset: settings.skill_assumption_settings.preset.clone(),
        item_progression_preset: settings.item_progression_settings.preset.clone(),
        difficulty: output.difficulty_tiers[0].clone(),
        quality_of_life_preset: settings.quality_of_life_settings.preset.clone(),
        supers_double: settings.quality_of_life_settings.supers_double,
        mother_brain_fight: to_variant_name(&settings.quality_of_life_settings.mother_brain_fight)
            .unwrap()
            .to_string(),
        escape_enemies_cleared: settings.quality_of_life_settings.escape_enemies_cleared,
        escape_refill: settings.quality_of_life_settings.escape_refill,
        escape_movement_items: settings.quality_of_life_settings.escape_movement_items,
        item_markers: to_variant_name(&settings.quality_of_life_settings.item_markers)
            .unwrap()
            .to_string(),
        all_items_spawn: settings.quality_of_life_settings.all_items_spawn,
        acid_chozo: settings.quality_of_life_settings.acid_chozo,
        remove_climb_lava: settings.quality_of_life_settings.remove_climb_lava,
        buffed_drops: settings.quality_of_life_settings.buffed_drops,
        fast_elevators: settings.quality_of_life_settings.fast_elevators,
        fast_doors: settings.quality_of_life_settings.fast_doors,
        fast_pause_menu: settings.quality_of_life_settings.fast_pause_menu,
        respin: settings.quality_of_life_settings.respin,
        infinite_space_jump: settings.quality_of_life_settings.infinite_space_jump,
        momentum_conservation: settings.quality_of_life_settings.momentum_conservation,
        fanfares: to_variant_name(&settings.quality_of_life_settings.fanfares)
            .unwrap()
            .to_string(),
        objectives: output
            .randomization
            .objectives
            .iter()
            .map(|x| to_variant_name(x).unwrap().to_string())
            .collect(),
        doors_preset: settings.doors_settings.preset.clone(),
        red_doors_count: settings.doors_settings.red_doors_count,
        green_doors_count: settings.doors_settings.green_doors_count,
        yellow_doors_count: settings.doors_settings.yellow_doors_count,
        charge_doors_count: settings.doors_settings.charge_doors_count,
        ice_doors_count: settings.doors_settings.ice_doors_count,
        wave_doors_count: settings.doors_settings.wave_doors_count,
        spazer_doors_count: settings.doors_settings.spazer_doors_count,
        plasma_doors_count: settings.doors_settings.plasma_doors_count,
        start_location_mode: if settings.start_location_settings.mode == StartLocationMode::Custom {
            output.randomization.start_location.name.clone()
        } else {
            to_variant_name(&settings.start_location_settings.mode)
                .unwrap()
                .to_string()
        },
        map_layout: settings.map_layout.clone(),
        save_animals: to_variant_name(&settings.save_animals).unwrap().to_string(),
        early_save: settings.quality_of_life_settings.early_save,
        wall_jump: to_variant_name(&settings.other_settings.wall_jump)
            .unwrap()
            .to_string(),
        vanilla_map: settings.map_layout == "Vanilla",
        ultra_low_qol: settings.other_settings.ultra_low_qol,
    };

    let seed_name = &output.randomization.seed_name;
    save_seed(
        seed_name,
        &seed_data,
        &req.settings.0,
        &req.spoiler_token.0,
        &settings,
        &output.randomization,
        &output.spoiler_log,
        &app_data,
    )
    .await
    .unwrap();

    HttpResponse::Ok().json(RandomizeResponse {
        seed_url: format!("/seed/{seed_name}/"),
    })
}

#[derive(Template)]
#[template(path = "about.html")]
struct AboutTemplate {
    version_info: VersionInfo,
    sprite_artists: Vec<String>,
    video_creators: Vec<(String, usize)>,
}

impl AboutTemplate {
    fn sprite_artists(&self) -> String {
        self.sprite_artists
            .iter()
            .map(|x| format!("<i>{x}</i>"))
            .collect::<Vec<String>>()
            .join(", ")
    }

    fn video_creators(&self) -> String {
        self.video_creators
            .iter()
            .map(|x| format!("<i>{}</i>", x.0))
            .collect::<Vec<String>>()
            .join(", ")
    }
}

#[get("/about")]
async fn about(app_data: actix_web::web::Data<AppData>) -> impl Responder {
    let mut sprite_artists = vec![];

    for category in &app_data.samus_sprite_categories {
        for info in &category.sprites {
            for author in &info.authors {
                if info.display_name != "Samus" {
                    sprite_artists.push(author.clone());
                }
            }
        }
    }
    sprite_artists.sort_by_key(|x| x.to_lowercase());
    sprite_artists.dedup();

    let mut video_creator_cnt: HashMap<String, usize> = HashMap::new();
    for (_, video_list) in app_data.game_data.strat_videos.iter() {
        for video in video_list {
            *video_creator_cnt
                .entry(video.created_user.clone())
                .or_default() += 1;
        }
    }
    let mut video_creators: Vec<(String, usize)> = video_creator_cnt.into_iter().collect();
    video_creators.sort_by_key(|x| x.1);
    video_creators.reverse();

    let about_template = AboutTemplate {
        version_info: app_data.version_info.clone(),
        sprite_artists,
        video_creators,
    };
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(about_template.render().unwrap())
}

#[post("/upgrade-settings")]
async fn upgrade_settings(
    settings_str: String,
    app_data: actix_web::web::Data<AppData>,
) -> impl Responder {
    match try_upgrade_settings(settings_str, &app_data.preset_data, true) {
        Ok((settings_str, _)) => HttpResponse::Ok()
            .content_type("application/json")
            .body(settings_str),
        Err(e) => {
            error!("Failed to upgrade settings: {e}");
            HttpResponse::BadRequest().body(e.to_string())
        }
    }
}

#[actix_web::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    let app_data = actix_web::web::Data::new(build_app_data());

    let port = app_data.port;

    HttpServer::new(move || {
        let mut app = App::new()
            .wrap(Compress::default())
            .app_data(app_data.clone())
            .app_data(
                MultipartFormConfig::default()
                    .memory_limit(16_000_000)
                    .total_limit(16_000_000),
            )
            .wrap(Logger::default())
            .service(home)
            .service(releases)
            .service(generate)
            .service(randomize)
            .service(about)
            .service(seed::scope())
            .service(logic::scope())
            .service(upgrade_settings)
            .service(actix_files::Files::new(
                "/static/sm-json-data",
                "../sm-json-data",
            ))
            .service(actix_files::Files::new("/static", "static"))
            .service(actix_files::Files::new("/wasm", "maprando-wasm/pkg"))
            .route("/favicon.ico", actix_web::web::get().to(fav_icon));

        if let Some(path) = &app_data.video_storage_path {
            app = app.service(actix_files::Files::new("/videos", path));
        }
        app
    })
    .workers(1)
    .worker_max_blocking_threads(1)
    .bind(("0.0.0.0", port))
    .unwrap()
    .run()
    .await
    .unwrap();
}
