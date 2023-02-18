use std::path::{Path, PathBuf};
use std::time::SystemTime;

use actix_easy_multipart::bytes::Bytes;
use actix_easy_multipart::text::Text;
use actix_easy_multipart::{MultipartForm, MultipartFormConfig};
use actix_web::http::header::{self, ContentDisposition, DispositionParam, DispositionType};
use actix_web::middleware::Logger;
use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use anyhow::{Context, Result};
use base64::Engine;
use clap::Parser;
use hashbrown::{HashMap, HashSet};
use log::{error, info};
use maprando::customize::{customize_rom, CustomizeSettings};
use maprando::game_data::{GameData, Map};
use maprando::patch::ips_write::create_ips_patch;
use maprando::patch::{make_rom, Rom};
use maprando::randomize::{DifficultyConfig, Randomization, Randomizer};
use maprando::seed_repository::{Seed, SeedFile, SeedRepository};
use maprando::spoiler_map;
use rand::{RngCore, SeedableRng};
use sailfish::TemplateOnce;
use serde_derive::{Deserialize, Serialize};

const VERSION: usize = 37;

#[derive(Serialize, Deserialize, Clone)]
struct Preset {
    name: String,
    shinespark_tiles: usize,
    resource_multiplier: f32,
    escape_timer_multiplier: f32,
    phantoon_proficiency: f32,
    draygon_proficiency: f32,
    ridley_proficiency: f32,
    botwoon_proficiency: f32,
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
    seed_repository: SeedRepository,
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
#[template(path = "errors/missing_input_rom.stpl")]
struct MissingInputRomTemplate {}

#[derive(TemplateOnce)]
#[template(path = "errors/invalid_rom.stpl")]
struct InvalidRomTemplate {}

#[derive(TemplateOnce)]
#[template(path = "errors/seed_not_found.stpl")]
struct SeedNotFoundTemplate {}

#[derive(TemplateOnce)]
#[template(path = "errors/file_not_found.stpl")]
struct FileNotFoundTemplate {}

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
    phantoon_proficiency: Text<f32>,
    draygon_proficiency: Text<f32>,
    ridley_proficiency: Text<f32>,
    botwoon_proficiency: Text<f32>,
    escape_timer_multiplier: Text<f32>,
    save_animals: Text<bool>,
    tech_json: Text<String>,
    race_mode: Text<String>,
    random_seed: Text<String>,
    quality_of_life_preset: Option<Text<bool>>,
    supers_double: Text<bool>,
    streamlined_escape: Text<bool>,
    mark_map_stations: Text<bool>,
    mark_uniques: Text<bool>,
    mark_tanks: Text<bool>,
    fast_elevators: Text<bool>,
}

#[derive(MultipartForm)]
struct CustomizeRequest {
    rom: Bytes,
    room_palettes: Text<String>,
}

#[derive(Serialize, Deserialize)]
struct SeedData {
    version: usize,
    timestamp: usize,
    peer_addr: String,
    http_headers: serde_json::Map<String, serde_json::Value>,
    random_seed: usize,
    map_seed: usize,
    item_placement_seed: usize,
    race_mode: bool,
    preset: Option<String>,
    difficulty: DifficultyConfig,
    quality_of_life_preset: Option<bool>,
    supers_double: bool,
    streamlined_escape: bool,
    mark_map_stations: bool,
    mark_uniques: bool,
    mark_tanks: bool,
    fast_elevators: bool,
}

fn get_seed_name(seed_data: &SeedData) -> String {
    let seed_data_str = serde_json::to_string(&seed_data).unwrap();
    let h128 = fasthash::spooky::hash128(seed_data_str);
    let base64_str = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(h128.to_le_bytes());
    base64_str
}

#[derive(TemplateOnce)]
#[template(path = "seed/seed_header.stpl")]
struct SeedHeaderTemplate<'a> {
    seed_name: String,
    timestamp: usize, // Milliseconds since UNIX epoch
    random_seed: usize,
    version: usize,
    race_mode: bool,
    preset: String,
    item_placement_strategy: String,
    difficulty: &'a DifficultyConfig,
    quality_of_life_preset: Option<bool>,
    supers_double: bool,
    streamlined_escape: bool,
    mark_map_stations: bool,
    mark_uniques: bool,
    mark_tanks: bool,
    fast_elevators: bool,
}

#[derive(TemplateOnce)]
#[template(path = "seed/seed_footer.stpl")]
struct SeedFooterTemplate {
    race_mode: bool,
}

#[derive(TemplateOnce)]
#[template(path = "seed/customize_seed.stpl")]
struct CustomizeSeedTemplate {
    version: usize,
    seed_header: String,
    seed_footer: String,
}

fn render_seed(seed_name: &str, seed_data: &SeedData) -> Result<(String, String)> {
    let seed_header_template = SeedHeaderTemplate {
        seed_name: seed_name.to_string(),
        version: VERSION,
        random_seed: seed_data.random_seed,
        race_mode: seed_data.race_mode,
        timestamp: seed_data.timestamp,
        preset: seed_data.preset.clone().unwrap_or("Custom".to_string()),
        item_placement_strategy: format!("{:?}", seed_data.difficulty.item_placement_strategy),
        difficulty: &seed_data.difficulty,
        quality_of_life_preset: seed_data.quality_of_life_preset,
        supers_double: seed_data.supers_double,
        streamlined_escape: seed_data.streamlined_escape,
        mark_map_stations: seed_data.mark_map_stations,
        mark_uniques: seed_data.mark_uniques,
        mark_tanks: seed_data.mark_tanks,
        fast_elevators: seed_data.fast_elevators,
    };
    let seed_header_html = seed_header_template.render_once()?;

    let seed_footer_template = SeedFooterTemplate {
        race_mode: seed_data.race_mode,
    };
    let seed_footer_html = seed_footer_template.render_once()?;
    Ok((seed_header_html, seed_footer_html))
}

async fn save_seed(
    seed_name: &str,
    seed_data: &SeedData,
    vanilla_rom: &Rom,
    output_rom: &Rom,
    randomization: &Randomization,
    app_data: &AppData,
) -> Result<()> {
    let mut files: Vec<SeedFile> = Vec::new();

    // Write the seed data JSON. This contains details about the seed and request origin,
    // so to protect user privacy and the integrity of race ROMs we do not make it public.
    let seed_data_str = serde_json::to_vec_pretty(&seed_data).unwrap();
    files.push(SeedFile::new("seed_data.json", seed_data_str.to_vec()));

    // Write the ROM patch.
    let patch_ips = create_ips_patch(&vanilla_rom.data, &output_rom.data);
    files.push(SeedFile::new("patch.ips", patch_ips));

    // Write the seed header HTML and footer HTML
    let (seed_header_html, seed_footer_html) = render_seed(seed_name, seed_data)?;
    files.push(SeedFile::new(
        "seed_header.html",
        seed_header_html.into_bytes(),
    ));
    files.push(SeedFile::new(
        "seed_footer.html",
        seed_footer_html.into_bytes(),
    ));

    if !seed_data.race_mode {
        // Write the spoiler log
        let spoiler_bytes = serde_json::to_vec_pretty(&randomization.spoiler_log).unwrap();
        files.push(SeedFile::new("public/spoiler.json", spoiler_bytes));

        // Write the spoiler maps
        let spoiler_map_assigned = spoiler_map::get_spoiler_map(
            &output_rom,
            &randomization.map,
            &app_data.game_data,
            false,
        )
        .unwrap();
        files.push(SeedFile::new(
            "public/map-assigned.png",
            spoiler_map_assigned,
        ));
        let spoiler_map_vanilla = spoiler_map::get_spoiler_map(
            &output_rom,
            &randomization.map,
            &app_data.game_data,
            true,
        )
        .unwrap();
        files.push(SeedFile::new("public/map-vanilla.png", spoiler_map_vanilla));
    }

    let seed = Seed {
        name: seed_name.to_string(),
        files,
    };
    app_data.seed_repository.put_seed(seed).await?;
    Ok(())
}

#[get("/seed/{name}")]
async fn view_seed_redirect(info: web::Path<(String,)>) -> impl Responder {
    // Redirect to the seed page (with the trailing slash):
    HttpResponse::Found()
        .insert_header((header::LOCATION, format!("{}/", info.0)))
        .finish()
}

#[get("/seed/{name}/")]
async fn view_seed(info: web::Path<(String,)>, app_data: web::Data<AppData>) -> impl Responder {
    let seed_name = &info.0;
    let (seed_header, seed_footer) = futures::join!(
        app_data
            .seed_repository
            .get_file(seed_name, "seed_header.html"),
        app_data
            .seed_repository
            .get_file(seed_name, "seed_footer.html")
    );

    match (seed_header, seed_footer) {
        (Ok(header), Ok(footer)) => {
            let customize_template = CustomizeSeedTemplate {
                version: VERSION,
                seed_header: String::from_utf8(header.to_vec()).unwrap(),
                seed_footer: String::from_utf8(footer.to_vec()).unwrap(),
            };
            HttpResponse::Ok().body(customize_template.render_once().unwrap())
        }
        (Err(err), _) => {
            error!("{}", err.to_string());
            let template = SeedNotFoundTemplate {};
            HttpResponse::NotFound().body(template.render_once().unwrap())
        }
        (_, Err(err)) => {
            error!("{}", err.to_string());
            let template = SeedNotFoundTemplate {};
            HttpResponse::NotFound().body(template.render_once().unwrap())
        }
    }
}

#[post("/seed/{name}/customize")]
async fn customize_seed(
    req: MultipartForm<CustomizeRequest>,
    info: web::Path<(String,)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let seed_name = &info.0;
    let patch_ips = app_data
        .seed_repository
        .get_file(seed_name, "patch.ips")
        .await
        .unwrap();
    let mut rom = Rom {
        data: req.rom.data.to_vec(),
    };

    if rom.data.len() < 0x300000 {
        return HttpResponse::BadRequest().body("Invalid base ROM.");
    }

    let settings = CustomizeSettings {
        area_themed_palette: req.room_palettes.0 == "area-themed",
    };
    info!("CustomizeSettings: {:?}", settings);
    match customize_rom(&mut rom, &patch_ips, &settings, &app_data.game_data) {
        Ok(()) => {}
        Err(err) => {
            return HttpResponse::InternalServerError()
                .body(format!("Error customizing ROM: {:?}", err))
        }
    }

    HttpResponse::Ok()
        .content_type("application/octet-stream")
        .insert_header(ContentDisposition {
            disposition: DispositionType::Attachment,
            parameters: vec![DispositionParam::Filename(
                "map-rando-".to_string() + seed_name + ".sfc",
            )],
        })
        .body(rom.data)
}

#[get("/seed/{name}/data/{filename}")]
async fn get_seed_file(
    info: web::Path<(String, String)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let seed_name = &info.0;
    let filename = &info.1;
    match app_data
        .seed_repository
        .get_file(seed_name, &("public/".to_string() + filename))
        .await
    {
        Ok(data) => {
            let ext = Path::new(filename)
                .extension()
                .map(|x| x.to_str().unwrap())
                .unwrap_or("bin");
            let mime = actix_files::file_extension_to_mime(ext);
            HttpResponse::Ok().content_type(mime).body(data)
        }
        // TODO: Use more refined error handling instead of always returning 404:
        Err(err) => {
            error!("{}", err.to_string());
            HttpResponse::NotFound().body(FileNotFoundTemplate {}.render_once().unwrap())
        }
    }
}

fn format_http_headers(req: &HttpRequest) -> serde_json::Map<String, serde_json::Value> {
    let map: serde_json::Map<String, serde_json::Value> = req
        .headers()
        .into_iter()
        .map(|(name, value)| {
            (
                name.to_string(),
                serde_json::Value::String(value.to_str().unwrap_or("").to_string()),
            )
        })
        .collect();
    map
}

fn get_random_seed() -> usize {
    (rand::rngs::StdRng::from_entropy().next_u64() & 0xFFFFFFFF) as usize
}

#[post("/randomize")]
async fn randomize(
    req: MultipartForm<RandomizeRequest>,
    http_req: HttpRequest,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let rom = Rom {
        data: req.rom.data.to_vec(),
    };

    if rom.data.len() == 0 {
        return HttpResponse::BadRequest().body(MissingInputRomTemplate {}.render_once().unwrap());
    }

    let rom_digest = crypto_hash::hex_digest(crypto_hash::Algorithm::SHA256, &rom.data);
    info!("Rom digest: {rom_digest}");
    if rom_digest != "12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72" {
        return HttpResponse::BadRequest().body(InvalidRomTemplate {}.render_once().unwrap());
    }

    let random_seed = if &req.random_seed.0 == "" {
        get_random_seed()
    } else {
        match req.random_seed.0.parse::<usize>() {
            Ok(x) => x,
            Err(_) => {
                return HttpResponse::BadRequest().body("Invalid random seed");
            }
        }
    };

    if req.ridley_proficiency.0 < 0.0 || req.ridley_proficiency.0 > 1.0 {
        return HttpResponse::BadRequest().body("Invalid Ridley proficiency");
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
        save_animals: req.save_animals.0,
        phantoon_proficiency: req.phantoon_proficiency.0,
        draygon_proficiency: req.draygon_proficiency.0,
        ridley_proficiency: req.ridley_proficiency.0,
        botwoon_proficiency: req.botwoon_proficiency.0,
        supers_double: req.supers_double.0,
        streamlined_escape: req.streamlined_escape.0,
        mark_map_stations: req.mark_map_stations.0,
        mark_uniques: req.mark_uniques.0,
        mark_tanks: req.mark_tanks.0,
        fast_elevators: req.fast_elevators.0,
        debug_options: None,
    };
    let race_mode = req.race_mode.0 == "Yes";
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&random_seed.to_le_bytes());
    rng_seed[9] = if race_mode { 1 } else { 0 };
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);
    let max_attempts = 100;
    let mut attempt_num = 0;
    let randomization: Randomization;
    info!("Random seed={random_seed}, difficulty={difficulty:?}");
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

    let timestamp = match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(n) => n.as_millis() as usize,
        Err(_) => panic!("SystemTime before UNIX EPOCH!"),
    };
    let seed_data = SeedData {
        version: VERSION,
        timestamp,
        peer_addr: http_req
            .peer_addr()
            .map(|x| format!("{:?}", x))
            .unwrap_or(String::new()),
        http_headers: format_http_headers(&http_req),
        random_seed: random_seed,
        map_seed,
        item_placement_seed,
        race_mode,
        preset: req.preset.as_ref().map(|x| x.0.clone()),
        difficulty,
        quality_of_life_preset: req.quality_of_life_preset.as_ref().map(|x| x.0),
        supers_double: req.supers_double.0,
        streamlined_escape: req.streamlined_escape.0,
        mark_map_stations: req.mark_map_stations.0,
        mark_uniques: req.mark_uniques.0,
        mark_tanks: req.mark_tanks.0,
        fast_elevators: req.fast_elevators.0,
    };

    let output_rom = make_rom(&rom, &randomization, &app_data.game_data).unwrap();

    let seed_name = get_seed_name(&seed_data);
    save_seed(
        &seed_name,
        &seed_data,
        &rom,
        &output_rom,
        &randomization,
        &app_data,
    )
    .await
    .unwrap();

    // Redirect to the seed page:
    HttpResponse::Found()
        .insert_header((header::LOCATION, format!("seed/{}/", seed_name)))
        .finish()
}

fn init_presets(presets: Vec<Preset>, game_data: &GameData) -> Vec<PresetData> {
    let mut out: Vec<PresetData> = Vec::new();
    let mut cumulative_tech: HashSet<String> = HashSet::new();

    // Tech which is currently not used by any strat in logic, so we avoid showing on the website:
    let ignored_tech: HashSet<String> = [
        "canWallIceClip",
        "canGrappleClip",
        "canUseSpeedEchoes",
        "canSamusEaterStandUp",
        "canRiskPermanentLossOfAccess",
    ]
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

#[derive(Parser)]
struct Args {
    #[arg(long)]
    seed_repository_url: String,
}

fn build_app_data() -> AppData {
    let args = Args::parse();
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let palette_path = Path::new("../palettes.json");
    let maps_path =
        Path::new("../maps/session-2022-06-03T17:19:29.727911.pkl-bk30-subarea-balance");

    let game_data = GameData::load(sm_json_data_path, room_geometry_path, palette_path).unwrap();
    let presets: Vec<Preset> =
        serde_json::from_str(&std::fs::read_to_string(&"data/presets.json").unwrap()).unwrap();
    let preset_data = init_presets(presets, &game_data);
    AppData {
        game_data,
        preset_data,
        map_repository: MapRepository::new(maps_path).unwrap(),
        seed_repository: SeedRepository::new(&args.seed_repository_url).unwrap(),
    }
}

#[actix_web::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    let app_data = web::Data::new(build_app_data());

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
            .service(view_seed)
            .service(get_seed_file)
            .service(customize_seed)
            .service(view_seed_redirect)
            .service(actix_files::Files::new("/static", "static"))
    })
    .bind("0.0.0.0:8080")
    .unwrap()
    .run()
    .await
    .unwrap();
}
