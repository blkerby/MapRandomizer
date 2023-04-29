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
use maprando::game_data::{GameData, IndexedVec, Item, Map};
use maprando::patch::ips_write::create_ips_patch;
use maprando::patch::{make_rom, Rom};
use maprando::randomize::{
    DebugOptions, DifficultyConfig, ItemMarkers, ItemPlacementStyle, ItemPriorityGroup,
    MotherBrainFight, Objectives, Randomization, Randomizer,
};
use maprando::seed_repository::{Seed, SeedFile, SeedRepository};
use maprando::spoiler_map;
use rand::{RngCore, SeedableRng};
use sailfish::TemplateOnce;
use serde_derive::{Deserialize, Serialize};

const VERSION: usize = 57;
const VISUALIZER_PATH: &'static str = "../visualizer/";

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
    notable_strats: Vec<String>,
}

struct PresetData {
    preset: Preset,
    tech_setting: Vec<(String, bool)>,
    notable_strat_setting: Vec<(String, bool)>,
}

struct MapRepository {
    base_path: PathBuf,
    filenames: Vec<String>,
}

struct AppData {
    game_data: GameData,
    preset_data: Vec<PresetData>,
    ignored_notable_strats: HashSet<String>,
    map_repository: MapRepository,
    seed_repository: SeedRepository,
    visualizer_files: Vec<(String, Vec<u8>)>, // (path, contents)
    debug: bool,
    static_visualizer: bool,
}

impl MapRepository {
    fn new(base_path: &Path) -> Result<Self> {
        let mut filenames: Vec<String> = Vec::new();
        for path in std::fs::read_dir(base_path)? {
            filenames.push(path?.file_name().into_string().unwrap());
        }
        filenames.sort();
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
    progression_rates: Vec<&'static str>,
    item_placement_styles: Vec<&'static str>,
    objectives: Vec<&'static str>,
    preset_data: &'a [PresetData],
    item_priorities: Vec<String>,
    prioritizable_items: Vec<String>,
    tech_description: &'a HashMap<String, String>,
    tech_dependencies: &'a HashMap<String, Vec<String>>,
    strat_dependencies: &'a HashMap<String, Vec<String>>,
    _strat_area: &'a HashMap<String, String>,
    strat_room: &'a HashMap<String, String>,
    strat_description: &'a HashMap<String, String>,
    strat_id_by_name: &'a HashMap<String, usize>,
}

#[get("/")]
async fn home(app_data: web::Data<AppData>) -> impl Responder {
    let mut prioritizable_items: Vec<String> = app_data
        .game_data
        .item_isv
        .keys
        .iter()
        .cloned()
        .filter(|x| x != "Missile")
        .collect();
    prioritizable_items.sort();
    let home_template = HomeTemplate {
        version: VERSION,
        progression_rates: vec!["Fast", "Normal", "Slow"],
        item_placement_styles: vec!["Neutral", "Forced"],
        objectives: vec!["Bosses", "Minibosses", "Metroids"],
        item_priorities: vec!["Early", "Default", "Late"]
            .iter()
            .map(|x| x.to_string())
            .collect(),
        prioritizable_items,
        preset_data: &app_data.preset_data,
        tech_description: &app_data.game_data.tech_description,
        tech_dependencies: &app_data.game_data.tech_dependencies,
        strat_dependencies: &app_data.game_data.strat_dependencies,
        _strat_area: &app_data.game_data.strat_area,
        strat_room: &app_data.game_data.strat_room,
        strat_description: &app_data.game_data.strat_description,
        strat_id_by_name: &app_data.game_data.notable_strat_isv.index_by_key,
    };
    HttpResponse::Ok().body(home_template.render_once().unwrap())
}

#[derive(MultipartForm)]
struct RandomizeRequest {
    rom: Bytes,
    preset: Option<Text<String>>,
    shinespark_tiles: Text<f32>,
    resource_multiplier: Text<f32>,
    phantoon_proficiency: Text<f32>,
    draygon_proficiency: Text<f32>,
    ridley_proficiency: Text<f32>,
    botwoon_proficiency: Text<f32>,
    escape_timer_multiplier: Text<f32>,
    save_animals: Text<bool>,
    tech_json: Text<String>,
    strat_json: Text<String>,
    progression_rate: Text<String>,
    item_placement_style: Text<String>,
    item_progression_preset: Option<Text<String>>,
    objectives: Text<String>,
    item_priority_json: Text<String>,
    filler_items_json: Text<String>,
    race_mode: Text<String>,
    random_seed: Text<String>,
    quality_of_life_preset: Option<Text<String>>,
    supers_double: Text<bool>,
    mother_brain_fight: Text<String>,
    escape_enemies_cleared: Text<bool>,
    escape_movement_items: Text<bool>,
    mark_map_stations: Text<bool>,
    item_markers: Text<String>,
    all_items_spawn: Text<bool>,
    fast_elevators: Text<bool>,
    fast_doors: Text<bool>,
    fast_pause_menu: Text<bool>,
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
    item_progression_preset: Option<String>,
    difficulty: DifficultyConfig,
    ignored_notable_strats: Vec<String>,
    quality_of_life_preset: Option<String>,
    supers_double: bool,
    mother_brain_fight: String,
    escape_enemies_cleared: bool,
    escape_movement_items: bool,
    mark_map_stations: bool,
    item_markers: String,
    all_items_spawn: bool,
    fast_elevators: bool,
    fast_doors: bool,
    fast_pause_menu: bool,
    objectives: String,
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
    item_progression_preset: String,
    progression_rate: String,
    filler_items: Vec<String>,
    item_placement_style: String,
    difficulty: &'a DifficultyConfig,
    notable_strats: Vec<String>,
    quality_of_life_preset: String,
    supers_double: bool,
    mother_brain_fight: String,
    escape_enemies_cleared: bool,
    escape_movement_items: bool,
    mark_map_stations: bool,
    item_markers: String,
    all_items_spawn: bool,
    fast_elevators: bool,
    fast_doors: bool,
    fast_pause_menu: bool,
    objectives: String,
}

#[derive(TemplateOnce)]
#[template(path = "seed/seed_footer.stpl")]
struct SeedFooterTemplate {
    race_mode: bool,
    all_items_spawn: bool,
    supers_double: bool,
}

#[derive(TemplateOnce)]
#[template(path = "seed/customize_seed.stpl")]
struct CustomizeSeedTemplate {
    version: usize,
    seed_header: String,
    seed_footer: String,
}

fn render_seed(seed_name: &str, seed_data: &SeedData) -> Result<(String, String)> {
    let ignored_notable_strats: HashSet<String> =
        seed_data.ignored_notable_strats.iter().cloned().collect();
    let notable_strats: Vec<String> = seed_data
        .difficulty
        .notable_strats
        .iter()
        .cloned()
        .filter(|x| !ignored_notable_strats.contains(x))
        .collect();
    let seed_header_template = SeedHeaderTemplate {
        seed_name: seed_name.to_string(),
        version: VERSION,
        random_seed: seed_data.random_seed,
        race_mode: seed_data.race_mode,
        timestamp: seed_data.timestamp,
        preset: seed_data.preset.clone().unwrap_or("Custom".to_string()),
        item_progression_preset: seed_data
            .item_progression_preset
            .clone()
            .unwrap_or("Custom".to_string()),
        progression_rate: format!("{:?}", seed_data.difficulty.progression_rate),
        filler_items: seed_data
            .difficulty
            .filler_items
            .iter()
            .map(|x| format!("{:?}", x))
            .collect(),
        item_placement_style: format!("{:?}", seed_data.difficulty.item_placement_style),
        difficulty: &seed_data.difficulty,
        notable_strats,
        quality_of_life_preset: seed_data
            .quality_of_life_preset
            .clone()
            .unwrap_or("Custom".to_string()),
        supers_double: seed_data.supers_double,
        mother_brain_fight: seed_data.mother_brain_fight.clone(),
        escape_enemies_cleared: seed_data.escape_enemies_cleared,
        escape_movement_items: seed_data.escape_movement_items,
        mark_map_stations: seed_data.mark_map_stations,
        item_markers: seed_data.item_markers.clone(),
        all_items_spawn: seed_data.all_items_spawn,
        fast_elevators: seed_data.fast_elevators,
        fast_doors: seed_data.fast_doors,
        fast_pause_menu: seed_data.fast_pause_menu,
        objectives: seed_data.objectives.clone(),
    };
    let seed_header_html = seed_header_template.render_once()?;

    let seed_footer_template = SeedFooterTemplate {
        race_mode: seed_data.race_mode,
        all_items_spawn: seed_data.all_items_spawn,
        supers_double: seed_data.supers_double,
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

        // Write the spoiler visualizer
        for (filename, data) in &app_data.visualizer_files {
            let path = format!("public/visualizer/{}", filename);
            files.push(SeedFile::new(&path, data.clone()));
        }
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
    let mut rom = Rom::new(req.rom.data.to_vec());

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

#[get("/seed/{name}/data/{filename:.*}")]
async fn get_seed_file(
    info: web::Path<(String, String)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let seed_name = &info.0;
    let filename = &info.1;
    println!("get_seed_file {}", filename);

    let data_result: Result<Vec<u8>> =
        if filename.starts_with("visualizer/") && app_data.static_visualizer {
            let path = Path::new(VISUALIZER_PATH).join(filename.strip_prefix("visualizer/").unwrap());
            std::fs::read(&path)
                .map_err(anyhow::Error::from)
                .with_context(|| format!("Error reading static file: {}", path.display()))
        } else {
            app_data
                .seed_repository
                .get_file(seed_name, &("public/".to_string() + filename))
                .await
        };

    match data_result {
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

// Computes the intersection of the selected difficulty with each preset. This
// gives a set of difficulty tiers below the selected difficulty. These are
// used in "forced mode" to try to identify locations at which to place
// key items which are reachable using the selected difficulty but not at
// lower difficulties.
fn get_difficulty_tiers(
    difficulty: &DifficultyConfig,
    app_data: &AppData,
) -> Vec<DifficultyConfig> {
    let presets = &app_data.preset_data;
    let mut out: Vec<DifficultyConfig> = vec![];
    let tech_set: HashSet<String> = difficulty.tech.iter().cloned().collect();
    let strat_set: HashSet<String> = difficulty.notable_strats.iter().cloned().collect();

    out.push(difficulty.clone());
    for preset_data in presets.iter().rev() {
        let preset = &preset_data.preset;
        let mut tech_vec: Vec<String> = Vec::new();
        for (tech, enabled) in &preset_data.tech_setting {
            if *enabled && tech_set.contains(tech) {
                tech_vec.push(tech.clone());
            }
        }

        let mut strat_vec: Vec<String> = app_data.ignored_notable_strats.iter().cloned().collect();
        for (strat, enabled) in &preset_data.notable_strat_setting {
            if *enabled && strat_set.contains(strat) {
                strat_vec.push(strat.clone());
            }
        }

        // TODO: move some fields out of here so we don't have clone as much irrelevant stuff:
        let new_difficulty = DifficultyConfig {
            tech: tech_vec,
            notable_strats: strat_vec,
            shine_charge_tiles: f32::max(
                difficulty.shine_charge_tiles,
                preset.shinespark_tiles as f32,
            ),
            progression_rate: difficulty.progression_rate,
            item_placement_style: difficulty.item_placement_style,
            item_priorities: difficulty.item_priorities.clone(),
            filler_items: difficulty.filler_items.clone(),
            resource_multiplier: f32::max(
                difficulty.resource_multiplier,
                preset.resource_multiplier,
            ),
            escape_timer_multiplier: difficulty.escape_timer_multiplier,
            save_animals: difficulty.save_animals,
            phantoon_proficiency: f32::min(
                difficulty.phantoon_proficiency,
                preset.phantoon_proficiency,
            ),
            draygon_proficiency: f32::min(
                difficulty.draygon_proficiency,
                preset.draygon_proficiency,
            ),
            ridley_proficiency: f32::min(difficulty.ridley_proficiency, preset.ridley_proficiency),
            botwoon_proficiency: f32::min(
                difficulty.botwoon_proficiency,
                preset.botwoon_proficiency,
            ),
            // Quality-of-life options:
            supers_double: difficulty.supers_double,
            mother_brain_fight: difficulty.mother_brain_fight,
            escape_enemies_cleared: difficulty.escape_enemies_cleared,
            escape_movement_items: difficulty.escape_movement_items,
            mark_map_stations: difficulty.mark_map_stations,
            item_markers: difficulty.item_markers,
            all_items_spawn: difficulty.all_items_spawn,
            fast_elevators: difficulty.fast_elevators,
            fast_doors: difficulty.fast_doors,
            fast_pause_menu: difficulty.fast_pause_menu,
            objectives: difficulty.objectives,
            debug_options: difficulty.debug_options.clone(),
        };
        if Some(&new_difficulty) != out.last() {
            out.push(new_difficulty);
        }
    }
    out
}

fn get_item_priorities(item_priority_json: serde_json::Value) -> Vec<ItemPriorityGroup> {
    let mut priorities: IndexedVec<String> = IndexedVec::default();
    priorities.add("Early");
    priorities.add("Default");
    priorities.add("Late");

    let mut out: Vec<ItemPriorityGroup> = Vec::new();
    for priority in &priorities.keys {
        out.push(ItemPriorityGroup {
            name: priority.clone(),
            items: vec![],
        });
    }
    for (k, v) in item_priority_json.as_object().unwrap() {
        let i = priorities.index_by_key[v.as_str().unwrap()];
        out[i].items.push(k.to_string());
    }
    out
}

#[post("/randomize")]
async fn randomize(
    req: MultipartForm<RandomizeRequest>,
    http_req: HttpRequest,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let rom = Rom::new(req.rom.data.to_vec());

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

    let strat_json: serde_json::Value = serde_json::from_str(&req.strat_json).unwrap();
    let mut strat_vec: Vec<String> = app_data.ignored_notable_strats.iter().cloned().collect();
    for (strat, is_enabled) in strat_json.as_object().unwrap().iter() {
        if is_enabled.as_bool().unwrap() {
            strat_vec.push(strat.to_string());
        }
    }

    info!("raw json: {}", req.item_priority_json.0);
    let item_priority_json: serde_json::Value =
        serde_json::from_str(&req.item_priority_json.0).unwrap();

    let filler_items_json: serde_json::Value =
        serde_json::from_str(&req.filler_items_json.0).unwrap();
    let mut filler_items = vec![Item::Missile];
    filler_items.extend(
        filler_items_json
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_k, v)| v.as_str().unwrap() == "true")
            .map(|(k, _v)| Item::try_from(app_data.game_data.item_isv.index_by_key[k]).unwrap()),
    );
    info!("Filler items: {:?}", filler_items);

    let difficulty = DifficultyConfig {
        tech: tech_vec,
        notable_strats: strat_vec,
        shine_charge_tiles: req.shinespark_tiles.0,
        progression_rate: match req.progression_rate.0.as_str() {
            "Slow" => maprando::randomize::ProgressionRate::Slow,
            "Normal" => maprando::randomize::ProgressionRate::Normal,
            "Fast" => maprando::randomize::ProgressionRate::Fast,
            _ => panic!(
                "Unrecognized progression rate {}",
                req.progression_rate.0.as_str()
            ),
        },
        filler_items: filler_items,
        item_placement_style: match req.item_placement_style.0.as_str() {
            "Neutral" => maprando::randomize::ItemPlacementStyle::Neutral,
            "Forced" => maprando::randomize::ItemPlacementStyle::Forced,
            _ => panic!(
                "Unrecognized item placement style {}",
                req.item_placement_style.0.as_str()
            ),
        },
        item_priorities: get_item_priorities(item_priority_json),
        resource_multiplier: req.resource_multiplier.0,
        escape_timer_multiplier: req.escape_timer_multiplier.0,
        save_animals: req.save_animals.0,
        phantoon_proficiency: req.phantoon_proficiency.0,
        draygon_proficiency: req.draygon_proficiency.0,
        ridley_proficiency: req.ridley_proficiency.0,
        botwoon_proficiency: req.botwoon_proficiency.0,
        supers_double: req.supers_double.0,
        mother_brain_fight: match req.mother_brain_fight.0.as_str() {
            "Vanilla" => MotherBrainFight::Vanilla,
            "Short" => MotherBrainFight::Short,
            "Skip" => MotherBrainFight::Skip,
            _ => panic!(
                "Unrecognized mother_brain_fight: {}",
                req.mother_brain_fight.0
            ),
        },
        escape_enemies_cleared: req.escape_enemies_cleared.0,
        escape_movement_items: req.escape_movement_items.0,
        mark_map_stations: req.mark_map_stations.0,
        item_markers: match req.item_markers.0.as_str() {
            "Basic" => ItemMarkers::Basic,
            "Majors" => ItemMarkers::Majors,
            "Uniques" => ItemMarkers::Uniques,
            "3-Tiered" => ItemMarkers::ThreeTiered,
            _ => panic!("Unrecognized item_markers: {}", req.item_markers.0),
        },
        all_items_spawn: req.all_items_spawn.0,
        fast_elevators: req.fast_elevators.0,
        fast_doors: req.fast_doors.0,
        fast_pause_menu: req.fast_pause_menu.0,
        objectives: match req.objectives.0.as_str() {
            "Bosses" => Objectives::Bosses,
            "Minibosses" => Objectives::Minibosses,
            "Metroids" => Objectives::Metroids,
            _ => panic!("Unrecognized objectives: {}", req.objectives.0),
        },
        debug_options: if app_data.debug {
            Some(DebugOptions {
                new_game_extra: true,
                extended_spoiler: true,
            })
        } else {
            None
        },
    };
    let difficulty_tiers = if difficulty.item_placement_style == ItemPlacementStyle::Forced {
        get_difficulty_tiers(&difficulty, &app_data)
    } else {
        vec![difficulty.clone()]
    };
    let race_mode = req.race_mode.0 == "Yes";
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&random_seed.to_le_bytes());
    rng_seed[9] = if race_mode { 1 } else { 0 };
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);
    let max_attempts = 200;
    let mut attempt_num = 0;
    let randomization: Randomization;
    // info!(
    //     "Random seed={random_seed}, difficulty={:?}",
    //     difficulty_tiers[0]
    // );
    info!(
        "Random seed={random_seed}, difficulty={:?}",
        difficulty_tiers[0]
    );
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
        let randomizer = Randomizer::new(&map, &difficulty_tiers, &app_data.game_data);
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
        item_progression_preset: req.item_progression_preset.as_ref().map(|x| x.0.clone()),
        difficulty: difficulty_tiers[0].clone(),
        ignored_notable_strats: app_data.ignored_notable_strats.iter().cloned().collect(),
        quality_of_life_preset: req.quality_of_life_preset.as_ref().map(|x| x.0.clone()),
        supers_double: req.supers_double.0,
        mother_brain_fight: req.mother_brain_fight.0.clone(),
        escape_enemies_cleared: req.escape_enemies_cleared.0,
        escape_movement_items: req.escape_movement_items.0,
        mark_map_stations: req.mark_map_stations.0,
        item_markers: req.item_markers.0.clone(),
        all_items_spawn: req.all_items_spawn.0,
        fast_elevators: req.fast_elevators.0,
        fast_doors: req.fast_doors.0,
        fast_pause_menu: req.fast_pause_menu.0,
        objectives: req.objectives.0.clone(),
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

fn init_presets(
    presets: Vec<Preset>,
    game_data: &GameData,
    ignored_notable_strats: &HashSet<String>,
) -> Vec<PresetData> {
    let mut out: Vec<PresetData> = Vec::new();
    let mut cumulative_tech: HashSet<String> = HashSet::new();
    let mut cumulative_strats: HashSet<String> = HashSet::new();

    // Tech which is currently not used by any strat in logic, so we avoid showing on the website:
    let ignored_tech: HashSet<String> = [
        "canWallIceClip",
        "canGrappleClip",
        "canShinesparkWithReserve",
        "canRiskPermanentLossOfAccess",
        "canIceZebetitesSkip",
        "canSpeedZebetitesSkip",
        "canRemorphZebetiteSkip",
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    for tech in &ignored_tech {
        if !game_data.tech_isv.index_by_key.contains_key(tech) {
            panic!("Unrecognized ignored tech \"{tech}\"");
        }
    }

    let all_notable_strats: HashSet<String> = game_data
        .links
        .iter()
        .filter_map(|x| x.notable_strat_name.clone())
        .collect();
    if !ignored_notable_strats.is_subset(&all_notable_strats) {
        let diff: Vec<String> = ignored_notable_strats
            .difference(&all_notable_strats)
            .cloned()
            .collect();
        panic!("Unrecognized ignored notable strats: {:?}", diff);
    }

    let visible_tech: Vec<String> = game_data
        .tech_isv
        .keys
        .iter()
        .filter(|&x| !ignored_tech.contains(x))
        .cloned()
        .collect();
    let visible_tech_set: HashSet<String> = visible_tech.iter().cloned().collect();

    let visible_notable_strats: HashSet<String> = all_notable_strats
        .iter()
        .filter(|&x| !ignored_notable_strats.contains(x))
        .cloned()
        .collect();

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

        for strat_name in &preset.notable_strats {
            if cumulative_strats.contains(strat_name) {
                panic!("Notable strat \"{strat_name}\" appears in presets more than once.");
            }
            cumulative_strats.insert(strat_name.clone());
        }
        let mut notable_strat_setting: Vec<(String, bool)> = Vec::new();
        for strat_name in &visible_notable_strats {
            notable_strat_setting
                .push((strat_name.clone(), cumulative_strats.contains(strat_name)));
        }

        out.push(PresetData {
            preset: preset,
            tech_setting: tech_setting,
            notable_strat_setting: notable_strat_setting,
        });
    }
    for tech in &visible_tech_set {
        if !cumulative_tech.contains(tech) {
            panic!("Tech \"{tech}\" not found in any preset.");
        }
    }

    if !visible_notable_strats.is_subset(&cumulative_strats) {
        let diff: Vec<String> = visible_notable_strats
            .difference(&cumulative_strats)
            .cloned()
            .collect();
        panic!("Notable strats not found in any preset: {:?}", diff);
    }
    if !cumulative_strats.is_subset(&visible_notable_strats) {
        let diff: Vec<String> = cumulative_strats
            .difference(&visible_notable_strats)
            .cloned()
            .collect();
        panic!("Unrecognized notable strats in presets: {:?}", diff);
    }

    out
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    seed_repository_url: String,
    #[arg(long, action)]
    debug: bool,
    #[arg(long, action)]
    static_visualizer: bool,
}

fn get_ignored_notable_strats() -> HashSet<String> {
    [
        "Frozen Geemer Alcatraz Escape",
        "Suitless Botwoon Kill",
        "Maridia Bug Room Frozen Menu Bridge",
        "Breaking the Maridia Tube Gravity Jump",
        "Crumble Shaft Consecutive Crumble Climb",
        "Metroid Room 1 PB Dodge Kill (Left to Right)",
        "Metroid Room 1 PB Dodge Kill (Right to Left)",
        "Metroid Room 2 PB Dodge Kill (Bottom to Top)",
        "Metroid Room 3 PB Dodge Kill (Left to Right)",
        "Metroid Room 3 PB Dodge Kill (Right to Left)",
        "Metroid Room 4 Three PB Kill (Top to Bottom)",
        "Metroid Room 4 Six PB Dodge Kill (Bottom to Top)",
        "Metroid Room 4 Three PB Dodge Kill (Bottom to Top)",
        "Partial Covern Ice Clip",
        "Basement Speedball (Phantoon Dead)",
        "Basement Speedball (Phantoon Alive)",
        "MickeyMouse Crumbleless MidAir Spring Ball",
        "Mickey Mouse Crumble IBJ",
        "Botwoon Hallway Puyo Ice Clip",
    ]
    .iter()
    .map(|x| x.to_string())
    .collect()
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
    let args = Args::parse();
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let palette_path = Path::new("../palettes.json");
    let escape_timings_path = Path::new("data/escape_timings.json");
    let maps_path =
        Path::new("../maps/session-2022-06-03T17:19:29.727911.pkl-bk30-subarea-balance-2");

    let game_data = GameData::load(sm_json_data_path, room_geometry_path, palette_path, escape_timings_path).unwrap();
    let presets: Vec<Preset> =
        serde_json::from_str(&std::fs::read_to_string(&"data/presets.json").unwrap()).unwrap();
    let ignored_notable_strats = get_ignored_notable_strats();
    let preset_data = init_presets(presets, &game_data, &ignored_notable_strats);
    AppData {
        game_data,
        preset_data,
        ignored_notable_strats,
        map_repository: MapRepository::new(maps_path).unwrap(),
        seed_repository: SeedRepository::new(&args.seed_repository_url).unwrap(),
        visualizer_files: load_visualizer_files(),
        debug: args.debug,
        static_visualizer: args.static_visualizer,
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
