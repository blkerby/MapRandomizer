use std::path::Path;
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
use maprando::customize::{customize_rom, CustomizeSettings, MusicSettings};
use maprando::game_data::{GameData, IndexedVec, Item};
use maprando::patch::ips_write::create_ips_patch;
use maprando::patch::{make_rom, Rom};
use maprando::randomize::{
    DebugOptions, DifficultyConfig, ItemMarkers, ItemDotChange, ItemPlacementStyle, ItemPriorityGroup,
    MotherBrainFight, Objectives, Randomization, Randomizer,
};
use maprando::seed_repository::{Seed, SeedFile, SeedRepository};
use maprando::spoiler_map;
use maprando::web::{AppData, MapRepository, Preset, PresetData, SamusSpriteCategory};
use rand::{RngCore, SeedableRng};
use sailfish::TemplateOnce;
use serde_derive::{Deserialize, Serialize};

use maprando::web::logic::LogicData;
use maprando::web::VERSION;

const VISUALIZER_PATH: &'static str = "../visualizer/";
const TECH_GIF_PATH: &'static str = "static/tech_gifs/";
const NOTABLE_GIF_PATH: &'static str = "static/notable_gifs/";

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
#[template(path = "errors/room_not_found.stpl")]
struct RoomNotFoundTemplate {}

#[derive(TemplateOnce)]
#[template(path = "errors/file_not_found.stpl")]
struct FileNotFoundTemplate {}

#[derive(TemplateOnce)]
#[template(path = "errors/invalid_token.stpl")]
struct InvalidTokenTemplate {}

#[derive(TemplateOnce)]
#[template(path = "errors/already_unlocked.stpl")]
struct AlreadyUnlockedTemplate {}

#[derive(TemplateOnce)]
#[template(path = "home.stpl")]
struct HomeTemplate {
    version: usize,
}

#[derive(TemplateOnce)]
#[template(path = "releases.stpl")]
struct ReleasesTemplate {
    version: usize,
}

#[derive(TemplateOnce)]
#[template(path = "about.stpl")]
struct AboutTemplate {
    version: usize,
    sprite_artists: Vec<String>,
}

#[derive(TemplateOnce)]
#[template(path = "generate/main.stpl")]
struct GenerateTemplate<'a> {
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
    strat_description: &'a HashMap<String, String>,
    strat_id_by_name: &'a HashMap<String, usize>,
    tech_gif_listing: &'a HashSet<String>,
    notable_gif_listing: &'a HashSet<String>,
    tech_strat_counts: &'a HashMap<String, usize>,
}

#[get("/")]
async fn home(_app_data: web::Data<AppData>) -> impl Responder {
    let home_template = HomeTemplate { version: VERSION };
    HttpResponse::Ok().body(home_template.render_once().unwrap())
}

#[get("/releases")]
async fn releases(_app_data: web::Data<AppData>) -> impl Responder {
    let changes_template = ReleasesTemplate { version: VERSION };
    HttpResponse::Ok().body(changes_template.render_once().unwrap())
}

#[get("/about")]
async fn about(app_data: web::Data<AppData>) -> impl Responder {
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
    let about_template = AboutTemplate {
        version: VERSION,
        sprite_artists,
    };
    HttpResponse::Ok().body(about_template.render_once().unwrap())
}

#[get("/generate")]
async fn generate(app_data: web::Data<AppData>) -> impl Responder {
    let mut prioritizable_items: Vec<String> = app_data
        .game_data
        .item_isv
        .keys
        .iter()
        .cloned()
        .filter(|x| x != "Missile")
        .collect();
    prioritizable_items.sort();
    let generate_template = GenerateTemplate {
        version: VERSION,
        progression_rates: vec!["Fast", "Normal", "Slow"],
        item_placement_styles: vec!["Neutral", "Forced"],
        objectives: vec!["Bosses", "Minibosses", "Metroids", "Chozos", "Pirates"],
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
        strat_description: &app_data.game_data.strat_description,
        strat_id_by_name: &app_data.game_data.notable_strat_isv.index_by_key,
        tech_gif_listing: &app_data.tech_gif_listing,
        notable_gif_listing: &app_data.notable_gif_listing,
        tech_strat_counts: &app_data.logic_data.tech_strat_counts,
    };
    HttpResponse::Ok().body(generate_template.render_once().unwrap())
}

#[derive(MultipartForm)]
struct RandomizeRequest {
    rom: Bytes,
    preset: Option<Text<String>>,
    shinespark_tiles: Text<f32>,
    resource_multiplier: Text<f32>,
    gate_glitch_leniency: Text<i32>,
    phantoon_proficiency: Text<f32>,
    draygon_proficiency: Text<f32>,
    ridley_proficiency: Text<f32>,
    botwoon_proficiency: Text<f32>,
    escape_timer_multiplier: Text<f32>,
    tech_json: Text<String>,
    strat_json: Text<String>,
    progression_rate: Text<String>,
    item_placement_style: Text<String>,
    item_progression_preset: Option<Text<String>>,
    item_priority_json: Text<String>,
    filler_items_json: Text<String>,
    race_mode: Text<String>,
    random_seed: Text<String>,
    spoiler_token: Text<String>,
    quality_of_life_preset: Option<Text<String>>,
    supers_double: Text<bool>,
    mother_brain_fight: Text<String>,
    escape_enemies_cleared: Text<bool>,
    escape_refill: Text<bool>,
    escape_movement_items: Text<bool>,
    mark_map_stations: Text<bool>,
    transition_letters: Text<bool>,
    item_markers: Text<String>,
    item_dot_change: Text<String>,
    all_items_spawn: Text<bool>,
    acid_chozo: Text<bool>,
    fast_elevators: Text<bool>,
    fast_doors: Text<bool>,
    fast_pause_menu: Text<bool>,
    respin: Text<bool>,
    infinite_space_jump: Text<bool>,
    momentum_conservation: Text<bool>,
    objectives: Text<String>,
    randomized_start: Text<bool>,
    save_animals: Text<bool>,
    disable_walljump: Text<bool>,
    maps_revealed: Text<bool>,
    vanilla_map: Text<bool>,
    ultra_low_qol: Text<bool>,
}

#[derive(Deserialize)]
struct UnlockRequest {
    spoiler_token: String,
}

#[derive(MultipartForm)]
struct CustomizeRequest {
    rom: Bytes,
    custom_samus_sprite: Text<bool>,
    custom_etank_color: Text<bool>,
    samus_sprite: Text<String>,
    vanilla_screw_attack_animation: Text<bool>,
    room_palettes: Text<String>,
    music: Text<String>,
    disable_beeping: Text<bool>,
    etank_color: Text<String>,
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
    escape_refill: bool,
    escape_movement_items: bool,
    mark_map_stations: bool,
    transition_letters: bool,
    item_markers: String,
    item_dot_change: String,
    all_items_spawn: bool,
    acid_chozo: bool,
    fast_elevators: bool,
    fast_doors: bool,
    fast_pause_menu: bool,
    respin: bool,
    infinite_space_jump: bool,
    momentum_conservation: bool,
    objectives: String,
    disable_walljump: bool,
    maps_revealed: bool,
    vanilla_map: bool,
    ultra_low_qol: bool,
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
    early_filler_items: Vec<String>,
    item_placement_style: String,
    difficulty: &'a DifficultyConfig,
    notable_strats: Vec<String>,
    quality_of_life_preset: String,
    supers_double: bool,
    mother_brain_fight: String,
    escape_enemies_cleared: bool,
    escape_refill: bool,
    escape_movement_items: bool,
    mark_map_stations: bool,
    transition_letters: bool,
    item_markers: String,
    item_dot_change: String,
    all_items_spawn: bool,
    acid_chozo: bool,
    fast_elevators: bool,
    fast_doors: bool,
    fast_pause_menu: bool,
    respin: bool,
    infinite_space_jump: bool,
    momentum_conservation: bool,
    objectives: String,
    disable_walljump: bool,
    maps_revealed: bool,
    vanilla_map: bool,
    ultra_low_qol: bool,
    preset_data: &'a [PresetData],
    enabled_tech: HashSet<String>,
    enabled_notables: HashSet<String>,
}

#[derive(TemplateOnce)]
#[template(path = "seed/seed_footer.stpl")]
struct SeedFooterTemplate {
    race_mode: bool,
    all_items_spawn: bool,
    supers_double: bool,
    ultra_low_qol: bool,
}

#[derive(TemplateOnce)]
#[template(path = "seed/customize_seed.stpl")]
struct CustomizeSeedTemplate {
    version: usize,
    spoiler_token_prefix: String,
    unlocked_timestamp_str: String,
    seed_header: String,
    seed_footer: String,
    samus_sprite_categories: Vec<SamusSpriteCategory>,
    etank_colors: Vec<Vec<String>>,
}

fn render_seed(
    seed_name: &str,
    seed_data: &SeedData,
    app_data: &AppData,
) -> Result<(String, String)> {
    let ignored_notable_strats: HashSet<String> =
        seed_data.ignored_notable_strats.iter().cloned().collect();
    let notable_strats: Vec<String> = seed_data
        .difficulty
        .notable_strats
        .iter()
        .cloned()
        .filter(|x| !ignored_notable_strats.contains(x))
        .collect();
    let enabled_tech: HashSet<String> = seed_data.difficulty.tech.iter().cloned().collect();
    let enabled_notables: HashSet<String> = seed_data
        .difficulty
        .notable_strats
        .iter()
        .cloned()
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
        early_filler_items: seed_data
            .difficulty
            .early_filler_items
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
        escape_refill: seed_data.escape_refill,
        escape_movement_items: seed_data.escape_movement_items,
        mark_map_stations: seed_data.mark_map_stations,
        item_markers: seed_data.item_markers.clone(),
        item_dot_change: seed_data.item_dot_change.clone(),
        transition_letters: seed_data.transition_letters,
        all_items_spawn: seed_data.all_items_spawn,
        acid_chozo: seed_data.acid_chozo,
        fast_elevators: seed_data.fast_elevators,
        fast_doors: seed_data.fast_doors,
        fast_pause_menu: seed_data.fast_pause_menu,
        respin: seed_data.respin,
        infinite_space_jump: seed_data.infinite_space_jump,
        momentum_conservation: seed_data.momentum_conservation,
        objectives: seed_data.objectives.clone(),
        disable_walljump: seed_data.disable_walljump,
        maps_revealed: seed_data.maps_revealed,
        vanilla_map: seed_data.vanilla_map,
        ultra_low_qol: seed_data.ultra_low_qol,
        preset_data: &app_data.preset_data,
        enabled_tech,
        enabled_notables,
    };
    let seed_header_html = seed_header_template.render_once()?;

    let seed_footer_template = SeedFooterTemplate {
        race_mode: seed_data.race_mode,
        all_items_spawn: seed_data.all_items_spawn,
        supers_double: seed_data.supers_double,
        ultra_low_qol: seed_data.ultra_low_qol,
    };
    let seed_footer_html = seed_footer_template.render_once()?;
    Ok((seed_header_html, seed_footer_html))
}

async fn save_seed(
    seed_name: &str,
    seed_data: &SeedData,
    spoiler_token: &str,
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
    let (seed_header_html, seed_footer_html) = render_seed(seed_name, seed_data, app_data)?;
    files.push(SeedFile::new(
        "seed_header.html",
        seed_header_html.into_bytes(),
    ));
    files.push(SeedFile::new(
        "seed_footer.html",
        seed_footer_html.into_bytes(),
    ));

    let prefix = if seed_data.race_mode {
        "locked"
    } else {
        "public"
    };

    if seed_data.race_mode {
        files.push(SeedFile::new(
            "spoiler_token.txt",
            spoiler_token.as_bytes().to_vec(),
        ));
    }

    // Write the spoiler log
    let spoiler_bytes = serde_json::to_vec_pretty(&randomization.spoiler_log).unwrap();
    files.push(SeedFile::new(
        &format!("{}/spoiler.json", prefix),
        spoiler_bytes,
    ));

    // Write the spoiler maps
    let spoiler_maps =
        spoiler_map::get_spoiler_map(&output_rom, &randomization.map, &app_data.game_data).unwrap();
    files.push(SeedFile::new(
        &format!("{}/map-assigned.png", prefix),
        spoiler_maps.assigned,
    ));
    files.push(SeedFile::new(
        &format!("{}/map-vanilla.png", prefix),
        spoiler_maps.vanilla,
    ));
    files.push(SeedFile::new(
        &format!("{}/map-grid.png", prefix),
        spoiler_maps.grid,
    ));

    // Write the spoiler visualizer
    for (filename, data) in &app_data.visualizer_files {
        let path = format!("{}/visualizer/{}", prefix, filename);
        files.push(SeedFile::new(&path, data.clone()));
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

#[get("/logic")]
async fn logic(app_data: web::Data<AppData>) -> impl Responder {
    HttpResponse::Ok().body(app_data.logic_data.index_html.clone())
}

#[get("/logic/room/{name}")]
async fn logic_room(info: web::Path<(String,)>, app_data: web::Data<AppData>) -> impl Responder {
    let room_name = &info.0;
    if let Some(html) = app_data.logic_data.room_html.get(room_name) {
        HttpResponse::Ok().body(html.clone())
    } else {
        let template = RoomNotFoundTemplate {};
        HttpResponse::NotFound().body(template.render_once().unwrap())
    }
}

#[get("/logic/room/{room_name}/{from_node}/{to_node}/{strat_name}")]
async fn logic_strat(
    info: web::Path<(String, usize, usize, String)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let room_name = &info.0;
    let from_node = info.1;
    let to_node = info.2;
    let strat_name = &info.3;
    if let Some(html) = app_data.logic_data.strat_html.get(&(
        room_name.clone(),
        from_node,
        to_node,
        strat_name.clone(),
    )) {
        HttpResponse::Ok().body(html.clone())
    } else {
        let template = RoomNotFoundTemplate {};
        HttpResponse::NotFound().body(template.render_once().unwrap())
    }
}

#[get("/logic/tech/{name}")]
async fn logic_tech(info: web::Path<(String,)>, app_data: web::Data<AppData>) -> impl Responder {
    let tech_name = &info.0;
    if let Some(html) = app_data.logic_data.tech_html.get(tech_name) {
        HttpResponse::Ok().body(html.clone())
    } else {
        let template = RoomNotFoundTemplate {};
        HttpResponse::NotFound().body(template.render_once().unwrap())
    }
}

#[get("/seed/{name}/")]
async fn view_seed(info: web::Path<(String,)>, app_data: web::Data<AppData>) -> impl Responder {
    let seed_name = &info.0;
    let (seed_header, seed_footer, unlocked_timestamp_str, spoiler_token) = futures::join!(
        app_data
            .seed_repository
            .get_file(seed_name, "seed_header.html"),
        app_data
            .seed_repository
            .get_file(seed_name, "seed_footer.html"),
        app_data
            .seed_repository
            .get_file(seed_name, "unlocked_timestamp.txt"),
        app_data
            .seed_repository
            .get_file(seed_name, "spoiler_token.txt"),
    );
    let spoiler_token = String::from_utf8(spoiler_token.unwrap_or(vec![])).unwrap();
    let spoiler_token_prefix = if spoiler_token.is_empty() {
        "".to_string()
    } else {
        spoiler_token[0..16].to_string()
    };
    match (seed_header, seed_footer) {
        (Ok(header), Ok(footer)) => {
            let customize_template = CustomizeSeedTemplate {
                version: VERSION,
                unlocked_timestamp_str: String::from_utf8(unlocked_timestamp_str.unwrap_or(vec![]))
                    .unwrap(),
                spoiler_token_prefix: spoiler_token_prefix.to_string(),
                seed_header: String::from_utf8(header.to_vec()).unwrap(),
                seed_footer: String::from_utf8(footer.to_vec()).unwrap(),
                samus_sprite_categories: app_data.samus_sprite_categories.clone(),
                etank_colors: app_data.etank_colors.clone(),
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

#[post("/seed/{name}/unlock")]
async fn unlock_seed(
    req: web::Form<UnlockRequest>,
    info: web::Path<(String,)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let seed_name = &info.0;
    let seed_spoiler_token = app_data
        .seed_repository
        .get_file(seed_name, "spoiler_token.txt")
        .await
        .unwrap();

    if req.spoiler_token.as_bytes() == seed_spoiler_token {
        let unlocked_timestamp_data = app_data
            .seed_repository
            .get_file(seed_name, "unlocked_timestamp.txt")
            .await;
        if unlocked_timestamp_data.is_ok() {
            // TODO: handle other errors that are not 404.
            let template = AlreadyUnlockedTemplate {};
            return HttpResponse::UnprocessableEntity().body(template.render_once().unwrap());
        }

        app_data
            .seed_repository
            .move_prefix(seed_name, "locked", "public")
            .await
            .unwrap();
        let timestamp = match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            Ok(n) => n.as_millis() as usize,
            Err(_) => panic!("SystemTime before UNIX EPOCH!"),
        };
        let unlock_time_str = format!("{}", timestamp);
        app_data
            .seed_repository
            .put_file(
                seed_name,
                "unlocked_timestamp.txt".to_string(),
                unlock_time_str.into_bytes(),
            )
            .await
            .unwrap();
    } else {
        let template = InvalidTokenTemplate {};
        return HttpResponse::Forbidden().body(template.render_once().unwrap());
    }
    HttpResponse::Found()
        .insert_header((header::LOCATION, format!("/seed/{}/", info.0)))
        .finish()
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
        samus_sprite: if req.custom_samus_sprite.0 && req.samus_sprite.0 != "" {
            Some(req.samus_sprite.0.clone())
        } else {
            None
        },
        vanilla_screw_attack_animation: req.vanilla_screw_attack_animation.0,
        area_themed_palette: req.room_palettes.0 == "area-themed",
        music: match req.music.0.as_str() {
            "vanilla" => MusicSettings::Vanilla,
            "area" => MusicSettings::AreaThemed,
            "disabled" => MusicSettings::Disabled,
            _ => panic!("Unexpected music option: {}", req.music.0.as_str()),
        },
        disable_beeping: req.disable_beeping.0,
        etank_color: if req.custom_etank_color.0 { Some((
            u8::from_str_radix(&req.etank_color.0[0..2], 16).unwrap() / 8,
            u8::from_str_radix(&req.etank_color.0[2..4], 16).unwrap() / 8,
            u8::from_str_radix(&req.etank_color.0[4..6], 16).unwrap() / 8,
        )) } else { None },
    };
    info!("CustomizeSettings: {:?}", settings);
    match customize_rom(
        &mut rom,
        &patch_ips,
        &settings,
        &app_data.game_data,
        &app_data.samus_sprite_categories,
    ) {
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

    let data_result: Result<Vec<u8>> = if filename.starts_with("visualizer/")
        && app_data.static_visualizer
    {
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
    out.last_mut().unwrap().tech.sort();
    out.last_mut().unwrap().notable_strats.sort();
    for preset_data in presets.iter().rev() {
        let preset = &preset_data.preset;
        let mut tech_vec: Vec<String> = Vec::new();
        for (tech, enabled) in &preset_data.tech_setting {
            if *enabled && tech_set.contains(tech) {
                tech_vec.push(tech.clone());
            }
        }
        tech_vec.sort();

        let mut strat_vec: Vec<String> = vec![]; //= app_data.ignored_notable_strats.iter().cloned().collect();
        for (strat, enabled) in &preset_data.notable_strat_setting {
            if *enabled && strat_set.contains(strat) {
                strat_vec.push(strat.clone());
            }
        }
        strat_vec.sort();

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
            early_filler_items: difficulty.early_filler_items.clone(),
            resource_multiplier: f32::max(
                difficulty.resource_multiplier,
                preset.resource_multiplier,
            ),
            gate_glitch_leniency: i32::max(
                difficulty.gate_glitch_leniency,
                preset.gate_glitch_leniency as i32,
            ),
            escape_timer_multiplier: difficulty.escape_timer_multiplier,
            randomized_start: difficulty.randomized_start,
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
            escape_refill: difficulty.escape_refill,
            escape_movement_items: difficulty.escape_movement_items,
            mark_map_stations: difficulty.mark_map_stations,
            transition_letters: difficulty.transition_letters,
            item_markers: difficulty.item_markers,
            item_dot_change: difficulty.item_dot_change,
            all_items_spawn: difficulty.all_items_spawn,
            acid_chozo: difficulty.acid_chozo,
            fast_elevators: difficulty.fast_elevators,
            fast_doors: difficulty.fast_doors,
            fast_pause_menu: difficulty.fast_pause_menu,
            respin: difficulty.respin,
            infinite_space_jump: difficulty.infinite_space_jump,
            momentum_conservation: difficulty.momentum_conservation,
            objectives: difficulty.objectives,
            disable_walljump: difficulty.disable_walljump,
            maps_revealed: difficulty.maps_revealed,
            vanilla_map: difficulty.vanilla_map,
            ultra_low_qol: difficulty.ultra_low_qol,
            skill_assumptions_preset: difficulty.skill_assumptions_preset.clone(),
            item_progression_preset: difficulty.item_progression_preset.clone(),
            quality_of_life_preset: difficulty.quality_of_life_preset.clone(),
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

    let race_mode = req.race_mode.0 == "Yes";
    let random_seed = if &req.random_seed.0 == "" || race_mode {
        get_random_seed()
    } else {
        match req.random_seed.0.parse::<usize>() {
            Ok(x) => x,
            Err(_) => {
                return HttpResponse::BadRequest().body("Invalid random seed");
            }
        }
    };
    let display_seed = if race_mode {
        get_random_seed()
    } else {
        random_seed
    };

    if req.ridley_proficiency.0 < 0.0 || req.ridley_proficiency.0 > 1.0 {
        return HttpResponse::BadRequest().body("Invalid Ridley proficiency");
    }

    let tech_json: serde_json::Value = serde_json::from_str(&req.tech_json).unwrap();
    let mut tech_vec: Vec<String> = app_data.implicit_tech.iter().cloned().collect();
    let walljump_tech = "canWalljump";
    assert!(tech_json.as_object().unwrap().contains_key(walljump_tech));
    for (tech, is_enabled) in tech_json.as_object().unwrap().iter() {
        if tech == walljump_tech && req.disable_walljump.0 {
            continue;
        }
        if is_enabled.as_bool().unwrap() {
            tech_vec.push(tech.to_string());
        }
    }
    if req.vanilla_map.0 {
        tech_vec.push("canEscapeMorphLocation".to_string());
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
            .filter(|(_k, v)| v.as_str().unwrap() != "No")
            .map(|(k, _v)| Item::try_from(app_data.game_data.item_isv.index_by_key[k]).unwrap()),
    );
    let early_filler_items: Vec<Item> = filler_items_json
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_k, v)| v.as_str().unwrap() == "Early")
        .map(|(k, _v)| Item::try_from(app_data.game_data.item_isv.index_by_key[k]).unwrap())
        .collect();

    info!("Filler items: {:?}", filler_items);
    info!("Early filler items: {:?}", early_filler_items);

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
        early_filler_items: early_filler_items,
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
        gate_glitch_leniency: req.gate_glitch_leniency.0,
        randomized_start: req.randomized_start.0,
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
        escape_refill: req.escape_refill.0,
        escape_movement_items: req.escape_movement_items.0,
        mark_map_stations: req.mark_map_stations.0,
        transition_letters: req.transition_letters.0,
        item_markers: match req.item_markers.0.as_str() {
            "Simple" => ItemMarkers::Simple,
            "Majors" => ItemMarkers::Majors,
            "Uniques" => ItemMarkers::Uniques,
            "3-Tiered" => ItemMarkers::ThreeTiered,
            _ => panic!("Unrecognized item_markers: {}", req.item_markers.0),
        },
        item_dot_change: match req.item_dot_change.0.as_str() {
            "Fade" => ItemDotChange::Fade,
            "Disappear" => ItemDotChange::Disappear,
            _ => panic!("Unrecognized item_dot_change: {}", req.item_dot_change.0),
        },
        all_items_spawn: req.all_items_spawn.0,
        acid_chozo: req.acid_chozo.0,
        fast_elevators: req.fast_elevators.0,
        fast_doors: req.fast_doors.0,
        fast_pause_menu: req.fast_pause_menu.0,
        respin: req.respin.0,
        infinite_space_jump: req.infinite_space_jump.0,
        momentum_conservation: req.momentum_conservation.0,
        objectives: match req.objectives.0.as_str() {
            "Bosses" => Objectives::Bosses,
            "Minibosses" => Objectives::Minibosses,
            "Metroids" => Objectives::Metroids,
            "Chozos" => Objectives::Chozos,
            "Pirates" => Objectives::Pirates,
            _ => panic!("Unrecognized objectives: {}", req.objectives.0),
        },
        disable_walljump: req.disable_walljump.0,
        maps_revealed: req.maps_revealed.0,
        vanilla_map: req.vanilla_map.0,
        ultra_low_qol: req.ultra_low_qol.0,
        skill_assumptions_preset: req.preset.as_ref().map(|x| x.0.clone()),
        item_progression_preset: req.item_progression_preset.as_ref().map(|x| x.0.clone()),
        quality_of_life_preset: req.quality_of_life_preset.as_ref().map(|x| x.0.clone()),
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
        let map = if difficulty.vanilla_map {
            app_data.map_repository.get_vanilla_map().unwrap()
        } else {
            app_data.map_repository.get_map(map_seed).unwrap()
        };
        item_placement_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        info!("Map seed={map_seed}, item placement seed={item_placement_seed}");
        let randomizer = Randomizer::new(&map, &difficulty_tiers, &app_data.game_data);
        randomization = match randomizer.randomize(item_placement_seed, display_seed) {
            Ok(r) => r,
            Err(e) => {
                info!("Failed randomization: {}", e);
                continue;
            }
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
        escape_refill: req.escape_refill.0,
        escape_movement_items: req.escape_movement_items.0,
        mark_map_stations: req.mark_map_stations.0,
        transition_letters: req.transition_letters.0,
        item_markers: req.item_markers.0.clone(),
        item_dot_change: req.item_dot_change.0.clone(),
        all_items_spawn: req.all_items_spawn.0,
        acid_chozo: req.acid_chozo.0,
        fast_elevators: req.fast_elevators.0,
        fast_doors: req.fast_doors.0,
        fast_pause_menu: req.fast_pause_menu.0,
        respin: req.respin.0,
        infinite_space_jump: req.infinite_space_jump.0,
        momentum_conservation: req.momentum_conservation.0,
        objectives: req.objectives.0.clone(),
        disable_walljump: req.disable_walljump.0,
        maps_revealed: req.maps_revealed.0,
        vanilla_map: req.vanilla_map.0,
        ultra_low_qol: req.ultra_low_qol.0,
    };

    let output_rom = make_rom(&rom, &randomization, &app_data.game_data).unwrap();

    let seed_name = get_seed_name(&seed_data);
    save_seed(
        &seed_name,
        &seed_data,
        &req.spoiler_token.0,
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
    implicit_tech: &HashSet<String>,
) -> Vec<PresetData> {
    let mut out: Vec<PresetData> = Vec::new();
    let mut cumulative_tech: HashSet<String> = HashSet::new();
    let mut cumulative_strats: HashSet<String> = HashSet::new();

    // Tech which is currently not used by any strat in logic, so we avoid showing on the website:
    let ignored_tech: HashSet<String> = [
        "canGrappleClip",
        "canShinesparkWithReserve",
        "canRiskPermanentLossOfAccess",
        "canIceZebetitesSkip",
        "canSpeedZebetitesSkip",
        "canRemorphZebetiteSkip",
        "canEscapeMorphLocation", // Special internal tech for "vanilla map" option
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    for tech in &ignored_tech {
        if !game_data.tech_isv.index_by_key.contains_key(tech) {
            panic!("Unrecognized ignored tech \"{tech}\"");
        }
    }
    for tech in implicit_tech {
        if !game_data.tech_isv.index_by_key.contains_key(tech) {
            panic!("Unrecognized implicit tech \"{tech}\"");
        }
        if ignored_tech.contains(tech) {
            panic!("Tech is both ignored and implicit: \"{tech}\"");
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
        .filter(|&x| !ignored_tech.contains(x) && !implicit_tech.contains(x))
        .cloned()
        .collect();
    let visible_tech_set: HashSet<String> = visible_tech.iter().cloned().collect();

    let visible_notable_strats: HashSet<String> = all_notable_strats
        .iter()
        .filter(|&x| !ignored_notable_strats.contains(x))
        .cloned()
        .collect();

    cumulative_tech.extend(implicit_tech.iter().cloned());
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
        for tech in implicit_tech {
            tech_setting.push((tech.clone(), true));
        }
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
            implicit_tech: implicit_tech.clone(),
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
        "Suitless Botwoon Kill",
        "Maridia Bug Room Frozen Menu Bridge",
        "Breaking the Maridia Tube Gravity Jump",
        "Metroid Room 1 PB Dodge Kill (Left to Right)",
        "Metroid Room 1 PB Dodge Kill (Right to Left)",
        "Metroid Room 2 PB Dodge Kill (Bottom to Top)",
        "Metroid Room 3 PB Dodge Kill (Left to Right)",
        "Metroid Room 3 PB Dodge Kill (Right to Left)",
        "Metroid Room 4 Three PB Kill (Top to Bottom)",
        "Metroid Room 4 Six PB Dodge Kill (Bottom to Top)",
        "Metroid Room 4 Three PB Dodge Kill (Bottom to Top)",
        "Partial Covern Ice Clip",
        "Mickey Mouse Crumble Jump IBJ",
        "G-Mode Morph Breaking the Maridia Tube Gravity Jump", // not usable because of canRiskPermanentLossOfAccess
        "Mt. Everest Cross Room Jump through Top Door", // currently unusable because of obstacleCleared requirement
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

fn list_tech_gif_files() -> HashSet<String> {
    let mut files: HashSet<String> = HashSet::new();
    for entry_res in std::fs::read_dir(TECH_GIF_PATH).unwrap() {
        let entry = entry_res.unwrap();
        let name = entry.file_name().to_str().unwrap().to_string();
        files.insert(name);
    }
    files
}

fn list_notable_gif_files() -> HashSet<String> {
    let mut files: HashSet<String> = HashSet::new();
    for entry_res in std::fs::read_dir(NOTABLE_GIF_PATH).unwrap() {
        let entry = entry_res.unwrap();
        let name = entry.file_name().to_str().unwrap().to_string();
        files.insert(name);
    }
    files
}

fn get_implicit_tech() -> HashSet<String> {
    [
        "canPseudoScrew",
        "canMidAirMorph",
        "canTurnaroundSpinJump",
        "canCameraManip",
        "canStopOnADime",
        "canUseGrapple",
        "canDisableEquipment",
        "canEscapeEnemyGrab",
        "canSpecialBeamAttack",
        "canDownBack",
    ]
    .into_iter()
    .map(|x| x.to_string())
    .collect()
}

fn build_app_data() -> AppData {
    let args = Args::parse();
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let palette_path = Path::new("../palettes.json");
    let escape_timings_path = Path::new("data/escape_timings.json");
    let start_locations_path = Path::new("data/start_locations.json");
    let hub_locations_path = Path::new("data/hub_locations.json");
    let etank_colors_path = Path::new("data/etank_colors.json");
    let maps_path =
        Path::new("../maps/session-2023-06-08T14:55:16.779895.pkl-bk24-subarea-balance-2");
    let samus_sprites_path = Path::new("../MapRandoSprites/samus_sprites/manifest.json");
    // let samus_spritesheet_layout_path = Path::new("data/samus_spritesheet_layout.json");

    let game_data = GameData::load(
        sm_json_data_path,
        room_geometry_path,
        palette_path,
        escape_timings_path,
        start_locations_path,
        hub_locations_path,
    )
    .unwrap();
    // let samus_customizer = SamusSpriteCustomizer::new(samus_spritesheet_layout_path).unwrap();
    let tech_gif_listing = list_tech_gif_files();
    let notable_gif_listing = list_notable_gif_files();
    let presets: Vec<Preset> =
        serde_json::from_str(&std::fs::read_to_string(&"data/presets.json").unwrap()).unwrap();
    let etank_colors: Vec<Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string(&etank_colors_path).unwrap()).unwrap();
    let ignored_notable_strats = get_ignored_notable_strats();
    let implicit_tech = get_implicit_tech();
    let preset_data = init_presets(presets, &game_data, &ignored_notable_strats, &implicit_tech);
    let logic_data = LogicData::new(&game_data, &tech_gif_listing, &preset_data);
    let samus_sprite_categories: Vec<SamusSpriteCategory> =
        serde_json::from_str(&std::fs::read_to_string(&samus_sprites_path).unwrap()).unwrap();
    AppData {
        game_data,
        preset_data,
        ignored_notable_strats,
        implicit_tech,
        map_repository: MapRepository::new(maps_path).unwrap(),
        seed_repository: SeedRepository::new(&args.seed_repository_url).unwrap(),
        visualizer_files: load_visualizer_files(),
        tech_gif_listing,
        notable_gif_listing,
        logic_data,
        samus_sprite_categories,
        // samus_customizer,
        debug: args.debug,
        static_visualizer: args.static_visualizer,
        etank_colors,
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
            .service(releases)
            .service(generate)
            .service(randomize)
            .service(about)
            .service(view_seed)
            .service(get_seed_file)
            .service(customize_seed)
            .service(unlock_seed)
            .service(view_seed_redirect)
            .service(logic)
            .service(logic_room)
            .service(logic_strat)
            .service(logic_tech)
            .service(actix_files::Files::new("/static", "static"))
    })
    .bind("0.0.0.0:8080")
    .unwrap()
    .run()
    .await
    .unwrap();
}
