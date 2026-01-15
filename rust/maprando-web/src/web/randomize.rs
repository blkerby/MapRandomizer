mod helpers;

use crate::web::{AppData, VERSION};
use actix_easy_multipart::{MultipartForm, text::Text};
use actix_web::{HttpRequest, HttpResponse, Responder, post, web};
use helpers::*;
use log::info;
use maprando::{
    randomize::{
        DifficultyConfig, Randomization, Randomizer, assign_map_areas, filter_links,
        get_difficulty_tiers, get_objectives, randomize_doors,
    },
    settings::{RandomizerSettings, StartLocationMode, try_upgrade_settings},
    spoiler_log::SpoilerLog,
};
use maprando_game::{LinksDataGroup, Map};
use rand::{RngCore, SeedableRng};
use serde_derive::{Deserialize, Serialize};
use serde_variant::to_variant_name;
use std::time::{Duration, Instant, SystemTime};

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
    doors: String,
    start_location_mode: String,
    map_layout: String,
    save_animals: String,
    early_save: bool,
    area_assignment: String,
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
    app_data: web::Data<AppData>,
) -> Result<AttemptOutput, AttemptError> {
    let skill_settings = &settings.skill_assumption_settings;
    let race_mode = settings.other_settings.race_mode;
    let random_seed = if settings.other_settings.random_seed.is_none() || race_mode {
        get_random_seed()
    } else {
        settings.other_settings.random_seed.unwrap()
    };
    let display_seed = if race_mode {
        get_random_seed()
    } else {
        random_seed
    };
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&random_seed.to_le_bytes());
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let implicit_tech = &app_data.preset_data.tech_by_difficulty["Implicit"];
    let implicit_notables = &app_data.preset_data.notables_by_difficulty["Implicit"];
    let difficulty = DifficultyConfig::new(
        skill_settings,
        &app_data.game_data,
        implicit_tech,
        implicit_notables,
    );
    let difficulty_tiers = get_difficulty_tiers(
        &settings,
        &app_data.preset_data.difficulty_tiers,
        &app_data.game_data,
        &app_data.preset_data.tech_by_difficulty["Implicit"],
        &app_data.preset_data.notables_by_difficulty["Implicit"],
    );

    let filtered_base_links =
        filter_links(&app_data.game_data.links, &app_data.game_data, &difficulty);
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
        assign_map_areas(&mut map, &settings, map_seed, &app_data.game_data);
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
                randomizer.randomize(attempt_num, item_placement_seed, display_seed);
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
    app_data: web::Data<AppData>,
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
        doors: to_variant_name(&settings.doors_mode).unwrap().to_string(),
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
        area_assignment: to_variant_name(&settings.other_settings.area_assignment)
            .unwrap()
            .to_string(),
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
