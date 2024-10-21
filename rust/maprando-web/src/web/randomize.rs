mod helpers;

use crate::web::{AppData, VERSION};
use actix_easy_multipart::{bytes::Bytes, text::Text, MultipartForm};
use actix_web::{post, web, HttpRequest, HttpResponse, Responder};
use askama::Template;
use helpers::*;
use log::info;
use maprando::{
    patch::{make_rom, Rom},
    randomize::{
        filter_links, randomize_doors, randomize_map_areas, DifficultyConfig, Objective,
        Randomization, Randomizer,
    },
    settings::{
        parse_randomizer_settings, AreaAssignment, FillerItemPriority, ItemPlacementStyle,
        ObjectivesMode, RandomizerSettings, StartLocationMode, WallJump,
    },
};
use maprando_game::{
    Capacity, Item, LinksDataGroup, NotableId, RoomId, TechId, TECH_ID_CAN_ESCAPE_MORPH_LOCATION,
};
use rand::{RngCore, SeedableRng};
use serde_derive::{Deserialize, Serialize};
use serde_variant::to_variant_name;
use std::time::{Instant, SystemTime};

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
    mark_map_stations: bool,
    transition_letters: bool,
    item_markers: String,
    item_dot_change: String,
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
    objectives: String,
    doors: String,
    start_location_mode: String,
    map_layout: String,
    save_animals: String,
    early_save: bool,
    area_assignment: String,
    wall_jump: String,
    etank_refill: String,
    maps_revealed: String,
    vanilla_map: bool,
    ultra_low_qol: bool,
}

#[derive(Template)]
#[template(path = "errors/missing_input_rom.html")]
struct MissingInputRomTemplate {}

#[derive(Template)]
#[template(path = "errors/invalid_rom.html")]
struct InvalidRomTemplate {}

#[derive(MultipartForm)]
struct RandomizeRequest {
    rom: Bytes,
    spoiler_token: Text<String>,
    settings: Text<String>,
}

#[derive(Serialize)]
struct RandomizeResponse {
    seed_url: String,
}

#[post("/randomize")]
async fn randomize(
    req: MultipartForm<RandomizeRequest>,
    http_req: HttpRequest,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let rom = Rom::new(req.rom.data.to_vec());

    if rom.data.len() == 0 {
        return HttpResponse::BadRequest().body(MissingInputRomTemplate {}.render().unwrap());
    }

    let rom_digest = crypto_hash::hex_digest(crypto_hash::Algorithm::SHA256, &rom.data);
    info!("Rom digest: {rom_digest}");
    if rom_digest != "12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72" {
        return HttpResponse::BadRequest().body(InvalidRomTemplate {}.render().unwrap());
    }

    let settings: RandomizerSettings = match parse_randomizer_settings(&req.settings.0) {
        Ok(s) => s,
        Err(e) => {
            return HttpResponse::BadRequest().body(e.to_string());
        }
    };
    let skill_settings = &settings.skill_assumption_settings;
    let item_settings = &settings.item_progression_settings;
    let qol_settings = &settings.quality_of_life_settings;
    let other_settings = &settings.other_settings;
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

    if skill_settings.ridley_proficiency < 0.0 || skill_settings.ridley_proficiency > 1.0 {
        return HttpResponse::BadRequest().body("Invalid Ridley proficiency");
    }

    let mut tech_vec: Vec<TechId> = vec![];
    for tech in &app_data.preset_data[0].preset.tech {
        // Include implicit tech (which is in the first preset):
        tech_vec.push(tech.tech_id);
    }
    for tech_setting in skill_settings.tech_settings.iter() {
        if tech_setting.enabled {
            tech_vec.push(tech_setting.id);
        }
    }

    let vanilla_map = settings.map_layout == "Vanilla";
    if vanilla_map {
        tech_vec.push(TECH_ID_CAN_ESCAPE_MORPH_LOCATION);
    }

    let mut notable_vec: Vec<(RoomId, NotableId)> = vec![];
    for notable in &app_data.preset_data[0].preset.notables {
        // Include implicit notables (which are in the first preset):
        notable_vec.push((notable.room_id, notable.notable_id));
    }
    for notable_setting in skill_settings.notable_settings.iter() {
        if notable_setting.enabled {
            notable_vec.push((notable_setting.room_id, notable_setting.notable_id));
        }
    }

    let semi_filler_items: Vec<Item> = item_settings
        .filler_items
        .iter()
        .filter(|(_k, &v)| v == FillerItemPriority::Semi)
        .map(|(k, _v)| *k)
        .collect();
    let mut filler_items = vec![Item::Missile, Item::Nothing];
    filler_items.extend(
        item_settings
            .filler_items
            .iter()
            .filter(|(_k, &v)| v == FillerItemPriority::Yes || v == FillerItemPriority::Early)
            .map(|(k, _v)| *k),
    );
    let early_filler_items: Vec<Item> = item_settings
        .filler_items
        .iter()
        .filter(|(_k, &v)| v == FillerItemPriority::Early)
        .map(|(k, _v)| *k)
        .collect();

    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&random_seed.to_le_bytes());
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let difficulty = DifficultyConfig {
        name: Some(
            skill_settings
                .preset
                .as_ref()
                .map(|x| x.to_string())
                .unwrap_or("Beyond".to_string()),
        ),
        tech: tech_vec,
        notables: notable_vec,
        shine_charge_tiles: skill_settings.shinespark_tiles,
        heated_shine_charge_tiles: skill_settings.heated_shinespark_tiles,
        speed_ball_tiles: skill_settings.speed_ball_tiles,
        shinecharge_leniency_frames: skill_settings.shinecharge_leniency_frames as Capacity,
        progression_rate: item_settings.progression_rate,
        item_priority_strength: item_settings.item_priority_strength,
        random_tank: item_settings.random_tank,
        spazer_before_plasma: item_settings.spazer_before_plasma,
        stop_item_placement_early: item_settings.stop_item_placement_early,
        item_pool: item_settings.item_pool.iter().map(|(&x, &y)| (x, y)).collect(),
        starting_items: item_settings.starting_items.iter().map(|(&x, &y)| (x, y)).collect(),
        filler_items,
        semi_filler_items,
        early_filler_items,
        item_placement_style: item_settings.item_placement_style,
        item_priorities: get_item_priorities(&item_settings.key_item_priority),
        resource_multiplier: skill_settings.resource_multiplier,
        escape_timer_multiplier: skill_settings.escape_timer_multiplier,
        gate_glitch_leniency: skill_settings.gate_glitch_leniency as Capacity,
        door_stuck_leniency: skill_settings.door_stuck_leniency as Capacity,
        start_location_mode: settings.start_location_mode,
        save_animals: settings.save_animals,
        phantoon_proficiency: skill_settings.phantoon_proficiency,
        draygon_proficiency: skill_settings.draygon_proficiency,
        ridley_proficiency: skill_settings.ridley_proficiency,
        botwoon_proficiency: skill_settings.botwoon_proficiency,
        mother_brain_proficiency: skill_settings.mother_brain_proficiency,
        supers_double: qol_settings.supers_double,
        mother_brain_fight: qol_settings.mother_brain_fight,
        escape_enemies_cleared: qol_settings.escape_enemies_cleared,
        escape_refill: qol_settings.escape_refill,
        escape_movement_items: qol_settings.escape_movement_items,
        mark_map_stations: qol_settings.mark_map_stations,
        room_outline_revealed: qol_settings.room_outline_revealed,
        opposite_area_revealed: qol_settings.opposite_area_revealed,
        transition_letters: other_settings.transition_letters,
        door_locks_size: other_settings.door_locks_size,
        item_markers: qol_settings.item_markers,
        item_dot_change: other_settings.item_dot_change,
        all_items_spawn: qol_settings.all_items_spawn,
        acid_chozo: qol_settings.acid_chozo,
        remove_climb_lava: qol_settings.remove_climb_lava,
        buffed_drops: qol_settings.buffed_drops,
        fast_elevators: qol_settings.fast_elevators,
        fast_doors: qol_settings.fast_doors,
        fast_pause_menu: qol_settings.fast_pause_menu,
        respin: qol_settings.respin,
        infinite_space_jump: qol_settings.infinite_space_jump,
        momentum_conservation: qol_settings.momentum_conservation,
        objectives: {
            use Objective::*;
            match settings.objectives_mode {
                ObjectivesMode::None => vec![],
                ObjectivesMode::Bosses => vec![Kraid, Phantoon, Draygon, Ridley],
                ObjectivesMode::Minibosses => vec![SporeSpawn, Crocomire, Botwoon, GoldenTorizo],
                ObjectivesMode::Metroids => {
                    vec![MetroidRoom1, MetroidRoom2, MetroidRoom3, MetroidRoom4]
                }
                ObjectivesMode::Chozos => {
                    vec![BombTorizo, BowlingStatue, AcidChozoStatue, GoldenTorizo]
                }
                ObjectivesMode::Pirates => {
                    vec![PitRoom, BabyKraidRoom, PlasmaRoom, MetalPiratesRoom]
                }
                ObjectivesMode::Random => {
                    rand::seq::SliceRandom::choose_multiple(Objective::get_all(), &mut rng, 4)
                        .copied()
                        .collect()
                }
            }
        },
        doors_mode: settings.doors_mode,
        early_save: qol_settings.early_save,
        area_assignment: other_settings.area_assignment,
        wall_jump: if settings.start_location_mode == StartLocationMode::Escape {
            WallJump::Vanilla
        } else {
            other_settings.wall_jump
        },
        etank_refill: other_settings.etank_refill,
        maps_revealed: other_settings.maps_revealed,
        map_station_reveal: other_settings.map_station_reveal,
        vanilla_map,
        energy_free_shinesparks: other_settings.energy_free_shinesparks,
        ultra_low_qol: other_settings.ultra_low_qol,
        skill_assumptions_preset: skill_settings.preset.as_ref().map(|x| x.clone()),
        item_progression_preset: item_settings.preset.as_ref().map(|x| x.clone()),
        quality_of_life_preset: qol_settings.preset.as_ref().map(|x| x.clone()),
        debug: app_data.debug,
    };
    let difficulty_tiers = if difficulty.item_placement_style == ItemPlacementStyle::Forced {
        get_difficulty_tiers(&difficulty, &app_data)
    } else {
        vec![difficulty.clone()]
    };

    let filtered_base_links =
        filter_links(&app_data.game_data.links, &app_data.game_data, &difficulty);
    let filtered_base_links_data = LinksDataGroup::new(
        filtered_base_links,
        app_data.game_data.vertex_isv.keys.len(),
        0,
    );
    let map_layout = settings.map_layout.clone();
    let max_attempts = 2000;
    let max_attempts_per_map = if difficulty.start_location_mode == StartLocationMode::Random {
        10
    } else {
        1
    };
    let max_map_attempts = max_attempts / max_attempts_per_map;
    info!(
        "Random seed={random_seed}, max_attempts_per_map={max_attempts_per_map}, max_map_attempts={max_map_attempts}, difficulty={:?}",
        difficulty_tiers[0]
    );

    struct AttemptOutput {
        map_seed: usize,
        door_randomization_seed: usize,
        item_placement_seed: usize,
        randomization: Randomization,
        output_rom: Rom,
    }

    let time_start_attempts = Instant::now();
    let mut attempt_num = 0;
    let mut output_opt: Option<AttemptOutput> = None;
    'attempts: for _ in 0..max_map_attempts {
        let map_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        let door_randomization_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;

        if !app_data.map_repositories.contains_key(&map_layout) {
            // TODO: it doesn't make sense to panic on things like this.
            panic!("Unrecognized map layout option: {}", map_layout);
        }
        let mut map = app_data.map_repositories[&map_layout]
            .get_map(attempt_num, map_seed, &app_data.game_data)
            .unwrap();
        if difficulty.area_assignment == AreaAssignment::Random {
            randomize_map_areas(&mut map, map_seed);
        }
        let locked_door_data = randomize_doors(
            &app_data.game_data,
            &map,
            &difficulty_tiers[0],
            door_randomization_seed,
        );
        let randomizer = Randomizer::new(
            &map,
            &locked_door_data,
            &difficulty_tiers,
            &app_data.game_data,
            &filtered_base_links_data,
        );
        for _ in 0..max_attempts_per_map {
            let item_placement_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
            attempt_num += 1;

            info!("Attempt {attempt_num}/{max_attempts}: Map seed={map_seed}, door randomization seed={door_randomization_seed}, item placement seed={item_placement_seed}");
            let randomization_result =
                randomizer.randomize(attempt_num, item_placement_seed, display_seed);
            let randomization = match randomization_result {
                Ok(x) => x,
                Err(e) => {
                    info!(
                        "Attempt {attempt_num}/{max_attempts}: Randomization failed: {}",
                        e
                    );
                    continue;
                }
            };
            let output_rom_result = make_rom(&rom, &randomization, &app_data.game_data);
            let output_rom = match output_rom_result {
                Ok(x) => x,
                Err(e) => {
                    info!(
                        "Attempt {attempt_num}/{max_attempts}: Failed to write ROM: {}",
                        e
                    );
                    continue;
                }
            };
            info!(
                "Successful attempt {attempt_num}/{attempt_num}/{max_attempts}: display_seed={}, random_seed={random_seed}, map_seed={map_seed}, door_randomization_seed={door_randomization_seed}, item_placement_seed={item_placement_seed}",
                randomization.display_seed,
            );
            output_opt = Some(AttemptOutput {
                map_seed,
                door_randomization_seed,
                item_placement_seed,
                randomization,
                output_rom,
            });
            break 'attempts;
        }
    }

    if output_opt.is_none() {
        return HttpResponse::InternalServerError().body("Failed too many randomization attempts");
    }
    let output = output_opt.unwrap();

    info!(
        "Wall-clock time for attempts: {:?} sec",
        time_start_attempts.elapsed().as_secs_f32()
    );
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
        map_seed: output.map_seed,
        door_randomization_seed: output.door_randomization_seed,
        item_placement_seed: output.item_placement_seed,
        race_mode,
        preset: skill_settings.preset.clone(),
        item_progression_preset: item_settings.preset.clone(),
        difficulty: difficulty_tiers[0].clone(),
        quality_of_life_preset: qol_settings.preset.clone(),
        supers_double: qol_settings.supers_double,
        mother_brain_fight: to_variant_name(&qol_settings.mother_brain_fight)
            .unwrap()
            .to_string(),
        escape_enemies_cleared: qol_settings.escape_enemies_cleared,
        escape_refill: qol_settings.escape_refill,
        escape_movement_items: qol_settings.escape_movement_items,
        mark_map_stations: qol_settings.mark_map_stations,
        transition_letters: other_settings.transition_letters,
        item_markers: to_variant_name(&qol_settings.item_markers)
            .unwrap()
            .to_string(),
        item_dot_change: to_variant_name(&other_settings.item_dot_change)
            .unwrap()
            .to_string(),
        all_items_spawn: qol_settings.all_items_spawn,
        acid_chozo: qol_settings.acid_chozo,
        remove_climb_lava: qol_settings.remove_climb_lava,
        buffed_drops: qol_settings.buffed_drops,
        fast_elevators: qol_settings.fast_elevators,
        fast_doors: qol_settings.fast_doors,
        fast_pause_menu: qol_settings.fast_pause_menu,
        respin: qol_settings.respin,
        infinite_space_jump: qol_settings.infinite_space_jump,
        momentum_conservation: qol_settings.momentum_conservation,
        objectives: to_variant_name(&settings.objectives_mode)
            .unwrap()
            .to_string(),
        doors: to_variant_name(&settings.doors_mode).unwrap().to_string(),
        start_location_mode: to_variant_name(&settings.start_location_mode)
            .unwrap()
            .to_string(),
        map_layout: settings.map_layout.clone(),
        save_animals: to_variant_name(&settings.save_animals).unwrap().to_string(),
        early_save: qol_settings.early_save,
        area_assignment: to_variant_name(&other_settings.area_assignment)
            .unwrap()
            .to_string(),
        wall_jump: to_variant_name(&other_settings.wall_jump)
            .unwrap()
            .to_string(),
        etank_refill: to_variant_name(&other_settings.etank_refill)
            .unwrap()
            .to_string(),
        maps_revealed: to_variant_name(&other_settings.maps_revealed)
            .unwrap()
            .to_string(),
        vanilla_map,
        ultra_low_qol: other_settings.ultra_low_qol,
    };

    let seed_name = &output.randomization.seed_name;
    save_seed(
        seed_name,
        &seed_data,
        &req.spoiler_token.0,
        &rom,
        &output.output_rom,
        &output.randomization,
        &app_data,
    )
    .await
    .unwrap();

    HttpResponse::Ok().json(RandomizeResponse {
        seed_url: format!("/seed/{}/", seed_name),
    })
}
