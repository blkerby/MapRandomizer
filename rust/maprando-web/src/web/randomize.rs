mod helpers;

use crate::web::{AppData, VERSION};
use actix_easy_multipart::{bytes::Bytes, text::Text, MultipartForm};
use actix_web::{
    http::header::{self},
    post, web, HttpRequest, HttpResponse, Responder,
};
use askama::Template;
use helpers::*;
use log::info;
use maprando::{
    patch::{make_rom, Rom},
    randomize::{
        filter_links, randomize_doors, randomize_map_areas, AreaAssignment, DebugOptions,
        DifficultyConfig, DoorLocksSize, DoorsMode, ItemDotChange, ItemMarkers, ItemPlacementStyle,
        ItemPriorityStrength, MotherBrainFight, Objective, Randomization, Randomizer, SaveAnimals,
        StartLocationMode,
    },
};
use maprando_game::{
    Capacity, Item, LinksDataGroup, NotableId, RoomId, TechId, TECH_ID_CAN_ESCAPE_MORPH_LOCATION,
    TECH_ID_CAN_WALLJUMP,
};
use rand::{RngCore, SeedableRng};
use serde_derive::{Deserialize, Serialize};
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
    preset: Option<Text<String>>,
    shinespark_tiles: Text<f32>,
    heated_shinespark_tiles: Text<f32>,
    speed_ball_tiles: Text<f32>,
    shinecharge_leniency_frames: Text<Capacity>,
    resource_multiplier: Text<f32>,
    gate_glitch_leniency: Text<Capacity>,
    door_stuck_leniency: Text<Capacity>,
    phantoon_proficiency: Text<f32>,
    draygon_proficiency: Text<f32>,
    ridley_proficiency: Text<f32>,
    botwoon_proficiency: Text<f32>,
    mother_brain_proficiency: Text<f32>,
    escape_timer_multiplier: Text<f32>,
    tech_json: Text<String>,
    notable_json: Text<String>,
    progression_rate: Text<String>,
    item_placement_style: Text<String>,
    item_priority_strength: Text<String>,
    random_tank: Text<String>,
    spazer_before_plasma: Text<String>,
    stop_item_placement_early: Text<String>,
    item_progression_preset: Option<Text<String>>,
    item_pool_json: Text<String>,
    starting_item_json: Text<String>,
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
    room_outline_revealed: Text<bool>,
    transition_letters: Text<bool>,
    door_locks_size: Text<String>,
    item_markers: Text<String>,
    item_dot_change: Text<String>,
    all_items_spawn: Text<bool>,
    acid_chozo: Text<bool>,
    buffed_drops: Text<bool>,
    fast_elevators: Text<bool>,
    fast_doors: Text<bool>,
    fast_pause_menu: Text<bool>,
    respin: Text<bool>,
    infinite_space_jump: Text<bool>,
    momentum_conservation: Text<bool>,
    objectives: Text<String>,
    doors: Text<String>,
    start_location: Text<String>,
    map_layout: Text<String>,
    save_animals: Text<String>,
    early_save: Text<bool>,
    area_assignment: Text<String>,
    wall_jump: Text<String>,
    etank_refill: Text<String>,
    maps_revealed: Text<String>,
    map_station_reveal: Text<String>,
    energy_free_shinesparks: Text<bool>,
    ultra_low_qol: Text<bool>,
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
    let mut tech_vec: Vec<TechId> = vec![];
    for tech in &app_data.preset_data[0].preset.tech {
        // Include implicit tech (which is in the first preset):
        tech_vec.push(tech.tech_id);
    }
    for tech_setting in tech_json.as_array().unwrap().iter() {
        let tech_id = tech_setting[0].as_i64().unwrap() as TechId;
        let is_enabled = tech_setting[1].as_bool().unwrap();
        if tech_id == TECH_ID_CAN_WALLJUMP && req.wall_jump.0 == "Disabled" {
            continue;
        }
        if is_enabled {
            tech_vec.push(tech_id);
        }
    }

    let vanilla_map = req.map_layout.0 == "Vanilla";
    if vanilla_map {
        tech_vec.push(TECH_ID_CAN_ESCAPE_MORPH_LOCATION);
    }
    let notable_json: serde_json::Value = serde_json::from_str(&req.notable_json).unwrap();
    let mut notable_vec: Vec<(RoomId, NotableId)> = vec![];
    for notable in &app_data.preset_data[0].preset.notables {
        // Include implicit notables (which are in the first preset):
        notable_vec.push((notable.room_id, notable.notable_id));
    }
    for notable_setting in notable_json.as_array().unwrap().iter() {
        let room_id = notable_setting[0].as_i64().unwrap() as RoomId;
        let notable_id = notable_setting[1].as_i64().unwrap() as NotableId;
        let is_enabled = notable_setting[2].as_bool().unwrap();
        if is_enabled {
            notable_vec.push((room_id, notable_id));
        }
    }

    let starting_item_json: serde_json::Value =
        serde_json::from_str(&req.starting_item_json).unwrap();
    info!("starting_item_json: {:?}", starting_item_json);
    let mut starting_items: Vec<(Item, usize)> = vec![];
    for (k, cnt_json) in starting_item_json.as_object().unwrap().iter() {
        let cnt_str = cnt_json.as_str().unwrap();
        let cnt = if cnt_str == "Yes" {
            1
        } else if cnt_str == "No" {
            0
        } else {
            usize::from_str_radix(cnt_str, 10).unwrap()
        };
        if cnt > 0 {
            starting_items.push((
                Item::try_from(app_data.game_data.item_isv.index_by_key[k]).unwrap(),
                cnt,
            ));
        }
    }
    info!("Starting items: {:?}", starting_items);

    let item_pool_json: serde_json::Value = serde_json::from_str(&req.item_pool_json).unwrap();
    info!("item_pool_json: {:?}", item_pool_json);
    let mut item_pool: Vec<(Item, usize)> = vec![];
    for (k, cnt_json) in item_pool_json.as_object().unwrap().iter() {
        let cnt_str = cnt_json.as_str().unwrap();
        let cnt = usize::from_str_radix(cnt_str, 10).unwrap();
        item_pool.push((
            Item::try_from(app_data.game_data.item_isv.index_by_key[k]).unwrap(),
            cnt,
        ));
    }
    info!("Item pool: {:?}", item_pool);

    let item_priority_json: serde_json::Value =
        serde_json::from_str(&req.item_priority_json.0).unwrap();

    let filler_items_json: serde_json::Value =
        serde_json::from_str(&req.filler_items_json.0).unwrap();
    let semi_filler_items: Vec<Item> = filler_items_json
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_k, v)| v.as_str().unwrap() == "Semi")
        .map(|(k, _v)| Item::try_from(app_data.game_data.item_isv.index_by_key[k]).unwrap())
        .collect();
    let mut filler_items = vec![Item::Missile, Item::Nothing];
    filler_items.extend(
        filler_items_json
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_k, v)| v.as_str().unwrap() == "Yes" || v.as_str().unwrap() == "Early")
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
    info!("Semi-filler items: {:?}", semi_filler_items);
    info!("Early filler items: {:?}", early_filler_items);

    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&random_seed.to_le_bytes());
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let difficulty = DifficultyConfig {
        name: Some(
            req.preset
                .as_ref()
                .map(|x| x.to_string())
                .unwrap_or("Beyond".to_string()),
        ),
        tech: tech_vec,
        notables: notable_vec,
        shine_charge_tiles: req.shinespark_tiles.0,
        heated_shine_charge_tiles: req.heated_shinespark_tiles.0,
        speed_ball_tiles: req.speed_ball_tiles.0,
        shinecharge_leniency_frames: req.shinecharge_leniency_frames.0,
        progression_rate: match req.progression_rate.0.as_str() {
            "Slow" => maprando::randomize::ProgressionRate::Slow,
            "Uniform" => maprando::randomize::ProgressionRate::Uniform,
            "Fast" => maprando::randomize::ProgressionRate::Fast,
            _ => panic!(
                "Unrecognized progression rate {}",
                req.progression_rate.0.as_str()
            ),
        },
        item_priority_strength: match req.item_priority_strength.0.as_str() {
            "Moderate" => ItemPriorityStrength::Moderate,
            "Heavy" => ItemPriorityStrength::Heavy,
            _ => panic!(
                "Unrecognized item priority strength {}",
                req.item_priority_strength.0,
            ),
        },
        random_tank: match req.random_tank.0.as_str() {
            "No" => false,
            "Yes" => true,
            _ => panic!("Unrecognized random_tank {}", req.random_tank.0.as_str()),
        },
        spazer_before_plasma: match req.spazer_before_plasma.0.as_str() {
            "No" => false,
            "Yes" => true,
            _ => panic!(
                "Unrecognized spazer_before_plasma {}",
                req.spazer_before_plasma.0.as_str()
            ),
        },
        stop_item_placement_early: match req.stop_item_placement_early.0.as_str() {
            "No" => false,
            "Yes" => true,
            _ => panic!(
                "Unrecognized stop_item_placement_early {}",
                req.stop_item_placement_early.0.as_str()
            ),
        },
        item_pool,
        starting_items,
        filler_items,
        semi_filler_items,
        early_filler_items,
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
        door_stuck_leniency: req.door_stuck_leniency.0,
        start_location_mode: match req.start_location.0.as_str() {
            "Ship" => maprando::randomize::StartLocationMode::Ship,
            "Random" => maprando::randomize::StartLocationMode::Random,
            "Escape" => maprando::randomize::StartLocationMode::Escape,
            _ => panic!("Unrecognized start_location: {}", req.start_location.0),
        },
        save_animals: match req.save_animals.0.as_str() {
            "No" => SaveAnimals::No,
            "Maybe" => SaveAnimals::Maybe,
            "Yes" => SaveAnimals::Yes,
            _ => panic!(
                "Unrecognized save_animals options {}",
                req.save_animals.0.as_str()
            ),
        },
        phantoon_proficiency: req.phantoon_proficiency.0,
        draygon_proficiency: req.draygon_proficiency.0,
        ridley_proficiency: req.ridley_proficiency.0,
        botwoon_proficiency: req.botwoon_proficiency.0,
        mother_brain_proficiency: req.mother_brain_proficiency.0,
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
        room_outline_revealed: req.room_outline_revealed.0,
        transition_letters: req.transition_letters.0,
        door_locks_size: match req.door_locks_size.0.as_str() {
            "small" => DoorLocksSize::Small,
            "large" => DoorLocksSize::Large,
            _ => panic!("Unrecognized door_locks_size: {}", req.door_locks_size.0),
        },
        item_markers: match req.item_markers.0.as_str() {
            "Simple" => ItemMarkers::Simple,
            "Majors" => ItemMarkers::Majors,
            "Uniques" => ItemMarkers::Uniques,
            "3-Tiered" => ItemMarkers::ThreeTiered,
            "4-Tiered" => ItemMarkers::FourTiered,
            _ => panic!("Unrecognized item_markers: {}", req.item_markers.0),
        },
        item_dot_change: match req.item_dot_change.0.as_str() {
            "Fade" => ItemDotChange::Fade,
            "Disappear" => ItemDotChange::Disappear,
            _ => panic!("Unrecognized item_dot_change: {}", req.item_dot_change.0),
        },
        all_items_spawn: req.all_items_spawn.0,
        acid_chozo: req.acid_chozo.0,
        buffed_drops: req.buffed_drops.0,
        fast_elevators: req.fast_elevators.0,
        fast_doors: req.fast_doors.0,
        fast_pause_menu: req.fast_pause_menu.0,
        respin: req.respin.0,
        infinite_space_jump: req.infinite_space_jump.0,
        momentum_conservation: req.momentum_conservation.0,
        objectives: {
            use Objective::*;
            match req.objectives.0.as_str() {
                "None" => vec![],
                "Bosses" => vec![Kraid, Phantoon, Draygon, Ridley],
                "Minibosses" => vec![SporeSpawn, Crocomire, Botwoon, GoldenTorizo],
                "Metroids" => vec![MetroidRoom1, MetroidRoom2, MetroidRoom3, MetroidRoom4],
                "Chozos" => vec![BombTorizo, BowlingStatue, AcidChozoStatue, GoldenTorizo],
                "Pirates" => vec![PitRoom, BabyKraidRoom, PlasmaRoom, MetalPiratesRoom],
                "Random" => {
                    rand::seq::SliceRandom::choose_multiple(Objective::get_all(), &mut rng, 4)
                        .copied()
                        .collect()
                }
                _ => panic!("Unrecognized objectives: {}", req.objectives.0),
            }
        },
        doors_mode: match req.doors.0.as_str() {
            "Blue" => DoorsMode::Blue,
            "Ammo" => DoorsMode::Ammo,
            "Beam" => DoorsMode::Beam,
            _ => panic!("Unrecognized doors mode: {}", req.doors.0),
        },
        early_save: req.early_save.0,
        area_assignment: match req.area_assignment.0.as_str() {
            "Standard" => AreaAssignment::Standard,
            "Random" => AreaAssignment::Random,
            _ => panic!("Unrecognized ship area option: {}", req.area_assignment.0),
        },
        wall_jump: if req.start_location.0.as_str() == "Escape" {
            maprando::randomize::WallJump::Vanilla
        } else {
            match req.wall_jump.0.as_str() {
                "Vanilla" => maprando::randomize::WallJump::Vanilla,
                "Collectible" => maprando::randomize::WallJump::Collectible,
                _ => panic!(
                    "Unrecognized wall_jump setting {}",
                    req.wall_jump.0.as_str()
                ),
            }
        },
        etank_refill: match req.etank_refill.0.as_str() {
            "Disabled" => maprando::randomize::EtankRefill::Disabled,
            "Vanilla" => maprando::randomize::EtankRefill::Vanilla,
            "Full" => maprando::randomize::EtankRefill::Full,
            _ => panic!(
                "Unrecognized etank_refill setting {}",
                req.etank_refill.0.as_str()
            ),
        },
        maps_revealed: match req.maps_revealed.0.as_str() {
            "No" => maprando::randomize::MapsRevealed::No,
            "Partial" => maprando::randomize::MapsRevealed::Partial,
            "Full" => maprando::randomize::MapsRevealed::Full,
            _ => panic!(
                "Unrecognized maps_revealed setting {}",
                req.maps_revealed.0.as_str()
            ),
        },
        map_station_reveal: match req.map_station_reveal.0.as_str() {
            "Partial" => maprando::randomize::MapStationReveal::Partial,
            "Full" => maprando::randomize::MapStationReveal::Full,
            _ => panic!(
                "Unrecognized maps_station_reveal setting {}",
                req.maps_revealed.0.as_str()
            ),
        },
        vanilla_map,
        energy_free_shinesparks: req.energy_free_shinesparks.0,
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

    let filtered_base_links =
        filter_links(&app_data.game_data.links, &app_data.game_data, &difficulty);
    let filtered_base_links_data = LinksDataGroup::new(
        filtered_base_links,
        app_data.game_data.vertex_isv.keys.len(),
        0,
    );
    let map_layout = req.map_layout.0.clone();
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
        preset: req.preset.as_ref().map(|x| x.0.clone()),
        item_progression_preset: req.item_progression_preset.as_ref().map(|x| x.0.clone()),
        difficulty: difficulty_tiers[0].clone(),
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
        buffed_drops: req.buffed_drops.0,
        fast_elevators: req.fast_elevators.0,
        fast_doors: req.fast_doors.0,
        fast_pause_menu: req.fast_pause_menu.0,
        respin: req.respin.0,
        infinite_space_jump: req.infinite_space_jump.0,
        momentum_conservation: req.momentum_conservation.0,
        objectives: req.objectives.0.clone(),
        doors: req.doors.0.clone(),
        start_location_mode: req.start_location.0.clone(),
        map_layout: req.map_layout.0.clone(),
        save_animals: req.save_animals.0.clone(),
        early_save: req.early_save.0,
        area_assignment: req.area_assignment.0.clone(),
        wall_jump: req.wall_jump.0.clone(),
        etank_refill: req.etank_refill.0.clone(),
        maps_revealed: req.maps_revealed.0.clone(),
        vanilla_map,
        ultra_low_qol: req.ultra_low_qol.0,
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

    // Redirect to the seed page:
    HttpResponse::Found()
        .insert_header((header::LOCATION, format!("seed/{}/", seed_name)))
        .finish()
}
