use anyhow::{bail, Context, Result};
use clap::Parser;
use log::info;
use maprando::customize::samus_sprite::SamusSpriteCategory;
use maprando::customize::{customize_rom, ControllerConfig, CustomizeSettings, MusicSettings};
use maprando::map_repository::MapRepository;
use maprando::patch::ips_write::create_ips_patch;
use maprando::patch::Rom;
use maprando::preset::Preset;
use maprando::randomize::{
    randomize_doors, randomize_map_areas, AreaAssignment, DoorsMode, ItemDotChange,
    ItemLocationRestriction, ItemMarkers, ItemPlacementStyle, ItemPriorityGroup,
    ItemPriorityStrength, MotherBrainFight, Objective, ProgressionRate, Randomization, Randomizer,
    SaveAnimals, StartLocationMode,
};
use maprando::spoiler_map;
use maprando::{patch::make_rom, randomize::DifficultyConfig};
use maprando_game::{Capacity, GameData, Item};
use rand::{RngCore, SeedableRng};
use std::path::{Path, PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 100 as usize)]
    test_cycles: usize,

    #[arg(long)]
    attempt_num: Option<u64>,

    #[arg(long)]
    input_rom: PathBuf,

    #[arg(long)]
    output_seeds: PathBuf,

    #[arg(long)]
    skill_preset: Option<String>,

    #[arg(long)]
    progression_preset: Option<String>,

    #[arg(long)]
    qol_preset: Option<String>,
}

fn create_difficulty_from_preset(preset: &Preset) -> DifficultyConfig {
    let diff = DifficultyConfig {
        name: None,
        // From the actual Preset
        tech: preset.tech.clone(),
        notable_strats: preset.notable_strats.clone(),
        shine_charge_tiles: preset.shinespark_tiles as f32,
        heated_shine_charge_tiles: preset.heated_shinespark_tiles as f32,
        speed_ball_tiles: preset.speed_ball_tiles as f32,
        shinecharge_leniency_frames: preset.shinecharge_leniency_frames as Capacity,
        resource_multiplier: preset.resource_multiplier,
        escape_timer_multiplier: preset.escape_timer_multiplier,
        gate_glitch_leniency: preset.gate_glitch_leniency as Capacity,
        door_stuck_leniency: preset.door_stuck_leniency as Capacity,
        phantoon_proficiency: preset.phantoon_proficiency,
        draygon_proficiency: preset.draygon_proficiency,
        ridley_proficiency: preset.ridley_proficiency,
        botwoon_proficiency: preset.botwoon_proficiency,
        mother_brain_proficiency: preset.mother_brain_proficiency,
        // Progression options, Normal preset
        progression_rate: ProgressionRate::Fast,
        item_location_restriction: ItemLocationRestriction::None,
        random_tank: true,
        spazer_before_plasma: true,
        stop_item_placement_early: false,
        item_pool: vec![],
        starting_items: vec![],
        filler_items: vec![
            Item::Missile,
            Item::ETank,
            Item::ReserveTank,
            Item::Super,
            Item::PowerBomb,
            Item::Charge,
            Item::Ice,
            Item::Wave,
            Item::Spazer,
        ],
        semi_filler_items: vec![],
        early_filler_items: vec![Item::ETank],
        item_placement_style: ItemPlacementStyle::Neutral,
        item_priority_strength: ItemPriorityStrength::Moderate,
        item_priorities: vec![
            ItemPriorityGroup {
                name: "Early".to_string(),
                items: vec!["ETank", "Morph"]
                    .into_iter()
                    .map(|x| x.to_string())
                    .collect(),
            },
            ItemPriorityGroup {
                name: "Default".to_string(),
                items: vec![
                    "ReserveTank",
                    "Super",
                    "PowerBomb",
                    "Charge",
                    "Ice",
                    "Wave",
                    "Spazer",
                    "Plasma",
                    "Bombs",
                    "Grapple",
                    "HiJump",
                    "SpeedBooster",
                    "SpringBall",
                    "XRayScope",
                    "WallJump",
                ]
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
            },
            ItemPriorityGroup {
                name: "Late".to_string(),
                items: vec!["SpaceJump", "ScrewAttack", "Varia", "Gravity"]
                    .into_iter()
                    .map(|x| x.to_string())
                    .collect(),
            },
        ],
        // QoL (Default)
        item_markers: ItemMarkers::ThreeTiered,
        mark_map_stations: true,
        room_outline_revealed: true,
        early_save: true,
        mother_brain_fight: MotherBrainFight::Short,
        supers_double: true,
        escape_movement_items: true,
        escape_refill: true,
        escape_enemies_cleared: true,
        fast_elevators: true,
        fast_doors: true,
        fast_pause_menu: true,
        respin: false,
        infinite_space_jump: false,
        momentum_conservation: false,
        all_items_spawn: true,
        acid_chozo: true,
        buffed_drops: true,

        // Game options
        objectives: vec![
            Objective::Kraid,
            Objective::Phantoon,
            Objective::Draygon,
            Objective::Ridley,
        ],
        vanilla_map: false,
        doors_mode: DoorsMode::Blue,
        start_location_mode: StartLocationMode::Random,
        save_animals: SaveAnimals::No,

        // Other options
        wall_jump: maprando::randomize::WallJump::Vanilla,
        etank_refill: maprando::randomize::EtankRefill::Vanilla,
        area_assignment: AreaAssignment::Standard,
        item_dot_change: ItemDotChange::Fade,
        transition_letters: false,
        door_locks_size: maprando::randomize::DoorLocksSize::Small,
        maps_revealed: maprando::randomize::MapsRevealed::No,
        map_station_reveal: maprando::randomize::MapStationReveal::Full,
        energy_free_shinesparks: false,
        ultra_low_qol: false,

        skill_assumptions_preset: Some(preset.name.clone()),
        item_progression_preset: Some("Normal".to_string()),
        quality_of_life_preset: Some("Default".to_string()),

        debug_options: None,
    };

    diff
}

fn set_item_progression_normal(diff: &mut DifficultyConfig) -> () {
    diff.progression_rate = ProgressionRate::Fast;
    diff.random_tank = true;
    diff.spazer_before_plasma = true;
    diff.filler_items = vec![
        Item::Missile,
        Item::ETank,
        Item::ReserveTank,
        Item::Super,
        Item::PowerBomb,
        Item::Charge,
        Item::Ice,
        Item::Wave,
        Item::Spazer,
        Item::Plasma,
        Item::XRayScope,
    ];
    diff.semi_filler_items = vec![];
    diff.early_filler_items = vec![Item::ETank];
    diff.item_placement_style = ItemPlacementStyle::Neutral;
    diff.item_priorities = vec![
        ItemPriorityGroup {
            name: "Early".to_string(),
            items: vec!["Morph"].into_iter().map(|x| x.to_string()).collect(),
        },
        ItemPriorityGroup {
            name: "Default".to_string(),
            items: vec![
                "ETank",
                "ReserveTank",
                "Super",
                "PowerBomb",
                "Charge",
                "Ice",
                "Wave",
                "Spazer",
                "Plasma",
                "Bombs",
                "Grapple",
                "HiJump",
                "SpeedBooster",
                "SpringBall",
                "XRayScope",
                "WallJump",
                "SpaceJump",
                "ScrewAttack",
            ]
            .into_iter()
            .map(|x| x.to_string())
            .collect(),
        },
        ItemPriorityGroup {
            name: "Late".to_string(),
            items: vec!["Varia", "Gravity"]
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
        },
    ];
    diff.item_progression_preset = Some("Normal".to_string());
    ();
}

fn set_item_progression_tricky(diff: &mut DifficultyConfig) -> () {
    diff.progression_rate = ProgressionRate::Uniform;
    diff.random_tank = true;
    diff.spazer_before_plasma = true;
    diff.filler_items = vec![
        Item::Missile,
        Item::ETank,
        Item::ReserveTank,
        Item::Super,
        Item::PowerBomb,
        Item::Charge,
        Item::Ice,
        Item::Wave,
        Item::Spazer,
        Item::Plasma,
        Item::XRayScope,
    ];
    diff.semi_filler_items = vec![];
    diff.early_filler_items = vec![];
    diff.item_placement_style = ItemPlacementStyle::Forced;
    diff.item_priorities = vec![
        ItemPriorityGroup {
            name: "Early".to_string(),
            items: vec![
                "Morph",
                "ETank",
                "ReserveTank",
                "Super",
                "PowerBomb",
                "Charge",
                "Ice",
                "Wave",
                "Spazer",
                "Plasma",
                "Bombs",
                "Grapple",
                "HiJump",
                "SpeedBooster",
                "SpringBall",
                "XRayScope",
            ]
            .into_iter()
            .map(|x| x.to_string())
            .collect(),
        },
        ItemPriorityGroup {
            name: "Default".to_string(),
            items: vec!["WallJump", "SpaceJump", "ScrewAttack"]
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
        },
        ItemPriorityGroup {
            name: "Late".to_string(),
            items: vec!["Varia", "Gravity"]
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
        },
    ];
    diff.item_progression_preset = Some("Tricky".to_string());
    ();
}

fn set_item_progression_challenge(diff: &mut DifficultyConfig) -> () {
    diff.progression_rate = ProgressionRate::Slow;
    diff.random_tank = true;
    diff.spazer_before_plasma = true;
    diff.stop_item_placement_early = false;
    diff.filler_items = vec![Item::Missile, Item::Charge, Item::Spazer];
    diff.semi_filler_items = vec![Item::Super, Item::PowerBomb];
    diff.early_filler_items = vec![];
    diff.item_placement_style = ItemPlacementStyle::Forced;
    diff.item_priorities = vec![
        ItemPriorityGroup {
            name: "Early".to_string(),
            items: vec![
                "ETank",
                "ReserveTank",
                "Super",
                "PowerBomb",
                "Charge",
                "Ice",
                "Wave",
                "Spazer",
                "Plasma",
                "Bombs",
                "Grapple",
                "HiJump",
                "SpeedBooster",
                "SpringBall",
                "XRayScope",
            ]
            .into_iter()
            .map(|x| x.to_string())
            .collect(),
        },
        ItemPriorityGroup {
            name: "Default".to_string(),
            items: vec!["Morph", "WallJump", "SpaceJump", "ScrewAttack"]
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
        },
        ItemPriorityGroup {
            name: "Late".to_string(),
            items: vec!["Varia", "Gravity"]
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
        },
    ];
    diff.item_progression_preset = Some("Challenge".to_string());
    ();
}

fn set_item_progression_desolate(diff: &mut DifficultyConfig) -> () {
    diff.progression_rate = ProgressionRate::Slow;
    diff.random_tank = true;
    diff.spazer_before_plasma = true;
    diff.stop_item_placement_early = true;
    diff.item_pool = vec![
        (Item::Missile, 12),
        (Item::Super, 6),
        (Item::PowerBomb, 6),
        (Item::ETank, 3),
        (Item::ReserveTank, 3),
    ];
    diff.filler_items = vec![Item::Missile, Item::Charge, Item::Spazer];
    diff.semi_filler_items = vec![Item::Super, Item::PowerBomb];
    diff.early_filler_items = vec![];
    diff.item_placement_style = ItemPlacementStyle::Forced;
    diff.item_priorities = vec![
        ItemPriorityGroup {
            name: "Early".to_string(),
            items: vec![
                "ETank",
                "ReserveTank",
                "Super",
                "PowerBomb",
                "Charge",
                "Ice",
                "Wave",
                "Spazer",
                "Plasma",
                "Bombs",
                "Grapple",
                "HiJump",
                "SpeedBooster",
                "SpringBall",
                "XRayScope",
            ]
            .into_iter()
            .map(|x| x.to_string())
            .collect(),
        },
        ItemPriorityGroup {
            name: "Default".to_string(),
            items: vec!["Morph", "WallJump", "SpaceJump", "ScrewAttack"]
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
        },
        ItemPriorityGroup {
            name: "Late".to_string(),
            items: vec!["Varia", "Gravity"]
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
        },
    ];
    diff.item_progression_preset = Some("Desolate".to_string());
    ();
}

fn set_qol_default(diff: &mut DifficultyConfig) -> () {
    diff.item_markers = ItemMarkers::ThreeTiered;
    diff.mark_map_stations = true;
    diff.room_outline_revealed = true;
    diff.early_save = true;
    diff.mother_brain_fight = MotherBrainFight::Short;
    diff.supers_double = true;
    diff.escape_movement_items = true;
    diff.escape_refill = true;
    diff.escape_enemies_cleared = true;
    diff.fast_elevators = true;
    diff.fast_doors = true;
    diff.fast_pause_menu = true;
    diff.respin = false;
    diff.infinite_space_jump = false;
    diff.momentum_conservation = false;
    diff.all_items_spawn = true;
    diff.acid_chozo = true;
    diff.buffed_drops = true;
    diff.ultra_low_qol = false;
    diff.quality_of_life_preset = Some("Default".to_string());
    ()
}

fn set_qol_max(diff: &mut DifficultyConfig) -> () {
    diff.item_markers = ItemMarkers::ThreeTiered;
    diff.mark_map_stations = true;
    diff.room_outline_revealed = true;
    diff.early_save = true;
    diff.mother_brain_fight = MotherBrainFight::Skip;
    diff.supers_double = true;
    diff.escape_movement_items = true;
    diff.escape_refill = true;
    diff.escape_enemies_cleared = true;
    diff.fast_elevators = true;
    diff.fast_doors = true;
    diff.fast_pause_menu = true;
    diff.respin = true;
    diff.infinite_space_jump = true;
    diff.momentum_conservation = true;
    diff.all_items_spawn = true;
    diff.acid_chozo = true;
    diff.buffed_drops = true;
    diff.ultra_low_qol = false;
    diff.quality_of_life_preset = Some("Max".to_string());
    ()
}

fn set_qol_low(diff: &mut DifficultyConfig) -> () {
    diff.item_markers = ItemMarkers::Uniques;
    diff.mark_map_stations = true;
    diff.room_outline_revealed = false;
    diff.early_save = false;
    diff.mother_brain_fight = MotherBrainFight::Short;
    diff.supers_double = true;
    diff.escape_movement_items = false;
    diff.escape_refill = false;
    diff.escape_enemies_cleared = false;
    diff.fast_elevators = true;
    diff.fast_doors = true;
    diff.fast_pause_menu = true;
    diff.respin = false;
    diff.infinite_space_jump = false;
    diff.momentum_conservation = false;
    diff.all_items_spawn = false;
    diff.acid_chozo = false;
    diff.buffed_drops = false;
    diff.ultra_low_qol = false;
    diff.quality_of_life_preset = Some("Low".to_string());
    ()
}

fn set_qol_off(diff: &mut DifficultyConfig) -> () {
    diff.item_markers = ItemMarkers::Simple;
    diff.mark_map_stations = false;
    diff.room_outline_revealed = false;
    diff.early_save = false;
    diff.mother_brain_fight = MotherBrainFight::Vanilla;
    diff.supers_double = false;
    diff.escape_movement_items = false;
    diff.escape_refill = false;
    diff.escape_enemies_cleared = false;
    diff.fast_elevators = false;
    diff.fast_doors = false;
    diff.fast_pause_menu = false;
    diff.respin = false;
    diff.infinite_space_jump = false;
    diff.momentum_conservation = false;
    diff.all_items_spawn = false;
    diff.acid_chozo = false;
    diff.buffed_drops = false;
    diff.ultra_low_qol = false;
    diff.quality_of_life_preset = Some("Off".to_string());
    ()
}

fn set_ultra_low_qol(mut diff: &mut DifficultyConfig) -> () {
    set_qol_off(&mut diff);
    diff.ultra_low_qol = true;
    ()
}

// Reduced version of web::AppData for test tool
struct TestAppData {
    attempt_num: Option<u64>,
    input_rom: Rom,
    output_dir: PathBuf,
    game_data: GameData,
    map_repos: Vec<(MapRepository, Option<fn(&mut DifficultyConfig) -> ()>)>,
    presets: Vec<Preset>,
    progressions: Vec<fn(&mut DifficultyConfig) -> ()>,
    qols: Vec<fn(&mut DifficultyConfig) -> ()>,
    etank_colors: Vec<(u8, u8, u8)>,
    samus_sprite_categories: Vec<SamusSpriteCategory>,
    samus_sprites: Vec<String>,
}

fn get_randomization(app: &TestAppData, seed: u64) -> Result<(Randomization, String)> {
    let game_data = &app.game_data;
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let preset_idx = rng.next_u64() as usize % app.presets.len();
    let progression_idx = rng.next_u64() as usize % app.progressions.len();
    let qol_idx = rng.next_u64() as usize % app.qols.len();
    let repo_idx = rng.next_u64() as usize % app.map_repos.len();

    let preset = &app.presets[preset_idx];
    let progression = &app.progressions[progression_idx];
    let qol = &app.qols[qol_idx];
    let (repo, repo_diff) = &app.map_repos[repo_idx];

    let mut diff = create_difficulty_from_preset(&preset);
    progression(&mut diff);
    qol(&mut diff);

    match repo_diff {
        Some(rd) => rd(&mut diff),
        None => (),
    };

    let skill_label = match &diff.skill_assumptions_preset {
        Some(s) => s.clone(),
        None => String::from("Custom"),
    };
    let prog_label = match &diff.item_progression_preset {
        Some(s) => s.clone(),
        None => String::from("Custom"),
    };
    let qol_label = match &diff.quality_of_life_preset {
        Some(s) => s.clone(),
        None => String::from("Custom"),
    };

    info!(
        "Generating seed using Skills {0}, Progression {1}, QoL {2}",
        skill_label, prog_label, qol_label
    );

    let difficulty_tiers = [diff.clone()]; // TODO needs to do the right thing for
                                           // ItemPlacementStyle::Forced
    let random_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&random_seed.to_le_bytes());
    rng_seed[9] = 0; // Not race-mode
    rng = rand::rngs::StdRng::from_seed(rng_seed);

    let max_attempts = 10000;
    let max_attempts_per_map = if diff.start_location_mode == StartLocationMode::Random {
        10
    } else {
        1
    };
    let max_map_attempts = max_attempts / max_attempts_per_map;
    let mut attempt_num = 0;

    let output_file_prefix = format!(
        "{0}-{1}-{2}-{3}",
        skill_label, prog_label, qol_label, random_seed
    );

    // Save a dump of the difficulty
    std::fs::write(
        Path::join(
            &app.output_dir,
            format!("{output_file_prefix}-difficulty.txt"),
        ),
        format!("{diff:?}"),
    )?;

    for _ in 0..max_map_attempts {
        let map_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        let door_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        let mut map = repo.get_map(attempt_num, map_seed, game_data)?;
        if diff.area_assignment == AreaAssignment::Random {
            randomize_map_areas(&mut map, map_seed);
        }
        let locked_door_data = randomize_doors(&game_data, &map, &diff, door_seed);
        let randomizer = Randomizer::new(
            &map,
            &locked_door_data,
            &difficulty_tiers,
            &game_data,
            &game_data.base_links_data,
        );
        for _ in 0..max_attempts_per_map {
            attempt_num += 1;
            let item_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
            info!("Attempt {attempt_num}/{max_attempts}: Map seed={map_seed}, door randomization seed={door_seed}, item placement seed={item_seed}");
            match randomizer.randomize(attempt_num, item_seed, 1) {
                Ok(randomization) => {
                    return Ok((randomization, output_file_prefix));
                }
                Err(e) => {
                    info!(
                        "Attempt {attempt_num}/{max_attempts}: Randomization failed: {}",
                        e
                    );
                }
            }
        }
    }
    bail!("Exhausted randomization attempts");
}

fn make_random_customization(app: &TestAppData) -> CustomizeSettings {
    let mut rng = rand::rngs::StdRng::from_entropy();

    let mut possible_sprites: Vec<Option<String>> = app
        .samus_sprites
        .clone()
        .into_iter()
        .map(|x| Some(x))
        .collect();
    possible_sprites.push(None);

    let mut possible_etanks: Vec<Option<(u8, u8, u8)>> = app
        .etank_colors
        .clone()
        .into_iter()
        .map(|x| Some(x))
        .collect();
    possible_etanks.push(None);

    let bits = rng.next_u64();

    let cust = CustomizeSettings {
        samus_sprite: possible_sprites[rng.next_u64() as usize % possible_sprites.len()].clone(),
        etank_color: possible_etanks[rng.next_u64() as usize % possible_etanks.len()],
        reserve_hud_style: bits & 0x01 != 0,
        vanilla_screw_attack_animation: bits & 0x02 != 0,
        palette_theme: maprando::customize::PaletteTheme::AreaThemed,
        tile_theme: maprando::customize::TileTheme::Vanilla,
        door_theme: maprando::customize::DoorTheme::Vanilla,
        music: match (bits & 0x04 != 0, bits & 0x08 != 0) {
            (true, true) => MusicSettings::Vanilla,
            (true, false) => MusicSettings::Disabled,
            (false, _) => MusicSettings::AreaThemed,
        },
        disable_beeping: bits & 0x10 != 0,
        shaking: match (bits & 0x20 != 0, bits & 0x40 != 0) {
            (true, true) => maprando::customize::ShakingSetting::Reduced,
            (true, false) => maprando::customize::ShakingSetting::Disabled,
            (false, _) => maprando::customize::ShakingSetting::Vanilla,
        },
        flashing: match bits & 0x80 != 0 {
            true => maprando::customize::FlashingSetting::Reduced,
            false => maprando::customize::FlashingSetting::Vanilla,
        },
        controller_config: ControllerConfig::default(),
    };

    cust
}

fn perform_test_cycle(app: &TestAppData, cycle_count: usize) -> Result<()> {
    let seed: u64 = app.attempt_num.unwrap_or(cycle_count as u64);

    info!("Test cycle {cycle_count} Start: seed={}", seed);

    // Perform randomization (map selection & item placement):
    let (randomization, output_file_prefix) = get_randomization(&app, seed)?;

    // Generate the patched ROM:
    let game_rom = make_rom(&app.input_rom, &randomization, &app.game_data)?;
    let ips_patch = create_ips_patch(&app.input_rom.data, &game_rom.data);

    let mut output_rom = app.input_rom.clone();
    let basic_customize_settings = CustomizeSettings {
        samus_sprite: None,
        etank_color: None,
        reserve_hud_style: true,
        vanilla_screw_attack_animation: true,
        palette_theme: maprando::customize::PaletteTheme::AreaThemed,
        tile_theme: maprando::customize::TileTheme::Vanilla,
        door_theme: maprando::customize::DoorTheme::Vanilla,
        music: MusicSettings::AreaThemed,
        disable_beeping: false,
        shaking: maprando::customize::ShakingSetting::Vanilla,
        flashing: maprando::customize::FlashingSetting::Vanilla,
        controller_config: ControllerConfig::default(),
    };
    customize_rom(
        &mut output_rom,
        &app.input_rom,
        &ips_patch,
        &Some(randomization.map.clone()),
        &basic_customize_settings,
        &app.game_data,
        &app.samus_sprite_categories,
        &vec![],
    )?;

    std::fs::write(
        Path::join(
            &app.output_dir,
            format!("{output_file_prefix}-basic-customize.txt"),
        ),
        format!("{basic_customize_settings:?}"),
    )?;

    let output_rom_path = Path::join(&app.output_dir, format!("{output_file_prefix}-rom.smc"));
    info!("Writing output ROM to {}", output_rom_path.display());
    output_rom.save(&output_rom_path)?;

    for custom in 0..5 {
        let custom_rom_path = Path::join(
            &app.output_dir,
            format!("{output_file_prefix}-custom-{}.smc", custom + 1),
        );
        output_rom = app.input_rom.clone();
        let customize_settings = make_random_customization(&app);
        customize_rom(
            &mut output_rom,
            &app.input_rom,
            &ips_patch,
            &Some(randomization.map.clone()),
            &customize_settings,
            &app.game_data,
            &app.samus_sprite_categories,
            &vec![],
        )?;
        info!(
            "Writing customization #{0} to {1}",
            custom + 1,
            custom_rom_path.display()
        );
        output_rom.save(&custom_rom_path)?;
        std::fs::write(
            Path::join(
                &app.output_dir,
                format!("{output_file_prefix}-customize-{}.txt", custom + 1),
            ),
            format!("{customize_settings:?}"),
        )?;
    }

    let output_spoiler_log_path = Path::join(
        &app.output_dir,
        format!("{output_file_prefix}-spoiler.json"),
    );
    info!(
        "Writing spoiler log to {}",
        output_spoiler_log_path.display()
    );
    let spoiler_str = serde_json::to_string_pretty(&randomization.spoiler_log)?;
    std::fs::write(output_spoiler_log_path, spoiler_str)?;

    let spoiler_maps =
        spoiler_map::get_spoiler_map(&output_rom, &randomization.map, &app.game_data)?;

    let output_spoiler_map_assigned_path = Path::join(
        &app.output_dir,
        format!("{output_file_prefix}-assigned.png"),
    );

    info!(
        "Writing spoiler map (assigned areas) to {}",
        output_spoiler_map_assigned_path.display()
    );
    let spoiler_map_assigned = spoiler_maps.assigned.clone();
    std::fs::write(output_spoiler_map_assigned_path, spoiler_map_assigned)?;

    let output_spoiler_map_vanilla_path =
        Path::join(&app.output_dir, format!("{output_file_prefix}-vanilla.png"));

    info!(
        "Writing spoiler map (vanilla areas) to {}",
        output_spoiler_map_vanilla_path.display()
    );
    let spoiler_map_vanilla = spoiler_maps.vanilla.clone();
    std::fs::write(output_spoiler_map_vanilla_path, spoiler_map_vanilla)?;

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args = Args::parse();
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let escape_timings_path = Path::new("data/escape_timings.json");
    let start_locations_path = Path::new("data/start_locations.json");
    let hub_locations_path = Path::new("data/hub_locations.json");
    let etank_colors_path = Path::new("data/etank_colors.json");
    let reduced_flashing_path = Path::new("data/reduced_flashing.json");
    let vanilla_map_path = Path::new("../maps/vanilla");
    let tame_maps_path = Path::new("../maps/v113-tame");
    let wild_maps_path = Path::new("../maps/v110c-wild");
    let samus_sprites_path = Path::new("../MapRandoSprites/samus_sprites/manifest.json");
    let title_screen_path = Path::new("../TitleScreen/Images");
    let game_data = GameData::load(
        sm_json_data_path,
        room_geometry_path,
        escape_timings_path,
        start_locations_path,
        hub_locations_path,
        title_screen_path,
        reduced_flashing_path,
    )?;

    if !args.output_seeds.is_dir() {
        bail!("{0} is not a directory", args.output_seeds.display());
    }

    info!("Loading logic preset data");
    let mut presets: Vec<Preset> =
        serde_json::from_str(&std::fs::read_to_string(&"data/presets.json")?)?;

    let mut implicit_tech: Vec<String> = vec![
        // Implicit tech
        "canSpecialBeamAttack",
        "canMidAirMorph",
        "canTurnaroundSpinJump",
        "canStopOnADime",
        "canUseGrapple",
        "canEscapeEnemyGrab",
        "canDownBack",
        "canTrivialUseFrozenEnemies",
    ]
    .into_iter()
    .map(|x| x.to_string())
    .collect();
    presets[0].tech.append(&mut implicit_tech);
    for ix in 0..(presets.len() - 1) {
        let mut tech = presets[ix].tech.clone();
        let mut strat = presets[ix].notable_strats.clone();
        presets[ix + 1].tech.append(&mut tech);
        presets[ix + 1].notable_strats.append(&mut strat);
    }

    // If we are using a locked-in preset, go ahead and remove all the others.
    if let Some(fixed_preset) = args.skill_preset {
        presets.retain(|x| x.name == fixed_preset);
        if presets.len() < 1 {
            bail!("Unknown skills preset {fixed_preset}");
        }
    } else {
        // Remove Beyond preset
        presets.retain(|x| x.name != "Beyond");
    }
    let etank_color_from_json: Vec<Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string(&etank_colors_path)?)?;
    let mut etank_colors_str: Vec<String> = vec![];
    for mut v in etank_color_from_json {
        etank_colors_str.append(&mut v);
    }

    let etank_colors = etank_colors_str
        .into_iter()
        .map(|x| {
            (
                u8::from_str_radix(&x[0..2], 16).unwrap() / 8,
                u8::from_str_radix(&x[2..4], 16).unwrap() / 8,
                u8::from_str_radix(&x[4..6], 16).unwrap() / 8,
            )
        })
        .collect();

    let samus_sprite_categories: Vec<SamusSpriteCategory> =
        serde_json::from_str(&std::fs::read_to_string(&samus_sprites_path)?)?;

    let progressions: Vec<fn(&mut DifficultyConfig) -> ()> =
        match args.progression_preset.as_deref() {
            Some("Normal") => {
                vec![set_item_progression_normal]
            }
            Some("Tricky") => {
                vec![set_item_progression_tricky]
            }
            Some("Challenge") => {
                vec![set_item_progression_challenge]
            }
            Some("Desolate") => {
                vec![set_item_progression_desolate]
            }
            Some(other) => {
                bail!("Unknown progression preset {other}");
            }
            None => {
                vec![
                    set_item_progression_normal,
                    set_item_progression_tricky,
                    set_item_progression_challenge,
                    set_item_progression_desolate,
                ]
            }
        };

    let qols: Vec<fn(&mut DifficultyConfig) -> ()> = match args.qol_preset.as_deref() {
        Some("Max") => {
            vec![set_qol_max]
        }
        Some("Default") => {
            vec![set_qol_default]
        }
        Some("Low") => {
            vec![set_qol_low]
        }
        Some("Off") => {
            vec![set_qol_off]
        }
        Some("Ultra-Low") => {
            vec![set_ultra_low_qol]
        }
        Some(other) => {
            bail!("Unknown QoL preset {other}");
        }
        None => {
            vec![set_qol_max, set_qol_default, set_qol_low, set_qol_off]
        }
    };

    let mut input_rom = Rom::load(&args.input_rom)?;
    if input_rom.data.len() < 0x300000 {
        bail!("Invalid base ROM");
    }
    let rom_digest = crypto_hash::hex_digest(crypto_hash::Algorithm::SHA256, &input_rom.data);
    if rom_digest != "12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72" {
        info!("Warning: use of non-vanilla base ROM! Digest = {rom_digest}");
    }
    input_rom.data.resize(0x400000, 0);

    let mut samus_sprites: Vec<String> = vec![];
    for cat in &samus_sprite_categories {
        for inf in &cat.sprites {
            samus_sprites.push(inf.name.clone());
        }
    }

    let app = TestAppData {
        attempt_num: args.attempt_num,
        input_rom,
        output_dir: args.output_seeds,
        game_data,
        map_repos: vec![
            (
                MapRepository::new("Vanilla", vanilla_map_path)?,
                Some(|diff| {
                    diff.vanilla_map = true;
                    diff.tech.push("canEscapeMorphLocation".to_string())
                }),
            ),
            (MapRepository::new("Tame", tame_maps_path)?, None),
            (MapRepository::new("Wild", wild_maps_path)?, None),
        ],
        presets,
        progressions,
        qols,
        etank_colors,
        samus_sprite_categories,
        samus_sprites,
    };

    for test_cycle in 0..args.test_cycles {
        perform_test_cycle(&app, test_cycle + 1)
            .with_context(|| "Failed during test cycle {test_cycle + 1}")?;
        if args.attempt_num.is_some() {
            break;
        }
    }

    Ok(())
}
