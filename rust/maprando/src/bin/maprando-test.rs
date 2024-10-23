use anyhow::{bail, Context, Result};
use clap::Parser;
use log::info;
use maprando::customize::samus_sprite::SamusSpriteCategory;
use maprando::customize::{customize_rom, ControllerConfig, CustomizeSettings, MusicSettings};
use maprando::map_repository::MapRepository;
use maprando::patch::ips_write::create_ips_patch;
use maprando::patch::Rom;
use maprando::preset::PresetData;
use maprando::randomize::{
    get_difficulty_tiers, get_objectives, randomize_doors, randomize_map_areas, ItemPriorityGroup, Objective, Randomization, Randomizer
};
use maprando::settings::{
    AreaAssignment, DoorLocksSize, DoorsMode, ETankRefill, ItemDotChange, ItemMarkers, ItemPlacementStyle, ItemPriorityStrength, ItemProgressionSettings, KeyItemPriority, MapStationReveal, MapsRevealed, MotherBrainFight, ProgressionRate, QualityOfLifeSettings, RandomizerSettings, SaveAnimals, SkillAssumptionSettings, StartLocationMode, WallJump
};
use maprando::spoiler_map;
use maprando::{patch::make_rom, randomize::DifficultyConfig};
use maprando_game::{
    Capacity, GameData, Item, NotableId, RoomId, TechId,
};
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
    preset: Option<String>,

    #[arg(long)]
    skill_preset: Option<String>,

    #[arg(long)]
    item_preset: Option<String>,

    #[arg(long)]
    qol_preset: Option<String>,
}

// Reduced version of web::AppData for test tool
struct TestAppData {
    attempt_num: Option<u64>,
    input_rom: Rom,
    output_dir: PathBuf,
    game_data: GameData,
    preset_data: PresetData,
    map_repos: Vec<MapRepository>,
    skill_presets: Vec<SkillAssumptionSettings>,
    item_presets: Vec<ItemProgressionSettings>,
    qol_presets: Vec<QualityOfLifeSettings>,
    etank_colors: Vec<(u8, u8, u8)>,
    samus_sprite_categories: Vec<SamusSpriteCategory>,
    samus_sprites: Vec<String>,
}

fn get_randomization(app: &TestAppData, seed: u64) -> Result<(Randomization, String)> {
    let game_data = &app.game_data;
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

    let preset_idx = rng.next_u64() as usize % app.skill_presets.len();
    let progression_idx = rng.next_u64() as usize % app.item_presets.len();
    let qol_idx = rng.next_u64() as usize % app.qol_presets.len();
    let repo_idx = rng.next_u64() as usize % app.map_repos.len();

    let skill_preset = &app.skill_presets[preset_idx];
    let item_preset = &app.item_presets[progression_idx];
    let qol_preset = &app.qol_presets[qol_idx];
    let map_repo = &app.map_repos[repo_idx];

    let mut settings = app.preset_data.default_preset.clone();
    settings.skill_assumption_settings = skill_preset.clone();
    settings.item_progression_settings = item_preset.clone();
    settings.quality_of_life_settings = qol_preset.clone();

    let skill_label = match &settings.skill_assumption_settings.preset {
        Some(s) => s.clone(),
        None => String::from("Custom"),
    };
    let item_label = match &settings.item_progression_settings.preset {
        Some(s) => s.clone(),
        None => String::from("Custom"),
    };
    let qol_label = match &settings.quality_of_life_settings.preset {
        Some(s) => s.clone(),
        None => String::from("Custom"),
    };

    info!(
        "Generating seed using Skills {0}, Progression {1}, QoL {2}",
        skill_label, item_label, qol_label
    );

    let difficulty_tiers = get_difficulty_tiers(
        &settings, &app.preset_data.difficulty_tiers, game_data,
        &app.preset_data.tech_by_difficulty["Implicit"], 
        &app.preset_data.notables_by_difficulty["Implicit"]);

    let random_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&random_seed.to_le_bytes());
    rng_seed[9] = 0; // Not race-mode
    rng = rand::rngs::StdRng::from_seed(rng_seed);

    let max_attempts = 10000;
    let max_attempts_per_map = if settings.start_location_mode == StartLocationMode::Random {
        10
    } else {
        1
    };
    let max_map_attempts = max_attempts / max_attempts_per_map;
    let mut attempt_num = 0;

    let output_file_prefix = format!(
        "{0}-{1}-{2}-{3}",
        skill_label, item_label, qol_label, random_seed
    );

    // Save a dump of the settings
    let settings_json = serde_json::to_string(&settings)?;
    std::fs::write(
        Path::join(
            &app.output_dir,
            format!("{output_file_prefix}-settings.json"),
        ),
        settings_json,
    )?;

    for _ in 0..max_map_attempts {
        let map_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        let door_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        let mut map = map_repo.get_map(attempt_num, map_seed, game_data)?;
        if settings.other_settings.area_assignment == AreaAssignment::Random {
            randomize_map_areas(&mut map, map_seed);
        }
        let objectives = get_objectives(&settings, &mut rng);
        let locked_door_data = randomize_doors(game_data, &map, &settings, &objectives, door_seed);
        let randomizer = Randomizer::new(
            &map,
            &locked_door_data,
            objectives,
            &settings,
            &difficulty_tiers,
            &game_data,
            &game_data.base_links_data,
            &mut rng,
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

fn build_app_data(args: &Args) -> Result<TestAppData> {
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let escape_timings_path = Path::new("data/escape_timings.json");
    let start_locations_path = Path::new("data/start_locations.json");
    let hub_locations_path = Path::new("data/hub_locations.json");
    let etank_colors_path = Path::new("data/etank_colors.json");
    let reduced_flashing_path = Path::new("data/reduced_flashing.json");
    let strat_videos_path = Path::new("data/strat_videos.json");
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
        strat_videos_path,
    )?;

    if !args.output_seeds.is_dir() {
        bail!("{0} is not a directory", args.output_seeds.display());
    }

    info!("Loading logic preset data");
    let tech_path = Path::new("data/tech_data.json");
    let notable_path = Path::new("data/notable_data.json");
    let presets_path = Path::new("data/presets");
    let preset_data = PresetData::load(tech_path, notable_path, presets_path, &game_data)?;
    
    let mut skill_presets = preset_data.skill_presets.clone();
    // If we are using a locked-in preset, go ahead and remove all the others.
    if let Some(fixed_preset) = &args.skill_preset {
        let path = format!("data/presets/skill-assumptions/{}.json", fixed_preset);
        let s = std::fs::read_to_string(path)?;
        let p: SkillAssumptionSettings = serde_json::from_str(&s)?;
        skill_presets = vec![p];
    } else {
        // Remove Implicit and Ignored preset
        skill_presets.retain(|x| x.preset.as_ref().unwrap() != "Implicit" && x.preset.as_ref().unwrap() != "Beyond");
    }

    let mut item_presets = preset_data.item_progression_presets.clone();
    // If we are using a locked-in preset, go ahead and remove all the others.
    if let Some(fixed_preset) = &args.item_preset {
        let path = format!("data/presets/item-progression/{}.json", fixed_preset);
        let s = std::fs::read_to_string(path)?;
        let p: ItemProgressionSettings = serde_json::from_str(&s)?;
        item_presets = vec![p];
    }

    let mut qol_presets = preset_data.quality_of_life_presets.clone();
    // If we are using a locked-in preset, go ahead and remove all the others.
    if let Some(fixed_preset) = &args.item_preset {
        let path = format!("data/presets/quality-of-life/{}.json", fixed_preset);
        let s = std::fs::read_to_string(path)?;
        let p: QualityOfLifeSettings = serde_json::from_str(&s)?;
        qol_presets = vec![p];
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

    let mut samus_sprites: Vec<String> = vec![];
    for cat in &samus_sprite_categories {
        for inf in &cat.sprites {
            samus_sprites.push(inf.name.clone());
        }
    }

    let mut input_rom = Rom::load(&args.input_rom)?;
    let rom_digest = crypto_hash::hex_digest(crypto_hash::Algorithm::SHA256, &input_rom.data);
    if rom_digest != "12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72" {
        info!("Warning: use of non-vanilla base ROM! Digest = {rom_digest}");
    }
    input_rom.data.resize(0x400000, 0);

    let app = TestAppData {
        attempt_num: args.attempt_num,
        input_rom,
        output_dir: args.output_seeds.clone(),
        game_data,
        preset_data,
        map_repos: vec![
            MapRepository::new("Vanilla", vanilla_map_path)?,
            MapRepository::new("Tame", tame_maps_path)?,
            MapRepository::new("Wild", wild_maps_path)?,
        ],
        skill_presets,
        item_presets,
        qol_presets,
        etank_colors,
        samus_sprite_categories,
        samus_sprites,
    };
    Ok(app)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args = Args::parse();
    let app_data = build_app_data(&args)?;

    for test_cycle in 0..args.test_cycles {
        perform_test_cycle(&app_data, test_cycle + 1)
            .with_context(|| "Failed during test cycle {test_cycle + 1}")?;
        if args.attempt_num.is_some() {
            break;
        }
    }

    Ok(())
}
