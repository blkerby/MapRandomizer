use anyhow::{Context, Result, bail};
use clap::Parser;
use log::{error, info};
use maprando::customize::samus_sprite::SamusSpriteCategory;
use maprando::customize::{ControllerConfig, CustomizeSettings, MusicSettings};
use maprando::difficulty::{get_full_global, get_link_difficulty_length};
use maprando::map_repository::MapRepository;
use maprando::patch::Rom;
use maprando::patch::make_rom;
use maprando::preset::PresetData;
use maprando::randomize::{
    Randomization, Randomizer, get_difficulty_tiers, get_objectives, order_map_areas,
    randomize_doors, randomize_map_areas,
};
use maprando::settings::{
    AreaAssignment, ItemProgressionSettings, QualityOfLifeSettings, RandomizerSettings,
    SkillAssumptionSettings, StartLocationMode,
};
use maprando::spoiler_log::SpoilerLog;
use maprando::spoiler_map;
use maprando_game::{GameData, Map};
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
    base_preset: RandomizerSettings,
    skill_presets: Vec<SkillAssumptionSettings>,
    item_presets: Vec<ItemProgressionSettings>,
    qol_presets: Vec<QualityOfLifeSettings>,
    etank_colors: Vec<(u8, u8, u8)>,
    samus_sprite_categories: Vec<SamusSpriteCategory>,
    samus_sprites: Vec<String>,
}

fn get_randomization(
    app: &TestAppData,
    seed: u64,
) -> Result<(RandomizerSettings, Randomization, SpoilerLog, String)> {
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

    let mut settings = app.base_preset.clone();
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

    info!("Generating seed using Skills {skill_label}, Progression {item_label}, QoL {qol_label}");

    let difficulty_tiers = get_difficulty_tiers(
        &settings,
        &app.preset_data.difficulty_tiers,
        game_data,
        &app.preset_data.tech_by_difficulty["Implicit"],
        &app.preset_data.notables_by_difficulty["Implicit"],
    );

    let random_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&random_seed.to_le_bytes());
    rng_seed[9] = 0; // Not race-mode
    rng = rand::rngs::StdRng::from_seed(rng_seed);

    let max_attempts = 10000;
    let max_attempts_per_map = if settings.start_location_settings.mode == StartLocationMode::Random
    {
        10
    } else {
        1
    };
    let max_map_attempts = max_attempts / max_attempts_per_map;
    let mut attempt_num = 0;

    let output_file_prefix = format!("{skill_label}-{item_label}-{qol_label}-{random_seed}");

    // Save a dump of the settings
    let settings_json = serde_json::to_string(&settings)?;
    std::fs::write(
        Path::join(
            &app.output_dir,
            format!("{output_file_prefix}-settings.json"),
        ),
        settings_json,
    )?;

    let mut map_batch: Vec<Map> = vec![];

    for _ in 0..max_map_attempts {
        let map_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        let door_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;

        if map_batch.is_empty() {
            map_batch = map_repo.get_map_batch(map_seed, game_data).unwrap();
        }

        let mut map = map_batch.pop().unwrap();
        match settings.other_settings.area_assignment {
            AreaAssignment::Ordered => {
                order_map_areas(&mut map, map_seed, game_data);
            }
            AreaAssignment::Random => {
                randomize_map_areas(&mut map, map_seed);
            }
            AreaAssignment::Standard => {}
        }
        let objectives = get_objectives(&settings, Some(&map), game_data, &mut rng);
        let locked_door_data = randomize_doors(game_data, &map, &settings, &objectives, door_seed);
        let randomizer = Randomizer::new(
            &map,
            &locked_door_data,
            objectives,
            &settings,
            &difficulty_tiers,
            game_data,
            &game_data.base_links_data,
            &mut rng,
        );
        for _ in 0..max_attempts_per_map {
            attempt_num += 1;
            let item_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
            info!(
                "Attempt {attempt_num}/{max_attempts}: Map seed={map_seed}, door randomization seed={door_seed}, item placement seed={item_seed}"
            );
            match randomizer.randomize(attempt_num, item_seed, 1) {
                Ok((randomization, spoiler_log)) => {
                    return Ok((settings, randomization, spoiler_log, output_file_prefix));
                }
                Err(e) => {
                    info!("Attempt {attempt_num}/{max_attempts}: Randomization failed: {e}");
                }
            }
        }
    }
    bail!("Exhausted randomization attempts");
}

fn make_random_customization(app: &TestAppData) -> CustomizeSettings {
    let mut rng = rand::rngs::StdRng::from_entropy();

    let mut possible_sprites: Vec<Option<String>> =
        app.samus_sprites.clone().into_iter().map(Some).collect();
    possible_sprites.push(None);

    let mut possible_etanks: Vec<Option<(u8, u8, u8)>> =
        app.etank_colors.clone().into_iter().map(Some).collect();
    possible_etanks.push(None);

    let bits = rng.next_u64();

    CustomizeSettings {
        samus_sprite: possible_sprites[rng.next_u64() as usize % possible_sprites.len()].clone(),
        etank_color: possible_etanks[rng.next_u64() as usize % possible_etanks.len()],
        item_dot_change: maprando::customize::ItemDotChange::Fade,
        transition_letters: true,
        reserve_hud_style: bits & 0x01 != 0,
        vanilla_screw_attack_animation: bits & 0x02 != 0,
        room_names: true,
        palette_theme: maprando::customize::PaletteTheme::AreaThemed,
        tile_theme: maprando::customize::TileTheme::Vanilla,
        door_theme: maprando::customize::DoorTheme::Vanilla,
        music: match bits & 0x04 != 0 {
            true => MusicSettings::Disabled,
            false => MusicSettings::AreaThemed,
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
    }
}

fn perform_test_cycle(app: &TestAppData, cycle_count: usize) -> Result<()> {
    let seed: u64 = app.attempt_num.unwrap_or(cycle_count as u64);

    info!("Test cycle {cycle_count} Start: seed={seed}");

    // Perform randomization (map selection & item placement):
    let (settings, randomization, spoiler_log, output_file_prefix) = get_randomization(app, seed)?;
    let customize_settings = make_random_customization(app);

    // Generate the patched ROM:
    let game_rom = make_rom(
        &app.input_rom,
        &settings,
        &customize_settings,
        &randomization,
        &app.game_data,
        &app.samus_sprite_categories,
        &[],
    )?;
    let output_rom = game_rom.clone();

    std::fs::write(
        Path::join(
            &app.output_dir,
            format!("{output_file_prefix}-customize.txt"),
        ),
        format!("{customize_settings:?}"),
    )?;

    let output_rom_path = Path::join(&app.output_dir, format!("{output_file_prefix}-rom.smc"));
    info!("Writing output ROM to {}", output_rom_path.display());
    output_rom.save(&output_rom_path)?;

    let output_spoiler_log_path = Path::join(
        &app.output_dir,
        format!("{output_file_prefix}-spoiler.json"),
    );
    info!(
        "Writing spoiler log to {}",
        output_spoiler_log_path.display()
    );
    let spoiler_str = serde_json::to_string_pretty(&spoiler_log)?;
    std::fs::write(output_spoiler_log_path, spoiler_str)?;

    let spoiler_maps =
        spoiler_map::get_spoiler_map(&randomization, &app.game_data, &settings, true)?;
    let spoiler_maps_small =
        spoiler_map::get_spoiler_map(&randomization, &app.game_data, &settings, true)?;

    let output_spoiler_map_explored_path = Path::join(
        &app.output_dir,
        format!("{output_file_prefix}-explored.png"),
    );

    info!(
        "Writing spoiler map (explored) to {}",
        output_spoiler_map_explored_path.display()
    );

    let spoiler_map_explored = spoiler_maps.explored.clone();
    std::fs::write(output_spoiler_map_explored_path, spoiler_map_explored)?;

    let output_spoiler_map_explored_small_path = Path::join(
        &app.output_dir,
        format!("{output_file_prefix}-explored-small.png"),
    );

    info!(
        "Writing spoiler map (explored) to {}",
        output_spoiler_map_explored_small_path.display()
    );

    let spoiler_map_explored_small = spoiler_maps_small.explored.clone();
    std::fs::write(
        output_spoiler_map_explored_small_path,
        spoiler_map_explored_small,
    )?;

    let output_spoiler_map_outline_path =
        Path::join(&app.output_dir, format!("{output_file_prefix}-outline.png"));

    info!(
        "Writing spoiler map (vanilla areas) to {}",
        output_spoiler_map_outline_path.display()
    );
    let spoiler_map_outline = spoiler_maps.outline.clone();
    std::fs::write(output_spoiler_map_outline_path, spoiler_map_outline)?;

    Ok(())
}

fn build_app_data(args: &Args) -> Result<TestAppData> {
    let etank_colors_path = Path::new("data/etank_colors.json");
    let vanilla_map_path = Path::new("../maps/vanilla");
    let small_maps_path = Path::new("../maps/v119-small-avro");
    let standard_maps_path = Path::new("../maps/v119-standard-avro");
    let wild_maps_path = Path::new("../maps/v119-wild-avro");
    let samus_sprites_path = Path::new("../MapRandoSprites/samus_sprites/manifest.json");
    let mut game_data = GameData::load(Path::new("."))?;

    if !args.output_seeds.is_dir() {
        bail!("{0} is not a directory", args.output_seeds.display());
    }

    info!("Loading logic preset data");
    let tech_path = Path::new("data/tech_data.json");
    let notable_path = Path::new("data/notable_data.json");
    let presets_path = Path::new("data/presets");
    let preset_data = PresetData::load(tech_path, notable_path, presets_path, &game_data)?;
    let global = get_full_global(&game_data);
    game_data.make_links_data(&|link, game_data| {
        get_link_difficulty_length(link, game_data, &preset_data, &global)
    });
    let mut base_preset = preset_data.default_preset.clone();

    base_preset.start_location_settings.mode = StartLocationMode::Random;

    if let Some(fixed_preset) = &args.preset {
        let path = if fixed_preset.ends_with(".json") {
            fixed_preset.clone()
        } else {
            format!("data/presets/full-settings/{fixed_preset}.json")
        };
        let s =
            std::fs::read_to_string(&path).context(format!("Unable to read {}", path.as_str()))?;
        base_preset = serde_json::from_str(&s)?;
    }

    let mut skill_presets = preset_data.skill_presets.clone();
    // If we are using a locked-in preset, go ahead and remove all the others.
    if let Some(fixed_preset) = &args.skill_preset {
        let path = format!("data/presets/skill-assumptions/{fixed_preset}.json");
        let s = std::fs::read_to_string(&path)
            .context(format!("Unable to load skill preset: {path}"))?;
        let p: SkillAssumptionSettings = serde_json::from_str(&s)?;
        skill_presets = vec![p];
    } else {
        // Remove Implicit and Ignored preset
        skill_presets.retain(|x| {
            x.preset.as_ref().unwrap() != "Implicit" && x.preset.as_ref().unwrap() != "Beyond"
        });
    }

    let mut item_presets = preset_data.item_progression_presets.clone();
    // If we are using a locked-in preset, go ahead and remove all the others.
    if let Some(fixed_preset) = &args.item_preset {
        let path = format!("data/presets/item-progression/{fixed_preset}.json");
        let s = std::fs::read_to_string(&path)
            .context(format!("Unable to load item progression preset: {path}"))?;
        let p: ItemProgressionSettings = serde_json::from_str(&s)?;
        item_presets = vec![p];
    }

    let mut qol_presets = preset_data.quality_of_life_presets.clone();
    // If we are using a locked-in preset, go ahead and remove all the others.
    if let Some(fixed_preset) = &args.qol_preset {
        let path = format!("data/presets/quality-of-life/{fixed_preset}.json");
        let s =
            std::fs::read_to_string(&path).context(format!("Unable to load QoL preset: {path}"))?;
        let p: QualityOfLifeSettings = serde_json::from_str(&s)?;
        qol_presets = vec![p];
    }

    let etank_color_from_json: Vec<Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string(etank_colors_path)?)?;
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
        serde_json::from_str(&std::fs::read_to_string(samus_sprites_path)?)?;

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
            MapRepository::new("Small", small_maps_path)?,
            MapRepository::new("Standard", standard_maps_path)?,
            MapRepository::new("Wild", wild_maps_path)?,
        ],
        base_preset,
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
    let mut error_vec = vec![];
    for test_cycle in 0..args.test_cycles {
        if let Err(e) = perform_test_cycle(&app_data, test_cycle) {
            error!("Failed during test cycle {test_cycle}: {e}");
            error_vec.push((test_cycle, e));
        }
        if args.attempt_num.is_some() {
            break;
        }
    }
    for (test_cycle, e) in error_vec {
        error!("Failed during test cycle {test_cycle}: {}", e);
        // error!("Failed during test cycle {test_cycle}: {e}");
    }

    Ok(())
}
