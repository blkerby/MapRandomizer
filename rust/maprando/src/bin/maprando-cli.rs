use anyhow::{Context, Result, bail};
use clap::Parser;
use log::info;
use maprando::customize::samus_sprite::{SamusSpriteCategory, SamusSpriteInfo};
use maprando::customize::{
    ControllerConfig, CustomizeSettings, MusicSettings, StatuesHallwayAudio, StatuesHallwayTiling,
};
use maprando::environment_logic::MAX_ENVIRONMENT_SOLVABILITY_ATTEMPTS;
use maprando::difficulty::{get_full_global, get_link_difficulty_length};
use maprando::map_repository::MapRepository;
use maprando::patch::Rom;
use maprando::patch::make_rom;
use maprando::preset::PresetData;
use maprando::randomize::{
    Randomization, Randomizer, RandomizerContext, assign_map_areas, get_difficulty_tiers,
    get_objectives, randomize_doors,
};
use maprando::settings::{DoorsSettings, RandomizerSettings, StartLocationMode};
use maprando::spoiler_log::SpoilerLog;
use maprando::spoiler_map;
use maprando_game::{GameData, Map};
use rand::{RngCore, SeedableRng};
use std::path::{Path, PathBuf};

#[derive(Parser)]
struct Args {
    /// Fixed map JSON file or directory (Vanilla layout only). Ignored for Standard/Wild/Small.
    #[arg(long)]
    map: Option<PathBuf>,

    /// Map pool: Standard (scrambled rooms), Wild, Small, or Vanilla (fixed layout).
    #[arg(long)]
    map_layout: Option<String>,

    #[arg(long)]
    preset: Option<String>,

    #[arg(long)]
    skill_preset: Option<String>,

    #[arg(long)]
    item_preset: Option<String>,

    #[arg(long)]
    qol_preset: Option<String>,

    #[arg(long)]
    random_seed: Option<usize>,

    #[arg(long)]
    start_location: Option<String>,

    #[arg(long)]
    item_placement_seed: Option<usize>,

    #[arg(long)]
    max_attempts: Option<usize>,

    #[arg(long)]
    input_rom: PathBuf,

    #[arg(long)]
    output_rom: Option<PathBuf>,

    #[arg(long)]
    output_spoiler_log: Option<PathBuf>,

    #[arg(long)]
    output_spoiler_map_explored: Option<PathBuf>,

    #[arg(long)]
    output_spoiler_map_outline: Option<PathBuf>,

    #[arg(long)]
    area_themed_palette: bool,

    /// Experimental: flood random dry rooms with water (logic, ROM FX, and pause map).
    #[arg(long)]
    randomize_water_environments: bool,

    /// Number of rooms to flood when --randomize-water-environments is set (default: 5).
    #[arg(long, default_value_t = 5)]
    water_room_count: u32,

    /// Flood dry rooms near the ship spawn (for testing). Uses blue doors only, skips
    /// logic changes so seed generation stays fast; swim physics and pause-map water apply.
    #[arg(long)]
    flood_near_spawn: bool,

    /// Experimental: heat random dry rooms (logic, ROM FX, and pause map).
    #[arg(long)]
    randomize_heat_environments: bool,

    /// Number of rooms to heat when --randomize-heat-environments is set (default: 5).
    #[arg(long, default_value_t = 5)]
    heat_room_count: u32,

    /// Heat dry rooms near the ship spawn (for testing). Uses blue doors only, skips
    /// logic changes so seed generation stays fast; heat FX and pause-map heat apply.
    #[arg(long)]
    heat_near_spawn: bool,

    /// Randomize the start location (instead of always starting at the ship).
    #[arg(long)]
    random_start: bool,
}

fn get_settings(args: &Args, preset_data: &PresetData) -> Result<RandomizerSettings> {
    let mut settings = preset_data.default_preset.clone();

    if let Some(preset) = &args.preset {
        let path = format!("data/presets/full-settings/{preset}.json");
        let s = std::fs::read_to_string(path)?;
        settings = serde_json::from_str(&s)?;
    }
    if let Some(skill_preset) = &args.skill_preset {
        let path = format!("data/presets/skill-assumptions/{skill_preset}.json");
        let s = std::fs::read_to_string(path)?;
        settings.skill_assumption_settings = serde_json::from_str(&s)?;
    }
    if let Some(item_preset) = &args.item_preset {
        let path = format!("data/presets/item-progression/{item_preset}.json");
        let s = std::fs::read_to_string(path)?;
        settings.item_progression_settings = serde_json::from_str(&s)?;
    }
    if let Some(qol_preset) = &args.qol_preset {
        let path = format!("data/presets/item-quality-of-life/{qol_preset}.json");
        let s = std::fs::read_to_string(path)?;
        settings.quality_of_life_settings = serde_json::from_str(&s)?;
    }
    settings.other_settings.random_seed = args.random_seed;
    if let Some(map_layout) = &args.map_layout {
        settings.map_layout = map_layout.clone();
    }
    settings.experimental_settings.randomize_water_environments =
        args.randomize_water_environments || args.flood_near_spawn;
    settings.experimental_settings.water_room_count = args.water_room_count;
    settings.experimental_settings.water_flood_near_spawn = args.flood_near_spawn;
    settings.experimental_settings.water_visual_only = args.flood_near_spawn;
    settings.experimental_settings.randomize_heat_environments =
        args.randomize_heat_environments || args.heat_near_spawn;
    settings.experimental_settings.heat_room_count = args.heat_room_count;
    settings.experimental_settings.heat_flood_near_spawn = args.heat_near_spawn;
    settings.experimental_settings.heat_visual_only = args.heat_near_spawn;
    if args.flood_near_spawn || args.heat_near_spawn {
        settings.doors_settings = DoorsSettings {
            preset: Some("Blue".to_string()),
            red_doors_count: 0,
            green_doors_count: 0,
            yellow_doors_count: 0,
            charge_doors_count: 0,
            ice_doors_count: 0,
            wave_doors_count: 0,
            spazer_doors_count: 0,
            plasma_doors_count: 0,
        };
    }
    if args.random_start {
        settings.start_location_settings.mode = StartLocationMode::Random;
    }
    Ok(settings)
}

fn map_repository_for_layout(map_layout: &str) -> Result<MapRepository> {
    let path = match map_layout {
        "Standard" => Path::new("../maps/v119-standard-avro"),
        "Wild" => Path::new("../maps/v119-wild-avro"),
        "Small" => Path::new("../maps/v119-small-avro"),
        "Vanilla" => Path::new("../maps/vanilla"),
        other => bail!("Unknown map layout {other:?}; expected Standard, Wild, Small, or Vanilla"),
    };
    MapRepository::new(map_layout, path)
        .with_context(|| format!("Unable to load {map_layout} map repository at {}", path.display()))
}

fn get_randomization(
    args: &Args,
    settings: &RandomizerSettings,
    game_data: &GameData,
    preset_data: &PresetData,
) -> Result<(Randomization, SpoilerLog)> {
    let implicit_tech = &preset_data.tech_by_difficulty["Implicit"];
    let implicit_notables = &preset_data.notables_by_difficulty["Implicit"];
    let difficulty_tiers = get_difficulty_tiers(
        settings,
        &preset_data.difficulty_tiers,
        game_data,
        implicit_tech,
        implicit_notables,
    );
    let map_layout = settings.map_layout.as_str();
    let use_map_repository = matches!(map_layout, "Standard" | "Wild" | "Small");
    let map_repository = if use_map_repository {
        Some(map_repository_for_layout(map_layout)?)
    } else {
        None
    };

    let mut json_map_filenames: Vec<String> = Vec::new();
    let single_json_map: Option<Map> = if use_map_repository {
        if args.map.is_some() {
            info!(
                "Ignoring --map because map_layout={map_layout} uses the scrambled map pool"
            );
        }
        None
    } else if let Some(map_path) = &args.map {
        if map_path.is_dir() {
            for path in std::fs::read_dir(map_path)
                .with_context(|| format!("Unable to read maps in directory {}", map_path.display()))?
            {
                json_map_filenames.push(path?.file_name().into_string().unwrap());
            }
            json_map_filenames.sort();
            info!(
                "{} maps available ({})",
                json_map_filenames.len(),
                map_path.display()
            );
            None
        } else {
            let map_string = std::fs::read_to_string(map_path)
                .with_context(|| format!("Unable to read map file at {}", map_path.display()))?;
            Some(serde_json::from_str(&map_string).with_context(|| {
                format!("Unable to parse map file at {}", map_path.display())
            })?)
        }
    } else {
        bail!(
            "Vanilla map layout requires --map pointing to a map JSON file or directory"
        );
    };
    let root_seed = match args.random_seed {
        Some(s) => s,
        None => (rand::rngs::StdRng::from_entropy().next_u64() & 0xFFFFFFFF) as usize,
    };
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&root_seed.to_le_bytes());
    rng_seed[9] = 0; // Not race-mode
    let mut rng = rand::rngs::StdRng::from_seed(rng_seed);
    let max_attempts = if args.item_placement_seed.is_some() {
        1
    } else {
        args.max_attempts.unwrap_or(10000) // Same as maprando-web.
    };
    let max_attempts_per_map = if settings.start_location_settings.mode == StartLocationMode::Random
        && game_data.start_locations.len() > 1
    {
        10
    } else {
        1
    };
    let max_map_attempts = max_attempts / max_attempts_per_map;
    let mut attempt_num = 0;
    let mut map_batch: Vec<Map> = vec![];
    for _ in 0..max_map_attempts {
        let map_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        let mut map = if let Some(map_repo) = &map_repository {
            if map_batch.is_empty() {
                map_batch = map_repo.get_map_batch(map_seed, game_data)?;
            }
            map_batch.pop().context("Map batch exhausted")?
        } else if let Some(ref m) = single_json_map {
            m.clone()
        } else {
            let map_path = args
                .map
                .as_ref()
                .context("Expected --map when using JSON map directory")?;
            let idx = map_seed % json_map_filenames.len();
            let path = map_path.join(&json_map_filenames[idx]);
            let map_string = std::fs::read_to_string(&path)
                .with_context(|| format!("Unable to read map file at {}", path.display()))?;
            info!("[attempt {attempt_num}] Map: {}", path.display());
            serde_json::from_str(&map_string).with_context(|| {
                format!("Unable to parse map file at {}", path.display())
            })?
        };
        if !assign_map_areas(&mut map, settings, map_seed, game_data) {
            info!("[attempt {attempt_num}] Area assignment failed for map seed={map_seed}");
            continue;
        }
        let door_seed = match args.item_placement_seed {
            Some(s) => s,
            None => (rng.next_u64() & 0xFFFFFFFF) as usize,
        };
        let objectives = get_objectives(settings, Some(&map), game_data, &mut rng);
        let locked_door_data = randomize_doors(game_data, &map, settings, &objectives, door_seed);
        let (ctx, water_assignments, heat_assignments, dry_water_assignments, dry_heat_assignments) =
            RandomizerContext::prepare_solvable(
                game_data,
                &map,
                settings,
                &locked_door_data,
                &objectives,
                &difficulty_tiers,
                door_seed,
                |gd| {
                    let global = get_full_global(gd);
                    gd.make_links_data(&|link, game_data| {
                        get_link_difficulty_length(link, game_data, preset_data, &global)
                    });
                },
            MAX_ENVIRONMENT_SOLVABILITY_ATTEMPTS,
            )?;
        let effective_game_data = ctx.effective_game_data(game_data);
        let randomizer = Randomizer::new(
            &map,
            &locked_door_data,
            objectives,
            settings,
            &difficulty_tiers,
            effective_game_data,
            &effective_game_data.base_links_data,
            water_assignments,
            heat_assignments,
            dry_water_assignments,
            dry_heat_assignments,
            &mut rng,
        );
        for _ in 0..max_attempts_per_map {
            attempt_num += 1;
            let item_seed = match args.item_placement_seed {
                Some(s) => s,
                None => (rng.next_u64() & 0xFFFFFFFF) as usize,
            };
            info!(
                "Attempt {attempt_num}/{max_attempts}: Map seed={map_seed}, door randomization seed={door_seed}, item placement seed={item_seed}"
            );
            match randomizer.randomize(attempt_num, item_seed, 1, true, false) {
                Ok(randomization) => {
                    return Ok(randomization);
                }
                Err(e) => {
                    info!("Attempt {attempt_num}/{max_attempts}: Randomization failed: {e}");
                }
            }
        }
    }
    bail!("Exhausted randomization attempts");
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args = Args::parse();
    let mut game_data = GameData::load(Path::new("."))?;

    if let Some(start_location_name) = &args.start_location {
        game_data
            .start_locations
            .retain(|x| &x.name == start_location_name);
    }

    let tech_path = Path::new("data/tech_data.json");
    let notable_path = Path::new("data/notable_data.json");
    let presets_path = Path::new("data/presets");
    let preset_data = PresetData::load(tech_path, notable_path, presets_path, &game_data)?;
    let global = get_full_global(&game_data);
    game_data.make_links_data(&|link, game_data| {
        get_link_difficulty_length(link, game_data, &preset_data, &global)
    });
    let settings = get_settings(&args, &preset_data)?;

    // Perform randomization (map selection & item placement):
    let (randomization, spoiler_log) =
        get_randomization(&args, &settings, &game_data, &preset_data)?;

    // Generate the patched ROM:
    let orig_rom = Rom::load(&args.input_rom)?;
    let mut input_rom = orig_rom.clone();
    input_rom.data.resize(0x400000, 0);

    let customize_settings = CustomizeSettings {
        samus_sprite: Some("samus_vanilla".to_string()),
        // samus_sprite: None,
        map_theme: maprando::customize::MapTheme::Light,
        etank_color: None,
        item_dot_change: maprando::customize::ItemDotChange::Fade,
        transition_letters: true,
        reserve_hud_style: true,
        vanilla_screw_attack_animation: true,
        save_icons: true,
        boss_icons: true,
        miniboss_icons: true,
        room_names: true,
        palette_theme: maprando::customize::PaletteTheme::AreaThemed,
        tile_theme: maprando::customize::TileTheme::Vanilla,
        door_theme: maprando::customize::DoorTheme::Vanilla,
        music: MusicSettings::AreaThemed,
        // music: MusicSettings::Vanilla,
        statues_hallway_tiling: StatuesHallwayTiling::Default,
        statues_hallway_audio: StatuesHallwayAudio::Enabled,
        disable_beeping: false,
        shaking: maprando::customize::ShakingSetting::Vanilla,
        flashing: maprando::customize::FlashingSetting::Vanilla,
        controller_config: ControllerConfig::default(),
    };

    let output_rom = make_rom(
        &input_rom,
        &settings,
        &customize_settings,
        &randomization,
        &game_data,
        &[SamusSpriteCategory {
            category_name: "category".to_string(),
            sprites: vec![SamusSpriteInfo {
                name: "samus_vanilla".to_string(),
                display_name: "Samus".to_string(),
                credits_name: None,
                authors: vec!["Nintendo".to_string()],
            }],
        }],
        &[],
    )?;

    // Save the outputs:
    if let Some(output_rom_path) = &args.output_rom {
        println!("Writing output ROM to {}", output_rom_path.display());
        output_rom.save(output_rom_path)?;
    }

    if let Some(output_spoiler_log_path) = &args.output_spoiler_log {
        println!(
            "Writing spoiler log to {}",
            output_spoiler_log_path.display()
        );
        let spoiler_str = serde_json::to_string_pretty(&spoiler_log)?;
        std::fs::write(output_spoiler_log_path, spoiler_str)?;
    }

    let spoiler_maps = spoiler_map::get_spoiler_map(&randomization, &game_data, &settings, true)?;
    let spoiler_maps_small =
        spoiler_map::get_spoiler_map(&randomization, &game_data, &settings, true)?;

    if let Some(output_spoiler_map_explored_path) = &args.output_spoiler_map_explored {
        println!(
            "Writing spoiler map (explored) to {}",
            output_spoiler_map_explored_path.display()
        );
        let spoiler_map_explored = spoiler_maps.explored.clone();
        std::fs::write(output_spoiler_map_explored_path, spoiler_map_explored)?;
        let spoiler_map_explored_small = spoiler_maps_small.explored.clone();
        std::fs::write(output_spoiler_map_explored_path, spoiler_map_explored_small)?;
    }

    if let Some(output_spoiler_map_outline_path) = &args.output_spoiler_map_outline {
        println!(
            "Writing spoiler map (outline) to {}",
            output_spoiler_map_outline_path.display()
        );
        let spoiler_map_outline = spoiler_maps.outline.clone();
        std::fs::write(output_spoiler_map_outline_path, spoiler_map_outline)?;
    }

    Ok(())
}
