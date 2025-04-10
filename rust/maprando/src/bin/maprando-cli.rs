use anyhow::{bail, Context, Result};
use clap::Parser;
use log::info;
use maprando::customize::samus_sprite::{SamusSpriteCategory, SamusSpriteInfo};
use maprando::customize::{customize_rom, ControllerConfig, CustomizeSettings, MusicSettings};
use maprando::patch::ips_write::create_ips_patch;
use maprando::patch::make_rom;
use maprando::patch::Rom;
use maprando::preset::PresetData;
use maprando::randomize::{
    get_difficulty_tiers, get_objectives, randomize_doors, Randomization, Randomizer,
};
use maprando::settings::{RandomizerSettings, StartLocationMode};
use maprando::spoiler_map;
use maprando_game::{GameData, Map};
use rand::{RngCore, SeedableRng};
use std::path::{Path, PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    map: PathBuf,

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
}

fn get_settings(args: &Args, preset_data: &PresetData) -> Result<RandomizerSettings> {
    let mut settings = preset_data.default_preset.clone();

    if let Some(preset) = &args.preset {
        let path = format!("data/presets/full-settings/{}.json", preset);
        let s = std::fs::read_to_string(path)?;
        settings = serde_json::from_str(&s)?;
    }
    if let Some(skill_preset) = &args.skill_preset {
        let path = format!("data/presets/skill-assumptions/{}.json", skill_preset);
        let s = std::fs::read_to_string(path)?;
        settings.skill_assumption_settings = serde_json::from_str(&s)?;
    }
    if let Some(item_preset) = &args.item_preset {
        let path = format!("data/presets/item-progression/{}.json", item_preset);
        let s = std::fs::read_to_string(path)?;
        settings.item_progression_settings = serde_json::from_str(&s)?;
    }
    if let Some(qol_preset) = &args.qol_preset {
        let path = format!("data/presets/item-quality-of-life/{}.json", qol_preset);
        let s = std::fs::read_to_string(path)?;
        settings.quality_of_life_settings = serde_json::from_str(&s)?;
    }
    settings.other_settings.random_seed = args.random_seed;
    Ok(settings)
}

fn get_randomization(
    args: &Args,
    settings: &RandomizerSettings,
    game_data: &GameData,
    preset_data: &PresetData,
) -> Result<Randomization> {
    let implicit_tech = &preset_data.tech_by_difficulty["Implicit"];
    let implicit_notables = &preset_data.notables_by_difficulty["Implicit"];
    let difficulty_tiers = get_difficulty_tiers(
        &settings,
        &preset_data.difficulty_tiers,
        game_data,
        implicit_tech,
        implicit_notables,
    );
    let single_map: Option<Map>;
    let mut filenames: Vec<String> = Vec::new();
    if args.map.is_dir() {
        for path in std::fs::read_dir(&args.map)
            .with_context(|| format!("Unable to read maps in directory {}", args.map.display()))?
        {
            filenames.push(path?.file_name().into_string().unwrap());
        }
        filenames.sort();
        info!(
            "{} maps available ({})",
            filenames.len(),
            args.map.display()
        );
        single_map = None;
    } else {
        let map_string = std::fs::read_to_string(&args.map)
            .with_context(|| format!("Unable to read map file at {}", args.map.display()))?;
        single_map = Some(
            serde_json::from_str(&map_string)
                .with_context(|| format!("Unable to parse map file at {}", args.map.display()))?,
        );
    }
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
        match args.max_attempts {
            Some(ma) => ma,
            None => 10000, // Same as maprando-web.
        }
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
    for _ in 0..max_map_attempts {
        let map_seed = (rng.next_u64() & 0xFFFFFFFF) as usize;
        let map = match single_map {
            Some(ref m) => m.clone(),
            None => {
                let idx = map_seed % filenames.len();
                let path = args.map.join(&filenames[idx]);
                let map_string = std::fs::read_to_string(&path)
                    .with_context(|| format!("Unable to read map file at {}", path.display()))?;
                info!("[attempt {attempt_num}] Map: {}", path.display());
                serde_json::from_str(&map_string).with_context(|| {
                    format!("Unable to parse map file at {}", args.map.display())
                })?
            }
        };
        let door_seed = match args.item_placement_seed {
            Some(s) => s,
            None => (rng.next_u64() & 0xFFFFFFFF) as usize,
        };
        let objectives = get_objectives(&settings, &mut rng);
        let locked_door_data = randomize_doors(game_data, &map, settings, &objectives, door_seed);
        let randomizer = Randomizer::new(
            &map,
            &locked_door_data,
            objectives,
            settings,
            &difficulty_tiers,
            &game_data,
            &game_data.base_links_data,
            &mut rng,
        );
        for _ in 0..max_attempts_per_map {
            attempt_num += 1;
            let item_seed = match args.item_placement_seed {
                Some(s) => s,
                None => (rng.next_u64() & 0xFFFFFFFF) as usize,
            };
            info!("Attempt {attempt_num}/{max_attempts}: Map seed={map_seed}, door randomization seed={door_seed}, item placement seed={item_seed}");
            match randomizer.randomize(attempt_num, item_seed, 1) {
                Ok(randomization) => {
                    return Ok(randomization);
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
    let reduced_flashing_path = Path::new("data/reduced_flashing.json");
    let strat_videos_path = Path::new("data/strat_videos.json");
    let title_screen_path = Path::new("../TitleScreen/Images");
    let map_tiles_path = Path::new("data/map_tiles.json");
    let mut game_data = GameData::load(
        sm_json_data_path,
        room_geometry_path,
        escape_timings_path,
        start_locations_path,
        hub_locations_path,
        title_screen_path,
        reduced_flashing_path,
        strat_videos_path,
        map_tiles_path,
    )?;

    if let Some(start_location_name) = &args.start_location {
        game_data.start_locations = game_data
            .start_locations
            .iter()
            .cloned()
            .filter(|x| &x.name == start_location_name)
            .collect();
    }

    let tech_path = Path::new("data/tech_data.json");
    let notable_path = Path::new("data/notable_data.json");
    let presets_path = Path::new("data/presets");
    let preset_data = PresetData::load(tech_path, notable_path, presets_path, &game_data)?;
    let settings = get_settings(&args, &preset_data)?;

    // Perform randomization (map selection & item placement):
    let randomization = get_randomization(&args, &settings, &game_data, &preset_data)?;

    // Generate the patched ROM:
    let orig_rom = Rom::load(&args.input_rom)?;
    let mut input_rom = orig_rom.clone();
    input_rom.data.resize(0x400000, 0);
    let game_rom = make_rom(&input_rom, &randomization, &game_data)?;
    let ips_patch = create_ips_patch(&input_rom.data, &game_rom.data);

    let mut output_rom = input_rom.clone();
    let customize_settings = CustomizeSettings {
        samus_sprite: Some("samus_vanilla".to_string()),
        // samus_sprite: None,
        etank_color: None,
        reserve_hud_style: true,
        vanilla_screw_attack_animation: true,
        palette_theme: maprando::customize::PaletteTheme::AreaThemed,
        tile_theme: maprando::customize::TileTheme::Vanilla,
        door_theme: maprando::customize::DoorTheme::Vanilla,
        music: MusicSettings::AreaThemed,
        // music: MusicSettings::Vanilla,
        disable_beeping: false,
        shaking: maprando::customize::ShakingSetting::Vanilla,
        flashing: maprando::customize::FlashingSetting::Vanilla,
        controller_config: ControllerConfig::default(),
    };
    customize_rom(
        &mut output_rom,
        &orig_rom,
        &ips_patch,
        &Some(randomization.map.clone()),
        &customize_settings,
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
        &vec![],
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
        let spoiler_str = serde_json::to_string_pretty(&randomization.spoiler_log)?;
        std::fs::write(output_spoiler_log_path, spoiler_str)?;
    }

    let spoiler_maps = spoiler_map::get_spoiler_map(&output_rom, &randomization.map, &game_data)?;

    if let Some(output_spoiler_map_explored_path) = &args.output_spoiler_map_explored {
        println!(
            "Writing spoiler map (explored) to {}",
            output_spoiler_map_explored_path.display()
        );
        let spoiler_map_explored = spoiler_maps.explored.clone();
        std::fs::write(output_spoiler_map_explored_path, spoiler_map_explored)?;
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
