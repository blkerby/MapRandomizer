use anyhow::{Context, Result, bail};
use clap::Parser;
use log::info;
use maprando::customize::samus_sprite::{SamusSpriteCategory, SamusSpriteInfo};
use maprando::customize::{ControllerConfig, CustomizeSettings, MusicSettings};
use maprando::patch::Rom;
use maprando::patch::make_rom;
use maprando::preset::PresetData;
use maprando::randomize::{
    Randomization, Randomizer, get_difficulty_tiers, get_objectives, randomize_doors,
};
use maprando::settings::{RandomizerSettings, StartLocationMode};
use maprando::spoiler_log::SpoilerLog;
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
    Ok(settings)
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
    let mut filenames: Vec<String> = Vec::new();
    let single_map: Option<Map> = if args.map.is_dir() {
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
        None
    } else {
        let map_string = std::fs::read_to_string(&args.map)
            .with_context(|| format!("Unable to read map file at {}", args.map.display()))?;
        Some(
            serde_json::from_str(&map_string)
                .with_context(|| format!("Unable to parse map file at {}", args.map.display()))?,
        )
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
        let objectives = get_objectives(settings, Some(&map), game_data, &mut rng);
        let locked_door_data = randomize_doors(game_data, &map, settings, &objectives, door_seed);
        let randomizer = Randomizer::new(
            &map,
            &locked_door_data,
            objectives,
            settings,
            &difficulty_tiers,
            game_data,
            &game_data.base_links_data,
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
            match randomizer.randomize(attempt_num, item_seed, 1) {
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
    let mut game_data = GameData::load()?;

    if let Some(start_location_name) = &args.start_location {
        game_data
            .start_locations
            .retain(|x| &x.name == start_location_name);
    }

    let tech_path = Path::new("data/tech_data.json");
    let notable_path = Path::new("data/notable_data.json");
    let presets_path = Path::new("data/presets");
    let preset_data = PresetData::load(tech_path, notable_path, presets_path, &game_data)?;
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
        etank_color: None,
        item_dot_change: maprando::customize::ItemDotChange::Fade,
        transition_letters: true,
        reserve_hud_style: true,
        vanilla_screw_attack_animation: true,
        room_names: true,
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

    if let Some(output_spoiler_map_explored_path) = &args.output_spoiler_map_explored {
        println!(
            "Writing spoiler map (explored) to {}",
            output_spoiler_map_explored_path.display()
        );
        let spoiler_map_explored = spoiler_maps.explored.clone();
        std::fs::write(output_spoiler_map_explored_path, spoiler_map_explored)?;
        let spoiler_map_explored_small = spoiler_maps.explored_small.clone();
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
