use anyhow::{bail, Context, Result};
use clap::Parser;
use maprando::game_data::Map;
use maprando::patch::Rom;
use maprando::randomize::{ItemPlacementStrategy, Randomization, Randomizer};
use maprando::spoiler_map;
use maprando::{game_data::GameData, patch::make_rom, randomize::DifficultyConfig};
use std::path::{Path, PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    map: PathBuf,

    #[arg(long)]
    input_rom: PathBuf,

    #[arg(long)]
    output_rom: Option<PathBuf>,

    #[arg(long)]
    output_spoiler_log: Option<PathBuf>,

    #[arg(long)]
    output_spoiler_map_assigned: Option<PathBuf>,

    #[arg(long)]
    output_spoiler_map_vanilla: Option<PathBuf>,
}

fn get_randomization(args: &Args, game_data: &GameData) -> Result<Randomization> {
    let map_string = std::fs::read_to_string(&args.map)
        .with_context(|| format!("Unable to read map file at {}", args.map.display()))?;
    let map: Map = serde_json::from_str(&map_string)
        .with_context(|| format!("Unable to parse map file at {}", args.map.display()))?;

    let difficulty = DifficultyConfig {
        tech: game_data.tech_isv.keys.clone(),
        shine_charge_tiles: 16,
        // item_placement_strategy: ItemPlacementStrategy::Closed,
        item_placement_strategy: ItemPlacementStrategy::Semiclosed,
        resource_multiplier: 1.0,
        escape_timer_multiplier: 1.0,
        save_animals: false,
    };
    let randomizer = Randomizer::new(&map, &difficulty, &game_data);
    let max_attempts = 1;
    for attempt_num in 0..max_attempts {
        if let Some(randomization) = randomizer.randomize(attempt_num) {
            return Ok(randomization);
        } else {
            println!("Failed randomization attempt");
        }
    }
    bail!("Exhausted randomization attempts");
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let game_data = GameData::load(sm_json_data_path, room_geometry_path)?;

    // Perform randomization (map selection & item placement):
    let randomization = get_randomization(&args, &game_data)?;

    // Generate the patched ROM:
    let rom = make_rom(&Rom::load(&args.input_rom)?, &randomization, &game_data)?;

    // Save the outputs:
    if let Some(output_rom_path) = &args.output_rom {
        println!("Writing output ROM to {}", output_rom_path.display());
        rom.save(output_rom_path)?;
    }

    if let Some(output_spoiler_log_path) = &args.output_spoiler_log {
        println!(
            "Writing spoiler log to {}",
            output_spoiler_log_path.display()
        );
        let spoiler_str = serde_json::to_string_pretty(&randomization.spoiler_log)?;
        std::fs::write(output_spoiler_log_path, spoiler_str)?;
    }

    if let Some(output_spoiler_map_assigned_path) = &args.output_spoiler_map_assigned {
        println!(
            "Writing spoiler map (assigned areas) to {}",
            output_spoiler_map_assigned_path.display()
        );
        let spoiler_map_assigned =
            spoiler_map::get_spoiler_map(&rom, &randomization.map, &game_data, false)?;
        std::fs::write(output_spoiler_map_assigned_path, spoiler_map_assigned)?;
    }

    if let Some(output_spoiler_map_vanilla_path) = &args.output_spoiler_map_vanilla {
        println!(
            "Writing spoiler map (vanilla areas) to {}",
            output_spoiler_map_vanilla_path.display()
        );
        let spoiler_map_vanilla =
            spoiler_map::get_spoiler_map(&rom, &randomization.map, &game_data, true)?;
        std::fs::write(output_spoiler_map_vanilla_path, spoiler_map_vanilla)?;
    }
    Ok(())
}
