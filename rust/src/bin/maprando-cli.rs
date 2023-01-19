use maprando::game_data::Map;
use maprando::{game_data::GameData, randomize::DifficultyConfig, patch::make_rom};
use maprando::randomize::{Randomizer, ItemPlacementStrategy, Randomization};
use rand::SeedableRng;
use std::{path::{Path, PathBuf}};
use clap::Parser;
use anyhow::{Context, Result, bail};

#[derive(Parser)]
struct Args {
    map: PathBuf,
    input_rom_path: PathBuf,
    output_rom_path: PathBuf,
    output_spoiler_path: PathBuf,
}

fn get_randomization(args: &Args, game_data: &GameData) -> Result<Randomization> {
    let map_string = std::fs::read_to_string(&args.map)
        .with_context(|| format!("Unable to read map file at {}", args.map.display()))?;
    let map: Map = serde_json::from_str(&map_string)
        .with_context(|| format!("Unable to parse map file at {}", args.map.display()))?;

    let difficulty = DifficultyConfig {
        tech: game_data.tech_isv.keys.clone(),
        shine_charge_tiles: 16,
        item_placement_strategy: ItemPlacementStrategy::Closed,
        // item_placement_strategy: ItemPlacementStrategy::Semiclosed,
    };
    let randomizer = Randomizer::new(&map, &difficulty, &game_data);
    let mut rng = rand::rngs::StdRng::from_seed([0; 32]);
    let max_attempts = 1;
    for _ in 0..max_attempts {
        let randomization_opt = randomizer.randomize(&mut rng);
        if let Some(randomization) = randomization_opt {
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
    let game_data = GameData::load(sm_json_data_path)?;
    let randomization = get_randomization(&args, &game_data)?;
    let rom = make_rom(&args.input_rom_path, &randomization, &game_data)?;
    rom.save(&args.output_rom_path)?;
    let spoiler_str = serde_json::to_string_pretty(&randomization.spoiler_log)?;
    std::fs::write(args.output_spoiler_path, spoiler_str)?;
    Ok(())
    // for link in randomizer.links {
    //     if link.from_vertex_id == 1208 {
    //         println!("Link: {link:?}");
    //     }
    // }
}
