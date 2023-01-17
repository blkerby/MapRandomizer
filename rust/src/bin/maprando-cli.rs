use maprando::{game_data::GameData, item_placement::DifficultyConfig};
use maprando::item_placement::{Randomizer, ItemPlacementStrategy};
use rand::SeedableRng;
use std::{path::{Path, PathBuf}, io, fs::File};
use clap::Parser;
use json;

#[derive(Parser)]
struct Args {
    map: PathBuf,
}

fn main() {
    let args = Args::parse();
    let sm_json_data_path = Path::new("../sm-json-data");
    let game_data = GameData::load(sm_json_data_path);

    let map_string = io::read_to_string(File::open(&args.map).unwrap()).unwrap();
    let map = json::parse(&map_string).unwrap();
    let difficulty = DifficultyConfig {
        tech: game_data.tech_isv.keys.clone(),
        shine_charge_tiles: 16,
        item_placement_strategy: ItemPlacementStrategy::Semiclosed,
    };
    let randomizer = Randomizer::new(&map, &difficulty, &game_data);
    let mut rng = rand::rngs::StdRng::from_seed([8; 32]);
    let max_attempts = 1;
    for _ in 0..max_attempts {
        let randomization_opt = randomizer.randomize(&mut rng);
        if let Some(randomization) = randomization_opt {
            println!("Success");
            break;
        } else {
            println!("Failed randomization attempt");
        }
    }

    // for link in randomizer.links {
    //     if link.from_vertex_id == 1208 {
    //         println!("Link: {link:?}");
    //     }
    // }
}
