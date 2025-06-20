use anyhow::{Context, Result};
use log::info;
use std::path::{Path, PathBuf};

use crate::randomize::Randomizer;
use maprando_game::{GameData, Map};

pub struct MapRepository {
    pub base_path: PathBuf,
    pub filenames: Vec<String>,
}

impl MapRepository {
    pub fn new(name: &str, base_path: &Path) -> Result<Self> {
        let mut filenames: Vec<String> = Vec::new();
        for path in std::fs::read_dir(base_path)? {
            filenames.push(path?.file_name().into_string().unwrap());
        }
        filenames.sort();
        info!(
            "{}: {} maps available ({})",
            name,
            filenames.len(),
            base_path.display()
        );
        Ok(MapRepository {
            base_path: base_path.to_owned(),
            filenames,
        })
    }

    pub fn get_map(
        &self,
        attempt_num_rando: usize,
        seed: usize,
        game_data: &GameData,
    ) -> Result<Map> {
        let idx = seed % self.filenames.len();
        let path = self.base_path.join(&self.filenames[idx]);
        let map_string = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "[attempt {attempt_num_rando}] Unable to read map file at {}",
                path.display()
            )
        })?;
        info!("[attempt {attempt_num_rando}] Map: {}", path.display());
        let mut map: Map = serde_json::from_str(&map_string).with_context(|| {
            format!(
                "[attempt {attempt_num_rando}] Unable to parse map file at {}",
                path.display()
            )
        })?;

        // Make Toilet area/subarea align with its intersecting room(s):
        // TODO: Push this upstream into the map generation
        let toilet_intersections = Randomizer::get_toilet_intersections(&map, game_data);
        if !toilet_intersections.is_empty() {
            let area = map.area[toilet_intersections[0]];
            let subarea = map.subarea[toilet_intersections[0]];
            for &t in &toilet_intersections {
                if map.area[t] != area {
                    panic!("Mismatched areas for Toilet intersection");
                }
                if map.subarea[t] != subarea {
                    panic!("Mismatched subareas for Toilet intersection");
                }
            }
            map.area[game_data.toilet_room_idx] = area;
            map.subarea[game_data.toilet_room_idx] = subarea;
        }
        Ok(map)
    }
}
