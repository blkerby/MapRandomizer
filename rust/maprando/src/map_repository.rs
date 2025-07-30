use anyhow::Result;
use log::info;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Deserialize;
use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use crate::randomize::Randomizer;
use maprando_game::{GameData, Map, RoomId};

pub struct MapRepository {
    pub base_path: PathBuf,
    pub filenames: Vec<String>,
}

#[derive(Deserialize)]
struct StoredMap {
    pub room_id: Vec<RoomId>,
    pub room_x: Vec<usize>,
    pub room_y: Vec<usize>,
    pub room_area: Vec<usize>,
    pub room_subarea: Vec<usize>,
    pub room_subsubarea: Vec<usize>,
    pub conn_from_room_id: Vec<usize>,
    pub conn_from_door_id: Vec<usize>,
    pub conn_to_room_id: Vec<usize>,
    pub conn_to_door_id: Vec<usize>,
    pub conn_bidirectional: Vec<bool>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MapsPerFile {
    Fixed(usize),
    Variable(Vec<usize>),
}

#[derive(Deserialize)]
struct MapManifest {
    pub maps_per_file: MapsPerFile,
    pub files: Vec<String>,
}

impl MapRepository {
    pub fn new(name: &str, base_path: &Path) -> Result<Self> {
        let manifest_bytes = std::fs::read(base_path.join("manifest.json"))?;
        let manifest: MapManifest = serde_json::from_slice(&manifest_bytes)?;

        let num_maps = match manifest.maps_per_file {
            MapsPerFile::Fixed(n) => manifest.files.len() * n,
            MapsPerFile::Variable(v) => v.iter().sum(),
        };
        info!(
            "{}: {} maps available ({})",
            name,
            num_maps,
            base_path.display()
        );
        Ok(MapRepository {
            base_path: base_path.to_owned(),
            filenames: manifest.files,
        })
    }

    pub fn get_map_batch(&self, seed: usize, game_data: &GameData) -> Result<Vec<Map>> {
        let idx = seed % self.filenames.len();
        let path = self.base_path.join(&self.filenames[idx]);
        info!("Map batch file: {}", path.display());

        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let avro_reader = apache_avro::Reader::new(buf_reader)?;
        let mut map_vec: Vec<Map> = vec![];
        let room_geometry = &game_data.room_geometry;

        for value in avro_reader {
            let stored_map: StoredMap = apache_avro::from_value(&value?)?;
            let num_rooms = stored_map.room_id.len();
            let num_conns = stored_map.conn_from_door_id.len();

            let mut room_mask: Vec<bool> = vec![false; room_geometry.len()];
            let mut rooms: Vec<(usize, usize)> = vec![(0, 0); room_geometry.len()];
            let mut areas: Vec<usize> = vec![0; room_geometry.len()];
            let mut subareas: Vec<usize> = vec![0; room_geometry.len()];
            let mut subsubareas: Vec<usize> = vec![0; room_geometry.len()];
            for i in 0..num_rooms {
                let room_id = stored_map.room_id[i];
                let room_ptr = game_data.room_ptr_by_id[&room_id];
                let room_idx = game_data.room_idx_by_ptr[&room_ptr];
                room_mask[room_idx] = true;
                rooms[room_idx] = (stored_map.room_x[i], stored_map.room_y[i]);
                areas[room_idx] = stored_map.room_area[i];
                subareas[room_idx] = stored_map.room_subarea[i];
                subsubareas[room_idx] = stored_map.room_subsubarea[i];
            }

            let mut doors = vec![];
            for i in 0..num_conns {
                let from_room_id = stored_map.conn_from_room_id[i];
                let from_room_ptr = game_data.room_ptr_by_id[&from_room_id];
                let from_room_idx = game_data.room_idx_by_ptr[&from_room_ptr];
                let from_door_id = stored_map.conn_from_door_id[i];
                let from_exit_ptr =
                    game_data.room_geometry[from_room_idx].doors[from_door_id].exit_ptr;
                let from_entrance_ptr =
                    game_data.room_geometry[from_room_idx].doors[from_door_id].entrance_ptr;
                let to_room_id = stored_map.conn_to_room_id[i];
                let to_room_ptr = game_data.room_ptr_by_id[&to_room_id];
                let to_room_idx = game_data.room_idx_by_ptr[&to_room_ptr];
                let to_door_id = stored_map.conn_to_door_id[i];
                let to_exit_ptr = game_data.room_geometry[to_room_idx].doors[to_door_id].exit_ptr;
                let to_entrance_ptr =
                    game_data.room_geometry[to_room_idx].doors[to_door_id].entrance_ptr;
                let bidirectional = stored_map.conn_bidirectional[i];
                doors.push((
                    (from_exit_ptr, from_entrance_ptr),
                    (to_exit_ptr, to_entrance_ptr),
                    bidirectional,
                ));
            }

            let mut map = Map {
                room_mask,
                rooms,
                doors,
                area: areas,
                subarea: subareas,
                subsubarea: subsubareas,
            };

            // Make Toilet area/subarea/subsubarea align with its intersecting room(s):
            // TODO: Push this upstream into the map generation
            let toilet_intersections = Randomizer::get_toilet_intersections(&map, game_data);
            if !toilet_intersections.is_empty() {
                let area = map.area[toilet_intersections[0]];
                let subarea = map.subarea[toilet_intersections[0]];
                let subsubarea = map.subsubarea[toilet_intersections[0]];
                for &t in &toilet_intersections {
                    if map.area[t] != area {
                        panic!("Mismatched areas for Toilet intersection");
                    }
                    if map.subarea[t] != subarea {
                        panic!("Mismatched subareas for Toilet intersection");
                    }
                    if map.subsubarea[t] != subsubarea {
                        panic!("Mismatched subsubareas for Toilet intersection");
                    }
                }
                map.area[game_data.toilet_room_idx] = area;
                map.subarea[game_data.toilet_room_idx] = subarea;
                map.subsubarea[game_data.toilet_room_idx] = subsubarea;
            }

            let toilet_top = (Some(0x1A60C), Some(0x1A5AC));
            let toilet_bottom = (Some(0x1A600), Some(0x1A678));
            let mut found_top: bool = false;
            let mut found_bottom: bool = false;
            for d in &map.doors {
                if d.0 == toilet_top || d.1 == toilet_top {
                    found_top = true;
                }
                if d.0 == toilet_bottom || d.1 == toilet_bottom {
                    found_bottom = true;
                }
            }
            if !found_top || !found_bottom {
                // If Toilet does not connect on both sides, then remove it,
                // since we can't put a wall inside it.
                // TODO: push this upstream to the small map extraction
                map.room_mask[game_data.toilet_room_idx] = false;
                map.doors.retain(|x| {
                    x.0 != toilet_top
                        && x.0 != toilet_bottom
                        && x.1 != toilet_top
                        && x.1 != toilet_bottom
                });
            }

            map_vec.push(map);
        }

        let mut rng_seed = [0u8; 32];
        rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
        let mut rng = StdRng::from_seed(rng_seed);
        map_vec.shuffle(&mut rng);

        Ok(map_vec)
    }
}
