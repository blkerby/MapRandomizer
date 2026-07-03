use anyhow::{Context, Result, bail};
use hashbrown::HashMap;
use log::info;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    time::Duration,
};

use crate::randomize::Randomizer;
use crate::settings::AreaAssignmentBaseOrder;
use maprando_game::{GameData, Map, NodeId, RoomId};

#[derive(Clone, Copy, Debug)]
pub struct MapSettings {
    pub small: bool,
    pub wild: bool,
    pub vanilla: bool,
    pub area_assignment_base_order: AreaAssignmentBaseOrder,
}

impl Default for MapSettings {
    fn default() -> Self {
        Self {
            small: false,
            wild: false,
            vanilla: false,
            area_assignment_base_order: AreaAssignmentBaseOrder::Size,
        }
    }
}

impl MapSettings {
    pub fn from_map_layout(map_layout: &str) -> Result<Self> {
        Ok(match map_layout {
            "Vanilla" => Self {
                vanilla: true,
                ..Default::default()
            },
            "Small" => Self {
                small: true,
                ..Default::default()
            },
            "Standard" => Self::default(),
            "Wild" => Self {
                wild: true,
                ..Default::default()
            },
            _ => bail!("Unrecognized map layout option: {map_layout}"),
        })
    }
}

pub trait MapRepository: Send + Sync {
    fn get_map_batch(
        &self,
        seed: usize,
        settings: MapSettings,
        game_data: &GameData,
    ) -> Result<Vec<Map>>;

    fn requires_area_assignment(&self, _settings: MapSettings) -> bool {
        true
    }
}

pub struct LocalVanillaMapRepository {
    vanilla_repository: Box<dyn MapRepository>,
    other_repository: Box<dyn MapRepository>,
}

impl LocalVanillaMapRepository {
    pub fn new(
        vanilla_repository: OfflineMapRepository,
        other_repository: Box<dyn MapRepository>,
    ) -> Self {
        Self {
            vanilla_repository: Box::new(vanilla_repository),
            other_repository,
        }
    }
}

impl MapRepository for LocalVanillaMapRepository {
    fn get_map_batch(
        &self,
        seed: usize,
        settings: MapSettings,
        game_data: &GameData,
    ) -> Result<Vec<Map>> {
        if settings.vanilla {
            self.vanilla_repository
                .get_map_batch(seed, settings, game_data)
        } else {
            self.other_repository
                .get_map_batch(seed, settings, game_data)
        }
    }

    fn requires_area_assignment(&self, settings: MapSettings) -> bool {
        if settings.vanilla {
            self.vanilla_repository.requires_area_assignment(settings)
        } else {
            self.other_repository.requires_area_assignment(settings)
        }
    }
}

pub struct OfflineMapRepository {
    pools: HashMap<String, OfflineMapPool>,
}

struct OfflineMapPool {
    base_path: PathBuf,
    filenames: Vec<String>,
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

impl OfflineMapRepository {
    pub fn new(pools: Vec<(&str, &Path)>) -> Result<Self> {
        Ok(Self {
            pools: pools
                .into_iter()
                .map(|(name, base_path)| {
                    let pool = OfflineMapPool::new(name, base_path)?;
                    Ok((name.to_string(), pool))
                })
                .collect::<Result<HashMap<_, _>>>()?,
        })
    }

    fn pool_name(settings: MapSettings) -> Result<&'static str> {
        let enabled = [settings.small, settings.wild, settings.vanilla]
            .into_iter()
            .filter(|x| *x)
            .count();
        if enabled > 1 {
            bail!("Offline maps support at most one map setting");
        }
        if settings.vanilla {
            Ok("Vanilla")
        } else if settings.small {
            Ok("Small")
        } else if settings.wild {
            Ok("Wild")
        } else {
            Ok("Standard")
        }
    }
}

impl MapRepository for OfflineMapRepository {
    fn get_map_batch(
        &self,
        seed: usize,
        settings: MapSettings,
        game_data: &GameData,
    ) -> Result<Vec<Map>> {
        let pool_name = Self::pool_name(settings)?;
        let pool = self
            .pools
            .get(pool_name)
            .with_context(|| format!("Map pool {pool_name} is not configured"))?;
        pool.get_map_batch(seed, game_data)
    }
}

impl OfflineMapPool {
    fn new(name: &str, base_path: &Path) -> Result<Self> {
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
        Ok(Self {
            base_path: base_path.to_owned(),
            filenames: manifest.files,
        })
    }

    fn get_map_batch(&self, seed: usize, game_data: &GameData) -> Result<Vec<Map>> {
        let idx = seed % self.filenames.len();
        let path = self.base_path.join(&self.filenames[idx]);
        info!("Map batch file: {}", path.display());

        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let avro_reader = apache_avro::Reader::new(buf_reader)?;
        let mut map_vec: Vec<Map> = vec![];

        for value in avro_reader {
            let stored_map: StoredMap = apache_avro::from_value(&value?)?;
            let mut map = stored_map_to_map(stored_map, game_data)?;
            normalize_toilet(&mut map, game_data);
            map_vec.push(map);
        }

        shuffle_maps(&mut map_vec, seed);
        Ok(map_vec)
    }
}

pub struct HttpMapRepository {
    host: String,
    port: u16,
    path: String,
}

#[derive(Serialize)]
struct GenerateRequest {
    episode_length: usize,
    recommended_candidates: usize,
    shortlist_candidates: usize,
    temperature: f64,
    proposal_temperature: f64,
    reward_door: f64,
    reward_connection: f64,
    reward_toilet: f64,
    reward_phantoon: f64,
    reward_balance: f64,
    reward_toilet_balance: f64,
    reward_frontier: f64,
    reward_graph_diameter: f64,
    reward_save_distance: f64,
    reward_refill_distance: f64,
    reward_missing_connect_utility: f64,
    small_map: bool,
    min_rooms: Option<usize>,
    max_rooms: Option<usize>,
    target_rooms: Option<usize>,
    area_assignment_base_order: GenerateAreaAssignmentBaseOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum GenerateAreaAssignmentBaseOrder {
    Size,
    Depth,
    Random,
}

impl From<AreaAssignmentBaseOrder> for GenerateAreaAssignmentBaseOrder {
    fn from(value: AreaAssignmentBaseOrder) -> Self {
        match value {
            AreaAssignmentBaseOrder::Size => Self::Size,
            AreaAssignmentBaseOrder::Depth => Self::Depth,
            AreaAssignmentBaseOrder::Random => Self::Random,
        }
    }
}

#[derive(Deserialize)]
struct GenerateResponse {
    rooms: GenerateResponseRooms,
    edges: GenerateResponseEdges,
}

#[derive(Deserialize)]
struct GenerateResponseRooms {
    id: Vec<Vec<RoomId>>,
    x: Vec<Vec<usize>>,
    y: Vec<Vec<usize>>,
    area: Vec<Vec<usize>>,
    subarea: Vec<Vec<usize>>,
    subsubarea: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct GenerateResponseEdges {
    from_room_placement_idx: Vec<Vec<usize>>,
    #[serde(rename = "from_door_id")]
    from_door_node_id: Vec<Vec<NodeId>>,
    to_room_placement_idx: Vec<Vec<usize>>,
    #[serde(rename = "to_door_id")]
    to_door_node_id: Vec<Vec<NodeId>>,
}

impl HttpMapRepository {
    pub fn new(server: &str) -> Result<Self> {
        let server = server
            .strip_prefix("http://")
            .unwrap_or(server)
            .trim_end_matches('/');
        if server.starts_with("https://") {
            bail!("Map generation server must use HTTP, not HTTPS");
        }
        let (host, port) = if let Some((host, port)) = server.rsplit_once(':') {
            (host.to_string(), port.parse()?)
        } else {
            bail!("Map generation server must be in host:port format");
        };
        Ok(Self {
            host,
            port,
            path: "/generate".to_string(),
        })
    }

    fn generate_request(settings: MapSettings) -> GenerateRequest {
        GenerateRequest {
            episode_length: 253,
            recommended_candidates: 4,
            shortlist_candidates: 16,
            temperature: 0.03,
            proposal_temperature: 0.3,
            reward_door: 1.0,
            reward_connection: 1.0,
            reward_toilet: 1.0,
            reward_phantoon: 1.0,
            reward_balance: 0.1,
            reward_toilet_balance: 0.1,
            reward_frontier: 0.0,
            reward_graph_diameter: 0.1,
            reward_save_distance: 0.1,
            reward_refill_distance: 0.1,
            reward_missing_connect_utility: if settings.wild { 0.0 } else { 1.0 },
            small_map: settings.small,
            min_rooms: settings.small.then_some(120),
            max_rooms: settings.small.then_some(180),
            target_rooms: settings.small.then_some(150),
            area_assignment_base_order: settings.area_assignment_base_order.into(),
        }
    }

    fn response_to_maps(response: GenerateResponse, game_data: &GameData) -> Result<Vec<Map>> {
        let map_count = response.rooms.id.len();
        validate_response_lengths(&response, map_count)?;
        let mut maps = Vec::with_capacity(map_count);
        for map_idx in 0..map_count {
            let mut map = response_map_to_map(&response, map_idx, game_data)?;
            normalize_toilet(&mut map, game_data);
            maps.push(map);
        }
        Ok(maps)
    }

    fn request_generate(&self, settings: MapSettings) -> Result<GenerateResponse> {
        let body = serde_json::to_vec(&Self::generate_request(settings))?;
        let mut stream =
            TcpStream::connect((self.host.as_str(), self.port)).with_context(|| {
                format!(
                    "Failed to connect to map generation server {}:{}",
                    self.host, self.port
                )
            })?;
        stream.set_read_timeout(Some(Duration::from_secs(300)))?;
        stream.set_write_timeout(Some(Duration::from_secs(30)))?;

        write!(
            stream,
            "POST {} HTTP/1.1\r\n\
             Host: {}:{}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n",
            self.path,
            self.host,
            self.port,
            body.len()
        )?;
        stream.write_all(&body)?;
        stream.flush()?;

        let mut response_bytes = Vec::new();
        stream.read_to_end(&mut response_bytes)?;
        parse_generate_response(&response_bytes).with_context(|| {
            format!(
                "Failed to parse response from map generation server {}:{}",
                self.host, self.port
            )
        })
    }
}

impl MapRepository for HttpMapRepository {
    fn get_map_batch(
        &self,
        seed: usize,
        settings: MapSettings,
        game_data: &GameData,
    ) -> Result<Vec<Map>> {
        let response = self.request_generate(settings)?;
        let mut maps = Self::response_to_maps(response, game_data)?;
        if maps.is_empty() {
            bail!("Map generation server returned no valid maps");
        }
        shuffle_maps(&mut maps, seed);
        Ok(maps)
    }

    fn requires_area_assignment(&self, _settings: MapSettings) -> bool {
        false
    }
}

fn stored_map_to_map(stored_map: StoredMap, game_data: &GameData) -> Result<Map> {
    let num_rooms = stored_map.room_id.len();
    let num_conns = stored_map.conn_from_door_id.len();
    let mut room_mask: Vec<bool> = vec![false; game_data.room_geometry.len()];
    let mut rooms: Vec<(usize, usize)> = vec![(0, 0); game_data.room_geometry.len()];
    let mut areas: Vec<usize> = vec![0; game_data.room_geometry.len()];
    let mut subareas: Vec<usize> = vec![0; game_data.room_geometry.len()];
    let mut subsubareas: Vec<usize> = vec![0; game_data.room_geometry.len()];
    for i in 0..num_rooms {
        let room_id = stored_map.room_id[i];
        let room_idx = room_idx_from_room_id(room_id, game_data)?;
        room_mask[room_idx] = true;
        rooms[room_idx] = (stored_map.room_x[i], stored_map.room_y[i]);
        areas[room_idx] = stored_map.room_area[i];
        subareas[room_idx] = stored_map.room_subarea[i];
        subsubareas[room_idx] = stored_map.room_subsubarea[i];
    }

    let mut doors = vec![];
    for i in 0..num_conns {
        let from_room_idx = room_idx_from_room_id(stored_map.conn_from_room_id[i], game_data)?;
        let to_room_idx = room_idx_from_room_id(stored_map.conn_to_room_id[i], game_data)?;
        let from_door_idx = stored_map.conn_from_door_id[i];
        let to_door_idx = stored_map.conn_to_door_id[i];
        doors.push((
            door_ptr_pair(from_room_idx, from_door_idx, game_data)?,
            door_ptr_pair(to_room_idx, to_door_idx, game_data)?,
            stored_map.conn_bidirectional[i],
        ));
    }

    Ok(Map {
        room_mask,
        rooms,
        doors,
        area: areas,
        subarea: subareas,
        subsubarea: subsubareas,
    })
}

fn response_map_to_map(
    response: &GenerateResponse,
    map_idx: usize,
    game_data: &GameData,
) -> Result<Map> {
    let num_rooms = response.rooms.id[map_idx].len();
    let mut placement_room_idx = Vec::with_capacity(num_rooms);
    let mut room_mask: Vec<bool> = vec![false; game_data.room_geometry.len()];
    let mut rooms: Vec<(usize, usize)> = vec![(0, 0); game_data.room_geometry.len()];
    let mut areas: Vec<usize> = vec![0; game_data.room_geometry.len()];
    let mut subareas: Vec<usize> = vec![0; game_data.room_geometry.len()];
    let mut subsubareas: Vec<usize> = vec![0; game_data.room_geometry.len()];

    for placement_idx in 0..num_rooms {
        let room_idx = room_idx_from_room_id(response.rooms.id[map_idx][placement_idx], game_data)?;
        placement_room_idx.push(room_idx);
        room_mask[room_idx] = true;
        rooms[room_idx] = (
            response.rooms.x[map_idx][placement_idx],
            response.rooms.y[map_idx][placement_idx],
        );
        areas[room_idx] = response.rooms.area[map_idx][placement_idx];
        subareas[room_idx] = response.rooms.subarea[map_idx][placement_idx];
        subsubareas[room_idx] = response.rooms.subsubarea[map_idx][placement_idx];
    }

    let num_edges = response.edges.from_room_placement_idx[map_idx].len();
    let mut doors = Vec::with_capacity(num_edges);
    for edge_idx in 0..num_edges {
        let from_placement_idx = response.edges.from_room_placement_idx[map_idx][edge_idx];
        let to_placement_idx = response.edges.to_room_placement_idx[map_idx][edge_idx];
        let from_room_idx = *placement_room_idx
            .get(from_placement_idx)
            .with_context(|| {
                format!("Edge references missing from-room placement index {from_placement_idx}")
            })?;
        let to_room_idx = *placement_room_idx.get(to_placement_idx).with_context(|| {
            format!("Edge references missing to-room placement index {to_placement_idx}")
        })?;
        let from_door_node_id = response.edges.from_door_node_id[map_idx][edge_idx];
        let to_door_node_id = response.edges.to_door_node_id[map_idx][edge_idx];
        let from_door_idx = door_idx_from_node_id(from_room_idx, from_door_node_id, game_data)?;
        let to_door_idx = door_idx_from_node_id(to_room_idx, to_door_node_id, game_data)?;
        let (src_room_idx, src_door_idx, dst_room_idx, dst_door_idx) = canonical_door_order(
            from_room_idx,
            from_door_idx,
            to_room_idx,
            to_door_idx,
            game_data,
        )?;
        doors.push((
            door_ptr_pair(src_room_idx, src_door_idx, game_data)?,
            door_ptr_pair(dst_room_idx, dst_door_idx, game_data)?,
            true,
        ));
    }

    Ok(Map {
        room_mask,
        rooms,
        doors,
        area: areas,
        subarea: subareas,
        subsubarea: subsubareas,
    })
}

fn room_idx_from_room_id(room_id: RoomId, game_data: &GameData) -> Result<usize> {
    let room_ptr = game_data
        .room_ptr_by_id
        .get(&room_id)
        .with_context(|| format!("Room ID {room_id} not found"))?;
    game_data
        .room_idx_by_ptr
        .get(room_ptr)
        .copied()
        .with_context(|| format!("Room pointer {room_ptr:?} for room ID {room_id} not found"))
}

fn door_ptr_pair(
    room_idx: usize,
    door_idx: usize,
    game_data: &GameData,
) -> Result<(Option<usize>, Option<usize>)> {
    let door = game_data
        .room_geometry
        .get(room_idx)
        .and_then(|room| room.doors.get(door_idx))
        .with_context(|| format!("Door index {door_idx} not found for room index {room_idx}"))?;
    Ok((door.exit_ptr, door.entrance_ptr))
}

fn door_idx_from_node_id(room_idx: usize, node_id: NodeId, game_data: &GameData) -> Result<usize> {
    let room = game_data
        .room_geometry
        .get(room_idx)
        .with_context(|| format!("Room index {room_idx} not found"))?;
    let room_id = room.room_id;
    if let Some(ptr_pair) = game_data.reverse_door_ptr_pair_map.get(&(room_id, node_id)) {
        let (mapped_room_idx, door_idx) = game_data
            .room_and_door_idxs_by_door_ptr_pair
            .get(ptr_pair)
            .copied()
            .with_context(|| {
                format!("Door pointer pair {ptr_pair:?} not found in room geometry index")
            })?;
        if mapped_room_idx != room_idx {
            bail!(
                "Door node ID {node_id} for room ID {room_id} maps to room index {mapped_room_idx}, expected {room_idx}"
            );
        }
        return Ok(door_idx);
    }

    door_idx_from_node_tile(room_idx, room_id, node_id, game_data)
}

fn canonical_door_order(
    from_room_idx: usize,
    from_door_idx: usize,
    to_room_idx: usize,
    to_door_idx: usize,
    game_data: &GameData,
) -> Result<(usize, usize, usize, usize)> {
    let from_door = game_data
        .room_geometry
        .get(from_room_idx)
        .and_then(|room| room.doors.get(from_door_idx))
        .with_context(|| {
            format!("Door index {from_door_idx} not found for room index {from_room_idx}")
        })?;
    let to_door = game_data
        .room_geometry
        .get(to_room_idx)
        .and_then(|room| room.doors.get(to_door_idx))
        .with_context(|| {
            format!("Door index {to_door_idx} not found for room index {to_room_idx}")
        })?;

    let should_swap = match (from_door.direction.as_str(), to_door.direction.as_str()) {
        ("right", "left") | ("down", "up") => false,
        ("left", "right") | ("up", "down") => true,
        _ => bail!(
            "Unexpected door direction pair: {} -> {}",
            from_door.direction,
            to_door.direction
        ),
    };
    if should_swap {
        Ok((to_room_idx, to_door_idx, from_room_idx, from_door_idx))
    } else {
        Ok((from_room_idx, from_door_idx, to_room_idx, to_door_idx))
    }
}

fn door_idx_from_node_tile(
    room_idx: usize,
    room_id: RoomId,
    node_id: NodeId,
    game_data: &GameData,
) -> Result<usize> {
    let node_key = (room_id, node_id);
    let node_tiles = game_data.node_tile_coords.get(&node_key).with_context(|| {
        format!("Node tile coordinates missing for room ID {room_id}, node ID {node_id}")
    })?;
    let direction = game_data
        .node_json_map
        .get(&node_key)
        .and_then(|node| node["doorOrientation"].as_str())
        .with_context(|| {
            format!("Door orientation missing for room ID {room_id}, node ID {node_id}")
        })?;
    let room = &game_data.room_geometry[room_idx];
    let matching_door_idxs: Vec<usize> = room
        .doors
        .iter()
        .enumerate()
        .filter_map(|(door_idx, door)| {
            (door.direction == direction && node_tiles.contains(&(door.x, door.y)))
                .then_some(door_idx)
        })
        .collect();
    match matching_door_idxs.as_slice() {
        [door_idx] => Ok(*door_idx),
        [] => bail!(
            "Door node ID {node_id} for room ID {room_id} has no matching geometry door by tile/orientation"
        ),
        _ => bail!(
            "Door node ID {node_id} for room ID {room_id} matched multiple geometry doors by tile/orientation: {matching_door_idxs:?}"
        ),
    }
}

fn normalize_toilet(map: &mut Map, game_data: &GameData) {
    // Make Toilet area/subarea/subsubarea align with its intersecting room(s):
    // TODO: Push this upstream into the map generation
    let toilet_intersections = Randomizer::get_toilet_intersections(map, game_data);
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
            x.0 != toilet_top && x.0 != toilet_bottom && x.1 != toilet_top && x.1 != toilet_bottom
        });
    }
}

fn shuffle_maps(map_vec: &mut [Map], seed: usize) {
    let mut rng_seed = [0u8; 32];
    rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
    let mut rng = StdRng::from_seed(rng_seed);
    map_vec.shuffle(&mut rng);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serialize_generate_request(settings: MapSettings) -> serde_json::Value {
        serde_json::to_value(HttpMapRepository::generate_request(settings)).unwrap()
    }

    #[test]
    fn generate_request_serializes_lowercase_area_assignment_base_order() {
        let cases = [
            (AreaAssignmentBaseOrder::Size, "size"),
            (AreaAssignmentBaseOrder::Depth, "depth"),
            (AreaAssignmentBaseOrder::Random, "random"),
        ];

        for (base_order, expected) in cases {
            let request = serialize_generate_request(MapSettings {
                area_assignment_base_order: base_order,
                ..Default::default()
            });
            assert_eq!(request["area_assignment_base_order"], expected);
        }
    }

    #[test]
    fn repository_area_assignment_capability_matches_source() {
        let settings = MapSettings::default();
        let offline_repository = OfflineMapRepository {
            pools: HashMap::new(),
        };
        let http_repository = HttpMapRepository::new("localhost:1234").unwrap();

        assert!(offline_repository.requires_area_assignment(settings));
        assert!(!http_repository.requires_area_assignment(settings));

        let local_vanilla_repository = LocalVanillaMapRepository::new(
            OfflineMapRepository {
                pools: HashMap::new(),
            },
            Box::new(http_repository),
        );
        assert!(!local_vanilla_repository.requires_area_assignment(settings));
        assert!(
            local_vanilla_repository.requires_area_assignment(MapSettings {
                vanilla: true,
                ..Default::default()
            })
        );
    }
}

fn validate_response_lengths(response: &GenerateResponse, map_count: usize) -> Result<()> {
    if response.rooms.x.len() != map_count
        || response.rooms.y.len() != map_count
        || response.rooms.area.len() != map_count
        || response.rooms.subarea.len() != map_count
        || response.rooms.subsubarea.len() != map_count
        || response.edges.from_room_placement_idx.len() != map_count
        || response.edges.from_door_node_id.len() != map_count
        || response.edges.to_room_placement_idx.len() != map_count
        || response.edges.to_door_node_id.len() != map_count
    {
        bail!("Map generation response has inconsistent top-level lengths");
    }
    for map_idx in 0..map_count {
        let room_count = response.rooms.id[map_idx].len();
        if response.rooms.x[map_idx].len() != room_count
            || response.rooms.y[map_idx].len() != room_count
            || response.rooms.area[map_idx].len() != room_count
            || response.rooms.subarea[map_idx].len() != room_count
            || response.rooms.subsubarea[map_idx].len() != room_count
        {
            bail!("Map generation response has inconsistent room lengths for map {map_idx}");
        }
        let edge_count = response.edges.from_room_placement_idx[map_idx].len();
        if response.edges.from_door_node_id[map_idx].len() != edge_count
            || response.edges.to_room_placement_idx[map_idx].len() != edge_count
            || response.edges.to_door_node_id[map_idx].len() != edge_count
        {
            bail!("Map generation response has inconsistent edge lengths for map {map_idx}");
        }
    }
    Ok(())
}

fn parse_generate_response(response_bytes: &[u8]) -> Result<GenerateResponse> {
    let header_end = response_bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .with_context(|| "HTTP response missing header terminator")?;
    let header_bytes = &response_bytes[..header_end];
    let body_bytes = &response_bytes[header_end + 4..];
    let header_text = std::str::from_utf8(header_bytes)?;
    let mut header_lines = header_text.lines();
    let status_line = header_lines
        .next()
        .with_context(|| "HTTP response missing status line")?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .with_context(|| format!("Malformed HTTP status line: {status_line}"))?
        .parse::<u16>()?;
    if !(200..300).contains(&status) {
        bail!(
            "Map generation server returned HTTP status {status}: {}",
            String::from_utf8_lossy(body_bytes)
        );
    }
    serde_json::from_slice(body_bytes).context("Failed to parse map generation response JSON")
}
