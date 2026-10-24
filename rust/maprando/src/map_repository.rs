use anyhow::{Context, Result, bail};
use hashbrown::HashMap;
use log::info;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Duration,
};

use crate::map_settings::Landmark;
use crate::randomize::Randomizer;
use crate::settings::{
    MapPreset, RandomizerSettings, SkillAssumptionSettings, StartLocationMode, WallJump,
};
use maprando_game::{GameData, Map, NodeId, RoomId};

pub trait MapRepository: Send + Sync {
    fn get_map_batch(
        &self,
        seed: usize,
        settings: &RandomizerSettings,
        game_data: &GameData,
    ) -> Result<Vec<Map>>;

    fn request_refill(&self, settings: &RandomizerSettings) -> Result<()>;
}

pub struct SelectableMapRepository {
    pub pregenerated: PregeneratedMapRepository,
    pub fresh: Option<HttpMapRepository>,
}

impl MapRepository for SelectableMapRepository {
    fn get_map_batch(
        &self,
        seed: usize,
        settings: &RandomizerSettings,
        game_data: &GameData,
    ) -> Result<Vec<Map>> {
        if settings.map_settings.source == "Fresh" {
            self.fresh
                .as_ref()
                .context("Fresh map generation is not configured")?
                .get_map_batch(seed, settings, game_data)
        } else {
            self.pregenerated.get_map_batch(seed, settings, game_data)
        }
    }

    fn request_refill(&self, settings: &RandomizerSettings) -> Result<()> {
        if settings.map_settings.source == "Fresh" {
            self.fresh
                .as_ref()
                .context("Fresh map generation is not configured")?
                .request_refill(settings)?;
        }
        Ok(())
    }
}

pub struct PregeneratedMapRepository {
    pools: HashMap<String, PregeneratedMapPool>,
}

struct PregeneratedMapPool {
    source: PregeneratedMapPoolSource,
    filenames: Vec<String>,
}

enum PregeneratedMapPoolSource {
    Local(PathBuf),
    Remote {
        base_url: reqwest::Url,
        client: reqwest::blocking::Client,
    },
}

#[derive(Deserialize)]
pub struct MapPoolsConfig {
    pub preset_fallbacks: BTreeMap<MapPreset, String>,
    pub pools: Vec<MapPoolConfig>,
}

#[derive(Deserialize)]
pub struct MapPoolConfig {
    pub name: String,
    pub display_name: String,
    pub base_url: Option<String>,
    #[serde(default)]
    pub small_map: bool,
}

impl MapPoolsConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(
            &std::fs::read(path).with_context(|| format!("Reading {}", path.display()))?,
        )
        .with_context(|| format!("Parsing {}", path.display()))?;
        let mut pool_names = HashSet::new();
        for pool in &config.pools {
            if pool.name == "Fresh" {
                bail!("Fresh is reserved for fresh map generation and cannot identify a pool");
            }
            if !pool_names.insert(pool.name.as_str()) {
                bail!("Duplicate map pool name {}", pool.name);
            }
        }
        for (preset, pool) in &config.preset_fallbacks {
            if !pool_names.contains(pool.as_str()) {
                bail!("Map preset {preset:?} refers to unconfigured pool {pool}");
            }
        }
        Ok(config)
    }
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

impl PregeneratedMapRepository {
    pub fn new(config: &MapPoolsConfig) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Creating map pool HTTP client")?;
        Ok(Self {
            pools: config
                .pools
                .iter()
                .map(|pool_config| {
                    let name = &pool_config.name;
                    let pool = PregeneratedMapPool::new(name, pool_config, &client)
                        .with_context(|| format!("Loading map pool {name}"))?;
                    Ok((name.clone(), pool))
                })
                .collect::<Result<HashMap<_, _>>>()?,
        })
    }
}

impl MapRepository for PregeneratedMapRepository {
    fn request_refill(&self, _settings: &RandomizerSettings) -> Result<()> {
        Ok(())
    }

    fn get_map_batch(
        &self,
        seed: usize,
        settings: &RandomizerSettings,
        game_data: &GameData,
    ) -> Result<Vec<Map>> {
        let pool_name = &settings.map_settings.source;
        let pool = self
            .pools
            .get(pool_name)
            .with_context(|| format!("Map pool {pool_name} is not configured"))?;
        pool.get_map_batch(seed, game_data)
    }
}

impl PregeneratedMapPool {
    fn new(name: &str, config: &MapPoolConfig, client: &reqwest::blocking::Client) -> Result<Self> {
        let path = Path::new("../maps").join(name);
        let manifest_path = path.join("manifest.json");
        let manifest_bytes = std::fs::read(&manifest_path)
            .with_context(|| format!("Reading {}", manifest_path.display()))?;
        let manifest: MapManifest = serde_json::from_slice(&manifest_bytes)?;
        if manifest.files.is_empty() {
            bail!("Manifest contains no batch files");
        }

        let local_files = std::fs::read_dir(&path)?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<std::io::Result<HashSet<_>>>()?;
        let num_local = manifest
            .files
            .iter()
            .filter(|filename| local_files.contains(std::ffi::OsStr::new(filename)))
            .count();
        let source = if num_local == manifest.files.len() {
            info!("{name}: using local batches ({})", path.display());
            PregeneratedMapPoolSource::Local(path)
        } else if num_local != 0 {
            bail!(
                "Partial local pool in {}: {} of {} batch files missing; finish downloading the pool or remove its batches",
                path.display(),
                manifest.files.len() - num_local,
                manifest.files.len()
            );
        } else {
            let base_url = config
                .base_url
                .as_ref()
                .context("No local batches and no base_url configured")?;
            let base_url = reqwest::Url::parse(&format!("{}/", base_url.trim_end_matches('/')))?;
            info!("{name}: using remote batches ({base_url})");
            PregeneratedMapPoolSource::Remote {
                base_url,
                client: client.clone(),
            }
        };

        let num_maps = match manifest.maps_per_file {
            MapsPerFile::Fixed(n) => manifest.files.len() * n,
            MapsPerFile::Variable(v) => v.iter().sum(),
        };
        info!("{name}: {num_maps} maps available");
        Ok(Self {
            source,
            filenames: manifest.files,
        })
    }

    fn get_map_batch(&self, seed: usize, game_data: &GameData) -> Result<Vec<Map>> {
        let idx = seed % self.filenames.len();
        let reader: Box<dyn Read> = match &self.source {
            PregeneratedMapPoolSource::Local(base_path) => {
                let path = base_path.join(&self.filenames[idx]);
                info!("Map batch file: {}", path.display());
                Box::new(File::open(&path).with_context(|| format!("Opening {}", path.display()))?)
            }
            PregeneratedMapPoolSource::Remote { base_url, client } => {
                let url = base_url.join(&self.filenames[idx])?;
                info!("Map batch URL: {url}");
                Box::new(
                    client
                        .get(url.clone())
                        .send()
                        .and_then(|response| response.error_for_status())
                        .with_context(|| format!("Fetching map batch {url}"))?,
                )
            }
        };
        let buf_reader = BufReader::new(reader);
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
    base_url: String,
    client: reqwest::blocking::Client,
    difficulty_bins: Vec<(String, f32)>,
}

#[derive(Serialize)]
struct GenerateRequest {
    num_maps: usize,
    episode_length: usize,
    recommended_candidates: usize,
    shortlist_candidates: usize,
    num_scored_invalid_candidates: usize,
    max_candidate_areas_per_placement: usize,
    recommended_candidates_same_frontier: bool,
    guarantee_phantoon_counterpart: bool,
    temperature: f64,
    proposal_temperature: f64,
    reward_door: f64,
    reward_connection: f64,
    reward_toilet: f64,
    reward_phantoon_pair: f64,
    reward_phantoon_area: f64,
    reward_success: f64,
    reward_initial_item: f64,
    initial_item_start: Option<(RoomId, NodeId)>,
    initial_item_difficulty: String,
    initial_item_wall_jump: bool,
    reward_ship_in_crateria: f64,
    reward_kraid_in_brinstar: f64,
    reward_ridley_in_norfair: f64,
    reward_phantoon_in_wrecked_ship: f64,
    reward_draygon_in_maridia: f64,
    reward_mother_brain_in_tourian: f64,
    force_ship_in_crateria: bool,
    force_kraid_in_brinstar: bool,
    force_ridley_in_norfair: bool,
    force_phantoon_in_wrecked_ship: bool,
    force_draygon_in_maridia: bool,
    force_mother_brain_in_tourian: bool,
    reward_frontier: f64,
    reward_graph_diameter: f64,
    maridia_water_preferred_probability_1: f64,
    maridia_water_preferred_probability_2: f64,
    maridia_water_preferred_probability_3: f64,
    norfair_heat_preferred_probability_1: f64,
    norfair_heat_preferred_probability_2: f64,
    norfair_heat_preferred_probability_3: f64,
    reward_save_distance: f64,
    reward_refill_distance: f64,
    reward_missing_connect_utility: f64,
    max_return_distance: Option<f64>,
    reward_area_distinct_crossing: f64,
    reward_area_crossing: f64,
    reward_area_size_valid: f64,
    reward_area_map_station: f64,
    reward_area_x: f64,
    reward_area_y: f64,
    target_area_rooms: [f64; 6],
    target_area_x: [f64; 6],
    target_area_y: [f64; 6],
    small_map: bool,
    min_rooms: Option<usize>,
    max_rooms: Option<usize>,
    target_rooms: Option<usize>,
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
    pub fn new(server: &str, skill_presets: &[SkillAssumptionSettings]) -> Result<Self> {
        let server = server
            .strip_prefix("http://")
            .unwrap_or(server)
            .trim_end_matches('/');
        if server.starts_with("https://") {
            bail!("Map generation server must use HTTP, not HTTPS");
        }
        let (host, port) = if let Some((host, port)) = server.rsplit_once(':') {
            (host, port.parse::<u16>()?)
        } else {
            bail!("Map generation server must be in host:port format");
        };
        let mut difficulty_bins: Vec<_> = skill_presets
            .iter()
            .filter(|preset| preset.preset.as_deref() != Some("Implicit"))
            .map(|preset| {
                (
                    preset.preset.clone().unwrap(),
                    preset.escape_timer_multiplier,
                )
            })
            .collect();
        difficulty_bins.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(Self {
            base_url: format!("http://{host}:{port}"),
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(300))
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .context("Failed to create map generation HTTP client")?,
            difficulty_bins,
        })
    }

    fn initial_item_difficulty(&self, skill: &SkillAssumptionSettings) -> Result<String> {
        if let Some(name) = &skill.preset {
            if !self
                .difficulty_bins
                .iter()
                .any(|(preset, _)| preset == name)
            {
                bail!("Unsupported initial-item difficulty preset: {name}");
            }
            return Ok(name.clone());
        }
        // Midpoints belong to the larger multiplier (the more conservative tier).
        let preset = self
            .difficulty_bins
            .windows(2)
            .find(|pair| skill.escape_timer_multiplier >= (pair[0].1 + pair[1].1) / 2.0);
        Ok(preset.map_or_else(
            || self.difficulty_bins.last().unwrap().0.clone(),
            |pair| pair[0].0.clone(),
        ))
    }

    fn generate_request(&self, settings: &RandomizerSettings) -> Result<GenerateRequest> {
        let g = &settings.map_settings.generation;
        let initial_item_start = match settings.start_location_settings.mode {
            StartLocationMode::Ship => Some((8, 5)),
            StartLocationMode::Custom => Some((
                settings
                    .start_location_settings
                    .room_id
                    .context("expected start room_id")?,
                settings
                    .start_location_settings
                    .node_id
                    .context("expected start node_id")?,
            )),
            StartLocationMode::Random | StartLocationMode::Escape => None,
        };
        let initial_item_wall_jump = settings.other_settings.wall_jump == WallJump::Vanilla;
        let water = g.maridia_water.effective_probabilities(g.area_share(4));
        let heat = g.norfair_heat.effective_probabilities(g.area_share(2));
        let ship = g.landmarks.contains(&Landmark::Ship);
        let kraid = g.landmarks.contains(&Landmark::Kraid);
        let ridley = g.landmarks.contains(&Landmark::Ridley);
        let phantoon = g.landmarks.contains(&Landmark::Phantoon);
        let draygon = g.landmarks.contains(&Landmark::Draygon);
        let mother_brain = g.landmarks.contains(&Landmark::MotherBrain);
        Ok(GenerateRequest {
            num_maps: 1,
            episode_length: 253,
            recommended_candidates: 8,
            shortlist_candidates: 1024,
            num_scored_invalid_candidates: 0,
            max_candidate_areas_per_placement: 2,
            recommended_candidates_same_frontier: false,
            guarantee_phantoon_counterpart: true,
            temperature: 0.04,
            proposal_temperature: 0.02,
            reward_door: 1.0,
            reward_connection: 1.0,
            reward_toilet: 1.0,
            reward_phantoon_pair: 1.0,
            reward_phantoon_area: 1.0,
            reward_success: 0.0,
            reward_initial_item: if initial_item_start.is_some() {
                1.0
            } else {
                0.0
            },
            initial_item_start,
            initial_item_difficulty: self
                .initial_item_difficulty(&settings.skill_assumption_settings)?,
            initial_item_wall_jump,
            reward_ship_in_crateria: if ship { 1.0 } else { 0.0 },
            reward_kraid_in_brinstar: if kraid { 1.0 } else { 0.0 },
            reward_ridley_in_norfair: if ridley { 1.0 } else { 0.0 },
            reward_phantoon_in_wrecked_ship: if phantoon { 1.0 } else { 0.0 },
            reward_draygon_in_maridia: if draygon { 1.0 } else { 0.0 },
            reward_mother_brain_in_tourian: if mother_brain { 1.0 } else { 0.0 },
            force_ship_in_crateria: ship,
            force_kraid_in_brinstar: kraid,
            force_ridley_in_norfair: ridley,
            force_phantoon_in_wrecked_ship: phantoon,
            force_draygon_in_maridia: draygon,
            force_mother_brain_in_tourian: mother_brain,
            reward_frontier: 0.0,
            reward_graph_diameter: g.reward_graph_diameter,
            maridia_water_preferred_probability_1: water[0],
            maridia_water_preferred_probability_2: water[1],
            maridia_water_preferred_probability_3: water[2],
            norfair_heat_preferred_probability_1: heat[0],
            norfair_heat_preferred_probability_2: heat[1],
            norfair_heat_preferred_probability_3: heat[2],
            reward_save_distance: g.reward_save_distance,
            reward_refill_distance: g.reward_refill_distance,
            reward_missing_connect_utility: g.reward_missing_connect_utility,
            max_return_distance: g.max_return_distance,
            reward_area_distinct_crossing: g.reward_area_distinct_crossing,
            reward_area_crossing: 0.0,
            reward_area_size_valid: 1.0,
            reward_area_map_station: 1.0,
            reward_area_x: g.reward_area_x,
            reward_area_y: g.reward_area_y,
            target_area_rooms: g.target_area_rooms,
            target_area_x: g.target_area_x,
            target_area_y: g.target_area_y,
            small_map: g.small_map,
            min_rooms: g.small_map.then_some(120),
            max_rooms: g.small_map.then_some(180),
            target_rooms: g.small_map.then_some(150),
        })
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

    fn request_generate(&self, settings: &RandomizerSettings) -> Result<GenerateResponse> {
        serde_json::from_value(self.post("/generate", settings)?)
            .context("Failed to parse map generation response JSON")
    }

    fn post(&self, path: &str, settings: &RandomizerSettings) -> Result<serde_json::Value> {
        let url = format!("{}{path}", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&self.generate_request(settings)?)
            .send()
            .with_context(|| format!("Map generation request failed: {url}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .context("Failed to read map generation error response")?;
            bail!(
                "Map generation server returned HTTP status {}: {body}",
                status.as_u16()
            );
        }
        response
            .json()
            .with_context(|| format!("Failed to parse response from map generation server {url}"))
    }
}

impl MapRepository for HttpMapRepository {
    fn get_map_batch(
        &self,
        _seed: usize,
        settings: &RandomizerSettings,
        game_data: &GameData,
    ) -> Result<Vec<Map>> {
        settings.map_settings.generation.validate()?;
        // TODO: if it is feasible to make fresh map generation reproducible,
        // plumb the `_seed` through to the request. It is ignored for now.
        let response = self.request_generate(settings)?;
        let maps = Self::response_to_maps(response, game_data)?;
        if maps.is_empty() {
            bail!("Map generation server returned no valid maps");
        }
        Ok(maps)
    }

    fn request_refill(&self, settings: &RandomizerSettings) -> Result<()> {
        self.post("/refill", settings)?;
        Ok(())
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

    // Green Brinstar Main Shaft internal door
    doors.push((
        (Some(0x18cee), Some(0x18d06)),
        (Some(0x18d06), Some(0x18cee)),
        true,
    ));

    // Pants Room internal door
    doors.push((
        (Some(0x1a78c), Some(0x1a7a4)),
        (Some(0x1a7a4), Some(0x1a78c)),
        true,
    ));

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
