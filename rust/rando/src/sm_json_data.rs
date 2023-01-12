use hashbrown::{HashMap, HashSet};
use json::{self, JsonValue};
use std::borrow::ToOwned;
use std::fs::File;
use std::hash::Hash;
use std::path::{Path, PathBuf};

type TechId = usize;
type ItemId = usize;
type FlagId = usize;
type RoomId = usize;
type NodeId = usize;
type NodePtr = usize;
type ObstacleMask = usize;
type WeaponMask = usize;
type Capacity = usize;

#[derive(Default)]
pub struct IndexedVec<T: Hash + Eq> {
    keys: Vec<T>,
    index_by_key: HashMap<T, usize>,
}

#[derive(Clone, Debug)]
pub enum Requirement {
    Free,
    Never,
    Tech(TechId),
    Item(ItemId),
    Flag(ItemId),
    ShineCharge {
        used_tiles: usize,
        shinespark_frames: usize,
    },
    HeatFrames(usize),
    LavaFrames(usize),
    LavaPhysicsFrames(usize),
    Damage(usize),
    EnemyKill(WeaponMask),
    EnergyDrain,
    // EnergyRefill,
    // MissileRefill,
    // SupersRefill,
    // PowerBombsRefill,
    Missiles(usize),
    Supers(usize),
    PowerBombs(usize),
    And(Vec<Requirement>),
    Or(Vec<Requirement>),
}

pub struct Link {
    from_vertex_id: usize,
    to_vertex_id: usize,
    requirement: Requirement,
    strat_name: String,
}

#[derive(Default)]
pub struct SMJsonData {
    path: PathBuf,
    tech_isv: IndexedVec<String>,
    item_isv: IndexedVec<String>,
    flag_isv: IndexedVec<String>,
    weapon_isv: IndexedVec<String>,
    enemy_attack_damage: HashMap<(String, String), Capacity>,
    enemy_vulnerabilities: HashMap<String, WeaponMask>,
    weapon_json_map: HashMap<String, JsonValue>,
    helper_json_map: HashMap<String, JsonValue>,
    helpers: HashMap<String, Option<Requirement>>,
    room_json_map: HashMap<RoomId, JsonValue>,
    node_json_map: HashMap<(RoomId, NodeId), JsonValue>,
    node_ptr_map: HashMap<(RoomId, NodeId), NodePtr>,
    door_ptr_pair_map: HashMap<(NodePtr, NodePtr), (RoomId, NodeId)>,
    vertex_isv: IndexedVec<(RoomId, NodeId, ObstacleMask)>,
    links: Vec<Link>,
}

impl<T: Hash + Eq> IndexedVec<T> {
    pub fn add<U: ToOwned<Owned = T> + ?Sized>(&mut self, name: &U) {
        self.index_by_key.insert(name.to_owned(), self.keys.len());
        self.keys.push(name.to_owned());
    }
}

fn read_json(path: &Path) -> JsonValue {
    let file = File::open(path).expect(&format!("unable to open {}", path.display()));
    let json_str =
        std::io::read_to_string(file).expect(&format!("unable to read {}", path.display()));
    let json_data = json::parse(&json_str).expect(&format!("unable to parse {}", path.display()));
    json_data
}

impl SMJsonData {
    fn load_tech(&mut self) {
        let full_tech_json = read_json(&self.path.join("tech.json"));
        assert!(full_tech_json["techCategories"].is_array());
        for tech_category in full_tech_json["techCategories"].members() {
            assert!(tech_category["techs"].is_array());
            for tech_json in tech_category["techs"].members() {
                self.process_tech_rec(&tech_json);
            }
        }
    }

    fn process_tech_rec(&mut self, tech_json: &JsonValue) {
        let name = tech_json["name"].as_str().unwrap();
        self.tech_isv.add(name);
        if tech_json.has_key("extensionTechs") {
            assert!(tech_json["extensionTechs"].is_array());
            for ext_tech in tech_json["extensionTechs"].members() {
                self.process_tech_rec(ext_tech);
            }    
        }
    }

    fn load_items_and_flags(&mut self) {
        let item_json = read_json(&self.path.join("items.json"));
        assert!(item_json["implicitItems"].is_array());
        for item_name in item_json["implicitItems"].members() {
            self.item_isv.add(item_name.as_str().unwrap());
        }
        assert!(item_json["upgradeItems"].is_array());
        for item in item_json["upgradeItems"].members() {
            self.item_isv.add(item["name"].as_str().unwrap());
        }
        assert!(item_json["expansionItems"].is_array());
        for item in item_json["expansionItems"].members() {
            self.item_isv.add(item["name"].as_str().unwrap());
        }
        assert!(item_json["gameFlags"].is_array());
        for flag_name in item_json["gameFlags"].members() {
            self.flag_isv.add(flag_name.as_str().unwrap());
        }
    }

    fn load_weapons(&mut self) {
        let weapons_json = read_json(&self.path.join("weapons/main.json"));
        assert!(weapons_json["weapons"].is_array());
        for weapon_json in weapons_json["weapons"].members() {
            let name = weapon_json["name"].as_str().unwrap();
            if weapon_json["situational"].as_bool().unwrap() {
                continue;
            }
            if weapon_json.contains("shotRequires") {
                // TODO: Take weapon ammo into account instead of skipping this.
                continue;
            }
            self.weapon_json_map
                .insert(name.to_string(), weapon_json.clone());
            self.weapon_isv.add(name);
        }
    }

    fn load_enemies(&mut self) {
        for file in ["main.json", "bosses/main.json"] {
            let enemies_json = read_json(&self.path.join("enemies").join(file));
            assert!(enemies_json["enemies"].is_array());
            for enemy_json in enemies_json["enemies"].members() {
                let enemy_name = enemy_json["name"].as_str().unwrap();
                assert!(enemy_json["attacks"].is_array());
                for attack in enemy_json["attacks"].members() {
                    let attack_name = attack["name"].as_str().unwrap();
                    let damage = attack["baseDamage"].as_usize().unwrap() as Capacity;
                    self.enemy_attack_damage
                        .insert((enemy_name.to_string(), attack_name.to_string()), damage);
                }
                self.enemy_vulnerabilities.insert(
                    enemy_name.to_string(),
                    self.get_enemy_vulnerabilities(enemy_json),
                );
            }
        }
    }

    fn get_enemy_vulnerabilities(&self, enemy_json: &JsonValue) -> WeaponMask {
        assert!(enemy_json["invul"].is_array());
        let invul: HashSet<String> = enemy_json["invul"]
            .members()
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        let mut vul_mask = 0;
        'weapon: for (i, weapon_name) in self.weapon_isv.keys.iter().enumerate() {
            let weapon_json = &self.weapon_json_map[weapon_name];
            if invul.contains(weapon_name) {
                continue;
            }
            assert!(weapon_json["categories"].is_array());
            for cat in weapon_json["categories"]
                .members()
                .map(|x| x.as_str().unwrap())
            {
                if invul.contains(cat) {
                    continue 'weapon;
                }
            }
            vul_mask |= 1 << i;
        }
        return vul_mask;
    }

    fn load_helpers(&mut self) {
        let helpers_json = read_json(&self.path.join("helpers.json"));
        assert!(helpers_json["helpers"].is_array());
        for helper in helpers_json["helpers"].members() {
            self.helper_json_map
                .insert(helper["name"].as_str().unwrap().to_owned(), helper.clone());
        }
        for helper in helpers_json["helpers"].members() {
            let _ = self.parse_helper(helper["name"].as_str().unwrap());
        }
    }

    fn parse_helper(&mut self, name: &str) -> Requirement {
        self.helpers.insert(name.to_owned(), None);
        let json_value = self.helper_json_map[name].clone();
        assert!(json_value["requires"].is_array());
        let req = Requirement::And(self.parse_requires_list(&json_value["requires"].members().as_slice()));
        *self.helpers.get_mut(name).unwrap() = Some(req.clone());
        req
    }

    fn parse_requires_list(&mut self, json_values: &[JsonValue]) -> Vec<Requirement> {
        let mut req_list: Vec<Requirement> = Vec::new();
        for json_req in json_values {
            req_list.push(self.parse_requirement(json_req));
        }
        req_list
    }

    fn parse_requirement(&mut self, json_value: &JsonValue) -> Requirement {
        if json_value.is_string() {
            let value = json_value.as_str().unwrap();
            if let Some(&tech_id) = self.tech_isv.index_by_key.get(value) {
                return Requirement::Tech(tech_id as TechId);
            } else if let Some(&item_id) = self.item_isv.index_by_key.get(value) {
                return Requirement::Item(item_id as ItemId);
            } else if let Some(&flag_id) = self.flag_isv.index_by_key.get(value) {
                return Requirement::Flag(flag_id as FlagId);
            } else if let Some(req) = self.helpers.get(value) {
                if req.is_none() {
                    panic!("Cyclic dependency in helper {}", value);
                }
                return req.clone().unwrap();
            } else if self.helper_json_map.contains_key(value) {
                return self.parse_helper(value);
            }
        } else if json_value.is_object() && json_value.len() == 1 {
            let (key, value) = json_value.entries().next().unwrap();
            if key == "or" {
                assert!(value.is_array());
                return Requirement::Or(self.parse_requires_list(value.members().as_slice()));
            } else if key == "and" {
                assert!(value.is_array());
                return Requirement::And(self.parse_requires_list(value.members().as_slice()));
            } else if key == "ammo" {
                let ammo_type = value["type"]
                    .as_str()
                    .expect(&format!("missing/invalid ammo type in {}", json_value));
                let count = value["count"]
                    .as_usize()
                    .expect(&format!("missing/invalid ammo count in {}", json_value));
                if ammo_type == "Missile" {
                    return Requirement::Missiles(count as Capacity);
                } else if ammo_type == "Super" {
                    return Requirement::Supers(count as Capacity);
                } else if ammo_type == "PowerBomb" {
                    return Requirement::PowerBombs(count as Capacity);
                } else {
                    panic!("Unexpected ammo type in {}", json_value);
                }
            } else if key == "ammoDrain" {
                // We patch out the ammo drain from the Mother Brain fight.
                return Requirement::Free;
            } else if key == "canShineCharge" {
                let used_tiles = value["usedTiles"]
                    .as_usize()
                    .expect(&format!("missing/invalid usedTiles in {}", json_value));
                let shinespark_frames = value["shinesparkFrames"].as_usize().expect(&format!(
                    "missing/invalid shinesparkFrames in {}",
                    json_value
                ));
                return Requirement::ShineCharge {
                    used_tiles,
                    shinespark_frames,
                };
            } else if key == "heatFrames" {
                let frames = value
                    .as_usize()
                    .expect(&format!("invalid heatFrames in {}", json_value));
                return Requirement::HeatFrames(frames);
            } else if key == "lavaFrames" {
                let frames = value
                    .as_usize()
                    .expect(&format!("invalid lavaFrames in {}", json_value));
                return Requirement::LavaFrames(frames);
            } else if key == "lavaPhysicsFrames" {
                let frames = value
                    .as_usize()
                    .expect(&format!("invalid lavaPhysicsFrames in {}", json_value));
                return Requirement::LavaPhysicsFrames(frames);
            } else if key == "acidFrames" {
                let frames = value
                    .as_usize()
                    .expect(&format!("invalid acidFrames in {}", json_value));
                return Requirement::Damage(3 * frames / 8);
            } else if key == "draygonElectricityFrames" {
                let frames = value.as_usize().expect(&format!(
                    "invalid draygonElectricityFrames in {}",
                    json_value
                ));
                return Requirement::Damage(frames);
            } else if key == "spikeHits" {
                let hits = value
                    .as_usize()
                    .expect(&format!("invalid spikeHits in {}", json_value));
                return Requirement::Damage(hits * 60);
            } else if key == "thornHits" {
                let hits = value
                    .as_usize()
                    .expect(&format!("invalid thornHits in {}", json_value));
                return Requirement::Damage(hits * 16);
            } else if key == "hibashiHits" {
                let hits = value
                    .as_usize()
                    .expect(&format!("invalid hibashiHits in {}", json_value));
                return Requirement::Damage(hits * 30);
            } else if key == "enemyDamage" {
                let enemy_name = value["enemy"].as_str().unwrap().to_string();
                let attack_name = value["type"].as_str().unwrap().to_string();
                let hits = value["hits"].as_usize().unwrap() as Capacity;
                let base_damage = self.enemy_attack_damage[&(enemy_name, attack_name)];
                return Requirement::Damage(hits * base_damage);
            } else if key == "enemyKill" {
                // We only consider enemy kill methods that are non-situational and do not require ammo.
                // TODO: Consider all methods.
                let mut enemy_set: HashSet<String> = HashSet::new();
                assert!(value["enemies"].is_array());
                for enemy_group in value["enemies"].members() {
                    assert!(enemy_group.is_array());
                    for enemy in enemy_group.members() {
                        enemy_set.insert(enemy.as_str().unwrap().to_string());
                    }
                }
                assert!(enemy_set.len() > 0);
                let mut allowed_weapons: WeaponMask = if value.contains("explicitWeapons") {
                    assert!(value["explicitWeapons"].is_array());
                    let mut weapon_mask = 0;
                    for weapon_name in value["explicitWeapons"].members() {
                        weapon_mask |=
                            1 << self.weapon_isv.index_by_key[weapon_name.as_str().unwrap()];
                    }
                    weapon_mask
                } else {
                    (1 << self.weapon_isv.keys.len()) - 1
                };
                if value.contains("excludedWeapons") {
                    assert!(value["excludedWeapons"].is_array());
                    for weapon_name in value["excludedWeapons"].members() {
                        allowed_weapons &=
                            !(1 << self.weapon_isv.index_by_key[weapon_name.as_str().unwrap()]);
                    }
                }
                let mut reqs: Vec<Requirement> = Vec::new();
                for enemy in &enemy_set {
                    let vul = self.enemy_vulnerabilities[enemy];
                    reqs.push(Requirement::EnemyKill(vul & allowed_weapons));
                }
                return Requirement::And(reqs);
            } else if key == "energyAtMost" {
                assert!(value.as_usize().unwrap() == 1);
                return Requirement::EnergyDrain;
            } else if key == "previousNode" {
                // Currently this is used only in the Early Supers quick crumble and Mission Impossible strats and is
                // redundant in both cases, so we treat it as free.
                return Requirement::Free;
            } else if key == "resetRoom" {
                // In all the places where this is required (excluding runways and canComeInCharged which we are not
                // yet taking into account), it seems to be essentially unnecessary (ignoring the
                // possibility of needing to take a small amount of heat damage in an adjacent room to exit and
                // reenter), so for now we treat it as free.
                return Requirement::Free;
            } else if key == "previousStratProperty" {
                // This is only used in one place in Crumble Shaft, where it doesn't seem to be necessary.
                return Requirement::Free;
            } else if key == "canComeInCharged" || key == "adjacentRunway" {
                // For now assume we can't do these.
                return Requirement::Never;
            }
        }
        panic!("Unable to parse requirement: {}", json_value)
    }

    fn load_regions(&mut self) {
        let region_pattern = self.path.to_str().unwrap().to_string() + "/region/**/*.json";
        for entry in glob::glob(&region_pattern).unwrap() {
            if let Ok(path) = entry {
                if !path.to_str().unwrap().contains("ceres") {
                    println!("{}", path.display());
                    self.process_region(&read_json(&path));
                }
            } else {
                panic!("Error processing region path: {}", entry.err().unwrap());
            }
        }
        // Add Pants Room in-room transition
        let from_vertex_id = self.vertex_isv.index_by_key[&(220, 2, 0)];  // Pants Room
        let to_vertex_id = self.vertex_isv.index_by_key[&(322, 1, 0)];  // East Pants Room
        self.links.push(Link {
            from_vertex_id, 
            to_vertex_id,
            requirement: Requirement::Free,
            strat_name: "Pants Room in-room transition".to_string(),
        });
    }

    fn process_region(&mut self, region_json: &JsonValue) {
        assert!(region_json["rooms"].is_array());
        for room_json in region_json["rooms"].members() {
            self.process_room(room_json);
        }
    }

    fn process_room(&mut self, room_json: &JsonValue) {
        println!("{}", room_json["name"]);

        let room_id = room_json["id"].as_usize().unwrap();
        self.room_json_map.insert(room_id, room_json.clone());

        // Process obstacles
        let obstacles_idx_map: HashMap<String, usize> = if room_json.contains("obstacles") {
            assert!(room_json["obstacles"].is_array());
            room_json["obstacles"]
            .members()
            .enumerate()
            .map(|(i, x)| (x["name"].as_str().unwrap().to_string(), i))
            .collect()
        } else {
            HashMap::new()
        };
        let num_obstacles = obstacles_idx_map.len();

        // Process nodes:
        assert!(room_json["nodes"].is_array());
        for node_json in room_json["nodes"].members() {
            let node_id = node_json["id"].as_usize().unwrap();
            self.node_json_map
                .insert((room_id, node_id), node_json.clone());
            if node_json.contains("nodeAddress") {
                let mut node_ptr =
                    usize::from_str_radix(node_json["nodeAddress"].as_str().unwrap(), 16).unwrap();
                // Convert East Pants Room door pointers to corresponding Pants Room pointers
                if node_ptr == 0x1A7BC {
                    node_ptr = 0x1A798;
                } else if node_ptr == 0x1A7B0 {
                    node_ptr = 0x1A7A4;
                }
                self.node_ptr_map.insert((room_id, node_id), node_ptr);
            }
            for obstacle_bitmask in 0..(1 << num_obstacles) {
                self.vertex_isv.add(&(room_id, node_id, obstacle_bitmask));
            }
        }

        // Process links:
        assert!(room_json["links"].is_array());
        for link_json in room_json["links"].members() {
            assert!(link_json["to"].is_array());
            for link_to_json in link_json["to"].members() {
                assert!(link_to_json["strats"].is_array());
                for strat_json in link_to_json["strats"].members() {
                    for from_obstacles_bitmask in 0..(1 << num_obstacles) {
                        let from_node_id = link_json["from"].as_usize().unwrap();
                        let to_node_id = link_to_json["id"].as_usize().unwrap();
                        let mut to_obstacles_bitmask = from_obstacles_bitmask;
                        assert!(strat_json["requires"].is_array());
                        let mut requires_json: Vec<JsonValue> = strat_json["requires"]
                            .members()
                            .map(|x| x.clone())
                            .collect();

                        if strat_json.contains("obstacles") {
                            assert!(strat_json["obstacles"].is_array());
                            for obstacle in strat_json["obstacles"].members() {
                                let obstacle_idx =
                                    obstacles_idx_map[obstacle["id"].as_str().unwrap()];
                                to_obstacles_bitmask |= 1 << obstacle_idx;
                                if (1 << obstacle_idx) & from_obstacles_bitmask == 0 {
                                    assert!(obstacle["requires"].is_array());
                                    requires_json.extend(obstacle["requires"].members().map(|x| x.clone()));
                                }
                                let room_obstacle = &room_json["obstacles"][obstacle_idx];
                                if room_obstacle.has_key("requires") {
                                    assert!(room_obstacle["requires"].is_array());
                                    requires_json.extend(room_obstacle["requires"].members().map(|x| x.clone()));
                                }
                                if obstacle.has_key("additionalObstacles") {
                                    assert!(obstacle["additionalObstacles"].is_array());
                                    for additional_obstacle_id in obstacle["additionalObstacles"].members() {
                                        let additional_obstacle_idx = obstacles_idx_map[additional_obstacle_id.as_str().unwrap()];
                                        to_obstacles_bitmask |= 1 << additional_obstacle_idx;
                                    }
                                }
                            }
                        }
                        let requirement = Requirement::And(self.parse_requires_list(&requires_json));
                        let from_vertex_id = self.vertex_isv.index_by_key[&(room_id, from_node_id, from_obstacles_bitmask)];
                        let to_vertex_id = self.vertex_isv.index_by_key[&(room_id, to_node_id, to_obstacles_bitmask)];
                        let strat_name = strat_json["name"].as_str().unwrap().to_string();
                        let link = Link {
                            from_vertex_id,
                            to_vertex_id,
                            requirement,
                            strat_name,
                        };
                        self.links.push(link);
                    }
                }
            }
        }
    }

    fn load_connections(&mut self) {
        
    }

    pub fn load(path: &Path) -> SMJsonData {
        let mut sm_json_data = SMJsonData::default();
        sm_json_data.path = path.to_owned();
        sm_json_data.load_tech();
        sm_json_data.load_items_and_flags();
        sm_json_data.load_weapons();
        sm_json_data.load_enemies();
        sm_json_data.load_helpers();
        sm_json_data.load_regions();
        sm_json_data.load_connections();
        sm_json_data
    }
}
