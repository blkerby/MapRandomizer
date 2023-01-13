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
type DoorPtrPair = (Option<NodePtr>, Option<NodePtr>);

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
    EnergyRefill,
    MissileRefill,
    SuperRefill,
    PowerBombRefill,
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
    unlocked_node_map: HashMap<(RoomId, NodeId), NodeId>,
    door_ptr_pair_map: HashMap<DoorPtrPair, (RoomId, NodeId)>,
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
            if weapon_json.has_key("shotRequires") {
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
        let req = Requirement::And(
            self.parse_requires_list(&json_value["requires"].members().as_slice()),
        );
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
            if value == "never" {
                return Requirement::Never;
            } else if let Some(&tech_id) = self.tech_isv.index_by_key.get(value) {
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
                let mut allowed_weapons: WeaponMask = if value.has_key("explicitWeapons") {
                    assert!(value["explicitWeapons"].is_array());
                    let mut weapon_mask = 0;
                    for weapon_name in value["explicitWeapons"].members() {
                        if self
                            .weapon_isv
                            .index_by_key
                            .contains_key(weapon_name.as_str().unwrap())
                        {
                            weapon_mask |=
                                1 << self.weapon_isv.index_by_key[weapon_name.as_str().unwrap()];
                        }
                    }
                    weapon_mask
                } else {
                    (1 << self.weapon_isv.keys.len()) - 1
                };
                if value.has_key("excludedWeapons") {
                    assert!(value["excludedWeapons"].is_array());
                    for weapon_name in value["excludedWeapons"].members() {
                        if self
                            .weapon_isv
                            .index_by_key
                            .contains_key(weapon_name.as_str().unwrap())
                        {
                            allowed_weapons &=
                                !(1 << self.weapon_isv.index_by_key[weapon_name.as_str().unwrap()]);
                        }
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
        let from_vertex_id = self.vertex_isv.index_by_key[&(220, 2, 0)]; // Pants Room
        let to_vertex_id = self.vertex_isv.index_by_key[&(322, 1, 0)]; // East Pants Room
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
            let preprocessed_room_json = self.preprocess_room(room_json);
            self.process_room(&preprocessed_room_json);
        }
    }

    fn preprocess_room(&mut self, room_json: &JsonValue) -> JsonValue {
        let mut new_room_json = room_json.clone();
        assert!(room_json["nodes"].is_array());
        let mut next_node_id = room_json["nodes"]
            .members()
            .map(|x| x["id"].as_usize().unwrap())
            .max()
            .unwrap()
            + 1;
        let mut extra_nodes: Vec<JsonValue> = Vec::new();
        let mut extra_links: Vec<JsonValue> = Vec::new();
        let room_id = room_json["id"].as_usize().unwrap();

        // Rooms where we want the logic to take into account the gray door locks (elsewhere the gray doors are changed to blue):
        let door_lock_allowed_room_ids = [
            84,  // Kraid Room
            193, // Draygon's Room
            142, // Ridley's Room
            150, // Golden Torizo Room
        ];

        // Flags for which we want to add an obstacle in the room, to allow progression through (or back out of) the room
        // after defeating the boss on the same graph traversal step (which cannot take into account the new flag).
        let obstacle_flags = [
            "f_DefeatedKraid",
            "f_DefeatedDraygon",
            "f_DefeatedRidley",
            "f_DefeatedGoldenTorizo",
            "f_DefeatedCrocomire",
            "f_DefeatedSporeSpawn",
            "f_DefeatedBotwoon",
            "f_MaridiaTubeBroken",
            "f_ShaktoolDoneDigging",
            "f_UsedAcidChozoStatue",
        ];

        let mut obstacle_flag: Option<String> = None;

        for node_json in new_room_json["nodes"].members_mut() {
            let node_id = node_json["id"].as_usize().unwrap();

            if room_json["name"] == "Shaktool Room" && node_json["name"] == "f_ShaktoolDoneDigging"
            {
                // Adding a dummy lock on Shaktool done digging event, so that the code below can pick it up
                // and construct a corresponding obstacle for the flag (as it expects there to be a lock).
                node_json["locks"] = json::array![{
                    "name": "Shaktool Lock",
                    "lockType": "triggeredEvent",
                    "unlockStrats": [
                        {
                            "name": "Base",
                            "notable": false,
                            "requires": [],
                        }
                    ]
                }];
            }

            if node_json.has_key("locks")
                && (!["door", "entrance"].contains(&node_json["nodeType"].as_str().unwrap())
                    || door_lock_allowed_room_ids.contains(&room_id))
            {
                assert!(node_json["locks"].len() == 1);
                let base_node_name = node_json["name"].as_str().unwrap().to_string();
                let lock = node_json["locks"][0].clone();
                let yields = node_json["yields"].clone();
                node_json.remove("locks");
                let mut unlocked_node_json = node_json.clone();
                if yields != JsonValue::Null {
                    node_json.remove("yields");
                }
                node_json["name"] = JsonValue::String(base_node_name.clone() + " (locked)");
                node_json["nodeType"] = JsonValue::String("junction".to_string());

                unlocked_node_json["id"] = next_node_id.into();
                self.unlocked_node_map
                    .insert((room_id, node_id), next_node_id.into());
                unlocked_node_json["name"] = JsonValue::String(base_node_name + " (unlocked)");
                if yields != JsonValue::Null {
                    unlocked_node_json["yields"] = yields.clone();
                }
                extra_nodes.push(unlocked_node_json);

                let mut link_forward = json::object! {
                    "from": node_id,
                    "to": [{
                        "id": next_node_id,
                        "strats": lock["unlockStrats"].clone(),
                    }]
                };

                if yields != JsonValue::Null
                    && obstacle_flags.contains(&yields[0].as_str().unwrap())
                {
                    obstacle_flag = Some(yields[0].as_str().unwrap().to_string());
                    for strat in link_forward["to"][0]["strats"].members_mut() {
                        strat["obstacles"] = json::array![{
                            "id": obstacle_flag.as_ref().unwrap().to_string(),
                            "requires": []
                        }]
                    }
                }

                let link_backward = json::object! {
                    "from": next_node_id,
                    "to": [{
                        "id": node_id,
                        "strats": [{
                            "name": "Base",
                            "notable": false,
                            "requires": [],
                        }],
                    }]
                };
                extra_links.push(link_forward);
                extra_links.push(link_backward);

                next_node_id += 1;
            }
        }

        for extra_node in extra_nodes {
            new_room_json["nodes"].push(extra_node).unwrap();
        }
        for extra_link in extra_links {
            new_room_json["links"].push(extra_link).unwrap();
        }

        if obstacle_flag.is_some() {
            let obstacle_flag_name = obstacle_flag.as_ref().unwrap();
            if !new_room_json.has_key("obstacles") {
                new_room_json["obstacles"] = json::array![];
            }
            new_room_json["obstacles"]
                .push(json::object! {
                    "id": obstacle_flag_name.to_string(),
                    "name": obstacle_flag_name.to_string(),
                })
                .unwrap();
            assert!(new_room_json["links"].is_array());
            for link in new_room_json["links"].members_mut() {
                assert!(link["to"].is_array());
                for to_json in link["to"].members_mut() {
                    let mut new_strats: Vec<JsonValue> = Vec::new();
                    assert!(to_json["strats"].is_array());
                    for strat in to_json["strats"].members() {
                        let json_obstacle_flag_name = JsonValue::String(obstacle_flag_name.clone());
                        let pos = strat["requires"]
                            .members()
                            .position(|x| x == &json_obstacle_flag_name);
                        if let Some(i) = pos {
                            let mut new_strat = strat.clone();
                            if !new_strat.has_key("obstacles") {
                                new_strat["obstacles"] = json::array![];
                            }
                            new_strat["requires"].array_remove(i);
                            new_strat["obstacles"]
                                .push(json::object! {
                                    "id": obstacle_flag_name.to_string(),
                                    "requires": ["never"]
                                })
                                .unwrap();
                            new_strats.push(new_strat);
                        }
                    }
                    for strat in new_strats {
                        to_json["strats"].push(strat).unwrap();
                    }
                }
            }
        }

        new_room_json
    }

    fn process_room(&mut self, room_json: &JsonValue) {
        println!("{}", room_json["name"]);

        let room_id = room_json["id"].as_usize().unwrap();
        self.room_json_map.insert(room_id, room_json.clone());

        // Process obstacles:
        let obstacles_idx_map: HashMap<String, usize> = if room_json.has_key("obstacles") {
            assert!(room_json["obstacles"].is_array());
            room_json["obstacles"]
                .members()
                .enumerate()
                .map(|(i, x)| (x["id"].as_str().unwrap().to_string(), i))
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
            if node_json.has_key("nodeAddress") {
                let mut node_ptr =
                    parse_int::parse::<usize>(node_json["nodeAddress"].as_str().unwrap()).unwrap();
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
        for node_json in room_json["nodes"].members() {
            let node_id = node_json["id"].as_usize().unwrap();
            if node_json.has_key("spawnAt") {
                let spawn_node_id = node_json["spawnAt"].as_usize().unwrap();
                let from_vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, 0)];
                let to_vertex_id = self.vertex_isv.index_by_key[&(room_id, spawn_node_id, 0)];
                self.links.push(Link {
                    from_vertex_id,
                    to_vertex_id,
                    requirement: Requirement::Free,
                    strat_name: "spawnAt".to_string(),
                });
            }
            if node_json.has_key("utility") {
                let utility = &node_json["utility"];
                for obstacle_bitmask in 0..(1 << num_obstacles) {
                    let vertex_id =
                        self.vertex_isv.index_by_key[&(room_id, node_id, obstacle_bitmask)];
                    let mut reqs: Vec<Requirement> = Vec::new();
                    assert!(utility.is_array());
                    if utility.contains("energy") {
                        reqs.push(Requirement::EnergyRefill);
                    }
                    if utility.contains("missile") {
                        reqs.push(Requirement::MissileRefill);
                    }
                    if utility.contains("super") {
                        reqs.push(Requirement::SuperRefill);
                    }
                    if utility.contains("powerbomb") {
                        reqs.push(Requirement::PowerBombRefill);
                    }
                    self.links.push(Link {
                        from_vertex_id: vertex_id,
                        to_vertex_id: vertex_id,
                        requirement: Requirement::And(reqs),
                        strat_name: "Refill".to_string(),
                    });
                }
            }
        }
        if room_json.has_key("enemies") {
            assert!(room_json["enemies"].is_array());
            for enemy in room_json["enemies"].members() {
                if !enemy.has_key("farmCycles") {
                    continue;
                }
                let drops = &enemy["drops"];
                let drops_pb = drops["powerBomb"].as_usize().map(|x| x > 0) == Some(true);
                let drops_super = drops["super"].as_usize().map(|x| x > 0) == Some(true);
                let drops_missile = drops["missile"].as_usize().map(|x| x > 0) == Some(true);
                let drops_big_energy = drops["bigEnergy"].as_usize().map(|x| x > 0) == Some(true);
                let drops_small_energy = drops["smallEnergy"].as_usize().map(|x| x > 0) == Some(true);
                let mut reqs: Vec<Requirement> = Vec::new();
                if drops_pb {
                    reqs.push(Requirement::PowerBombRefill);
                }
                if drops_super {
                    reqs.push(Requirement::SuperRefill);
                }
                if drops_missile {
                    reqs.push(Requirement::MissileRefill);
                }
                if drops_big_energy || drops_small_energy {
                    reqs.push(Requirement::EnergyRefill);
                }
                let farm_name = format!("Farm {}", enemy["enemyName"].as_str().unwrap());
                assert!(enemy["homeNodes"].is_array());
                for node_id_json in enemy["homeNodes"].members() {
                    let node_id = node_id_json.as_usize().unwrap();
                    for obstacle_bitmask in 0..(1 << num_obstacles) {
                        let vertex_id =
                            self.vertex_isv.index_by_key[&(room_id, node_id, obstacle_bitmask)];
                        self.links.push(Link {
                            from_vertex_id: vertex_id,
                            to_vertex_id: vertex_id,
                            requirement: Requirement::And(reqs.clone()),
                            strat_name: farm_name.to_string(),
                        })
                    }
                }
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

                        if strat_json.has_key("obstacles") {
                            assert!(strat_json["obstacles"].is_array());
                            for obstacle in strat_json["obstacles"].members() {
                                let obstacle_idx =
                                    obstacles_idx_map[obstacle["id"].as_str().unwrap()];
                                to_obstacles_bitmask |= 1 << obstacle_idx;
                                if (1 << obstacle_idx) & from_obstacles_bitmask == 0 {
                                    assert!(obstacle["requires"].is_array());
                                    requires_json
                                        .extend(obstacle["requires"].members().map(|x| x.clone()));
                                }
                                let room_obstacle = &room_json["obstacles"][obstacle_idx];
                                if room_obstacle.has_key("requires") {
                                    assert!(room_obstacle["requires"].is_array());
                                    requires_json.extend(
                                        room_obstacle["requires"].members().map(|x| x.clone()),
                                    );
                                }
                                if obstacle.has_key("additionalObstacles") {
                                    assert!(obstacle["additionalObstacles"].is_array());
                                    for additional_obstacle_id in
                                        obstacle["additionalObstacles"].members()
                                    {
                                        let additional_obstacle_idx = obstacles_idx_map
                                            [additional_obstacle_id.as_str().unwrap()];
                                        to_obstacles_bitmask |= 1 << additional_obstacle_idx;
                                    }
                                }
                            }
                        }
                        let requirement =
                            Requirement::And(self.parse_requires_list(&requires_json));
                        let from_vertex_id = self.vertex_isv.index_by_key
                            [&(room_id, from_node_id, from_obstacles_bitmask)];
                        let to_vertex_id = self.vertex_isv.index_by_key
                            [&(room_id, to_node_id, to_obstacles_bitmask)];
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
        let connection_pattern = self.path.to_str().unwrap().to_string() + "/connection/**/*.json";
        for entry in glob::glob(&connection_pattern).unwrap() {
            if let Ok(path) = entry {
                if !path.to_str().unwrap().contains("ceres") {
                    println!("{}", path.display());
                    self.process_connections(&read_json(&path));
                }
            } else {
                panic!("Error processing connection path: {}", entry.err().unwrap());
            }
        }
    }

    fn process_connections(&mut self, connection_file_json: &JsonValue) {
        assert!(connection_file_json["connections"].is_array());
        for connection in connection_file_json["connections"].members() {
            assert!(connection["nodes"].is_array());
            assert!(connection["nodes"].len() == 2);
            let src_pair = (
                connection["nodes"][0]["roomid"].as_usize().unwrap(),
                connection["nodes"][0]["nodeid"].as_usize().unwrap(),
            );
            let dst_pair = (
                connection["nodes"][1]["roomid"].as_usize().unwrap(),
                connection["nodes"][1]["nodeid"].as_usize().unwrap(),
            );
            self.add_connection(src_pair, dst_pair);
            self.add_connection(dst_pair, src_pair);
        }
    }

    fn add_connection(&mut self, mut src: (RoomId, NodeId), dst: (RoomId, NodeId)) {
        if self.unlocked_node_map.contains_key(&src) {
            let src_room_id = src.0;
            src = (src_room_id, self.unlocked_node_map[&src])
        }
        let src_ptr = self.node_ptr_map.get(&src).map(|x| *x);
        let dst_ptr = self.node_ptr_map.get(&dst).map(|x| *x);
        if src_ptr.is_some() || dst_ptr.is_some() {
            self.door_ptr_pair_map.insert((src_ptr, dst_ptr), src);
        }
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
        println!("num vertices: {}", sm_json_data.vertex_isv.keys.len());
        println!("num links: {}", sm_json_data.links.len());
        sm_json_data
    }
}
