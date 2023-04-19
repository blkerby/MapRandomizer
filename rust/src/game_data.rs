use crate::randomize::escape_timer::{get_base_room_door_graph, RoomDoorGraph};
use anyhow::{bail, ensure, Context, Result};
use hashbrown::{HashMap, HashSet};
use json::{self, JsonValue};
use num_enum::TryFromPrimitive;
use serde::Serialize;
use serde_derive::Deserialize;
use std::borrow::ToOwned;
use std::fs::File;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use strum::VariantNames;
use strum_macros::{EnumString, EnumVariantNames};

#[derive(Deserialize, Clone)]
pub struct Map {
    pub rooms: Vec<(usize, usize)>, // (x, y) of upper-left corner of room on map
    pub doors: Vec<(
        (Option<usize>, Option<usize>), // Source (exit_ptr, entrance_ptr)
        (Option<usize>, Option<usize>), // Destination (exit_ptr, entrance_ptr)
        bool,                           // bidirectional
    )>,
    pub area: Vec<usize>,    // Area number: 0, 1, 2, 3, 4, or 5
    pub subarea: Vec<usize>, // Subarea number: 0 or 1
}

pub type TechId = usize; // Index into GameData.tech_isv.keys: distinct tech names from sm-json-data
pub type StratId = usize; // Index into GameData.notable_strats_isv.keys: distinct notable strat names from sm-json-data
pub type ItemId = usize; // Index into GameData.item_isv.keys: 21 distinct item names
pub type FlagId = usize; // Index into GameData.flag_isv.keys: distinct game flag names from sm-json-data
pub type RoomId = usize; // Room ID from sm-json-data
pub type RoomPtr = usize; // Room pointer (PC address of room header)
pub type NodeId = usize; // Node ID from sm-json-data (only unique within a room)
pub type NodePtr = usize; // nodeAddress from sm-json-data: for items this is the PC address of PLM, for doors it is PC address of door data
pub type VertexId = usize; // Index into GameData.vertex_isv.keys: (room_id, node_id, obstacle_bitmask) combinations
pub type ItemLocationId = usize; // Index into GameData.item_locations: 100 nodes each containing an item
pub type ObstacleMask = usize; // Bitmask where `i`th bit (from least significant) indicates `i`th obstacle cleared within a room
pub type WeaponMask = usize; // Bitmask where `i`th bit indicates availability of (or vulnerability to) `i`th weapon.
pub type Capacity = i32; // Data type used to represent quantities of energy, ammo, etc.
pub type DoorPtr = usize; // PC address of door data for exiting given door
pub type DoorPtrPair = (Option<DoorPtr>, Option<DoorPtr>); // PC addresses of door data for exiting & entering given door (from vanilla door connection)
pub type TilesetIdx = usize; // Tileset index
pub type AreaIdx = usize; // Area index (0..5)

#[derive(Default, Clone)]
pub struct IndexedVec<T: Hash + Eq> {
    pub keys: Vec<T>,
    pub index_by_key: HashMap<T, usize>,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    EnumString,
    EnumVariantNames,
    TryFromPrimitive,
    Serialize,
    Deserialize,
)]
#[repr(usize)]
// Note: the ordering of these items is significant; it must correspond to the ordering of PLM types:
pub enum Item {
    ETank,        // 0
    Missile,      // 1
    Super,        // 2
    PowerBomb,    // 3
    Bombs,        // 4
    Charge,       // 5
    Ice,          // 6
    HiJump,       // 7
    SpeedBooster, // 8
    Wave,         // 9
    Spazer,       // 10
    SpringBall,   // 11
    Varia,        // 12
    Gravity,      // 13
    XRayScope,    // 14
    Plasma,       // 15
    Grapple,      // 16
    SpaceJump,    // 17
    ScrewAttack,  // 18
    Morph,        // 19
    ReserveTank,  // 20
}

impl Item {
    pub fn is_unique(self) -> bool {
        ![
            Item::Missile,
            Item::Super,
            Item::PowerBomb,
            Item::ETank,
            Item::ReserveTank,
        ]
        .contains(&self)
    }
}

#[derive(Clone, Debug)]
pub enum Requirement {
    Free,
    Never,
    Tech(TechId),
    Strat(StratId),
    Item(ItemId),
    Flag(FlagId),
    ShineCharge {
        shinespark_tech_id: usize,
        used_tiles: f32,
        shinespark_frames: i32,
        excess_shinespark_frames: i32,
    },
    HeatFrames(i32),
    LavaFrames(i32),
    LavaPhysicsFrames(i32),
    AcidFrames(i32),
    Damage(i32),
    // Energy(i32),
    Missiles(i32),
    MissilesCapacity(i32),
    Supers(i32),
    PowerBombs(i32),
    EnergyRefill,
    ReserveRefill,
    MissileRefill,
    SuperRefill,
    PowerBombRefill,
    EnergyDrain,
    ReserveTrigger {
        min_reserve_energy: i32,
        max_reserve_energy: i32,
    },
    EnemyKill {
        count: i32,
        vul: EnemyVulnerabilities,
    },
    PhantoonFight {},
    DraygonFight {
        can_be_patient_tech_id: usize,
    },
    RidleyFight {
        can_be_patient_tech_id: usize,
    },
    BotwoonFight {
        second_phase: bool,
    },
    AdjacentRunway {
        room_id: RoomId,
        node_id: NodeId,
        used_tiles: f32,
        use_frames: Option<i32>,
        physics: Option<String>,
        override_runway_requirements: bool,
    },
    CanComeInCharged {
        shinespark_tech_id: usize,
        room_id: RoomId,
        node_id: NodeId,
        frames_remaining: i32,
        shinespark_frames: i32,
        excess_shinespark_frames: i32,
    },
    ComeInWithGMode {
        room_id: RoomId,
        node_ids: Vec<NodeId>,
        mode: String,
        artificial_morph: bool,
    },
    And(Vec<Requirement>),
    Or(Vec<Requirement>),
}

impl Requirement {
    pub fn make_and(reqs: Vec<Requirement>) -> Requirement {
        let mut out_reqs: Vec<Requirement> = vec![];
        for req in reqs {
            if let Requirement::Never = req {
                return Requirement::Never;
            } else if let Requirement::Free = req {
                continue;
            }
            out_reqs.push(req);
        }
        if out_reqs.len() == 0 {
            Requirement::Free
        } else if out_reqs.len() == 1 {
            out_reqs.into_iter().next().unwrap()
        } else {
            Requirement::And(out_reqs)
        }
    }

    pub fn make_or(reqs: Vec<Requirement>) -> Requirement {
        let mut out_reqs: Vec<Requirement> = vec![];
        for req in reqs {
            if let Requirement::Never = req {
                continue;
            } else if let Requirement::Free = req {
                return Requirement::Free;
            }
            out_reqs.push(req);
        }
        if out_reqs.len() == 0 {
            Requirement::Never
        } else if out_reqs.len() == 1 {
            out_reqs.into_iter().next().unwrap()
        } else {
            Requirement::Or(out_reqs)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Runway {
    // TODO: add more details like slopes, openEnd
    pub name: String,
    pub length: i32,
    pub open_end: i32,
    pub requirement: Requirement,
    pub physics: String,
    pub heated: bool,
    pub usable_coming_in: bool,
}

#[derive(Debug)]
pub struct CanLeaveCharged {
    // TODO: add more details like slopes, openEnd
    pub frames_remaining: i32,
    pub used_tiles: i32,
    pub requirement: Requirement,
    pub shinespark_frames: Option<i32>,
}

#[derive(Debug)]
pub struct LeaveWithGModeSetup {
    pub requirement: Requirement,
}

#[derive(Debug)]
pub struct LeaveWithGMode {
    pub artificial_morph: bool,
    pub requirement: Requirement,
}

#[derive(Clone, Debug)]
pub struct Link {
    pub from_vertex_id: VertexId,
    pub to_vertex_id: VertexId,
    pub requirement: Requirement,
    pub notable_strat_name: Option<String>,
    pub strat_name: String,
    pub strat_notes: Vec<String>,
}

#[derive(Deserialize, Default, Clone)]
pub struct RoomGeometryDoor {
    pub direction: String,
    pub x: usize,
    pub y: usize,
    pub exit_ptr: Option<usize>,
    pub entrance_ptr: Option<usize>,
    pub subtype: String,
}

#[derive(Deserialize, Default, Clone)]
pub struct RoomGeometryItem {
    pub x: usize,
    pub y: usize,
    pub addr: usize,
}

pub type RoomGeometryRoomIdx = usize;
pub type RoomGeometryDoorIdx = usize;
pub type RoomGeometryPartIdx = usize;

#[derive(Deserialize, Default, Clone)]
pub struct RoomGeometry {
    pub name: String,
    pub area: usize,
    pub rom_address: usize,
    pub twin_rom_address: Option<usize>,
    pub map: Vec<Vec<u8>>,
    pub doors: Vec<RoomGeometryDoor>,
    pub parts: Vec<Vec<RoomGeometryDoorIdx>>,
    pub durable_part_connections: Vec<(RoomGeometryPartIdx, RoomGeometryPartIdx)>,
    pub transient_part_connections: Vec<(RoomGeometryPartIdx, RoomGeometryPartIdx)>,
    pub items: Vec<RoomGeometryItem>,
    pub node_tiles: Vec<(usize, Vec<(usize, usize)>)>,
    pub twin_node_tiles: Option<Vec<(usize, Vec<(usize, usize)>)>>,
}

#[derive(Clone, Debug)]
pub struct EnemyVulnerabilities {
    pub hp: i32,
    pub non_ammo_vulnerabilities: WeaponMask,
    pub missile_damage: i32,
    pub super_damage: i32,
    pub power_bomb_damage: i32,
}

// TODO: Clean this up, e.g. pull out a separate structure to hold
// temporary data used only during loading, replace any
// remaining JsonValue types in the main struct with something
// more structured; combine maps with the same keys; also maybe unify the room geometry data
// with sm-json-data and cut back on the amount of different
// keys/IDs/indexes for rooms, nodes, and doors.
#[derive(Default)]
pub struct GameData {
    sm_json_data_path: PathBuf,
    pub tech_isv: IndexedVec<String>,
    pub notable_strat_isv: IndexedVec<String>,
    pub flag_isv: IndexedVec<String>,
    pub item_isv: IndexedVec<String>,
    weapon_isv: IndexedVec<String>,
    enemy_attack_damage: HashMap<(String, String), Capacity>,
    enemy_vulnerabilities: HashMap<String, EnemyVulnerabilities>,
    enemy_json: HashMap<String, JsonValue>,
    weapon_json_map: HashMap<String, JsonValue>,
    non_ammo_weapon_mask: WeaponMask,
    tech_json_map: HashMap<String, JsonValue>,
    helper_json_map: HashMap<String, JsonValue>,
    tech: HashMap<String, Option<Requirement>>,
    pub helpers: HashMap<String, Option<Requirement>>,
    pub room_json_map: HashMap<RoomId, JsonValue>,
    pub room_obstacle_idx_map: HashMap<RoomId, HashMap<String, usize>>,
    pub node_json_map: HashMap<(RoomId, NodeId), JsonValue>,
    pub node_spawn_at_map: HashMap<(RoomId, NodeId), NodeId>,
    pub node_runways_map: HashMap<(RoomId, NodeId), Vec<Runway>>,
    pub node_can_leave_charged_map: HashMap<(RoomId, NodeId), Vec<CanLeaveCharged>>,
    pub node_leave_with_gmode_map: HashMap<(RoomId, NodeId), Vec<LeaveWithGMode>>,
    pub node_leave_with_gmode_setup_map: HashMap<(RoomId, NodeId), Vec<LeaveWithGModeSetup>>,
    pub node_ptr_map: HashMap<(RoomId, NodeId), NodePtr>,
    unlocked_node_map: HashMap<(RoomId, NodeId), NodeId>,
    pub room_num_obstacles: HashMap<RoomId, usize>,
    pub door_ptr_pair_map: HashMap<DoorPtrPair, (RoomId, NodeId)>,
    pub vertex_isv: IndexedVec<(RoomId, NodeId, ObstacleMask)>,
    pub item_locations: Vec<(RoomId, NodeId)>,
    pub item_vertex_ids: Vec<Vec<VertexId>>,
    pub flag_locations: Vec<(RoomId, NodeId, FlagId)>,
    pub flag_vertex_ids: Vec<Vec<VertexId>>,
    pub links: Vec<Link>,
    pub room_geometry: Vec<RoomGeometry>,
    pub room_and_door_idxs_by_door_ptr_pair:
        HashMap<DoorPtrPair, (RoomGeometryRoomIdx, RoomGeometryDoorIdx)>,
    pub room_ptr_by_id: HashMap<RoomId, RoomPtr>,
    pub room_id_by_ptr: HashMap<RoomPtr, RoomId>,
    pub raw_room_id_by_ptr: HashMap<RoomPtr, RoomId>, // Does not replace twin room pointer with corresponding main room pointer
    pub room_idx_by_ptr: HashMap<RoomPtr, RoomGeometryRoomIdx>,
    pub room_idx_by_name: HashMap<String, RoomGeometryRoomIdx>,
    pub node_tile_coords: HashMap<(RoomId, NodeId), Vec<(usize, usize)>>,
    pub base_room_door_graph: RoomDoorGraph,
    pub area_names: Vec<String>,
    pub area_map_ptrs: Vec<isize>,
    pub tech_description: HashMap<String, String>,
    pub tech_dependencies: HashMap<String, Vec<String>>,
    pub strat_dependencies: HashMap<String, Vec<String>>,
    pub strat_area: HashMap<String, String>,
    pub strat_room: HashMap<String, String>,
    pub strat_description: HashMap<String, String>,
    pub palette_data: Vec<HashMap<TilesetIdx, [[u8; 3]; 128]>>,
}

impl<T: Hash + Eq> IndexedVec<T> {
    pub fn add<U: ToOwned<Owned = T> + ?Sized>(&mut self, name: &U) -> usize {
        if !self.index_by_key.contains_key(&name.to_owned()) {
            let idx = self.keys.len();
            self.index_by_key.insert(name.to_owned(), self.keys.len());
            self.keys.push(name.to_owned());
            idx
        } else {
            self.index_by_key[&name.to_owned()]
        }
    }
}

fn read_json(path: &Path) -> Result<JsonValue> {
    let file = File::open(path).with_context(|| format!("unable to open {}", path.display()))?;
    let json_str = std::io::read_to_string(file)
        .with_context(|| format!("unable to read {}", path.display()))?;
    let json_data =
        json::parse(&json_str).with_context(|| format!("unable to parse {}", path.display()))?;
    Ok(json_data)
}

#[derive(Default)]
struct RequirementContext<'a> {
    room_id: RoomId,
    _from_node_id: NodeId, // Usable for debugging
    from_obstacles_bitmask: ObstacleMask,
    obstacles_idx_map: Option<&'a HashMap<String, usize>>,
}

impl GameData {
    fn load_tech(&mut self) -> Result<()> {
        let full_tech_json = read_json(&self.sm_json_data_path.join("tech.json"))?;
        ensure!(full_tech_json["techCategories"].is_array());
        for tech_category in full_tech_json["techCategories"].members() {
            ensure!(tech_category["techs"].is_array());
            for tech_json in tech_category["techs"].members() {
                self.load_tech_rec(tech_json)?;
            }
        }
        Ok(())
    }

    fn load_tech_rec(&mut self, tech_json: &JsonValue) -> Result<()> {
        let name = tech_json["name"]
            .as_str()
            .context("Missing 'name' in tech")?;
        self.tech_isv.add(name);

        let desc = if tech_json["note"].is_string() {
            tech_json["note"].as_str().unwrap().to_string()
        } else if tech_json["note"].is_array() {
            let notes: Vec<String> = tech_json["note"]
                .members()
                .map(|x| x.as_str().unwrap().to_string())
                .collect();
            notes.join(" ")
        } else {
            String::new()
        };

        self.tech_description.insert(name.to_string(), desc);
        self.tech_json_map
            .insert(name.to_string(), tech_json.clone());
        if tech_json.has_key("extensionTechs") {
            ensure!(tech_json["extensionTechs"].is_array());
            for ext_tech in tech_json["extensionTechs"].members() {
                self.load_tech_rec(ext_tech)?;
            }
        }
        Ok(())
    }

    fn extract_tech_dependencies(&self, req: &Requirement) -> HashSet<String> {
        match req {
            Requirement::Tech(tech_id) => vec![self.tech_isv.keys[*tech_id].clone()]
                .into_iter()
                .collect(),
            Requirement::And(sub_reqs) => {
                let mut out: HashSet<String> = HashSet::new();
                for r in sub_reqs {
                    out.extend(self.extract_tech_dependencies(r));
                }
                out
            }
            _ => HashSet::new(),
        }
    }

    fn get_tech_requirement(&mut self, tech_name: &str) -> Result<Requirement> {
        if let Some(req_opt) = self.tech.get(tech_name) {
            if let Some(req) = req_opt {
                return Ok(req.clone());
            } else {
                bail!("Circular dependence in tech: {}", tech_name);
            }
        }
        // if self.tech.contains_key(tech_name) {
        //     return self.tech[tech_name].clone().unwrap();
        // }

        // Temporarily insert a None value to act as a sentinel for detecting circular dependencies:
        self.tech.insert(tech_name.to_string(), None);

        let tech_json = &self.tech_json_map[tech_name].clone();
        let req = if tech_json.has_key("requires") {
            let ctx = RequirementContext::default();
            let mut reqs =
                self.parse_requires_list(tech_json["requires"].members().as_slice(), &ctx)?;
            reqs.push(Requirement::Tech(self.tech_isv.index_by_key[tech_name]));
            Requirement::make_and(reqs)
        } else {
            Requirement::Tech(self.tech_isv.index_by_key[tech_name])
        };
        *self.tech.get_mut(tech_name).unwrap() = Some(req.clone());
        Ok(req)
    }

    fn load_items_and_flags(&mut self) -> Result<()> {
        let item_json = read_json(&self.sm_json_data_path.join("items.json"))?;

        for item_name in Item::VARIANTS {
            self.item_isv.add(&item_name.to_string());
        }
        ensure!(item_json["gameFlags"].is_array());
        for flag_name in item_json["gameFlags"].members() {
            self.flag_isv.add(flag_name.as_str().unwrap());
        }

        // Add randomizer-specific flags:
        self.flag_isv.add("f_AllItemsSpawn");

        Ok(())
    }

    fn load_weapons(&mut self) -> Result<()> {
        let weapons_json = read_json(&self.sm_json_data_path.join("weapons/main.json"))?;
        ensure!(weapons_json["weapons"].is_array());
        for weapon_json in weapons_json["weapons"].members() {
            let name = weapon_json["name"].as_str().unwrap();
            if weapon_json["situational"].as_bool().unwrap() {
                continue;
            }
            self.weapon_json_map
                .insert(name.to_string(), weapon_json.clone());
            self.weapon_isv.add(name);
        }

        self.non_ammo_weapon_mask = 0;
        for (i, weapon) in self.weapon_isv.keys.iter().enumerate() {
            let weapon_json = &self.weapon_json_map[weapon];
            if !weapon_json.has_key("shotRequires") {
                self.non_ammo_weapon_mask |= 1 << i;
            }
        }
        Ok(())
    }

    fn load_enemies(&mut self) -> Result<()> {
        for file in ["main.json", "bosses/main.json"] {
            let enemies_json = read_json(&self.sm_json_data_path.join("enemies").join(file))?;
            ensure!(enemies_json["enemies"].is_array());
            for enemy_json in enemies_json["enemies"].members() {
                let enemy_name = enemy_json["name"].as_str().unwrap();
                ensure!(enemy_json["attacks"].is_array());
                for attack in enemy_json["attacks"].members() {
                    let attack_name = attack["name"].as_str().unwrap();
                    let damage = attack["baseDamage"].as_i32().unwrap() as Capacity;
                    self.enemy_attack_damage
                        .insert((enemy_name.to_string(), attack_name.to_string()), damage);
                }
                self.enemy_vulnerabilities.insert(
                    enemy_name.to_string(),
                    self.get_enemy_vulnerabilities(enemy_json)?,
                );
                self.enemy_json.insert(enemy_name.to_string(), enemy_json.clone());
            }
        }
        Ok(())
    }

    fn get_enemy_damage_multiplier(&self, enemy_json: &JsonValue, weapon_name: &str) -> f32 {
        for multiplier in enemy_json["damageMultipliers"].members() {
            if multiplier["weapon"] == weapon_name {
                return multiplier["value"].as_f32().unwrap();
            }
        }
        1.0
    }

    fn get_enemy_damage_weapon(
        &self,
        enemy_json: &JsonValue,
        weapon_name: &str,
        vul_mask: WeaponMask,
    ) -> i32 {
        let multiplier = self.get_enemy_damage_multiplier(enemy_json, weapon_name);
        let weapon_idx = self.weapon_isv.index_by_key[weapon_name];
        if vul_mask & (1 << weapon_idx) == 0 {
            return 0;
        }
        match weapon_name {
            "Missile" => (100.0 * multiplier) as i32,
            "Super" => (300.0 * multiplier) as i32,
            "PowerBomb" => (400.0 * multiplier) as i32,
            _ => panic!("Unsupported weapon: {}", weapon_name),
        }
    }

    fn get_enemy_vulnerabilities(&self, enemy_json: &JsonValue) -> Result<EnemyVulnerabilities> {
        ensure!(enemy_json["invul"].is_array());
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
            ensure!(weapon_json["categories"].is_array());
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

        Ok(EnemyVulnerabilities {
            non_ammo_vulnerabilities: vul_mask & self.non_ammo_weapon_mask,
            hp: enemy_json["hp"].as_i32().unwrap(),
            missile_damage: self.get_enemy_damage_weapon(enemy_json, "Missile", vul_mask),
            super_damage: self.get_enemy_damage_weapon(enemy_json, "Super", vul_mask),
            power_bomb_damage: self.get_enemy_damage_weapon(enemy_json, "PowerBomb", vul_mask),
        })
    }

    fn load_helpers(&mut self) -> Result<()> {
        let helpers_json = read_json(&self.sm_json_data_path.join("helpers.json"))?;
        ensure!(helpers_json["helpers"].is_array());
        for helper in helpers_json["helpers"].members() {
            self.helper_json_map
                .insert(helper["name"].as_str().unwrap().to_owned(), helper.clone());
        }
        Ok(())
    }

    fn get_helper(&mut self, name: &str) -> Result<Requirement> {
        if self.helpers.contains_key(name) {
            if self.helpers[name].is_none() {
                bail!("Circular dependence in helper {}", name);
            }
            return Ok(self.helpers[name].clone().unwrap());
        }
        self.helpers.insert(name.to_owned(), None);
        let json_value = self.helper_json_map[name].clone();
        ensure!(json_value["requires"].is_array());
        let ctx = RequirementContext::default();
        let req = Requirement::make_and(
            self.parse_requires_list(&json_value["requires"].members().as_slice(), &ctx)?,
        );
        *self.helpers.get_mut(name).unwrap() = Some(req.clone());
        Ok(req)
    }

    fn parse_requires_list(
        &mut self,
        req_jsons: &[JsonValue],
        ctx: &RequirementContext,
    ) -> Result<Vec<Requirement>> {
        let mut reqs: Vec<Requirement> = Vec::new();
        for req_json in req_jsons {
            reqs.push(self.parse_requirement(req_json, ctx)?);
        }
        Ok(reqs)
    }

    fn parse_requirement(
        &mut self,
        req_json: &JsonValue,
        ctx: &RequirementContext,
    ) -> Result<Requirement> {
        if req_json.is_string() {
            let value = req_json.as_str().unwrap();
            if value == "never" {
                return Ok(Requirement::Never);
            } else if let Some(&item_id) = self.item_isv.index_by_key.get(value) {
                return Ok(Requirement::Item(item_id as ItemId));
            } else if let Some(&flag_id) = self.flag_isv.index_by_key.get(value) {
                return Ok(Requirement::Flag(flag_id as FlagId));
            } else if self.tech_json_map.contains_key(value) {
                return self.get_tech_requirement(value);
            } else if self.helper_json_map.contains_key(value) {
                return self.get_helper(value);
            }
        } else if req_json.is_object() && req_json.len() == 1 {
            let (key, value) = req_json.entries().next().unwrap();
            if key == "or" {
                ensure!(value.is_array());
                return Ok(Requirement::make_or(
                    self.parse_requires_list(value.members().as_slice(), ctx)?,
                ));
            } else if key == "and" {
                ensure!(value.is_array());
                return Ok(Requirement::make_and(
                    self.parse_requires_list(value.members().as_slice(), ctx)?,
                ));
            } else if key == "not" {
                // For now, assume we can't do these, since they could get us permanently stuck.
                return Ok(Requirement::Never);
            } else if key == "ammo" {
                let ammo_type = value["type"]
                    .as_str()
                    .expect(&format!("missing/invalid ammo type in {}", req_json));
                let count = value["count"]
                    .as_i32()
                    .expect(&format!("missing/invalid ammo count in {}", req_json));
                if ammo_type == "Missile" {
                    return Ok(Requirement::Missiles(count as Capacity));
                } else if ammo_type == "Super" {
                    return Ok(Requirement::Supers(count as Capacity));
                } else if ammo_type == "PowerBomb" {
                    return Ok(Requirement::PowerBombs(count as Capacity));
                } else {
                    bail!("Unexpected ammo type in {}", req_json);
                }
            } else if key == "resourceCapacity" {
                ensure!(value.members().len() == 1);
                let value0 = value.members().next().unwrap();
                let resource_type = value0["type"]
                    .as_str()
                    .expect(&format!("missing/invalid resource type in {}", req_json));
                let count = value0["count"]
                    .as_i32()
                    .expect(&format!("missing/invalid resource count in {}", req_json));
                if resource_type == "Missile" {
                    return Ok(Requirement::MissilesCapacity(count as Capacity));
                } else {
                    bail!("Unexpected ammo type in {}", req_json);
                }
            } else if key == "ammoDrain" {
                // We patch out the ammo drain from the Mother Brain fight.
                return Ok(Requirement::Free);
            } else if key == "canShineCharge" {
                let used_tiles = value["usedTiles"]
                    .as_f32()
                    .expect(&format!("missing/invalid usedTiles in {}", req_json));
                let shinespark_frames = value["shinesparkFrames"]
                    .as_i32()
                    .expect(&format!("missing/invalid shinesparkFrames in {}", req_json));
                let excess_shinespark_frames =
                    value["excessShinesparkFrames"].as_i32().unwrap_or(0);
                // TODO: take slopes into account
                return Ok(Requirement::ShineCharge {
                    used_tiles,
                    shinespark_frames,
                    excess_shinespark_frames,
                    shinespark_tech_id: self.tech_isv.index_by_key["canShinespark"],
                });
            } else if key == "heatFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid heatFrames in {}", req_json));
                return Ok(Requirement::HeatFrames(frames));
            } else if key == "lavaFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid lavaFrames in {}", req_json));
                return Ok(Requirement::LavaFrames(frames));
            } else if key == "lavaPhysicsFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid lavaPhysicsFrames in {}", req_json));
                return Ok(Requirement::LavaPhysicsFrames(frames));
            } else if key == "acidFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid acidFrames in {}", req_json));
                return Ok(Requirement::AcidFrames(frames));
                // return Ok(Requirement::Damage(3 * frames / 2));
            } else if key == "draygonElectricityFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid draygonElectricityFrames in {}", req_json));
                return Ok(Requirement::Damage(frames));
            } else if key == "samusEaterFrames" {
                let frames = value
                    .as_i32()
                    .expect(&format!("invalid samusEaterFrames in {}", req_json));
                return Ok(Requirement::Damage(frames / 8));
            } else if key == "spikeHits" {
                let hits = value
                    .as_i32()
                    .expect(&format!("invalid spikeHits in {}", req_json));
                return Ok(Requirement::Damage(hits * 60));
            } else if key == "thornHits" {
                let hits = value
                    .as_i32()
                    .expect(&format!("invalid thornHits in {}", req_json));
                return Ok(Requirement::Damage(hits * 16));
            } else if key == "hibashiHits" {
                let hits = value
                    .as_i32()
                    .expect(&format!("invalid hibashiHits in {}", req_json));
                return Ok(Requirement::Damage(hits * 30));
            } else if key == "enemyDamage" {
                let enemy_name = value["enemy"].as_str().unwrap().to_string();
                let attack_name = value["type"].as_str().unwrap().to_string();
                let hits = value["hits"].as_i32().unwrap() as Capacity;
                let base_damage = self.enemy_attack_damage[&(enemy_name, attack_name)];
                return Ok(Requirement::Damage(hits * base_damage));
            } else if key == "enemyKill" {
                // We only consider enemy kill methods that are non-situational and do not require ammo.
                // TODO: Consider all methods.
                let mut enemy_set: HashSet<String> = HashSet::new();
                let mut enemy_list: Vec<(String, i32)> = Vec::new();
                ensure!(value["enemies"].is_array());
                for enemy_group in value["enemies"].members() {
                    ensure!(enemy_group.is_array());
                    let mut last_enemy_name: Option<String> = None;
                    let mut cnt = 0;
                    for enemy in enemy_group.members() {
                        let enemy_name = enemy.as_str().unwrap().to_string();
                        enemy_set.insert(enemy_name.clone());
                        if Some(&enemy_name) == last_enemy_name.as_ref() {
                            cnt += 1;
                        } else {
                            if cnt > 0 {
                                enemy_list.push((last_enemy_name.unwrap(), cnt));
                            }
                            last_enemy_name = Some(enemy_name);
                            cnt = 1;
                        }
                    }
                    if cnt > 0 {
                        enemy_list.push((last_enemy_name.unwrap(), cnt));
                    }
                }

                if enemy_set.contains("Phantoon") {
                    return Ok(Requirement::PhantoonFight {});
                } else if enemy_set.contains("Draygon") {
                    return Ok(Requirement::DraygonFight {
                        can_be_patient_tech_id: self.tech_isv.index_by_key["canBePatient"],
                    });
                } else if enemy_set.contains("Ridley") {
                    return Ok(Requirement::RidleyFight {
                        can_be_patient_tech_id: self.tech_isv.index_by_key["canBePatient"],
                    });
                } else if enemy_set.contains("Botwoon 1") {
                    return Ok(Requirement::BotwoonFight {
                        second_phase: false,
                    });
                } else if enemy_set.contains("Botwoon 2") {
                    return Ok(Requirement::BotwoonFight { second_phase: true });
                }

                let mut allowed_weapons: WeaponMask = if value.has_key("explicitWeapons") {
                    ensure!(value["explicitWeapons"].is_array());
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
                    ensure!(value["excludedWeapons"].is_array());
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
                for (enemy_name, count) in &enemy_list {
                    let mut vul = self.enemy_vulnerabilities[enemy_name].clone();
                    vul.non_ammo_vulnerabilities &= allowed_weapons;
                    if allowed_weapons & (1 << self.weapon_isv.index_by_key["Missile"]) == 0 {
                        vul.missile_damage = 0;
                    }
                    if allowed_weapons & (1 << self.weapon_isv.index_by_key["Super"]) == 0 {
                        vul.super_damage = 0;
                    }
                    if allowed_weapons & (1 << self.weapon_isv.index_by_key["PowerBomb"]) == 0 {
                        vul.power_bomb_damage = 0;
                    }
                    reqs.push(Requirement::EnemyKill {
                        count: *count,
                        vul: vul,
                    });
                }
                return Ok(Requirement::make_and(reqs));
            } else if key == "energyAtMost" {
                ensure!(value.as_i32().unwrap() == 1);
                return Ok(Requirement::EnergyDrain);
            } else if key == "previousNode" {
                // Currently this is used only in the Early Supers quick crumble and Mission Impossible strats and is
                // redundant in both cases, so we treat it as free.
                return Ok(Requirement::Free);
            } else if key == "resetRoom" {
                // In all the places where this is required (excluding runways and canComeInCharged which we are not
                // yet taking into account), it seems to be essentially unnecessary (ignoring the
                // possibility of needing to take a small amount of heat damage in an adjacent room to exit and
                // reenter), so for now we treat it as free.
                return Ok(Requirement::Free);
            } else if key == "previousStratProperty" {
                // This is only used in one place in Crumble Shaft, where it doesn't seem to be necessary.
                return Ok(Requirement::Free);
            } else if key == "obstaclesCleared" {
                ensure!(value.is_array());
                if let Some(obstacles_idx_map) = ctx.obstacles_idx_map {
                    for obstacle_name_json in value.members() {
                        let obstacle_name = obstacle_name_json.as_str().unwrap();
                        if let Some(obstacle_idx) = obstacles_idx_map.get(obstacle_name) {
                            if (1 << obstacle_idx) & ctx.from_obstacles_bitmask == 0 {
                                return Ok(Requirement::Never);
                            }
                        } else {
                            bail!("Obstacle name {} not found", obstacle_name);
                        }
                    }
                    return Ok(Requirement::Free);
                } else {
                    // No obstacle state in context. This happens with cross-room strats. We're not ready to
                    // deal with obstacles yet here, so we just keep these out of logic.
                    return Ok(Requirement::Never);
                }
            } else if key == "obstaclesNotCleared" {
                ensure!(value.is_array());
                if let Some(obstacles_idx_map) = ctx.obstacles_idx_map {
                    for obstacle_name_json in value.members() {
                        let obstacle_name = obstacle_name_json.as_str().unwrap();
                        if let Some(obstacle_idx) = obstacles_idx_map.get(obstacle_name) {
                            if (1 << obstacle_idx) & ctx.from_obstacles_bitmask != 0 {
                                return Ok(Requirement::Never);
                            }
                        } else {
                            bail!("Obstacle name {} not found", obstacle_name);
                        }
                    }
                } else {
                    // No obstacle state in context. This happens with cross-room strats. We're not ready to
                    // deal with obstacles yet here, so we just keep these out of logic.
                    return Ok(Requirement::Never);
                }
                return Ok(Requirement::Free);
            } else if key == "adjacentRunway" {
                if ctx.from_obstacles_bitmask != 0 {
                    return Ok(Requirement::Never);
                }
                let physics: Option<String> = if value.has_key("physics") {
                    ensure!(value["physics"].len() == 1);
                    Some(value["physics"][0].as_str().unwrap().to_string())
                } else {
                    None
                };
                let use_frames: Option<i32> = if value.has_key("useFrames") {
                    Some(
                        value["useFrames"]
                            .as_i32()
                            .context("Expecting integer for useFrames")?,
                    )
                } else {
                    None
                };
                let mut unlocked_node_id = value["fromNode"].as_usize().unwrap();
                if self
                    .unlocked_node_map
                    .contains_key(&(ctx.room_id, unlocked_node_id))
                {
                    unlocked_node_id = self.unlocked_node_map[&(ctx.room_id, unlocked_node_id)];
                }

                return Ok(Requirement::AdjacentRunway {
                    room_id: ctx.room_id,
                    node_id: unlocked_node_id,
                    used_tiles: value["usedTiles"].as_f32().unwrap(),
                    use_frames,
                    physics: physics,
                    override_runway_requirements: value["overrideRunwayRequirements"]
                        .as_bool()
                        .unwrap_or(false),
                });
            } else if key == "canComeInCharged" {
                if ctx.from_obstacles_bitmask != 0 {
                    return Ok(Requirement::Never);
                }
                let frames_remaining = value["framesRemaining"]
                    .as_i32()
                    .with_context(|| format!("missing/invalid framesRemaining in {}", req_json))?;
                let shinespark_frames = value["shinesparkFrames"]
                    .as_i32()
                    .with_context(|| format!("missing/invalid shinesparkFrames in {}", req_json))?;
                let excess_shinespark_frames =
                    value["excessShinesparkFrames"].as_i32().unwrap_or(0);
                // if value["fromNode"].as_usize().unwrap() != ctx.src_node_id {
                //     println!("In roomId={}, canComeInCharged fromNode={}, from nodeId={}", ctx.room_id,
                //         value["fromNode"].as_usize().unwrap(), ctx.src_node_id);
                // }
                let mut unlocked_node_id = value["fromNode"].as_usize().unwrap();
                if self
                    .unlocked_node_map
                    .contains_key(&(ctx.room_id, unlocked_node_id))
                {
                    unlocked_node_id = self.unlocked_node_map[&(ctx.room_id, unlocked_node_id)];
                }
                return Ok(Requirement::CanComeInCharged {
                    shinespark_tech_id: self.tech_isv.index_by_key["canShinespark"],
                    room_id: ctx.room_id,
                    node_id: unlocked_node_id,
                    // node_id: ctx.src_node_id,
                    frames_remaining,
                    shinespark_frames,
                    excess_shinespark_frames,
                });
                // return Ok(Requirement::Never);
            } else if key == "comeInWithGMode" {
                if ctx.from_obstacles_bitmask != 0 {
                    return Ok(Requirement::Never);
                }
                let mut node_ids: Vec<NodeId> = Vec::new();
                for from_node in value["fromNodes"].members() {
                    let mut unlocked_node_id = from_node.as_usize().unwrap();
                    if self
                        .unlocked_node_map
                        .contains_key(&(ctx.room_id, unlocked_node_id))
                    {
                        unlocked_node_id = self.unlocked_node_map[&(ctx.room_id, unlocked_node_id)];
                    }
                    node_ids.push(unlocked_node_id);
                }
                let mode = value["mode"]
                    .as_str()
                    .with_context(|| format!("missing/invalid artificialMorph in {}", req_json))?;
                let artificial_morph = value["artificialMorph"]
                    .as_bool()
                    .with_context(|| format!("missing/invalid artificialMorph in {}", req_json))?;

                return Ok(Requirement::ComeInWithGMode {
                    room_id: ctx.room_id,
                    node_ids,
                    mode: mode.to_string(),
                    artificial_morph,
                });
            }
        }
        bail!("Unable to parse requirement: {}", req_json);
    }

    fn load_regions(&mut self) -> Result<()> {
        let region_pattern =
            self.sm_json_data_path.to_str().unwrap().to_string() + "/region/**/*.json";
        for entry in glob::glob(&region_pattern).unwrap() {
            if let Ok(path) = entry {
                let path_str = path.to_str().with_context(|| {
                    format!("Unable to convert path to string: {}", path.display())
                })?;
                if path_str.contains("ceres") || path_str.contains("roomDiagrams") {
                    continue;
                }
                self.process_region(&read_json(&path)?)
                    .with_context(|| format!("Processing {}", path_str))?;
            } else {
                bail!("Error processing region path: {}", entry.err().unwrap());
            }
        }
        // Add Pants Room in-room transition
        let from_vertex_id = self.vertex_isv.index_by_key[&(220, 2, 0)]; // Pants Room
        let to_vertex_id = self.vertex_isv.index_by_key[&(322, 1, 0)]; // East Pants Room
        self.links.push(Link {
            from_vertex_id,
            to_vertex_id,
            requirement: Requirement::Free,
            notable_strat_name: None,
            strat_name: "Pants Room in-room transition".to_string(),
            strat_notes: vec![],
        });
        Ok(())
    }

    fn process_region(&mut self, region_json: &JsonValue) -> Result<()> {
        ensure!(region_json["rooms"].is_array());
        for room_json in region_json["rooms"].members() {
            let room_name = room_json["name"].clone();
            let preprocessed_room_json = self
                .preprocess_room(room_json)
                .with_context(|| format!("Preprocessing room {}", room_name))?;
            self.process_room(&preprocessed_room_json)
                .with_context(|| format!("Processing room {}", room_name))?;
        }
        Ok(())
    }

    fn override_shaktool_room(&mut self, room_json: &mut JsonValue) {
        for node_json in room_json["nodes"].members_mut() {
            if node_json["name"] == "f_ShaktoolDoneDigging" {
                // Adding a dummy lock on Shaktool done digging event, so that the code in `preprocess_room`
                // can pick it up and construct a corresponding obstacle for the flag (as it expects there
                // to be a lock).
                node_json["locks"] = json::array![{
                    "name": "Shaktool Lock",
                    "lockType": "triggeredEvent",
                    "unlockStrats": [
                        {
                            "name": "Base",
                            "notable": false,
                            "requires": ["h_canUsePowerBombs"],
                        }
                    ]
                }];
            }
        }

        room_json["links"] = json::array![
            {
                "from": 1,
                "to": [
                {
                    "id": 3,
                    "strats": [
                    {
                        "name": "Base",
                        "notable": false,
                        "requires": []
                    }
                    ]
                }
                ]
            },
            {
                "from": 2,
                "to": [
                {
                    "id": 3,
                    "strats": [
                    {
                        "name": "Base",
                        "notable": false,
                        "requires": []
                    }
                    ],
                    "note": "Use the snails to dig through the sand."
                }
                ]
            },
            {
                "from": 3,
                "to": [
                {
                    "id": 1,
                    "strats": [
                    {
                        "name": "Base",
                        "notable": false,
                        "requires": [ "f_ShaktoolDoneDigging" ]
                    }
                    ]
                },
                {
                    "id": 2,
                    "strats": [
                    {
                        "name": "Base",
                        "notable": false,
                        "requires": [ "f_ShaktoolDoneDigging" ]
                    }
                    ]
                }
                ]
            }
        ];
    }

    fn preprocess_room(&mut self, room_json: &JsonValue) -> Result<JsonValue> {
        // We apply some changes to the sm-json-data specific to Map Rando.
        let mut new_room_json = room_json.clone();
        ensure!(room_json["nodes"].is_array());
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
        // Be sure to keep this consistent with patches where the gray doors are actually changed in the ROM, in
        // "patch.rs", "bomb_torizo.asm", and "gray_doors.asm".
        let door_lock_allowed_room_ids = [
            12,  // Pit Room
            82,  // Baby Kraid Room
            84,  // Kraid Room
            139, // Metal Pirates Room
            142, // Ridley's Room
            150, // Golden Torizo Room
            193, // Draygon's Room
            219, // Plasma Room
        ];

        // Flags for which we want to add an obstacle in the room, to allow progression through (or back out of) the room
        // after setting the flag on the same graph traversal step (which cannot take into account the new flag).
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

        if room_json["name"] == "Shaktool Room" {
            self.override_shaktool_room(&mut new_room_json);
        }

        for node_json in new_room_json["nodes"].members_mut() {
            let node_id = node_json["id"].as_usize().unwrap();

            if node_json.has_key("locks")
                && (!["door", "entrance"].contains(&node_json["nodeType"].as_str().unwrap())
                    || door_lock_allowed_room_ids.contains(&room_id))
            {
                ensure!(node_json["locks"].len() == 1);
                let base_node_name = node_json["name"].as_str().unwrap().to_string();
                let lock = node_json["locks"][0].clone();
                let mut yields = node_json["yields"].clone();
                if lock["yields"] != JsonValue::Null {
                    yields = lock["yields"].clone();
                }
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
                // Adding spawnAt helps shorten/clean spoiler log but interferes with the implicit leaveWithGMode:
                // unlocked_node_json["spawnAt"] = node_id.into();
                unlocked_node_json["name"] =
                    JsonValue::String(base_node_name.clone() + " (unlocked)");
                if yields != JsonValue::Null {
                    unlocked_node_json["yields"] = yields.clone();
                }

                let mut unlock_strats = lock["unlockStrats"].clone();
                if lock["name"].as_str().unwrap() == "Phantoon Fight" {
                    unlock_strats = json::array![
                        {
                            "name": "Base",
                            "requires": [
                                {"enemyKill":{
                                    "enemies": [
                                        [ "Phantoon" ]
                                    ]
                                }},
                            ]
                        }
                    ];
                }

                if unlocked_node_json["nodeType"] == "item" {
                    unlock_strats.push(json::object! {
                        "name": "All Items Spawn From Start",
                        "requires": ["f_AllItemsSpawn"]
                    })?;
                }

                extra_nodes.push(unlocked_node_json);

                if lock.has_key("lock") {
                    ensure!(lock["lock"].is_array());
                    for strat in &mut unlock_strats.members_mut() {
                        for req in lock["lock"].members() {
                            strat["requires"].push(req.clone())?;
                        }
                    }
                }

                let mut link_forward = json::object! {
                    "from": node_id,
                    "to": [{
                        "id": next_node_id,
                        "strats": unlock_strats.clone(),
                    }]
                };

                if yields != JsonValue::Null
                    && obstacle_flags.contains(&yields[0].as_str().unwrap())
                {
                    obstacle_flag = Some(yields[0].as_str().unwrap().to_string());
                    for strat in link_forward["to"][0]["strats"].members_mut() {
                        let mut obstacles = if strat["obstacles"].is_array() {
                            strat["obstacles"].clone()
                        } else {
                            JsonValue::Array(vec![])
                        };
                        obstacles.push(json::object! {
                            "id": obstacle_flag.as_ref().unwrap().to_string(),
                            "requires": []
                        })?;
                        strat["obstacles"] = obstacles;
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
            ensure!(new_room_json["links"].is_array());
            for link in new_room_json["links"].members_mut() {
                ensure!(link["to"].is_array());
                for to_json in link["to"].members_mut() {
                    let mut new_strats: Vec<JsonValue> = Vec::new();
                    ensure!(to_json["strats"].is_array());
                    // For each strat requiring one of the "obstacle flags" listed above, create an alternative strat
                    // depending on the corresponding obstacle instead:
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
        Ok(new_room_json)
    }

    pub fn get_obstacle_data(
        &self,
        strat_json: &JsonValue,
        room_json: &JsonValue,
        from_obstacles_bitmask: ObstacleMask,
        obstacles_idx_map: &HashMap<String, usize>,
        requires_json: &mut Vec<JsonValue>,
    ) -> Result<ObstacleMask> {
        let mut to_obstacles_bitmask = from_obstacles_bitmask;
        if strat_json.has_key("obstacles") {
            ensure!(strat_json["obstacles"].is_array());
            for obstacle in strat_json["obstacles"].members() {
                let obstacle_idx = obstacles_idx_map[obstacle["id"].as_str().unwrap()];
                to_obstacles_bitmask |= 1 << obstacle_idx;
                if (1 << obstacle_idx) & from_obstacles_bitmask == 0 {
                    ensure!(obstacle["requires"].is_array());
                    requires_json.extend(obstacle["requires"].members().map(|x| x.clone()));
                    let room_obstacle = &room_json["obstacles"][obstacle_idx];
                    if room_obstacle.has_key("requires") {
                        ensure!(room_obstacle["requires"].is_array());
                        requires_json
                            .extend(room_obstacle["requires"].members().map(|x| x.clone()));
                    }
                    if obstacle.has_key("additionalObstacles") {
                        ensure!(obstacle["additionalObstacles"].is_array());
                        for additional_obstacle_id in obstacle["additionalObstacles"].members() {
                            let additional_obstacle_idx =
                                obstacles_idx_map[additional_obstacle_id.as_str().unwrap()];
                            to_obstacles_bitmask |= 1 << additional_obstacle_idx;
                        }
                    }
                }
            }
        }
        Ok(to_obstacles_bitmask)
    }

    pub fn parse_note(&self, note: &JsonValue) -> Vec<String> {
        if note.is_string() {
            vec![note.as_str().unwrap().to_string()]
        } else if note.is_array() {
            note.members()
                .map(|x| x.as_str().unwrap().to_string())
                .collect()
        } else {
            vec![]
        }
    }

    fn get_node_physics(&self, node_json: &JsonValue) -> Result<String> {
        // TODO: handle case with multiple environments
        ensure!(node_json["doorEnvironments"].is_array());
        ensure!(node_json["doorEnvironments"].len() == 1);
        return Ok(node_json["doorEnvironments"][0]["physics"]
            .as_str()
            .unwrap()
            .to_string());
    }

    fn get_room_heated(&self, room_json: &JsonValue, node_id: NodeId) -> Result<bool> {
        ensure!(room_json["roomEnvironments"].is_array());
        for env in room_json["roomEnvironments"].members() {
            if env.has_key("entranceNodes") {
                ensure!(env["entranceNodes"].is_array());
                if !env["entranceNodes"].members().any(|x| x == node_id) {
                    continue;
                }
            }
            return Ok(env["heated"]
                .as_bool()
                .context("Expecting 'heated' to be a bool")?);
        }
        bail!("No match for node {} in roomEnvironments", node_id);
    }

    // fn get_origin_node(&self, requirement_json: &JsonValue) -> Option<NodeId> {

    // }

    fn process_room(&mut self, room_json: &JsonValue) -> Result<()> {
        let room_id = room_json["id"].as_usize().unwrap();
        self.room_json_map.insert(room_id, room_json.clone());

        let mut room_ptr =
            parse_int::parse::<usize>(room_json["roomAddress"].as_str().unwrap()).unwrap();
        self.raw_room_id_by_ptr.insert(room_ptr, room_id);
        if room_ptr == 0x7D408 {
            room_ptr = 0x7D5A7; // Treat Toilet Bowl as part of Aqueduct
        } else if room_ptr == 0x7D69A {
            room_ptr = 0x7D646; // Treat East Pants Room as part of Pants Room
        } else if room_ptr == 0x7968F {
            room_ptr = 0x793FE; // Treat Homing Geemer Room as part of West Ocean
        } else {
            self.room_id_by_ptr.insert(room_ptr, room_id);
        }
        self.room_ptr_by_id.insert(room_id, room_ptr);

        // Process obstacles:
        let obstacles_idx_map: HashMap<String, usize> = if room_json.has_key("obstacles") {
            ensure!(room_json["obstacles"].is_array());
            room_json["obstacles"]
                .members()
                .enumerate()
                .map(|(i, x)| (x["id"].as_str().unwrap().to_string(), i))
                .collect()
        } else {
            HashMap::new()
        };
        let num_obstacles = obstacles_idx_map.len();
        self.room_num_obstacles.insert(room_id, num_obstacles);
        self.room_obstacle_idx_map
            .insert(room_id, obstacles_idx_map.clone());

        // Process nodes:
        ensure!(room_json["nodes"].is_array());
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
            if node_json.has_key("runways") {
                ensure!(node_json["runways"].is_array());
                let mut runway_vec: Vec<Runway> = vec![];
                for runway_json in node_json["runways"].members() {
                    ensure!(runway_json["strats"].is_array());
                    for strat_json in runway_json["strats"].members() {
                        ensure!(strat_json["requires"].is_array());
                        let requires_json: Vec<JsonValue> = strat_json["requires"]
                            .members()
                            .map(|x| x.clone())
                            .collect();
                        let ctx = RequirementContext::default();
                        let requirement =
                            Requirement::make_and(self.parse_requires_list(&requires_json, &ctx)?);
                        if strat_json.has_key("obstacles") {
                            // TODO: handle obstacles in runways
                            continue;
                        }
                        let heated = self.get_room_heated(room_json, node_id)?;
                        let physics_res = self.get_node_physics(node_json);
                        if let Ok(physics) = physics_res {
                            let runway = Runway {
                                name: runway_json["name"].as_str().unwrap().to_string(),
                                length: runway_json["length"].as_i32().unwrap(),
                                open_end: runway_json["openEnd"].as_i32().unwrap(),
                                requirement,
                                physics,
                                heated,
                                usable_coming_in: runway_json["usableComingIn"]
                                    .as_bool()
                                    .unwrap_or(true),
                            };
                            runway_vec.push(runway);
                        }
                    }
                }
                self.node_runways_map.insert((room_id, node_id), runway_vec);
            } else {
                self.node_runways_map.insert((room_id, node_id), vec![]);
            }

            if node_json.has_key("canLeaveCharged") {
                ensure!(node_json["canLeaveCharged"].is_array());
                let mut can_leave_charged_vec: Vec<CanLeaveCharged> = vec![];
                for can_leave_charged_json in node_json["canLeaveCharged"].members() {
                    if can_leave_charged_json.has_key("initiateRemotely") {
                        // TODO: handle case with initiateRemotely
                        continue;
                    }
                    ensure!(can_leave_charged_json["strats"].is_array());
                    for strat_json in can_leave_charged_json["strats"].members() {
                        ensure!(strat_json["requires"].is_array());
                        let requires_json: Vec<JsonValue> = strat_json["requires"]
                            .members()
                            .map(|x| x.clone())
                            .collect();
                        let ctx = RequirementContext::default();
                        let requirement =
                            Requirement::make_and(self.parse_requires_list(&requires_json, &ctx)?);
                        if strat_json.has_key("obstacles") {
                            // TODO: handle obstacles
                            continue;
                        }
                        let can_leave_charged = CanLeaveCharged {
                            used_tiles: can_leave_charged_json["usedTiles"]
                                .as_i32()
                                .context("Expecting integer usedTiles")?,
                            frames_remaining: can_leave_charged_json["framesRemaining"]
                                .as_i32()
                                .context("Expecting integer framesRemaining")?,
                            shinespark_frames: can_leave_charged_json["shinesparkFrames"].as_i32(),
                            requirement,
                        };
                        can_leave_charged_vec.push(can_leave_charged);
                    }
                }
                self.node_can_leave_charged_map
                    .insert((room_id, node_id), can_leave_charged_vec);
            } else {
                self.node_can_leave_charged_map
                    .insert((room_id, node_id), vec![]);
            }

            if node_json.has_key("leaveWithGModeSetup") {
                ensure!(node_json["leaveWithGModeSetup"].is_array());
                let mut leave_with_gmode_setup_vec: Vec<LeaveWithGModeSetup> = vec![];
                for leave_with_gmode_setup_json in node_json["leaveWithGModeSetup"].members() {
                    ensure!(leave_with_gmode_setup_json["strats"].is_array());
                    for strat_json in leave_with_gmode_setup_json["strats"].members() {
                        ensure!(strat_json["requires"].is_array());
                        let requires_json: Vec<JsonValue> = strat_json["requires"]
                            .members()
                            .map(|x| x.clone())
                            .collect();
                        let mut ctx = RequirementContext::default();
                        ctx.room_id = room_id;
                        let requirement =
                            Requirement::make_and(self.parse_requires_list(&requires_json, &ctx)?);
                        let leave_with_gmode_setup = LeaveWithGModeSetup {
                            requirement,
                        };
                        leave_with_gmode_setup_vec.push(leave_with_gmode_setup);
                    }
                }
                self.node_leave_with_gmode_setup_map
                    .insert((room_id, node_id), leave_with_gmode_setup_vec);
            } else {
                self.node_leave_with_gmode_setup_map
                    .insert((room_id, node_id), vec![]);
            }

            // Explicit leaveWithGMode:
            if node_json.has_key("leaveWithGMode") {
                ensure!(node_json["leaveWithGMode"].is_array());
                let mut leave_with_gmode_vec: Vec<LeaveWithGMode> = vec![];
                for leave_with_gmode_json in node_json["leaveWithGMode"].members() {
                    ensure!(leave_with_gmode_json["strats"].is_array());
                    for strat_json in leave_with_gmode_json["strats"].members() {
                        ensure!(strat_json["requires"].is_array());
                        let requires_json: Vec<JsonValue> = strat_json["requires"]
                            .members()
                            .map(|x| x.clone())
                            .collect();
                        let mut ctx = RequirementContext::default();
                        ctx.room_id = room_id;
                        let requirement =
                            Requirement::make_and(self.parse_requires_list(&requires_json, &ctx)?);
                        let leave_with_gmode = LeaveWithGMode {
                            artificial_morph: leave_with_gmode_json["leavesWithArtificialMorph"]
                                .as_bool()
                                .context("Expecting field leavesWithArtificialMorph")?,                            
                            requirement,
                        };
                        leave_with_gmode_vec.push(leave_with_gmode);
                    }
                }
                self.node_leave_with_gmode_map
                    .insert((room_id, node_id), leave_with_gmode_vec);
            } else {
                self.node_leave_with_gmode_map
                    .insert((room_id, node_id), vec![]);
            }

            // Implicit leaveWithGMode:
            if !node_json.has_key("spawnAt") && node_json["nodeType"].as_str().unwrap() == "door" {
                for artificial_morph in [false, true] {
                    self.node_leave_with_gmode_map.get_mut(&(room_id, node_id)).unwrap().push(LeaveWithGMode {
                        artificial_morph,
                        requirement: Requirement::ComeInWithGMode {
                            room_id,
                            node_ids: vec![node_id],
                            mode: "direct".to_string(),
                            artificial_morph
                        }
                    });    
                }
            }

            if node_json.has_key("spawnAt") {
                let spawn_node_id = node_json["spawnAt"].as_usize().unwrap();
                self.node_spawn_at_map
                    .insert((room_id, node_id), spawn_node_id);
            }
            if node_json.has_key("utility") {
                let utility = &node_json["utility"];
                for obstacle_bitmask in 0..(1 << num_obstacles) {
                    let vertex_id =
                        self.vertex_isv.index_by_key[&(room_id, node_id, obstacle_bitmask)];
                    let mut reqs: Vec<Requirement> = Vec::new();
                    ensure!(utility.is_array());
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
                        requirement: Requirement::make_and(reqs),
                        notable_strat_name: None,
                        strat_name: "Refill".to_string(),
                        strat_notes: vec![],
                    });
                }
            }
        }
        if room_json.has_key("enemies") {
            ensure!(room_json["enemies"].is_array());
            for enemy in room_json["enemies"].members() {
                // TODO: implement other types of enemy farms, aside from those with farmCycles
                // (using a requirement to reset the room).
                if !enemy.has_key("farmCycles") {
                    continue;
                }
                if !enemy["homeNodes"].is_array() {
                    continue;
                }
                let enemy_name = enemy["enemyName"].as_str().unwrap();
                let enemy_json = self.enemy_json.get(enemy_name).with_context(|| format!("Unknown enemy: {}", enemy_name))?;

                let drops = &enemy_json["drops"];
                let drops_pb = drops["powerBomb"].as_i32().map(|x| x > 0) == Some(true);
                let drops_super = drops["super"].as_i32().map(|x| x > 0) == Some(true);
                let drops_missile = drops["missile"].as_i32().map(|x| x > 0) == Some(true);
                let drops_big_energy = drops["bigEnergy"].as_i32().map(|x| x > 0) == Some(true);
                let drops_small_energy = drops["smallEnergy"].as_i32().map(|x| x > 0) == Some(true);
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
                    reqs.push(Requirement::ReserveRefill);
                }
                let farm_name = format!("Farm {}", enemy["enemyName"].as_str().unwrap());
                for node_id_json in enemy["homeNodes"].members() {
                    let node_id = node_id_json.as_usize().unwrap();
                    for obstacle_bitmask in 0..(1 << num_obstacles) {
                        let vertex_id =
                            self.vertex_isv.index_by_key[&(room_id, node_id, obstacle_bitmask)];
                        self.links.push(Link {
                            from_vertex_id: vertex_id,
                            to_vertex_id: vertex_id,
                            requirement: Requirement::make_and(reqs.clone()),
                            notable_strat_name: None,
                            strat_name: farm_name.to_string(),
                            strat_notes: vec![],
                        });
                    }
                }
            }
        }

        // Process roomwide reusable strats:
        let mut roomwide_notable: HashMap<String, JsonValue> = HashMap::new();
        for strat in room_json["reusableRoomwideNotable"].members() {
            roomwide_notable.insert(strat["name"].as_str().unwrap().to_string(), strat.clone());
        }

        // Process links:
        ensure!(room_json["links"].is_array());
        for link_json in room_json["links"].members() {
            ensure!(link_json["to"].is_array());
            for link_to_json in link_json["to"].members() {
                ensure!(link_to_json["strats"].is_array());
                for strat_json in link_to_json["strats"].members() {
                    for from_obstacles_bitmask in 0..(1 << num_obstacles) {
                        let from_node_id = link_json["from"].as_usize().unwrap();
                        let to_node_id = link_to_json["id"].as_usize().unwrap();
                        ensure!(strat_json["requires"].is_array());
                        let mut requires_json: Vec<JsonValue> = strat_json["requires"]
                            .members()
                            .map(|x| x.clone())
                            .collect();

                        let to_obstacles_bitmask = self.get_obstacle_data(
                            strat_json,
                            room_json,
                            from_obstacles_bitmask,
                            &obstacles_idx_map,
                            &mut requires_json,
                        )?;
                        let ctx = RequirementContext {
                            room_id,
                            _from_node_id: from_node_id,
                            from_obstacles_bitmask,
                            obstacles_idx_map: Some(&obstacles_idx_map),
                        };
                        let mut requires_vec = self.parse_requires_list(&requires_json, &ctx)?;
                        let strat_name = strat_json["name"].as_str().unwrap().to_string();
                        let strat_notes = self.parse_note(&strat_json["note"]);
                        let notable = strat_json["notable"].as_bool().unwrap_or(false);
                        let mut notable_strat_name = strat_name.clone();
                        if notable {
                            let mut notable_strat_note: Vec<String> = strat_notes.clone();
                            if strat_json.has_key("reusableRoomwideNotable") {
                                notable_strat_name = strat_json["reusableRoomwideNotable"]
                                    .as_str()
                                    .unwrap()
                                    .to_string();
                                if !roomwide_notable.contains_key(&notable_strat_name) {
                                    bail!(
                                        "Unrecognized reusable notable strat name: {}",
                                        notable_strat_name
                                    );
                                }
                                notable_strat_note =
                                    self.parse_note(&roomwide_notable[&notable_strat_name]["note"]);
                            }
                            let strat_id = self.notable_strat_isv.add(&notable_strat_name);
                            requires_vec.push(Requirement::Strat(strat_id));
                            let area = format!(
                                "{} - {}",
                                room_json["area"].as_str().unwrap(),
                                room_json["subarea"].as_str().unwrap()
                            );
                            self.strat_area.insert(notable_strat_name.clone(), area);
                            self.strat_room.insert(
                                notable_strat_name.clone(),
                                room_json["name"].as_str().unwrap().to_string(),
                            );
                            self.strat_description
                                .insert(notable_strat_name.clone(), notable_strat_note.join(" "));
                        }
                        let requirement = Requirement::make_and(requires_vec);
                        let from_vertex_id = self.vertex_isv.index_by_key
                            [&(room_id, from_node_id, from_obstacles_bitmask)];
                        let to_vertex_id = self.vertex_isv.index_by_key
                            [&(room_id, to_node_id, to_obstacles_bitmask)];
                        let link = Link {
                            from_vertex_id,
                            to_vertex_id,
                            requirement,
                            notable_strat_name: if notable {
                                Some(notable_strat_name)
                            } else {
                                None
                            },
                            strat_name,
                            strat_notes,
                        };
                        self.links.push(link);
                    }
                }
            }
        }
        Ok(())
    }

    fn load_connections(&mut self) -> Result<()> {
        let connection_pattern =
            self.sm_json_data_path.to_str().unwrap().to_string() + "/connection/**/*.json";
        for entry in glob::glob(&connection_pattern)? {
            if let Ok(path) = entry {
                if !path.to_str().unwrap().contains("ceres") {
                    self.process_connections(&read_json(&path)?)?;
                }
            } else {
                bail!("Error processing connection path: {}", entry.err().unwrap());
            }
        }
        Ok(())
    }

    fn process_connections(&mut self, connection_file_json: &JsonValue) -> Result<()> {
        ensure!(connection_file_json["connections"].is_array());
        for connection in connection_file_json["connections"].members() {
            ensure!(connection["nodes"].is_array());
            ensure!(connection["nodes"].len() == 2);
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
        Ok(())
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

    fn populate_target_locations(&mut self) -> Result<()> {
        // Flags that are relevant to track in the randomizer:
        let flag_set: HashSet<String> = [
            "f_ZebesAwake",
            "f_MaridiaTubeBroken",
            "f_ShaktoolDoneDigging",
            "f_UsedAcidChozoStatue",
            "f_DefeatedBotwoon",
            "f_DefeatedCrocomire",
            "f_DefeatedSporeSpawn",
            "f_DefeatedGoldenTorizo",
            "f_DefeatedKraid",
            "f_DefeatedPhantoon",
            "f_DefeatedDraygon",
            "f_DefeatedRidley",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();

        for (&(room_id, node_id), node_json) in &self.node_json_map {
            if node_json["nodeType"] == "item" {
                self.item_locations.push((room_id, node_id));
            }
            if node_json.has_key("yields") {
                ensure!(node_json["yields"].len() >= 1);
                let flag_id = self.flag_isv.index_by_key[node_json["yields"][0].as_str().unwrap()];
                if flag_set.contains(&self.flag_isv.keys[flag_id]) {
                    self.flag_locations.push((room_id, node_id, flag_id));
                }
            }
        }

        for &(room_id, node_id) in &self.item_locations {
            let num_obstacles = self.room_num_obstacles[&room_id];
            let mut vertex_ids: Vec<VertexId> = Vec::new();
            for obstacle_bitmask in 0..(1 << num_obstacles) {
                let vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, obstacle_bitmask)];
                vertex_ids.push(vertex_id);
            }
            self.item_vertex_ids.push(vertex_ids);
        }

        for &(room_id, node_id, _flag_id) in &self.flag_locations {
            let num_obstacles = self.room_num_obstacles[&room_id];
            let mut vertex_ids: Vec<VertexId> = Vec::new();
            for obstacle_bitmask in 0..(1 << num_obstacles) {
                let vertex_id = self.vertex_isv.index_by_key[&(room_id, node_id, obstacle_bitmask)];
                vertex_ids.push(vertex_id);
            }
            self.flag_vertex_ids.push(vertex_ids);
        }
        Ok(())
    }

    pub fn get_weapon_mask(&self, items: &[bool]) -> WeaponMask {
        let mut weapon_mask = 0;
        // TODO: possibly make this more efficient. We could avoid dealing with strings
        // and just use a pre-computed item bitmask per weapon. But not sure yet if it matters.
        'weapon: for (i, weapon_name) in self.weapon_isv.keys.iter().enumerate() {
            let weapon = &self.weapon_json_map[weapon_name];
            assert!(weapon["useRequires"].is_array());
            for item_name_json in weapon["useRequires"].members() {
                let item_name = item_name_json.as_str().unwrap();
                if item_name == "PowerBeam" {
                    continue;
                }
                let item_idx = self.item_isv.index_by_key[item_name];
                if !items[item_idx] {
                    continue 'weapon;
                }
            }
            weapon_mask |= 1 << i;
        }
        weapon_mask
    }

    fn load_room_geometry(&mut self, room_geometry_path: &Path) -> Result<()> {
        let room_geometry_str = std::fs::read_to_string(room_geometry_path)?;
        self.room_geometry = serde_json::from_str(&room_geometry_str)?;
        for (room_idx, room) in self.room_geometry.iter().enumerate() {
            self.room_idx_by_name.insert(room.name.clone(), room_idx);
            self.room_idx_by_ptr.insert(room.rom_address, room_idx);
            if let Some(twin_rom_address) = room.twin_rom_address {
                self.room_idx_by_ptr.insert(twin_rom_address, room_idx);
            }
            for (door_idx, door) in room.doors.iter().enumerate() {
                let door_ptr_pair = (door.exit_ptr, door.entrance_ptr);
                self.room_and_door_idxs_by_door_ptr_pair
                    .insert(door_ptr_pair, (room_idx, door_idx));
            }

            let room_id = self.room_id_by_ptr[&room.rom_address];
            for (node_id, tiles) in &room.node_tiles {
                self.node_tile_coords
                    .insert((room_id, *node_id), tiles.clone());
            }

            if let Some(twin_rom_address) = room.twin_rom_address {
                let room_id = self.raw_room_id_by_ptr[&twin_rom_address];
                for (node_id, tiles) in room.twin_node_tiles.as_ref().unwrap() {
                    self.node_tile_coords
                        .insert((room_id, *node_id), tiles.clone());
                }
            }
        }
        Ok(())
    }

    fn load_palette(&mut self, json_path: &Path) -> Result<()> {
        let file = File::open(json_path)?;
        let json_value: serde_json::Value = serde_json::from_reader(file)?;
        for area_json in json_value.as_array().unwrap() {
            let mut pal_map: HashMap<TilesetIdx, [[u8; 3]; 128]> = HashMap::new();
            for (tileset_idx_str, palette) in area_json.as_object().unwrap().iter() {
                let tileset_idx: usize = tileset_idx_str.parse()?;
                let mut pal = [[0u8; 3]; 128];
                for (i, color) in palette.as_array().unwrap().iter().enumerate() {
                    let color_arr = color.as_array().unwrap();
                    let r = color_arr[0].as_i64().unwrap();
                    let g = color_arr[1].as_i64().unwrap();
                    let b = color_arr[2].as_i64().unwrap();
                    pal[i][0] = r as u8;
                    pal[i][1] = g as u8;
                    pal[i][2] = b as u8;
                }

                // for i in 0..128 {
                //     for j in 0..3 {
                //         pal[i][j] = 0;
                //     }
                // }
                pal_map.insert(tileset_idx, pal);
            }
            self.palette_data.push(pal_map);
        }
        Ok(())
    }

    fn extract_all_tech_dependencies(&mut self) -> Result<()> {
        let tech_vec = self.tech_isv.keys.clone();
        for tech in &tech_vec {
            let req = self.get_tech_requirement(tech)?;
            let deps: Vec<String> = self
                .extract_tech_dependencies(&req)
                .into_iter()
                .filter(|x| x != tech)
                .collect();
            self.tech_dependencies.insert(tech.clone(), deps);
        }
        Ok(())
    }

    fn extract_all_strat_dependencies(&mut self) -> Result<()> {
        let links = self.links.clone();
        for link in &links {
            if let Some(notable_strat_name) = link.notable_strat_name.clone() {
                let deps: HashSet<String> = self.extract_tech_dependencies(&link.requirement);
                self.strat_dependencies
                    .insert(notable_strat_name.clone(), deps.into_iter().collect());
            }
        }
        Ok(())
    }

    pub fn load(
        sm_json_data_path: &Path,
        room_geometry_path: &Path,
        palette_path: &Path,
    ) -> Result<GameData> {
        let mut game_data = GameData::default();
        game_data.sm_json_data_path = sm_json_data_path.to_owned();

        game_data.load_items_and_flags()?;
        game_data.load_tech()?;
        game_data.load_helpers()?;

        // Patch the h_heatProof and h_heatResistant to take into account the complementary suit
        // patch, where only Varia (and not Gravity) provides heat protection:
        *game_data.helper_json_map.get_mut("h_heatProof").unwrap() = json::object! {
            "name": "h_heatProof",
            "requires": ["Varia"],
        };
        *game_data
            .helper_json_map
            .get_mut("h_heatResistant")
            .unwrap() = json::object! {
            "name": "h_heatResistant",
            "requires": ["Varia"],
        };

        game_data.load_weapons()?;
        game_data.load_enemies()?;
        game_data.load_regions()?;
        game_data.load_connections()?;
        game_data.populate_target_locations()?;
        game_data.extract_all_tech_dependencies()?;
        game_data.extract_all_strat_dependencies()?;

        game_data
            .load_room_geometry(room_geometry_path)
            .context("Unable to load room geometry")?;
        game_data.base_room_door_graph = get_base_room_door_graph(&game_data);
        game_data.area_names = vec![
            "Crateria",
            "Brinstar",
            "Norfair",
            "Wrecked Ship",
            "Maridia",
            "Tourian",
        ]
        .into_iter()
        .map(|x| x.to_owned())
        .collect();
        game_data.area_map_ptrs = vec![
            0x1A9000, // Crateria
            0x1A8000, // Brinstar
            0x1AA000, // Norfair
            0x1AB000, // Wrecked ship
            0x1AC000, // Maridia
            0x1AD000, // Tourian
        ];
        game_data.load_palette(palette_path)?;

        Ok(game_data)
    }
}
