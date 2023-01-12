use hashbrown::HashMap;
use json::{self, JsonValue};
use std::fs::File;
use std::io::Read;
use std::path::Path;

type TechId = u32;
type ItemId = u32;
type FlagId = u32;
type Capacity = u32;

pub struct IndexedStringVec {
    names: Vec<String>,
    id_by_name: HashMap<String, usize>,
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
    EnergyRefill,
    MissileRefill,
    SupersRefill,
    PowerBombsRefill,
    Missiles(Capacity),
    Supers(Capacity),
    PowerBombs(Capacity),
    And(Vec<Requirement>),
    Or(Vec<Requirement>),
}

pub struct SMJsonData {
    tech_isv: IndexedStringVec,
    items_isv: IndexedStringVec,
    flags_isv: IndexedStringVec,
}

struct ParsingContext<'a> {
    tech_isv: &'a IndexedStringVec,
    items_isv: &'a IndexedStringVec,
    flags_isv: &'a IndexedStringVec,
    helpers_json_map: &'a HashMap<String, JsonValue>,
    helpers: &'a mut HashMap<String, Option<Requirement>>,
}

impl IndexedStringVec {
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            id_by_name: HashMap::new(),
        }
    }

    pub fn add(&mut self, name: &str) {
        self.id_by_name.insert(name.to_owned(), self.names.len());
        self.names.push(name.to_owned());
    }
}

fn read_json(path: &Path) -> JsonValue {
    let file = File::open(path).expect(&format!("unable to open {}", path.display()));
    let json_str =
        std::io::read_to_string(file).expect(&format!("unable to read {}", path.display()));
    let json_data = json::parse(&json_str).expect(&format!("unable to parse {}", path.display()));
    json_data
}

fn load_tech(sm_json_data_path: &Path) -> IndexedStringVec {
    let tech_json = read_json(&sm_json_data_path.join("tech.json"));
    let mut tech_isv = IndexedStringVec::new();
    for tech_category in tech_json["techCategories"].members() {
        for tech in tech_category["techs"].members() {
            tech_isv.add(tech["name"].as_str().unwrap());
        }
    }
    tech_isv
}

fn load_items_and_flags(sm_json_data_path: &Path) -> (IndexedStringVec, IndexedStringVec) {
    let item_json = read_json(&sm_json_data_path.join("items.json"));
    let mut item_isv = IndexedStringVec::new();
    for item_name in item_json["implicitItems"].members() {
        item_isv.add(item_name.as_str().unwrap());
    }
    for item in item_json["upgradeItems"].members() {
        item_isv.add(item["name"].as_str().unwrap());
    }
    for item in item_json["expansionItems"].members() {
        item_isv.add(item["name"].as_str().unwrap());
    }

    let mut flag_isv = IndexedStringVec::new();
    for flag_name in item_json["gameFlags"].members() {
        flag_isv.add(flag_name.as_str().unwrap());
    }
    (item_isv, flag_isv)
}

fn parse_helper(name: &str, json_value: &JsonValue, ctx: &mut ParsingContext) -> Requirement {
    ctx.helpers.insert(name.to_owned(), None);
    let req = Requirement::And(parse_requires_list(&json_value["requires"], ctx));
    *ctx.helpers.get_mut(name).unwrap() = Some(req.clone());
    req
}

fn parse_requires_list(json_value: &JsonValue, ctx: &mut ParsingContext) -> Vec<Requirement> {
    let mut req_list: Vec<Requirement> = Vec::new();
    for json_req in json_value.members() {
        req_list.push(parse_requirement(json_req, ctx));
    }
    req_list
}

fn parse_requirement(json_value: &JsonValue, ctx: &mut ParsingContext) -> Requirement {
    if json_value.is_string() {
        let value = json_value.as_str().unwrap();
        if let Some(&tech_id) = ctx.tech_isv.id_by_name.get(value) {
            return Requirement::Tech(tech_id as TechId);
        } else if let Some(&item_id) = ctx.items_isv.id_by_name.get(value) {
            return Requirement::Item(item_id as ItemId);
        } else if let Some(&flag_id) = ctx.flags_isv.id_by_name.get(value) {
            return Requirement::Flag(flag_id as FlagId);
        } else if let Some(req) = ctx.helpers.get(value) {
            if req.is_none() {
                panic!("Cyclic dependency in helper {}", value);
            }
            return req.clone().unwrap();
        } else if let Some(helper_j) = ctx.helpers_json_map.get(value) {
            return parse_helper(value, helper_j, ctx);
        }
    } else if json_value.is_object() && json_value.len() == 1 {
        let (key, value) = json_value.entries().next().unwrap();
        if key == "or" {
            return Requirement::Or(parse_requires_list(value, ctx));
        } else if key == "and" {
            return Requirement::And(parse_requires_list(value, ctx));
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
            let hits = value.as_usize().expect(&format!(
                "invalid spikeHits in {}",
                json_value
            ));
            return Requirement::Damage(hits * 60);
        } else if key == "thornHits" {
            let hits = value.as_usize().expect(&format!(
                "invalid thornHits in {}",
                json_value
            ));
            return Requirement::Damage(hits * 16);
        } else if key == "hibashiHits" {
            let hits = value.as_usize().expect(&format!(
                "invalid hibashiHits in {}",
                json_value
            ));
            return Requirement::Damage(hits * 30);
        } else if key == "previousNode" {
            // Currently this is used only in the Early Supers quick crumble and Mission Impossible strats and is
            // redundant in both cases, so we treat it as free.
            return Requirement::Free
        } else if key == "resetRoom" {
            // In all the places where this is required (excluding runways and canComeInCharged which we are not
            // yet taking into account), it seems to be essentially unnecessary (ignoring the
            // possibility of needing to take a small amount of heat damage in an adjacent room to exit and
            // reenter), so for now we treat it as free.
            return Requirement::Free
        } else if key == "previousStratProperty" {
            // This is only used in one place in Crumble Shaft, where it doesn't seem to be necessary.
            return Requirement::Free
        } else if key == "canComeInCharged" || key == "adjacentRunway" {
            // For now assume we can't do these.
            return Requirement::Never
        }
    }
    panic!("Unable to parse requirement: {}", json_value)
}

fn load_helpers(
    sm_json_data_path: &Path,
    tech_isv: &IndexedStringVec,
    items_isv: &IndexedStringVec,
    flags_isv: &IndexedStringVec,
) -> HashMap<String, Option<Requirement>> {
    let helpers_json = read_json(&sm_json_data_path.join("helpers.json"));
    let mut helpers_json_map: HashMap<String, JsonValue> = HashMap::new();
    for helper in helpers_json["helpers"].members() {
        helpers_json_map.insert(helper["name"].as_str().unwrap().to_owned(), helper.clone());
    }

    let mut helpers: HashMap<String, Option<Requirement>> = HashMap::new();
    let mut ctx = ParsingContext {
        tech_isv,
        items_isv,
        flags_isv,
        helpers_json_map: &helpers_json_map,
        helpers: &mut helpers,
    };
    for helper in helpers_json["helpers"].members() {
        let _ = parse_helper(helper["name"].as_str().unwrap(), helper, &mut ctx);
    }
    helpers
}

fn load_regions(sm_json_data_path: &Path, ctx: &ParsingContext) {
    let region_pattern = sm_json_data_path.to_str().unwrap().to_string() + "/region/**/*.json";
    for entry in glob::glob(&region_pattern).unwrap() {
        if let Ok(path) = entry {
            println!("{}", path.display());
        } else {
            panic!("Error processing region path: {}", entry.err().unwrap());
        }
    }
}

pub fn load_sm_json_data(sm_json_data_path: &Path) -> SMJsonData {
    let tech_isv = load_tech(sm_json_data_path);
    let (items_isv, flags_isv) = load_items_and_flags(sm_json_data_path);
    let helpers = load_helpers(sm_json_data_path, &tech_isv, &items_isv, &flags_isv);
    let helpers_json_map: HashMap<String, JsonValue> = HashMap::new();
    let mut ctx = ParsingContext {
        tech_isv: &tech_isv,
        items_isv: &items_isv,
        flags_isv: &flags_isv,
        helpers_json_map: &helpers_json_map,
        helpers: &mut helpers,
    };
    load_regions(sm_json_data_path, &ctx);
    // for tech in &tech.names {
    //     println!("{}", tech);
    // }
    // for item_name in &items_isv.names {
    //     println!("{}", item_name);
    // }
    // for flag_name in &flags.names {
    //     println!("{}", flag_name);
    // }
    for (name, reqs) in &helpers {
        println!("{}: {:?}", name, reqs);
    }

    SMJsonData {
        tech_isv,
        items_isv,
        flags_isv,
    }
}
