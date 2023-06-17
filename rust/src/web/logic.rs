use glob::glob;
use hashbrown::HashMap;
use json::JsonValue;
use sailfish::TemplateOnce;

use crate::game_data::GameData;
use crate::web::VERSION;

#[derive(Clone)]
struct RoomStrat {
    name: String,
    notable: bool,
    from_node_id: usize,
    from_node_name: String,
    to_node_id: usize,
    to_node_name: String,
    note: String,
    requires_json: String,
}

#[derive(TemplateOnce, Clone)]
#[template(path = "logic/room.stpl")]
struct RoomTemplate {
    version: usize,
    room_name: String,
    room_diagram_path: String,
    nodes: Vec<(usize, String)>,
    strats: Vec<RoomStrat>,
}

#[derive(Default)]
pub struct LogicData {
    pub room_html: HashMap<String, String>, // Map from room name (with whitespace removed) to rendered HTML.
}

fn list_room_diagram_files() -> HashMap<usize, String> {
    let mut out: HashMap<usize, String> = HashMap::new();
    for entry in glob("static/sm-json-data/region/*/roomDiagrams/*.png").unwrap() {
        match entry {
            Ok(path) => {
                let path_string = path.to_str().unwrap().to_string();
                let segments: Vec<&str> = path_string.split("_").collect();
                let subregion = segments[0];
                if subregion == "ceres" {
                    continue;
                }
                let room_id: usize = str::parse(segments[1]).unwrap();
                // let img = image::open(path).unwrap();
                // println!("{:?}", img.dimensions());
                out.insert(room_id, path_string);
            }
            Err(e) => panic!("Failure reading room diagrams: {:?}", e),
        }
    }
    out
}

fn make_requires(requires_json: &JsonValue) -> String {
    let mut out: Vec<String> = vec![];
    for req in requires_json.members() {
        out.push(req.pretty(2));
    }
    out.join("\n")
}

fn make_room_template(
    room_json: &JsonValue,
    room_diagram_listing: &HashMap<usize, String>,
    game_data: &GameData,
) -> RoomTemplate {
    let mut strats: Vec<RoomStrat> = vec![];
    let room_id = room_json["id"].as_usize().unwrap();
    let mut node_name_map: HashMap<usize, String> = HashMap::new();
    let mut nodes: Vec<(usize, String)> = vec![];
    for node_json in room_json["nodes"].members() {
        let node_id = node_json["id"].as_usize().unwrap();
        let node_name = node_json["name"].as_str().unwrap();
        node_name_map.insert(node_id, node_name.to_string());
        nodes.push((node_id, node_name.to_string()));
    }
    for link_json in room_json["links"].members() {
        for link_to_json in link_json["to"].members() {
            for strat_json in link_to_json["strats"].members() {
                let from_node_id = link_json["from"].as_usize().unwrap();
                let to_node_id = link_to_json["id"].as_usize().unwrap();
                strats.push(RoomStrat {
                    name: strat_json["name"].as_str().unwrap().to_string(),
                    notable: strat_json["notable"].as_bool().unwrap_or(false),
                    from_node_id,
                    from_node_name: node_name_map[&from_node_id].clone(),
                    to_node_id,
                    to_node_name: node_name_map[&to_node_id].clone(),
                    note: game_data.parse_note(&strat_json["note"]).join(" "),
                    requires_json: make_requires(&strat_json["requires"]),
                });
            }
        }
    }
    // let shape = *game_data.room_shape.get(&room_id).unwrap_or(&(1, 1));
    RoomTemplate {
        version: VERSION,
        room_name: room_json["name"].as_str().unwrap().to_string(),
        room_diagram_path: room_diagram_listing[&room_id].clone(),
        nodes,
        strats,
    }
}

fn make_url_safe(s: &str) -> String {
    s.chars().filter(|x| x.is_ascii_alphanumeric()).collect()
}

impl LogicData {
    pub fn new(game_data: &GameData) -> LogicData {
        let mut out = LogicData::default();
        let room_diagram_listing = list_room_diagram_files();
        for (_, room_json) in game_data.room_json_map.iter() {
            let template = make_room_template(room_json, &room_diagram_listing, &game_data);
            let html = template.clone().render_once().unwrap();
            out.room_html.insert(make_url_safe(&template.room_name), html);
        }
        out
    }
}
