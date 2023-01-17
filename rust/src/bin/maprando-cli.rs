use maprando::game_data::GameData;
use std::path::Path;

fn main() {
    let sm_json_data_path = Path::new("../sm-json-data");
    let sm_json_data = GameData::load(sm_json_data_path);
}
