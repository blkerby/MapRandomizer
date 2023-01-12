mod sm_json_data;

use sm_json_data::SMJsonData;

use std::path::Path;

fn main() {
    let sm_json_data_path = Path::new("../../sm-json-data");
    let sm_json_data = SMJsonData::load(sm_json_data_path);
}
