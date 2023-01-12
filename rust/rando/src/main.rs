mod sm_json_data;

use std::path::Path;

fn main() {
    let sm_json_data_path = Path::new("../../sm-json-data");
    let sm_json_data = sm_json_data::load_sm_json_data(sm_json_data_path);
}
