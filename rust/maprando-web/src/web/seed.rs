mod customize_seed;
mod get_seed_file;
mod unlock_seed;
mod view_seed;

use super::AppData;
use maprando::randomize::DifficultyConfig;
use serde::{Deserialize, Serialize};

pub fn scope() -> actix_web::Scope {
    actix_web::web::scope("/seed")
        .service(view_seed::view_seed)
        .service(get_seed_file::get_seed_file)
        .service(customize_seed::customize_seed)
        .service(unlock_seed::unlock_request)
        .service(view_seed::view_seed_redirect)
}

#[derive(Serialize, Deserialize)]
pub struct SeedData {
    pub version: usize,
    pub timestamp: usize,
    pub peer_addr: String,
    pub http_headers: serde_json::Map<String, serde_json::Value>,
    pub random_seed: usize,
    pub map_seed: usize,
    pub door_randomization_seed: usize,
    pub item_placement_seed: usize,
    pub race_mode: bool,
    pub preset: Option<String>,
    pub item_progression_preset: Option<String>,
    pub difficulty: DifficultyConfig,
    pub quality_of_life_preset: Option<String>,
    pub supers_double: bool,
    pub mother_brain_fight: String,
    pub escape_enemies_cleared: bool,
    pub escape_refill: bool,
    pub escape_movement_items: bool,
    pub mark_map_stations: bool,
    pub transition_letters: bool,
    pub item_markers: String,
    pub item_dot_change: String,
    pub all_items_spawn: bool,
    pub acid_chozo: bool,
    pub buffed_drops: bool,
    pub fast_elevators: bool,
    pub fast_doors: bool,
    pub fast_pause_menu: bool,
    pub respin: bool,
    pub infinite_space_jump: bool,
    pub momentum_conservation: bool,
    pub objectives: String,
    pub doors: String,
    pub start_location_mode: String,
    pub map_layout: String,
    pub save_animals: String,
    pub early_save: bool,
    pub area_assignment: String,
    pub wall_jump: String,
    pub etank_refill: String,
    pub maps_revealed: String,
    pub vanilla_map: bool,
    pub ultra_low_qol: bool,
}

impl SeedData {
    /// Returns the [`SeedData`] of the specified seed from the seed repository.
    ///
    /// # Panics
    /// Panics if the specified seed doesn't exist.
    async fn from_repository(app_data: &AppData, seed_name: &str) -> SeedData {
        let bytes = app_data
            .seed_repository
            .get_file(seed_name, "seed_data.json")
            .await
            .unwrap();
        let json_string = String::from_utf8(bytes).unwrap();
        serde_json::from_str(&json_string).unwrap()
    }
}
