use super::SeedData;
use crate::web::{AppData, VersionInfo};
use actix_web::HttpRequest;
use anyhow::{Result, bail};
use askama::Template;
use hashbrown::{HashMap, HashSet};
use maprando::{
    helpers::get_item_priorities,
    preset::PresetData,
    randomize::{DifficultyConfig, ItemPriorityGroup, Randomization, get_starting_items},
    seed_repository::{Seed, SeedFile},
    settings::{
        AreaAssignmentBaseOrder, AreaAssignmentPreset, DisableETankSetting, DoorLocksSize,
        ETankRefill, FillerItemPriority, ItemCount, RandomizerSettings, SpeedBooster, WallJump,
        get_objective_groups,
    },
    spoiler_log::SpoilerLog,
    spoiler_map,
};
use maprando_game::{GameData, NotableId, RoomId, TechId};
use rand::{RngCore, SeedableRng};
use serde::Serialize;

#[derive(Template)]
#[template(path = "seed/seed_header.html")]
pub struct SeedHeaderTemplate<'a> {
    seed_name: String,
    timestamp: usize, // Milliseconds since UNIX epoch
    random_seed: usize,
    version_info: VersionInfo,
    settings: &'a RandomizerSettings,
    item_priority_groups: Vec<ItemPriorityGroup>,
    objective_names: HashMap<String, String>,
    race_mode: bool,
    preset: String,
    item_progression_preset: String,
    progression_rate: String,
    random_tank: bool,
    filler_items: Vec<String>,
    semi_filler_items: Vec<String>,
    early_filler_items: Vec<String>,
    item_placement_style: String,
    difficulty: &'a DifficultyConfig,
    quality_of_life_preset: String,
    supers_double: bool,
    mother_brain_fight: String,
    escape_enemies_cleared: bool,
    escape_refill: bool,
    escape_movement_items: bool,
    item_markers: String,
    all_items_spawn: bool,
    acid_chozo: bool,
    buffed_drops: bool,
    fast_elevators: bool,
    fast_doors: bool,
    fast_pause_menu: bool,
    respin: bool,
    infinite_space_jump: bool,
    momentum_conservation: bool,
    fanfares: String,
    etank_refill: String,
    disableable_etanks: String,
    doors: String,
    start_location_mode: String,
    map_layout: String,
    save_animals: String,
    early_save: bool,
    preset_data: &'a PresetData,
    enabled_tech: HashSet<TechId>,
    enabled_notables: HashSet<(RoomId, NotableId)>,
    starting_items: Vec<ItemCount>,
}

impl SeedHeaderTemplate<'_> {
    fn percent_enabled(&self, preset_name: &str) -> isize {
        let tech = &self.preset_data.tech_by_difficulty[preset_name];
        let tech_enabled_count = tech
            .iter()
            .filter(|&x| self.enabled_tech.contains(x))
            .count();

        let notables = &self.preset_data.notables_by_difficulty[preset_name];
        let notable_enabled_count = notables
            .iter()
            .filter(|&x| self.enabled_notables.contains(x))
            .count();
        let total_enabled_count = tech_enabled_count + notable_enabled_count;
        let total_count = tech.len() + notables.len();
        let frac_enabled = (total_enabled_count as f32) / (total_count as f32);
        let mut percent_enabled = (frac_enabled * 100.0) as isize;
        if percent_enabled == 0 && frac_enabled > 0.0 {
            percent_enabled = 1;
        }
        if percent_enabled == 100 && frac_enabled < 1.0 {
            percent_enabled = 99;
        }
        percent_enabled
    }

    fn item_pool_strs(&self) -> String {
        self.settings
            .item_progression_settings
            .item_pool
            .iter()
            .filter(|x| x.count > 0)
            .map(|x| {
                if x.count > 1 {
                    format!("{:?} ({})", x.item, x.count)
                } else {
                    format!("{:?}", x.item)
                }
            })
            .collect::<Vec<String>>()
            .join(", ")
    }

    fn starting_items_strs(&self) -> String {
        self.settings
            .item_progression_settings
            .starting_items
            .iter()
            .filter(|x| x.count > 0)
            .map(|x| {
                if x.count > 1 {
                    format!("{:?} ({})", x.item, x.count)
                } else {
                    format!("{:?}", x.item)
                }
            })
            .collect::<Vec<String>>()
            .join(", ")
    }

    fn game_variations(&self) -> Vec<&str> {
        let mut game_variations = vec![];
        let other_settings = &self.settings.other_settings;
        if other_settings.area_assignment.preset != Some(AreaAssignmentPreset::Standard) {
            match other_settings.area_assignment.base_order {
                AreaAssignmentBaseOrder::Size => {
                    game_variations.push("Size-based area assignment");
                }
                AreaAssignmentBaseOrder::Depth => {
                    game_variations.push("Depth-based area assignment");
                }
                AreaAssignmentBaseOrder::Random => {
                    game_variations.push("Random area assignment");
                }
            }
            if other_settings.area_assignment.ship_in_crateria {
                game_variations.push("Ship in Crateria");
            }
            if other_settings.area_assignment.mother_brain_in_tourian {
                game_variations.push("Mother Brain in Tourian");
            }
        }
        if other_settings.door_locks_size == DoorLocksSize::Small {
            game_variations.push("Door locks drawn smaller on map");
        }
        if other_settings.wall_jump == WallJump::Collectible {
            game_variations.push("Collectible wall jump");
        }
        if other_settings.map_station_reveal == maprando::settings::MapStationReveal::Partial {
            game_variations.push("Map stations give partial reveal");
        }

        if other_settings.energy_free_shinesparks {
            game_variations.push("Energy-free shinesparks");
        }
        if other_settings.ultra_low_qol {
            game_variations.push("Ultra-low quality of life");
        }
        if other_settings.speed_booster == SpeedBooster::Split {
            game_variations.push("Split Speed Booster");
        }
        if other_settings.disable_spikesuit {
            game_variations.push("Spikesuit disabled");
        }
        if other_settings.disable_bluesuit {
            game_variations.push("Blue suit disabled");
        }
        if other_settings.enable_major_glitches {
            game_variations.push("Major glitches enabled");
        }
        game_variations
    }
}

#[derive(Template)]
#[template(path = "seed/seed_footer.html")]
pub struct SeedFooterTemplate {
    race_mode: bool,
    all_items_spawn: bool,
    supers_double: bool,
    ultra_low_qol: bool,
    settings: RandomizerSettings,
}

pub fn get_random_seed() -> usize {
    (rand::rngs::StdRng::from_entropy().next_u64() & 0xFFFFFFFF) as usize + 1
}

pub async fn save_seed(
    seed_name: &str,
    seed_data: &SeedData,
    input_settings: &str,
    spoiler_token: &str,
    settings: &RandomizerSettings,
    randomization: &Randomization,
    spoiler_log: &SpoilerLog,
    app_data: &AppData,
) -> Result<()> {
    if check_seed_exists(seed_name, app_data).await {
        bail!("Seed name already exists: {}", seed_name);
    }

    let mut files: Vec<SeedFile> = Vec::new();

    // Write the seed data JSON. This contains details about the seed and request origin,
    // so to protect user privacy and the integrity of race ROMs we do not make it public.
    let seed_data_str = serde_json::to_vec_pretty(&seed_data).unwrap();
    files.push(SeedFile::new("seed_data.json", seed_data_str.to_vec()));

    files.push(SeedFile::new(
        "input_settings.json",
        input_settings.as_bytes().to_owned(),
    ));

    // Write the seed header HTML and footer HTML
    let (seed_header_html, seed_footer_html) = render_seed(seed_name, seed_data, app_data)?;
    files.push(SeedFile::new(
        "seed_header.html",
        seed_header_html.into_bytes(),
    ));
    files.push(SeedFile::new(
        "seed_footer.html",
        seed_footer_html.into_bytes(),
    ));

    let prefix = if seed_data.race_mode {
        "locked"
    } else {
        "public"
    };

    if seed_data.race_mode {
        files.push(SeedFile::new(
            "spoiler_token.txt",
            spoiler_token.as_bytes().to_vec(),
        ));
    }

    // Write the map data
    files.push(SeedFile::new(
        "map.json",
        serde_json::to_string(&randomization.map)?
            .as_bytes()
            .to_vec(),
    ));

    // Write the randomizer settings:
    let mut buf = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    settings.serialize(&mut ser).unwrap();
    files.push(SeedFile::new("public/settings.json", buf));

    // Write the Randomization struct:
    files.push(SeedFile::new(
        "randomization.json",
        serde_json::to_string(&randomization)?.as_bytes().to_vec(),
    ));

    // Write the spoiler log
    let spoiler_bytes = serde_json::to_vec(&spoiler_log).unwrap();
    files.push(SeedFile::new(
        &format!("{prefix}/spoiler.json"),
        spoiler_bytes,
    ));

    // Write the spoiler maps
    let mut door_settings = settings.clone();
    door_settings.other_settings.door_locks_size = DoorLocksSize::Large;
    let spoiler_maps =
        spoiler_map::get_spoiler_map(randomization, &app_data.game_data, &door_settings, false)
            .unwrap();
    door_settings.other_settings.door_locks_size = DoorLocksSize::Small;
    let spoiler_maps_small =
        spoiler_map::get_spoiler_map(randomization, &app_data.game_data, &door_settings, false)
            .unwrap();
    files.push(SeedFile::new(
        &format!("{prefix}/map-explored.png"),
        spoiler_maps.explored,
    ));
    files.push(SeedFile::new(
        &format!("{prefix}/map-explored-small.png"),
        spoiler_maps_small.explored,
    ));
    files.push(SeedFile::new(
        &format!("{prefix}/map-outline.png"),
        spoiler_maps.outline,
    ));

    // Write the spoiler visualizer
    for (filename, data) in &app_data.visualizer_files {
        let path = format!("{prefix}/visualizer/{filename}");
        files.push(SeedFile::new(&path, data.clone()));
    }

    let seed = Seed {
        name: seed_name.to_string(),
        files,
    };
    app_data.seed_repository.put_seed(seed).await?;
    Ok(())
}

pub fn format_http_headers(req: &HttpRequest) -> serde_json::Map<String, serde_json::Value> {
    let map: serde_json::Map<String, serde_json::Value> = req
        .headers()
        .into_iter()
        .map(|(name, value)| {
            (
                name.to_string(),
                serde_json::Value::String(value.to_str().unwrap_or("").to_string()),
            )
        })
        .collect();
    map
}

pub async fn check_seed_exists(seed_name: &str, app_data: &AppData) -> bool {
    app_data
        .seed_repository
        .get_file(seed_name, "seed_data.json")
        .await
        .is_ok()
}

fn get_enabled_tech(tech: &[bool], game_data: &GameData) -> HashSet<TechId> {
    let mut tech_set: HashSet<TechId> = HashSet::new();
    for (i, &tech_id) in game_data.tech_isv.keys.iter().enumerate() {
        if tech[i] {
            tech_set.insert(tech_id);
        }
    }
    tech_set
}

fn get_enabled_notables(notables: &[bool], game_data: &GameData) -> HashSet<(RoomId, NotableId)> {
    let mut notable_set: HashSet<(RoomId, NotableId)> = HashSet::new();
    for (i, &(room_id, notable_id)) in game_data.notable_isv.keys.iter().enumerate() {
        if notables[i] {
            notable_set.insert((room_id, notable_id));
        }
    }
    notable_set
}

pub fn render_seed(
    seed_name: &str,
    seed_data: &SeedData,
    app_data: &AppData,
) -> Result<(String, String)> {
    let enabled_tech: HashSet<TechId> =
        get_enabled_tech(&seed_data.difficulty.tech, &app_data.game_data);
    let enabled_notables: HashSet<(RoomId, NotableId)> =
        get_enabled_notables(&seed_data.difficulty.notables, &app_data.game_data);
    let objective_names: HashMap<String, String> = get_objective_groups()
        .iter()
        .flat_map(|x| x.objectives.clone())
        .collect();
    let seed_header_template = SeedHeaderTemplate {
        seed_name: seed_name.to_string(),
        version_info: app_data.version_info.clone(),
        random_seed: seed_data.random_seed,
        settings: &seed_data.settings,
        item_priority_groups: get_item_priorities(
            &seed_data
                .settings
                .item_progression_settings
                .key_item_priority,
        ),
        objective_names,
        race_mode: seed_data.race_mode,
        timestamp: seed_data.timestamp,
        preset: seed_data.preset.clone().unwrap_or("Custom".to_string()),
        item_progression_preset: seed_data
            .item_progression_preset
            .clone()
            .unwrap_or("Custom".to_string()),
        progression_rate: format!(
            "{:?}",
            seed_data
                .settings
                .item_progression_settings
                .progression_rate
        ),
        random_tank: seed_data.settings.item_progression_settings.random_tank,
        filler_items: seed_data
            .settings
            .item_progression_settings
            .filler_items
            .iter()
            .filter(|x| {
                x.priority == FillerItemPriority::Yes || x.priority == FillerItemPriority::Early
            })
            .map(|x| format!("{:?}", x.item))
            .collect(),
        semi_filler_items: seed_data
            .settings
            .item_progression_settings
            .filler_items
            .iter()
            .filter(|x| x.priority == FillerItemPriority::Semi)
            .map(|x| format!("{:?}", x.item))
            .collect(),
        early_filler_items: seed_data
            .settings
            .item_progression_settings
            .filler_items
            .iter()
            .filter(|x| x.priority == FillerItemPriority::Early)
            .map(|x| format!("{:?}", x.item))
            .collect(),
        item_placement_style: format!(
            "{:?}",
            seed_data
                .settings
                .item_progression_settings
                .item_placement_style
        ),
        difficulty: &seed_data.difficulty,
        quality_of_life_preset: seed_data
            .quality_of_life_preset
            .clone()
            .unwrap_or("Custom".to_string()),
        supers_double: seed_data.supers_double,
        mother_brain_fight: seed_data.mother_brain_fight.clone(),
        escape_enemies_cleared: seed_data.escape_enemies_cleared,
        escape_refill: seed_data.escape_refill,
        escape_movement_items: seed_data.escape_movement_items,
        item_markers: seed_data.item_markers.clone(),
        all_items_spawn: seed_data.all_items_spawn,
        acid_chozo: seed_data.acid_chozo,
        buffed_drops: seed_data.buffed_drops,
        fast_elevators: seed_data.fast_elevators,
        fast_doors: seed_data.fast_doors,
        fast_pause_menu: seed_data.fast_pause_menu,
        respin: seed_data.respin,
        infinite_space_jump: seed_data.infinite_space_jump,
        momentum_conservation: seed_data.momentum_conservation,
        fanfares: seed_data.fanfares.clone(),
        etank_refill: match seed_data.settings.quality_of_life_settings.etank_refill {
            ETankRefill::Disabled => "Disabled",
            ETankRefill::Vanilla => "Vanilla",
            ETankRefill::Full => "Full",
        }
        .to_string(),
        disableable_etanks: match seed_data
            .settings
            .quality_of_life_settings
            .disableable_etanks
        {
            DisableETankSetting::Off => "Off",
            DisableETankSetting::Standard => "Standard",
            DisableETankSetting::Unrestricted => "Unrestricted",
        }
        .to_string(),
        doors: seed_data.doors.clone(),
        start_location_mode: seed_data.start_location_mode.clone(),
        map_layout: seed_data.map_layout.clone(),
        save_animals: seed_data.save_animals.clone(),
        early_save: seed_data.early_save,
        preset_data: &app_data.preset_data,
        enabled_tech,
        enabled_notables,
        starting_items: get_starting_items(&seed_data.settings),
    };
    let seed_header_html = seed_header_template.render()?;

    let seed_footer_template = SeedFooterTemplate {
        race_mode: seed_data.race_mode,
        all_items_spawn: seed_data.all_items_spawn,
        supers_double: seed_data.supers_double,
        ultra_low_qol: seed_data.ultra_low_qol,
        settings: seed_data.settings.clone(),
    };
    let seed_footer_html = seed_footer_template.render()?;
    Ok((seed_header_html, seed_footer_html))
}
