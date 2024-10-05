use super::SeedData;
use crate::web::{AppData, PresetData, VersionInfo};
use actix_web::HttpRequest;
use anyhow::{bail, Result};
use askama::Template;
use hashbrown::HashSet;
use maprando::{
    patch::{ips_write::create_ips_patch, Rom},
    randomize::{
        DifficultyConfig, DoorLocksSize, EtankRefill, ItemPriorityGroup, Randomization, WallJump,
    },
    seed_repository::{Seed, SeedFile},
    spoiler_map,
};
use maprando_game::{Capacity, IndexedVec, Item, NotableId, RoomId, TechId};
use rand::{RngCore, SeedableRng};

#[derive(Template)]
#[template(path = "seed/seed_header.html")]
pub struct SeedHeaderTemplate<'a> {
    seed_name: String,
    timestamp: usize, // Milliseconds since UNIX epoch
    random_seed: usize,
    version_info: VersionInfo,
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
    mark_map_stations: bool,
    transition_letters: bool,
    item_markers: String,
    item_dot_change: String,
    all_items_spawn: bool,
    acid_chozo: bool,
    buffed_drops: bool,
    fast_elevators: bool,
    fast_doors: bool,
    fast_pause_menu: bool,
    respin: bool,
    infinite_space_jump: bool,
    momentum_conservation: bool,
    objectives: String,
    doors: String,
    start_location_mode: String,
    map_layout: String,
    save_animals: String,
    early_save: bool,
    area_assignment: String,
    ultra_low_qol: bool,
    preset_data: &'a [PresetData],
    enabled_tech: HashSet<TechId>,
    enabled_notables: HashSet<(RoomId, NotableId)>,
}

impl<'a> SeedHeaderTemplate<'a> {
    fn percent_enabled(&self, p: &PresetData) -> isize {
        let tech_enabled_count = p
            .preset
            .tech
            .iter()
            .filter(|&x| self.enabled_tech.contains(&x.tech_id))
            .count();
        let notable_enabled_count = p
            .preset
            .notables
            .iter()
            .filter(|x| self.enabled_notables.contains(&(x.room_id, x.notable_id)))
            .count();
        let total_enabled_count = tech_enabled_count + notable_enabled_count;
        let total_count = p.preset.tech.len() + p.preset.notables.len();
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
        self.difficulty
            .item_pool
            .iter()
            .map(|(x, cnt)| {
                if *cnt > 1 {
                    format!("{:?} ({})", x, cnt)
                } else {
                    format!("{:?}", x)
                }
            })
            .collect::<Vec<String>>()
            .join(", ")
    }

    fn starting_items_strs(&self) -> String {
        self.difficulty
            .starting_items
            .iter()
            .map(|(x, cnt)| {
                if *cnt > 1 {
                    format!("{:?} ({})", x, cnt)
                } else {
                    format!("{:?}", x)
                }
            })
            .collect::<Vec<String>>()
            .join(", ")
    }

    fn game_variations(&self) -> Vec<&str> {
        let mut game_variations = vec![];
        if self.area_assignment == "Random" {
            game_variations.push("Random area assignment");
        }
        if self.item_dot_change == "Disappear" {
            game_variations.push("Item dots disappear after collection");
        }
        if !self.transition_letters {
            game_variations.push("Area transitions marked as arrows");
        }
        if self.difficulty.door_locks_size == DoorLocksSize::Small {
            game_variations.push("Door locks drawn smaller on map");
        }
        match self.difficulty.wall_jump {
            WallJump::Collectible => {
                game_variations.push("Collectible wall jump");
            }
            _ => {}
        }
        match self.difficulty.etank_refill {
            EtankRefill::Disabled => {
                game_variations.push("E-Tank refill disabled");
            }
            EtankRefill::Full => {
                game_variations.push("E-Tanks refill reserves");
            }
            _ => {}
        }
        if self.difficulty.maps_revealed == maprando::randomize::MapsRevealed::Partial {
            game_variations.push("Maps partially revealed from start");
        }
        if self.difficulty.maps_revealed == maprando::randomize::MapsRevealed::Full {
            game_variations.push("Maps revealed from start");
        }
        if self.difficulty.map_station_reveal == maprando::randomize::MapStationReveal::Partial {
            game_variations.push("Map stations give partial reveal");
        }

        if self.difficulty.energy_free_shinesparks {
            game_variations.push("Energy-free shinesparks");
        }
        if self.ultra_low_qol {
            game_variations.push("Ultra-low quality of life");
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
}

pub fn get_random_seed() -> usize {
    (rand::rngs::StdRng::from_entropy().next_u64() & 0xFFFFFFFF) as usize
}

pub fn get_item_priorities(item_priority_json: serde_json::Value) -> Vec<ItemPriorityGroup> {
    let mut priorities: IndexedVec<String> = IndexedVec::default();
    priorities.add("Early");
    priorities.add("Default");
    priorities.add("Late");

    let mut out: Vec<ItemPriorityGroup> = Vec::new();
    for priority in &priorities.keys {
        out.push(ItemPriorityGroup {
            name: priority.clone(),
            items: vec![],
        });
    }
    for (k, v) in item_priority_json.as_object().unwrap() {
        let i = priorities.index_by_key[v.as_str().unwrap()];
        out[i].items.push(k.to_string());
    }
    out
}

// Computes the intersection of the selected difficulty with each preset. This
// gives a set of difficulty tiers below the selected difficulty. These are
// used in "forced mode" to try to identify locations at which to place
// key items which are reachable using the selected difficulty but not at
// lower difficulties.
pub fn get_difficulty_tiers(
    difficulty: &DifficultyConfig,
    app_data: &AppData,
) -> Vec<DifficultyConfig> {
    let presets = &app_data.preset_data;
    let mut out: Vec<DifficultyConfig> = vec![];
    let tech_set: HashSet<TechId> = difficulty.tech.iter().cloned().collect();
    let notable_set: HashSet<(RoomId, NotableId)> = difficulty.notables.iter().cloned().collect();

    out.push(difficulty.clone());
    out.last_mut().unwrap().tech.sort();
    out.last_mut().unwrap().notables.sort();
    for preset_data in presets[1..presets.len() - 1].iter().rev() {
        let preset = &preset_data.preset;
        let mut tech_vec: Vec<TechId> = Vec::new();
        for (tech_setting, enabled) in &preset_data.tech_setting {
            if *enabled && tech_set.contains(&tech_setting.tech_id) {
                tech_vec.push(tech_setting.tech_id);
            }
        }
        tech_vec.sort();

        let mut notable_vec: Vec<(RoomId, NotableId)> = vec![];
        for (notable_setting, enabled) in &preset_data.notable_setting {
            let room_id = notable_setting.room_id;
            let notable_id = notable_setting.notable_id;
            if *enabled && notable_set.contains(&(room_id, notable_id)) {
                notable_vec.push((room_id, notable_id));
            }
        }
        notable_vec.sort();

        // TODO: move some fields out of here so we don't have clone as much irrelevant stuff:
        let new_difficulty = DifficultyConfig {
            name: Some(preset.name.clone()),
            tech: tech_vec,
            notables: notable_vec,
            shine_charge_tiles: f32::max(
                difficulty.shine_charge_tiles,
                preset.shinespark_tiles as f32,
            ),
            heated_shine_charge_tiles: f32::max(
                difficulty.heated_shine_charge_tiles,
                preset.heated_shinespark_tiles as f32,
            ),
            speed_ball_tiles: f32::max(difficulty.speed_ball_tiles, preset.speed_ball_tiles as f32),
            shinecharge_leniency_frames: Capacity::max(
                difficulty.shinecharge_leniency_frames,
                preset.shinecharge_leniency_frames as Capacity,
            ),
            progression_rate: difficulty.progression_rate,
            random_tank: difficulty.random_tank,
            spazer_before_plasma: difficulty.spazer_before_plasma,
            stop_item_placement_early: difficulty.stop_item_placement_early,
            item_placement_style: difficulty.item_placement_style,
            item_priority_strength: difficulty.item_priority_strength,
            item_priorities: difficulty.item_priorities.clone(),
            item_pool: difficulty.item_pool.clone(),
            starting_items: difficulty.starting_items.clone(),
            semi_filler_items: difficulty.semi_filler_items.clone(),
            filler_items: difficulty.filler_items.clone(),
            early_filler_items: difficulty.early_filler_items.clone(),
            resource_multiplier: f32::max(
                difficulty.resource_multiplier,
                preset.resource_multiplier,
            ),
            gate_glitch_leniency: Capacity::max(
                difficulty.gate_glitch_leniency,
                preset.gate_glitch_leniency as Capacity,
            ),
            door_stuck_leniency: Capacity::max(
                difficulty.door_stuck_leniency,
                preset.door_stuck_leniency as Capacity,
            ),
            escape_timer_multiplier: difficulty.escape_timer_multiplier,
            start_location_mode: difficulty.start_location_mode,
            save_animals: difficulty.save_animals,
            phantoon_proficiency: f32::min(
                difficulty.phantoon_proficiency,
                preset.phantoon_proficiency,
            ),
            draygon_proficiency: f32::min(
                difficulty.draygon_proficiency,
                preset.draygon_proficiency,
            ),
            ridley_proficiency: f32::min(difficulty.ridley_proficiency, preset.ridley_proficiency),
            botwoon_proficiency: f32::min(
                difficulty.botwoon_proficiency,
                preset.botwoon_proficiency,
            ),
            mother_brain_proficiency: f32::min(
                difficulty.mother_brain_proficiency,
                preset.mother_brain_proficiency,
            ),
            // Quality-of-life options:
            supers_double: difficulty.supers_double,
            mother_brain_fight: difficulty.mother_brain_fight,
            escape_enemies_cleared: difficulty.escape_enemies_cleared,
            escape_refill: difficulty.escape_refill,
            escape_movement_items: difficulty.escape_movement_items,
            mark_map_stations: difficulty.mark_map_stations,
            room_outline_revealed: difficulty.room_outline_revealed,
            transition_letters: difficulty.transition_letters,
            door_locks_size: difficulty.door_locks_size,
            item_markers: difficulty.item_markers,
            item_dot_change: difficulty.item_dot_change,
            all_items_spawn: difficulty.all_items_spawn,
            acid_chozo: difficulty.acid_chozo,
            buffed_drops: difficulty.buffed_drops,
            fast_elevators: difficulty.fast_elevators,
            fast_doors: difficulty.fast_doors,
            fast_pause_menu: difficulty.fast_pause_menu,
            respin: difficulty.respin,
            infinite_space_jump: difficulty.infinite_space_jump,
            momentum_conservation: difficulty.momentum_conservation,
            objectives: difficulty.objectives.clone(),
            doors_mode: difficulty.doors_mode,
            early_save: difficulty.early_save,
            area_assignment: difficulty.area_assignment,
            wall_jump: difficulty.wall_jump,
            etank_refill: difficulty.etank_refill,
            maps_revealed: difficulty.maps_revealed,
            map_station_reveal: difficulty.map_station_reveal,
            energy_free_shinesparks: difficulty.energy_free_shinesparks,
            vanilla_map: difficulty.vanilla_map,
            ultra_low_qol: difficulty.ultra_low_qol,
            skill_assumptions_preset: difficulty.skill_assumptions_preset.clone(),
            item_progression_preset: difficulty.item_progression_preset.clone(),
            quality_of_life_preset: difficulty.quality_of_life_preset.clone(),
            debug_options: difficulty.debug_options.clone(),
        };
        if !is_equivalent_difficulty(&new_difficulty, out.last().as_ref().unwrap()) {
            out.push(new_difficulty);
        }
    }
    out
}

// A simplified measure of "equivalence" in difficulty for the purposes of Forced mode logic, to reduce
// the amount of tiers the randomizer has to consider:
pub fn is_equivalent_difficulty(a: &DifficultyConfig, b: &DifficultyConfig) -> bool {
    return a.tech == b.tech && a.notables == b.notables;
}

pub async fn save_seed(
    seed_name: &str,
    seed_data: &SeedData,
    spoiler_token: &str,
    vanilla_rom: &Rom,
    output_rom: &Rom,
    randomization: &Randomization,
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

    // Write the ROM patch.
    let patch_ips = create_ips_patch(&vanilla_rom.data, &output_rom.data);
    files.push(SeedFile::new("patch.ips", patch_ips));

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

    // Write the spoiler log
    let spoiler_bytes = serde_json::to_vec_pretty(&randomization.spoiler_log).unwrap();
    files.push(SeedFile::new(
        &format!("{}/spoiler.json", prefix),
        spoiler_bytes,
    ));

    // Write the spoiler maps
    let spoiler_maps =
        spoiler_map::get_spoiler_map(&output_rom, &randomization.map, &app_data.game_data).unwrap();
    files.push(SeedFile::new(
        &format!("{}/map-assigned.png", prefix),
        spoiler_maps.assigned,
    ));
    files.push(SeedFile::new(
        &format!("{}/map-vanilla.png", prefix),
        spoiler_maps.vanilla,
    ));
    files.push(SeedFile::new(
        &format!("{}/map-grid.png", prefix),
        spoiler_maps.grid,
    ));

    // Write the spoiler visualizer
    for (filename, data) in &app_data.visualizer_files {
        let path = format!("{}/visualizer/{}", prefix, filename);
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

pub fn render_seed(
    seed_name: &str,
    seed_data: &SeedData,
    app_data: &AppData,
) -> Result<(String, String)> {
    let enabled_tech: HashSet<TechId> = seed_data.difficulty.tech.iter().cloned().collect();
    let enabled_notables: HashSet<(RoomId, NotableId)> =
        seed_data.difficulty.notables.iter().cloned().collect();
    let seed_header_template = SeedHeaderTemplate {
        seed_name: seed_name.to_string(),
        version_info: app_data.version_info.clone(),
        random_seed: seed_data.random_seed,
        race_mode: seed_data.race_mode,
        timestamp: seed_data.timestamp,
        preset: seed_data.preset.clone().unwrap_or("Custom".to_string()),
        item_progression_preset: seed_data
            .item_progression_preset
            .clone()
            .unwrap_or("Custom".to_string()),
        progression_rate: format!("{:?}", seed_data.difficulty.progression_rate),
        random_tank: seed_data.difficulty.random_tank,
        filler_items: seed_data
            .difficulty
            .filler_items
            .iter()
            .filter(|&&x| x != Item::Nothing)
            .map(|x| format!("{:?}", x))
            .collect(),
        semi_filler_items: seed_data
            .difficulty
            .semi_filler_items
            .iter()
            .map(|x| format!("{:?}", x))
            .collect(),
        early_filler_items: seed_data
            .difficulty
            .early_filler_items
            .iter()
            .map(|x| format!("{:?}", x))
            .collect(),
        item_placement_style: format!("{:?}", seed_data.difficulty.item_placement_style),
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
        mark_map_stations: seed_data.mark_map_stations,
        item_markers: seed_data.item_markers.clone(),
        item_dot_change: seed_data.item_dot_change.clone(),
        transition_letters: seed_data.transition_letters,
        all_items_spawn: seed_data.all_items_spawn,
        acid_chozo: seed_data.acid_chozo,
        buffed_drops: seed_data.buffed_drops,
        fast_elevators: seed_data.fast_elevators,
        fast_doors: seed_data.fast_doors,
        fast_pause_menu: seed_data.fast_pause_menu,
        respin: seed_data.respin,
        infinite_space_jump: seed_data.infinite_space_jump,
        momentum_conservation: seed_data.momentum_conservation,
        objectives: seed_data.objectives.clone(),
        doors: seed_data.doors.clone(),
        start_location_mode: seed_data.start_location_mode.clone(),
        map_layout: seed_data.map_layout.clone(),
        save_animals: seed_data.save_animals.clone(),
        early_save: seed_data.early_save,
        area_assignment: seed_data.area_assignment.clone(),
        ultra_low_qol: seed_data.ultra_low_qol,
        preset_data: &app_data.preset_data,
        enabled_tech,
        enabled_notables,
    };
    let seed_header_html = seed_header_template.render()?;

    let seed_footer_template = SeedFooterTemplate {
        race_mode: seed_data.race_mode,
        all_items_spawn: seed_data.all_items_spawn,
        supers_double: seed_data.supers_double,
        ultra_low_qol: seed_data.ultra_low_qol,
    };
    let seed_footer_html = seed_footer_template.render()?;
    Ok((seed_header_html, seed_footer_html))
}
