use crate::web::AppData;
use actix_web::{HttpResponse, Responder, post, web};
use anyhow::{Context, Result, bail};
use hashbrown::HashMap;
use log::error;
use maprando::settings::{
    InitialMapRevealSettings, MapRevealLevel, NotableSetting, RandomizerSettings, TechSetting,
    parse_randomizer_settings,
};
use maprando_game::{NotableId, RoomId, TechId};

use super::VERSION;

fn assign_presets(settings: &mut serde_json::Value, app_data: &AppData) -> Result<()> {
    if let Some(preset) = settings["skill_assumption_settings"]["preset"].as_str() {
        let preset = preset.to_owned();
        for p in &app_data.preset_data.skill_presets {
            if p.preset.as_ref() == Some(&preset) {
                *settings.get_mut("skill_assumption_settings").unwrap() = serde_json::to_value(p)?;
            }
        }
    }
    if let Some(preset) = settings["item_progression_settings"]["preset"].as_str() {
        let preset = preset.to_owned();
        for p in &app_data.preset_data.item_progression_presets {
            if p.preset.as_ref() == Some(&preset) {
                *settings.get_mut("item_progression_settings").unwrap() = serde_json::to_value(p)?;
            }
        }
    }
    if let Some(preset) = settings["quality_of_life_settings"]["preset"].as_str() {
        let preset = preset.to_owned();
        for p in &app_data.preset_data.quality_of_life_presets {
            if p.preset.as_ref() == Some(&preset) {
                *settings.get_mut("quality_of_life_settings").unwrap() = serde_json::to_value(p)?;
            }
        }
    }
    if let Some(preset) = settings["objective_settings"]["preset"].as_str() {
        let preset = preset.to_owned();
        for p in &app_data.preset_data.objective_presets {
            if p.preset.as_ref() == Some(&preset) {
                *settings.get_mut("objective_settings").unwrap() = serde_json::to_value(p)?;
            }
        }
    }
    if let Some(preset) = settings["name"].as_str() {
        let preset = preset.to_owned();
        for p in &app_data.preset_data.full_presets {
            if p.name.as_ref() == Some(&preset) {
                *settings = serde_json::to_value(p)?;
            }
        }
    }
    Ok(())
}

fn upgrade_tech_settings(settings: &mut serde_json::Value, app_data: &AppData) -> Result<()> {
    // This updates the names of tech, discards any obsolete tech settings, and disables
    // any new tech that are not referenced in the settings.
    let mut tech_map: HashMap<TechId, bool> = HashMap::new();
    for tech_setting in settings["skill_assumption_settings"]["tech_settings"]
        .as_array()
        .context("missing tech_settings")?
    {
        let tech_id = tech_setting["id"]
            .as_i64()
            .context("tech_setting missing id field")? as TechId;
        let enabled = tech_setting["enabled"]
            .as_bool()
            .context("tech_setting missing enabled field")?;
        tech_map.insert(tech_id, enabled);
    }

    let mut new_tech_settings: Vec<TechSetting> = vec![];
    for t in &app_data
        .preset_data
        .default_preset
        .skill_assumption_settings
        .tech_settings
    {
        new_tech_settings.push(TechSetting {
            id: t.id,
            name: t.name.clone(),
            enabled: tech_map.get(&t.id).copied().unwrap_or(false),
        });
    }
    *settings
        .get_mut("skill_assumption_settings")
        .unwrap()
        .get_mut("tech_settings")
        .unwrap() = serde_json::to_value(new_tech_settings)?;

    Ok(())
}

fn upgrade_notable_settings(settings: &mut serde_json::Value, app_data: &AppData) -> Result<()> {
    // This updates the names of notables, discards any obsolete notables settings, and disables
    // any new notables that are not referenced in the settings.
    let mut notable_map: HashMap<(RoomId, NotableId), bool> = HashMap::new();
    for notable_setting in settings["skill_assumption_settings"]["notable_settings"]
        .as_array()
        .context("missing notable_settings")?
    {
        let room_id = notable_setting["room_id"]
            .as_i64()
            .context("notable_setting missing room_id field")? as RoomId;
        let notable_id = notable_setting["notable_id"]
            .as_i64()
            .context("notable_setting missing notable_id field")?
            as RoomId;
        let enabled = notable_setting["enabled"]
            .as_bool()
            .context("notable_setting missing enabled field")?;
        notable_map.insert((room_id, notable_id), enabled);
    }

    let mut new_notable_settings: Vec<NotableSetting> = vec![];
    for s in &app_data
        .preset_data
        .default_preset
        .skill_assumption_settings
        .notable_settings
    {
        new_notable_settings.push(NotableSetting {
            room_id: s.room_id,
            notable_id: s.notable_id,
            room_name: s.room_name.clone(),
            notable_name: s.notable_name.clone(),
            enabled: notable_map
                .get(&(s.room_id, s.notable_id))
                .copied()
                .unwrap_or(false),
        });
    }
    *settings
        .get_mut("skill_assumption_settings")
        .unwrap()
        .get_mut("notable_settings")
        .unwrap() = serde_json::to_value(new_notable_settings)?;

    Ok(())
}

fn upgrade_other_skill_settings(settings: &mut serde_json::Value) -> Result<()> {
    let skill_assumption_settings = settings
        .get_mut("skill_assumption_settings")
        .context("missing skill_assumption_settings")?
        .as_object_mut()
        .context("skill_assumption_settings is not object")?;
    if !skill_assumption_settings.contains_key("bomb_into_cf_leniency") {
        skill_assumption_settings.insert("bomb_into_cf_leniency".to_string(), 5.into());
    }
    if !skill_assumption_settings.contains_key("jump_into_cf_leniency") {
        skill_assumption_settings.insert("jump_into_cf_leniency".to_string(), 9.into());
    }
    if !skill_assumption_settings.contains_key("spike_xmode_leniency") {
        skill_assumption_settings.insert("spike_xmode_leniency".to_string(), 2.into());
    }
    if !skill_assumption_settings.contains_key("farm_time_limit") {
        skill_assumption_settings.insert("farm_time_limit".to_string(), (60.0).into());
    }

    Ok(())
}

fn upgrade_item_progression_settings(settings: &mut serde_json::Value) -> Result<()> {
    let item_progression_settings = settings
        .get_mut("item_progression_settings")
        .context("missing item_progression_settings")?
        .as_object_mut()
        .context("item_progression_settings is not object")?;
    if !item_progression_settings.contains_key("ammo_collect_fraction") {
        item_progression_settings.insert("ammo_collect_fraction".to_string(), (0.7).into());
    }

    Ok(())
}

fn upgrade_initial_map_reveal_settings(settings: &mut serde_json::Value) -> Result<()> {
    if settings["quality_of_life_settings"]
        .as_object()
        .unwrap()
        .contains_key("initial_map_reveal_settings")
    {
        return Ok(());
    }

    let maps_revealed = settings["other_settings"]
        .as_object_mut()
        .context("missing 'other_settings'")?["maps_revealed"]
        .as_str()
        .context("expected 'maps_revealed' to be a string")?
        .to_owned();

    let qol_settings = settings["quality_of_life_settings"]
        .as_object_mut()
        .unwrap();
    if let Some(mark_map_stations) = qol_settings["mark_map_stations"].as_bool() {
        let reveal_level = match maps_revealed.as_str() {
            "No" => MapRevealLevel::No,
            "Partial" => MapRevealLevel::Partial,
            "Full" => MapRevealLevel::Full,
            _ => bail!("Unexpected value of 'maps_revealed'"),
        };
        let initial_map_reveal_settings = InitialMapRevealSettings {
            preset: Some(match reveal_level {
                MapRevealLevel::No => {
                    if mark_map_stations {
                        "Maps".to_string()
                    } else {
                        "No".to_string()
                    }
                }
                MapRevealLevel::Partial => "Partial".to_string(),
                MapRevealLevel::Full => "Full".to_string(),
            }),
            map_stations: if reveal_level == MapRevealLevel::No {
                if mark_map_stations {
                    MapRevealLevel::Full
                } else {
                    MapRevealLevel::No
                }
            } else {
                reveal_level
            },
            save_stations: reveal_level,
            refill_stations: reveal_level,
            ship: reveal_level,
            objectives: reveal_level,
            area_transitions: reveal_level,
            items1: reveal_level,
            items2: reveal_level,
            items3: reveal_level,
            items4: reveal_level,
            other: reveal_level,
            all_areas: reveal_level != MapRevealLevel::No,
        };
        qol_settings.insert(
            "initial_map_reveal_settings".to_string(),
            serde_json::to_value(initial_map_reveal_settings)?,
        );
    };

    Ok(())
}

fn upgrade_qol_settings(settings: &mut serde_json::Value) -> Result<()> {
    let etank_refill = settings["other_settings"]["etank_refill"]
        .as_str()
        .unwrap_or("Vanilla")
        .to_string();
    let qol_settings = settings
        .get_mut("quality_of_life_settings")
        .context("missing quality_of_life_settings")?
        .as_object_mut()
        .context("quality_of_life_settings is not object")?;
    if !qol_settings.contains_key("fanfares") {
        qol_settings.insert("fanfares".to_string(), "Off".into());
    }
    if !qol_settings.contains_key("etank_refill") {
        qol_settings.insert("etank_refill".to_string(), etank_refill.into());
    }
    if !qol_settings.contains_key("energy_station_reserves") {
        qol_settings.insert("energy_station_reserves".to_string(), false.into());
    }
    if !qol_settings.contains_key("disableable_etanks") {
        qol_settings.insert("disableable_etanks".to_string(), false.into());
    }
    if !qol_settings.contains_key("reserve_backward_transfer") {
        qol_settings.insert("reserve_backward_transfer".to_string(), false.into());
    }
    upgrade_initial_map_reveal_settings(settings)?;
    Ok(())
}

fn upgrade_map_setting(settings: &mut serde_json::Value) -> Result<()> {
    if settings["map_layout"].as_str() == Some("Tame") {
        *settings.get_mut("map_layout").unwrap() = "Standard".into();
    }
    Ok(())
}

fn upgrade_start_location_setings(settings: &mut serde_json::Value) -> Result<()> {
    if !settings
        .as_object()
        .unwrap()
        .contains_key("start_location_settings")
    {
        let start_location_mode: String = settings["start_location_mode"].as_str().unwrap().into();
        settings.as_object_mut().unwrap().insert(
            "start_location_settings".to_string(),
            serde_json::Value::Object(
                vec![("mode".to_string(), start_location_mode.into())]
                    .into_iter()
                    .collect(),
            ),
        );
    }
    Ok(())
}

fn upgrade_animals_setting(settings: &mut serde_json::Value) -> Result<()> {
    if settings["save_animals"].as_str() == Some("Maybe") {
        *settings.get_mut("save_animals").unwrap() = "Optional".into();
    }
    Ok(())
}

fn upgrade_objective_settings(settings: &mut serde_json::Value, app_data: &AppData) -> Result<()> {
    let settings_obj = settings
        .as_object_mut()
        .context("expected settings to be object")?;

    if !settings_obj.contains_key("objective_settings") {
        settings_obj.insert(
            "objective_settings".to_string(),
            serde_json::to_value(app_data.preset_data.objective_presets[1].clone()).unwrap(),
        );
    }
    if settings_obj.contains_key("objectives_mode") {
        *settings_obj
            .get_mut("objective_settings")
            .unwrap()
            .get_mut("preset")
            .unwrap() = settings_obj["objectives_mode"].as_str().unwrap().into();
    }
    if !settings_obj["objective_settings"]
        .as_object()
        .unwrap()
        .contains_key("objective_screen")
    {
        settings_obj
            .get_mut("objective_settings")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("objective_screen".to_string(), "Enabled".into());
    }
    Ok(())
}

pub fn try_upgrade_settings(
    settings_str: String,
    app_data: &AppData,
    apply_presets: bool,
) -> Result<(String, RandomizerSettings)> {
    let mut settings: serde_json::Value = serde_json::from_str(&settings_str)?;

    upgrade_objective_settings(&mut settings, app_data)?;
    if apply_presets {
        assign_presets(&mut settings, app_data)?;
    }
    upgrade_tech_settings(&mut settings, app_data)?;
    upgrade_notable_settings(&mut settings, app_data)?;
    upgrade_other_skill_settings(&mut settings)?;
    upgrade_item_progression_settings(&mut settings)?;
    upgrade_qol_settings(&mut settings)?;
    upgrade_map_setting(&mut settings)?;
    upgrade_start_location_setings(&mut settings)?;
    upgrade_animals_setting(&mut settings)?;

    // Update version field to current version:
    *settings
        .get_mut("version")
        .context("missing version field")? = VERSION.into();

    // Validate that the upgraded settings will parse as a RandomizerSettings struct:
    let settings_str = settings.to_string();
    let settings_out = parse_randomizer_settings(&settings_str)?;
    let settings_out_str = serde_json::to_string(&settings_out)?;
    Ok((settings_out_str, settings_out))
}

#[post("/upgrade-settings")]
async fn upgrade_settings(settings_str: String, app_data: web::Data<AppData>) -> impl Responder {
    match try_upgrade_settings(settings_str, &app_data, true) {
        Ok((settings_str, _)) => HttpResponse::Ok()
            .content_type("application/json")
            .body(settings_str),
        Err(e) => {
            error!("Failed to upgrade settings: {e}");
            HttpResponse::BadRequest().body(e.to_string())
        }
    }
}
