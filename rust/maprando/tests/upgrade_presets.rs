use std::path::Path;

use anyhow::{Context, Result, bail};
use maprando::{preset::PresetData, settings::try_upgrade_settings};
use maprando_game::GameData;

// Test that settings presets from old versions can be upgraded
// correctly, and that upgrading a second time has no effect.
#[test]
fn test_upgrade_old_presets() -> Result<()> {
    let base_path = Path::new("..");
    let game_data = GameData::load(base_path).context("Unable to load game data")?;
    let tech_path = base_path.join("data/tech_data.json");
    let notable_path = base_path.join("data/notable_data.json");
    let presets_path = base_path.join("data/presets");
    let preset_data = PresetData::load(&tech_path, &notable_path, &presets_path, &game_data)?;

    for entry in std::fs::read_dir("tests/presets")? {
        let entry = entry?;
        println!("Checking preset: {}", entry.path().display());
        let settings0_str = std::fs::read_to_string(entry.path())
            .with_context(|| format!("Unable to load preset at {}", entry.path().display()))?;
        let (settings1_str, settings1) = try_upgrade_settings(settings0_str, &preset_data, false)
            .with_context(|| {
            format!("Unable to upgrade preset at {}", entry.path().display())
        })?;
        let (_, settings2) = try_upgrade_settings(settings1_str, &preset_data, false)
            .with_context(|| {
                format!(
                    "Unable to double-upgrade preset at {}",
                    entry.path().display()
                )
            })?;
        if settings1 != settings2 {
            bail!(
                "Settings upgrade not idempotent for preset at {}",
                entry.path().display()
            );
        }
    }
    Ok(())
}

// Test that current settings presets are unchanged by upgrading.
#[test]
fn test_upgrade_current_presets() -> Result<()> {
    let base_path = Path::new("..");
    let game_data = GameData::load(base_path).context("Unable to load game data")?;
    let tech_path = base_path.join("data/tech_data.json");
    let notable_path = base_path.join("data/notable_data.json");
    let presets_path = base_path.join("data/presets");
    let preset_data = PresetData::load(&tech_path, &notable_path, &presets_path, &game_data)?;

    for preset in preset_data.full_presets.iter() {
        println!("Checking preset: {}", preset.name.as_ref().unwrap());
        let settings_str = serde_json::to_string_pretty(preset)
            .with_context(|| "Unable to serialize current preset")?;
        let (_, upgraded_settings) = try_upgrade_settings(settings_str, &preset_data, false)
            .with_context(|| "Unable to upgrade current preset")?;
        if preset != &upgraded_settings {
            bail!("Current preset changed by upgrade");
        }
    }
    Ok(())
}

use serde_json::Value;

// Set (or remove) a key inside a top-level settings section, if the section exists.
fn tweak(settings: &mut Value, section: &str, key: &str, value: Option<Value>) {
    if let Some(obj) = settings.get_mut(section).and_then(|v| v.as_object_mut()) {
        match value {
            Some(v) => {
                obj.insert(key.to_string(), v);
            }
            None => {
                obj.remove(key);
            }
        }
    }
}

// A null value (what the web form can send for an unchecked radio group)
// should behave exactly like a missing value.
#[test]
fn test_upgrade_null_equals_missing() -> Result<()> {
    let base_path = Path::new("..");
    let game_data = GameData::load(base_path).context("Unable to load game data")?;
    let tech_path = base_path.join("data/tech_data.json");
    let notable_path = base_path.join("data/notable_data.json");
    let presets_path = base_path.join("data/presets");
    let preset_data = PresetData::load(&tech_path, &notable_path, &presets_path, &game_data)?;

    let fields = [
        ("objective_settings", "objective_areas"),
        ("objective_settings", "objective_screen"),
        ("other_settings", "savestate"),
        ("other_settings", "speed_booster"),
    ];

    for preset in preset_data.full_presets.iter() {
        let base: Value = serde_json::to_value(preset)?;
        for (section, key) in fields {
            let mut missing = base.clone();
            tweak(&mut missing, section, key, None);
            let mut null = base.clone();
            tweak(&mut null, section, key, Some(Value::Null));

            let (_, from_missing) =
                try_upgrade_settings(missing.to_string(), &preset_data, false)
                    .with_context(|| format!("missing {section}.{key} failed"))?;
            let (_, from_null) = try_upgrade_settings(null.to_string(), &preset_data, false)
                .with_context(|| format!("null {section}.{key} failed"))?;

            if from_missing != from_null {
                bail!("null and missing differ for {section}.{key}");
            }
        }
    }
    Ok(())
}

use std::collections::BTreeSet;

const ALLOWED_REQUIRED: &[(&str, &str)] = &[
    ("item_progression_settings", "item_placement_style"),
    ("item_progression_settings", "item_priority_strength"),
    ("item_progression_settings", "progression_rate"),
    ("item_progression_settings", "random_tank"),
    ("item_progression_settings", "spazer_before_plasma"),
    ("item_progression_settings", "stop_item_placement_early"),
    ("objective_settings", "max_objectives"),
    ("objective_settings", "min_objectives"),
    ("other_settings", "door_locks_size"),
    ("other_settings", "energy_free_shinesparks"),
    ("other_settings", "race_mode"),
    ("other_settings", "wall_jump"),
    ("quality_of_life_settings", "acid_chozo"),
    ("quality_of_life_settings", "all_items_spawn"),
    ("quality_of_life_settings", "early_save"),
    ("quality_of_life_settings", "escape_enemies_cleared"),
    ("quality_of_life_settings", "escape_movement_items"),
    ("quality_of_life_settings", "escape_refill"),
    ("quality_of_life_settings", "fast_doors"),
    ("quality_of_life_settings", "fast_elevators"),
    ("quality_of_life_settings", "fast_pause_menu"),
    ("quality_of_life_settings", "infinite_space_jump"),
    ("quality_of_life_settings", "item_markers"),
    ("quality_of_life_settings", "momentum_conservation"),
    ("quality_of_life_settings", "mother_brain_fight"),
    ("quality_of_life_settings", "opposite_area_revealed"),
    ("quality_of_life_settings", "remove_climb_lava"),
    ("quality_of_life_settings", "respin"),
    ("quality_of_life_settings", "room_outline_revealed"),
    ("quality_of_life_settings", "supers_double"),
];

// For every scalar field in the main sections, a null or missing value
// should upgrade successfully and give the same result.
#[test]
fn test_upgrade_null_and_missing_all_scalars() -> Result<()> {
    let base_path = Path::new("..");
    let game_data = GameData::load(base_path).context("Unable to load game data")?;
    let tech_path = base_path.join("data/tech_data.json");
    let notable_path = base_path.join("data/notable_data.json");
    let presets_path = base_path.join("data/presets");
    let preset_data = PresetData::load(&tech_path, &notable_path, &presets_path, &game_data)?;

    let sections = [
        "other_settings",
        "objective_settings",
        "quality_of_life_settings",
        "item_progression_settings",
    ];

    let mut failures: BTreeSet<String> = BTreeSet::new();

    for preset in preset_data.full_presets.iter() {
        let base: Value = serde_json::to_value(preset)?;
        for section in sections {
            let Some(obj) = base.get(section).and_then(|v| v.as_object()) else {
                continue;
            };
            for (key, val) in obj {
                let is_scalar = val.is_string() || val.is_boolean() || val.is_number();
                if !is_scalar || ALLOWED_REQUIRED.contains(&(section, key.as_str())) {
                    continue;
                }

                let mut missing = base.clone();
                tweak(&mut missing, section, key, None);
                let mut null = base.clone();
                tweak(&mut null, section, key, Some(Value::Null));

                let r_missing = try_upgrade_settings(missing.to_string(), &preset_data, false);
                let r_null = try_upgrade_settings(null.to_string(), &preset_data, false);

match (&r_missing, &r_null) {
                    (Ok((_, a)), Ok((_, b))) => {
                        if a != b {
                            failures.insert(format!("{section}.{key}: null and missing differ"));
                        }
                    }
                    _ => {
                        if r_missing.is_err() {
                            failures.insert(format!("{section}.{key} [missing]"));
                        }
                        if r_null.is_err() {
                            failures.insert(format!("{section}.{key} [null]"));
                        }
                    }
                }
            }
        }
    }

    if !failures.is_empty() {
        for f in &failures {
            println!("FAIL {f}");
        }
        bail!("{} field(s) lack a default for null/missing", failures.len());
    }
    Ok(())
}