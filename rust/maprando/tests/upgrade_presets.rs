use std::path::Path;

use anyhow::{Context, Result, bail};
use maprando::{preset::PresetData, settings::try_upgrade_settings};
use maprando_game::GameData;

/// Test that settings presets from old versions can be upgraded
/// correctly, and that upgrading a second time has no effect.
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

/// Test that current settings presets are unchanged by upgrading.
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
