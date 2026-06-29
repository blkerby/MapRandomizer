use crate::settings::{ExperimentalSettings, RandomizerSettings, StartLocationMode};

pub fn environment_respect_logic_enabled(settings: &RandomizerSettings) -> bool {
    if settings.start_location_settings.mode == StartLocationMode::Escape {
        return false;
    }
    let exp = &settings.experimental_settings;
    (exp.randomize_water_environments && !exp.water_visual_only)
        || (exp.randomize_heat_environments && !exp.heat_visual_only)
}

pub fn environment_needs_solvability_probe(settings: &RandomizerSettings) -> bool {
    environment_respect_logic_enabled(settings)
        && !settings.experimental_settings.water_flood_near_spawn
        && !settings.experimental_settings.heat_flood_near_spawn
}

pub const MAX_ENVIRONMENT_SOLVABILITY_ATTEMPTS: usize = 128;

/// Item-placement trials per environment overlay. One seed can fail on precedence luck alone.
pub const ENVIRONMENT_PROBE_ITEM_SEEDS: usize = 8;

/// Cap balanced flood/dry (or heat/cool) pairs — fewer pairs means higher solvability on scrambled maps.
pub fn effective_environment_pair_count(settings: &ExperimentalSettings, base: u32) -> u32 {
    if settings.randomize_water_environments && settings.randomize_heat_environments {
        base.min(1)
    } else if settings.randomize_water_environments || settings.randomize_heat_environments {
        base.min(2)
    } else {
        base
    }
}

/// Probe variants: user's start mode (early-save kept when both water and heat are on),
/// plus Ship as a fallback when the player uses random start.
pub fn environment_probe_settings_variants(
    settings: &RandomizerSettings,
) -> Vec<RandomizerSettings> {
    let both_env = settings.experimental_settings.randomize_water_environments
        && settings.experimental_settings.randomize_heat_environments;
    let mut probe_settings = settings.clone();
    if !both_env {
        probe_settings.quality_of_life_settings.early_save = false;
    }
    let mut variants = vec![probe_settings.clone()];
    if settings.start_location_settings.mode == StartLocationMode::Random {
        let mut ship = probe_settings;
        ship.start_location_settings.mode = StartLocationMode::Ship;
        variants.push(ship);
    }
    variants
}

/// After this many failed overlays on one map, retry with a single flood/dry pair.
pub fn environment_reduced_pair_attempt(settings: &RandomizerSettings) -> RandomizerSettings {
    let mut reduced = settings.clone();
    reduced.experimental_settings.water_room_count = 1;
    reduced.experimental_settings.heat_room_count = 1;
    reduced
}
