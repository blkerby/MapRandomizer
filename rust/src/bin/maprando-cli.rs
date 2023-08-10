use anyhow::{bail, Context, Result};
use clap::Parser;
use maprando::customize::{customize_rom, CustomizeSettings, MusicSettings};
use maprando::game_data::{Item, Map};
use maprando::patch::ips_write::create_ips_patch;
use maprando::patch::Rom;
use maprando::randomize::{
    DebugOptions, ItemMarkers, ItemPlacementStyle, ItemPriorityGroup, MotherBrainFight, Objectives,
    ProgressionRate, Randomization, Randomizer,
};
use maprando::spoiler_map;
use maprando::web::{SamusSpriteInfo, SamusSpriteCategory};
use maprando::{game_data::GameData, patch::make_rom, randomize::DifficultyConfig};
use std::path::{Path, PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    map: PathBuf,

    #[arg(long)]
    item_placement_seed: Option<usize>,

    #[arg(long)]
    input_rom: PathBuf,

    #[arg(long)]
    output_rom: Option<PathBuf>,

    #[arg(long)]
    output_spoiler_log: Option<PathBuf>,

    #[arg(long)]
    output_spoiler_map_assigned: Option<PathBuf>,

    #[arg(long)]
    output_spoiler_map_vanilla: Option<PathBuf>,

    #[arg(long)]
    area_themed_palette: bool,
}

fn get_randomization(args: &Args, game_data: &GameData) -> Result<Randomization> {
    let map_string = std::fs::read_to_string(&args.map)
        .with_context(|| format!("Unable to read map file at {}", args.map.display()))?;
    let map: Map = serde_json::from_str(&map_string)
        .with_context(|| format!("Unable to parse map file at {}", args.map.display()))?;

    // let ignored_tech: Vec<String> = ["canWallIceClip", "canGrappleClip", "canUseSpeedEchoes"].iter().map(|x| x.to_string()).collect();
    // let tech: Vec<String> = game_data.tech_isv.keys.iter().filter(|&x| !ignored_tech.contains(&x)).cloned().collect();
    // let tech: Vec<String> = vec![
    //     "canCrouchJump",
    //     "canDownGrab",
    //     "canHeatRun",
    //     "canIBJ",
    //     "canShinespark",
    //     "canSuitlessMaridia",
    //     "canWalljump"
    //     //   "can3HighMidAirMorph",
    //     // "canBlueSpaceJump",
    //     // "canBombAboveIBJ",
    //     // "canBombHorizontally",
    //     // "canBombJumpWaterEscape",
    //     // "canCeilingClip",
    //     // "canCrabClimb",
    //     // "canCrouchJump",
    //     // "canCrumbleJump",
    //     // "canCrumbleSpinJump",
    //     // "canCrystalFlash",
    //     // "canCrystalFlashForceStandup",
    //     // "canDamageBoost",
    //     // "canDownGrab",
    //     // "canGateGlitch",
    //     // "canGrappleJump",
    //     // "canGravityJump",
    //     // "canHBJ",
    //     // "canHeatRun",
    //     // "canHitbox",
    //     // "canIBJ",
    //     // "canIceZebetitesSkip",
    //     // "canIframeSpikeJump",
    //     // "canJumpIntoIBJ",
    //     // "canLateralMidAirMorph",
    //     // "canLavaGravityJump",
    //     // "canMaridiaTubeClip",
    //     // "canMetroidAvoid",
    //     // "canMochtroidIceClimb",
    //     // "canMochtroidIceClip",
    //     // "canMockball",
    //     // "canMoonfall",
    //     // "canPreciseWalljump",
    //     // "canQuickLowTideWalljumpWaterEscape",
    //     // "canSandMochtroidIceClimb",
    //     // "canShinespark",
    //     // "canShotBlockOverload",
    //     // "canSnailClimb",
    //     // "canSnailClip",
    //     // "canSpringBallJump",
    //     // "canSpringBallJumpMidAir",
    //     // "canStationaryLateralMidAirMorph",
    //     // "canStationarySpinJump",
    //     // "canSuitlessLavaDive",
    //     // "canSuitlessMaridia",
    //     // "canSuperReachAround",
    //     // "canTrickyJump",
    //     // "canTrickyUseFrozenEnemies",
    //     // "canTunnelCrawl",
    //     // "canTurnaroundAimCancel",
    //     // "canTwoTileSqueeze",
    //     // "canUnmorphBombBoost",
    //     // "canUseEnemies",
    //     // "canUseFrozenEnemies",
    //     // "canWalljump",
    //     // "canWrapAroundShot",
    //     // "canXRayStandUp"
    // ].iter().map(|x| x.to_string()).collect();
    // let tech = vec![];

    let difficulty = DifficultyConfig {
        tech: game_data.tech_isv.keys.clone(),
        notable_strats: vec![],
        // tech,
        shine_charge_tiles: 16.0,
        // shine_charge_tiles: 32,
        progression_rate: ProgressionRate::Fast,
        filler_items: vec![Item::Missile],
        early_filler_items: vec![],
        item_placement_style: ItemPlacementStyle::Neutral,
        item_priorities: vec![
            ItemPriorityGroup {
                name: "Default".to_string(),
                items: game_data
                    .item_isv
                    .keys
                    .iter()
                    .filter(|x| x != &"Varia" && x != &"Gravity")
                    .cloned()
                    .collect(),
            },
            ItemPriorityGroup {
                name: "Late".to_string(),
                items: vec!["Varia".to_string(), "Gravity".to_string()],
            },
        ],
        resource_multiplier: 1.0,
        escape_timer_multiplier: 3.0,
        gate_glitch_leniency: 0,
        phantoon_proficiency: 1.0,
        draygon_proficiency: 1.0,
        ridley_proficiency: 1.0,
        botwoon_proficiency: 1.0,
        supers_double: true,
        mother_brain_fight: MotherBrainFight::Short,
        escape_enemies_cleared: true,
        escape_refill: true,
        escape_movement_items: true,
        mark_map_stations: true,
        transition_letters: true,
        item_markers: ItemMarkers::ThreeTiered,
        all_items_spawn: true,
        acid_chozo: true,
        fast_elevators: true,
        fast_doors: true,
        fast_pause_menu: true,
        respin: false,
        infinite_space_jump: false,
        momentum_conservation: false,
        objectives: Objectives::Bosses,
        randomized_start: false,
        save_animals: false,
        disable_walljump: false,
        maps_revealed: true,
        vanilla_map: false,
        ultra_low_qol: false,
        skill_assumptions_preset: Some("None".to_string()),
        item_progression_preset: Some("None".to_string()),
        quality_of_life_preset: Some("None".to_string()),
        debug_options: Some(DebugOptions {
            new_game_extra: true,
            extended_spoiler: true,
        }),
    };
    let difficulty_tiers = [difficulty];
    let randomizer = Randomizer::new(&map, &difficulty_tiers, &game_data);
    let max_attempts = if args.item_placement_seed.is_some() {
        1
    } else {
        10
    };
    for attempt_num in 0..max_attempts {
        let seed = match args.item_placement_seed {
            Some(s) => s,
            None => attempt_num,
        };
        if let Ok(randomization) = randomizer.randomize(seed, 1) {
            return Ok(randomization);
        } else {
            println!("Failed randomization attempt");
        }
    }
    bail!("Exhausted randomization attempts");
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args = Args::parse();
    let sm_json_data_path = Path::new("../sm-json-data");
    let room_geometry_path = Path::new("../room_geometry.json");
    let palettes_path = Path::new("../palettes.json");
    let escape_timings_path = Path::new("data/escape_timings.json");
    let start_locations_path = Path::new("data/start_locations.json");
    let hub_locations_path = Path::new("data/hub_locations.json");
    let game_data = GameData::load(
        sm_json_data_path,
        room_geometry_path,
        palettes_path,
        escape_timings_path,
        start_locations_path,
        hub_locations_path,
    )?;

    // Perform randomization (map selection & item placement):
    let randomization = get_randomization(&args, &game_data)?;

    // // Override start location:
    // randomization.start_location = game_data.start_locations.last().unwrap().clone();

    // Generate the patched ROM:
    let mut input_rom = Rom::load(&args.input_rom)?;
    input_rom.data.resize(0x400000, 0);
    let game_rom = make_rom(&input_rom, &randomization, &game_data)?;
    let ips_patch = create_ips_patch(&input_rom.data, &game_rom.data);

    let mut output_rom = input_rom.clone();
    let customize_settings = CustomizeSettings {
        samus_sprite: Some("samus".to_string()),
        // samus_sprite: None,
        vanilla_screw_attack_animation: true,
        area_themed_palette: false,
        // music: MusicSettings::AreaThemed,
        music: MusicSettings::Vanilla,
        disable_beeping: false,
        etank_color: None,
    };
    customize_rom(&mut output_rom, &ips_patch, &customize_settings, &game_data, &[
        SamusSpriteCategory {
            category_name: "category".to_string(),
            sprites: vec![
                SamusSpriteInfo {
                    name: "samus".to_string(),
                    display_name: "Samus".to_string(),
                    credits_name: None,
                    authors: vec!["Nintendo".to_string()],
                }
            ]
        }
    ])?;

    // Save the outputs:
    if let Some(output_rom_path) = &args.output_rom {
        println!("Writing output ROM to {}", output_rom_path.display());
        output_rom.save(output_rom_path)?;
    }

    if let Some(output_spoiler_log_path) = &args.output_spoiler_log {
        println!(
            "Writing spoiler log to {}",
            output_spoiler_log_path.display()
        );
        let spoiler_str = serde_json::to_string_pretty(&randomization.spoiler_log)?;
        std::fs::write(output_spoiler_log_path, spoiler_str)?;
    }

    let spoiler_maps = spoiler_map::get_spoiler_map(&output_rom, &randomization.map, &game_data)?;

    if let Some(output_spoiler_map_assigned_path) = &args.output_spoiler_map_assigned {
        println!(
            "Writing spoiler map (assigned areas) to {}",
            output_spoiler_map_assigned_path.display()
        );
        let spoiler_map_assigned = spoiler_maps.assigned.clone();
        std::fs::write(output_spoiler_map_assigned_path, spoiler_map_assigned)?;
    }

    if let Some(output_spoiler_map_vanilla_path) = &args.output_spoiler_map_vanilla {
        println!(
            "Writing spoiler map (vanilla areas) to {}",
            output_spoiler_map_vanilla_path.display()
        );
        let spoiler_map_vanilla = spoiler_maps.vanilla.clone();
        std::fs::write(output_spoiler_map_vanilla_path, spoiler_map_vanilla)?;
    }

    Ok(())
}
