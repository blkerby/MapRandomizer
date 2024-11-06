use std::cmp::{max, min};

use hashbrown::HashMap;

use crate::{
    randomize::{DifficultyConfig, LockedDoor, Objective},
    settings::{MotherBrainFight, RandomizerSettings, WallJump},
};
use maprando_game::{
    BeamType, Capacity, DoorType, EnemyDrop, EnemyVulnerabilities, GameData, Item, Link, LinkIdx,
    LinksDataGroup, NodeId, Requirement, RoomId, VertexId,
};
use maprando_logic::{
    boss_requirements::{
        apply_botwoon_requirement, apply_draygon_requirement, apply_mother_brain_2_requirement,
        apply_phantoon_requirement, apply_ridley_requirement,
    },
    helpers::{suit_damage_factor, validate_energy},
    GlobalState, Inventory, LocalState,
};

fn apply_enemy_kill_requirement(
    global: &GlobalState,
    mut local: LocalState,
    count: Capacity,
    vul: &EnemyVulnerabilities,
) -> Option<LocalState> {
    // Prioritize using weapons that do not require ammo:
    if global.weapon_mask & vul.non_ammo_vulnerabilities != 0 {
        return Some(local);
    }

    let mut hp = vul.hp; // HP per enemy

    // Next use Missiles:
    if vul.missile_damage > 0 {
        let missiles_available = global.inventory.max_missiles - local.missiles_used;
        let missiles_to_use_per_enemy = max(
            0,
            min(
                missiles_available / count,
                (hp + vul.missile_damage - 1) / vul.missile_damage,
            ),
        );
        hp -= missiles_to_use_per_enemy * vul.missile_damage as Capacity;
        local.missiles_used += missiles_to_use_per_enemy * count;
    }

    // Then use Supers (some overkill is possible, where we could have used fewer Missiles, but we ignore that):
    if vul.super_damage > 0 {
        let supers_available = global.inventory.max_supers - local.supers_used;
        let supers_to_use_per_enemy = max(
            0,
            min(
                supers_available / count,
                (hp + vul.super_damage - 1) / vul.super_damage,
            ),
        );
        hp -= supers_to_use_per_enemy * vul.super_damage as Capacity;
        local.supers_used += supers_to_use_per_enemy * count;
    }

    // Finally, use Power Bombs (overkill is possible, where we could have used fewer Missiles or Supers, but we ignore that):
    if vul.power_bomb_damage > 0 && global.inventory.items[Item::Morph as usize] {
        let pbs_available = global.inventory.max_power_bombs - local.power_bombs_used;
        let pbs_to_use = max(
            0,
            min(
                pbs_available,
                (hp + vul.power_bomb_damage - 1) / vul.power_bomb_damage,
            ),
        );
        hp -= pbs_to_use * vul.power_bomb_damage as Capacity;
        // Power bombs hit all enemies in the group, so we do not multiply by the count.
        local.power_bombs_used += pbs_to_use;
    }

    if hp <= 0 {
        Some(local)
    } else {
        None
    }
}

pub const IMPOSSIBLE_LOCAL_STATE: LocalState = LocalState {
    energy_used: 0x3FFF,
    reserves_used: 0x3FFF,
    missiles_used: 0x3FFF,
    supers_used: 0x3FFF,
    power_bombs_used: 0x3FFF,
    shinecharge_frames_remaining: 0x3FFF,
};

pub const NUM_COST_METRICS: usize = 2;

fn compute_cost(local: LocalState, inventory: &Inventory) -> [f32; NUM_COST_METRICS] {
    let eps = 1e-15;
    let energy_cost = (local.energy_used as f32) / (inventory.max_energy as f32 + eps);
    let reserve_cost = (local.reserves_used as f32) / (inventory.max_reserves as f32 + eps);
    let missiles_cost = (local.missiles_used as f32) / (inventory.max_missiles as f32 + eps);
    let supers_cost = (local.supers_used as f32) / (inventory.max_supers as f32 + eps);
    let power_bombs_cost =
        (local.power_bombs_used as f32) / (inventory.max_power_bombs as f32 + eps);
    let shinecharge_cost = -(local.shinecharge_frames_remaining as f32) / 180.0;

    let ammo_sensitive_cost_metric = energy_cost
        + reserve_cost
        + 100.0 * (missiles_cost + supers_cost + power_bombs_cost + shinecharge_cost);
    let energy_sensitive_cost_metric = 100.0 * (energy_cost + reserve_cost)
        + missiles_cost
        + supers_cost
        + power_bombs_cost
        + shinecharge_cost;
    [ammo_sensitive_cost_metric, energy_sensitive_cost_metric]
}

fn validate_energy_no_auto_reserve(
    mut local: LocalState,
    global: &GlobalState,
    game_data: &GameData,
    difficulty: &DifficultyConfig,
) -> Option<LocalState> {
    if local.energy_used >= global.inventory.max_energy {
        if difficulty.tech[game_data.manage_reserves_tech_idx] {
            // Assume that just enough reserve energy is manually converted to regular energy.
            local.reserves_used += local.energy_used - (global.inventory.max_energy - 1);
            local.energy_used = global.inventory.max_energy - 1;
        } else {
            // Assume that reserves cannot be used (e.g. during a shinespark or enemy hit).
            return None;
        }
    }
    if local.reserves_used > global.inventory.max_reserves {
        return None;
    }
    Some(local)
}

fn validate_missiles(local: LocalState, global: &GlobalState) -> Option<LocalState> {
    if local.missiles_used > global.inventory.max_missiles {
        None
    } else {
        Some(local)
    }
}

fn validate_supers(local: LocalState, global: &GlobalState) -> Option<LocalState> {
    if local.supers_used > global.inventory.max_supers {
        None
    } else {
        Some(local)
    }
}

fn validate_power_bombs(local: LocalState, global: &GlobalState) -> Option<LocalState> {
    if local.power_bombs_used > global.inventory.max_power_bombs {
        None
    } else {
        Some(local)
    }
}

fn apply_gate_glitch_leniency(
    mut local: LocalState,
    global: &GlobalState,
    green: bool,
    heated: bool,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
) -> Option<LocalState> {
    if heated && !global.inventory.items[Item::Varia as usize] {
        local.energy_used += (difficulty.gate_glitch_leniency as f32
            * difficulty.resource_multiplier
            * 60.0) as Capacity;
        local = match validate_energy(
            local,
            &global.inventory,
            difficulty.tech[game_data.manage_reserves_tech_idx],
        ) {
            Some(x) => x,
            None => return None,
        };
    }
    if green {
        local.supers_used += difficulty.gate_glitch_leniency;
        return validate_supers(local, global);
    } else {
        let missiles_available = global.inventory.max_missiles - local.missiles_used;
        if missiles_available >= difficulty.gate_glitch_leniency {
            local.missiles_used += difficulty.gate_glitch_leniency;
            return validate_missiles(local, global);
        } else {
            local.missiles_used = global.inventory.max_missiles;
            local.supers_used += difficulty.gate_glitch_leniency - missiles_available;
            return validate_supers(local, global);
        }
    }
}

fn is_objective_complete(
    global: &GlobalState,
    _difficulty: &DifficultyConfig,
    objectives: &[Objective],
    game_data: &GameData,
    obj_id: usize,
) -> bool {
    // TODO: What to do when obj_id is out of bounds?
    if let Some(obj) = objectives.get(obj_id) {
        let flag_name = obj.get_flag_name();
        let flag_idx = game_data.flag_isv.index_by_key[flag_name];
        global.flags[flag_idx]
    } else {
        true
    }
}

fn apply_heat_frames(
    frames: Capacity,
    local: LocalState,
    global: &GlobalState,
    game_data: &GameData,
    difficulty: &DifficultyConfig,
) -> Option<LocalState> {
    let varia = global.inventory.items[Item::Varia as usize];
    let mut new_local = local;
    if varia {
        Some(new_local)
    } else {
        if !difficulty.tech[game_data.heat_run_tech_idx] {
            None
        } else {
            new_local.energy_used +=
                (frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity;
            validate_energy(
                new_local,
                &global.inventory,
                difficulty.tech[game_data.manage_reserves_tech_idx],
            )
        }
    }
}

fn get_enemy_drop_value(
    drop: &EnemyDrop,
    local: LocalState,
    reverse: bool,
    buffed_drops: bool,
) -> Capacity {
    let mut p_small = drop.small_energy_weight.get();
    let mut p_large = drop.large_energy_weight.get();
    let mut p_missile = drop.missile_weight.get();
    let p_tier1 = p_small + p_large + p_missile;
    let rel_small = p_small / p_tier1;
    let rel_large = p_large / p_tier1;
    let rel_missile = p_missile / p_tier1;
    let p_super = drop.super_weight.get();
    let p_pb = drop.power_bomb_weight.get();

    if !reverse {
        // For the forward traversal, we take into account how ammo drops roll over to energy if full.
        // This could also be done for the reverse traversal, but it would require branching on multiple
        // possibilities, which would need additional cost metrics in order to be effective;
        // we pass on that for now.
        //
        // In theory, health bomb could also be modeled, except that the "heatFramesWithEnergyDrops" requirement
        // does not identify exactly when the drops are collected. In cases where it significantly matters, this
        // could be handled by having a boolean property in "heatFramesWithEnergyDrops" to indicate that the
        // drops are obtained at the very end of the heat frames? We ignore it for now.
        if local.power_bombs_used == 0 {
            p_small += p_pb * rel_small;
            p_large += p_pb * rel_large;
            p_missile += p_pb * rel_missile;
        }
        if local.supers_used == 0 {
            p_small += p_super * rel_small;
            p_large += p_super * rel_large;
            p_missile += p_super * rel_missile;
        }
        if local.missiles_used == 0 {
            p_small += p_missile * p_small / (p_small + p_large);
            p_large += p_missile * p_large / (p_small + p_large);
        }
    }
    let expected_energy = p_small * if buffed_drops { 10.0 } else { 5.0 } + p_large * 20.0;
    (expected_energy * drop.count as f32) as Capacity
}

fn apply_heat_frames_with_energy_drops(
    frames: Capacity,
    drops: &[EnemyDrop],
    local: LocalState,
    global: &GlobalState,
    game_data: &GameData,
    settings: &RandomizerSettings,
    difficulty: &DifficultyConfig,
    reverse: bool,
) -> Option<LocalState> {
    let varia = global.inventory.items[Item::Varia as usize];
    let mut new_local = local;
    if varia {
        Some(new_local)
    } else {
        if !difficulty.tech[game_data.heat_run_tech_idx] {
            None
        } else {
            let mut total_drop_value = 0;
            for drop in drops {
                total_drop_value += get_enemy_drop_value(
                    drop,
                    local,
                    reverse,
                    settings.quality_of_life_settings.buffed_drops,
                )
            }
            let heat_energy =
                (frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity;
            total_drop_value = Capacity::min(total_drop_value, heat_energy);
            new_local.energy_used += heat_energy;
            if let Some(x) = validate_energy(
                new_local,
                &global.inventory,
                difficulty.tech[game_data.manage_reserves_tech_idx],
            ) {
                new_local = x;
            } else {
                return None;
            }
            if total_drop_value <= new_local.energy_used {
                new_local.energy_used -= total_drop_value;
            } else {
                new_local.reserves_used -= total_drop_value - new_local.energy_used;
                new_local.energy_used = 0;
            }
            Some(new_local)
        }
    }
}

#[derive(Clone)]
pub struct LockedDoorData {
    pub locked_doors: Vec<LockedDoor>,
    pub locked_door_node_map: HashMap<(RoomId, NodeId), usize>,
    pub locked_door_vertex_ids: Vec<Vec<VertexId>>,
}

pub fn apply_link(
    link: &Link,
    global: &GlobalState,
    local: LocalState,
    reverse: bool,
    settings: &RandomizerSettings,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
    locked_door_data: &LockedDoorData,
    objectives: &[Objective],
) -> Option<LocalState> {
    if reverse {
        if !link.end_with_shinecharge && local.shinecharge_frames_remaining > 0 {
            return None;
        }
    } else {
        if link.start_with_shinecharge && local.shinecharge_frames_remaining == 0 {
            return None;
        }
    }
    let new_local = apply_requirement(
        &link.requirement,
        global,
        local,
        reverse,
        settings,
        difficulty,
        game_data,
        locked_door_data,
        objectives,
    );
    if let Some(mut new_local) = new_local {
        if reverse {
            if !link.start_with_shinecharge {
                new_local.shinecharge_frames_remaining = 0;
            }
        } else {
            if !link.end_with_shinecharge {
                new_local.shinecharge_frames_remaining = 0;
            }
        }
        Some(new_local)
    } else {
        None
    }
}

fn has_beam(beam: BeamType, inventory: &Inventory) -> bool {
    let item = match beam {
        BeamType::Charge => Item::Charge,
        BeamType::Ice => Item::Ice,
        BeamType::Wave => Item::Wave,
        BeamType::Spazer => Item::Spazer,
        BeamType::Plasma => Item::Plasma,
    };
    inventory.items[item as usize]
}

fn get_heated_speedball_tiles(difficulty: &DifficultyConfig) -> f32 {
    let heat_leniency = difficulty.heated_shine_charge_tiles - difficulty.shine_charge_tiles;
    difficulty.speed_ball_tiles + heat_leniency
}

pub fn debug_requirement(
    req: &Requirement,
    global: &GlobalState,
    local: LocalState,
    reverse: bool,
    settings: &RandomizerSettings,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
    locked_door_data: &LockedDoorData,
    objectives: &[Objective],
) {
    println!(
        "{:?}: {:?}",
        req,
        apply_requirement(
            req,
            global,
            local,
            reverse,
            settings,
            difficulty,
            game_data,
            locked_door_data,
            objectives
        )
    );
    match req {
        Requirement::And(reqs) => {
            for r in reqs {
                debug_requirement(
                    r,
                    global,
                    local,
                    reverse,
                    settings,
                    difficulty,
                    game_data,
                    locked_door_data,
                    objectives,
                );
            }
        }
        Requirement::Or(reqs) => {
            for r in reqs {
                debug_requirement(
                    r,
                    global,
                    local,
                    reverse,
                    settings,
                    difficulty,
                    game_data,
                    locked_door_data,
                    objectives,
                );
            }
        }
        _ => {}
    }
}

pub fn apply_requirement(
    req: &Requirement,
    global: &GlobalState,
    local: LocalState,
    reverse: bool,
    settings: &RandomizerSettings,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
    locked_door_data: &LockedDoorData,
    objectives: &[Objective],
) -> Option<LocalState> {
    let can_manage_reserves = difficulty.tech[game_data.manage_reserves_tech_idx];
    match req {
        Requirement::Free => Some(local),
        Requirement::Never => None,
        Requirement::Tech(tech_idx) => {
            if difficulty.tech[*tech_idx] {
                Some(local)
            } else {
                None
            }
        }
        Requirement::Notable(notable_idx) => {
            if difficulty.notables[*notable_idx] {
                Some(local)
            } else {
                None
            }
        }
        Requirement::Item(item_id) => {
            if global.inventory.items[*item_id] {
                Some(local)
            } else {
                None
            }
        }
        Requirement::Flag(flag_id) => {
            if global.flags[*flag_id] {
                Some(local)
            } else {
                None
            }
        }
        Requirement::NotFlag(_flag_id) => {
            // We're ignoring this for now. It should be safe because all strats relying on a "not" flag will be
            // guarded by "canRiskPermanentLossOfAccess" if there is not an alternative strat with the flag set.
            Some(local)
        }
        Requirement::Objective(obj_id) => {
            if is_objective_complete(global, difficulty, objectives, game_data, *obj_id) {
                Some(local)
            } else {
                None
            }
        }
        Requirement::Walljump => match settings.other_settings.wall_jump {
            WallJump::Vanilla => {
                if difficulty.tech[game_data.wall_jump_tech_idx] {
                    Some(local)
                } else {
                    None
                }
            }
            WallJump::Collectible => {
                if difficulty.tech[game_data.wall_jump_tech_idx]
                    && global.inventory.items[Item::WallJump as usize]
                {
                    Some(local)
                } else {
                    None
                }
            }
        },
        Requirement::ClimbWithoutLava => {
            if settings.quality_of_life_settings.remove_climb_lava {
                Some(local)
            } else {
                None
            }
        }
        Requirement::HeatFrames(frames) => {
            apply_heat_frames(*frames, local, global, game_data, difficulty)
        }
        Requirement::HeatFramesWithEnergyDrops(frames, enemy_drops) => {
            apply_heat_frames_with_energy_drops(
                *frames,
                enemy_drops,
                local,
                global,
                game_data,
                settings,
                difficulty,
                reverse,
            )
        }
        Requirement::MainHallElevatorFrames => {
            if settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(188, local, global, game_data, difficulty)
            } else {
                apply_heat_frames(436, local, global, game_data, difficulty)
            }
        }
        Requirement::LowerNorfairElevatorDownFrames => {
            if settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(30, local, global, game_data, difficulty)
            } else {
                apply_heat_frames(60, local, global, game_data, difficulty)
            }
        }
        Requirement::LowerNorfairElevatorUpFrames => {
            if settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(48, local, global, game_data, difficulty)
            } else {
                apply_heat_frames(108, local, global, game_data, difficulty)
            }
        }
        Requirement::LavaFrames(frames) => {
            let varia = global.inventory.items[Item::Varia as usize];
            let gravity = global.inventory.items[Item::Gravity as usize];
            let mut new_local = local;
            if gravity && varia {
                Some(new_local)
            } else if gravity || varia {
                new_local.energy_used +=
                    (*frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity;
                validate_energy(new_local, &global.inventory, can_manage_reserves)
            } else {
                new_local.energy_used +=
                    (*frames as f32 * difficulty.resource_multiplier / 2.0).ceil() as Capacity;
                validate_energy(new_local, &global.inventory, can_manage_reserves)
            }
        }
        Requirement::GravitylessLavaFrames(frames) => {
            let varia = global.inventory.items[Item::Varia as usize];
            let mut new_local = local;
            if varia {
                new_local.energy_used +=
                    (*frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity
            } else {
                new_local.energy_used +=
                    (*frames as f32 * difficulty.resource_multiplier / 2.0).ceil() as Capacity
            }
            validate_energy(new_local, &global.inventory, can_manage_reserves)
        }
        Requirement::AcidFrames(frames) => {
            let mut new_local = local;
            new_local.energy_used += (*frames as f32 * difficulty.resource_multiplier * 1.5
                / suit_damage_factor(&global.inventory) as f32)
                .ceil() as Capacity;
            validate_energy(new_local, &global.inventory, can_manage_reserves)
        }
        Requirement::GravitylessAcidFrames(frames) => {
            let varia = global.inventory.items[Item::Varia as usize];
            let mut new_local = local;
            if varia {
                new_local.energy_used +=
                    (*frames as f32 * difficulty.resource_multiplier * 0.75).ceil() as Capacity;
            } else {
                new_local.energy_used +=
                    (*frames as f32 * difficulty.resource_multiplier * 1.5).ceil() as Capacity;
            }
            validate_energy(new_local, &global.inventory, can_manage_reserves)
        }
        Requirement::MetroidFrames(frames) => {
            let mut new_local = local;
            new_local.energy_used += (*frames as f32 * difficulty.resource_multiplier * 0.75
                / suit_damage_factor(&global.inventory) as f32)
                .ceil() as Capacity;
            validate_energy(new_local, &global.inventory, can_manage_reserves)
        }
        Requirement::Damage(base_energy) => {
            let mut new_local = local;
            let energy = base_energy / suit_damage_factor(&global.inventory);
            if energy >= global.inventory.max_energy
                && !difficulty.tech[game_data.pause_abuse_tech_idx]
            {
                None
            } else {
                new_local.energy_used += energy;
                validate_energy_no_auto_reserve(new_local, global, game_data, difficulty)
            }
        }
        Requirement::Missiles(count) => {
            let mut new_local = local;
            new_local.missiles_used += *count;
            validate_missiles(new_local, global)
        }
        Requirement::Supers(count) => {
            let mut new_local = local;
            new_local.supers_used += *count;
            validate_supers(new_local, global)
        }
        Requirement::PowerBombs(count) => {
            let mut new_local = local;
            new_local.power_bombs_used += *count;
            validate_power_bombs(new_local, global)
        }
        Requirement::GateGlitchLeniency { green, heated } => {
            apply_gate_glitch_leniency(local, global, *green, *heated, difficulty, game_data)
        }
        Requirement::HeatedDoorStuckLeniency { heat_frames } => {
            if !global.inventory.items[Item::Varia as usize] {
                let mut new_local = local;
                new_local.energy_used += (difficulty.door_stuck_leniency as f32
                    * difficulty.resource_multiplier
                    * *heat_frames as f32
                    / 4.0) as Capacity;
                validate_energy(new_local, &global.inventory, can_manage_reserves)
            } else {
                Some(local)
            }
        }
        Requirement::MissilesAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.inventory.max_missiles < *count {
                    None
                } else {
                    new_local.missiles_used = Capacity::max(new_local.missiles_used, *count);
                    Some(new_local)
                }
            } else {
                if global.inventory.max_missiles - local.missiles_used < *count {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::SupersAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.inventory.max_supers < *count {
                    None
                } else {
                    new_local.supers_used = Capacity::max(new_local.supers_used, *count);
                    Some(new_local)
                }
            } else {
                if global.inventory.max_supers - local.supers_used < *count {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::PowerBombsAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.inventory.max_power_bombs < *count {
                    None
                } else {
                    new_local.power_bombs_used = Capacity::max(new_local.power_bombs_used, *count);
                    Some(new_local)
                }
            } else {
                if global.inventory.max_power_bombs - local.power_bombs_used < *count {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::RegularEnergyAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.inventory.max_energy < *count {
                    None
                } else {
                    new_local.energy_used = Capacity::max(new_local.energy_used, *count);
                    Some(new_local)
                }
            } else {
                if global.inventory.max_energy - local.energy_used < *count {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::ReserveEnergyAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.inventory.max_reserves < *count {
                    None
                } else {
                    new_local.reserves_used = Capacity::max(new_local.reserves_used, *count);
                    Some(new_local)
                }
            } else {
                if global.inventory.max_reserves - local.reserves_used < *count {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::EnergyAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.inventory.max_energy + global.inventory.max_reserves < *count {
                    None
                } else {
                    if global.inventory.max_energy < *count {
                        new_local.energy_used = global.inventory.max_energy;
                        new_local.reserves_used = Capacity::max(
                            new_local.reserves_used,
                            *count - global.inventory.max_energy,
                        );
                        Some(new_local)
                    } else {
                        new_local.energy_used = Capacity::max(new_local.energy_used, *count);
                        Some(new_local)
                    }
                }
            } else {
                if global.inventory.max_reserves - local.reserves_used + global.inventory.max_energy
                    - local.energy_used
                    < *count
                {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::MissilesCapacity(count) => {
            if global.inventory.max_missiles >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::SupersCapacity(count) => {
            if global.inventory.max_supers >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::PowerBombsCapacity(count) => {
            if global.inventory.max_power_bombs >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::RegularEnergyCapacity(count) => {
            if global.inventory.max_energy >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::ReserveEnergyCapacity(count) => {
            if global.inventory.max_reserves >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::EnergyRefill(limit) => {
            let limit_reserves = max(0, *limit - global.inventory.max_energy);
            if reverse {
                let mut new_local = local;
                if local.energy_used < *limit {
                    new_local.energy_used = 0;
                }
                if local.reserves_used <= limit_reserves {
                    new_local.reserves_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.energy_used > global.inventory.max_energy - limit {
                    new_local.energy_used = max(0, global.inventory.max_energy - limit);
                }
                if local.reserves_used > global.inventory.max_reserves - limit_reserves {
                    new_local.reserves_used =
                        max(0, global.inventory.max_reserves - limit_reserves);
                }
                Some(new_local)
            }
        }
        Requirement::RegularEnergyRefill(limit) => {
            if reverse {
                let mut new_local = local;
                if local.energy_used < *limit {
                    new_local.energy_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.energy_used > global.inventory.max_energy - limit {
                    new_local.energy_used = max(0, global.inventory.max_energy - limit);
                }
                Some(new_local)
            }
        }
        Requirement::ReserveRefill(limit) => {
            if reverse {
                let mut new_local = local;
                if local.reserves_used <= *limit {
                    new_local.reserves_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.reserves_used > global.inventory.max_reserves - limit {
                    new_local.reserves_used = max(0, global.inventory.max_reserves - limit);
                }
                Some(new_local)
            }
        }
        Requirement::MissileRefill(limit) => {
            if reverse {
                let mut new_local = local;
                if local.missiles_used <= *limit {
                    new_local.missiles_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.missiles_used > global.inventory.max_missiles - limit {
                    new_local.missiles_used = max(0, global.inventory.max_missiles - limit);
                }
                Some(new_local)
            }
        }
        Requirement::SuperRefill(limit) => {
            if reverse {
                let mut new_local = local;
                if local.supers_used <= *limit {
                    new_local.supers_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.supers_used > global.inventory.max_supers - limit {
                    new_local.supers_used = max(0, global.inventory.max_supers - limit);
                }
                Some(new_local)
            }
        }
        Requirement::PowerBombRefill(limit) => {
            if reverse {
                let mut new_local = local;
                if local.power_bombs_used <= *limit {
                    new_local.power_bombs_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.power_bombs_used > global.inventory.max_power_bombs - limit {
                    new_local.power_bombs_used = max(0, global.inventory.max_power_bombs - limit);
                }
                Some(new_local)
            }
        }
        Requirement::AmmoStationRefill => {
            let mut new_local = local;
            new_local.missiles_used = 0;
            if !settings.other_settings.ultra_low_qol {
                new_local.supers_used = 0;
                new_local.power_bombs_used = 0;
            }
            Some(new_local)
        }
        Requirement::AmmoStationRefillAll => {
            if settings.other_settings.ultra_low_qol {
                None
            } else {
                Some(local)
            }
        }
        Requirement::SupersDoubleDamageMotherBrain => {
            if settings.quality_of_life_settings.supers_double {
                Some(local)
            } else {
                None
            }
        }
        Requirement::ShinesparksCostEnergy => {
            if settings.other_settings.energy_free_shinesparks {
                None
            } else {
                Some(local)
            }
        }
        Requirement::RegularEnergyDrain(count) => {
            if reverse {
                let mut new_local = local;
                let amt = Capacity::max(0, local.energy_used - count + 1);
                new_local.reserves_used += amt;
                new_local.energy_used -= amt;
                if new_local.reserves_used > global.inventory.max_reserves {
                    None
                } else {
                    Some(new_local)
                }
            } else {
                let mut new_local = local;
                new_local.energy_used =
                    Capacity::max(local.energy_used, global.inventory.max_energy - count);
                Some(new_local)
            }
        }
        Requirement::ReserveEnergyDrain(count) => {
            if reverse {
                if local.reserves_used > *count {
                    None
                } else {
                    Some(local)
                }
            } else {
                let mut new_local = local;
                // TODO: Drained reserve energy could potentially be transferred into regular energy, but it wouldn't
                // be consistent with how "resourceAtMost" is currently defined.
                new_local.reserves_used =
                    Capacity::max(local.reserves_used, global.inventory.max_reserves - count);
                Some(new_local)
            }
        }
        Requirement::ReserveTrigger {
            min_reserve_energy,
            max_reserve_energy,
            heated,
        } => {
            if reverse {
                if local.reserves_used > 0 {
                    None
                } else {
                    let mut new_local = local;
                    new_local.energy_used = 0;
                    let energy_needed = if *heated {
                        (local.energy_used * 4 + 2) / 3
                    } else {
                        local.energy_used
                    };
                    new_local.reserves_used = max(energy_needed + 1, *min_reserve_energy);
                    if new_local.reserves_used > *max_reserve_energy
                        || new_local.reserves_used > global.inventory.max_reserves
                    {
                        None
                    } else {
                        Some(new_local)
                    }
                }
            } else {
                let reserve_energy = min(
                    global.inventory.max_reserves - local.reserves_used,
                    *max_reserve_energy,
                );
                let usable_reserve_energy = if *heated {
                    reserve_energy * 3 / 4
                } else {
                    reserve_energy
                };
                if reserve_energy >= *min_reserve_energy {
                    let mut new_local = local;
                    new_local.reserves_used = global.inventory.max_reserves;
                    new_local.energy_used =
                        max(0, global.inventory.max_energy - usable_reserve_energy);
                    Some(new_local)
                } else {
                    None
                }
            }
        }
        Requirement::EnemyKill { count, vul } => {
            apply_enemy_kill_requirement(global, local, *count, vul)
        }
        Requirement::PhantoonFight {} => apply_phantoon_requirement(
            &global.inventory,
            local,
            difficulty.phantoon_proficiency,
            can_manage_reserves,
        ),
        Requirement::DraygonFight {
            can_be_very_patient_tech_idx: can_be_very_patient_tech_id,
        } => apply_draygon_requirement(
            &global.inventory,
            local,
            difficulty.draygon_proficiency,
            can_manage_reserves,
            difficulty.tech[*can_be_very_patient_tech_id],
        ),
        Requirement::RidleyFight {
            can_be_very_patient_tech_idx: can_be_very_patient_tech_id,
        } => apply_ridley_requirement(
            &global.inventory,
            local,
            difficulty.ridley_proficiency,
            can_manage_reserves,
            difficulty.tech[*can_be_very_patient_tech_id],
        ),
        Requirement::BotwoonFight { second_phase } => apply_botwoon_requirement(
            &global.inventory,
            local,
            difficulty.botwoon_proficiency,
            *second_phase,
            can_manage_reserves,
        ),
        Requirement::MotherBrain2Fight {
            can_be_very_patient_tech_id,
            r_mode,
        } => {
            if settings.quality_of_life_settings.mother_brain_fight == MotherBrainFight::Skip {
                return Some(local);
            }
            apply_mother_brain_2_requirement(
                &global.inventory,
                local,
                difficulty.mother_brain_proficiency,
                settings.quality_of_life_settings.supers_double,
                can_manage_reserves,
                difficulty.tech[*can_be_very_patient_tech_id],
                *r_mode,
            )
        }
        Requirement::SpeedBall { used_tiles, heated } => {
            if !difficulty.tech[game_data.speed_ball_tech_idx]
                || !global.inventory.items[Item::Morph as usize]
            {
                None
            } else {
                let used_tiles = used_tiles.get();
                let tiles_limit = if *heated && !global.inventory.items[Item::Varia as usize] {
                    get_heated_speedball_tiles(difficulty)
                } else {
                    difficulty.speed_ball_tiles
                };
                if global.inventory.items[Item::SpeedBooster as usize] && used_tiles >= tiles_limit
                {
                    Some(local)
                } else {
                    None
                }
            }
        }
        Requirement::GetBlueSpeed { used_tiles, heated } => {
            let used_tiles = used_tiles.get();
            let tiles_limit = if *heated && !global.inventory.items[Item::Varia as usize] {
                difficulty.heated_shine_charge_tiles
            } else {
                difficulty.shine_charge_tiles
            };
            if global.inventory.items[Item::SpeedBooster as usize] && used_tiles >= tiles_limit {
                Some(local)
            } else {
                None
            }
        }
        Requirement::ShineCharge { used_tiles, heated } => {
            let used_tiles = used_tiles.get();
            let tiles_limit = if *heated && !global.inventory.items[Item::Varia as usize] {
                difficulty.heated_shine_charge_tiles
            } else {
                difficulty.shine_charge_tiles
            };
            if global.inventory.items[Item::SpeedBooster as usize] && used_tiles >= tiles_limit {
                let mut new_local = local;
                if reverse {
                    new_local.shinecharge_frames_remaining = 0;
                } else {
                    new_local.shinecharge_frames_remaining =
                        180 - difficulty.shinecharge_leniency_frames;
                }
                Some(new_local)
            } else {
                None
            }
        }
        Requirement::ShineChargeFrames(frames) => {
            let mut new_local = local;
            if reverse {
                new_local.shinecharge_frames_remaining += frames;
                if new_local.shinecharge_frames_remaining
                    <= 180 - difficulty.shinecharge_leniency_frames
                {
                    Some(new_local)
                } else {
                    None
                }
            } else {
                new_local.shinecharge_frames_remaining -= frames;
                if new_local.shinecharge_frames_remaining >= 0 {
                    Some(new_local)
                } else {
                    None
                }
            }
        }
        Requirement::Shinespark {
            frames,
            excess_frames,
            shinespark_tech_idx: shinespark_tech_id,
        } => {
            if difficulty.tech[*shinespark_tech_id] {
                let mut new_local = local;
                if settings.other_settings.energy_free_shinesparks {
                    return Some(new_local);
                }
                if reverse {
                    if new_local.energy_used <= 28 {
                        new_local.energy_used = 28 + frames - excess_frames;
                    } else {
                        new_local.energy_used += frames;
                    }
                    validate_energy_no_auto_reserve(new_local, global, game_data, difficulty)
                } else {
                    new_local.energy_used += frames - excess_frames + 28;
                    if let Some(mut new_local) =
                        validate_energy_no_auto_reserve(new_local, global, game_data, difficulty)
                    {
                        let energy_remaining =
                            global.inventory.max_energy - new_local.energy_used - 1;
                        new_local.energy_used += std::cmp::min(*excess_frames, energy_remaining);
                        new_local.energy_used -= 28;
                        Some(new_local)
                    } else {
                        None
                    }
                }
            } else {
                None
            }
        }
        Requirement::DoorUnlocked { room_id, node_id } => {
            if let Some(locked_door_idx) = locked_door_data
                .locked_door_node_map
                .get(&(*room_id, *node_id))
            {
                if global.doors_unlocked[*locked_door_idx] {
                    Some(local)
                } else {
                    None
                }
            } else {
                Some(local)
            }
        }
        Requirement::DoorType {
            room_id,
            node_id,
            door_type,
        } => {
            let actual_door_type = if let Some(locked_door_idx) = locked_door_data
                .locked_door_node_map
                .get(&(*room_id, *node_id))
            {
                locked_door_data.locked_doors[*locked_door_idx].door_type
            } else {
                DoorType::Blue
            };
            if *door_type == actual_door_type {
                Some(local)
            } else {
                None
            }
        }
        Requirement::UnlockDoor {
            room_id,
            node_id,
            requirement_red,
            requirement_green,
            requirement_yellow,
            requirement_charge,
        } => {
            if let Some(locked_door_idx) = locked_door_data
                .locked_door_node_map
                .get(&(*room_id, *node_id))
            {
                let door_type = locked_door_data.locked_doors[*locked_door_idx].door_type;
                if global.doors_unlocked[*locked_door_idx] {
                    return Some(local);
                }
                match door_type {
                    DoorType::Blue => Some(local),
                    DoorType::Red => apply_requirement(
                        requirement_red,
                        global,
                        local,
                        reverse,
                        settings,
                        difficulty,
                        game_data,
                        locked_door_data,
                        objectives,
                    ),
                    DoorType::Green => apply_requirement(
                        requirement_green,
                        global,
                        local,
                        reverse,
                        settings,
                        difficulty,
                        game_data,
                        locked_door_data,
                        objectives,
                    ),
                    DoorType::Yellow => apply_requirement(
                        requirement_yellow,
                        global,
                        local,
                        reverse,
                        settings,
                        difficulty,
                        game_data,
                        locked_door_data,
                        objectives,
                    ),
                    DoorType::Beam(beam) => {
                        if has_beam(beam, &global.inventory) {
                            if let BeamType::Charge = beam {
                                apply_requirement(
                                    requirement_charge,
                                    global,
                                    local,
                                    reverse,
                                    settings,
                                    difficulty,
                                    game_data,
                                    locked_door_data,
                                    objectives,
                                )
                            } else {
                                Some(local)
                            }
                        } else {
                            None
                        }
                    }
                    DoorType::Gray => {
                        panic!("Unexpected gray door while processing Requirement::UnlockDoor")
                    }
                }
            } else {
                Some(local)
            }
        }
        Requirement::EscapeMorphLocation => {
            if settings.map_layout == "Vanilla" {
                Some(local)
            } else {
                None
            }
        }
        Requirement::And(reqs) => {
            let mut new_local = local;
            if reverse {
                for req in reqs.into_iter().rev() {
                    new_local = apply_requirement(
                        req,
                        global,
                        new_local,
                        reverse,
                        settings,
                        difficulty,
                        game_data,
                        locked_door_data,
                        objectives,
                    )?;
                }
            } else {
                for req in reqs {
                    new_local = apply_requirement(
                        req,
                        global,
                        new_local,
                        reverse,
                        settings,
                        difficulty,
                        game_data,
                        locked_door_data,
                        objectives,
                    )?;
                }
            }
            Some(new_local)
        }
        Requirement::Or(reqs) => {
            let mut best_local = None;
            let mut best_cost = [f32::INFINITY; NUM_COST_METRICS];
            for req in reqs {
                if let Some(new_local) = apply_requirement(
                    req,
                    global,
                    local,
                    reverse,
                    settings,
                    difficulty,
                    game_data,
                    locked_door_data,
                    objectives,
                ) {
                    let cost = compute_cost(new_local, &global.inventory);
                    // TODO: Maybe do something better than just using the first cost metric.
                    if cost[0] < best_cost[0] {
                        best_cost = cost;
                        best_local = Some(new_local);
                    }
                }
            }
            best_local
        }
    }
}

pub fn is_reachable_state(local: LocalState) -> bool {
    local.energy_used != IMPOSSIBLE_LOCAL_STATE.energy_used
}

pub fn is_bireachable_state(
    global: &GlobalState,
    forward: LocalState,
    reverse: LocalState,
) -> bool {
    if forward.reserves_used + reverse.reserves_used > global.inventory.max_reserves {
        return false;
    }
    let forward_total_energy_used = forward.energy_used + forward.reserves_used;
    let reverse_total_energy_used = reverse.energy_used + reverse.reserves_used;
    let max_total_energy = global.inventory.max_energy + global.inventory.max_reserves;
    if forward_total_energy_used + reverse_total_energy_used >= max_total_energy {
        return false;
    }
    if forward.missiles_used + reverse.missiles_used > global.inventory.max_missiles {
        return false;
    }
    if forward.supers_used + reverse.supers_used > global.inventory.max_supers {
        return false;
    }
    if forward.power_bombs_used + reverse.power_bombs_used > global.inventory.max_power_bombs {
        return false;
    }
    if reverse.shinecharge_frames_remaining > forward.shinecharge_frames_remaining {
        return false;
    }
    true
}

// If the given vertex is bireachable, returns a pair of cost metric indexes (between 0 and NUM_COST_METRICS),
// indicating which forward route and backward route, respectively, combine to give a successful full route.
// Otherwise returns None.
pub fn get_bireachable_idxs(
    global: &GlobalState,
    vertex_id: usize,
    forward: &TraverseResult,
    reverse: &TraverseResult,
) -> Option<(usize, usize)> {
    for forward_cost_idx in 0..NUM_COST_METRICS {
        for reverse_cost_idx in 0..NUM_COST_METRICS {
            let forward_state = forward.local_states[vertex_id][forward_cost_idx];
            let reverse_state = reverse.local_states[vertex_id][reverse_cost_idx];
            if is_bireachable_state(global, forward_state, reverse_state) {
                // A valid combination of forward & return routes has been found.
                return Some((forward_cost_idx, reverse_cost_idx));
            }
        }
    }
    None
}

// If the given vertex is reachable, returns a cost metric index (between 0 and NUM_COST_METRICS),
// indicating a forward route. Otherwise returns None.
pub fn get_one_way_reachable_idx(vertex_id: usize, forward: &TraverseResult) -> Option<usize> {
    for forward_cost_idx in 0..NUM_COST_METRICS {
        let forward_state = forward.local_states[vertex_id][forward_cost_idx];
        if is_reachable_state(forward_state) {
            return Some(forward_cost_idx);
        }
    }
    None
}

pub type StepTrailId = i32;

#[derive(Clone)]
pub struct StepTrail {
    pub prev_trail_id: StepTrailId,
    pub link_idx: LinkIdx,
}

#[derive(Clone)]
pub struct TraverseResult {
    pub local_states: Vec<[LocalState; NUM_COST_METRICS]>,
    pub cost: Vec<[f32; NUM_COST_METRICS]>,
    pub step_trails: Vec<StepTrail>,
    pub start_trail_ids: Vec<[StepTrailId; NUM_COST_METRICS]>,
}

pub fn traverse(
    base_links_data: &LinksDataGroup,
    seed_links_data: &LinksDataGroup,
    init_opt: Option<TraverseResult>,
    global: &GlobalState,
    init_local: LocalState,
    num_vertices: usize,
    start_vertex_id: usize,
    reverse: bool,
    settings: &RandomizerSettings,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
    locked_door_data: &LockedDoorData,
    objectives: &[Objective],
) -> TraverseResult {
    let mut modified_vertices: HashMap<usize, [bool; NUM_COST_METRICS]> = HashMap::new();
    let mut result: TraverseResult;

    if let Some(init) = init_opt {
        for (v, cost) in init.cost.iter().enumerate() {
            let valid = cost.map(|x| f32::is_finite(x));
            if valid.iter().any(|&x| x) {
                modified_vertices.insert(v, valid);
            }
        }
        result = init;
    } else {
        result = TraverseResult {
            local_states: vec![[IMPOSSIBLE_LOCAL_STATE; NUM_COST_METRICS]; num_vertices],
            cost: vec![[f32::INFINITY; NUM_COST_METRICS]; num_vertices],
            step_trails: Vec::with_capacity(num_vertices * 10),
            start_trail_ids: vec![[-1; NUM_COST_METRICS]; num_vertices],
        };
        let first_metric = {
            let mut x = [false; NUM_COST_METRICS];
            x[0] = true;
            x
        };
        result.local_states[start_vertex_id] = [init_local; NUM_COST_METRICS];
        result.start_trail_ids[start_vertex_id] = [-1; NUM_COST_METRICS];
        result.cost[start_vertex_id] = compute_cost(init_local, &global.inventory);
        modified_vertices.insert(start_vertex_id, first_metric);
    }

    let base_links_by_src: &Vec<Vec<(LinkIdx, Link)>> = if reverse {
        &base_links_data.links_by_dst
    } else {
        &base_links_data.links_by_src
    };
    let seed_links_by_src: &Vec<Vec<(LinkIdx, Link)>> = if reverse {
        &seed_links_data.links_by_dst
    } else {
        &seed_links_data.links_by_src
    };

    while modified_vertices.len() > 0 {
        let mut new_modified_vertices: HashMap<usize, [bool; NUM_COST_METRICS]> = HashMap::new();
        for (&src_id, &modified_costs) in &modified_vertices {
            let src_local_state_arr = result.local_states[src_id];
            let src_trail_id_arr = result.start_trail_ids[src_id];
            for src_cost_idx in 0..NUM_COST_METRICS {
                if !modified_costs[src_cost_idx] {
                    continue;
                }
                let src_trail_id = src_trail_id_arr[src_cost_idx];
                let src_local_state = src_local_state_arr[src_cost_idx];
                let all_src_links = base_links_by_src[src_id]
                    .iter()
                    .chain(seed_links_by_src[src_id].iter());
                for &(link_idx, ref link) in all_src_links {
                    let dst_id = link.to_vertex_id;
                    let dst_old_cost_arr = result.cost[dst_id];
                    if let Some(dst_new_local_state) = apply_link(
                        &link,
                        global,
                        src_local_state,
                        reverse,
                        settings,
                        difficulty,
                        game_data,
                        locked_door_data,
                        objectives,
                    ) {
                        let dst_new_cost_arr = compute_cost(dst_new_local_state, &global.inventory);

                        let new_step_trail = StepTrail {
                            prev_trail_id: src_trail_id,
                            link_idx,
                        };
                        let new_trail_id = result.step_trails.len() as StepTrailId;
                        let mut any_improvement: bool = false;
                        let mut improved_arr: [bool; NUM_COST_METRICS] = new_modified_vertices
                            .get(&dst_id)
                            .map(|x| *x)
                            .unwrap_or([false; NUM_COST_METRICS]);
                        for dst_cost_idx in 0..NUM_COST_METRICS {
                            if dst_new_cost_arr[dst_cost_idx] < dst_old_cost_arr[dst_cost_idx] {
                                result.local_states[dst_id][dst_cost_idx] = dst_new_local_state;
                                result.start_trail_ids[dst_id][dst_cost_idx] = new_trail_id;
                                result.cost[dst_id][dst_cost_idx] = dst_new_cost_arr[dst_cost_idx];
                                improved_arr[dst_cost_idx] = true;
                                any_improvement = true;
                            }
                        }
                        if any_improvement {
                            let check_value = |name: &'static str, v: Capacity| {
                                if v < 0 {
                                    panic!("Resource {} is negative, with value {}: old_state={:?}, new_state={:?}, link={:?}", 
                                        name, v, src_local_state, dst_new_local_state, link);
                                }
                            };
                            check_value("energy", dst_new_local_state.energy_used);
                            check_value("reserves", dst_new_local_state.reserves_used);
                            check_value("missiles", dst_new_local_state.missiles_used);
                            check_value("supers", dst_new_local_state.supers_used);
                            check_value("power_bombs", dst_new_local_state.power_bombs_used);
                            check_value(
                                "shinecharge_frames",
                                dst_new_local_state.shinecharge_frames_remaining,
                            );
                            new_modified_vertices.insert(dst_id, improved_arr);
                            result.step_trails.push(new_step_trail);
                        }
                    }
                }
            }
        }
        modified_vertices = new_modified_vertices;
    }

    result
}

pub fn get_spoiler_route(
    traverse_result: &TraverseResult,
    vertex_id: usize,
    cost_idx: usize,
) -> Vec<LinkIdx> {
    let mut trail_id = traverse_result.start_trail_ids[vertex_id][cost_idx];
    let mut steps: Vec<LinkIdx> = Vec::new();
    while trail_id != -1 {
        let step_trail = &traverse_result.step_trails[trail_id as usize];
        steps.push(step_trail.link_idx);
        trail_id = step_trail.prev_trail_id;
    }
    steps.reverse();
    steps
}
