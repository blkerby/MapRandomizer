use std::cmp::{max, min};

use hashbrown::HashMap;
use log::debug;
use serde::{Deserialize, Serialize};

use crate::{
    randomize::{DifficultyConfig, LockedDoor},
    settings::{MotherBrainFight, Objective, RandomizerSettings, WallJump},
};
use maprando_game::{
    BeamType, Capacity, DoorType, EnemyDrop, EnemyVulnerabilities, GameData, Item, Link, LinkIdx,
    LinksDataGroup, NodeId, Requirement, RoomId, TECH_ID_CAN_BE_EXTREMELY_PATIENT,
    TECH_ID_CAN_BE_PATIENT, TECH_ID_CAN_BE_VERY_PATIENT, TECH_ID_CAN_SUITLESS_LAVA_DIVE, VertexId,
};
use maprando_logic::{
    GlobalState, Inventory, LocalState,
    boss_requirements::{
        apply_botwoon_requirement, apply_draygon_requirement, apply_mother_brain_2_requirement,
        apply_phantoon_requirement, apply_ridley_requirement,
    },
    helpers::{suit_damage_factor, validate_energy},
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
    let mut missiles_used = 0;

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
        missiles_used += missiles_to_use_per_enemy * count;
    }

    // Then use Supers:
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

    // If the enemy would be overkilled, refund some of the missile shots, if applicable:
    if vul.missile_damage > 0 {
        let missiles_overkill = -hp / vul.missile_damage;
        missiles_used = max(0, missiles_used - missiles_overkill * count);
    }
    local.missiles_used += missiles_used;

    if hp <= 0 { Some(local) } else { None }
}

pub const IMPOSSIBLE_LOCAL_STATE: LocalState = LocalState {
    energy_used: 0x3FFF,
    reserves_used: 0x3FFF,
    missiles_used: 0x3FFF,
    supers_used: 0x3FFF,
    power_bombs_used: 0x3FFF,
    shinecharge_frames_remaining: 0x3FFF,
    cycle_frames: 0x3FFF,
    farm_baseline_energy_used: 0x3FFF,
    farm_baseline_reserves_used: 0x3FFF,
    farm_baseline_missiles_used: 0x3FFF,
    farm_baseline_supers_used: 0x3FFF,
    farm_baseline_power_bombs_used: 0x3FFF,
};

pub const NUM_COST_METRICS: usize = 2;

fn compute_cost(
    local: LocalState,
    inventory: &Inventory,
    reverse: bool,
) -> [f32; NUM_COST_METRICS] {
    let eps = 1e-15;
    let energy_cost = (local.energy_used as f32) / (inventory.max_energy as f32 + eps);
    let reserve_cost = (local.reserves_used as f32) / (inventory.max_reserves as f32 + eps);
    let missiles_cost = (local.missiles_used as f32) / (inventory.max_missiles as f32 + eps);
    let supers_cost = (local.supers_used as f32) / (inventory.max_supers as f32 + eps);
    let power_bombs_cost =
        (local.power_bombs_used as f32) / (inventory.max_power_bombs as f32 + eps);
    let mut shinecharge_cost = -(local.shinecharge_frames_remaining as f32) / 180.0;
    if reverse {
        shinecharge_cost = -shinecharge_cost;
    }
    let cycle_frames_cost = (local.cycle_frames as f32) * 0.0001;

    let energy_sensitive_cost_metric = 100.0 * (energy_cost + reserve_cost)
        + missiles_cost
        + supers_cost
        + power_bombs_cost
        + shinecharge_cost
        + cycle_frames_cost;
    let ammo_sensitive_cost_metric = energy_cost
        + reserve_cost
        + 100.0
            * (missiles_cost
                + supers_cost
                + power_bombs_cost
                + shinecharge_cost
                + cycle_frames_cost);
    [energy_sensitive_cost_metric, ammo_sensitive_cost_metric]
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
        local = validate_energy(
            local,
            &global.inventory,
            difficulty.tech[game_data.manage_reserves_tech_idx],
        )?;
    }
    if green {
        local.supers_used += difficulty.gate_glitch_leniency;
        validate_supers(local, global)
    } else {
        let missiles_available = global.inventory.max_missiles - local.missiles_used;
        if missiles_available >= difficulty.gate_glitch_leniency {
            local.missiles_used += difficulty.gate_glitch_leniency;
            validate_missiles(local, global)
        } else {
            local.missiles_used = global.inventory.max_missiles;
            local.supers_used += difficulty.gate_glitch_leniency - missiles_available;
            validate_supers(local, global)
        }
    }
}

fn is_mother_brain_barrier_clear(
    global: &GlobalState,
    _difficulty: &DifficultyConfig,
    objectives: &[Objective],
    game_data: &GameData,
    obj_id: usize,
) -> bool {
    if objectives.len() > 4 {
        for obj in objectives {
            let flag_name = obj.get_flag_name();
            let flag_idx = game_data.flag_isv.index_by_key[flag_name];
            if !global.flags[flag_idx] {
                return false;
            }
        }
        true
    } else if let Some(obj) = objectives.get(obj_id) {
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
    simple: bool,
) -> Option<LocalState> {
    let varia = global.inventory.items[Item::Varia as usize];
    let mut new_local = local;
    if varia {
        Some(new_local)
    } else if !difficulty.tech[game_data.heat_run_tech_idx] {
        None
    } else {
        if simple {
            new_local.energy_used += (frames as f32 / 4.0).ceil() as Capacity;
        } else {
            new_local.energy_used +=
                (frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity;
        }
        validate_energy(
            new_local,
            &global.inventory,
            difficulty.tech[game_data.manage_reserves_tech_idx],
        )
    }
}

fn get_enemy_drop_energy_value(
    drop: &EnemyDrop,
    local: LocalState,
    reverse: bool,
    buffed_drops: bool,
) -> Capacity {
    let p_nothing = drop.nothing_weight.get();
    let mut p_small = drop.small_energy_weight.get();
    let mut p_large = drop.large_energy_weight.get();
    let mut p_missile = drop.missile_weight.get();
    let p_tier1 = p_nothing + p_small + p_large + p_missile;
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

fn get_enemy_drop_values(
    drop: &EnemyDrop,
    full_energy: bool,
    full_missiles: bool,
    full_supers: bool,
    full_power_bombs: bool,
    buffed_drops: bool,
) -> [f32; 4] {
    let p_nothing = drop.nothing_weight.get();
    let mut p_small = drop.small_energy_weight.get();
    let mut p_large = drop.large_energy_weight.get();
    let mut p_missile = drop.missile_weight.get();
    let p_tier1 = p_nothing + p_small + p_large + p_missile;
    let rel_small = p_small / p_tier1;
    let rel_large = p_large / p_tier1;
    let rel_missile = p_missile / p_tier1;
    let mut p_super = drop.super_weight.get();
    let mut p_pb = drop.power_bomb_weight.get();

    if full_power_bombs {
        p_small += p_pb * rel_small;
        p_large += p_pb * rel_large;
        p_missile += p_pb * rel_missile;
        p_pb = 0.0;
    }
    if full_supers {
        p_small += p_super * rel_small;
        p_large += p_super * rel_large;
        p_missile += p_super * rel_missile;
        p_super = 0.0;
    }
    if full_missiles && (p_small + p_large > 0.0) {
        p_small += p_missile * p_small / (p_small + p_large);
        p_large += p_missile * p_large / (p_small + p_large);
        p_missile = 0.0;
    } else if full_energy && p_missile > 0.0_ {
        p_missile += p_small + p_large;
        p_small = 0.0;
        p_large = 0.0;
    }
    let expected_energy = p_small * if buffed_drops { 10.0 } else { 5.0 } + p_large * 20.0;
    let expected_missiles = p_missile * 2.0;
    let expected_supers = p_super;
    let expected_pbs = p_pb * if buffed_drops { 2.0 } else { 1.0 };
    let count = drop.count as f32;
    [
        expected_energy * count,
        expected_missiles * count,
        expected_supers * count,
        expected_pbs * count,
    ]
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
    } else if !difficulty.tech[game_data.heat_run_tech_idx] {
        None
    } else {
        let mut total_drop_value = 0;
        for drop in drops {
            total_drop_value += get_enemy_drop_energy_value(
                drop,
                local,
                reverse,
                settings.quality_of_life_settings.buffed_drops,
            )
        }
        let heat_energy = (frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity;
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

fn apply_lava_frames_with_energy_drops(
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
    let gravity = global.inventory.items[Item::Gravity as usize];
    let mut new_local = local;
    if gravity && varia {
        Some(new_local)
    } else if !difficulty.tech[game_data.tech_isv.index_by_key[&TECH_ID_CAN_SUITLESS_LAVA_DIVE]] {
        None
    } else {
        let mut total_drop_value = 0;
        for drop in drops {
            total_drop_value += get_enemy_drop_energy_value(
                drop,
                local,
                reverse,
                settings.quality_of_life_settings.buffed_drops,
            )
        }
        let lava_energy = if gravity || varia {
            (frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity
        } else {
            (frames as f32 * difficulty.resource_multiplier / 2.0).ceil() as Capacity
        };
        total_drop_value = Capacity::min(total_drop_value, lava_energy);
        new_local.energy_used += lava_energy;
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
    door_map: &HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
    locked_door_data: &LockedDoorData,
    objectives: &[Objective],
) -> Option<LocalState> {
    if reverse {
        if !link.end_with_shinecharge && local.shinecharge_frames_remaining > 0 {
            return None;
        }
    } else if link.start_with_shinecharge && local.shinecharge_frames_remaining == 0 {
        return None;
    }
    let new_local = apply_requirement(
        &link.requirement,
        global,
        local,
        reverse,
        settings,
        difficulty,
        game_data,
        door_map,
        locked_door_data,
        objectives,
    );
    if let Some(mut new_local) = new_local {
        if reverse {
            if !link.start_with_shinecharge {
                new_local.shinecharge_frames_remaining = 0;
            }
        } else if !link.end_with_shinecharge {
            new_local.shinecharge_frames_remaining = 0;
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
    door_map: &HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
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
            door_map,
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
                    door_map,
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
                    door_map,
                    locked_door_data,
                    objectives,
                );
            }
        }
        _ => {}
    }
}

fn apply_missiles_available_req(
    local: LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> Option<LocalState> {
    if reverse {
        let mut new_local = local;
        if global.inventory.max_missiles < count {
            None
        } else {
            new_local.missiles_used = Capacity::max(new_local.missiles_used, count);
            Some(new_local)
        }
    } else if global.inventory.max_missiles - local.missiles_used < count {
        None
    } else {
        Some(local)
    }
}

fn apply_supers_available_req(
    local: LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> Option<LocalState> {
    if reverse {
        let mut new_local = local;
        if global.inventory.max_supers < count {
            None
        } else {
            new_local.supers_used = Capacity::max(new_local.supers_used, count);
            Some(new_local)
        }
    } else if global.inventory.max_supers - local.supers_used < count {
        None
    } else {
        Some(local)
    }
}

fn apply_power_bombs_available_req(
    local: LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> Option<LocalState> {
    if reverse {
        let mut new_local = local;
        if global.inventory.max_power_bombs < count {
            None
        } else {
            new_local.power_bombs_used = Capacity::max(new_local.power_bombs_used, count);
            Some(new_local)
        }
    } else if global.inventory.max_power_bombs - local.power_bombs_used < count {
        None
    } else {
        Some(local)
    }
}

fn apply_regular_energy_available_req(
    local: LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> Option<LocalState> {
    if reverse {
        let mut new_local = local;
        if global.inventory.max_energy < count {
            None
        } else {
            new_local.energy_used = Capacity::max(new_local.energy_used, count);
            Some(new_local)
        }
    } else if global.inventory.max_energy - local.energy_used < count {
        None
    } else {
        Some(local)
    }
}

fn apply_reserve_energy_available_req(
    local: LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> Option<LocalState> {
    if reverse {
        let mut new_local = local;
        if global.inventory.max_reserves < count {
            None
        } else {
            new_local.reserves_used = Capacity::max(new_local.reserves_used, count);
            Some(new_local)
        }
    } else if global.inventory.max_reserves - local.reserves_used < count {
        None
    } else {
        Some(local)
    }
}

fn apply_energy_available_req(
    local: LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> Option<LocalState> {
    if reverse {
        let mut new_local = local;
        if global.inventory.max_energy + global.inventory.max_reserves < count {
            None
        } else if global.inventory.max_energy < count {
            new_local.energy_used = global.inventory.max_energy;
            new_local.reserves_used =
                Capacity::max(new_local.reserves_used, count - global.inventory.max_energy);
            Some(new_local)
        } else {
            new_local.energy_used = Capacity::max(new_local.energy_used, count);
            Some(new_local)
        }
    } else if global.inventory.max_reserves - local.reserves_used + global.inventory.max_energy
        - local.energy_used
        < count
    {
        None
    } else {
        Some(local)
    }
}

pub fn get_total_enemy_drop_values(
    drops: &[EnemyDrop],
    full_energy: bool,
    full_missiles: bool,
    full_supers: bool,
    full_power_bombs: bool,
    buffed_drops: bool,
) -> [f32; 4] {
    let mut total_values = [0.0; 4];
    for drop in drops {
        let drop_values = get_enemy_drop_values(
            drop,
            full_energy,
            full_missiles,
            full_supers,
            full_power_bombs,
            buffed_drops,
        );
        for i in 0..4 {
            total_values[i] += drop_values[i];
        }
    }
    total_values
}

pub fn apply_farm_requirement(
    req: &Requirement,
    drops: &[EnemyDrop],
    global: &GlobalState,
    local: LocalState,
    reverse: bool,
    full_energy: bool,
    full_missiles: bool,
    full_supers: bool,
    full_power_bombs: bool,
    settings: &RandomizerSettings,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
    door_map: &HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
    locked_door_data: &LockedDoorData,
    objectives: &[Objective],
) -> Option<LocalState> {
    if !reverse && (full_energy || full_missiles || full_supers || full_power_bombs) {
        return None;
    }

    let mut start_local = local;
    // An initial cycle_frames of 1 is used to mark this as a farming strat, as this can affect
    // the processing of some requirements (currently just ResetRoom).
    start_local.cycle_frames = 1;
    let end_local_result = apply_requirement(
        req,
        global,
        start_local,
        reverse,
        settings,
        difficulty,
        game_data,
        door_map,
        locked_door_data,
        objectives,
    );
    if let Some(end_local) = end_local_result {
        if end_local.cycle_frames < 100 {
            panic!(
                "bad farm: expected cycle_frames >= 100: end_local={end_local:#?},\n req={req:#?}"
            );
        }
        let cycle_frames = (end_local.cycle_frames - 1) as f32;
        let cycle_energy = (end_local.energy_used + end_local.reserves_used
            - local.energy_used
            - local.reserves_used) as f32;
        let cycle_missiles = (end_local.missiles_used - local.missiles_used) as f32;
        let cycle_supers = (end_local.supers_used - local.supers_used) as f32;
        let cycle_pbs = (end_local.power_bombs_used - local.power_bombs_used) as f32;
        let mut patience_frames = 5400.0;
        if difficulty.tech[game_data.tech_isv.index_by_key[&TECH_ID_CAN_BE_EXTREMELY_PATIENT]] {
            patience_frames *= 8.0;
        } else if difficulty.tech[game_data.tech_isv.index_by_key[&TECH_ID_CAN_BE_VERY_PATIENT]] {
            patience_frames *= 4.0;
        } else if difficulty.tech[game_data.tech_isv.index_by_key[&TECH_ID_CAN_BE_PATIENT]] {
            patience_frames *= 2.0;
        }
        let mut num_cycles = (patience_frames / cycle_frames).floor() as i32;

        let mut new_local = local;
        if new_local.farm_baseline_energy_used < new_local.energy_used {
            new_local.farm_baseline_energy_used = new_local.energy_used;
        }
        if new_local.farm_baseline_reserves_used < new_local.reserves_used {
            new_local.farm_baseline_reserves_used = new_local.reserves_used;
        }
        if new_local.farm_baseline_missiles_used < new_local.missiles_used {
            new_local.farm_baseline_missiles_used = new_local.missiles_used;
        }
        if new_local.farm_baseline_supers_used < new_local.supers_used {
            new_local.farm_baseline_supers_used = new_local.supers_used;
        }
        if new_local.farm_baseline_power_bombs_used < new_local.power_bombs_used {
            new_local.farm_baseline_power_bombs_used = new_local.power_bombs_used;
        }
        new_local.energy_used = new_local.farm_baseline_energy_used;
        new_local.reserves_used = new_local.farm_baseline_reserves_used;
        new_local.missiles_used = new_local.farm_baseline_missiles_used;
        new_local.supers_used = new_local.farm_baseline_supers_used;
        new_local.power_bombs_used = new_local.farm_baseline_power_bombs_used;

        if reverse {
            // Handling reverse traversals is tricky because in the reverse traversal we don't know
            // the current resource levels, so we don't know if they can be full (affecting the drop
            // rates of other resource types). We address this by constructing variants of farm strats
            // (as separate Links) with different requirements on which combinations will be filled to full.
            // There is a limited ability for these different variants to propagate through the graph
            // traversal, due to the limitations in the cost metrics that we are using. But it is better
            // than nothing and could be refined later if needed.
            //
            // We also treat filling the given resources to full as having a separate "patience" allocation
            // of cycle frames. So in a worst-case scenario the total time required for the farm could be up
            // to double what would be allowed in forward traversal. But because we allocate only a modest
            // 45 seconds for farming (assuming no patience tech), in the worst case this still stays under
            // the limit of 1.5 minutes associated with `canBePatient`.
            //
            let [drop_energy, drop_missiles, drop_supers, drop_pbs] = get_total_enemy_drop_values(
                drops,
                full_energy,
                full_missiles,
                full_supers,
                full_power_bombs,
                settings.quality_of_life_settings.buffed_drops,
            );

            let net_energy = ((drop_energy - cycle_energy) * num_cycles as f32) as Capacity;
            let net_missiles = ((drop_missiles - cycle_missiles) * num_cycles as f32) as Capacity;
            let net_supers = ((drop_supers - cycle_supers) * num_cycles as f32) as Capacity;
            let net_pbs = ((drop_pbs - cycle_pbs) * num_cycles as f32) as Capacity;

            if net_energy < 0 || net_missiles < 0 || net_supers < 0 || net_pbs < 0 {
                return None;
            }

            // Now calculate refill rates assuming no resources are full. This is what we use to determine
            // how close to full the given resources must start out:
            let [raw_energy, raw_missiles, raw_supers, raw_pbs] = get_total_enemy_drop_values(
                drops,
                false,
                false,
                false,
                false,
                settings.quality_of_life_settings.buffed_drops,
            );

            let fill_energy = ((raw_energy - cycle_energy) * num_cycles as f32) as Capacity;
            let fill_missiles = ((raw_missiles - cycle_missiles) * num_cycles as f32) as Capacity;
            let fill_supers = ((raw_supers - cycle_supers) * num_cycles as f32) as Capacity;
            let fill_pbs = ((raw_pbs - cycle_pbs) * num_cycles as f32) as Capacity;

            if full_energy {
                if fill_energy > global.inventory.max_reserves {
                    new_local.reserves_used = 0;
                    new_local.energy_used = global.inventory.max_energy
                        - 1
                        - (fill_energy - global.inventory.max_reserves);
                    if new_local.energy_used < 0 {
                        new_local.energy_used = 0;
                    }
                } else {
                    new_local.energy_used = global.inventory.max_energy - 1;
                    new_local.reserves_used = global.inventory.max_reserves - fill_energy;
                }
            } else {
                if new_local.reserves_used > 0 {
                    // There may be a way to refine this by having an option to fill regular energy (not reserves),
                    // but it probably wouldn't work without creating a new cost metric anyway. It probably only
                    // applies in scenarios involving Big Boy drain?
                    new_local.energy_used = global.inventory.max_energy - 1;
                }
                if net_energy > new_local.reserves_used {
                    new_local.energy_used -= net_energy - new_local.reserves_used;
                    new_local.reserves_used = 0;
                    if new_local.energy_used < 0 {
                        new_local.energy_used = 0;
                    }
                } else {
                    new_local.reserves_used -= net_energy;
                }
            }
            if full_missiles {
                new_local.missiles_used = global.inventory.max_missiles - fill_missiles;
            } else {
                new_local.missiles_used -= net_missiles;
            }
            if new_local.missiles_used < 0 {
                new_local.missiles_used = 0;
            }
            if full_supers {
                new_local.supers_used = global.inventory.max_supers - fill_supers;
            } else {
                new_local.supers_used -= net_supers;
            }
            if new_local.supers_used < 0 {
                new_local.supers_used = 0;
            }
            if full_power_bombs {
                new_local.power_bombs_used = global.inventory.max_power_bombs - fill_pbs;
            } else {
                new_local.power_bombs_used -= net_pbs;
            }
            if new_local.power_bombs_used < 0 {
                new_local.power_bombs_used = 0;
            }
            return Some(new_local);
        } else {
            let mut energy = new_local.energy_used as f32;
            let mut reserves = new_local.reserves_used as f32;
            let mut missiles = new_local.missiles_used as f32;
            let mut supers = new_local.supers_used as f32;
            let mut pbs = new_local.power_bombs_used as f32;

            while num_cycles > 0 {
                let [drop_energy, drop_missiles, drop_supers, drop_pbs] =
                    get_total_enemy_drop_values(
                        drops,
                        energy == 0.0 && reserves == 0.0,
                        missiles == 0.0,
                        supers == 0.0,
                        pbs == 0.0,
                        settings.quality_of_life_settings.buffed_drops,
                    );

                let net_energy = drop_energy - cycle_energy;
                let net_missiles = drop_missiles - cycle_missiles;
                let net_supers = drop_supers - cycle_supers;
                let net_pbs = drop_pbs - cycle_pbs;

                if net_energy < 0.0 || net_missiles < 0.0 || net_supers < 0.0 || net_pbs < 0.0 {
                    return None;
                }

                energy -= net_energy;
                if energy < 0.0 {
                    reserves += energy;
                    energy = 0.0;
                }
                if reserves < 0.0 {
                    reserves = 0.0;
                }
                missiles -= net_missiles;
                if missiles < 0.0 {
                    missiles = 0.0;
                }
                supers -= net_supers;
                if supers < 0.0 {
                    supers = 0.0;
                }
                pbs -= net_pbs;
                if pbs < 0.0 {
                    pbs = 0.0;
                }

                new_local.energy_used = energy.ceil() as Capacity;
                new_local.reserves_used = reserves.ceil() as Capacity;
                new_local.missiles_used = missiles.ceil() as Capacity;
                new_local.supers_used = supers.ceil() as Capacity;
                new_local.power_bombs_used = pbs.ceil() as Capacity;

                // TODO: process multiple cycles at once, for more efficient computation.
                num_cycles -= 1;
            }
        }

        if new_local.energy_used == 0 && new_local.reserves_used == 0 {
            new_local.farm_baseline_energy_used = 0;
            new_local.farm_baseline_reserves_used = 0;
        }
        if new_local.missiles_used == 0 {
            new_local.farm_baseline_missiles_used = 0;
        }
        if new_local.supers_used == 0 {
            new_local.farm_baseline_supers_used = 0;
        }
        if new_local.power_bombs_used == 0 {
            new_local.farm_baseline_power_bombs_used = 0;
        }
        Some(new_local)
    } else {
        None
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
    door_map: &HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
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
        Requirement::MotherBrainBarrierClear(obj_id) => {
            if is_mother_brain_barrier_clear(global, difficulty, objectives, game_data, *obj_id) {
                Some(local)
            } else {
                None
            }
        }
        Requirement::DisableableETank => {
            if settings.quality_of_life_settings.disableable_etanks {
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
            apply_heat_frames(*frames, local, global, game_data, difficulty, false)
        }
        Requirement::SimpleHeatFrames(frames) => {
            apply_heat_frames(*frames, local, global, game_data, difficulty, true)
        }
        Requirement::HeatFramesWithEnergyDrops(frames, enemy_drops, enemy_drops_buffed) => {
            let drops = if settings.quality_of_life_settings.buffed_drops {
                enemy_drops_buffed
            } else {
                enemy_drops
            };
            apply_heat_frames_with_energy_drops(
                *frames, drops, local, global, game_data, settings, difficulty, reverse,
            )
        }
        Requirement::LavaFramesWithEnergyDrops(frames, enemy_drops, enemy_drops_buffed) => {
            let drops = if settings.quality_of_life_settings.buffed_drops {
                enemy_drops_buffed
            } else {
                enemy_drops
            };
            apply_lava_frames_with_energy_drops(
                *frames, drops, local, global, game_data, settings, difficulty, reverse,
            )
        }
        Requirement::MainHallElevatorFrames => {
            if settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(188, local, global, game_data, difficulty, true)
            } else if !global.inventory.items[Item::Varia as usize]
                && global.inventory.max_energy < 149
            {
                None
            } else {
                apply_heat_frames(436, local, global, game_data, difficulty, true)
            }
        }
        Requirement::LowerNorfairElevatorDownFrames => {
            if settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(30, local, global, game_data, difficulty, true)
            } else {
                apply_heat_frames(60, local, global, game_data, difficulty, true)
            }
        }
        Requirement::LowerNorfairElevatorUpFrames => {
            if settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(48, local, global, game_data, difficulty, true)
            } else {
                apply_heat_frames(108, local, global, game_data, difficulty, true)
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
        Requirement::CycleFrames(frames) => {
            let mut new_local: LocalState = local;
            new_local.cycle_frames +=
                (*frames as f32 * difficulty.resource_multiplier).ceil() as Capacity;
            Some(new_local)
        }
        Requirement::SimpleCycleFrames(frames) => {
            let mut new_local: LocalState = local;
            new_local.cycle_frames += frames;
            Some(new_local)
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
        Requirement::Energy(count) => {
            let mut new_local = local;
            new_local.energy_used += *count;
            validate_energy(new_local, &global.inventory, can_manage_reserves)
        }
        Requirement::RegularEnergy(count) => {
            // For now, we assume reserve energy can be converted to regular energy, so this is
            // implemented the same as the Energy requirement above.
            let mut new_local = local;
            new_local.energy_used += *count;
            validate_energy(new_local, &global.inventory, can_manage_reserves)
        }
        Requirement::ReserveEnergy(count) => {
            let mut new_local = local;
            new_local.reserves_used += *count;
            validate_energy(new_local, &global.inventory, can_manage_reserves)
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
        Requirement::BombIntoCrystalFlashClipLeniency {} => {
            let mut new_local = local;
            new_local.power_bombs_used += difficulty.bomb_into_cf_leniency;
            validate_power_bombs(new_local, global)
        }
        Requirement::JumpIntoCrystalFlashClipLeniency {} => {
            let mut new_local = local;
            new_local.power_bombs_used += difficulty.jump_into_cf_leniency;
            validate_power_bombs(new_local, global)
        }
        Requirement::XModeSpikeHitLeniency {} => {
            let mut new_local = local;
            new_local.energy_used +=
                difficulty.spike_xmode_leniency * 60 / suit_damage_factor(&global.inventory);
            validate_energy(new_local, &global.inventory, can_manage_reserves)
        }
        Requirement::XModeThornHitLeniency {} => {
            let mut new_local = local;
            new_local.energy_used +=
                difficulty.spike_xmode_leniency * 16 / suit_damage_factor(&global.inventory);
            validate_energy(new_local, &global.inventory, can_manage_reserves)
        }
        Requirement::MissilesAvailable(count) => {
            apply_missiles_available_req(local, global, *count, reverse)
        }
        Requirement::SupersAvailable(count) => {
            apply_supers_available_req(local, global, *count, reverse)
        }
        Requirement::PowerBombsAvailable(count) => {
            apply_power_bombs_available_req(local, global, *count, reverse)
        }
        Requirement::RegularEnergyAvailable(count) => {
            apply_regular_energy_available_req(local, global, *count, reverse)
        }
        Requirement::ReserveEnergyAvailable(count) => {
            apply_reserve_energy_available_req(local, global, *count, reverse)
        }
        Requirement::EnergyAvailable(count) => {
            apply_energy_available_req(local, global, *count, reverse)
        }
        Requirement::MissilesMissingAtMost(count) => apply_missiles_available_req(
            local,
            global,
            global.inventory.max_missiles - *count,
            reverse,
        ),
        Requirement::SupersMissingAtMost(count) => {
            apply_supers_available_req(local, global, global.inventory.max_supers - *count, reverse)
        }
        Requirement::PowerBombsMissingAtMost(count) => apply_power_bombs_available_req(
            local,
            global,
            global.inventory.max_power_bombs - *count,
            reverse,
        ),
        Requirement::RegularEnergyMissingAtMost(count) => apply_regular_energy_available_req(
            local,
            global,
            global.inventory.max_energy - *count,
            reverse,
        ),
        Requirement::ReserveEnergyMissingAtMost(count) => apply_reserve_energy_available_req(
            local,
            global,
            global.inventory.max_reserves - *count,
            reverse,
        ),
        Requirement::EnergyMissingAtMost(count) => apply_energy_available_req(
            local,
            global,
            global.inventory.max_energy + global.inventory.max_reserves - *count,
            reverse,
        ),
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
        Requirement::Farm {
            requirement,
            enemy_drops,
            enemy_drops_buffed,
            full_energy,
            full_missiles,
            full_power_bombs,
            full_supers,
        } => {
            let drops = if settings.quality_of_life_settings.buffed_drops {
                enemy_drops_buffed
            } else {
                enemy_drops
            };
            apply_farm_requirement(
                requirement,
                drops,
                global,
                local,
                reverse,
                *full_energy,
                *full_missiles,
                *full_supers,
                *full_power_bombs,
                settings,
                difficulty,
                game_data,
                door_map,
                locked_door_data,
                objectives,
            )
        }
        Requirement::EnergyRefill(limit) => {
            let limit_reserves = max(0, *limit - global.inventory.max_energy);
            if reverse {
                let mut new_local = local;
                if local.energy_used < *limit {
                    new_local.energy_used = 0;
                    new_local.farm_baseline_energy_used = 0;
                }
                if local.reserves_used <= limit_reserves {
                    new_local.reserves_used = 0;
                    new_local.farm_baseline_reserves_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.energy_used > global.inventory.max_energy - limit {
                    new_local.energy_used = max(0, global.inventory.max_energy - limit);
                    new_local.farm_baseline_energy_used = new_local.energy_used;
                }
                if local.reserves_used > global.inventory.max_reserves - limit_reserves {
                    new_local.reserves_used =
                        max(0, global.inventory.max_reserves - limit_reserves);
                    new_local.farm_baseline_reserves_used = new_local.reserves_used;
                }
                Some(new_local)
            }
        }
        Requirement::RegularEnergyRefill(limit) => {
            if reverse {
                let mut new_local = local;
                if local.energy_used < *limit {
                    new_local.energy_used = 0;
                    new_local.farm_baseline_energy_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.energy_used > global.inventory.max_energy - limit {
                    new_local.energy_used = max(0, global.inventory.max_energy - limit);
                    new_local.farm_baseline_energy_used = new_local.energy_used;
                }
                Some(new_local)
            }
        }
        Requirement::ReserveRefill(limit) => {
            if reverse {
                let mut new_local = local;
                if local.reserves_used <= *limit {
                    new_local.reserves_used = 0;
                    new_local.farm_baseline_reserves_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.reserves_used > global.inventory.max_reserves - limit {
                    new_local.reserves_used = max(0, global.inventory.max_reserves - limit);
                    new_local.farm_baseline_reserves_used = new_local.reserves_used;
                }
                Some(new_local)
            }
        }
        Requirement::MissileRefill(limit) => {
            if reverse {
                let mut new_local = local;
                if local.missiles_used <= *limit {
                    new_local.missiles_used = 0;
                    new_local.farm_baseline_missiles_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.missiles_used > global.inventory.max_missiles - limit {
                    new_local.missiles_used = max(0, global.inventory.max_missiles - limit);
                    new_local.farm_baseline_missiles_used = new_local.missiles_used;
                }
                Some(new_local)
            }
        }
        Requirement::SuperRefill(limit) => {
            if reverse {
                let mut new_local = local;
                if local.supers_used <= *limit {
                    new_local.supers_used = 0;
                    new_local.farm_baseline_supers_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.supers_used > global.inventory.max_supers - limit {
                    new_local.supers_used = max(0, global.inventory.max_supers - limit);
                    new_local.farm_baseline_supers_used = new_local.supers_used;
                }
                Some(new_local)
            }
        }
        Requirement::PowerBombRefill(limit) => {
            if reverse {
                let mut new_local = local;
                if local.power_bombs_used <= *limit {
                    new_local.power_bombs_used = 0;
                    new_local.farm_baseline_power_bombs_used = 0;
                }
                Some(new_local)
            } else {
                let mut new_local = local;
                if local.power_bombs_used > global.inventory.max_power_bombs - limit {
                    new_local.power_bombs_used = max(0, global.inventory.max_power_bombs - limit);
                    new_local.farm_baseline_power_bombs_used = new_local.power_bombs_used;
                }
                Some(new_local)
            }
        }
        Requirement::AmmoStationRefill => {
            let mut new_local = local;
            new_local.missiles_used = 0;
            new_local.farm_baseline_missiles_used = 0;
            if !settings.other_settings.ultra_low_qol {
                new_local.supers_used = 0;
                new_local.farm_baseline_supers_used = 0;
                new_local.power_bombs_used = 0;
                new_local.farm_baseline_power_bombs_used = 0;
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
        Requirement::EnergyStationRefill => {
            let mut new_local = local;
            new_local.energy_used = 0;
            new_local.farm_baseline_energy_used = 0;
            if settings.quality_of_life_settings.energy_station_reserves
                || settings.quality_of_life_settings.reserve_backward_transfer
            {
                new_local.reserves_used = 0;
                new_local.farm_baseline_reserves_used = 0;
            }
            Some(new_local)
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
        Requirement::MissileDrain(count) => {
            if reverse {
                if local.missiles_used > *count {
                    None
                } else {
                    Some(local)
                }
            } else {
                let mut new_local = local;
                new_local.missiles_used =
                    Capacity::max(local.missiles_used, global.inventory.max_missiles - count);
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
            can_be_patient_tech_idx,
            can_be_very_patient_tech_idx,
            can_be_extremely_patient_tech_idx,
            power_bombs,
            g_mode,
            stuck,
        } => apply_ridley_requirement(
            &global.inventory,
            local,
            difficulty.ridley_proficiency,
            can_manage_reserves,
            difficulty.tech[*can_be_patient_tech_idx],
            difficulty.tech[*can_be_very_patient_tech_idx],
            difficulty.tech[*can_be_extremely_patient_tech_idx],
            *power_bombs,
            *g_mode,
            *stuck,
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
                        if frames == excess_frames {
                            // If all frames are excess frames and energy is at 29 or lower, then the spark does not require any energy:
                            return Some(new_local);
                        }
                        new_local.energy_used = 28 + frames - excess_frames;
                    } else {
                        new_local.energy_used += frames;
                    }
                    validate_energy_no_auto_reserve(new_local, global, game_data, difficulty)
                } else {
                    if frames == excess_frames
                        && new_local.energy_used >= global.inventory.max_energy - 29
                    {
                        // If all frames are excess frames and energy is at 29 or lower, then the spark does not require any energy:
                        return Some(new_local);
                    }
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
                        door_map,
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
                        door_map,
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
                        door_map,
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
                                    door_map,
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
                    DoorType::Wall => None,
                }
            } else {
                Some(local)
            }
        }
        &Requirement::ResetRoom { room_id, node_id } => {
            // TODO: add more requirements here
            let mut new_local = local;
            if new_local.cycle_frames > 0 {
                // We assume the it takes 400 frames to go through the door transition, shoot open the door, and return.
                // The actual time can vary based on room load time and whether fast doors are enabled.
                new_local.cycle_frames += 400;
            }

            if let Some(&(other_room_id, other_node_id)) = door_map.get(&(room_id, node_id)) {
                let reset_req =
                    &game_data.node_reset_room_requirement[&(other_room_id, other_node_id)];
                new_local = apply_requirement(
                    reset_req,
                    global,
                    new_local,
                    reverse,
                    settings,
                    difficulty,
                    game_data,
                    door_map,
                    locked_door_data,
                    objectives,
                )?;
            }
            Some(new_local)
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
                for req in reqs.iter().rev() {
                    new_local = apply_requirement(
                        req,
                        global,
                        new_local,
                        reverse,
                        settings,
                        difficulty,
                        game_data,
                        door_map,
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
                        door_map,
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
                    door_map,
                    locked_door_data,
                    objectives,
                ) {
                    let cost = compute_cost(new_local, &global.inventory, reverse);
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

#[derive(Clone, Serialize, Deserialize)]
pub struct StepTrail {
    pub prev_trail_id: StepTrailId,
    pub link_idx: LinkIdx,
    pub local_state: LocalState,
}

#[derive(Clone)]
pub struct TraverseResult {
    pub traversal_number: usize,
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
    door_map: &HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
    locked_door_data: &LockedDoorData,
    objectives: &[Objective],
    traversal_number: &mut usize,
) -> TraverseResult {
    debug!("Traversal number {traversal_number} (reverse: {reverse})");
    let mut modified_vertices: HashMap<usize, [bool; NUM_COST_METRICS]> = HashMap::new();
    let mut result: TraverseResult;

    if let Some(init) = init_opt {
        for (v, cost) in init.cost.iter().enumerate() {
            let valid = cost.map(f32::is_finite);
            if valid.iter().any(|&x| x) {
                modified_vertices.insert(v, valid);
            }
        }
        result = init;
    } else {
        result = TraverseResult {
            traversal_number: 0,
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
        result.cost[start_vertex_id] = compute_cost(init_local, &global.inventory, reverse);
        modified_vertices.insert(start_vertex_id, first_metric);
    }
    result.traversal_number = *traversal_number;
    *traversal_number += 1;

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

    while !modified_vertices.is_empty() {
        let mut new_modified_vertices: HashMap<usize, [bool; NUM_COST_METRICS]> = HashMap::new();
        let modified_vertices_vec = {
            // Process the vertices in sorted order, to make the traversal deterministic.
            let mut m: Vec<(usize, [bool; NUM_COST_METRICS])> =
                modified_vertices.into_iter().collect();
            m.sort();
            m
        };
        for &(src_id, modified_costs) in &modified_vertices_vec {
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
                        link,
                        global,
                        src_local_state,
                        reverse,
                        settings,
                        difficulty,
                        game_data,
                        door_map,
                        locked_door_data,
                        objectives,
                    ) {
                        let dst_new_cost_arr =
                            compute_cost(dst_new_local_state, &global.inventory, reverse);

                        let new_step_trail = StepTrail {
                            prev_trail_id: src_trail_id,
                            local_state: dst_new_local_state,
                            link_idx,
                        };
                        let new_trail_id = result.step_trails.len() as StepTrailId;
                        let mut any_improvement: bool = false;
                        let mut improved_arr: [bool; NUM_COST_METRICS] = new_modified_vertices
                            .get(&dst_id)
                            .copied()
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
                                    panic!(
                                        "Resource {name} is negative, with value {v}: old_state={src_local_state:?}, new_state={dst_new_local_state:?}, link={link:?}"
                                    );
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
