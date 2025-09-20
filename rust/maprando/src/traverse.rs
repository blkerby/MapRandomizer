use std::{
    cmp::{max, min},
    fmt::Debug,
};

use arrayvec::ArrayVec;
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use crate::{
    randomize::{DifficultyConfig, LockedDoor},
    settings::{MotherBrainFight, Objective, RandomizerSettings, WallJump},
};
use maprando_game::{
    BeamType, Capacity, DoorType, EnemyDrop, EnemyVulnerabilities, GameData, Item, Link, LinkIdx,
    LinksDataGroup, NodeId, Requirement, RoomId, StepTrailId, TECH_ID_CAN_SUITLESS_LAVA_DIVE,
    VertexId,
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
    local: &mut LocalState,
    count: Capacity,
    vul: &EnemyVulnerabilities,
) -> bool {
    // Prioritize using weapons that do not require ammo:
    if global.weapon_mask & vul.non_ammo_vulnerabilities != 0 {
        return true;
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

    hp <= 0
}

pub const NUM_COST_METRICS: usize = 3;

fn compute_cost(
    local: &LocalState,
    inventory: &Inventory,
    reverse: bool,
) -> [f32; NUM_COST_METRICS] {
    let eps = 1e-15;
    let energy = match (reverse, local.energy) {
        (false, maprando_logic::ResourceLevel::Consumed(x)) => x,
        (false, maprando_logic::ResourceLevel::Remaining(x)) => inventory.max_energy - x,
        (true, maprando_logic::ResourceLevel::Consumed(x)) => inventory.max_energy - x,
        (true, maprando_logic::ResourceLevel::Remaining(x)) => x,
    };
    let energy_cost = (energy as f32) / (inventory.max_energy as f32 + eps);
    let reserve_cost = (local.reserves_used as f32) / (inventory.max_reserves as f32 + eps);
    let missiles_cost = (local.missiles_used as f32) / (inventory.max_missiles as f32 + eps);
    let supers_cost = (local.supers_used as f32) / (inventory.max_supers as f32 + eps);
    let power_bombs_cost =
        (local.power_bombs_used as f32) / (inventory.max_power_bombs as f32 + eps);
    let mut shinecharge_cost = if local.flash_suit {
        // For the purposes of the cost metrics, treat flash suit as equivalent
        // to a large amount of shinecharge frames remaining:
        -10.0
    } else {
        -(local.shinecharge_frames_remaining as f32) / 180.0
    };
    if reverse {
        shinecharge_cost = -shinecharge_cost;
    }
    let cycle_frames_cost = (local.cycle_frames as f32) * 0.0001;
    let total_energy_cost = energy_cost + reserve_cost;
    let total_ammo_cost = missiles_cost + supers_cost + power_bombs_cost;

    let energy_sensitive_cost_metric =
        100.0 * total_energy_cost + total_ammo_cost + shinecharge_cost + cycle_frames_cost;
    let ammo_sensitive_cost_metric =
        total_energy_cost + 100.0 * total_ammo_cost + shinecharge_cost + cycle_frames_cost;
    let shinecharge_sensitive_cost_metric =
        total_energy_cost + total_ammo_cost + 100.0 * shinecharge_cost + cycle_frames_cost;
    [
        energy_sensitive_cost_metric,
        ammo_sensitive_cost_metric,
        shinecharge_sensitive_cost_metric,
    ]
}

fn validate_energy_no_auto_reserve(
    local: &mut LocalState,
    global: &GlobalState,
    game_data: &GameData,
    difficulty: &DifficultyConfig,
) -> bool {
    if local.energy >= global.inventory.max_energy {
        if difficulty.tech[game_data.manage_reserves_tech_idx] {
            // Assume that just enough reserve energy is manually converted to regular energy.
            local.reserves_used += local.energy - (global.inventory.max_energy - 1);
            local.energy = global.inventory.max_energy - 1;
        } else {
            // Assume that reserves cannot be used (e.g. during a shinespark or enemy hit).
            return false;
        }
    }
    if local.reserves_used > global.inventory.max_reserves {
        return false;
    }
    true
}

fn validate_missiles(local: &LocalState, global: &GlobalState) -> bool {
    local.missiles_used <= global.inventory.max_missiles
}

fn validate_supers(local: &LocalState, global: &GlobalState) -> bool {
    local.supers_used <= global.inventory.max_supers
}

fn validate_power_bombs(local: &LocalState, global: &GlobalState) -> bool {
    local.power_bombs_used <= global.inventory.max_power_bombs
}

fn apply_gate_glitch_leniency(
    local: &mut LocalState,
    global: &GlobalState,
    green: bool,
    heated: bool,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
) -> bool {
    if heated && !global.inventory.items[Item::Varia as usize] {
        local.energy += (difficulty.gate_glitch_leniency as f32
            * difficulty.resource_multiplier
            * 60.0) as Capacity;
        if !validate_energy(
            local,
            &global.inventory,
            difficulty.tech[game_data.manage_reserves_tech_idx],
        ) {
            return false;
        }
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
    local: &mut LocalState,
    global: &GlobalState,
    game_data: &GameData,
    difficulty: &DifficultyConfig,
    simple: bool,
) -> bool {
    let varia = global.inventory.items[Item::Varia as usize];
    if varia {
        true
    } else if !difficulty.tech[game_data.heat_run_tech_idx] {
        false
    } else {
        if simple {
            local.energy += (frames as f32 / 4.0).ceil() as Capacity;
        } else {
            local.energy +=
                (frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity;
        }
        validate_energy(
            local,
            &global.inventory,
            difficulty.tech[game_data.manage_reserves_tech_idx],
        )
    }
}

fn get_enemy_drop_energy_value(
    drop: &EnemyDrop,
    local: &LocalState,
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
    local: &mut LocalState,
    global: &GlobalState,
    game_data: &GameData,
    settings: &RandomizerSettings,
    difficulty: &DifficultyConfig,
    reverse: bool,
) -> bool {
    let varia = global.inventory.items[Item::Varia as usize];
    if varia {
        true
    } else if !difficulty.tech[game_data.heat_run_tech_idx] {
        false
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
        local.energy += heat_energy;
        if !validate_energy(
            local,
            &global.inventory,
            difficulty.tech[game_data.manage_reserves_tech_idx],
        ) {
            return false;
        }
        if total_drop_value <= local.energy {
            local.energy -= total_drop_value;
        } else {
            local.reserves_used -= total_drop_value - local.energy;
            local.energy = 0;
        }
        true
    }
}

fn apply_lava_frames_with_energy_drops(
    frames: Capacity,
    drops: &[EnemyDrop],
    local: &mut LocalState,
    global: &GlobalState,
    game_data: &GameData,
    settings: &RandomizerSettings,
    difficulty: &DifficultyConfig,
    reverse: bool,
) -> bool {
    let varia = global.inventory.items[Item::Varia as usize];
    let gravity = global.inventory.items[Item::Gravity as usize];
    if gravity && varia {
        true
    } else if !difficulty.tech[game_data.tech_isv.index_by_key[&TECH_ID_CAN_SUITLESS_LAVA_DIVE]] {
        false
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
        local.energy += lava_energy;
        if !validate_energy(
            local,
            &global.inventory,
            difficulty.tech[game_data.manage_reserves_tech_idx],
        ) {
            return false;
        }
        if total_drop_value <= local.energy {
            local.energy -= total_drop_value;
        } else {
            local.reserves_used -= total_drop_value - local.energy;
            local.energy = 0;
        }
        true
    }
}

#[derive(Clone)]
pub struct LockedDoorData {
    pub locked_doors: Vec<LockedDoor>,
    pub locked_door_node_map: HashMap<(RoomId, NodeId), usize>,
    pub locked_door_vertex_ids: Vec<Vec<VertexId>>,
}

type LocalStateArray = ArrayVec<LocalState, NUM_COST_METRICS>;

fn apply_link(link: &Link, mut local: LocalStateArray, cx: &TraversalContext) -> LocalStateArray {
    if cx.reverse {
        if !link.end_with_shinecharge {
            local.retain(|x| x.shinecharge_frames_remaining == 0);
        }
    } else if link.start_with_shinecharge {
        local.retain(|x| x.shinecharge_frames_remaining > 0);
    }
    local = apply_requirement_complex(&link.requirement, local, cx);
    if cx.reverse {
        if !link.start_with_shinecharge {
            for loc in &mut local {
                loc.shinecharge_frames_remaining = 0;
            }
        }
    } else if !link.end_with_shinecharge {
        for loc in &mut local {
            loc.shinecharge_frames_remaining = 0;
        }
    }
    local
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
    local: &mut LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> bool {
    if reverse {
        if global.inventory.max_missiles < count {
            false
        } else {
            local.missiles_used = Capacity::max(local.missiles_used, count);
            true
        }
    } else {
        global.inventory.max_missiles - local.missiles_used >= count
    }
}

fn apply_supers_available_req(
    local: &mut LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> bool {
    if reverse {
        if global.inventory.max_supers < count {
            false
        } else {
            local.supers_used = Capacity::max(local.supers_used, count);
            true
        }
    } else {
        global.inventory.max_supers - local.supers_used >= count
    }
}

fn apply_power_bombs_available_req(
    local: &mut LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> bool {
    if reverse {
        if global.inventory.max_power_bombs < count {
            false
        } else {
            local.power_bombs_used = Capacity::max(local.power_bombs_used, count);
            true
        }
    } else {
        global.inventory.max_power_bombs - local.power_bombs_used >= count
    }
}

fn apply_regular_energy_available_req(
    local: &mut LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> bool {
    if reverse {
        if global.inventory.max_energy < count {
            false
        } else {
            local.energy = Capacity::max(local.energy, count);
            true
        }
    } else {
        global.inventory.max_energy - local.energy >= count
    }
}

fn apply_reserve_energy_available_req(
    local: &mut LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> bool {
    if reverse {
        if global.inventory.max_reserves < count {
            false
        } else {
            local.reserves_used = Capacity::max(local.reserves_used, count);
            true
        }
    } else {
        global.inventory.max_reserves - local.reserves_used >= count
    }
}

fn apply_energy_available_req(
    local: &mut LocalState,
    global: &GlobalState,
    count: Capacity,
    reverse: bool,
) -> bool {
    if reverse {
        if global.inventory.max_energy + global.inventory.max_reserves < count {
            false
        } else if global.inventory.max_energy < count {
            local.energy = global.inventory.max_energy;
            local.reserves_used =
                Capacity::max(local.reserves_used, count - global.inventory.max_energy);
            true
        } else {
            local.energy = Capacity::max(local.energy, count);
            false
        }
    } else {
        global.inventory.max_reserves - local.reserves_used + global.inventory.max_energy
            - local.energy
            >= count
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
    let end_local = end_local_result?;
    if end_local.cycle_frames < 100 {
        panic!("bad farm: expected cycle_frames >= 100: end_local={end_local:#?},\n req={req:#?}");
    }
    let cycle_frames = (end_local.cycle_frames - 1) as f32;
    let cycle_energy =
        (end_local.energy + end_local.reserves_used - local.energy - local.reserves_used) as f32;
    let cycle_missiles = (end_local.missiles_used - local.missiles_used) as f32;
    let cycle_supers = (end_local.supers_used - local.supers_used) as f32;
    let cycle_pbs = (end_local.power_bombs_used - local.power_bombs_used) as f32;
    let patience_frames = difficulty.farm_time_limit * 60.0;
    let mut num_cycles = (patience_frames / cycle_frames).floor() as i32;

    let mut new_local = local;
    if new_local.farm_baseline_energy_used < new_local.energy {
        new_local.farm_baseline_energy_used = new_local.energy;
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
    new_local.energy = new_local.farm_baseline_energy_used;
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
                new_local.energy =
                    global.inventory.max_energy - 1 - (fill_energy - global.inventory.max_reserves);
                if new_local.energy < 0 {
                    new_local.energy = 0;
                }
            } else {
                new_local.energy = global.inventory.max_energy - 1;
                new_local.reserves_used = global.inventory.max_reserves - fill_energy;
            }
        } else {
            if new_local.reserves_used > 0 {
                // There may be a way to refine this by having an option to fill regular energy (not reserves),
                // but it probably wouldn't work without creating a new cost metric anyway. It probably only
                // applies in scenarios involving Big Boy drain?
                new_local.energy = global.inventory.max_energy - 1;
            }
            if net_energy > new_local.reserves_used {
                new_local.energy -= net_energy - new_local.reserves_used;
                new_local.reserves_used = 0;
                if new_local.energy < 0 {
                    new_local.energy = 0;
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
    } else {
        let mut energy = new_local.energy as f32;
        let mut reserves = new_local.reserves_used as f32;
        let mut missiles = new_local.missiles_used as f32;
        let mut supers = new_local.supers_used as f32;
        let mut pbs = new_local.power_bombs_used as f32;

        while num_cycles > 0 {
            let [drop_energy, drop_missiles, drop_supers, drop_pbs] = get_total_enemy_drop_values(
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

            new_local.energy = energy.round() as Capacity;
            new_local.reserves_used = reserves.round() as Capacity;
            new_local.missiles_used = missiles.round() as Capacity;
            new_local.supers_used = supers.round() as Capacity;
            new_local.power_bombs_used = pbs.round() as Capacity;

            // TODO: process multiple cycles at once, for more efficient computation.
            num_cycles -= 1;
        }
    }

    new_local.energy = Capacity::min(new_local.energy, local.energy);
    new_local.reserves_used = Capacity::min(new_local.reserves_used, local.reserves_used);
    new_local.missiles_used = Capacity::min(new_local.missiles_used, local.missiles_used);
    new_local.supers_used = Capacity::min(new_local.supers_used, local.supers_used);
    new_local.power_bombs_used = Capacity::min(new_local.power_bombs_used, local.power_bombs_used);

    if new_local.energy == 0 && new_local.reserves_used == 0 {
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
}

// We parametrize the "trail_id" type. In the Traverser data structure, an i32 is used;
// but during internal requirement processing (e.g. resolving `or`s) a unit type is used
// as we do not create trails at that level of detail.
#[derive(Clone, Debug)]
pub struct LocalStateReducer<T: Copy + Debug> {
    pub local: ArrayVec<LocalState, NUM_COST_METRICS>,
    pub trail_ids: ArrayVec<T, NUM_COST_METRICS>,
    pub best_cost_values: [f32; NUM_COST_METRICS],
    pub best_cost_idxs: [u8; NUM_COST_METRICS],
}

impl<T: Copy + Debug> LocalStateReducer<T> {
    fn new() -> Self {
        Self {
            local: ArrayVec::new(),
            trail_ids: ArrayVec::new(),
            best_cost_values: [f32::MAX; NUM_COST_METRICS],
            best_cost_idxs: [0; NUM_COST_METRICS],
        }
    }

    fn push(
        &mut self,
        local: LocalState,
        trail_id: T,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        let cost = compute_cost(&local, inventory, reverse);
        let n = self.local.len() as u8;
        let mut improved_any: bool = false;
        let mut improved_all: bool = true;
        for i in 0..NUM_COST_METRICS {
            if cost[i] < self.best_cost_values[i] {
                self.best_cost_values[i] = cost[i];
                self.best_cost_idxs[i] = n;
                improved_any = true;
            }
            if cost[i] > self.best_cost_values[i] {
                improved_all = false;
            }
        }

        // Handle the common, easy cases first:
        if !improved_any {
            // No strict improvement across any metrics, so do nothing.
            return false;
        }

        if improved_all {
            // Weak improvement across all metrics, so replace all existing states with this one.
            let mut new_local = ArrayVec::new();
            new_local.push(local);
            self.local = new_local;

            let mut new_trail_ids = ArrayVec::new();
            new_trail_ids.push(trail_id);
            self.trail_ids = new_trail_ids;

            self.best_cost_idxs = [0; NUM_COST_METRICS];
            self.best_cost_values = cost;
            return true;
        }

        // The general case: some metrics are improved, others were better with an existing state.
        // Filter the states, keeping only those that are optimal with respect to at least one cost metric.
        let mut idxs = self.best_cost_idxs;
        idxs.sort();
        let mut idxs = idxs.to_vec();
        idxs.dedup();
        let mut new_local: ArrayVec<LocalState, NUM_COST_METRICS> = ArrayVec::new();
        let mut new_trail_ids: ArrayVec<T, NUM_COST_METRICS> = ArrayVec::new();
        for &i in &idxs {
            if i == n {
                new_local.push(local);
                new_trail_ids.push(trail_id);
            } else {
                new_local.push(self.local[i as usize]);
                new_trail_ids.push(self.trail_ids[i as usize]);
            }
        }
        self.local = new_local;
        self.trail_ids = new_trail_ids;
        'outer: for i in 0..NUM_COST_METRICS {
            let j0 = self.best_cost_idxs[i];
            for (k, j) in idxs.iter().copied().enumerate() {
                if j == j0 {
                    self.best_cost_idxs[i] = k as u8;
                    continue 'outer;
                }
            }
            panic!("internal error");
        }
        true
    }
}

struct TraversalContext<'a> {
    global: &'a GlobalState,
    reverse: bool,
    settings: &'a RandomizerSettings,
    difficulty: &'a DifficultyConfig,
    game_data: &'a GameData,
    door_map: &'a HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
    locked_door_data: &'a LockedDoorData,
    objectives: &'a [Objective],
}

// TODO: get rid of this function?
pub fn apply_requirement(
    req: &Requirement,
    global: &GlobalState,
    mut local: LocalState,
    reverse: bool,
    settings: &RandomizerSettings,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
    door_map: &HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
    locked_door_data: &LockedDoorData,
    objectives: &[Objective],
) -> Option<LocalState> {
    let cx = TraversalContext {
        global,
        reverse,
        settings,
        difficulty,
        game_data,
        door_map,
        locked_door_data,
        objectives,
    };
    match apply_requirement_simple(req, &mut local, &cx) {
        SimpleResult::Failure => None,
        SimpleResult::Success => Some(local),
        SimpleResult::_ExtraState(_) => Some(local),
    }
}

enum SimpleResult {
    Failure,
    Success,
    _ExtraState(LocalState),
}

impl From<bool> for SimpleResult {
    fn from(value: bool) -> Self {
        if value {
            SimpleResult::Success
        } else {
            SimpleResult::Failure
        }
    }
}

fn apply_requirement_complex(
    req: &Requirement,
    mut local: ArrayVec<LocalState, NUM_COST_METRICS>,
    cx: &TraversalContext,
) -> ArrayVec<LocalState, NUM_COST_METRICS> {
    match req {
        Requirement::And(sub_reqs) => {
            for r in sub_reqs {
                local = apply_requirement_complex(r, local, cx);
                if local.is_empty() {
                    break;
                }
            }
            local
        }
        Requirement::Or(sub_reqs) => {
            let mut reducer: LocalStateReducer<()> = LocalStateReducer::new();
            for r in sub_reqs {
                for loc in apply_requirement_complex(r, local.clone(), cx) {
                    reducer.push(loc, (), &cx.global.inventory, cx.reverse);
                }
            }
            reducer.local
        }
        _ => {
            let mut reducer: LocalStateReducer<()> = LocalStateReducer::new();
            for mut loc in local {
                match apply_requirement_simple(req, &mut loc, cx) {
                    SimpleResult::Failure => {}
                    SimpleResult::Success => {
                        reducer.push(loc, (), &cx.global.inventory, cx.reverse);
                    }
                    SimpleResult::_ExtraState(extra_state) => {
                        reducer.push(loc, (), &cx.global.inventory, cx.reverse);
                        reducer.push(extra_state, (), &cx.global.inventory, cx.reverse);
                    }
                }
            }
            reducer.local
        }
    }
}

fn apply_requirement_simple(
    req: &Requirement,
    local: &mut LocalState,
    cx: &TraversalContext,
) -> SimpleResult {
    let can_manage_reserves = cx.difficulty.tech[cx.game_data.manage_reserves_tech_idx];
    match req {
        Requirement::Free => SimpleResult::Success,
        Requirement::Never => SimpleResult::Failure,
        Requirement::Tech(tech_idx) => cx.difficulty.tech[*tech_idx].into(),
        Requirement::Notable(notable_idx) => cx.difficulty.notables[*notable_idx].into(),
        Requirement::Item(item_id) => cx.global.inventory.items[*item_id].into(),
        Requirement::Flag(flag_id) => cx.global.flags[*flag_id].into(),
        Requirement::NotFlag(_flag_id) => {
            // We're ignoring this for now. It should be safe because all strats relying on a "not" flag will be
            // guarded by "canRiskPermanentLossOfAccess" if there is not an alternative strat with the flag set.
            SimpleResult::Success
        }
        Requirement::MotherBrainBarrierClear(obj_id) => is_mother_brain_barrier_clear(
            cx.global,
            cx.difficulty,
            cx.objectives,
            cx.game_data,
            *obj_id,
        )
        .into(),
        Requirement::DisableableETank => cx
            .settings
            .quality_of_life_settings
            .disableable_etanks
            .into(),
        Requirement::Walljump => match cx.settings.other_settings.wall_jump {
            WallJump::Vanilla => cx.difficulty.tech[cx.game_data.wall_jump_tech_idx].into(),
            WallJump::Collectible => (cx.difficulty.tech[cx.game_data.wall_jump_tech_idx]
                && cx.global.inventory.items[Item::WallJump as usize])
                .into(),
        },
        Requirement::ClimbWithoutLava => cx
            .settings
            .quality_of_life_settings
            .remove_climb_lava
            .into(),
        Requirement::HeatFrames(frames) => apply_heat_frames(
            *frames,
            local,
            cx.global,
            cx.game_data,
            cx.difficulty,
            false,
        )
        .into(),
        Requirement::SimpleHeatFrames(frames) => {
            apply_heat_frames(*frames, local, cx.global, cx.game_data, cx.difficulty, true).into()
        }
        Requirement::HeatFramesWithEnergyDrops(frames, enemy_drops, enemy_drops_buffed) => {
            let drops = if cx.settings.quality_of_life_settings.buffed_drops {
                enemy_drops_buffed
            } else {
                enemy_drops
            };
            apply_heat_frames_with_energy_drops(
                *frames,
                drops,
                local,
                cx.global,
                cx.game_data,
                cx.settings,
                cx.difficulty,
                cx.reverse,
            )
            .into()
        }
        Requirement::LavaFramesWithEnergyDrops(frames, enemy_drops, enemy_drops_buffed) => {
            let drops = if cx.settings.quality_of_life_settings.buffed_drops {
                enemy_drops_buffed
            } else {
                enemy_drops
            };
            apply_lava_frames_with_energy_drops(
                *frames,
                drops,
                local,
                cx.global,
                cx.game_data,
                cx.settings,
                cx.difficulty,
                cx.reverse,
            )
            .into()
        }
        Requirement::MainHallElevatorFrames => {
            if cx.settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(188, local, cx.global, cx.game_data, cx.difficulty, true).into()
            } else if !cx.global.inventory.items[Item::Varia as usize]
                && cx.global.inventory.max_energy < 149
            {
                SimpleResult::Failure
            } else {
                apply_heat_frames(436, local, cx.global, cx.game_data, cx.difficulty, true).into()
            }
        }
        Requirement::LowerNorfairElevatorDownFrames => {
            if cx.settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(30, local, cx.global, cx.game_data, cx.difficulty, true).into()
            } else {
                apply_heat_frames(60, local, cx.global, cx.game_data, cx.difficulty, true).into()
            }
        }
        Requirement::LowerNorfairElevatorUpFrames => {
            if cx.settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(48, local, cx.global, cx.game_data, cx.difficulty, true).into()
            } else {
                apply_heat_frames(108, local, cx.global, cx.game_data, cx.difficulty, true).into()
            }
        }
        Requirement::LavaFrames(frames) => {
            let varia = cx.global.inventory.items[Item::Varia as usize];
            let gravity = cx.global.inventory.items[Item::Gravity as usize];
            if gravity && varia {
                SimpleResult::Success
            } else if gravity || varia {
                local.energy +=
                    (*frames as f32 * cx.difficulty.resource_multiplier / 4.0).ceil() as Capacity;
                validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
            } else {
                local.energy +=
                    (*frames as f32 * cx.difficulty.resource_multiplier / 2.0).ceil() as Capacity;
                validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
            }
        }
        Requirement::GravitylessLavaFrames(frames) => {
            let varia = cx.global.inventory.items[Item::Varia as usize];
            if varia {
                local.energy +=
                    (*frames as f32 * cx.difficulty.resource_multiplier / 4.0).ceil() as Capacity
            } else {
                local.energy +=
                    (*frames as f32 * cx.difficulty.resource_multiplier / 2.0).ceil() as Capacity
            }
            validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
        }
        Requirement::AcidFrames(frames) => {
            local.energy += (*frames as f32 * cx.difficulty.resource_multiplier * 1.5
                / suit_damage_factor(&cx.global.inventory) as f32)
                .ceil() as Capacity;
            validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
        }
        Requirement::GravitylessAcidFrames(frames) => {
            let varia = cx.global.inventory.items[Item::Varia as usize];
            if varia {
                local.energy +=
                    (*frames as f32 * cx.difficulty.resource_multiplier * 0.75).ceil() as Capacity;
            } else {
                local.energy +=
                    (*frames as f32 * cx.difficulty.resource_multiplier * 1.5).ceil() as Capacity;
            }
            validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
        }
        Requirement::MetroidFrames(frames) => {
            local.energy += (*frames as f32 * cx.difficulty.resource_multiplier * 0.75
                / suit_damage_factor(&cx.global.inventory) as f32)
                .ceil() as Capacity;
            validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
        }
        Requirement::CycleFrames(frames) => {
            local.cycle_frames +=
                (*frames as f32 * cx.difficulty.resource_multiplier).ceil() as Capacity;
            SimpleResult::Success
        }
        Requirement::SimpleCycleFrames(frames) => {
            local.cycle_frames += frames;
            SimpleResult::Success
        }
        Requirement::Damage(base_energy) => {
            let energy = base_energy / suit_damage_factor(&cx.global.inventory);
            if energy >= cx.global.inventory.max_energy
                && !cx.difficulty.tech[cx.game_data.pause_abuse_tech_idx]
            {
                SimpleResult::Failure
            } else {
                local.energy += energy;
                validate_energy_no_auto_reserve(local, cx.global, cx.game_data, cx.difficulty)
                    .into()
            }
        }
        Requirement::Energy(count) => {
            local.energy += *count;
            validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
        }
        Requirement::RegularEnergy(count) => {
            // For now, we assume reserve energy can be converted to regular energy, so this is
            // implemented the same as the Energy requirement above.
            local.energy += *count;
            validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
        }
        Requirement::ReserveEnergy(count) => {
            local.reserves_used += *count;
            validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
        }
        Requirement::Missiles(count) => {
            local.missiles_used += *count;
            validate_missiles(local, cx.global).into()
        }
        Requirement::Supers(count) => {
            local.supers_used += *count;
            validate_supers(local, cx.global).into()
        }
        Requirement::PowerBombs(count) => {
            local.power_bombs_used += *count;
            validate_power_bombs(local, cx.global).into()
        }
        Requirement::GateGlitchLeniency { green, heated } => apply_gate_glitch_leniency(
            local,
            cx.global,
            *green,
            *heated,
            cx.difficulty,
            cx.game_data,
        )
        .into(),
        Requirement::HeatedDoorStuckLeniency { heat_frames } => {
            if !cx.global.inventory.items[Item::Varia as usize] {
                local.energy += (cx.difficulty.door_stuck_leniency as f32
                    * cx.difficulty.resource_multiplier
                    * *heat_frames as f32
                    / 4.0) as Capacity;
                validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
            } else {
                SimpleResult::Success
            }
        }
        Requirement::BombIntoCrystalFlashClipLeniency {} => {
            local.power_bombs_used += cx.difficulty.bomb_into_cf_leniency;
            validate_power_bombs(local, cx.global).into()
        }
        Requirement::JumpIntoCrystalFlashClipLeniency {} => {
            local.power_bombs_used += cx.difficulty.jump_into_cf_leniency;
            validate_power_bombs(local, cx.global).into()
        }
        Requirement::XModeSpikeHitLeniency {} => {
            local.energy +=
                cx.difficulty.spike_xmode_leniency * 60 / suit_damage_factor(&cx.global.inventory);
            validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
        }
        Requirement::XModeThornHitLeniency {} => {
            local.energy +=
                cx.difficulty.spike_xmode_leniency * 16 / suit_damage_factor(&cx.global.inventory);
            validate_energy(local, &cx.global.inventory, can_manage_reserves).into()
        }
        Requirement::MissilesAvailable(count) => {
            apply_missiles_available_req(local, cx.global, *count, cx.reverse).into()
        }
        Requirement::SupersAvailable(count) => {
            apply_supers_available_req(local, cx.global, *count, cx.reverse).into()
        }
        Requirement::PowerBombsAvailable(count) => {
            apply_power_bombs_available_req(local, cx.global, *count, cx.reverse).into()
        }
        Requirement::RegularEnergyAvailable(count) => {
            apply_regular_energy_available_req(local, cx.global, *count, cx.reverse).into()
        }
        Requirement::ReserveEnergyAvailable(count) => {
            apply_reserve_energy_available_req(local, cx.global, *count, cx.reverse).into()
        }
        Requirement::EnergyAvailable(count) => {
            apply_energy_available_req(local, cx.global, *count, cx.reverse).into()
        }
        Requirement::MissilesMissingAtMost(count) => apply_missiles_available_req(
            local,
            cx.global,
            cx.global.inventory.max_missiles - *count,
            cx.reverse,
        )
        .into(),
        Requirement::SupersMissingAtMost(count) => apply_supers_available_req(
            local,
            cx.global,
            cx.global.inventory.max_supers - *count,
            cx.reverse,
        )
        .into(),
        Requirement::PowerBombsMissingAtMost(count) => apply_power_bombs_available_req(
            local,
            cx.global,
            cx.global.inventory.max_power_bombs - *count,
            cx.reverse,
        )
        .into(),
        Requirement::RegularEnergyMissingAtMost(count) => apply_regular_energy_available_req(
            local,
            cx.global,
            cx.global.inventory.max_energy - *count,
            cx.reverse,
        )
        .into(),
        Requirement::ReserveEnergyMissingAtMost(count) => apply_reserve_energy_available_req(
            local,
            cx.global,
            cx.global.inventory.max_reserves - *count,
            cx.reverse,
        )
        .into(),
        Requirement::EnergyMissingAtMost(count) => apply_energy_available_req(
            local,
            cx.global,
            cx.global.inventory.max_energy + cx.global.inventory.max_reserves - *count,
            cx.reverse,
        )
        .into(),
        Requirement::MissilesCapacity(count) => (cx.global.inventory.max_missiles >= *count).into(),
        Requirement::SupersCapacity(count) => (cx.global.inventory.max_supers >= *count).into(),
        Requirement::PowerBombsCapacity(count) => {
            (cx.global.inventory.max_power_bombs >= *count).into()
        }
        Requirement::RegularEnergyCapacity(count) => {
            (cx.global.inventory.max_energy >= *count).into()
        }
        Requirement::ReserveEnergyCapacity(count) => {
            (cx.global.inventory.max_reserves >= *count).into()
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
            let drops = if cx.settings.quality_of_life_settings.buffed_drops {
                enemy_drops_buffed
            } else {
                enemy_drops
            };
            if let Some(new_local) = apply_farm_requirement(
                requirement,
                drops,
                cx.global,
                *local,
                cx.reverse,
                *full_energy,
                *full_missiles,
                *full_supers,
                *full_power_bombs,
                cx.settings,
                cx.difficulty,
                cx.game_data,
                cx.door_map,
                cx.locked_door_data,
                cx.objectives,
            ) {
                *local = new_local;
                SimpleResult::Success
            } else {
                SimpleResult::Failure
            }
        }
        Requirement::EnergyRefill(limit) => {
            let limit_reserves = max(0, *limit - cx.global.inventory.max_energy);
            if cx.reverse {
                if local.energy < *limit {
                    local.energy = 0;
                    local.farm_baseline_energy_used = 0;
                }
                if local.reserves_used <= limit_reserves {
                    local.reserves_used = 0;
                    local.farm_baseline_reserves_used = 0;
                }
            } else {
                if local.energy > cx.global.inventory.max_energy - limit {
                    local.energy = max(0, cx.global.inventory.max_energy - limit);
                    local.farm_baseline_energy_used = local.energy;
                }
                if local.reserves_used > cx.global.inventory.max_reserves - limit_reserves {
                    local.reserves_used = max(0, cx.global.inventory.max_reserves - limit_reserves);
                    local.farm_baseline_reserves_used = local.reserves_used;
                }
            }
            SimpleResult::Success
        }
        Requirement::RegularEnergyRefill(limit) => {
            if cx.reverse {
                if local.energy < *limit {
                    local.energy = 0;
                    local.farm_baseline_energy_used = 0;
                }
            } else if local.energy > cx.global.inventory.max_energy - limit {
                local.energy = max(0, cx.global.inventory.max_energy - limit);
                local.farm_baseline_energy_used = local.energy;
            }
            SimpleResult::Success
        }
        Requirement::ReserveRefill(limit) => {
            if cx.reverse {
                if local.reserves_used <= *limit {
                    local.reserves_used = 0;
                    local.farm_baseline_reserves_used = 0;
                }
            } else if local.reserves_used > cx.global.inventory.max_reserves - limit {
                local.reserves_used = max(0, cx.global.inventory.max_reserves - limit);
                local.farm_baseline_reserves_used = local.reserves_used;
            }
            SimpleResult::Success
        }
        Requirement::MissileRefill(limit) => {
            if cx.reverse {
                if local.missiles_used <= *limit {
                    local.missiles_used = 0;
                    local.farm_baseline_missiles_used = 0;
                }
            } else if local.missiles_used > cx.global.inventory.max_missiles - limit {
                local.missiles_used = max(0, cx.global.inventory.max_missiles - limit);
                local.farm_baseline_missiles_used = local.missiles_used;
            }
            SimpleResult::Success
        }
        Requirement::SuperRefill(limit) => {
            if cx.reverse {
                if local.supers_used <= *limit {
                    local.supers_used = 0;
                    local.farm_baseline_supers_used = 0;
                }
            } else if local.supers_used > cx.global.inventory.max_supers - limit {
                local.supers_used = max(0, cx.global.inventory.max_supers - limit);
                local.farm_baseline_supers_used = local.supers_used;
            }
            SimpleResult::Success
        }
        Requirement::PowerBombRefill(limit) => {
            if cx.reverse {
                if local.power_bombs_used <= *limit {
                    local.power_bombs_used = 0;
                    local.farm_baseline_power_bombs_used = 0;
                }
            } else if local.power_bombs_used > cx.global.inventory.max_power_bombs - limit {
                local.power_bombs_used = max(0, cx.global.inventory.max_power_bombs - limit);
                local.farm_baseline_power_bombs_used = local.power_bombs_used;
            }
            SimpleResult::Success
        }
        Requirement::AmmoStationRefill => {
            local.missiles_used = 0;
            local.farm_baseline_missiles_used = 0;
            if !cx.settings.other_settings.ultra_low_qol {
                local.supers_used = 0;
                local.farm_baseline_supers_used = 0;
                local.power_bombs_used = 0;
                local.farm_baseline_power_bombs_used = 0;
            }
            SimpleResult::Success
        }
        Requirement::AmmoStationRefillAll => (!cx.settings.other_settings.ultra_low_qol).into(),
        Requirement::EnergyStationRefill => {
            local.energy = 0;
            local.farm_baseline_energy_used = 0;
            if cx.settings.quality_of_life_settings.energy_station_reserves
                || cx
                    .settings
                    .quality_of_life_settings
                    .reserve_backward_transfer
            {
                local.reserves_used = 0;
                local.farm_baseline_reserves_used = 0;
            }
            SimpleResult::Success
        }
        Requirement::SupersDoubleDamageMotherBrain => {
            cx.settings.quality_of_life_settings.supers_double.into()
        }
        Requirement::ShinesparksCostEnergy => {
            cx.settings.other_settings.energy_free_shinesparks.into()
        }
        Requirement::RegularEnergyDrain(count) => {
            if cx.reverse {
                let amt = Capacity::max(0, local.energy - count + 1);
                local.reserves_used += amt;
                local.energy -= amt;
                (local.reserves_used <= cx.global.inventory.max_reserves).into()
            } else {
                local.energy = Capacity::max(local.energy, cx.global.inventory.max_energy - count);
                SimpleResult::Success
            }
        }
        Requirement::ReserveEnergyDrain(count) => {
            if cx.reverse {
                (local.reserves_used <= *count).into()
            } else {
                // TODO: Drained reserve energy could potentially be transferred into regular energy, but it wouldn't
                // be consistent with how "resourceAtMost" is currently defined.
                local.reserves_used = Capacity::max(
                    local.reserves_used,
                    cx.global.inventory.max_reserves - count,
                );
                SimpleResult::Success
            }
        }
        Requirement::MissileDrain(count) => {
            if cx.reverse {
                (local.missiles_used <= *count).into()
            } else {
                local.missiles_used = Capacity::max(
                    local.missiles_used,
                    cx.global.inventory.max_missiles - count,
                );
                SimpleResult::Success
            }
        }
        Requirement::ReserveTrigger {
            min_reserve_energy,
            max_reserve_energy,
            heated,
        } => {
            if cx.reverse {
                if local.reserves_used > 0 {
                    SimpleResult::Failure
                } else {
                    local.energy = 0;
                    let energy_needed = if *heated {
                        (local.energy * 4 + 2) / 3
                    } else {
                        local.energy
                    };
                    local.reserves_used = max(energy_needed + 1, *min_reserve_energy);
                    (local.reserves_used <= *max_reserve_energy
                        && local.reserves_used <= cx.global.inventory.max_reserves)
                        .into()
                }
            } else {
                let reserve_energy = min(
                    cx.global.inventory.max_reserves - local.reserves_used,
                    *max_reserve_energy,
                );
                let usable_reserve_energy = if *heated {
                    reserve_energy * 3 / 4
                } else {
                    reserve_energy
                };
                if reserve_energy >= *min_reserve_energy {
                    local.reserves_used = cx.global.inventory.max_reserves;
                    local.energy = max(0, cx.global.inventory.max_energy - usable_reserve_energy);
                    SimpleResult::Success
                } else {
                    SimpleResult::Failure
                }
            }
        }
        Requirement::EnemyKill { count, vul } => {
            apply_enemy_kill_requirement(cx.global, local, *count, vul).into()
        }
        Requirement::PhantoonFight {} => apply_phantoon_requirement(
            &cx.global.inventory,
            local,
            cx.difficulty.phantoon_proficiency,
            can_manage_reserves,
        )
        .into(),
        Requirement::DraygonFight {
            can_be_very_patient_tech_idx: can_be_very_patient_tech_id,
        } => apply_draygon_requirement(
            &cx.global.inventory,
            local,
            cx.difficulty.draygon_proficiency,
            can_manage_reserves,
            cx.difficulty.tech[*can_be_very_patient_tech_id],
        )
        .into(),
        Requirement::RidleyFight {
            can_be_patient_tech_idx,
            can_be_very_patient_tech_idx,
            can_be_extremely_patient_tech_idx,
            power_bombs,
            g_mode,
            stuck,
        } => apply_ridley_requirement(
            &cx.global.inventory,
            local,
            cx.difficulty.ridley_proficiency,
            can_manage_reserves,
            cx.difficulty.tech[*can_be_patient_tech_idx],
            cx.difficulty.tech[*can_be_very_patient_tech_idx],
            cx.difficulty.tech[*can_be_extremely_patient_tech_idx],
            *power_bombs,
            *g_mode,
            *stuck,
        )
        .into(),
        Requirement::BotwoonFight { second_phase } => apply_botwoon_requirement(
            &cx.global.inventory,
            local,
            cx.difficulty.botwoon_proficiency,
            *second_phase,
            can_manage_reserves,
        )
        .into(),
        Requirement::MotherBrain2Fight {
            can_be_very_patient_tech_id,
            r_mode,
        } => {
            if cx.settings.quality_of_life_settings.mother_brain_fight == MotherBrainFight::Skip {
                return SimpleResult::Success;
            }
            apply_mother_brain_2_requirement(
                &cx.global.inventory,
                local,
                cx.difficulty.mother_brain_proficiency,
                cx.settings.quality_of_life_settings.supers_double,
                can_manage_reserves,
                cx.difficulty.tech[*can_be_very_patient_tech_id],
                *r_mode,
            )
            .into()
        }
        Requirement::SpeedBall { used_tiles, heated } => {
            if !cx.difficulty.tech[cx.game_data.speed_ball_tech_idx]
                || !cx.global.inventory.items[Item::Morph as usize]
            {
                SimpleResult::Failure
            } else {
                let used_tiles = used_tiles.get();
                let tiles_limit = if *heated && !cx.global.inventory.items[Item::Varia as usize] {
                    get_heated_speedball_tiles(cx.difficulty)
                } else {
                    cx.difficulty.speed_ball_tiles
                };
                (cx.global.inventory.items[Item::SpeedBooster as usize]
                    && used_tiles >= tiles_limit)
                    .into()
            }
        }
        Requirement::GetBlueSpeed { used_tiles, heated } => {
            let used_tiles = used_tiles.get();
            let tiles_limit = if *heated && !cx.global.inventory.items[Item::Varia as usize] {
                cx.difficulty.heated_shine_charge_tiles
            } else {
                cx.difficulty.shine_charge_tiles
            };
            (cx.global.inventory.items[Item::SpeedBooster as usize] && used_tiles >= tiles_limit)
                .into()
        }
        Requirement::ShineCharge { used_tiles, heated } => {
            let used_tiles = used_tiles.get();
            let tiles_limit = if *heated && !cx.global.inventory.items[Item::Varia as usize] {
                cx.difficulty.heated_shine_charge_tiles
            } else {
                cx.difficulty.shine_charge_tiles
            };
            if cx.global.inventory.items[Item::SpeedBooster as usize] && used_tiles >= tiles_limit {
                if cx.reverse {
                    local.shinecharge_frames_remaining = 0;
                    if local.flash_suit {
                        return SimpleResult::Failure;
                    }
                } else {
                    local.shinecharge_frames_remaining =
                        180 - cx.difficulty.shinecharge_leniency_frames;
                    local.flash_suit = false;
                }
                SimpleResult::Success
            } else {
                SimpleResult::Failure
            }
        }
        Requirement::ShineChargeFrames(frames) => {
            if cx.reverse {
                local.shinecharge_frames_remaining += frames;
                (local.shinecharge_frames_remaining
                    <= 180 - cx.difficulty.shinecharge_leniency_frames)
                    .into()
            } else {
                local.shinecharge_frames_remaining -= frames;
                (local.shinecharge_frames_remaining >= 0).into()
            }
        }
        Requirement::Shinespark {
            frames,
            excess_frames,
            shinespark_tech_idx: shinespark_tech_id,
        } => {
            if cx.difficulty.tech[*shinespark_tech_id] {
                if cx.settings.other_settings.energy_free_shinesparks {
                    return SimpleResult::Success;
                }
                if cx.reverse {
                    if local.energy <= 28 {
                        if frames == excess_frames {
                            // If all frames are excess frames and energy is at 29 or lower, then the spark does not require any energy:
                            return SimpleResult::Success;
                        }
                        local.energy = 28 + frames - excess_frames;
                    } else {
                        local.energy += frames;
                    }
                    validate_energy_no_auto_reserve(local, cx.global, cx.game_data, cx.difficulty)
                        .into()
                } else {
                    if frames == excess_frames
                        && local.energy >= cx.global.inventory.max_energy - 29
                    {
                        // If all frames are excess frames and energy is at 29 or lower, then the spark does not require any energy:
                        return SimpleResult::Success;
                    }
                    local.energy += frames - excess_frames + 28;
                    if !validate_energy_no_auto_reserve(
                        local,
                        cx.global,
                        cx.game_data,
                        cx.difficulty,
                    ) {
                        return SimpleResult::Failure;
                    }
                    let energy_remaining = cx.global.inventory.max_energy - local.energy - 1;
                    local.energy += std::cmp::min(*excess_frames, energy_remaining);
                    local.energy -= 28;
                    SimpleResult::Success
                }
            } else {
                SimpleResult::Failure
            }
        }
        Requirement::GainFlashSuit => {
            #[allow(clippy::needless_bool_assign)]
            if cx.reverse {
                local.flash_suit = false;
            } else {
                local.flash_suit = true;
            }
            SimpleResult::Success
        }
        Requirement::NoFlashSuit => {
            if cx.reverse {
                (!local.flash_suit).into()
            } else {
                local.flash_suit = false;
                SimpleResult::Success
            }
        }
        &Requirement::UseFlashSuit {
            carry_flash_suit_tech_idx,
        } => {
            if !cx.difficulty.tech[carry_flash_suit_tech_idx] {
                // It isn't strictly necessary to check the tech here (since it already checked
                // when obtaining the flash suit), but it could affect Forced item placement.
                return SimpleResult::Failure;
            }
            if cx.reverse {
                local.flash_suit = true;
                SimpleResult::Success
            } else if !local.flash_suit {
                SimpleResult::Failure
            } else {
                local.flash_suit = false;
                SimpleResult::Success
            }
        }
        Requirement::DoorUnlocked { room_id, node_id } => {
            if let Some(locked_door_idx) = cx
                .locked_door_data
                .locked_door_node_map
                .get(&(*room_id, *node_id))
            {
                cx.global.doors_unlocked[*locked_door_idx].into()
            } else {
                SimpleResult::Success
            }
        }
        Requirement::DoorType {
            room_id,
            node_id,
            door_type,
        } => {
            let actual_door_type = if let Some(locked_door_idx) = cx
                .locked_door_data
                .locked_door_node_map
                .get(&(*room_id, *node_id))
            {
                cx.locked_door_data.locked_doors[*locked_door_idx].door_type
            } else {
                DoorType::Blue
            };
            (*door_type == actual_door_type).into()
        }
        Requirement::UnlockDoor {
            room_id,
            node_id,
            requirement_red,
            requirement_green,
            requirement_yellow,
            requirement_charge,
        } => {
            if let Some(locked_door_idx) = cx
                .locked_door_data
                .locked_door_node_map
                .get(&(*room_id, *node_id))
            {
                let door_type = cx.locked_door_data.locked_doors[*locked_door_idx].door_type;
                if cx.global.doors_unlocked[*locked_door_idx] {
                    return SimpleResult::Success;
                }
                match door_type {
                    DoorType::Blue => SimpleResult::Success,
                    DoorType::Red => apply_requirement_simple(requirement_red, local, cx),
                    DoorType::Green => apply_requirement_simple(requirement_green, local, cx),
                    DoorType::Yellow => apply_requirement_simple(requirement_yellow, local, cx),
                    DoorType::Beam(beam) => {
                        if has_beam(beam, &cx.global.inventory) {
                            if let BeamType::Charge = beam {
                                apply_requirement_simple(requirement_charge, local, cx)
                            } else {
                                SimpleResult::Success
                            }
                        } else {
                            SimpleResult::Failure
                        }
                    }
                    DoorType::Gray => {
                        panic!("Unexpected gray door while processing Requirement::UnlockDoor")
                    }
                    DoorType::Wall => SimpleResult::Failure,
                }
            } else {
                SimpleResult::Success
            }
        }
        &Requirement::ResetRoom { room_id, node_id } => {
            if local.cycle_frames > 0 {
                // We assume the it takes 400 frames to go through the door transition, shoot open the door, and return.
                // The actual time can vary based on room load time and whether fast doors are enabled.
                local.cycle_frames += 400;
            }

            let Some(&(mut other_room_id, mut other_node_id)) =
                cx.door_map.get(&(room_id, node_id))
            else {
                // When simulating logic for the logic pages, the `cx.door_map` may be empty,
                // but we still treat the requirement as satisfiable.
                return SimpleResult::Success;
            };

            if other_room_id == 321 {
                // Passing through the Toilet adds to the time taken to reset the room.
                if local.cycle_frames > 0 {
                    local.cycle_frames += 600;
                }
                let opposite_node_id = match other_node_id {
                    1 => 2,
                    2 => 1,
                    _ => panic!("unexpected Toilet node ID: {}", other_node_id),
                };
                (other_room_id, other_node_id) = cx.door_map[&(321, opposite_node_id)];
            }
            let reset_req =
                &cx.game_data.node_reset_room_requirement[&(other_room_id, other_node_id)];
            apply_requirement_simple(reset_req, local, cx)
        }
        Requirement::EscapeMorphLocation => (cx.settings.map_layout == "Vanilla").into(),
        Requirement::And(reqs) => {
            if cx.reverse {
                for req in reqs.iter().rev() {
                    match apply_requirement_simple(req, local, cx) {
                        SimpleResult::Failure => return SimpleResult::Failure,
                        SimpleResult::Success => {}
                        SimpleResult::_ExtraState(_) => todo!(),
                    }
                }
            } else {
                for req in reqs {
                    match apply_requirement_simple(req, local, cx) {
                        SimpleResult::Failure => return SimpleResult::Failure,
                        SimpleResult::Success => {}
                        SimpleResult::_ExtraState(_) => todo!(),
                    }
                }
            }
            SimpleResult::Success
        }
        Requirement::Or(reqs) => {
            let mut best_local = None;
            let mut best_cost = [f32::INFINITY; NUM_COST_METRICS];
            let orig_local = *local;
            for req in reqs {
                *local = orig_local;
                match apply_requirement_simple(req, local, cx) {
                    SimpleResult::Failure => continue,
                    SimpleResult::Success => {}
                    SimpleResult::_ExtraState(_) => todo!(),
                }
                let cost = compute_cost(local, &cx.global.inventory, cx.reverse);
                // TODO: Maybe do something better than just using the first cost metric.
                if cost[0] < best_cost[0] {
                    best_cost = cost;
                    best_local = Some(*local);
                }
            }
            if let Some(new_local) = best_local {
                *local = new_local;
                SimpleResult::Success
            } else {
                SimpleResult::Failure
            }
        }
    }
}

pub fn is_bireachable_state(
    global: &GlobalState,
    forward: LocalState,
    reverse: LocalState,
) -> bool {
    if forward.reserves_used + reverse.reserves_used > global.inventory.max_reserves {
        return false;
    }
    let forward_total_energy_used = forward.energy + forward.reserves_used;
    let reverse_total_energy_used = reverse.energy + reverse.reserves_used;
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
    if reverse.flash_suit && !forward.flash_suit {
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
    forward: &Traverser,
    reverse: &Traverser,
) -> Option<(usize, usize)> {
    for (forward_idx, &forward_state) in forward.lsr[vertex_id].local.iter().enumerate() {
        for (reverse_idx, &reverse_state) in reverse.lsr[vertex_id].local.iter().enumerate() {
            if is_bireachable_state(global, forward_state, reverse_state) {
                // A valid combination of forward & return routes has been found.
                return Some((forward_idx, reverse_idx));
            }
        }
    }
    None
}

// If the given vertex is reachable, returns an index (between 0 and NUM_COST_METRICS),
// indicating a forward route. Otherwise returns None.
pub fn get_one_way_reachable_idx(vertex_id: usize, forward: &Traverser) -> Option<usize> {
    if !forward.lsr[vertex_id].local.is_empty() {
        return Some(0);
    }
    None
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StepTrail {
    pub link_idx: LinkIdx,
    pub local_state: LocalState,
}

#[derive(Clone)]
pub struct TraversalUpdate {
    pub vertex_id: VertexId,
    pub old_lsr: LocalStateReducer<StepTrailId>,
}

#[derive(Clone)]
pub struct TraversalStep {
    pub updates: Vec<TraversalUpdate>,
    pub start_step_trail_idx: usize,
    pub step_num: usize,
    pub global_state: GlobalState,
}

#[derive(Clone)]
pub struct Traverser {
    pub reverse: bool,
    pub step_trails: Vec<StepTrail>,
    pub lsr: Vec<LocalStateReducer<StepTrailId>>,
    pub step: TraversalStep,
    pub past_steps: Vec<TraversalStep>,
}

impl Traverser {
    pub fn new(num_vertices: usize, reverse: bool, global_state: &GlobalState) -> Self {
        Self {
            reverse,
            step_trails: Vec::with_capacity(num_vertices * 10),
            lsr: vec![LocalStateReducer::new(); num_vertices],
            step: TraversalStep {
                updates: vec![],
                start_step_trail_idx: 0,
                step_num: 0,
                global_state: global_state.clone(),
            },
            past_steps: vec![],
        }
    }

    fn add_trail(&mut self, vertex_id: VertexId, lsr: LocalStateReducer<StepTrailId>) {
        let u = TraversalUpdate {
            vertex_id,
            old_lsr: self.lsr[vertex_id].clone(),
        };
        self.lsr[vertex_id] = lsr;
        self.step.updates.push(u);
    }

    pub fn add_origin(
        &mut self,
        init_local: LocalState,
        start_vertex_id: usize,
        global: &GlobalState,
    ) {
        let mut lsr = LocalStateReducer::<StepTrailId>::new();
        lsr.push(init_local, -1, &global.inventory, self.reverse);
        self.add_trail(start_vertex_id, lsr);
    }

    pub fn finish_step(&mut self, step_num: usize) {
        let mut step = TraversalStep {
            updates: vec![],
            start_step_trail_idx: self.step_trails.len(),
            step_num: 0,
            global_state: self.step.global_state.clone(),
        };
        std::mem::swap(&mut self.step, &mut step);
        step.step_num = step_num;
        self.past_steps.push(step);
    }

    pub fn pop_step(&mut self) {
        let step = self.past_steps.pop().unwrap();
        for u in step.updates.iter().rev() {
            self.lsr[u.vertex_id] = u.old_lsr.clone();
        }
        self.step_trails.truncate(step.start_step_trail_idx);
        self.step.start_step_trail_idx = self.step_trails.len();
    }

    pub fn traverse(
        &mut self,
        base_links_data: &LinksDataGroup,
        seed_links_data: &LinksDataGroup,
        global: &GlobalState,
        settings: &RandomizerSettings,
        difficulty: &DifficultyConfig,
        game_data: &GameData,
        door_map: &HashMap<(RoomId, NodeId), (RoomId, NodeId)>,
        locked_door_data: &LockedDoorData,
        objectives: &[Objective],
        step_num: usize,
    ) {
        self.step.global_state = global.clone();
        let mut modified_vertices: HashSet<usize> = HashSet::new();

        for v in 0..self.lsr.len() {
            if !self.lsr[v].local.is_empty() {
                modified_vertices.insert(v);
            }
        }

        let base_links_by_src: &Vec<Vec<(StepTrailId, Link)>> = if self.reverse {
            &base_links_data.links_by_dst
        } else {
            &base_links_data.links_by_src
        };
        let seed_links_by_src: &Vec<Vec<(StepTrailId, Link)>> = if self.reverse {
            &seed_links_data.links_by_dst
        } else {
            &seed_links_data.links_by_src
        };

        let cx = TraversalContext {
            global,
            reverse: self.reverse,
            settings,
            difficulty,
            game_data,
            door_map,
            locked_door_data,
            objectives,
        };
        while !modified_vertices.is_empty() {
            let mut new_modified_vertices: HashSet<usize> = HashSet::new();
            let modified_vertices_vec = {
                // Process the vertices in sorted order, to make the traversal deterministic.
                // This also improves performance, possibly due to better locality:
                // neighboring vertices would tend to have their data stored next to each other.
                let mut m: Vec<usize> = modified_vertices.into_iter().collect();
                m.sort();
                m
            };
            for &src_id in &modified_vertices_vec {
                let mut src_local_arr = self.lsr[src_id].local.clone();
                for (i, local) in src_local_arr.iter_mut().enumerate() {
                    local.prev_trail_id = self.lsr[src_id].trail_ids[i];
                }
                let all_src_links = base_links_by_src[src_id]
                    .iter()
                    .chain(seed_links_by_src[src_id].iter());
                for &(link_idx, ref link) in all_src_links {
                    let dst_id = link.to_vertex_id;
                    let mut local_arr = src_local_arr.clone();
                    let mut any_improvement: bool = false;
                    let mut new_lsr = self.lsr[dst_id].clone();
                    local_arr = apply_link(link, local_arr, &cx);
                    for local in local_arr {
                        let new_trail_id = self.step_trails.len() as StepTrailId;
                        if new_lsr.push(local, new_trail_id, &cx.global.inventory, cx.reverse) {
                            let new_step_trail = StepTrail {
                                local_state: local,
                                link_idx,
                            };
                            self.step_trails.push(new_step_trail);
                            any_improvement = true;
                        }
                    }
                    if any_improvement {
                        self.add_trail(dst_id, new_lsr);
                        new_modified_vertices.insert(dst_id);
                    }
                }
            }
            modified_vertices = new_modified_vertices;
        }
        self.finish_step(step_num);
    }
}

pub fn get_spoiler_trail_ids(
    traverser: &Traverser,
    vertex_id: usize,
    idx: usize,
) -> Vec<StepTrailId> {
    let mut trail_id = traverser.lsr[vertex_id].trail_ids[idx];
    let mut steps: Vec<StepTrailId> = Vec::new();
    while trail_id != -1 {
        let step_trail = &traverser.step_trails[trail_id as usize];
        steps.push(trail_id);
        trail_id = step_trail.local_state.prev_trail_id;
    }
    steps.reverse();
    steps
}
