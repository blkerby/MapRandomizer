use std::{
    cmp::{Reverse, max, min},
    collections::BinaryHeap,
    fmt::Debug,
};

use arrayvec::ArrayVec;
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use crate::{
    randomize::{DifficultyConfig, LockedDoor},
    settings::{DisableETankSetting, MotherBrainFight, Objective, RandomizerSettings, WallJump},
};
use maprando_game::{
    BeamType, Capacity, DoorType, EnemyDrop, EnemyVulnerabilities, GameData, Item, Link, LinkIdx,
    LinkLength, LinksDataGroup, NodeId, Requirement, RoomId, StepTrailId,
    TECH_ID_CAN_SUITLESS_LAVA_DIVE, VertexId,
};
use maprando_logic::{
    GlobalState, Inventory, LocalState, ResourceLevel,
    boss_requirements::{
        apply_botwoon_requirement, apply_draygon_requirement, apply_mother_brain_2_requirement,
        apply_phantoon_requirement, apply_ridley_requirement,
    },
    helpers::suit_damage_factor,
};

type CostMetricIdx = u8;

fn apply_enemy_kill_requirement(
    global: &GlobalState,
    local: &mut LocalState,
    count: Capacity,
    vul: &EnemyVulnerabilities,
    reverse: bool,
) -> bool {
    // Prioritize using weapons that do not require ammo:
    if global.weapon_mask & vul.non_ammo_vulnerabilities != 0 {
        return true;
    }

    let mut hp = vul.hp; // HP per enemy
    let mut missiles_used = 0;

    // Next use Missiles:
    if vul.missile_damage > 0 {
        let missiles_available = local.missiles_available(&global.inventory, reverse);
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
        let supers_available = local.supers_available(&global.inventory, reverse);
        let supers_to_use_per_enemy = max(
            0,
            min(
                supers_available / count,
                (hp + vul.super_damage - 1) / vul.super_damage,
            ),
        );
        hp -= supers_to_use_per_enemy * vul.super_damage as Capacity;
        assert!(local.use_supers(supers_to_use_per_enemy * count, &global.inventory, reverse));
    }

    // Finally, use Power Bombs (overkill is possible, where we could have used fewer Missiles or Supers, but we ignore that):
    if vul.power_bomb_damage > 0 && global.inventory.items[Item::Morph as usize] {
        let pbs_available = local.power_bombs_available(&global.inventory, reverse);
        let pbs_to_use = max(
            0,
            min(
                pbs_available,
                (hp + vul.power_bomb_damage - 1) / vul.power_bomb_damage,
            ),
        );
        hp -= pbs_to_use * vul.power_bomb_damage as Capacity;
        // Power bombs hit all enemies in the group, so we do not multiply by the count.
        assert!(local.use_power_bombs(pbs_to_use, &global.inventory, reverse));
    }

    if hp > 0 {
        return false;
    }

    // If the enemy would be overkilled, refund some of the missile shots, if applicable:
    if vul.missile_damage > 0 {
        let missiles_overkill = -hp / vul.missile_damage;
        missiles_used = max(0, missiles_used - missiles_overkill * count);
    }
    assert!(local.use_missiles(missiles_used, &global.inventory, reverse));
    true
}

// Ended up abandoning the original purpose of introducing this struct, but keeping it for possible future use.
#[derive(Clone)]
pub struct CostConfig {}

pub const NUM_COST_METRICS: usize = 5;
type CostValue = i32;

fn compute_cost(
    local: &LocalState,
    inventory: &Inventory,
    _cost_config: &CostConfig,
    reverse: bool,
) -> [CostValue; NUM_COST_METRICS] {
    let mut energy_cost = match local.energy() {
        ResourceLevel::Consumed(x) => x as CostValue * 2,
        ResourceLevel::Remaining(x) => (inventory.max_energy - x) as CostValue * 2 + 1,
    };
    let mut reserve_cost = match local.reserves() {
        ResourceLevel::Consumed(x) => x as CostValue * 2,
        ResourceLevel::Remaining(x) => (inventory.max_reserves - x) as CostValue * 2 + 1,
    };
    let mut missiles_cost = match local.missiles() {
        ResourceLevel::Consumed(x) => x as CostValue * 2,
        ResourceLevel::Remaining(x) => (inventory.max_missiles - x) as CostValue * 2 + 1,
    };
    let mut supers_cost = match local.supers() {
        ResourceLevel::Consumed(x) => x as CostValue * 2,
        ResourceLevel::Remaining(x) => (inventory.max_supers - x) as CostValue * 2 + 1,
    };
    let mut power_bombs_cost = match local.power_bombs() {
        ResourceLevel::Consumed(x) => x as CostValue * 2,
        ResourceLevel::Remaining(x) => (inventory.max_power_bombs - x) as CostValue * 2 + 1,
    };
    let mut shinecharge_cost = -if local.flash_suit > 0 {
        // For the purposes of the cost metrics, treat flash suit as equivalent
        // to a large amount of shinecharge frames remaining:
        181 + (local.flash_suit as CostValue)
    } else {
        local.shinecharge_frames_remaining as CostValue
    };
    let mut blue_suit_cost = -(local.blue_suit as CostValue);
    if reverse {
        energy_cost = -energy_cost;
        reserve_cost = -reserve_cost;
        missiles_cost = -missiles_cost;
        supers_cost = -supers_cost;
        power_bombs_cost = -power_bombs_cost;
        shinecharge_cost = -shinecharge_cost;
        blue_suit_cost = -blue_suit_cost;
    }
    let cycle_frames_cost = local.cycle_frames as CostValue;
    let total_energy_cost = energy_cost + reserve_cost;
    let total_ammo_cost = missiles_cost + 10 * supers_cost + 20 * power_bombs_cost;

    let energy_sensitive_cost_metric = 100000 * total_energy_cost
        + 100 * reserve_cost
        + total_ammo_cost
        + shinecharge_cost
        + blue_suit_cost
        + cycle_frames_cost;
    let ammo_sensitive_cost_metric = total_energy_cost
        + 100000 * total_ammo_cost
        + shinecharge_cost
        + blue_suit_cost
        + cycle_frames_cost;
    let shinecharge_sensitive_cost_metric = total_energy_cost
        + total_ammo_cost
        + 100000 * shinecharge_cost
        + blue_suit_cost
        + cycle_frames_cost;
    let blue_suit_energy_sensitive_cost_metric = 2000 * total_energy_cost
        + total_ammo_cost
        + shinecharge_cost
        + 5000000 * blue_suit_cost
        + cycle_frames_cost;
    let blue_suit_ammo_sensitive_cost_metric = total_energy_cost
        + 2000 * total_ammo_cost
        + shinecharge_cost
        + 5000000 * blue_suit_cost
        + cycle_frames_cost;
    [
        energy_sensitive_cost_metric,
        ammo_sensitive_cost_metric,
        shinecharge_sensitive_cost_metric,
        blue_suit_energy_sensitive_cost_metric,
        blue_suit_ammo_sensitive_cost_metric,
    ]
}

fn apply_blue_gate_glitch_leniency(
    local: &mut LocalState,
    global: &GlobalState,
    heated: bool,
    difficulty: &DifficultyConfig,
    reverse: bool,
) -> bool {
    if heated && !global.inventory.items[Item::Varia as usize] {
        let energy_used = (difficulty.gate_glitch_leniency as f32
            * difficulty.resource_multiplier
            * 60.0) as Capacity;
        if !local.use_energy(energy_used, true, &global.inventory, reverse) {
            return false;
        }
    }
    let missiles_available = local.missiles_available(&global.inventory, reverse);
    if missiles_available >= difficulty.gate_glitch_leniency {
        local.use_missiles(difficulty.gate_glitch_leniency, &global.inventory, reverse)
    } else {
        assert!(local.use_missiles(missiles_available, &global.inventory, reverse));
        local.use_supers(
            difficulty.gate_glitch_leniency - missiles_available,
            &global.inventory,
            reverse,
        )
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
    reverse: bool,
) -> bool {
    let varia = global.inventory.items[Item::Varia as usize];
    if varia {
        true
    } else if !difficulty.tech[game_data.heat_run_tech_idx] {
        false
    } else {
        let energy_used = if simple {
            (frames as f32 / 4.0).ceil() as Capacity
        } else {
            (frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity
        };
        local.use_energy(energy_used, true, &global.inventory, reverse)
    }
}

fn apply_suitless_heat_frames(
    frames: Capacity,
    local: &mut LocalState,
    global: &GlobalState,
    game_data: &GameData,
    difficulty: &DifficultyConfig,
    simple: bool,
    reverse: bool,
) -> bool {
    if !difficulty.tech[game_data.heat_run_tech_idx] {
        false
    } else {
        let energy_used = if simple {
            (frames as f32 / 4.0).ceil() as Capacity
        } else {
            (frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity
        };
        local.use_energy(energy_used, true, &global.inventory, reverse)
    }
}

fn get_enemy_drop_energy_value(
    drop: &EnemyDrop,
    local: &mut LocalState,
    reverse: bool,
    buffed_drops: bool,
    full_ammo: bool,
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

    if !reverse || full_ammo {
        // In theory, health bomb could also be modeled, except that the "heatFramesWithEnergyDrops" requirement
        // does not identify exactly when the drops are collected. In cases where it significantly matters, this
        // could be handled by having a boolean property in "heatFramesWithEnergyDrops" to indicate that the
        // drops are obtained at the very end of the heat frames? We ignore it for now.
        if local.power_bombs() == ResourceLevel::Consumed(0)
            || (reverse && full_ammo && p_pb > 0.05)
        {
            p_small += p_pb * rel_small;
            p_large += p_pb * rel_large;
            p_missile += p_pb * rel_missile;
            if reverse {
                local.power_bombs = ResourceLevel::Consumed(0).into();
            }
        }
        if local.supers() == ResourceLevel::Consumed(0) || (reverse && full_ammo && p_super > 0.05)
        {
            p_small += p_super * rel_small;
            p_large += p_super * rel_large;
            p_missile += p_super * rel_missile;
            if reverse {
                local.supers = ResourceLevel::Consumed(0).into();
            }
        }
        if local.missiles() == ResourceLevel::Consumed(0)
            || (reverse && full_ammo && p_missile > 0.05)
        {
            p_small += p_missile * p_small / (p_small + p_large);
            p_large += p_missile * p_large / (p_small + p_large);
            if reverse {
                local.missiles = ResourceLevel::Consumed(0).into();
            }
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
    let w_nothing = drop.nothing_weight.get();
    let w_small = if full_energy {
        0.0
    } else {
        drop.small_energy_weight.get()
    };
    let w_large = if full_energy {
        0.0
    } else {
        drop.large_energy_weight.get()
    };
    let w_missile = if full_missiles {
        0.0
    } else {
        drop.missile_weight.get()
    };
    let w_super = if full_supers {
        0.0
    } else {
        drop.super_weight.get()
    };
    let w_pb = if full_power_bombs {
        0.0
    } else {
        drop.power_bomb_weight.get()
    };
    let w_total = w_nothing + w_small + w_large + w_missile + w_super + w_pb;

    let p_small = w_small / w_total;
    let p_large = w_large / w_total;
    let p_missile = w_missile / w_total;
    let p_super = w_super;
    let p_pb = w_pb;

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
) -> SimpleResult {
    let varia = global.inventory.items[Item::Varia as usize];
    if varia {
        true.into()
    } else if !difficulty.tech[game_data.heat_run_tech_idx] {
        false.into()
    } else {
        let full_ammo_settings: Vec<bool> = if reverse {
            vec![false, true]
        } else {
            vec![false]
        };
        let mut state_output: Vec<LocalState> = vec![];
        for full_ammo in full_ammo_settings {
            let mut new_local = *local;
            let mut total_drop_value = 0;
            for drop in drops {
                total_drop_value += get_enemy_drop_energy_value(
                    drop,
                    &mut new_local,
                    reverse,
                    settings.quality_of_life_settings.buffed_drops,
                    full_ammo,
                )
            }
            let heat_energy =
                (frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity;
            total_drop_value = Capacity::min(total_drop_value, heat_energy);
            if reverse {
                new_local.refill_energy(total_drop_value, true, &global.inventory, reverse);
                if new_local.use_energy(heat_energy, true, &global.inventory, reverse) {
                    state_output.push(new_local);
                }
            } else {
                if !new_local.use_energy(heat_energy, true, &global.inventory, reverse) {
                    continue;
                }
                new_local.refill_energy(total_drop_value, true, &global.inventory, reverse);
                state_output.push(new_local);
            }
        }
        match state_output.len() {
            0 => false.into(),
            1 => {
                *local = state_output[0];
                true.into()
            }
            2 => {
                *local = state_output[0];
                SimpleResult::ExtraState(state_output[1])
            }
            _ => {
                panic!("internal error");
            }
        }
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
) -> SimpleResult {
    let varia = global.inventory.items[Item::Varia as usize];
    let gravity = global.inventory.items[Item::Gravity as usize];
    if gravity && varia {
        true.into()
    } else if !difficulty.tech[game_data.tech_isv.index_by_key[&TECH_ID_CAN_SUITLESS_LAVA_DIVE]] {
        false.into()
    } else {
        let full_ammo_settings: Vec<bool> = if reverse {
            vec![false, true]
        } else {
            vec![false]
        };
        let mut state_output: Vec<LocalState> = vec![];
        for full_ammo in full_ammo_settings {
            let mut new_local = *local;
            let mut total_drop_value = 0;
            for drop in drops {
                total_drop_value += get_enemy_drop_energy_value(
                    drop,
                    &mut new_local,
                    reverse,
                    settings.quality_of_life_settings.buffed_drops,
                    full_ammo,
                )
            }
            let lava_energy = if gravity || varia {
                (frames as f32 * difficulty.resource_multiplier / 4.0).ceil() as Capacity
            } else {
                (frames as f32 * difficulty.resource_multiplier / 2.0).ceil() as Capacity
            };
            total_drop_value = Capacity::min(total_drop_value, lava_energy);
            if reverse {
                new_local.refill_energy(total_drop_value, true, &global.inventory, reverse);
                if new_local.use_energy(lava_energy, true, &global.inventory, reverse) {
                    state_output.push(new_local);
                }
            } else {
                if !new_local.use_energy(lava_energy, true, &global.inventory, reverse) {
                    continue;
                }
                new_local.refill_energy(total_drop_value, true, &global.inventory, reverse);
                state_output.push(new_local);
            }
        }
        match state_output.len() {
            0 => false.into(),
            1 => {
                *local = state_output[0];
                true.into()
            }
            2 => {
                *local = state_output[0];
                SimpleResult::ExtraState(state_output[1])
            }
            _ => {
                panic!("internal error");
            }
        }
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
    // if link.from_vertex_id == 17615
    //     && link.strat_id == Some(206)
    //     && cx.door_map[&(187, 2)] == (89, 1)
    //     && !cx.reverse
    // {
    //     info!("debug");
    // }
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
    for x in &mut local {
        x.length += link.length;
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
    cost_config: &CostConfig,
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
            objectives,
            cost_config
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
                    cost_config,
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
                    cost_config,
                );
            }
        }
        _ => {}
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

pub fn simple_cost_config() -> CostConfig {
    CostConfig {}
}

pub fn update_farm_baseline(local: &mut LocalState, inventory: &Inventory, reverse: bool) {
    if local.farm_baseline_energy_available(inventory, reverse)
        > local.energy_available(inventory, false, reverse)
    {
        local.farm_baseline_energy = local.energy;
    }
    if local.farm_baseline_reserves_available(inventory, reverse)
        > local.reserves_available(inventory, reverse)
    {
        local.farm_baseline_reserves = local.reserves;
    }
    if local.farm_baseline_missiles_available(inventory, reverse)
        > local.missiles_available(inventory, reverse)
    {
        local.farm_baseline_missiles = local.missiles;
    }
    if local.farm_baseline_supers_available(inventory, reverse)
        > local.supers_available(inventory, reverse)
    {
        local.farm_baseline_supers = local.supers;
    }
    if local.farm_baseline_power_bombs_available(inventory, reverse)
        > local.power_bombs_available(inventory, reverse)
    {
        local.farm_baseline_power_bombs = local.power_bombs;
    }
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
    start_local.energy = ResourceLevel::full(reverse).into();
    start_local.reserves = ResourceLevel::full(reverse).into();
    start_local.missiles = ResourceLevel::full(reverse).into();
    start_local.supers = ResourceLevel::full(reverse).into();
    start_local.power_bombs = ResourceLevel::full(reverse).into();
    let cost_config = simple_cost_config();
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
        &cost_config,
    );
    let end_local = end_local_result?;
    if end_local.cycle_frames < 100 {
        panic!("bad farm: expected cycle_frames >= 100: end_local={end_local:#?},\n req={req:#?}");
    }
    let cycle_frames = (end_local.cycle_frames - start_local.cycle_frames) as f32;
    let cycle_energy = (end_local.energy_available(&global.inventory, true, reverse)
        - start_local.energy_available(&global.inventory, true, reverse))
        as f32;
    let cycle_missiles = (end_local.missiles_available(&global.inventory, reverse)
        - start_local.missiles_available(&global.inventory, reverse))
        as f32;
    let cycle_supers = (end_local.supers_available(&global.inventory, reverse)
        - start_local.supers_available(&global.inventory, reverse)) as f32;
    let cycle_pbs = (end_local.power_bombs_available(&global.inventory, reverse)
        - start_local.power_bombs_available(&global.inventory, reverse)) as f32;
    let patience_frames = difficulty.farm_time_limit * 60.0;
    let num_cycles = (patience_frames / cycle_frames).floor() as i32;

    // We assume, somewhat simplisticly, that the maximum drop rate for each resource can be
    // obtained, by filling up on the other resource types.
    let drop_energy: f32 = get_total_enemy_drop_values(
        drops,
        false,
        true,
        true,
        true,
        settings.quality_of_life_settings.buffed_drops,
    )[0];
    let drop_missiles: f32 = get_total_enemy_drop_values(
        drops,
        true,
        false,
        true,
        true,
        settings.quality_of_life_settings.buffed_drops,
    )[1];
    let [_, _, drop_supers, drop_pbs] = get_total_enemy_drop_values(
        drops,
        false,
        false,
        false,
        false,
        settings.quality_of_life_settings.buffed_drops,
    );

    let net_energy = ((drop_energy + cycle_energy) * num_cycles as f32) as Capacity;
    let net_missiles = ((drop_missiles + cycle_missiles) * num_cycles as f32) as Capacity;
    let net_supers = ((drop_supers + cycle_supers) * num_cycles as f32) as Capacity;
    let net_pbs = ((drop_pbs + cycle_pbs) * num_cycles as f32) as Capacity;

    if net_energy < 0 || net_missiles < 0 || net_supers < 0 || net_pbs < 0 {
        return None;
    }

    let mut new_local = local;
    if !reverse {
        // In forward traversal, we first apply the requirement, then refill the resources gained from farming.
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
            &cost_config,
        )?;
        new_local.cycle_frames = 0;
    }

    update_farm_baseline(&mut new_local, &global.inventory, reverse);
    new_local.energy = new_local.farm_baseline_energy;
    new_local.reserves = new_local.farm_baseline_reserves;
    new_local.missiles = new_local.farm_baseline_missiles;
    new_local.supers = new_local.farm_baseline_supers;
    new_local.power_bombs = new_local.farm_baseline_power_bombs;

    new_local.refill_energy(net_energy, true, &global.inventory, reverse);
    new_local.refill_missiles(net_missiles, &global.inventory, reverse);
    new_local.refill_supers(net_supers, &global.inventory, reverse);
    new_local.refill_power_bombs(net_pbs, &global.inventory, reverse);

    if reverse {
        // In reverse traversal, we first refill the resources gained from farming, then apply the requirement.
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
            &cost_config,
        )?;
        new_local.cycle_frames = 0;
    }

    if local.energy_available(&global.inventory, false, reverse)
        > new_local.energy_available(&global.inventory, false, reverse)
    {
        new_local.energy = local.energy;
    }
    if local.reserves_available(&global.inventory, reverse)
        > new_local.reserves_available(&global.inventory, reverse)
    {
        new_local.reserves = local.reserves;
    }
    if local.missiles_available(&global.inventory, reverse)
        > new_local.missiles_available(&global.inventory, reverse)
    {
        new_local.missiles = local.missiles;
    }
    if local.supers_available(&global.inventory, reverse)
        > new_local.supers_available(&global.inventory, reverse)
    {
        new_local.supers = local.supers;
    }
    if local.power_bombs_available(&global.inventory, reverse)
        > new_local.power_bombs_available(&global.inventory, reverse)
    {
        new_local.power_bombs = local.power_bombs;
    }

    if net_energy >= global.pool_inventory.max_energy + global.pool_inventory.max_reserves {
        new_local.energy = ResourceLevel::full_energy(reverse).into();
        new_local.reserves = ResourceLevel::full(reverse).into();
        new_local.farm_baseline_energy = new_local.energy;
        new_local.farm_baseline_reserves = new_local.reserves;
    }
    if net_missiles >= global.pool_inventory.max_missiles {
        new_local.missiles = ResourceLevel::full(reverse).into();
        new_local.farm_baseline_missiles = new_local.missiles;
    }
    if net_supers >= global.pool_inventory.max_supers {
        new_local.supers = ResourceLevel::full(reverse).into();
        new_local.farm_baseline_supers = new_local.supers;
    }
    if net_pbs >= global.pool_inventory.max_power_bombs {
        new_local.power_bombs = ResourceLevel::full(reverse).into();
        new_local.farm_baseline_power_bombs = new_local.power_bombs;
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
    pub best_cost_values: [CostValue; NUM_COST_METRICS],
    pub best_cost_idxs: [CostMetricIdx; NUM_COST_METRICS],
}

impl<T: Copy + Debug> Default for LocalStateReducer<T> {
    fn default() -> Self {
        Self {
            local: ArrayVec::new(),
            trail_ids: ArrayVec::new(),
            best_cost_values: [CostValue::MAX; NUM_COST_METRICS],
            best_cost_idxs: [0; NUM_COST_METRICS],
        }
    }
}

impl<T: Copy + Debug> LocalStateReducer<T> {
    pub fn push(
        &mut self,
        local: LocalState,
        inventory: &Inventory,
        trail_id: T,
        cost_config: &CostConfig,
        reverse: bool,
    ) -> bool {
        let cost = compute_cost(&local, inventory, cost_config, reverse);
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
    cost_config: CostConfig,
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
    cost_config: &CostConfig,
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
        cost_config: cost_config.clone(),
    };
    match apply_requirement_simple(req, &mut local, &cx) {
        SimpleResult::Failure => None,
        SimpleResult::Success => Some(local),
        SimpleResult::ExtraState(_) => Some(local),
    }
}

enum SimpleResult {
    Failure,
    Success,
    ExtraState(LocalState),
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
            if cx.reverse {
                for r in sub_reqs.iter().rev() {
                    local = apply_requirement_complex(r, local, cx);
                    if local.is_empty() {
                        break;
                    }
                }
            } else {
                for r in sub_reqs {
                    local = apply_requirement_complex(r, local, cx);
                    if local.is_empty() {
                        break;
                    }
                }
            }
            local
        }
        Requirement::Or(sub_reqs) => {
            let mut reducer: LocalStateReducer<()> = LocalStateReducer::default();
            for r in sub_reqs {
                for loc in apply_requirement_complex(r, local.clone(), cx) {
                    reducer.push(loc, &cx.global.inventory, (), &cx.cost_config, cx.reverse);
                }
            }
            reducer.local
        }
        _ => {
            let mut reducer: LocalStateReducer<()> = LocalStateReducer::default();
            for mut loc in local {
                match apply_requirement_simple(req, &mut loc, cx) {
                    SimpleResult::Failure => {}
                    SimpleResult::Success => {
                        reducer.push(loc, &cx.global.inventory, (), &cx.cost_config, cx.reverse);
                    }
                    SimpleResult::ExtraState(extra_state) => {
                        reducer.push(loc, &cx.global.inventory, (), &cx.cost_config, cx.reverse);
                        reducer.push(
                            extra_state,
                            &cx.global.inventory,
                            (),
                            &cx.cost_config,
                            cx.reverse,
                        );
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
        Requirement::DisableableETank => (cx.settings.quality_of_life_settings.disableable_etanks
            != DisableETankSetting::Off)
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
        Requirement::HeatFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            apply_heat_frames(
                frames,
                local,
                cx.global,
                cx.game_data,
                cx.difficulty,
                false,
                cx.reverse,
            )
            .into()
        }
        Requirement::SuitlessHeatFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            apply_suitless_heat_frames(
                frames,
                local,
                cx.global,
                cx.game_data,
                cx.difficulty,
                false,
                cx.reverse,
            )
            .into()
        }
        Requirement::SimpleHeatFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            apply_heat_frames(
                frames,
                local,
                cx.global,
                cx.game_data,
                cx.difficulty,
                true,
                cx.reverse,
            )
            .into()
        }
        Requirement::HeatFramesWithEnergyDrops(frames, enemy_drops, enemy_drops_buffed) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            let drops = if cx.settings.quality_of_life_settings.buffed_drops {
                enemy_drops_buffed
            } else {
                enemy_drops
            };
            apply_heat_frames_with_energy_drops(
                frames,
                drops,
                local,
                cx.global,
                cx.game_data,
                cx.settings,
                cx.difficulty,
                cx.reverse,
            )
        }
        Requirement::LavaFramesWithEnergyDrops(frames, enemy_drops, enemy_drops_buffed) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            let drops = if cx.settings.quality_of_life_settings.buffed_drops {
                enemy_drops_buffed
            } else {
                enemy_drops
            };
            apply_lava_frames_with_energy_drops(
                frames,
                drops,
                local,
                cx.global,
                cx.game_data,
                cx.settings,
                cx.difficulty,
                cx.reverse,
            )
        }
        Requirement::MainHallElevatorFrames => {
            if cx.settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(
                    188,
                    local,
                    cx.global,
                    cx.game_data,
                    cx.difficulty,
                    true,
                    cx.reverse,
                )
                .into()
            } else if !cx.global.inventory.items[Item::Varia as usize]
                && cx.global.inventory.max_energy < 149
            {
                SimpleResult::Failure
            } else {
                apply_heat_frames(
                    436,
                    local,
                    cx.global,
                    cx.game_data,
                    cx.difficulty,
                    true,
                    cx.reverse,
                )
                .into()
            }
        }
        Requirement::EquipmentScreenCycleFrames => {
            if cx.settings.quality_of_life_settings.fast_pause_menu {
                local.cycle_frames += 300;
            } else {
                local.cycle_frames += 150;
            }
            SimpleResult::Success
        }
        Requirement::LowerNorfairElevatorDownFrames => {
            if cx.settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(
                    30,
                    local,
                    cx.global,
                    cx.game_data,
                    cx.difficulty,
                    true,
                    cx.reverse,
                )
                .into()
            } else {
                apply_heat_frames(
                    60,
                    local,
                    cx.global,
                    cx.game_data,
                    cx.difficulty,
                    true,
                    cx.reverse,
                )
                .into()
            }
        }
        Requirement::LowerNorfairElevatorUpFrames => {
            if cx.settings.quality_of_life_settings.fast_elevators {
                apply_heat_frames(
                    48,
                    local,
                    cx.global,
                    cx.game_data,
                    cx.difficulty,
                    true,
                    cx.reverse,
                )
                .into()
            } else {
                apply_heat_frames(
                    108,
                    local,
                    cx.global,
                    cx.game_data,
                    cx.difficulty,
                    true,
                    cx.reverse,
                )
                .into()
            }
        }
        Requirement::LavaFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            let varia = cx.global.inventory.items[Item::Varia as usize];
            let gravity = cx.global.inventory.items[Item::Gravity as usize];
            if gravity && varia {
                SimpleResult::Success
            } else if gravity || varia {
                let energy_used =
                    (frames as f32 * cx.difficulty.resource_multiplier / 4.0).ceil() as Capacity;
                local
                    .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                    .into()
            } else {
                let energy_used =
                    (frames as f32 * cx.difficulty.resource_multiplier / 2.0).ceil() as Capacity;
                local
                    .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                    .into()
            }
        }
        Requirement::GravitylessLavaFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            let varia = cx.global.inventory.items[Item::Varia as usize];
            let energy_used = if varia {
                (frames as f32 * cx.difficulty.resource_multiplier / 4.0).ceil() as Capacity
            } else {
                (frames as f32 * cx.difficulty.resource_multiplier / 2.0).ceil() as Capacity
            };
            local
                .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::AcidFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            let energy_used = (frames as f32 * cx.difficulty.resource_multiplier * 1.5
                / suit_damage_factor(&cx.global.inventory) as f32)
                .ceil() as Capacity;
            local
                .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::GravitylessAcidFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            let varia = cx.global.inventory.items[Item::Varia as usize];
            let energy_used = if varia {
                (frames as f32 * cx.difficulty.resource_multiplier * 0.75).ceil() as Capacity
            } else {
                (frames as f32 * cx.difficulty.resource_multiplier * 1.5).ceil() as Capacity
            };
            local
                .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::MetroidFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            let energy_used = (frames as f32 * cx.difficulty.resource_multiplier * 0.75
                / suit_damage_factor(&cx.global.inventory) as f32)
                .ceil() as Capacity;
            local
                .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::CycleFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            local.cycle_frames +=
                (frames as f32 * cx.difficulty.resource_multiplier).ceil() as Capacity;
            SimpleResult::Success
        }
        Requirement::SimpleCycleFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            local.cycle_frames += frames;
            SimpleResult::Success
        }
        Requirement::Damage {
            unit_energy,
            quantity,
        } => {
            let quantity = quantity.resolve(&cx.difficulty.numerics);
            let base_energy = unit_energy * quantity;
            let energy = base_energy / suit_damage_factor(&cx.global.inventory);
            if energy >= cx.global.inventory.max_energy
                && !cx.difficulty.tech[cx.game_data.manage_reserves_tech_idx]
            {
                SimpleResult::Failure
            } else {
                local
                    .use_energy(energy, true, &cx.global.inventory, cx.reverse)
                    .into()
            }
        }
        Requirement::Energy(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .use_energy(count, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::RegularEnergy(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            // For now, we assume reserve energy can be converted to regular energy, so this is
            // implemented the same as the Energy requirement above.
            local
                .use_energy(count, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::ReserveEnergy(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .use_reserve_energy(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::Missiles(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .use_missiles(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::Supers(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .use_supers(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::PowerBombs(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .use_power_bombs(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::BlueGateGlitchLeniency { heated } => {
            apply_blue_gate_glitch_leniency(local, cx.global, *heated, cx.difficulty, cx.reverse)
                .into()
        }
        Requirement::HeatedDoorStuckLeniency { heat_frames } => {
            if !cx.global.inventory.items[Item::Varia as usize] {
                let energy_used = (cx.difficulty.door_stuck_leniency as f32
                    * cx.difficulty.resource_multiplier
                    * *heat_frames as f32
                    / 4.0) as Capacity;
                local
                    .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                    .into()
            } else {
                SimpleResult::Success
            }
        }
        Requirement::MissilesAvailable(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_missiles_available(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::SupersAvailable(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_supers_available(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::PowerBombsAvailable(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_power_bombs_available(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::RegularEnergyAvailable(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            // Note: It could make more sense to treat reserve energy satisfying
            // the requirement for regular energy (by manually transfering it over);
            // but currently all uses are after an auto-reserve trigger, where no reserves
            // are available anyway.
            local
                .ensure_energy_available(count, false, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::ReserveEnergyAvailable(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_reserves_available(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::EnergyAvailable(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_energy_available(count, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::MissilesMissingAtMost(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_missiles_missing_at_most(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::SupersMissingAtMost(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_supers_missing_at_most(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::PowerBombsMissingAtMost(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_power_bombs_missing_at_most(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::RegularEnergyMissingAtMost(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_energy_missing_at_most(count, false, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::ReserveEnergyMissingAtMost(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_reserves_missing_at_most(count, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::EnergyMissingAtMost(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            local
                .ensure_energy_missing_at_most(count, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::MissilesCapacity(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            (cx.global.inventory.max_missiles >= count).into()
        }
        Requirement::SupersCapacity(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            (cx.global.inventory.max_supers >= count).into()
        }
        Requirement::PowerBombsCapacity(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            (cx.global.inventory.max_power_bombs >= count).into()
        }
        Requirement::RegularEnergyCapacity(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            (cx.global.inventory.max_energy >= count).into()
        }
        Requirement::ReserveEnergyCapacity(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            (cx.global.inventory.max_reserves >= count).into()
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
            let limit = limit.resolve(&cx.difficulty.numerics);
            let limit_energy = min(limit, cx.global.inventory.max_energy);
            let limit_reserves = max(
                0,
                min(
                    limit - cx.global.inventory.max_energy,
                    cx.global.inventory.max_reserves,
                ),
            );
            let energy_remaining = local.energy_remaining(&cx.global.inventory, false);
            let reserves_remaining = local.reserves_remaining(&cx.global.inventory);
            if cx.reverse {
                if energy_remaining <= limit_energy {
                    local.energy = ResourceLevel::Remaining(1).into();
                    local.farm_baseline_energy = local.energy;
                }
                if reserves_remaining <= limit_reserves {
                    local.reserves = ResourceLevel::Remaining(0).into();
                    local.farm_baseline_reserves = local.reserves;
                }
            } else {
                if limit >= cx.global.pool_inventory.max_energy {
                    local.energy = ResourceLevel::Consumed(0).into();
                    local.farm_baseline_energy = local.energy;
                } else if energy_remaining < limit_energy {
                    local.energy = ResourceLevel::Remaining(limit_energy).into();
                    local.farm_baseline_energy = local.energy;
                }
                if limit
                    >= cx.global.pool_inventory.max_energy + cx.global.pool_inventory.max_reserves
                {
                    local.reserves = ResourceLevel::Consumed(0).into();
                    local.farm_baseline_reserves = local.reserves;
                } else if reserves_remaining < limit_reserves {
                    local.reserves = ResourceLevel::Remaining(limit_reserves).into();
                    local.farm_baseline_reserves = local.reserves;
                }
            }
            SimpleResult::Success
        }
        Requirement::RegularEnergyRefill(limit) => {
            let limit = limit.resolve(&cx.difficulty.numerics);
            let energy_remaining = local.energy_remaining(&cx.global.inventory, false);
            if limit >= cx.global.pool_inventory.max_energy {
                local.energy = ResourceLevel::full_energy(cx.reverse).into();
                local.farm_baseline_energy = local.energy;
            } else if cx.reverse {
                if energy_remaining <= limit {
                    local.energy = ResourceLevel::Remaining(1).into();
                    local.farm_baseline_energy = ResourceLevel::Remaining(1).into();
                }
            } else if energy_remaining < limit {
                local.energy =
                    ResourceLevel::Remaining(min(limit, cx.global.inventory.max_energy)).into();
                local.farm_baseline_energy = local.energy;
            }
            SimpleResult::Success
        }
        Requirement::ReserveRefill(limit) => {
            let limit = limit.resolve(&cx.difficulty.numerics);
            let reserves_remaining = local.reserves_remaining(&cx.global.inventory);
            if limit >= cx.global.pool_inventory.max_reserves {
                local.reserves = ResourceLevel::full(cx.reverse).into();
                local.farm_baseline_reserves = local.reserves;
            } else if cx.reverse {
                if reserves_remaining <= limit {
                    local.reserves = ResourceLevel::Remaining(0).into();
                    local.farm_baseline_reserves = ResourceLevel::Remaining(0).into();
                }
            } else if reserves_remaining < limit {
                local.reserves =
                    ResourceLevel::Remaining(min(limit, cx.global.inventory.max_reserves)).into();
                local.farm_baseline_reserves = local.reserves;
            }
            SimpleResult::Success
        }
        Requirement::MissileRefill(limit) => {
            let limit = limit.resolve(&cx.difficulty.numerics);
            let missiles_remaining = local.missiles_remaining(&cx.global.inventory);
            if limit >= cx.global.pool_inventory.max_missiles {
                local.missiles = ResourceLevel::full(cx.reverse).into();
                local.farm_baseline_missiles = local.missiles;
            } else if cx.reverse {
                if missiles_remaining <= limit {
                    local.missiles = ResourceLevel::Remaining(0).into();
                    local.farm_baseline_missiles = ResourceLevel::Remaining(0).into();
                }
            } else if missiles_remaining < limit {
                local.missiles =
                    ResourceLevel::Remaining(min(limit, cx.global.inventory.max_missiles)).into();
                local.farm_baseline_missiles = local.missiles;
            }
            SimpleResult::Success
        }
        Requirement::SuperRefill(limit) => {
            let limit = limit.resolve(&cx.difficulty.numerics);
            let supers_remaining = local.supers_remaining(&cx.global.inventory);
            if limit >= cx.global.pool_inventory.max_supers {
                local.supers = ResourceLevel::full(cx.reverse).into();
                local.farm_baseline_supers = local.supers;
            } else if cx.reverse {
                if supers_remaining <= limit {
                    local.supers = ResourceLevel::Remaining(0).into();
                    local.farm_baseline_supers = ResourceLevel::Remaining(0).into();
                }
            } else if supers_remaining < limit {
                local.supers =
                    ResourceLevel::Remaining(min(limit, cx.global.inventory.max_supers)).into();
                local.farm_baseline_supers = local.supers;
            }
            SimpleResult::Success
        }
        Requirement::PowerBombRefill(limit) => {
            let limit = limit.resolve(&cx.difficulty.numerics);
            let power_bombs_remaining = local.power_bombs_remaining(&cx.global.inventory);
            if limit >= cx.global.pool_inventory.max_power_bombs {
                local.power_bombs = ResourceLevel::full(cx.reverse).into();
                local.farm_baseline_power_bombs = local.power_bombs;
            } else if cx.reverse {
                if power_bombs_remaining <= limit {
                    local.power_bombs = ResourceLevel::Remaining(0).into();
                    local.farm_baseline_power_bombs = ResourceLevel::Remaining(0).into();
                }
            } else if power_bombs_remaining < limit {
                local.power_bombs =
                    ResourceLevel::Remaining(min(limit, cx.global.inventory.max_power_bombs))
                        .into();
                local.farm_baseline_power_bombs = local.power_bombs;
            }
            SimpleResult::Success
        }
        Requirement::AmmoStationRefill => {
            local.missiles = ResourceLevel::full(cx.reverse).into();
            local.farm_baseline_missiles = local.missiles;
            if !cx.settings.other_settings.ultra_low_qol {
                local.supers = ResourceLevel::full(cx.reverse).into();
                local.farm_baseline_supers = local.supers;
                local.power_bombs = ResourceLevel::full(cx.reverse).into();
                local.farm_baseline_power_bombs = local.power_bombs
            }
            SimpleResult::Success
        }
        Requirement::AmmoStationRefillAll => (!cx.settings.other_settings.ultra_low_qol).into(),
        Requirement::EnergyStationRefill => {
            local.energy = ResourceLevel::full_energy(cx.reverse).into();
            local.farm_baseline_energy = local.energy;
            if cx.settings.quality_of_life_settings.energy_station_reserves
                || cx
                    .settings
                    .quality_of_life_settings
                    .reserve_backward_transfer
            {
                local.reserves = ResourceLevel::full(cx.reverse).into();
                local.farm_baseline_reserves = local.reserves;
            }
            SimpleResult::Success
        }
        Requirement::SupersDoubleDamageMotherBrain => {
            cx.settings.quality_of_life_settings.supers_double.into()
        }
        Requirement::ShinesparksCostEnergy => {
            cx.settings.other_settings.energy_free_shinesparks.into()
        }
        Requirement::AllItemsSpawn => cx.settings.quality_of_life_settings.all_items_spawn.into(),
        Requirement::AcidChozoWithoutSpaceJump => {
            cx.settings.quality_of_life_settings.acid_chozo.into()
        }
        Requirement::KraidCameraFix => (!cx.settings.other_settings.ultra_low_qol).into(),
        Requirement::CrocomireCameraFix => (!cx.settings.other_settings.ultra_low_qol).into(),
        Requirement::RegularEnergyDrain(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            let energy_remaining = local.energy_remaining(&cx.global.inventory, false);
            if cx.reverse {
                let amt = Capacity::max(0, energy_remaining - count);
                local
                    .use_reserve_energy(amt, &cx.global.inventory, cx.reverse)
                    .into()
            } else {
                local.energy =
                    ResourceLevel::Remaining(Capacity::min(count, energy_remaining)).into();
                SimpleResult::Success
            }
        }
        Requirement::ReserveEnergyDrain(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            let reserves_remaining = local.reserves_remaining(&cx.global.inventory);
            if cx.reverse {
                (reserves_remaining <= count).into()
            } else {
                local.reserves =
                    ResourceLevel::Remaining(Capacity::min(count, reserves_remaining)).into();
                SimpleResult::Success
            }
        }
        Requirement::MissileDrain(count) => {
            let count = count.resolve(&cx.difficulty.numerics);
            let missiles_remaining = local.missiles_remaining(&cx.global.inventory);
            if cx.reverse {
                (missiles_remaining <= count).into()
            } else {
                local.missiles =
                    ResourceLevel::Remaining(Capacity::min(count, missiles_remaining)).into();
                SimpleResult::Success
            }
        }
        Requirement::ReserveTrigger {
            min_reserve_energy,
            max_reserve_energy,
            heated,
        } => {
            let min_reserve_energy = min_reserve_energy.resolve(&cx.difficulty.numerics);
            let max_reserve_energy = max_reserve_energy.resolve(&cx.difficulty.numerics);
            local
                .auto_reserve_trigger(
                    min_reserve_energy,
                    max_reserve_energy,
                    &cx.global.inventory,
                    *heated,
                    cx.reverse,
                )
                .into()
        }
        Requirement::EnemyKill { count, vul } => {
            apply_enemy_kill_requirement(cx.global, local, *count, vul, cx.reverse).into()
        }
        Requirement::PhantoonFight {} => apply_phantoon_requirement(
            &cx.global.inventory,
            local,
            cx.difficulty.phantoon_proficiency,
            cx.reverse,
        )
        .into(),
        Requirement::DraygonFight {
            can_be_patient_tech_idx,
            can_be_very_patient_tech_idx,
            can_be_extremely_patient_tech_idx,
        } => apply_draygon_requirement(
            &cx.global.inventory,
            local,
            cx.difficulty.draygon_proficiency,
            cx.difficulty.tech[*can_be_patient_tech_idx],
            cx.difficulty.tech[*can_be_very_patient_tech_idx],
            cx.difficulty.tech[*can_be_extremely_patient_tech_idx],
            cx.reverse,
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
            cx.difficulty.tech[*can_be_patient_tech_idx],
            cx.difficulty.tech[*can_be_very_patient_tech_idx],
            cx.difficulty.tech[*can_be_extremely_patient_tech_idx],
            *power_bombs,
            *g_mode,
            *stuck,
            cx.reverse,
        )
        .into(),
        Requirement::BotwoonFight { second_phase } => apply_botwoon_requirement(
            &cx.global.inventory,
            local,
            cx.difficulty.botwoon_proficiency,
            *second_phase,
            cx.reverse,
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
                cx.difficulty.tech[*can_be_very_patient_tech_id],
                *r_mode,
                cx.reverse,
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
                if used_tiles < tiles_limit {
                    return SimpleResult::Failure;
                }
                if !cx.global.inventory.items[Item::SpeedBooster as usize]
                    && !cx.global.inventory.items[Item::BlueBooster as usize]
                {
                    return SimpleResult::Failure;
                }
                SimpleResult::Success
            }
        }
        Requirement::GetBlueSpeed { used_tiles, heated } => {
            let used_tiles = used_tiles.get();
            let tiles_limit = if *heated && !cx.global.inventory.items[Item::Varia as usize] {
                cx.difficulty.heated_shine_charge_tiles
            } else {
                cx.difficulty.shine_charge_tiles
            };
            if used_tiles < tiles_limit {
                return SimpleResult::Failure;
            }
            if !cx.global.inventory.items[Item::SpeedBooster as usize]
                && !cx.global.inventory.items[Item::BlueBooster as usize]
                && !cx.global.inventory.items[Item::SparkBooster as usize]
            {
                return SimpleResult::Failure;
            }
            if cx.reverse {
                if local.blue_suit > 0 {
                    return SimpleResult::Failure;
                }
            } else {
                local.blue_suit = 0;
            }
            SimpleResult::Success
        }
        Requirement::ShineCharge { used_tiles, heated } => {
            let used_tiles = used_tiles.get();
            let tiles_limit = if *heated && !cx.global.inventory.items[Item::Varia as usize] {
                cx.difficulty.heated_shine_charge_tiles
            } else {
                cx.difficulty.shine_charge_tiles
            };
            if used_tiles < tiles_limit {
                return SimpleResult::Failure;
            }
            if cx.global.inventory.items[Item::SpeedBooster as usize]
                || cx.global.inventory.items[Item::SparkBooster as usize]
            {
                if cx.reverse {
                    local.shinecharge_frames_remaining = 0;
                    if local.flash_suit > 0 || local.blue_suit > 0 {
                        return SimpleResult::Failure;
                    }
                } else {
                    // We measure shinecharge frames starting from 181 (rather than 180) because the
                    // the sm-json-data is written in a way where 180 shinecharge frames can be consumed
                    // while being in logic, while 1 frame must still be remaining in order to activate a shinespark.
                    // Essentially the shineChargeFrames can be understood as including the first frame of the shinespark.
                    // This is a bit awkward; if this gets changed in the sm-json-data at some point, we could adapt here.
                    local.shinecharge_frames_remaining =
                        181 - cx.difficulty.shinecharge_leniency_frames;
                    local.flash_suit = 0;
                    local.blue_suit = 0;
                }
                SimpleResult::Success
            } else if cx.global.inventory.items[Item::BlueBooster as usize] {
                // With Blue Booster, shinecharging is still possible for temporary blue,
                // but it isn't possible to use it to shinespark. We represent this
                // by treating it as a shinecharge with zero shinecharge frames remaining.
                if cx.reverse {
                    if local.flash_suit > 0
                        || local.blue_suit > 0
                        || local.shinecharge_frames_remaining > 0
                    {
                        return SimpleResult::Failure;
                    }
                } else {
                    local.shinecharge_frames_remaining = 0;
                    local.flash_suit = 0;
                    local.blue_suit = 0;
                }
                SimpleResult::Success
            } else {
                SimpleResult::Failure
            }
        }
        Requirement::ShineChargeFrames(frames) => {
            let frames = frames.resolve(&cx.difficulty.numerics);
            if cx.reverse {
                local.shinecharge_frames_remaining += frames;
                (local.shinecharge_frames_remaining
                    <= 181 - cx.difficulty.shinecharge_leniency_frames)
                    .into()
            } else {
                local.shinecharge_frames_remaining -= frames;
                (local.shinecharge_frames_remaining >= 1).into()
            }
        }
        &Requirement::Shinespark {
            frames,
            excess_frames,
            shinespark_tech_idx: shinespark_tech_id,
        } => {
            if cx.difficulty.tech[shinespark_tech_id] {
                if cx.settings.other_settings.energy_free_shinesparks {
                    return SimpleResult::Success;
                }
                let can_manage_reserves = cx.difficulty.tech[cx.game_data.manage_reserves_tech_idx];
                let regular_energy_remaining = local.energy_remaining(&cx.global.inventory, false);
                let energy_remaining =
                    local.energy_remaining(&cx.global.inventory, can_manage_reserves);
                let min_frames = frames - excess_frames;
                if cx.reverse {
                    // Require at least 1 shinecharge frame remaining.
                    local.shinecharge_frames_remaining = 1;
                    if regular_energy_remaining <= 29
                        && let ResourceLevel::Remaining(_) = local.energy()
                    {
                        if frames == excess_frames {
                            // If all frames are excess frames and energy is at 29 or lower, then the spark does not require any energy:
                            return SimpleResult::Success;
                        }
                        local
                            .ensure_energy_available(
                                29 + min_frames,
                                false,
                                &cx.global.inventory,
                                cx.reverse,
                            )
                            .into()
                    } else {
                        // TODO: a second case could be considered, in which some reserve energy would be
                        // expected to be transferred after the spark. This would involve outputting a
                        // second LocalState, and we would need cost metrics designed to handle a trade-off
                        // in regular vs. reserve energy.
                        local
                            .use_energy(
                                frames,
                                can_manage_reserves,
                                &cx.global.inventory,
                                cx.reverse,
                            )
                            .into()
                    }
                } else {
                    if local.shinecharge_frames_remaining <= 0 {
                        // Shinesparking requires at least 1 shinecharge frame remaining.
                        // Note: we do not reset shinecharge frames here, since the logic may
                        // have multiple `shinespark` requirements in a row.
                        return SimpleResult::Failure;
                    }
                    if frames == excess_frames && regular_energy_remaining <= 29 {
                        // If all frames are excess frames and energy is at 29 or lower, then the spark does not require any energy:
                        return SimpleResult::Success;
                    }
                    if energy_remaining < 29 + min_frames {
                        return SimpleResult::Failure;
                    }
                    if regular_energy_remaining >= 29 + frames {
                        local
                            .use_energy(
                                frames,
                                can_manage_reserves,
                                &cx.global.inventory,
                                cx.reverse,
                            )
                            .into()
                    } else {
                        let reserves_needed =
                            Capacity::max(0, 29 + min_frames - regular_energy_remaining);
                        local.energy = ResourceLevel::Remaining(29).into();
                        local
                            .use_reserve_energy(reserves_needed, &cx.global.inventory, cx.reverse)
                            .into()
                    }
                }
            } else {
                SimpleResult::Failure
            }
        }
        Requirement::DoorTransition => {
            if cx.reverse {
                if local.flash_suit > 0 {
                    local.flash_suit = local.flash_suit.saturating_add(1);
                }
                if local.blue_suit > 0 {
                    local.blue_suit = local.blue_suit.saturating_add(1);
                }
            } else {
                local.flash_suit = local.flash_suit.saturating_sub(1);
                local.blue_suit = local.blue_suit.saturating_sub(1);
            }
            SimpleResult::Success
        }
        Requirement::GainFlashSuit => {
            if cx.reverse {
                local.flash_suit = 0;
            } else {
                local.flash_suit = cx.settings.skill_assumption_settings.flash_suit_distance;
            }
            SimpleResult::Success
        }
        Requirement::NoFlashSuit => {
            if cx.reverse {
                (local.flash_suit == 0).into()
            } else {
                local.flash_suit = 0;
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
                if local.flash_suit > 0 {
                    return SimpleResult::Failure;
                }
                local.flash_suit = 1;
                local.shinecharge_frames_remaining = 0;
                SimpleResult::Success
            } else if local.flash_suit == 0 {
                SimpleResult::Failure
            } else {
                local.flash_suit = 0;
                // Set shinecharge frames remaining to the max, to allow `comeInShinecharged`
                // strats to be satisfied by a flash suit.
                // (And at least 1 shinecharge frame is required in order to satisfy a `shinespark` requirement.)
                local.shinecharge_frames_remaining =
                    181 - cx.difficulty.shinecharge_leniency_frames;
                SimpleResult::Success
            }
        }
        Requirement::GainBlueSuit => {
            if cx.reverse {
                local.blue_suit = 0;
            } else {
                local.blue_suit = cx.settings.skill_assumption_settings.blue_suit_distance;
            }
            SimpleResult::Success
        }
        Requirement::NoBlueSuit => {
            if cx.reverse {
                (local.blue_suit == 0).into()
            } else {
                local.blue_suit = 0;
                SimpleResult::Success
            }
        }
        &Requirement::HaveBlueSuit {
            carry_blue_suit_tech_idx,
        } => {
            if !cx.difficulty.tech[carry_blue_suit_tech_idx] {
                // It isn't strictly necessary to check the tech here (since it already checked
                // when obtaining the blue suit), but it could affect Forced item placement.
                return SimpleResult::Failure;
            }
            if cx.reverse {
                local.blue_suit = 1;
                SimpleResult::Success
            } else {
                (local.blue_suit != 0).into()
            }
        }
        &Requirement::BlueSuitShineCharge {
            carry_blue_suit_tech_idx,
        } => {
            if !cx.difficulty.tech[carry_blue_suit_tech_idx] {
                // It isn't strictly necessary to check the tech here (since it already checked
                // when obtaining the blue suit), but it could affect Forced item placement.
                return SimpleResult::Failure;
            }
            if cx.reverse {
                if local.blue_suit != 0 {
                    return SimpleResult::Failure;
                }
                local.blue_suit = 1;
                local.shinecharge_frames_remaining = 0;
                SimpleResult::Success
            } else if local.blue_suit == 0 {
                SimpleResult::Failure
            } else {
                local.blue_suit = 0;
                // Set shinecharge frames remaining to the max, to allow `comeInShinecharged`
                // strats to be satisfied by a blue suit.
                local.shinecharge_frames_remaining =
                    181 - cx.difficulty.shinecharge_leniency_frames;
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
                        SimpleResult::ExtraState(_) => todo!(),
                    }
                }
            } else {
                for req in reqs {
                    match apply_requirement_simple(req, local, cx) {
                        SimpleResult::Failure => return SimpleResult::Failure,
                        SimpleResult::Success => {}
                        SimpleResult::ExtraState(_) => todo!(),
                    }
                }
            }
            SimpleResult::Success
        }
        Requirement::Or(reqs) => {
            let mut best_local = None;
            let mut best_cost = [CostValue::MAX; NUM_COST_METRICS];
            let orig_local = *local;
            for req in reqs {
                *local = orig_local;
                match apply_requirement_simple(req, local, cx) {
                    SimpleResult::Failure => continue,
                    SimpleResult::Success => {}
                    SimpleResult::ExtraState(_) => todo!(),
                }
                let cost = compute_cost(local, &cx.global.inventory, &cx.cost_config, cx.reverse);
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
    mut forward: LocalState,
    mut reverse: LocalState,
) -> bool {
    update_farm_baseline(&mut forward, &global.inventory, false);
    update_farm_baseline(&mut reverse, &global.inventory, true);

    // At the end, only use a farm in either the forward or reverse direction, not both.
    if forward.farm_baseline_reserves_remaining(&global.inventory)
        < reverse.reserves_remaining(&global.inventory)
        && forward.reserves_remaining(&global.inventory)
            < reverse.farm_baseline_reserves_remaining(&global.inventory)
    {
        return false;
    }
    if forward.farm_baseline_energy_remaining(&global.inventory, true)
        < reverse.energy_remaining(&global.inventory, true)
        && forward.energy_remaining(&global.inventory, true)
            < reverse.farm_baseline_energy_remaining(&global.inventory, true)
    {
        return false;
    }
    if forward.farm_baseline_missiles_remaining(&global.inventory)
        < reverse.missiles_remaining(&global.inventory)
        && forward.missiles_remaining(&global.inventory)
            < reverse.farm_baseline_missiles_remaining(&global.inventory)
    {
        return false;
    }
    if forward.farm_baseline_supers_remaining(&global.inventory)
        < reverse.supers_remaining(&global.inventory)
        && forward.supers_remaining(&global.inventory)
            < reverse.farm_baseline_supers_remaining(&global.inventory)
    {
        return false;
    }
    if forward.farm_baseline_power_bombs_remaining(&global.inventory)
        < reverse.power_bombs_remaining(&global.inventory)
        && forward.power_bombs_remaining(&global.inventory)
            < reverse.farm_baseline_power_bombs_remaining(&global.inventory)
    {
        return false;
    }
    if reverse.shinecharge_frames_remaining > forward.shinecharge_frames_remaining {
        return false;
    }
    if reverse.flash_suit > forward.flash_suit {
        return false;
    }
    if reverse.blue_suit > forward.blue_suit {
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

// If the given vertex is bireachable, returns a pair of trail IDs,
// indicating which forward route and backward route, respectively, combine to give a successful full route.
// Otherwise returns None.
pub fn get_short_bireachable_trails(
    global: &GlobalState,
    vertex_ids: &[VertexId],
    forward: &Traverser,
    reverse: &Traverser,
    forward_trails_by_vertex: &HashMap<VertexId, Vec<StepTrailId>>,
    reverse_trails_by_vertex: &HashMap<VertexId, Vec<StepTrailId>>,
) -> Option<(StepTrailId, StepTrailId)> {
    let mut best_length: u32 = u32::MAX;
    let mut best_trail_pair: Option<(StepTrailId, StepTrailId)> = None;
    for &vertex_id in vertex_ids {
        for &forward_trail_id in &forward_trails_by_vertex[&vertex_id] {
            for &reverse_trail_id in &reverse_trails_by_vertex[&vertex_id] {
                if forward_trail_id == -1 || reverse_trail_id == -1 {
                    continue;
                }
                let forward_state = forward.step_trails[forward_trail_id as usize].local_state;
                let reverse_state = reverse.step_trails[reverse_trail_id as usize].local_state;
                let combined_length = forward_state.length + reverse_state.length;
                if combined_length >= best_length {
                    continue;
                }
                if is_bireachable_state(global, forward_state, reverse_state) {
                    // A valid combination of forward & return routes has been found.
                    best_length = combined_length;
                    best_trail_pair = Some((forward_trail_id, reverse_trail_id));
                }
            }
        }
    }
    best_trail_pair
}

// If the given vertex is reachable, returns an index (between 0 and NUM_COST_METRICS),
// indicating a forward route. Otherwise returns None.
pub fn get_one_way_reachable_idx(vertex_id: usize, forward: &Traverser) -> Option<usize> {
    if !forward.lsr[vertex_id].local.is_empty() {
        return Some(0);
    }
    None
}

// If the given vertex is reachable, returns a step trail ID for a shortest length trail
// for reaching the vertex. Otherwise returns None.
pub fn get_short_one_way_reachable_trail(
    vertex_ids: &[VertexId],
    forward: &Traverser,
    forward_trails_by_vertex: &HashMap<VertexId, Vec<StepTrailId>>,
) -> Option<StepTrailId> {
    let mut best_length: u32 = u32::MAX;
    let mut best_trail: Option<StepTrailId> = None;
    for &vertex_id in vertex_ids {
        for &forward_trail_id in &forward_trails_by_vertex[&vertex_id] {
            let forward_state = forward.step_trails[forward_trail_id as usize].local_state;
            let new_length = forward_state.length;
            if new_length < best_length {
                best_length = new_length;
                best_trail = Some(forward_trail_id);
            }
        }
    }
    best_trail
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StepTrail {
    pub vertex_id: VertexId,
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
    pub initial_local_state: LocalState,
    pub cost_config: CostConfig,
    pub step_trails: Vec<StepTrail>,
    pub lsr: Vec<LocalStateReducer<StepTrailId>>,
    pub step: TraversalStep,
    pub past_steps: Vec<TraversalStep>,
}

impl Traverser {
    pub fn new(
        num_vertices: usize,
        reverse: bool,
        initial_local_state: LocalState,
        global_state: &GlobalState,
    ) -> Self {
        Self {
            reverse,
            initial_local_state,
            step_trails: Vec::with_capacity(num_vertices * 10),
            lsr: vec![LocalStateReducer::default(); num_vertices],
            step: TraversalStep {
                updates: vec![],
                start_step_trail_idx: 0,
                step_num: 0,
                global_state: global_state.clone(),
            },
            past_steps: vec![],
            cost_config: simple_cost_config(),
        }
    }

    fn add_trail(&mut self, vertex_id: VertexId) {
        let u = TraversalUpdate {
            vertex_id,
            old_lsr: self.lsr[vertex_id].clone(),
        };
        self.step.updates.push(u);
    }

    pub fn add_origin(
        &mut self,
        init_local: LocalState,
        inventory: &Inventory,
        start_vertex_id: usize,
    ) {
        let mut lsr = LocalStateReducer::<StepTrailId>::default();
        lsr.push(init_local, inventory, -1, &self.cost_config, self.reverse);
        self.add_trail(start_vertex_id);
        self.lsr[start_vertex_id] = lsr;
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

    pub fn unfinish_step(&mut self) {
        self.step = self.past_steps.pop().unwrap();
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
            cost_config: self.cost_config.clone(),
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
                    let old_lsr = &self.lsr[dst_id];
                    local_arr = apply_link(link, local_arr, &cx);
                    if local_arr.is_empty() {
                        continue;
                    }
                    let mut new_lsr = LocalStateReducer::default();
                    for i in 0..old_lsr.local.len() {
                        // Rebuild the LocalStateReducer in order to update costs, which
                        // may have changed due to new inventory.
                        new_lsr.push(
                            old_lsr.local[i],
                            &cx.global.inventory,
                            old_lsr.trail_ids[i],
                            &cx.cost_config,
                            cx.reverse,
                        );
                    }
                    for local in local_arr {
                        let new_trail_id = self.step_trails.len() as StepTrailId;
                        if new_lsr.push(
                            local,
                            &cx.global.inventory,
                            new_trail_id,
                            &cx.cost_config,
                            cx.reverse,
                        ) {
                            let new_step_trail = StepTrail {
                                vertex_id: dst_id,
                                local_state: local,
                                link_idx,
                            };
                            self.step_trails.push(new_step_trail);
                            any_improvement = true;
                        }
                    }
                    if any_improvement {
                        self.add_trail(dst_id);
                        self.lsr[dst_id] = new_lsr;
                        new_modified_vertices.insert(dst_id);
                    }
                }
            }
            modified_vertices = new_modified_vertices;
        }
        self.finish_step(step_num);
    }

    /// A Dijkstra-type traversal which ensures that shortest-length paths are
    /// explored first. This gives shorter, cleaner spoiler routes but is much slower,
    /// so it is only used at the end of a successful randomization attempt.
    pub fn traverse_dijkstra(
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
        start_vertex_id: VertexId,
        step_num: usize,
    ) {
        self.step.global_state = global.clone();
        let mut trail_ends: BinaryHeap<(Reverse<LinkLength>, StepTrailId)> =
            BinaryHeap::with_capacity(10000);

        let start_trail_id = self.step_trails.len() as StepTrailId;
        self.step_trails.push(StepTrail {
            vertex_id: start_vertex_id,
            link_idx: -1,
            local_state: self.initial_local_state,
        });
        trail_ends.push((Reverse(0), start_trail_id));

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
            cost_config: self.cost_config.clone(),
        };
        while let Some((Reverse(src_length), src_trail_id)) = trail_ends.pop() {
            let src_trail = &self.step_trails[src_trail_id as usize];
            let src_vertex_id = src_trail.vertex_id;
            let mut src_lsr = self.lsr[src_vertex_id].clone();
            let mut src_local = src_trail.local_state;

            src_local.prev_trail_id = src_trail_id;
            if !src_lsr.push(
                src_local,
                &cx.global.inventory,
                src_trail_id,
                &cx.cost_config,
                cx.reverse,
            ) {
                continue;
            }

            self.add_trail(src_vertex_id);
            self.lsr[src_vertex_id] = src_lsr;

            let all_src_links = base_links_by_src[src_vertex_id]
                .iter()
                .chain(seed_links_by_src[src_vertex_id].iter());
            for &(link_idx, ref link) in all_src_links {
                let dst_vertex_id = link.to_vertex_id;
                let mut local_arr: ArrayVec<LocalState, NUM_COST_METRICS> = ArrayVec::new();
                local_arr.push(src_local);
                local_arr = apply_link(link, local_arr, &cx);
                let dst_length = src_length + link.length;
                for local in local_arr {
                    let new_trail_id = self.step_trails.len() as StepTrailId;
                    let new_step_trail = StepTrail {
                        vertex_id: dst_vertex_id,
                        local_state: local,
                        link_idx,
                    };
                    self.step_trails.push(new_step_trail);
                    trail_ends.push((Reverse(dst_length), new_trail_id));
                }
            }
        }
        self.finish_step(step_num);
    }
}

pub fn get_link<'a>(
    idx: usize,
    base_links_data: &'a LinksDataGroup,
    seed_links_data: &'a LinksDataGroup,
) -> &'a Link {
    let base_links_len = base_links_data.links.len();
    if idx < base_links_len {
        &base_links_data.links[idx]
    } else {
        &seed_links_data.links[idx - base_links_len]
    }
}

pub fn get_spoiler_trail_ids(traverser: &Traverser, mut trail_id: StepTrailId) -> Vec<StepTrailId> {
    let mut steps: Vec<StepTrailId> = Vec::new();
    while trail_id != -1 {
        let step_trail = &traverser.step_trails[trail_id as usize];
        if step_trail.link_idx == -1 {
            break;
        }
        steps.push(trail_id);
        trail_id = step_trail.local_state.prev_trail_id;
    }
    steps.reverse();
    steps
}

pub fn get_spoiler_trail_ids_by_idx(
    traverser: &Traverser,
    vertex_id: usize,
    idx: usize,
) -> Vec<StepTrailId> {
    let trail_id = traverser.lsr[vertex_id].trail_ids[idx];
    get_spoiler_trail_ids(traverser, trail_id)
}

// This could be mostly covered by logic scenario tests, which require less maintenance,
// so we don't really want to add too many unit tests like this. These are unique though
// in how they check all cost metrics comprehensively, which can be helpful.
#[cfg(test)]
mod tests {
    use super::*;

    fn default_inventory() -> Inventory {
        Inventory {
            items: vec![],
            max_energy: 99,
            max_reserves: 0,
            max_missiles: 0,
            max_supers: 0,
            max_power_bombs: 0,
            collectible_missile_packs: 0,
            collectible_super_packs: 0,
            collectible_power_bomb_packs: 0,
            collectible_reserve_tanks: 0,
        }
    }

    #[test]
    fn compute_cost_consumed_vs_remaining_energy() {
        let inventory = default_inventory();

        let mut local1 = LocalState::empty();
        local1.energy = ResourceLevel::Remaining(50).into();

        let mut local2 = LocalState::empty();
        local2.energy = ResourceLevel::Consumed(50).into();

        // Forward:
        let cost1 = compute_cost(&local1, &inventory, &simple_cost_config(), false);
        let cost2 = compute_cost(&local2, &inventory, &simple_cost_config(), false);
        for i in 0..NUM_COST_METRICS {
            println!("forward: cost metric {}", i);
            assert!(cost1[i] < cost2[i]);
        }

        // Reverse:
        let cost1 = compute_cost(&local1, &inventory, &simple_cost_config(), true);
        let cost2 = compute_cost(&local2, &inventory, &simple_cost_config(), true);
        for i in 0..NUM_COST_METRICS {
            println!("reverse: cost metric {}", i);
            assert!(cost1[i] > cost2[i]);
        }
    }

    #[test]
    fn compute_cost_consumed_vs_remaining_energy_tie_break() {
        let inventory = default_inventory();

        let mut local1 = LocalState::empty();
        local1.energy = ResourceLevel::Remaining(50).into();

        let mut local2 = LocalState::empty();
        local2.energy = ResourceLevel::Consumed(49).into();

        // Forward:
        let cost1 = compute_cost(&local1, &inventory, &simple_cost_config(), false);
        let cost2 = compute_cost(&local2, &inventory, &simple_cost_config(), false);
        for i in 0..NUM_COST_METRICS {
            println!("forward: cost metric {}", i);
            assert!(cost1[i] > cost2[i]);
        }

        // Reverse:
        let cost1 = compute_cost(&local1, &inventory, &simple_cost_config(), true);
        let cost2 = compute_cost(&local2, &inventory, &simple_cost_config(), true);
        for i in 0..NUM_COST_METRICS {
            println!("reverse: cost metric {}", i);
            assert!(cost1[i] < cost2[i]);
        }
    }
}
