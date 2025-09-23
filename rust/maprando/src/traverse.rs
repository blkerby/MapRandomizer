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
    GlobalState, Inventory, LocalState, ResourceLevel,
    boss_requirements::{
        apply_botwoon_requirement, apply_draygon_requirement, apply_mother_brain_2_requirement,
        apply_phantoon_requirement, apply_ridley_requirement,
    },
    helpers::suit_damage_factor,
};

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

pub const NUM_COST_METRICS: usize = 3;

fn compute_cost(
    local: &LocalState,
    inventory: &Inventory,
    reverse: bool,
) -> [f32; NUM_COST_METRICS] {
    let mut energy_cost = match local.energy() {
        ResourceLevel::Consumed(x) => x as f32,
        ResourceLevel::Remaining(x) => (inventory.max_energy - x) as f32 + 0.5,
    } / 1500.0;
    let mut reserve_cost = match local.reserves() {
        ResourceLevel::Consumed(x) => x as f32,
        ResourceLevel::Remaining(x) => (inventory.max_reserves - x) as f32 + 0.5,
    } / 400.0;
    let mut missiles_cost = match local.missiles() {
        ResourceLevel::Consumed(x) => x as f32,
        ResourceLevel::Remaining(x) => (inventory.max_missiles - x) as f32 + 0.5,
    } / 500.0;
    let mut supers_cost = match local.supers() {
        ResourceLevel::Consumed(x) => x as f32,
        ResourceLevel::Remaining(x) => (inventory.max_supers - x) as f32 + 0.5,
    } / 100.0;
    let mut power_bombs_cost = match local.power_bombs() {
        ResourceLevel::Consumed(x) => x as f32,
        ResourceLevel::Remaining(x) => (inventory.max_power_bombs - x) as f32 + 0.5,
    } / 100.0;
    let mut shinecharge_cost = if local.flash_suit {
        // For the purposes of the cost metrics, treat flash suit as equivalent
        // to a large amount of shinecharge frames remaining:
        0.0
    } else {
        (180 - local.shinecharge_frames_remaining) as f32
    } / 180.0;
    if reverse {
        energy_cost = -energy_cost;
        reserve_cost = -reserve_cost;
        missiles_cost = -missiles_cost;
        supers_cost = -supers_cost;
        power_bombs_cost = -power_bombs_cost;
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

fn apply_gate_glitch_leniency(
    local: &mut LocalState,
    global: &GlobalState,
    green: bool,
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
    if green {
        local.use_supers(difficulty.gate_glitch_leniency, &global.inventory, reverse)
    } else {
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
        if local.power_bombs() == ResourceLevel::Consumed(0) {
            p_small += p_pb * rel_small;
            p_large += p_pb * rel_large;
            p_missile += p_pb * rel_missile;
        }
        if local.supers() == ResourceLevel::Consumed(0) {
            p_small += p_super * rel_small;
            p_large += p_super * rel_large;
            p_missile += p_super * rel_missile;
        }
        if local.missiles() == ResourceLevel::Consumed(0) {
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
        if reverse {
            local.refill_energy(total_drop_value, true, &global.inventory, reverse);
            local.use_energy(heat_energy, true, &global.inventory, reverse)
        } else {
            if !local.use_energy(heat_energy, true, &global.inventory, reverse) {
                return false;
            }
            local.refill_energy(total_drop_value, true, &global.inventory, reverse);
            true
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
        if reverse {
            local.refill_energy(total_drop_value, true, &global.inventory, reverse);
            local.use_energy(lava_energy, true, &global.inventory, reverse)
        } else {
            if !local.use_energy(lava_energy, true, &global.inventory, reverse) {
                return false;
            }
            local.refill_energy(total_drop_value, true, &global.inventory, reverse);
            true
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
        false,
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
    let cycle_energy = (end_local.energy_remaining(&global.inventory, true)
        - local.energy_remaining(&global.inventory, true)) as f32;
    let cycle_missiles = (end_local.missiles_remaining(&global.inventory)
        - local.missiles_remaining(&global.inventory)) as f32;
    let cycle_supers = (end_local.supers_remaining(&global.inventory)
        - local.supers_remaining(&global.inventory)) as f32;
    let cycle_pbs = (end_local.power_bombs_remaining(&global.inventory)
        - local.power_bombs_remaining(&global.inventory)) as f32;
    let patience_frames = difficulty.farm_time_limit * 60.0;
    let num_cycles = (patience_frames / cycle_frames).floor() as i32;

    let mut new_local = local;
    if new_local.farm_baseline_energy_available(&global.inventory, reverse)
        > new_local.energy_available(&global.inventory, false, false)
    {
        new_local.farm_baseline_energy = new_local.energy;
    }
    if new_local.farm_baseline_reserves_available(&global.inventory, reverse)
        > new_local.reserves_remaining(&global.inventory)
    {
        new_local.farm_baseline_reserves = new_local.reserves;
    }
    if new_local.farm_baseline_missiles_available(&global.inventory, reverse)
        > new_local.missiles_available(&global.inventory, reverse)
    {
        new_local.farm_baseline_missiles = new_local.missiles;
    }
    if new_local.farm_baseline_supers_available(&global.inventory, reverse)
        > new_local.supers_available(&global.inventory, reverse)
    {
        new_local.farm_baseline_supers = new_local.supers;
    }
    if new_local.farm_baseline_power_bombs_available(&global.inventory, reverse)
        > new_local.power_bombs_available(&global.inventory, reverse)
    {
        new_local.farm_baseline_power_bombs = new_local.power_bombs;
    }
    new_local.energy = new_local.farm_baseline_energy;
    new_local.reserves = new_local.farm_baseline_reserves;
    new_local.missiles = new_local.farm_baseline_missiles;
    new_local.supers = new_local.farm_baseline_supers;
    new_local.power_bombs = new_local.farm_baseline_power_bombs;

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

    let net_energy = ((drop_energy - cycle_energy) * num_cycles as f32) as Capacity;
    let net_missiles = ((drop_missiles - cycle_missiles) * num_cycles as f32) as Capacity;
    let net_supers = ((drop_supers - cycle_supers) * num_cycles as f32) as Capacity;
    let net_pbs = ((drop_pbs - cycle_pbs) * num_cycles as f32) as Capacity;

    if net_energy < 0 || net_missiles < 0 || net_supers < 0 || net_pbs < 0 {
        return None;
    }

    new_local.refill_energy(net_energy, true, &global.inventory, reverse);
    new_local.refill_missiles(net_missiles, &global.inventory, reverse);
    new_local.refill_supers(net_supers, &global.inventory, reverse);
    new_local.refill_power_bombs(net_pbs, &global.inventory, reverse);

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
    }
    if net_missiles >= global.pool_inventory.max_missiles {
        new_local.missiles = ResourceLevel::full(reverse).into();
    }
    if net_supers >= global.pool_inventory.max_supers {
        new_local.supers = ResourceLevel::full(reverse).into();
    }
    if net_pbs >= global.pool_inventory.max_power_bombs {
        new_local.power_bombs = ResourceLevel::full(reverse).into();
    }

    if new_local.energy_available(&global.inventory, true, reverse)
        == global.inventory.max_energy + global.inventory.max_reserves
    {
        new_local.farm_baseline_energy = new_local.energy;
        new_local.farm_baseline_reserves = new_local.reserves;
    }
    if new_local.missiles_available(&global.inventory, reverse) == global.inventory.max_missiles {
        new_local.farm_baseline_missiles = new_local.missiles;
    }
    if new_local.supers_available(&global.inventory, reverse) == global.inventory.max_supers {
        new_local.farm_baseline_supers = new_local.supers;
    }
    if new_local.power_bombs_available(&global.inventory, reverse)
        == global.inventory.max_power_bombs
    {
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
        inventory: &Inventory,
        trail_id: T,
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
                    reducer.push(loc, &cx.global.inventory, (), cx.reverse);
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
                        reducer.push(loc, &cx.global.inventory, (), cx.reverse);
                    }
                    SimpleResult::_ExtraState(extra_state) => {
                        reducer.push(loc, &cx.global.inventory, (), cx.reverse);
                        reducer.push(extra_state, &cx.global.inventory, (), cx.reverse);
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
            cx.reverse,
        )
        .into(),
        Requirement::SimpleHeatFrames(frames) => apply_heat_frames(
            *frames,
            local,
            cx.global,
            cx.game_data,
            cx.difficulty,
            true,
            cx.reverse,
        )
        .into(),
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
            let varia = cx.global.inventory.items[Item::Varia as usize];
            let gravity = cx.global.inventory.items[Item::Gravity as usize];
            if gravity && varia {
                SimpleResult::Success
            } else if gravity || varia {
                let energy_used =
                    (*frames as f32 * cx.difficulty.resource_multiplier / 4.0).ceil() as Capacity;
                local
                    .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                    .into()
            } else {
                let energy_used =
                    (*frames as f32 * cx.difficulty.resource_multiplier / 2.0).ceil() as Capacity;
                local
                    .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                    .into()
            }
        }
        Requirement::GravitylessLavaFrames(frames) => {
            let varia = cx.global.inventory.items[Item::Varia as usize];
            let energy_used = if varia {
                (*frames as f32 * cx.difficulty.resource_multiplier / 4.0).ceil() as Capacity
            } else {
                (*frames as f32 * cx.difficulty.resource_multiplier / 2.0).ceil() as Capacity
            };
            local
                .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::AcidFrames(frames) => {
            let energy_used = (*frames as f32 * cx.difficulty.resource_multiplier * 1.5
                / suit_damage_factor(&cx.global.inventory) as f32)
                .ceil() as Capacity;
            local
                .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::GravitylessAcidFrames(frames) => {
            let varia = cx.global.inventory.items[Item::Varia as usize];
            let energy_used = if varia {
                (*frames as f32 * cx.difficulty.resource_multiplier * 0.75).ceil() as Capacity
            } else {
                (*frames as f32 * cx.difficulty.resource_multiplier * 1.5).ceil() as Capacity
            };
            local
                .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::MetroidFrames(frames) => {
            let energy_used = (*frames as f32 * cx.difficulty.resource_multiplier * 0.75
                / suit_damage_factor(&cx.global.inventory) as f32)
                .ceil() as Capacity;
            local
                .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                .into()
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
            if energy >= cx.global.inventory.max_energy {
                if cx.difficulty.tech[cx.game_data.manage_reserves_tech_idx] {
                    // With canManageReserves, assume low energy is put into reserves,
                    // in order to survive the damage with an auto-refill while keeping
                    // almost all the available i-frames.
                    local
                        .auto_reserve_trigger(1, 1, &cx.global.inventory, false, cx.reverse)
                        .into()
                } else {
                    SimpleResult::Failure
                }
            } else {
                local
                    .use_energy(energy, true, &cx.global.inventory, cx.reverse)
                    .into()
            }
        }
        &Requirement::Energy(count) => local
            .use_energy(count, true, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::RegularEnergy(count) => {
            // For now, we assume reserve energy can be converted to regular energy, so this is
            // implemented the same as the Energy requirement above.
            local
                .use_energy(count, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        &Requirement::ReserveEnergy(count) => local
            .use_reserve_energy(count, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::Missiles(count) => local
            .use_missiles(count, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::Supers(count) => local
            .use_supers(count, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::PowerBombs(count) => local
            .use_power_bombs(count, &cx.global.inventory, cx.reverse)
            .into(),
        Requirement::GateGlitchLeniency { green, heated } => {
            apply_gate_glitch_leniency(local, cx.global, *green, *heated, cx.difficulty, cx.reverse)
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
        Requirement::BombIntoCrystalFlashClipLeniency {} => local
            .use_power_bombs(
                cx.difficulty.bomb_into_cf_leniency,
                &cx.global.inventory,
                cx.reverse,
            )
            .into(),
        Requirement::JumpIntoCrystalFlashClipLeniency {} => local
            .use_power_bombs(
                cx.difficulty.jump_into_cf_leniency,
                &cx.global.inventory,
                cx.reverse,
            )
            .into(),
        Requirement::XModeSpikeHitLeniency {} => {
            let energy_used =
                cx.difficulty.spike_xmode_leniency * 60 / suit_damage_factor(&cx.global.inventory);
            local
                .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        Requirement::XModeThornHitLeniency {} => {
            let energy_used =
                cx.difficulty.spike_xmode_leniency * 16 / suit_damage_factor(&cx.global.inventory);
            local
                .use_energy(energy_used, true, &cx.global.inventory, cx.reverse)
                .into()
        }
        &Requirement::MissilesAvailable(count) => local
            .ensure_missiles_available(count, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::SupersAvailable(count) => local
            .ensure_supers_available(count, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::PowerBombsAvailable(count) => local
            .ensure_power_bombs_available(count, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::RegularEnergyAvailable(count) => local
            .ensure_energy_available(count, false, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::ReserveEnergyAvailable(count) => local
            .ensure_reserves_available(count, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::EnergyAvailable(count) => local
            .ensure_energy_available(count, true, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::MissilesMissingAtMost(count) => local
            .ensure_missiles_missing_at_most(count, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::SupersMissingAtMost(count) => local
            .ensure_supers_missing_at_most(count, &cx.global.inventory, cx.reverse)
            .into(),
        &Requirement::PowerBombsMissingAtMost(count) => local
            .ensure_power_bombs_missing_at_most(count, &cx.global.inventory, cx.reverse)
            .into(),
        Requirement::RegularEnergyMissingAtMost(_) => unimplemented!(),
        &Requirement::ReserveEnergyMissingAtMost(count) => local
            .ensure_reserves_missing_at_most(count, &cx.global.inventory, cx.reverse)
            .into(),
        Requirement::EnergyMissingAtMost(_) => unimplemented!(),
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
        &Requirement::EnergyRefill(limit) => {
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
        &Requirement::RegularEnergyRefill(limit) => {
            let energy_remaining = local.energy_remaining(&cx.global.inventory, false);
            if limit >= cx.global.pool_inventory.max_energy {
                local.energy = ResourceLevel::full(cx.reverse).into();
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
        &Requirement::ReserveRefill(limit) => {
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
        &Requirement::MissileRefill(limit) => {
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
        &Requirement::SuperRefill(limit) => {
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
        &Requirement::PowerBombRefill(limit) => {
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
        &Requirement::RegularEnergyDrain(count) => {
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
        &Requirement::ReserveEnergyDrain(count) => {
            let reserves_remaining = local.reserves_remaining(&cx.global.inventory);
            if cx.reverse {
                (reserves_remaining <= count).into()
            } else {
                local.reserves =
                    ResourceLevel::Remaining(Capacity::min(count, reserves_remaining)).into();
                SimpleResult::Success
            }
        }
        &Requirement::MissileDrain(count) => {
            let missiles_remaining = local.missiles_remaining(&cx.global.inventory);
            if cx.reverse {
                (missiles_remaining <= count).into()
            } else {
                local.missiles =
                    ResourceLevel::Remaining(Capacity::min(count, missiles_remaining)).into();
                SimpleResult::Success
            }
        }
        &Requirement::ReserveTrigger {
            min_reserve_energy,
            max_reserve_energy,
            heated,
        } => local
            .auto_reserve_trigger(
                min_reserve_energy,
                max_reserve_energy,
                &cx.global.inventory,
                heated,
                cx.reverse,
            )
            .into(),
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
            can_be_very_patient_tech_idx: can_be_very_patient_tech_id,
        } => apply_draygon_requirement(
            &cx.global.inventory,
            local,
            cx.difficulty.draygon_proficiency,
            cx.difficulty.tech[*can_be_very_patient_tech_id],
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
                    if regular_energy_remaining <= 29 {
                        if frames == excess_frames {
                            // If all frames are excess frames and energy is at 29 or lower, then the spark does not require any energy:
                            return SimpleResult::Success;
                        }
                        local
                            .ensure_energy_available(
                                29 + min_frames,
                                can_manage_reserves,
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
    if forward.reserves_remaining(&global.inventory) < reverse.reserves_remaining(&global.inventory)
    {
        return false;
    }
    if forward.energy_remaining(&global.inventory, true)
        < reverse.energy_remaining(&global.inventory, true)
    {
        return false;
    }
    if forward.missiles_remaining(&global.inventory) < reverse.missiles_remaining(&global.inventory)
    {
        return false;
    }
    if forward.supers_remaining(&global.inventory) < reverse.supers_remaining(&global.inventory) {
        return false;
    }
    if forward.power_bombs_remaining(&global.inventory)
        < reverse.power_bombs_remaining(&global.inventory)
    {
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
    pub initial_local_state: LocalState,
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
        inventory: &Inventory,
        start_vertex_id: usize,
    ) {
        let mut lsr = LocalStateReducer::<StepTrailId>::new();
        lsr.push(init_local, inventory, -1, self.reverse);
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
                    let old_lsr = &self.lsr[dst_id];
                    let mut new_lsr = LocalStateReducer::new();
                    for i in 0..old_lsr.local.len() {
                        // Rebuild the LocalStateReducer in order to update costs, which
                        // may have changed due to new inventory.
                        new_lsr.push(
                            old_lsr.local[i],
                            &cx.global.inventory,
                            old_lsr.trail_ids[i],
                            cx.reverse,
                        );
                    }
                    local_arr = apply_link(link, local_arr, &cx);
                    for local in local_arr {
                        let new_trail_id = self.step_trails.len() as StepTrailId;
                        if new_lsr.push(local, &cx.global.inventory, new_trail_id, cx.reverse) {
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
