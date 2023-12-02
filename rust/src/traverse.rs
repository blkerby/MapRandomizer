use std::{
    cmp::{max, min},
    mem::swap,
};

use hashbrown::{HashMap, HashSet};

use crate::{
    game_data::{
        Capacity, EnemyVulnerabilities, GameData, Item, Link, LinkIdx, LinksDataGroup, Requirement,
        WeaponMask,
    },
    randomize::{DifficultyConfig, WallJump},
};

// TODO: move tech and notable_strats out of this struct, since these do not change from step to step.
#[derive(Clone, Debug)]
pub struct GlobalState {
    pub tech: Vec<bool>,
    pub notable_strats: Vec<bool>,
    pub items: Vec<bool>,
    pub flags: Vec<bool>,
    pub max_energy: Capacity,
    pub max_reserves: Capacity,
    pub max_missiles: Capacity,
    pub max_supers: Capacity,
    pub max_power_bombs: Capacity,
    pub weapon_mask: WeaponMask,
    pub shine_charge_tiles: f32,
}

impl GlobalState {
    pub fn print_debug(&self, game_data: &GameData) {
        for (i, item) in game_data.item_isv.keys.iter().enumerate() {
            if self.items[i] {
                println!("{:?}", item);
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LocalState {
    pub energy_used: Capacity,
    pub reserves_used: Capacity,
    pub missiles_used: Capacity,
    pub supers_used: Capacity,
    pub power_bombs_used: Capacity,
}

impl LocalState {
    pub fn new() -> Self {
        Self {
            energy_used: 0,
            reserves_used: 0,
            missiles_used: 0,
            supers_used: 0,
            power_bombs_used: 0,
        }
    }
}

fn get_charge_damage(global: &GlobalState) -> f32 {
    if !global.items[Item::Charge as usize] {
        return 0.0;
    }
    let plasma = global.items[Item::Plasma as usize];
    let spazer = global.items[Item::Spazer as usize];
    let wave = global.items[Item::Wave as usize];
    let ice = global.items[Item::Spazer as usize];
    return match (plasma, spazer, wave, ice) {
        (false, false, false, false) => 20.0,
        (false, false, false, true) => 30.0,
        (false, false, true, false) => 50.0,
        (false, false, true, true) => 60.0,
        (false, true, false, false) => 40.0,
        (false, true, false, true) => 60.0,
        (false, true, true, false) => 70.0,
        (false, true, true, true) => 100.0,
        (true, _, false, false) => 150.0,
        (true, _, false, true) => 200.0,
        (true, _, true, false) => 250.0,
        (true, _, true, true) => 300.0,
    } * 3.0;
}

fn apply_enemy_kill_requirement(
    global: &GlobalState,
    mut local: LocalState,
    count: i32,
    vul: &EnemyVulnerabilities,
) -> Option<LocalState> {
    // Prioritize using weapons that do not require ammo:
    if global.weapon_mask & vul.non_ammo_vulnerabilities != 0 {
        return Some(local);
    }

    let mut hp = vul.hp; // HP per enemy

    // Next use Missiles:
    if vul.missile_damage > 0 {
        let missiles_available = global.max_missiles - local.missiles_used;
        let missiles_to_use_per_enemy = max(
            0,
            min(
                missiles_available / count,
                (hp + vul.missile_damage - 1) / vul.missile_damage,
            ),
        );
        hp -= missiles_to_use_per_enemy * vul.missile_damage as i32;
        local.missiles_used += missiles_to_use_per_enemy * count;
    }

    // Then use Supers (some overkill is possible, where we could have used fewer Missiles, but we ignore that):
    if vul.super_damage > 0 {
        let supers_available = global.max_supers - local.supers_used;
        let supers_to_use_per_enemy = max(
            0,
            min(
                supers_available / count,
                (hp + vul.super_damage - 1) / vul.super_damage,
            ),
        );
        hp -= supers_to_use_per_enemy * vul.super_damage as i32;
        local.supers_used += supers_to_use_per_enemy * count;
    }

    // Finally, use Power Bombs (overkill is possible, where we could have used fewer Missiles or Supers, but we ignore that):
    if vul.power_bomb_damage > 0 && global.items[Item::Morph as usize] {
        let pbs_available = global.max_power_bombs - local.power_bombs_used;
        let pbs_to_use = max(
            0,
            min(
                pbs_available,
                (hp + vul.power_bomb_damage - 1) / vul.power_bomb_damage,
            ),
        );
        hp -= pbs_to_use * vul.power_bomb_damage as i32;
        // Power bombs hit all enemies in the group, so we do not multiply by the count.
        local.power_bombs_used += pbs_to_use;
    }

    if hp <= 0 {
        Some(local)
    } else {
        None
    }
}

fn apply_phantoon_requirement(
    global: &GlobalState,
    mut local: LocalState,
    proficiency: f32,
) -> Option<LocalState> {
    // We only consider simple, safer strats here, where we try to damage Phantoon as much as possible
    // as soon as he opens his eye. Faster or more complex strats are not relevant, since at
    // high proficiency the fight is considered free anyway (as long as Charge or any ammo is available)
    // since all damage can be avoided.
    let boss_hp: f32 = 2500.0;
    let charge_damage = get_charge_damage(&global);

    // Assume a firing rate of between 50% (on lowest difficulty) to 100% (on highest).
    // This represents missing the opportunity to hit Phantoon when he first opens his eye,
    // and having to wait for the invisible phase.
    let firing_rate = 0.5 + 0.5 * proficiency;

    let mut possible_kill_times: Vec<f32> = vec![];
    if charge_damage > 0.0 {
        let charge_shots_to_use = f32::ceil(boss_hp / charge_damage);
        // Assume max 1 charge shot per 10 seconds. With weaker beams, a higher firing rate is
        // possible, but we leave it like this to roughly account for the higher risk of
        // damage from Phantoon's body when one shot won't immediately despawn him.
        let time = charge_shots_to_use as f32 * 10.0 / firing_rate;
        possible_kill_times.push(time);
    }
    if global.max_missiles > 0 {
        // We don't worry about ammo quantity since they can be farmed from the flames.
        let missiles_to_use = f32::ceil(boss_hp / 100.0);
        // Assume max average rate of 3 missiles per 10 seconds:
        let time = missiles_to_use as f32 * 10.0 / 3.0 / firing_rate;
        possible_kill_times.push(time);
    }
    if global.max_supers > 0 {
        // We don't worry about ammo quantity since they can be farmed from the flames.
        let supers_to_use = f32::ceil(boss_hp / 600.0);
        let time = supers_to_use as f32 * 30.0; // Assume average rate of 1 Super per 30 seconds
        possible_kill_times.push(time);
    }

    let kill_time = match possible_kill_times.iter().min_by(|x, y| x.total_cmp(y)) {
        Some(t) => t,
        None => {
            return None;
        }
    };

    // Assumed rate of damage to Samus per second.
    let base_hit_dps = 10.0 * (1.0 - 0.6 * proficiency);

    // Assumed average energy per second gained from farming flames:
    let farm_rate = 4.0 * (0.25 + 0.75 * proficiency);

    // Net damage taken by Samus per second, taking into account suit protection and farms:
    let mut net_dps = base_hit_dps / suit_damage_factor(global) as f32 - farm_rate;
    if net_dps < 0.0 {
        // We could assume we could refill on energy or ammo using farms, but by omitting this for now
        // we're just making the logic a little more conservative in favor of the player.
        net_dps = 0.0;
    }

    local.energy_used += (net_dps * kill_time) as Capacity;

    validate_energy(local, global)
}

fn apply_draygon_requirement(
    global: &GlobalState,
    mut local: LocalState,
    proficiency: f32,
    can_be_very_patient_tech_id: usize,
) -> Option<LocalState> {
    let boss_hp: f32 = 6000.0;
    let charge_damage = get_charge_damage(&global);

    // Assume an accuracy of between 60% (on lowest difficulty) to 100% (on highest).
    let accuracy = 0.6 + 0.4 * proficiency;

    // Assume a firing rate of between 60% (on lowest difficulty) to 100% (on highest).
    let firing_rate = 0.6 + 0.4 * proficiency;

    let mut possible_kill_times: Vec<f32> = vec![];
    if charge_damage > 0.0 {
        let charge_shots_to_use = f32::ceil(boss_hp / charge_damage / accuracy);
        // Assume max 1 charge shot per 3 seconds.
        let time = charge_shots_to_use as f32 * 3.0 / firing_rate;
        possible_kill_times.push(time);
    }
    if global.max_missiles > 0 {
        // We don't worry about ammo quantity since they can be farmed from the goops.
        let missiles_to_use = f32::ceil(boss_hp / 100.0 / accuracy);
        // Assume max average rate of 1 missiles per second:
        let time = missiles_to_use as f32 * 1.0 / firing_rate;
        possible_kill_times.push(time);
    }
    // We ignore the possibility of using Supers since farming them is very slow and the
    // potential benefit over using Missiles is limited.

    let kill_time = match possible_kill_times.iter().min_by(|x, y| x.total_cmp(y)) {
        Some(&t) => t,
        None => {
            return None;
        }
    };

    if kill_time >= 180.0 && !global.tech[can_be_very_patient_tech_id] {
        // We don't have enough patience to finish the fight:
        return None;
    }

    // Assumed rate of damage to Samus per second.
    let base_hit_dps = 20.0 * (1.0 - 0.9 * proficiency);

    // TODO: we should take into account key items like Morph, Gravity, and Screw Attack.
    // We ignore this for now; the strats already ensure either Morph or Gravity is available.

    // Assumed average energy per second gained from farming goops:
    let farm_rate = 3.0 * (0.5 + 0.5 * proficiency);

    // Net damage taken by Samus per second, taking into account suit protection and farms:
    let mut net_dps = base_hit_dps / suit_damage_factor(global) as f32 - farm_rate;
    if net_dps < 0.0 {
        // We could assume we could refill on energy or ammo using farms, but by omitting this for now
        // we're just making the logic a little more conservative in favor of the player.
        net_dps = 0.0;
    }

    local.energy_used += (net_dps * kill_time) as Capacity;

    validate_energy(local, global)
}

fn apply_ridley_requirement(
    global: &GlobalState,
    mut local: LocalState,
    proficiency: f32,
    can_be_very_patient_tech_id: usize,
) -> Option<LocalState> {
    let mut boss_hp: f32 = 18000.0;
    let mut time: f32 = 0.0; // Cumulative time in seconds for the fight
    let charge_damage = get_charge_damage(&global);

    // Assume an ammo accuracy rate of between 50% (on lowest difficulty) to 100% (on highest):
    let accuracy = 0.5 + 0.5 * proficiency;

    // Assume a firing rate of between 50% (on lowest difficulty) to 100% (on highest):
    let firing_rate = 0.5 + 0.5 * proficiency;

    // Prioritize using supers:
    let supers_available = global.max_supers - local.supers_used;
    let supers_to_use = min(
        supers_available,
        f32::ceil(boss_hp / (600.0 * accuracy)) as Capacity,
    );
    local.supers_used += supers_to_use;
    boss_hp -= supers_to_use as f32 * 600.0 * accuracy;
    time += supers_to_use as f32 * 0.5 / firing_rate; // Assumes max average rate of 2 supers per second

    // Use Charge Beam if it's powerful enough
    // 500 is the point at which Charge Beam has better DPS than Missiles, this happens with Charge + Plasma + (Ice and/or Wave)
    if charge_damage >= 500.0 {
        let powerful_charge_shots_to_use = max(
            0,
            f32::ceil(boss_hp / (charge_damage * accuracy)) as Capacity,
        );
        boss_hp = 0.0;
        time += powerful_charge_shots_to_use as f32 * 1.5 / firing_rate; // Assume max 1 charge shot per 1.5 seconds
    }

    // Then use available missiles:
    let missiles_available = global.max_missiles - local.missiles_used;
    let missiles_to_use = max(
        0,
        min(
            missiles_available,
            f32::ceil(boss_hp / (100.0 * accuracy)) as Capacity,
        ),
    );
    local.missiles_used += missiles_to_use;
    boss_hp -= missiles_to_use as f32 * 100.0 * accuracy;
    time += missiles_to_use as f32 * 0.3 / firing_rate; // Assume max average rate of 1 missile per 0.3 seconds

    if global.items[Item::Charge as usize] {
        // Then finish with Charge shots:
        // (TODO: it would be a little better to prioritize Charge shots over Supers/Missiles in
        // some cases).
        let charge_shots_to_use = max(
            0,
            f32::ceil(boss_hp / (charge_damage * accuracy)) as Capacity,
        );
        boss_hp = 0.0;
        time += charge_shots_to_use as f32 * 1.5 / firing_rate; // Assume max 1 charge shot per 1.5 seconds
    } else if global.items[Item::Morph as usize] {
        // Only use Power Bombs if Charge is not available:
        let pbs_available = global.max_power_bombs - local.power_bombs_used;
        let pbs_to_use = max(
            0,
            min(
                pbs_available,
                f32::ceil(boss_hp / (400.0 * accuracy)) as Capacity,
            ),
        );
        local.power_bombs_used += pbs_to_use;
        boss_hp -= pbs_to_use as f32 * 400.0 * accuracy; // Assumes double hits (or single hits for 50% accuracy)
        time += pbs_to_use as f32 * 3.0 * firing_rate; // Assume max average rate of 1 power bomb per 3 seconds
    }

    if boss_hp > 0.0 {
        // We don't have enough ammo to finish the fight:
        return None;
    }

    if time >= 180.0 && !global.tech[can_be_very_patient_tech_id] {
        // We don't have enough patience to finish the fight:
        return None;
    }

    let morph = global.items[Item::Morph as usize];
    let screw = global.items[Item::ScrewAttack as usize];

    // Assumed rate of Ridley damage to Samus (per second), given minimal dodging skill:
    let base_ridley_attack_dps = 40.0;

    // Multiplier to Ridley damage based on items (Morph and Screw) and proficiency (in dodging).
    // This is a rough guess which could be refined. We could also take into account other items
    // (HiJump and SpaceJump). We assume that at Expert level (proficiency=1.0) it is possible
    // to avoid all damage from Ridley using either Morph or Screw.
    let hit_rate = match (morph, screw) {
        (false, false) => 1.0 - 0.8 * proficiency,
        (false, true) => 0.5 - 0.5 * proficiency,
        (true, false) => 0.5 - 0.5 * proficiency,
        (true, true) => 0.3 - 0.3 * proficiency,
    };
    let damage = base_ridley_attack_dps * hit_rate * time;
    local.energy_used += (damage / suit_damage_factor(global) as f32) as Capacity;

    if !global.items[Item::Varia as usize] {
        // Heat run case: We do not explicitly check canHeatRun tech here, because it is
        // already required to reach the boss node from the doors.
        // Include time pre- and post-fight when Samus must still take heat damage:
        let heat_time = time + 20.0;
        // let heat_energy_used = if global.items[Item::Gravity as usize] {
        //     (heat_time * 7.5) as Capacity
        // } else {
        //     (heat_time * 15.0) as Capacity
        // };
        let heat_energy_used = (heat_time * 15.0) as Capacity;
        local.energy_used += heat_energy_used;
    }

    // TODO: We could add back some energy and/or ammo by assuming we get drops.
    // By omitting this for now we're just making the logic a little more conservative in favor of
    // the player.
    validate_energy(local, global)
}

fn apply_botwoon_requirement(
    global: &GlobalState,
    mut local: LocalState,
    proficiency: f32,
    second_phase: bool,
) -> Option<LocalState> {
    // We aim to be a little lenient here. For example, we don't take SBAs (e.g. X-factors) into account,
    // assuming instead the player just uses ammo and/or regular charged shots.

    let mut boss_hp: f32 = 1500.0; // HP for one phase of the fight.
    let mut time: f32 = 0.0; // Cumulative time in seconds for the phase
    let charge_damage = get_charge_damage(&global);

    // Assume an ammo accuracy rate of between 25% (on lowest difficulty) to 90% (on highest):
    let accuracy = 0.25 + 0.65 * proficiency;

    // Assume a firing rate of between 30% (on lowest difficulty) to 100% (on highest),
    let firing_rate = 0.3 + 0.7 * proficiency;

    // The firing rates below are for the first phase (since the rate doesn't matter for
    // the second phase):
    let use_supers = |local: &mut LocalState, boss_hp: &mut f32, time: &mut f32| {
        let supers_available = global.max_supers - local.supers_used;
        let supers_to_use = min(
            supers_available,
            f32::ceil(*boss_hp / (300.0 * accuracy)) as Capacity,
        );
        local.supers_used += supers_to_use;
        *boss_hp -= supers_to_use as f32 * 300.0 * accuracy;
        // Assume a max average rate of one super shot per 2.0 second:
        *time += supers_to_use as f32 * 2.0 / firing_rate;
    };

    let use_missiles = |local: &mut LocalState, boss_hp: &mut f32, time: &mut f32| {
        let missiles_available = global.max_missiles - local.missiles_used;
        let missiles_to_use = max(
            0,
            min(
                missiles_available,
                f32::ceil(*boss_hp / (100.0 * accuracy)) as Capacity,
            ),
        );
        local.missiles_used += missiles_to_use;
        *boss_hp -= missiles_to_use as f32 * 100.0 * accuracy;
        // Assume a max average rate of one missile shot per 1.0 seconds:
        *time += missiles_to_use as f32 * 1.0 / firing_rate;
    };

    let use_charge = |boss_hp: &mut f32, time: &mut f32| {
        if charge_damage == 0.0 {
            return;
        }
        let charge_shots_to_use = max(
            0,
            f32::ceil(*boss_hp / (charge_damage * accuracy)) as Capacity,
        );
        *boss_hp = 0.0;
        // Assume max average rate of one charge shot per 3.0 seconds
        *time += charge_shots_to_use as f32 * 3.0 / firing_rate;
    };

    if second_phase {
        // In second phase, prioritize using Charge beam if available. Even if it is slow, we are not
        // taking damage so we want to conserve ammo.
        if global.items[Item::Charge as usize] {
            use_charge(&mut boss_hp, &mut time);
        } else {
            // Prioritize using missiles over supers. This is slower but the idea is to conserve supers
            // since they are generally more valuable to save for later.
            use_missiles(&mut local, &mut boss_hp, &mut time);
            use_supers(&mut local, &mut boss_hp, &mut time);
        }
    } else {
        // In the first phase, prioritize using the highest-DPS weapons, to finish the fight faster and
        // hence minimize damage taken:
        if charge_damage >= 450.0 {
            use_charge(&mut boss_hp, &mut time);
        } else {
            use_supers(&mut local, &mut boss_hp, &mut time);
            use_missiles(&mut local, &mut boss_hp, &mut time);
            use_charge(&mut boss_hp, &mut time);
        }
    }

    if boss_hp > 0.0 {
        // We don't have enough ammo to finish the fight:
        return None;
    }

    if !second_phase {
        let morph = global.items[Item::Morph as usize];
        let gravity = global.items[Item::Gravity as usize];

        // Assumed average rate of boss attacks to Samus (per second), given minimal dodging skill:
        let base_hit_rate = 0.1;

        // Multiplier to boss damage based on items (Morph and Gravity) and proficiency (in dodging).
        let hit_rate_multiplier = match (morph, gravity) {
            (false, false) => 1.0 * (1.0 - proficiency) + 0.25 * proficiency,
            (false, true) => 0.8 * (1.0 - proficiency) + 0.2 * proficiency,
            (true, false) => 0.7 * (1.0 - proficiency) + 0.125 * proficiency,
            (true, true) => 0.5 * (1.0 - proficiency) + 0.0 * proficiency,
        };
        let hits = f32::round(base_hit_rate * hit_rate_multiplier * time);
        let damage_per_hit = 96.0 / suit_damage_factor(global) as f32;
        local.energy_used += (hits * damage_per_hit) as Capacity;
    }

    // TODO: We could add back some energy and/or ammo by assuming we get drops.
    // By omitting this for now we're just making the logic a little more conservative in favor of
    // the player.
    validate_energy(local, global)
}

pub const IMPOSSIBLE_LOCAL_STATE: LocalState = LocalState {
    energy_used: 0x3FFF,
    reserves_used: 0x3FFF,
    missiles_used: 0x3FFF,
    supers_used: 0x3FFF,
    power_bombs_used: 0x3FFF,
};

pub const NUM_COST_METRICS: usize = 2;

fn compute_cost(local: LocalState, global: &GlobalState) -> [f32; NUM_COST_METRICS] {
    let eps = 1e-15;
    let energy_cost = (local.energy_used as f32) / (global.max_energy as f32 + eps);
    let reserve_cost = (local.reserves_used as f32) / (global.max_reserves as f32 + eps);
    let missiles_cost = (local.missiles_used as f32) / (global.max_missiles as f32 + eps);
    let supers_cost = (local.supers_used as f32) / (global.max_supers as f32 + eps);
    let power_bombs_cost = (local.power_bombs_used as f32) / (global.max_power_bombs as f32 + eps);
    
    let ammo_sensitive_cost_metric =
        energy_cost + reserve_cost + 10.0 * (missiles_cost + supers_cost + power_bombs_cost);
    let energy_sensitive_cost_metric =
        10.0 * (energy_cost + reserve_cost) + missiles_cost + supers_cost + power_bombs_cost;
    [ammo_sensitive_cost_metric, energy_sensitive_cost_metric]
}

fn validate_energy(mut local: LocalState, global: &GlobalState) -> Option<LocalState> {
    if local.energy_used >= global.max_energy {
        local.reserves_used += local.energy_used - (global.max_energy - 1);
        local.energy_used = global.max_energy - 1;
    }
    if local.reserves_used > global.max_reserves {
        return None;
    }
    Some(local)
}

fn validate_missiles(local: LocalState, global: &GlobalState) -> Option<LocalState> {
    if local.missiles_used > global.max_missiles {
        None
    } else {
        Some(local)
    }
}

fn validate_supers(local: LocalState, global: &GlobalState) -> Option<LocalState> {
    if local.supers_used > global.max_supers {
        None
    } else {
        Some(local)
    }
}

fn validate_power_bombs(local: LocalState, global: &GlobalState) -> Option<LocalState> {
    if local.power_bombs_used > global.max_power_bombs {
        None
    } else {
        Some(local)
    }
}

fn multiply(amount: Capacity, difficulty: &DifficultyConfig) -> Capacity {
    ((amount as f32) * difficulty.resource_multiplier) as Capacity
}

fn suit_damage_factor(global: &GlobalState) -> Capacity {
    let varia = global.items[Item::Varia as usize];
    let gravity = global.items[Item::Gravity as usize];
    if gravity && varia {
        4
    } else if gravity || varia {
        2
    } else {
        1
    }
}

fn apply_gate_glitch_leniency(
    mut local: LocalState,
    global: &GlobalState,
    green: bool,
    heated: bool,
    difficulty: &DifficultyConfig,
) -> Option<LocalState> {
    if heated && !global.items[Item::Varia as usize] {
        local.energy_used +=
            (difficulty.gate_glitch_leniency as f32 * difficulty.resource_multiplier * 60.0) as i32;
        local = match validate_energy(local, global) {
            Some(x) => x,
            None => return None,
        };
    }
    if green {
        local.supers_used += difficulty.gate_glitch_leniency;
        return validate_supers(local, global);
    } else {
        let missiles_available = global.max_missiles - local.missiles_used;
        if missiles_available >= difficulty.gate_glitch_leniency {
            local.missiles_used += difficulty.gate_glitch_leniency;
            return validate_missiles(local, global);
        } else {
            local.missiles_used = global.max_missiles;
            local.supers_used += difficulty.gate_glitch_leniency - missiles_available;
            return validate_supers(local, global);
        }
    }
}

pub fn apply_requirement(
    req: &Requirement,
    global: &GlobalState,
    local: LocalState,
    reverse: bool,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
) -> Option<LocalState> {
    match req {
        Requirement::Free => Some(local),
        Requirement::Never => None,
        Requirement::Tech(tech_id) => {
            if global.tech[*tech_id] {
                Some(local)
            } else {
                None
            }
        }
        Requirement::Strat(strat_id) => {
            if global.notable_strats[*strat_id] {
                Some(local)
            } else {
                None
            }
        }
        Requirement::Item(item_id) => {
            if global.items[*item_id] {
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
            // if !global.flags[*flag_id] {
            //     Some(local)
            // } else {
            //     None
            // }
        }
        Requirement::Walljump => {
            match difficulty.wall_jump {
                WallJump::Vanilla => {
                    if global.tech[game_data.wall_jump_tech_id] {
                        Some(local)
                    } else {
                        None
                    }        
                }
                WallJump::Collectible => {
                    if global.tech[game_data.wall_jump_tech_id] && global.items[Item::WallJump as usize] {
                        Some(local)
                    } else {
                        None
                    }        
                },
                WallJump::Disabled => {
                    None
                }
            }
        }
        Requirement::HeatFrames(frames) => {
            let varia = global.items[Item::Varia as usize];
            let mut new_local = local;
            if varia {
                Some(new_local)
            } else {
                if !global.tech[game_data.heat_run_tech_id] {
                    None
                } else {
                    new_local.energy_used += multiply(frames / 4, difficulty);
                    validate_energy(new_local, global)
                }
            }
        }
        Requirement::LavaFrames(frames) => {
            let varia = global.items[Item::Varia as usize];
            let gravity = global.items[Item::Gravity as usize];
            let mut new_local = local;
            // if gravity {
            if gravity && varia {
                Some(new_local)
            } else if gravity || varia {
                new_local.energy_used += multiply(frames / 4, difficulty);
                validate_energy(new_local, global)
            } else {
                new_local.energy_used += multiply(frames / 2, difficulty);
                validate_energy(new_local, global)
            }
        }
        Requirement::GravitylessLavaFrames(frames) => {
            let varia = global.items[Item::Varia as usize];
            let mut new_local = local;
            if varia {
                new_local.energy_used += multiply(frames / 4, difficulty);
            } else {
                new_local.energy_used += multiply(frames / 2, difficulty);
            }
            validate_energy(new_local, global)
        }
        Requirement::AcidFrames(frames) => {
            let mut new_local = local;
            new_local.energy_used +=
                multiply(3 * frames / 2, difficulty) / suit_damage_factor(global);
            validate_energy(new_local, global)
        }
        Requirement::MetroidFrames(frames) => {
            let mut new_local = local;
            new_local.energy_used +=
                multiply(3 * frames / 4, difficulty) / suit_damage_factor(global);
            validate_energy(new_local, global)
        }
        Requirement::Damage(base_energy) => {
            let mut new_local = local;
            new_local.energy_used += base_energy / suit_damage_factor(global);
            validate_energy(new_local, global)
        }
        // Requirement::Energy(count) => {
        //     let mut new_local = local;
        //     new_local.energy_used += count;
        //     validate_energy(new_local, global)
        // },
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
            apply_gate_glitch_leniency(local, global, *green, *heated, difficulty)
        }
        Requirement::HeatedDoorStuckLeniency { heat_frames } => {
            if !global.items[Item::Varia as usize] {
                let mut new_local = local;
                new_local.energy_used += (difficulty.door_stuck_leniency as f32
                    * difficulty.resource_multiplier
                    * *heat_frames as f32) as i32;
                validate_energy(new_local, global)
            } else {
                Some(local)
            }
        }
        Requirement::MissilesCapacity(count) => {
            if global.max_missiles >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::SupersCapacity(count) => {
            if global.max_supers >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::PowerBombsCapacity(count) => {
            if global.max_power_bombs >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::EnergyRefill => {
            let mut new_local = local;
            new_local.energy_used = 0;
            Some(new_local)
        }
        Requirement::ReserveRefill => {
            let mut new_local = local;
            new_local.reserves_used = 0;
            Some(new_local)
        }
        Requirement::MissileRefill => {
            let mut new_local = local;
            new_local.missiles_used = 0;
            Some(new_local)
        }
        Requirement::SuperRefill => {
            let mut new_local = local;
            new_local.supers_used = 0;
            Some(new_local)
        }
        Requirement::PowerBombRefill => {
            let mut new_local = local;
            new_local.power_bombs_used = 0;
            Some(new_local)
        }
        Requirement::AmmoStationRefill => {
            let mut new_local = local;
            new_local.missiles_used = 0;
            if !difficulty.ultra_low_qol {
                new_local.supers_used = 0;
                new_local.power_bombs_used = 0;
            }
            Some(new_local)
        }
        Requirement::EnergyDrain => {
            if reverse {
                let mut new_local = local;
                new_local.reserves_used += new_local.energy_used;
                new_local.energy_used = 0;
                if new_local.reserves_used > global.max_reserves {
                    None
                } else {
                    Some(new_local)
                }
            } else {
                let mut new_local = local;
                new_local.energy_used = global.max_energy - 1;
                Some(new_local)
            }
        }
        Requirement::ReserveTrigger {
            min_reserve_energy,
            max_reserve_energy,
        } => {
            if reverse {
                if local.reserves_used > 0
                    || local.energy_used >= min(*max_reserve_energy, global.max_reserves)
                {
                    None
                } else {
                    let mut new_local = local;
                    new_local.energy_used = 0;
                    new_local.reserves_used = max(local.energy_used + 1, *min_reserve_energy);
                    Some(new_local)
                }
            } else {
                let reserve_energy = min(
                    global.max_reserves - local.reserves_used,
                    *max_reserve_energy,
                );
                if reserve_energy >= *min_reserve_energy {
                    let mut new_local = local;
                    new_local.reserves_used = global.max_reserves;
                    new_local.energy_used = max(0, global.max_energy - reserve_energy);
                    Some(new_local)
                } else {
                    None
                }
            }
        }
        Requirement::EnemyKill { count, vul } => {
            apply_enemy_kill_requirement(global, local, *count, vul)
        }
        Requirement::PhantoonFight {} => {
            apply_phantoon_requirement(global, local, difficulty.phantoon_proficiency)
        }
        Requirement::DraygonFight {
            can_be_very_patient_tech_id,
        } => apply_draygon_requirement(
            global,
            local,
            difficulty.draygon_proficiency,
            *can_be_very_patient_tech_id,
        ),
        Requirement::RidleyFight {
            can_be_very_patient_tech_id,
        } => apply_ridley_requirement(
            global,
            local,
            difficulty.ridley_proficiency,
            *can_be_very_patient_tech_id,
        ),
        Requirement::BotwoonFight { second_phase } => {
            apply_botwoon_requirement(global, local, difficulty.botwoon_proficiency, *second_phase)
        }
        Requirement::ShineCharge { used_tiles } => {
            if global.items[Item::SpeedBooster as usize] && *used_tiles >= global.shine_charge_tiles
            {
                Some(local)
            } else {
                None
            }
        }
        Requirement::Shinespark {
            frames,
            excess_frames,
            shinespark_tech_id,
        } => {
            if global.tech[*shinespark_tech_id] {
                let mut new_local = local;
                if reverse {
                    if new_local.energy_used <= 28 {
                        new_local.energy_used = 28 + frames - excess_frames;
                    } else {
                        new_local.energy_used += frames;
                    }
                    validate_energy(new_local, global)
                } else {
                    new_local.energy_used += frames - excess_frames + 28;
                    if let Some(mut new_local) = validate_energy(new_local, global) {
                        let energy_remaining = global.max_energy - new_local.energy_used - 1;
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
        Requirement::AdjacentRunway { .. } => {
            panic!("AdjacentRunway should be resolved during preprocessing")
        }
        Requirement::AdjacentJumpway { .. } => {
            panic!("AdjacentJumpway should be resolved during preprocessing")
        }
        Requirement::CanComeInCharged { .. } => {
            panic!("CanComeInCharged should be resolved during preprocessing")
        }
        Requirement::ComeInWithRMode { .. } => {
            panic!("ComeInWithRMode should be resolved during preprocessing")
        }
        Requirement::ComeInWithGMode { .. } => {
            panic!("ComeInWithGMode should be resolved during preprocessing")
        }
        Requirement::DoorUnlocked { .. } => {
            panic!("DoorUnlocked should be resolved during preprocessing")
        }
        Requirement::And(reqs) => {
            let mut new_local = local;
            if reverse {
                for req in reqs.into_iter().rev() {
                    new_local =
                        apply_requirement(req, global, new_local, reverse, difficulty, game_data)?;
                }
            } else {
                for req in reqs {
                    new_local =
                        apply_requirement(req, global, new_local, reverse, difficulty, game_data)?;
                }
            }
            Some(new_local)
        }
        Requirement::Or(reqs) => {
            let mut best_local = None;
            let mut best_cost = [f32::INFINITY; NUM_COST_METRICS];
            for req in reqs {
                if let Some(new_local) =
                    apply_requirement(req, global, local, reverse, difficulty, game_data)
                {
                    let cost = compute_cost(new_local, global);
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

pub fn is_bireachable_state(
    global: &GlobalState,
    forward: LocalState,
    reverse: LocalState,
) -> bool {
    if forward.reserves_used + reverse.reserves_used > global.max_reserves {
        return false;
    }
    let forward_total_energy_used = forward.energy_used + forward.reserves_used;
    let reverse_total_energy_used = reverse.energy_used + reverse.reserves_used;
    let max_total_energy = global.max_energy + global.max_reserves;
    if forward_total_energy_used + reverse_total_energy_used >= max_total_energy {
        return false;
    }
    if forward.missiles_used + reverse.missiles_used > global.max_missiles {
        return false;
    }
    if forward.supers_used + reverse.supers_used > global.max_supers {
        return false;
    }
    if forward.power_bombs_used + reverse.power_bombs_used > global.max_power_bombs {
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
    difficulty: &DifficultyConfig,
    game_data: &GameData,
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
        result.cost[start_vertex_id] = compute_cost(init_local, global);
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
                    if let Some(dst_new_local_state) = apply_requirement(
                        &link.requirement,
                        global,
                        src_local_state,
                        reverse,
                        difficulty,
                        game_data,
                    ) {
                        let dst_new_cost_arr = compute_cost(dst_new_local_state, global);
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
                            new_modified_vertices.insert(dst_id, improved_arr);
                            result.step_trails.push(new_step_trail);
                        }
                    }
                }
            }
        }
        modified_vertices = new_modified_vertices;
    }

    // for x in &result.local_state_history {
    //     println!("{}", x.len());
    // }

    result
}

impl GlobalState {
    pub fn collect(&mut self, item: Item, game_data: &GameData) {
        self.items[item as usize] = true;
        match item {
            Item::Missile => {
                self.max_missiles += 5;
            }
            Item::Super => {
                self.max_supers += 5;
            }
            Item::PowerBomb => {
                self.max_power_bombs += 5;
            }
            Item::ETank => {
                self.max_energy += 100;
            }
            Item::ReserveTank => {
                self.max_reserves += 100;
            }
            _ => {}
        }
        self.weapon_mask = game_data.get_weapon_mask(&self.items);
    }
}

pub fn get_spoiler_route(traverse_result: &TraverseResult, vertex_id: usize, cost_idx: usize) -> Vec<LinkIdx> {
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
