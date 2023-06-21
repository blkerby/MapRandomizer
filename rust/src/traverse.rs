use std::{
    cmp::{max, min},
    mem::swap,
};

use hashbrown::HashSet;

use crate::{
    game_data::{Capacity, EnemyVulnerabilities, GameData, Item, Link, Requirement, WeaponMask},
    randomize::DifficultyConfig,
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
    can_be_patient_tech_id: usize,
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

    if kill_time >= 180.0 && !global.tech[can_be_patient_tech_id] {
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
    can_be_patient_tech_id: usize,
) -> Option<LocalState> {
    let mut boss_hp: f32 = 18000.0;
    let mut time: f32 = 0.0; // Cumulative time in seconds for the fight

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
        let charge_damage = get_charge_damage(&global);
        let charge_shots_to_use = max(
            0,
            f32::ceil(boss_hp / (charge_damage * accuracy)) as Capacity,
        );
        boss_hp = 0.0;
        time += charge_shots_to_use as f32 * 1.5 / firing_rate; // Assume max 1 charge shot per 1.5 seconds
    } else if global.items[Item::Morph as usize]{
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

    if time >= 180.0 && !global.tech[can_be_patient_tech_id] {
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

fn compute_cost(local: LocalState, global: &GlobalState) -> f32 {
    let eps = 1e-15;
    let energy_cost = (local.energy_used as f32) / (global.max_energy as f32 + eps);
    let reserve_cost = (local.reserves_used as f32) / (global.max_reserves as f32 + eps);
    let missiles_cost = (local.missiles_used as f32) / (global.max_missiles as f32 + eps);
    let supers_cost = (local.supers_used as f32) / (global.max_supers as f32 + eps);
    let power_bombs_cost = (local.power_bombs_used as f32) / (global.max_power_bombs as f32 + eps);
    energy_cost + reserve_cost + missiles_cost + supers_cost + power_bombs_cost
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

pub fn apply_requirement(
    req: &Requirement,
    global: &GlobalState,
    local: LocalState,
    reverse: bool,
    difficulty: &DifficultyConfig,
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
        Requirement::NotFlag(flag_id) => {
            if !global.flags[*flag_id] {
                Some(local)
            } else {
                None
            }
        }
        Requirement::HeatFrames(frames) => {
            let varia = global.items[Item::Varia as usize];
            // let gravity = global.items[Item::Gravity as usize];
            let mut new_local = local;
            if varia {
                Some(new_local)
            // } else if gravity {
            //     new_local.energy_used += multiply(frames / 8, difficulty);
            //     validate_energy(new_local, global)
            } else {
                new_local.energy_used += multiply(frames / 4, difficulty);
                validate_energy(new_local, global)
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
        Requirement::LavaPhysicsFrames(frames) => {
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
            new_local.missiles_used += multiply(*count, difficulty);
            validate_missiles(new_local, global)
        }
        Requirement::MissilesCapacity(count) => {
            if global.max_missiles >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::Supers(count) => {
            let mut new_local = local;
            new_local.supers_used += multiply(*count, difficulty);
            validate_supers(new_local, global)
        }
        Requirement::PowerBombs(count) => {
            let mut new_local = local;
            new_local.power_bombs_used += multiply(*count, difficulty);
            validate_power_bombs(new_local, global)
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
        Requirement::ReserveTrigger { min_reserve_energy, max_reserve_energy } => {
            if reverse {
                if local.reserves_used > 0 || local.energy_used >= min(*max_reserve_energy, global.max_reserves) {
                    None
                } else {
                    let mut new_local = local;
                    new_local.energy_used = 0;
                    new_local.reserves_used = max(local.energy_used + 1, *min_reserve_energy);
                    Some(new_local)
                }
            } else {
                let reserve_energy = min(global.max_reserves - local.reserves_used, *max_reserve_energy);
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
            can_be_patient_tech_id,
        } => apply_draygon_requirement(
            global,
            local,
            difficulty.draygon_proficiency,
            *can_be_patient_tech_id,
        ),
        Requirement::RidleyFight {
            can_be_patient_tech_id,
        } => apply_ridley_requirement(
            global,
            local,
            difficulty.ridley_proficiency,
            *can_be_patient_tech_id,
        ),
        Requirement::BotwoonFight { second_phase } => {
            apply_botwoon_requirement(global, local, difficulty.botwoon_proficiency, *second_phase)
        }
        Requirement::ShineCharge {
            used_tiles,
            shinespark_frames,
            excess_shinespark_frames,
            shinespark_tech_id,
        } => {
            if (global.tech[*shinespark_tech_id] || *shinespark_frames == 0)
                && global.items[Item::SpeedBooster as usize]
                && *used_tiles >= global.shine_charge_tiles
            {
                let mut new_local = local;
                if *shinespark_frames == 0 {
                    Some(new_local)
                } else {
                    if reverse {
                        if new_local.energy_used <= 28 {
                            new_local.energy_used = 28 + shinespark_frames - excess_shinespark_frames;
                        } else {
                            new_local.energy_used += shinespark_frames;
                        }
                        validate_energy(new_local, global)
                    } else {
                        new_local.energy_used += shinespark_frames - excess_shinespark_frames + 28;
                        if let Some(mut new_local) = validate_energy(new_local, global) {
                            let energy_remaining = global.max_energy - new_local.energy_used - 1;
                            new_local.energy_used +=
                                std::cmp::min(*excess_shinespark_frames, energy_remaining);
                            new_local.energy_used -= 28;
                            Some(new_local)
                        } else {
                            None
                        }
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
        Requirement::And(reqs) => {
            let mut new_local = local;
            if reverse {
                for req in reqs.into_iter().rev() {
                    new_local = apply_requirement(req, global, new_local, reverse, difficulty)?;
                }
            } else {
                for req in reqs {
                    new_local = apply_requirement(req, global, new_local, reverse, difficulty)?;
                }
            }
            Some(new_local)
        }
        Requirement::Or(reqs) => {
            let mut best_local = None;
            let mut best_cost = f32::INFINITY;
            for req in reqs {
                if let Some(new_local) = apply_requirement(req, global, local, reverse, difficulty)
                {
                    let cost = compute_cost(new_local, global);
                    if cost < best_cost {
                        best_cost = cost;
                        best_local = Some(new_local);
                    }
                }
            }
            best_local
        }
    }
}

pub fn is_bireachable(
    global: &GlobalState,
    forward_local_state: &Option<LocalState>,
    reverse_local_state: &Option<LocalState>,
) -> bool {
    if forward_local_state.is_none() || reverse_local_state.is_none() {
        return false;
    }
    let forward = forward_local_state.unwrap();
    let reverse = reverse_local_state.unwrap();
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

pub type StepTrailId = i32;
pub type LinkIdx = u32;

#[derive(Clone)]
pub struct StepTrail {
    pub prev_trail_id: StepTrailId,
    pub link_idx: LinkIdx,
}

#[derive(Clone)]
pub struct TraverseResult {
    pub local_states: Vec<Option<LocalState>>,
    pub cost: Vec<f32>,
    pub step_trails: Vec<StepTrail>,
    pub start_trail_ids: Vec<Option<StepTrailId>>,
}

pub fn traverse(
    links: &[Link],
    init_opt: Option<TraverseResult>,
    global: &GlobalState,
    num_vertices: usize,
    start_vertex_id: usize,
    reverse: bool,
    difficulty: &DifficultyConfig,
    _game_data: &GameData, // May be used for debugging
) -> TraverseResult {
    let mut modified_vertices: HashSet<usize> = HashSet::new();
    let mut result: TraverseResult;

    if let Some(init) = init_opt {
        for (v, local) in init.local_states.iter().enumerate() {
            if local.is_some() {
                modified_vertices.insert(v);
            }
        }
        result = init;
    } else {
        result = TraverseResult {
            local_states: vec![None; num_vertices],
            cost: vec![f32::INFINITY; num_vertices],
            step_trails: Vec::with_capacity(num_vertices * 10),
            start_trail_ids: vec![None; num_vertices],
        };
        result.local_states[start_vertex_id] = Some(LocalState::new());
        result.start_trail_ids[start_vertex_id] = Some(-1);
        result.cost[start_vertex_id] =
            compute_cost(result.local_states[start_vertex_id].unwrap(), global);
        modified_vertices.insert(start_vertex_id);
    }

    let mut links_by_src: Vec<Vec<(LinkIdx, Link)>> = vec![Vec::new(); num_vertices];
    for (idx, link) in links.iter().enumerate() {
        if reverse {
            let mut reversed_link = link.clone();
            swap(
                &mut reversed_link.from_vertex_id,
                &mut reversed_link.to_vertex_id,
            );
            links_by_src[reversed_link.from_vertex_id].push((idx as LinkIdx, reversed_link));
        } else {
            links_by_src[link.from_vertex_id].push((idx as LinkIdx, link.clone()));
        }
    }

    while modified_vertices.len() > 0 {
        let mut new_modified_vertices: HashSet<usize> = HashSet::new();
        for &src_id in &modified_vertices {
            let src_local_state = result.local_states[src_id].unwrap();
            let src_trail_id = result.start_trail_ids[src_id].unwrap();
            for &(link_idx, ref link) in &links_by_src[src_id] {
                let dst_id = link.to_vertex_id;
                let dst_old_cost = result.cost[dst_id];
                if let Some(dst_new_local_state) = apply_requirement(
                    &link.requirement,
                    global,
                    src_local_state,
                    reverse,
                    difficulty,
                ) {
                    let dst_new_cost = compute_cost(dst_new_local_state, global);
                    if dst_new_cost < dst_old_cost {
                        let new_step_trail = StepTrail {
                            prev_trail_id: src_trail_id,
                            link_idx: link_idx,
                        };
                        let new_trail_id = result.step_trails.len() as StepTrailId;
                        result.step_trails.push(new_step_trail);
                        result.local_states[dst_id] = Some(dst_new_local_state);
                        result.start_trail_ids[dst_id] = Some(new_trail_id);
                        result.cost[dst_id] = dst_new_cost;
                        new_modified_vertices.insert(dst_id);
                    }
                }
            }
        }
        modified_vertices = new_modified_vertices;
    }

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

pub fn get_spoiler_route(traverse_result: &TraverseResult, vertex_id: usize) -> Vec<LinkIdx> {
    let mut trail_id = traverse_result.start_trail_ids[vertex_id].unwrap();
    let mut steps: Vec<LinkIdx> = Vec::new();
    while trail_id != -1 {
        let step_trail = &traverse_result.step_trails[trail_id as usize];
        steps.push(step_trail.link_idx);
        trail_id = step_trail.prev_trail_id;
    }
    steps.reverse();
    steps
}
