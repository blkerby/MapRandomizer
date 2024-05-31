use std::{
    cmp::{max, min},
    mem::swap,
};

use hashbrown::{HashMap, HashSet};

use crate::{
    game_data::{
        self, Capacity, EnemyDrop, EnemyVulnerabilities, GameData, Item, Link, LinkIdx, LinksDataGroup, NodeId, Requirement, RoomId, VertexId, WeaponMask
    },
    randomize::{BeamType, DifficultyConfig, DoorType, LockedDoor, MotherBrainFight, Objective, WallJump},
};

use log::info;

// TODO: move tech and notable_strats out of this struct, since these do not change from step to step.
#[derive(Clone, Debug)]
pub struct GlobalState {
    pub tech: Vec<bool>,
    pub notable_strats: Vec<bool>,
    pub items: Vec<bool>,
    pub flags: Vec<bool>,
    pub doors_unlocked: Vec<bool>,
    pub max_energy: Capacity,
    pub max_reserves: Capacity,
    pub max_missiles: Capacity,
    pub max_supers: Capacity,
    pub max_power_bombs: Capacity,
    pub weapon_mask: WeaponMask,
    pub shine_charge_tiles: f32,
    pub heated_shine_charge_tiles: f32,
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
    pub shinecharge_frames_remaining: Capacity,
}

impl LocalState {
    pub fn new() -> Self {
        Self {
            energy_used: 0,
            reserves_used: 0,
            missiles_used: 0,
            supers_used: 0,
            power_bombs_used: 0,
            shinecharge_frames_remaining: 0,
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
    let ice = global.items[Item::Ice as usize];
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
        let missiles_available = global.max_missiles - local.missiles_used;
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
        let supers_available = global.max_supers - local.supers_used;
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
    if vul.power_bomb_damage > 0 && global.items[Item::Morph as usize] {
        let pbs_available = global.max_power_bombs - local.power_bombs_used;
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

fn apply_phantoon_requirement(
    global: &GlobalState,
    mut local: LocalState,
    proficiency: f32,
    game_data: &GameData,
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

    validate_energy(local, global, game_data)
}

fn apply_draygon_requirement(
    global: &GlobalState,
    local: LocalState,
    proficiency: f32,
    can_be_very_patient_tech_id: usize,
    game_data: &GameData,
) -> Option<LocalState> {
    let mut boss_hp: f32 = 6000.0;
    let charge_damage = get_charge_damage(&global);

    // Assume an accuracy of between 40% (on lowest difficulty) to 100% (on highest).
    let accuracy = 0.4 + 0.6 * proficiency;

    // Assume a firing rate of between 60% (on lowest difficulty) to 100% (on highest).
    let firing_rate = 0.6 + 0.4 * proficiency;

    const GOOP_CYCLES_PER_SECOND: f32 = 1.0 / 15.0;
    const SWOOP_CYCLES_PER_SECOND: f32 = GOOP_CYCLES_PER_SECOND * 2.0;

    // Assume a maximum of 1 charge shot per goop phase, and 1 charge shot per swoop.
    let charge_firing_rate = (SWOOP_CYCLES_PER_SECOND + GOOP_CYCLES_PER_SECOND) * firing_rate;
    let charge_damage_rate = charge_firing_rate * charge_damage * accuracy;

    let farm_proficiency = 0.2 + 0.8 * proficiency;
    let base_goop_farms_per_cycle = match (
        global.items[Item::Plasma as usize],
        global.items[Item::Wave as usize],
    ) {
        (false, _) => 7.0,    // Basic beam
        (true, false) => 10.0, // Plasma can hit multiple goops at once.
        (true, true) => 13.0, // Wave+Plasma can hit even more goops at once.
    };
    let goop_farms_per_cycle = if global.items[Item::Gravity as usize] {
        farm_proficiency * base_goop_farms_per_cycle
    } else {
        // Without Gravity you can't farm as many goops since you have to spend more time avoiding Draygon.
        0.75 * farm_proficiency * base_goop_farms_per_cycle
    };
    let energy_farm_rate =
        GOOP_CYCLES_PER_SECOND * goop_farms_per_cycle * (5.0 * 0.02 + 20.0 * 0.12);
    let missile_farm_rate = GOOP_CYCLES_PER_SECOND * goop_farms_per_cycle * (2.0 * 0.44);

    let base_hit_dps = if global.items[Item::Gravity as usize] {
        // With Gravity, assume one Draygon hit per two cycles as the maximum rate of damage to Samus:
        160.0 * 0.5 * (GOOP_CYCLES_PER_SECOND + SWOOP_CYCLES_PER_SECOND) * (1.0 - proficiency)
    } else {
        // Without Gravity, assume one Draygon hit per cycle as the maximum rate of damage to Samus:
        160.0 * (GOOP_CYCLES_PER_SECOND + SWOOP_CYCLES_PER_SECOND) * (1.0 - proficiency)
    };

    // We assume as many Supers are available can be used immediately (e.g. on the first goop cycle):
    let supers_available = global.max_supers - local.supers_used;
    boss_hp -= (supers_available as f32) * accuracy * 300.0;
    if boss_hp < 0.0 {
        return Some(local);
    }

    let missiles_available = global.max_missiles - local.missiles_used;
    let missile_firing_rate = 20.0 * GOOP_CYCLES_PER_SECOND * firing_rate;
    let net_missile_use_rate = missile_firing_rate - missile_farm_rate;

    let initial_missile_damage_rate = 100.0 * missile_firing_rate * accuracy;
    let overall_damage_rate = initial_missile_damage_rate + charge_damage_rate;
    let time_boss_dead = f32::ceil(boss_hp / overall_damage_rate);
    let time_missiles_exhausted = if global.max_missiles == 0 {
        0.0
    } else if net_missile_use_rate > 0.0 {
        (missiles_available as f32) / net_missile_use_rate
    } else {
        f32::INFINITY
    };
    let mut time = f32::min(time_boss_dead, time_missiles_exhausted);
    if time_missiles_exhausted < time_boss_dead {
        // Boss is not dead yet after exhausting all Missiles (if any).
        // Continue the fight using Missiles only at the lower rate at which they can be farmed (if available).
        boss_hp -= time * overall_damage_rate;

        let farming_missile_damage_rate = if global.max_missiles > 0 {
            100.0 * missile_farm_rate * accuracy
        } else {
            0.0
        };
        let overall_damage_rate = farming_missile_damage_rate + charge_damage_rate;
        if overall_damage_rate == 0.0 {
            return None;
        }
        time += boss_hp / overall_damage_rate;
    }

    if time < 180.0 || global.tech[can_be_very_patient_tech_id] {
        let mut net_dps = base_hit_dps / suit_damage_factor(global) as f32 - energy_farm_rate;
        if net_dps < 0.0 {
            net_dps = 0.0;
        }
        let total_damage = (net_dps * time) as Capacity;
        // We don't account for resources used, since they can be farmed or picked up after the fight, and we don't
        // want the fight to go out of logic due to not saving enough Missiles to open some red doors for example.
        let result = LocalState {
            energy_used: local.energy_used + (net_dps * time) as Capacity,
            ..local
        };
        if validate_energy(result, global, game_data).is_some() {
            return Some(local);
        } else {
            return None;
        }
    } else {
        return None;
    }
}

pub fn apply_ridley_requirement(
    global: &GlobalState,
    mut local: LocalState,
    proficiency: f32,
    can_be_very_patient_tech_id: usize,
    game_data: &GameData,
) -> Option<LocalState> {
    let mut boss_hp: f32 = 18000.0;
    let mut time: f32 = 0.0; // Cumulative time in seconds for the fight
    let charge_damage = get_charge_damage(&global);

    // Assume an ammo accuracy rate of between 60% (on lowest difficulty) to 100% (on highest):
    let accuracy = 0.6 + 0.4 * proficiency;

    // Assume a firing rate of between 30% (on lowest difficulty) to 100% (on highest):
    let firing_rate = 0.3 + 0.7 * proficiency;

    let charge_time = 1.4; // minimum of 1.4 seconds between charge shots

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
        time += powerful_charge_shots_to_use as f32 * charge_time / firing_rate;
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
    time += missiles_to_use as f32 * 0.34 / firing_rate; // Assume max average rate of 1 missile per 0.34 seconds

    if global.items[Item::Charge as usize] {
        // Then finish with Charge shots:
        // (TODO: it would be a little better to prioritize Charge shots over Supers/Missiles in
        // some cases).
        let charge_shots_to_use = max(
            0,
            f32::ceil(boss_hp / (charge_damage * accuracy)) as Capacity,
        );
        boss_hp = 0.0;
        time += charge_shots_to_use as f32 * charge_time / firing_rate;
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
    let base_ridley_attack_dps = 50.0;

    // Multiplier to Ridley damage based on items (Morph and Screw) and proficiency (in dodging).
    // This is a rough guess which could be refined. We could also take into account other items
    // (HiJump and SpaceJump). We assume that at Insane level (proficiency=1.0) it is possible
    // to avoid all damage from Ridley using either Morph or Screw.
    let hit_rate = match (morph, screw) {
        (false, false) => 1.0 - 0.8 * proficiency,
        (false, true) => 1.0 - 1.0 * proficiency,
        (true, false) => 0.5 - 0.5 * proficiency,
        (true, true) => 0.5 - 0.5 * proficiency,
    };
    let damage = base_ridley_attack_dps * hit_rate * time;
    local.energy_used += (damage / suit_damage_factor(global) as f32) as Capacity;

    if !global.items[Item::Varia as usize] {
        // Heat run case: We do not explicitly check canHeatRun tech here, because it is
        // already required to reach the boss node from the doors.
        // Include time pre- and post-fight when Samus must still take heat damage:
        let heat_time = time + 16.0;
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
    validate_energy(local, global, game_data)
}

fn apply_botwoon_requirement(
    global: &GlobalState,
    mut local: LocalState,
    proficiency: f32,
    second_phase: bool,
    game_data: &GameData,
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
    validate_energy(local, global, game_data)
}

fn apply_mother_brain_2_requirement(
    global: &GlobalState,
    mut local: LocalState,
    difficulty: &DifficultyConfig,
    can_be_very_patient_tech_id: usize,
    r_mode: bool,
    game_data: &GameData,
) -> Option<LocalState> {
    if difficulty.mother_brain_fight == MotherBrainFight::Skip {
        return Some(local);
    }

    if global.max_energy < 199 && !r_mode {
        // Need at least one ETank to survive rainbow beam, except in R-mode (where energy requirements are handled elsewhere
        // in the strat logic)
        return None;
    }

    let proficiency = difficulty.mother_brain_proficiency;
    let mut boss_hp: f32 = 18000.0;
    let mut time: f32 = 0.0; // Cumulative time in seconds for the fight
    let charge_damage = get_charge_damage(&global);

    // Assume an ammo accuracy rate of between 75% (on lowest difficulty) to 100% (on highest):
    let accuracy = 0.75 + 0.25 * proficiency;

    // Assume a firing rate of between 60% (on lowest difficulty) to 100% (on highest):
    let firing_rate = 0.6 + 0.4 * proficiency;

    let charge_time = 1.1; // minimum of 1.1 seconds between charge shots
    let super_time = 0.35; // minimum of 0.35 seconds between Super shots
    let missile_time = 0.17;

    // Prioritize using supers:
    let supers_available = global.max_supers - local.supers_used;
    let super_damage = if difficulty.supers_double {
        600.0
    } else {
        300.0
    };
    let supers_to_use = min(
        supers_available,
        f32::ceil(boss_hp / (super_damage * accuracy)) as Capacity,
    );
    local.supers_used += supers_to_use;
    boss_hp -= supers_to_use as f32 * super_damage * accuracy;
    time += supers_to_use as f32 * super_time / firing_rate;

    // Use Charge Beam if it's powerful enough
    // 500 is the point at which Charge Beam has better DPS than Missiles, this happens with Charge + Plasma + (Ice and/or Wave)
    if charge_damage >= 500.0 {
        let powerful_charge_shots_to_use = max(
            0,
            f32::ceil(boss_hp / (charge_damage * accuracy)) as Capacity,
        );
        boss_hp = 0.0;
        time += powerful_charge_shots_to_use as f32 * charge_time / firing_rate;
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
    time += missiles_to_use as f32 * missile_time / firing_rate;

    if global.items[Item::Charge as usize] {
        // Then finish with Charge shots:
        let charge_shots_to_use = max(
            0,
            f32::ceil(boss_hp / (charge_damage * accuracy)) as Capacity,
        );
        boss_hp = 0.0;
        time += charge_shots_to_use as f32 * charge_time / firing_rate;
    }

    if boss_hp > 0.0 {
        // We don't have enough ammo to finish the fight:
        return None;
    }

    if time >= 180.0 && !global.tech[can_be_very_patient_tech_id] {
        // We don't have enough patience to finish the fight:
        return None;
    }

    // Assumed rate of Mother Brain damage to Samus (per second), given minimal dodging skill:
    // For simplicity we assume a uniform rate of damage (even though the ketchup phase has the highest risk of damage).
    let base_mb_attack_dps = 20.0;
    let hit_rate = 1.0 - proficiency;
    let damage = base_mb_attack_dps * hit_rate * time;
    if !r_mode {
        local.energy_used += (damage / suit_damage_factor(global) as f32) as Capacity;

        // Account for Rainbow beam damage:
        if global.items[Item::Varia as usize] {
            local.energy_used += 300;
        } else {
            local.energy_used += 600;
        }
    }

    validate_energy(local, global, game_data)
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

fn compute_cost(local: LocalState, global: &GlobalState) -> [f32; NUM_COST_METRICS] {
    let eps = 1e-15;
    let energy_cost = (local.energy_used as f32) / (global.max_energy as f32 + eps);
    let reserve_cost = (local.reserves_used as f32) / (global.max_reserves as f32 + eps);
    let missiles_cost = (local.missiles_used as f32) / (global.max_missiles as f32 + eps);
    let supers_cost = (local.supers_used as f32) / (global.max_supers as f32 + eps);
    let power_bombs_cost = (local.power_bombs_used as f32) / (global.max_power_bombs as f32 + eps);
    let shinecharge_cost = -(local.shinecharge_frames_remaining as f32) / 180.0;

    let ammo_sensitive_cost_metric =
        energy_cost + reserve_cost + 100.0 * (missiles_cost + supers_cost + power_bombs_cost + shinecharge_cost);
    let energy_sensitive_cost_metric =
        100.0 * (energy_cost + reserve_cost) + missiles_cost + supers_cost + power_bombs_cost + shinecharge_cost;
    [ammo_sensitive_cost_metric, energy_sensitive_cost_metric]
}

fn validate_energy(
    mut local: LocalState,
    global: &GlobalState,
    game_data: &GameData,
) -> Option<LocalState> {
    if local.energy_used >= global.max_energy {
        if global.tech[game_data.manage_reserves_tech_id] {
            // Assume that just enough reserve energy is manually converted to regular energy.
            local.reserves_used += local.energy_used - (global.max_energy - 1);
            local.energy_used = global.max_energy - 1;
        } else {
            // Assume that reserves auto-trigger, leaving reserves empty.
            let reserves_available = std::cmp::min(global.max_reserves - local.reserves_used, global.max_energy);
            local.reserves_used = global.max_reserves;
            local.energy_used = std::cmp::max(0, local.energy_used - reserves_available);
            if local.energy_used >= global.max_energy {
                return None;
            }
        }
    }
    if local.reserves_used > global.max_reserves {
        return None;
    }
    Some(local)
}

fn validate_energy_no_auto_reserve(
    mut local: LocalState,
    global: &GlobalState,
    game_data: &GameData,
) -> Option<LocalState> {
    if local.energy_used >= global.max_energy {
        if global.tech[game_data.manage_reserves_tech_id] {
            // Assume that just enough reserve energy is manually converted to regular energy.
            local.reserves_used += local.energy_used - (global.max_energy - 1);
            local.energy_used = global.max_energy - 1;
        } else {
            // Assume that reserves cannot be used (e.g. during a shinespark or enemy hit).
            return None;
        }
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
    game_data: &GameData,
) -> Option<LocalState> {
    if heated && !global.items[Item::Varia as usize] {
        local.energy_used +=
            (difficulty.gate_glitch_leniency as f32 * difficulty.resource_multiplier * 60.0) as Capacity;
        local = match validate_energy(local, global, game_data) {
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

fn is_objective_complete(
    global: &GlobalState,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
    obj_id: usize,
) -> bool {
    // TODO: What to do when obj_id is out of bounds?
    if let Some(obj) = difficulty.objectives.get(obj_id) {
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
    let varia = global.items[Item::Varia as usize];
    let mut new_local = local;
    if varia {
        Some(new_local)
    } else {
        if !global.tech[game_data.heat_run_tech_id] {
            None
        } else {
            new_local.energy_used += (multiply(frames, difficulty) + 3) / 4;
            validate_energy(new_local, global, game_data)
        }
    }
}

fn get_enemy_drop_value(drop: &EnemyDrop, local: LocalState, reverse: bool, buffed_drops: bool) -> Capacity {
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
    difficulty: &DifficultyConfig,
    reverse: bool,
) -> Option<LocalState> {
    let varia = global.items[Item::Varia as usize];
    let mut new_local = local;
    if varia {
        Some(new_local)
    } else {
        if !global.tech[game_data.heat_run_tech_id] {
            None
        } else {
            let mut total_drop_value = 0;
            for drop in drops {
                total_drop_value += get_enemy_drop_value(drop, local, reverse, difficulty.buffed_drops)
            }
            let heat_energy = (multiply(frames, difficulty) + 3) / 4;
            total_drop_value = Capacity::min(total_drop_value, heat_energy);
            new_local.energy_used += heat_energy;
            if let Some(x) = validate_energy(new_local, global, game_data) {
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
    difficulty: &DifficultyConfig,
    game_data: &GameData,
    locked_door_data: &LockedDoorData,
) -> Option<LocalState> {
    let mut new_local = apply_requirement(&link.requirement, global, local, reverse, difficulty, game_data, locked_door_data);
    if let Some(mut new_local) = new_local {
        if new_local.shinecharge_frames_remaining != 0 && !link.end_with_shinecharge {
            new_local.shinecharge_frames_remaining = 0;
        }
        Some(new_local)
    } else {
        None
    }
}

fn has_beam(beam: BeamType, global: &GlobalState) -> bool {
    let item = match beam {
        BeamType::Charge => Item::Charge,
        BeamType::Ice => Item::Ice,
        BeamType::Wave => Item::Wave,
        BeamType::Spazer => Item::Spazer,
        BeamType::Plasma => Item::Plasma,
    };
    global.items[item as usize]
}

pub fn apply_requirement(
    req: &Requirement,
    global: &GlobalState,
    local: LocalState,
    reverse: bool,
    difficulty: &DifficultyConfig,
    game_data: &GameData,
    locked_door_data: &LockedDoorData,
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
        Requirement::Objective(obj_id) => {
            if is_objective_complete(global, difficulty, game_data, *obj_id) {
                Some(local)
            } else {
                None
            }
        }
        Requirement::Walljump => match difficulty.wall_jump {
            WallJump::Vanilla => {
                if global.tech[game_data.wall_jump_tech_id] {
                    Some(local)
                } else {
                    None
                }
            }
            WallJump::Collectible => {
                if global.tech[game_data.wall_jump_tech_id] && global.items[Item::WallJump as usize]
                {
                    Some(local)
                } else {
                    None
                }
            }
        },
        Requirement::HeatFrames(frames) => {
            apply_heat_frames(*frames, local, global, game_data, difficulty)
        }
        Requirement::HeatFramesWithEnergyDrops(frames, enemy_drops) => {
            apply_heat_frames_with_energy_drops(*frames, enemy_drops, local, global, game_data, difficulty, reverse)
        }
        Requirement::MainHallElevatorFrames => {
            if difficulty.fast_elevators {
                apply_heat_frames(160, local, global, game_data, difficulty)
            } else {
                apply_heat_frames(436, local, global, game_data, difficulty)
            }
        }
        Requirement::LowerNorfairElevatorDownFrames => {
            if difficulty.fast_elevators {
                apply_heat_frames(24, local, global, game_data, difficulty)
            } else {
                apply_heat_frames(60, local, global, game_data, difficulty)
            }
        }
        Requirement::LowerNorfairElevatorUpFrames => {
            if difficulty.fast_elevators {
                apply_heat_frames(40, local, global, game_data, difficulty)
            } else {
                apply_heat_frames(108, local, global, game_data, difficulty)
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
                new_local.energy_used += (multiply(*frames, difficulty) + 3) / 4;
                validate_energy(new_local, global, game_data)
            } else {
                new_local.energy_used += (multiply(*frames, difficulty) + 1) / 2;
                validate_energy(new_local, global, game_data)
            }
        }
        Requirement::GravitylessLavaFrames(frames) => {
            let varia = global.items[Item::Varia as usize];
            let mut new_local = local;
            if varia {
                new_local.energy_used += (multiply(*frames, difficulty) + 3) / 4;
            } else {
                new_local.energy_used += (multiply(*frames, difficulty) + 1) / 2;
            }
            validate_energy(new_local, global, game_data)
        }
        Requirement::AcidFrames(frames) => {
            let mut new_local = local;
            new_local.energy_used +=
                multiply((3 * frames + 1) / 2, difficulty) / suit_damage_factor(global);
            validate_energy(new_local, global, game_data)
        }
        Requirement::GravitylessAcidFrames(frames) => {
            let varia = global.items[Item::Varia as usize];
            let mut new_local = local;
            if varia {
                new_local.energy_used += multiply((3 * frames + 3) / 4, difficulty);
            } else {
                new_local.energy_used += multiply((3 * frames + 1) / 2, difficulty);
            }
            validate_energy(new_local, global, game_data)
        }
        Requirement::MetroidFrames(frames) => {
            let mut new_local = local;
            new_local.energy_used +=
                multiply((3 * frames + 3) / 4, difficulty) / suit_damage_factor(global);
            validate_energy(new_local, global, game_data)
        }
        Requirement::Damage(base_energy) => {
            let mut new_local = local;
            let energy = base_energy / suit_damage_factor(global);
            if energy >= global.max_energy && !global.tech[game_data.pause_abuse_tech_id] {
                None
            } else {
                new_local.energy_used += energy;
                validate_energy_no_auto_reserve(new_local, global, game_data)    
            }
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
            apply_gate_glitch_leniency(local, global, *green, *heated, difficulty, game_data)
        }
        Requirement::HeatedDoorStuckLeniency { heat_frames } => {
            if !global.items[Item::Varia as usize] {
                let mut new_local = local;
                new_local.energy_used += (difficulty.door_stuck_leniency as f32
                    * difficulty.resource_multiplier
                    * *heat_frames as f32
                    / 4.0) as Capacity;
                validate_energy(new_local, global, game_data)
            } else {
                Some(local)
            }
        }
        Requirement::MissilesAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.max_missiles < *count {
                    None
                } else {
                    new_local.missiles_used = Capacity::max(new_local.missiles_used, *count);
                    Some(new_local)
                }
            } else {
                if global.max_missiles - local.missiles_used < *count {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::SupersAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.max_supers < *count {
                    None
                } else {
                    new_local.supers_used = Capacity::max(new_local.supers_used, *count);
                    Some(new_local)
                }
            } else {
                if global.max_supers - local.supers_used < *count {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::PowerBombsAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.max_power_bombs < *count {
                    None
                } else {
                    new_local.power_bombs_used = Capacity::max(new_local.power_bombs_used, *count);
                    Some(new_local)
                }
            } else {
                if global.max_power_bombs - local.power_bombs_used < *count {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::RegularEnergyAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.max_energy < *count {
                    None
                } else {
                    new_local.energy_used = Capacity::max(new_local.energy_used, *count);
                    Some(new_local)
                }
            } else {
                if global.max_energy - local.energy_used < *count {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::ReserveEnergyAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.max_reserves < *count {
                    None
                } else {
                    new_local.reserves_used = Capacity::max(new_local.reserves_used, *count);
                    Some(new_local)
                }
            } else {
                if global.max_reserves - local.reserves_used < *count {
                    None
                } else {
                    Some(local)
                }
            }
        }
        Requirement::EnergyAvailable(count) => {
            if reverse {
                let mut new_local = local;
                if global.max_energy + global.max_reserves < *count {
                    None
                } else {
                    if global.max_energy < *count {
                        new_local.energy_used = global.max_energy;
                        new_local.reserves_used = Capacity::max(new_local.reserves_used, *count - global.max_energy);
                        Some(new_local)
                    } else {
                        new_local.energy_used = Capacity::max(new_local.energy_used, *count);
                        Some(new_local)
                    }
                }
            } else {
                if global.max_reserves - local.reserves_used + global.max_energy - local.energy_used < *count {
                    None
                } else {
                    Some(local)
                }
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
        Requirement::RegularEnergyCapacity(count) => {
            if global.max_energy >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::ReserveEnergyCapacity(count) => {
            if global.max_reserves >= *count {
                Some(local)
            } else {
                None
            }
        }
        Requirement::EnergyRefill(limit) => {
            let limit_reserves = max(0, *limit - global.max_energy);
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
                if local.energy_used > global.max_energy - limit {
                    new_local.energy_used = max(0, global.max_energy - limit);
                }
                if local.reserves_used > global.max_reserves - limit_reserves {
                    new_local.reserves_used = max(0, global.max_reserves - limit_reserves);
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
                if local.energy_used > global.max_energy - limit {
                    new_local.energy_used = max(0, global.max_energy - limit);
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
                if local.reserves_used > global.max_reserves - limit {
                    new_local.reserves_used = max(0, global.max_reserves - limit);
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
                if local.missiles_used > global.max_missiles - limit {
                    new_local.missiles_used = max(0, global.max_missiles - limit);
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
                if local.supers_used > global.max_supers - limit {
                    new_local.supers_used = max(0, global.max_supers - limit);
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
                if local.power_bombs_used > global.max_power_bombs - limit {
                    new_local.power_bombs_used = max(0, global.max_power_bombs - limit);
                }
                Some(new_local)    
            }
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
        Requirement::AmmoStationRefillAll => {
            if difficulty.ultra_low_qol {
                None
            } else {
                Some(local)
            }
        }
        Requirement::SupersDoubleDamageMotherBrain => {
            if difficulty.supers_double {
                Some(local)
            } else {
                None
            }
        }
        Requirement::ShinesparksCostEnergy => {
            if difficulty.energy_free_shinesparks {
                None
            } else {
                Some(local)
            }            
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
            apply_phantoon_requirement(global, local, difficulty.phantoon_proficiency, game_data)
        }
        Requirement::DraygonFight {
            can_be_very_patient_tech_id,
        } => apply_draygon_requirement(
            global,
            local,
            difficulty.draygon_proficiency,
            *can_be_very_patient_tech_id,
            game_data,
        ),
        Requirement::RidleyFight {
            can_be_very_patient_tech_id,
        } => apply_ridley_requirement(
            global,
            local,
            difficulty.ridley_proficiency,
            *can_be_very_patient_tech_id,
            game_data,
        ),
        Requirement::BotwoonFight { second_phase } => apply_botwoon_requirement(
            global,
            local,
            difficulty.botwoon_proficiency,
            *second_phase,
            game_data,
        ),
        Requirement::MotherBrain2Fight {
            can_be_very_patient_tech_id,
            r_mode,
        } => apply_mother_brain_2_requirement(
            global,
            local,
            difficulty,
            *can_be_very_patient_tech_id,
            *r_mode,
            game_data,
        ),
        Requirement::ShineCharge { used_tiles, heated } => {
            let used_tiles = used_tiles.get();
            let tiles_limit = if *heated && !global.items[Item::Varia as usize] {
                global.heated_shine_charge_tiles
            } else {
                global.shine_charge_tiles
            };
            if global.items[Item::SpeedBooster as usize] && used_tiles >= tiles_limit {
                let mut new_local = local;
                if reverse {
                    new_local.shinecharge_frames_remaining = 0;
                } else {
                    new_local.shinecharge_frames_remaining = 180 - difficulty.shinecharge_leniency_frames;
                }
                Some(local)
            } else {
                None
            }
        }
        Requirement::ShineChargeFrames(frames) => {
            let mut new_local = local;
            if reverse {
                new_local.shinecharge_frames_remaining += frames;
                if new_local.shinecharge_frames_remaining <= 180 - difficulty.shinecharge_leniency_frames {
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
            shinespark_tech_id,
        } => {
            if global.tech[*shinespark_tech_id] {
                let mut new_local = local;
                if difficulty.energy_free_shinesparks {
                    return Some(new_local);
                }
                if reverse {
                    if new_local.energy_used <= 28 {
                        new_local.energy_used = 28 + frames - excess_frames;
                    } else {
                        new_local.energy_used += frames;
                    }
                    validate_energy_no_auto_reserve(new_local, global, game_data)
                } else {
                    new_local.energy_used += frames - excess_frames + 28;
                    if let Some(mut new_local) =
                        validate_energy_no_auto_reserve(new_local, global, game_data)
                    {
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
        Requirement::DoorUnlocked { room_id, node_id } => {
            if let Some(locked_door_idx) = locked_door_data.locked_door_node_map.get(&(*room_id, *node_id)) {
                if global.doors_unlocked[*locked_door_idx] {
                    Some(local)
                } else {
                    None
                }
            } else {
                Some(local)
            }
        }
        Requirement::DoorType { room_id, node_id, door_type } => {
            let actual_door_type = if let Some(locked_door_idx) = locked_door_data.locked_door_node_map.get(&(*room_id, *node_id)) {
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
        Requirement::UnlockDoor { room_id, node_id, requirement_red, requirement_green, requirement_yellow, requirement_charge } => {
            if let Some(locked_door_idx) = locked_door_data.locked_door_node_map.get(&(*room_id, *node_id)) {
                let door_type = locked_door_data.locked_doors[*locked_door_idx].door_type;
                if global.doors_unlocked[*locked_door_idx] {
                    return Some(local);
                }
                match door_type {
                    DoorType::Blue => {
                        Some(local)
                    }
                    DoorType::Red => {
                        apply_requirement(requirement_red, global, local, reverse, difficulty, game_data, locked_door_data)
                    }
                    DoorType::Green => {
                        apply_requirement(requirement_green, global, local, reverse, difficulty, game_data, locked_door_data)
                    }
                    DoorType::Yellow => {
                        apply_requirement(requirement_yellow, global, local, reverse, difficulty, game_data, locked_door_data)
                    }
                    DoorType::Beam(beam) => {
                        if has_beam(beam, global) { 
                            if let BeamType::Charge = beam {
                                apply_requirement(requirement_charge, global, local, reverse, difficulty, game_data, locked_door_data)
                            } else {
                                Some(local) 
                            }
                        } else { 
                            None 
                        }
                    }
                    DoorType::Gray => panic!("Unexpected gray door while processing Requirement::UnlockDoor")
                }
            } else {
                Some(local)
            }
        }
        Requirement::And(reqs) => {
            let mut new_local = local;
            if reverse {
                for req in reqs.into_iter().rev() {
                    new_local =
                        apply_requirement(req, global, new_local, reverse, difficulty, game_data, locked_door_data)?;
                }
            } else {
                for req in reqs {
                    new_local =
                        apply_requirement(req, global, new_local, reverse, difficulty, game_data, locked_door_data)?;
                }
            }
            Some(new_local)
        }
        Requirement::Or(reqs) => {
            let mut best_local = None;
            let mut best_cost = [f32::INFINITY; NUM_COST_METRICS];
            for req in reqs {
                if let Some(new_local) =
                    apply_requirement(req, global, local, reverse, difficulty, game_data, locked_door_data)
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

pub fn is_reachable_state(
    local: LocalState
) -> bool {
    local.energy_used != IMPOSSIBLE_LOCAL_STATE.energy_used
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

// If the given vertex is reachable, returns a cost metric index (between 0 and NUM_COST_METRICS),
// indicating a forward route. Otherwise returns None.
pub fn get_one_way_reachable_idx(
    global: &GlobalState,
    vertex_id: usize,
    forward: &TraverseResult,
) -> Option<usize> {
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
    difficulty: &DifficultyConfig,
    game_data: &GameData,
    locked_door_data: &LockedDoorData,
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
        // let mut cnt = 0;
        // let mut total_cost = 0.0;
        // for v in 0..result.cost.len() {
        //     for k in 0..NUM_COST_METRICS {
        //         if f32::is_finite(result.cost[v][k]) {
        //             cnt += 1;
        //             total_cost += result.cost[v][k];
        //         }
        //     }
        // }
        // println!("modified vertices: {}, cnt_finite: {}, cost={}", modified_vertices.len(), cnt, total_cost);

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
                    if let Some(mut dst_new_local_state) = apply_link(
                        &link,
                        global,
                        src_local_state,
                        reverse,
                        difficulty,
                        game_data,
                        locked_door_data,
                    ) {
                        let dst_new_cost_arr = compute_cost(dst_new_local_state, global);

                        // for k in dst_new_cost_arr {
                        //     if k < 0.0 {
                        //         println!("{:?}", link);
                        //         println!("{:?} {:?}", game_data.vertex_isv.keys[link.from_vertex_id], game_data.vertex_isv.keys[link.to_vertex_id]);
                        //         panic!("negative cost");
                        //     }
                        // }

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
                            check_value("shinecharge_frames", dst_new_local_state.shinecharge_frames_remaining);
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
