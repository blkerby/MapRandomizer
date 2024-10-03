use std::cmp::{max, min};

use crate::helpers::*;
use crate::{Inventory, LocalState};
use maprando_game::{Capacity, Item};

pub fn apply_phantoon_requirement(
    inventory: &Inventory,
    mut local: LocalState,
    proficiency: f32,
    can_manage_reserves: bool,
) -> Option<LocalState> {
    // We only consider simple, safer strats here, where we try to damage Phantoon as much as possible
    // as soon as he opens his eye. Faster or more complex strats are not relevant, since at
    // high proficiency the fight is considered free anyway (as long as Charge or any ammo is available)
    // since all damage can be avoided.
    let boss_hp: f32 = 2500.0;
    let charge_damage = get_charge_damage(&inventory);

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
    if inventory.max_missiles > 0 {
        // We don't worry about ammo quantity since they can be farmed from the flames.
        let missiles_to_use = f32::ceil(boss_hp / 100.0);
        // Assume max average rate of 3 missiles per 10 seconds:
        let time = missiles_to_use as f32 * 10.0 / 3.0 / firing_rate;
        possible_kill_times.push(time);
    }
    if inventory.max_supers > 0 {
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
    let mut net_dps = base_hit_dps / suit_damage_factor(inventory) as f32 - farm_rate;
    if net_dps < 0.0 {
        // We could assume we could refill on energy or ammo using farms, but by omitting this for now
        // we're just making the logic a little more conservative in favor of the player.
        net_dps = 0.0;
    }

    // Overflow safeguard - bail here if Samus takes calamitous damage.
    if net_dps * kill_time > 10000.0 {
        return None;
    }

    local.energy_used += (net_dps * kill_time) as Capacity;

    validate_energy(local, inventory, can_manage_reserves)
}

pub fn apply_draygon_requirement(
    inventory: &Inventory,
    local: LocalState,
    proficiency: f32,
    can_manage_reserves: bool,
    can_be_very_patient: bool,
) -> Option<LocalState> {
    let mut boss_hp: f32 = 6000.0;
    let charge_damage = get_charge_damage(&inventory);

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
        inventory.items[Item::Plasma as usize],
        inventory.items[Item::Wave as usize],
    ) {
        (false, _) => 7.0,     // Basic beam
        (true, false) => 10.0, // Plasma can hit multiple goops at once.
        (true, true) => 13.0,  // Wave+Plasma can hit even more goops at once.
    };
    let goop_farms_per_cycle = if inventory.items[Item::Gravity as usize] {
        farm_proficiency * base_goop_farms_per_cycle
    } else {
        // Without Gravity you can't farm as many goops since you have to spend more time avoiding Draygon.
        0.75 * farm_proficiency * base_goop_farms_per_cycle
    };
    let energy_farm_rate =
        GOOP_CYCLES_PER_SECOND * goop_farms_per_cycle * (5.0 * 0.02 + 20.0 * 0.12);
    let missile_farm_rate = GOOP_CYCLES_PER_SECOND * goop_farms_per_cycle * (2.0 * 0.44);

    let base_hit_dps = if inventory.items[Item::Gravity as usize] {
        // With Gravity, assume one Draygon hit per two cycles as the maximum rate of damage to Samus:
        160.0 * 0.5 * (GOOP_CYCLES_PER_SECOND + SWOOP_CYCLES_PER_SECOND) * (1.0 - proficiency)
    } else {
        // Without Gravity, assume one Draygon hit per cycle as the maximum rate of damage to Samus:
        160.0 * (GOOP_CYCLES_PER_SECOND + SWOOP_CYCLES_PER_SECOND) * (1.0 - proficiency)
    };

    // We assume as many Supers are available can be used immediately (e.g. on the first goop cycle):
    let supers_available = inventory.max_supers - local.supers_used;
    boss_hp -= (supers_available as f32) * accuracy * 300.0;
    if boss_hp < 0.0 {
        return Some(local);
    }

    let missiles_available = inventory.max_missiles - local.missiles_used;
    let missile_firing_rate = 20.0 * GOOP_CYCLES_PER_SECOND * firing_rate;
    let net_missile_use_rate = missile_firing_rate - missile_farm_rate;

    let initial_missile_damage_rate = 100.0 * missile_firing_rate * accuracy;
    let overall_damage_rate = initial_missile_damage_rate + charge_damage_rate;
    let time_boss_dead = f32::ceil(boss_hp / overall_damage_rate);
    let time_missiles_exhausted = if inventory.max_missiles == 0 {
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

        let farming_missile_damage_rate = if inventory.max_missiles > 0 {
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

    if time < 180.0 || can_be_very_patient {
        let mut net_dps = base_hit_dps / suit_damage_factor(inventory) as f32 - energy_farm_rate;
        if net_dps < 0.0 {
            net_dps = 0.0;
        }
        // Overflow safeguard - bail here if Samus takes calamitous damage.
        if net_dps * time > 10000.0 {
            return None;
        }
        // We don't account for resources used, since they can be farmed or picked up after the fight, and we don't
        // want the fight to go out of logic due to not saving enough Missiles to open some red doors for example.
        let result = LocalState {
            energy_used: local.energy_used + (net_dps * time) as Capacity,
            ..local
        };
        if validate_energy(result, inventory, can_manage_reserves).is_some() {
            return Some(local);
        } else {
            return None;
        }
    } else {
        return None;
    }
}

pub fn apply_ridley_requirement(
    inventory: &Inventory,
    mut local: LocalState,
    proficiency: f32,
    can_manage_reserves: bool,
    can_be_very_patient: bool,
) -> Option<LocalState> {
    let mut boss_hp: f32 = 18000.0;
    let mut time: f32 = 0.0; // Cumulative time in seconds for the fight
    let charge_damage = get_charge_damage(&inventory);

    // Assume an ammo accuracy rate of between 60% (on lowest difficulty) to 100% (on highest):
    let accuracy = 0.6 + 0.4 * proficiency;

    // Assume a firing rate of between 30% (on lowest difficulty) to 100% (on highest):
    let firing_rate = 0.3 + 0.7 * proficiency;

    let super_time = 0.5 / firing_rate; // minimum of 0.5 seconds between Super shots
    let charge_time = 1.4 / firing_rate; // minimum of 1.4 seconds between charge shots
    let missile_time = 0.34 / firing_rate; // minimum of 0.34 seconds between Missile shots
    let power_bomb_time = 3.0 / firing_rate; // minimum of 3.0 seconds between Power Bomb shots

    let charge_dps = charge_damage * accuracy / charge_time;
    let missiles_dps = 100.0 * accuracy / missile_time;
    let power_bomb_dps = 400.0 * accuracy / power_bomb_time;

    // Prioritize using supers:
    let supers_available = inventory.max_supers - local.supers_used;
    let supers_to_use = min(
        supers_available,
        f32::ceil(boss_hp / (600.0 * accuracy)) as Capacity,
    );
    local.supers_used += supers_to_use;
    boss_hp -= supers_to_use as f32 * 600.0 * accuracy;
    time += supers_to_use as f32 * super_time;

    // Use Charge if it's higher DPS than Missiles (which happens with Charge + Plasma).
    // For less than full beam combo, a player could squeeze out more DPS by using
    // Missiles during pogo and Charge during swoops, but we don't try to model this.
    if charge_dps >= missiles_dps {
        let charge_shots_to_use = max(
            0,
            f32::ceil(boss_hp / (charge_damage * accuracy)) as Capacity,
        );
        boss_hp = 0.0;
        time += charge_shots_to_use as f32 * charge_time;
    }

    // Then use available missiles:
    let missiles_available = inventory.max_missiles - local.missiles_used;
    let missiles_to_use = max(
        0,
        min(
            missiles_available,
            f32::ceil(boss_hp / (100.0 * accuracy)) as Capacity,
        ),
    );
    local.missiles_used += missiles_to_use;
    boss_hp -= missiles_to_use as f32 * 100.0 * accuracy;
    time += missiles_to_use as f32 * missile_time;

    // Use Charge if it's more powerful than Power Bomb:
    if charge_dps >= power_bomb_dps {
        let charge_shots_to_use = max(
            0,
            f32::ceil(boss_hp / (charge_damage * accuracy)) as Capacity,
        );
        boss_hp = 0.0;
        time += charge_shots_to_use as f32 * charge_time;
    }

    if inventory.items[Item::Morph as usize] {
        // Use Power Bombs:
        let pbs_available = inventory.max_power_bombs - local.power_bombs_used;
        let pbs_to_use = max(
            0,
            min(
                pbs_available,
                f32::ceil(boss_hp / (400.0 * accuracy)) as Capacity,
            ),
        );
        local.power_bombs_used += pbs_to_use;
        boss_hp -= pbs_to_use as f32 * 400.0 * accuracy;
        time += pbs_to_use as f32 * power_bomb_time;
    }

    // Use Charge, if available:
    if charge_damage > 0.0 {
        let charge_shots_to_use = max(
            0,
            f32::ceil(boss_hp / (charge_damage * accuracy)) as Capacity,
        );
        boss_hp = 0.0;
        time += charge_shots_to_use as f32 * charge_time;
    }

    if boss_hp > 0.0 {
        // We don't have enough ammo to finish the fight:
        return None;
    }

    if time >= 180.0 && !can_be_very_patient {
        // We don't have enough patience to finish the fight:
        return None;
    }

    let morph = inventory.items[Item::Morph as usize];
    let screw = inventory.items[Item::ScrewAttack as usize];

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
    // Overflow safeguard - bail here if Samus takes calamitous damage.
    if damage > 10000.0 {
        return None;
    }
    local.energy_used += (damage / suit_damage_factor(inventory) as f32) as Capacity;

    if !inventory.items[Item::Varia as usize] {
        // Heat run case: We do not explicitly check canHeatRun tech here, because it is
        // already required to reach the boss node from the doors.
        // Include time pre- and post-fight when Samus must still take heat damage:
        let heat_time = time + 16.0;
        let heat_energy_used = (heat_time * 15.0) as Capacity;
        local.energy_used += heat_energy_used;
    }

    // TODO: We could add back some energy and/or ammo by assuming we get drops.
    // By omitting this for now we're just making the logic a little more conservative in favor of
    // the player.
    validate_energy(local, inventory, can_manage_reserves)
}

pub fn apply_botwoon_requirement(
    inventory: &Inventory,
    mut local: LocalState,
    proficiency: f32,
    second_phase: bool,
    can_manage_reserves: bool,
) -> Option<LocalState> {
    // We aim to be a little lenient here. For example, we don't take SBAs (e.g. X-factors) into account,
    // assuming instead the player just uses ammo and/or regular charged shots.

    let mut boss_hp: f32 = 1500.0; // HP for one phase of the fight.
    let mut time: f32 = 0.0; // Cumulative time in seconds for the phase
    let charge_damage = get_charge_damage(&inventory);

    // Assume an ammo accuracy rate of between 25% (on lowest difficulty) to 90% (on highest):
    let accuracy = 0.25 + 0.65 * proficiency;

    // Assume a firing rate of between 30% (on lowest difficulty) to 100% (on highest),
    let firing_rate = 0.3 + 0.7 * proficiency;

    // The firing rates below are for the first phase (since the rate doesn't matter for
    // the second phase):
    let use_supers = |local: &mut LocalState, boss_hp: &mut f32, time: &mut f32| {
        let supers_available = inventory.max_supers - local.supers_used;
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
        let missiles_available = inventory.max_missiles - local.missiles_used;
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
        if inventory.items[Item::Charge as usize] {
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
        let morph = inventory.items[Item::Morph as usize];
        let gravity = inventory.items[Item::Gravity as usize];

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
        let damage_per_hit = 96.0 / suit_damage_factor(inventory) as f32;
        // Overflow safeguard - bail here if Samus takes calamitous damage.
        if hits * damage_per_hit > 10000.0 {
            return None;
        }
        local.energy_used += (hits * damage_per_hit) as Capacity;
    }

    // TODO: We could add back some energy and/or ammo by assuming we get drops.
    // By omitting this for now we're just making the logic a little more conservative in favor of
    // the player.
    validate_energy(local, inventory, can_manage_reserves)
}

pub fn apply_mother_brain_2_requirement(
    inventory: &Inventory,
    mut local: LocalState,
    proficiency: f32,
    supers_double: bool,
    can_manage_reserves: bool,
    can_be_very_patient: bool,
    r_mode: bool,
) -> Option<LocalState> {
    if inventory.max_energy < 199 && !r_mode {
        // Need at least one ETank to survive rainbow beam, except in R-mode (where energy requirements are handled elsewhere
        // in the strat logic)
        return None;
    }

    let mut boss_hp: f32 = 18000.0;
    let mut time: f32 = 0.0; // Cumulative time in seconds for the fight
    let charge_damage = get_charge_damage(&inventory);

    // Assume an ammo accuracy rate of between 75% (on lowest difficulty) to 100% (on highest):
    let accuracy = 0.75 + 0.25 * proficiency;

    // Assume a firing rate of between 60% (on lowest difficulty) to 100% (on highest):
    let firing_rate = 0.6 + 0.4 * proficiency;

    let charge_time = 1.1; // minimum of 1.1 seconds between charge shots
    let super_time = 0.35; // minimum of 0.35 seconds between Super shots
    let missile_time = 0.17;

    // Prioritize using supers:
    let supers_available = inventory.max_supers - local.supers_used;
    let super_damage = if supers_double { 600.0 } else { 300.0 };
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
    let missiles_available = inventory.max_missiles - local.missiles_used;
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

    if inventory.items[Item::Charge as usize] {
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

    if time >= 180.0 && !can_be_very_patient {
        // We don't have enough patience to finish the fight:
        return None;
    }

    // Assumed rate of Mother Brain damage to Samus (per second), given minimal dodging skill:
    // For simplicity we assume a uniform rate of damage (even though the ketchup phase has the highest risk of damage).
    let base_mb_attack_dps = 20.0;
    let hit_rate = 1.0 - proficiency;
    let damage = base_mb_attack_dps * hit_rate * time;
    if !r_mode {
        local.energy_used += (damage / suit_damage_factor(inventory) as f32) as Capacity;

        // Account for Rainbow beam damage:
        if inventory.items[Item::Varia as usize] {
            local.energy_used += 300;
        } else {
            local.energy_used += 600;
        }
    }
    // Overflow safeguard - bail here if Samus takes calamitous damage.
    if damage > 10000.0 {
        return None;
    }

    validate_energy(local, inventory, can_manage_reserves)
}
