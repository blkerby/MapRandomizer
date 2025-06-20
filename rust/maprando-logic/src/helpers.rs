use crate::{Inventory, LocalState};
use maprando_game::{Capacity, Item};

pub fn get_charge_damage(inventory: &Inventory) -> f32 {
    if !inventory.items[Item::Charge as usize] {
        return 0.0;
    }
    let plasma = inventory.items[Item::Plasma as usize];
    let spazer = inventory.items[Item::Spazer as usize];
    let wave = inventory.items[Item::Wave as usize];
    let ice = inventory.items[Item::Ice as usize];
    3.0 * match (plasma, spazer, wave, ice) {
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
    }
}

pub fn suit_damage_factor(inventory: &Inventory) -> Capacity {
    let varia = inventory.items[Item::Varia as usize];
    let gravity = inventory.items[Item::Gravity as usize];
    if gravity && varia {
        4
    } else if gravity || varia {
        2
    } else {
        1
    }
}

pub fn validate_energy(
    mut local: LocalState,
    inventory: &Inventory,
    can_manage_reserves: bool,
) -> Option<LocalState> {
    if local.energy_used >= inventory.max_energy {
        if can_manage_reserves {
            // Assume that just enough reserve energy is manually converted to regular energy.
            local.reserves_used += local.energy_used - (inventory.max_energy - 1);
            local.energy_used = inventory.max_energy - 1;
        } else {
            // Assume that reserves auto-trigger, leaving reserves empty.
            let reserves_available = std::cmp::min(
                inventory.max_reserves - local.reserves_used,
                inventory.max_energy,
            );
            local.reserves_used = inventory.max_reserves;
            local.energy_used = std::cmp::max(0, local.energy_used - reserves_available);
            if local.energy_used >= inventory.max_energy {
                return None;
            }
        }
    }
    if local.reserves_used > inventory.max_reserves {
        return None;
    }
    Some(local)
}
