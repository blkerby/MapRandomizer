// TODO: consider removing this later. It's not a bad lint but I don't want to deal with it now.
#![allow(clippy::too_many_arguments)]

pub mod boss_requirements;
pub mod helpers;

use maprando_game::{Capacity, GameData, Item, StepTrailId, WeaponMask};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
pub struct Inventory {
    pub items: Vec<bool>,
    pub max_energy: Capacity,
    pub max_reserves: Capacity,
    pub max_missiles: Capacity,
    pub max_supers: Capacity,
    pub max_power_bombs: Capacity,
    pub collectible_missile_packs: Capacity,
    pub collectible_super_packs: Capacity,
    pub collectible_power_bomb_packs: Capacity,
}

// TODO: move tech and notable_strats out of this struct, since these do not change from step to step.
#[derive(Clone, Debug)]
pub struct GlobalState {
    pub inventory: Inventory,
    pub flags: Vec<bool>,
    pub doors_unlocked: Vec<bool>,
    pub weapon_mask: WeaponMask,
}

impl GlobalState {
    pub fn print_debug(&self, game_data: &GameData) {
        for (i, item) in game_data.item_isv.keys.iter().enumerate() {
            if self.inventory.items[i] {
                println!("{item:?}");
            }
        }
    }

    pub fn collect(
        &mut self,
        item: Item,
        game_data: &GameData,
        ammo_collect_fraction: f32,
        tech: &[bool],
        starting_local_state: &mut LocalState,
    ) {
        self.inventory.items[item as usize] = true;
        match item {
            Item::Missile => {
                self.inventory.collectible_missile_packs += 1;
                let new_max_missiles = (ammo_collect_fraction
                    * self.inventory.collectible_missile_packs as f32)
                    .round() as Capacity
                    * 5;
                starting_local_state.missiles_used +=
                    new_max_missiles - self.inventory.max_missiles;
                self.inventory.max_missiles = new_max_missiles;
            }
            Item::Super => {
                self.inventory.collectible_super_packs += 1;
                let new_max_supers = (ammo_collect_fraction
                    * self.inventory.collectible_super_packs as f32)
                    .round() as Capacity
                    * 5;
                starting_local_state.supers_used += new_max_supers - self.inventory.max_supers;
                self.inventory.max_supers = new_max_supers;
            }
            Item::PowerBomb => {
                self.inventory.collectible_power_bomb_packs += 1;
                let new_max_power_bombs = (ammo_collect_fraction
                    * self.inventory.collectible_power_bomb_packs as f32)
                    .round() as Capacity
                    * 5;
                starting_local_state.power_bombs_used +=
                    new_max_power_bombs - self.inventory.max_power_bombs;
                self.inventory.max_power_bombs = new_max_power_bombs;
            }
            Item::ETank => {
                self.inventory.max_energy += 100;
                starting_local_state.energy += 100;
            }
            Item::ReserveTank => {
                self.inventory.max_reserves += 100;
                starting_local_state.reserves_used += 100;
            }
            _ => {}
        }
        self.weapon_mask = game_data.get_weapon_mask(&self.inventory.items, tech);
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ResourceLevel {
    Consumed(Capacity),
    Remaining(Capacity),
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LocalState {
    // Positive values are relative to empty, zero and negative values are relative to full:
    pub energy: ResourceLevel,
    pub reserves: ResourceLevel,
    pub missiles_used: Capacity,
    pub supers_used: Capacity,
    pub power_bombs_used: Capacity,
    pub shinecharge_frames_remaining: Capacity,
    pub cycle_frames: Capacity,
    pub farm_baseline_energy_used: Capacity,
    pub farm_baseline_reserves_used: Capacity,
    pub farm_baseline_missiles_used: Capacity,
    pub farm_baseline_supers_used: Capacity,
    pub farm_baseline_power_bombs_used: Capacity,
    pub flash_suit: bool,
    pub prev_trail_id: StepTrailId,
}

impl LocalState {
    pub fn empty(global: &GlobalState) -> Self {
        LocalState {
            energy: ResourceLevel::Remaining(1),
            reserves: ResourceLevel::Remaining(0),
            missiles_used: global.inventory.max_missiles,
            supers_used: global.inventory.max_supers,
            power_bombs_used: global.inventory.max_power_bombs,
            shinecharge_frames_remaining: 0,
            cycle_frames: 0,
            farm_baseline_energy_used: 0,
            farm_baseline_reserves_used: 0,
            farm_baseline_missiles_used: global.inventory.max_missiles,
            farm_baseline_supers_used: global.inventory.max_supers,
            farm_baseline_power_bombs_used: global.inventory.max_power_bombs,
            flash_suit: false,
            prev_trail_id: -1,
        }
    }

    pub fn full(reverse: bool) -> Self {
        LocalState {
            energy: if reverse {
                ResourceLevel::Remaining(1)
            } else {
                ResourceLevel::Consumed(0)
            },
            reserves: if reverse {
                ResourceLevel::Remaining(0)
            } else {
                ResourceLevel::Consumed(0)
            },
            missiles_used: 0,
            supers_used: 0,
            power_bombs_used: 0,
            shinecharge_frames_remaining: 0,
            cycle_frames: 0,
            farm_baseline_energy_used: 0,
            farm_baseline_reserves_used: 0,
            farm_baseline_missiles_used: 0,
            farm_baseline_supers_used: 0,
            farm_baseline_power_bombs_used: 0,
            flash_suit: false,
            prev_trail_id: -1,
        }
    }

    fn energy_remaining(&self, inventory: &Inventory) -> Capacity {
        match self.energy {
            ResourceLevel::Consumed(x) => inventory.max_energy - x,
            ResourceLevel::Remaining(x) => x,
        }
    }

    fn reserves_remaining(&self, inventory: &Inventory) -> Capacity {
        match self.reserves {
            ResourceLevel::Consumed(x) => inventory.max_reserves - x,
            ResourceLevel::Remaining(x) => x,
        }
    }

    pub fn auto_reserve_trigger(&mut self, inventory: &Inventory, reverse: bool) -> bool {
        if reverse {
            let energy_remaining = self.energy_remaining(inventory);
        } else {
            let reserves_remaining = self.reserves_remaining(inventory);
            if reserves_remaining == 0 {
                false
            } else {
                self.energy = ResourceLevel::Remaining(Capacity::min(
                    reserves_remaining,
                    inventory.max_energy,
                ));
                true
            }
        }
    }

    pub fn use_energy(
        &mut self,
        amt: Capacity,
        can_auto_reserve: bool,
        can_manage_reserves: bool,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        match (reverse, self.energy) {
            (false, ResourceLevel::Consumed(x)) => {
                if x + amt >= inventory.max_energy {
                    if can_manage_reserves {
                        // Assume that just enough reserve energy is manually converted to regular energy.
                        self.energy = ResourceLevel::Consumed(inventory.max_energy - 1);
                        self.use_reserve_energy(
                            x + amt - (inventory.max_energy - 1),
                            inventory,
                            reverse,
                        )
                    } else if can_auto_reserve {
                        // let new_energy = x + amt - local.res
                        self.energy = ResourceLevel::Consumed(x + amt);
                        self.reserves = ResourceLevel::Remaining(0);
                        false
                    } else {
                        false
                    }
                } else {
                    self.energy = ResourceLevel::Consumed(x + amt);
                    true
                }
            }
            (false, ResourceLevel::Remaining(x)) => {
                if x <= amt {
                    if can_manage_reserves {
                        // Assume that just enough reserve energy is manually converted to regular energy.
                        self.energy = ResourceLevel::Remaining(1);
                        self.use_reserve_energy(amt - x + 1, inventory, reverse)
                    } else {
                        false
                    }
                } else {
                    self.energy = ResourceLevel::Remaining(x - amt);
                    true
                }
            }
            (true, ResourceLevel::Consumed(x)) => {
                if x <= amt {
                    if can_manage_reserves {
                        // Assume that just enough reserve energy is manually converted to regular energy.
                        self.energy = ResourceLevel::Consumed(0);
                        self.use_reserve_energy(amt - x, inventory, reverse)
                    } else {
                        false
                    }
                } else {
                    self.energy = ResourceLevel::Consumed(x - amt);
                    true
                }
            }
            (true, ResourceLevel::Remaining(x)) => {
                if x + amt >= inventory.max_energy {
                    if can_manage_reserves {
                        // Assume that just enough reserve energy is manually converted to regular energy.
                        self.energy = ResourceLevel::Remaining(inventory.max_energy);
                        self.use_reserve_energy(x + amt - inventory.max_energy, inventory, reverse)
                    } else {
                        false
                    }
                } else {
                    self.energy = ResourceLevel::Remaining(x + amt);
                    true
                }
            }
        }
    }

    pub fn use_reserve_energy(
        &mut self,
        amt: Capacity,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        match (reverse, self.reserves) {
            (false, ResourceLevel::Consumed(x)) => {
                if x + amt > inventory.max_reserves {
                    false
                } else {
                    self.reserves = ResourceLevel::Consumed(x + amt);
                    true
                }
            }
            (false, ResourceLevel::Remaining(x)) => {
                if amt > x {
                    false
                } else {
                    self.reserves = ResourceLevel::Remaining(x - amt);
                    true
                }
            }
            (true, ResourceLevel::Consumed(x)) => {
                if amt > x {
                    false
                } else {
                    self.reserves = ResourceLevel::Consumed(x - amt);
                    true
                }
            }
            (true, ResourceLevel::Remaining(x)) => {
                if x + amt > inventory.max_reserves {
                    false
                } else {
                    self.reserves = ResourceLevel::Remaining(x + amt);
                    true
                }
            }
        }
    }
}
