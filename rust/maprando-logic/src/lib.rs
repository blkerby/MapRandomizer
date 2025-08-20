// TODO: consider removing this later. It's not a bad lint but I don't want to deal with it now.
#![allow(clippy::too_many_arguments)]

pub mod boss_requirements;
pub mod helpers;

use maprando_game::{Capacity, GameData, Item, WeaponMask};
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
                starting_local_state.energy_used += 100;
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
pub struct LocalState {
    pub energy_used: Capacity,
    pub reserves_used: Capacity,
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

impl LocalState {
    pub fn empty(global: &GlobalState) -> Self {
        LocalState {
            energy_used: global.inventory.max_energy - 1,
            reserves_used: global.inventory.max_reserves,
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
        }
    }

    pub fn full() -> Self {
        LocalState {
            energy_used: 0,
            reserves_used: 0,
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
        }
    }

    pub fn is_impossible(&self) -> bool {
        self.energy_used == IMPOSSIBLE_LOCAL_STATE.energy_used
    }
}
