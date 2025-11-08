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
    pub collectible_reserve_tanks: Capacity,
}

// TODO: move tech and notable_strats out of this struct, since these do not change from step to step.
#[derive(Clone, Debug)]
pub struct GlobalState {
    pub inventory: Inventory,
    pub pool_inventory: Inventory,
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
    ) {
        self.inventory.items[item as usize] = true;
        match item {
            Item::Missile => {
                self.inventory.collectible_missile_packs += 1;
                let new_max_missiles = (ammo_collect_fraction
                    * self.inventory.collectible_missile_packs as f32)
                    .round() as Capacity
                    * 5;
                self.inventory.max_missiles = new_max_missiles;
            }
            Item::Super => {
                self.inventory.collectible_super_packs += 1;
                let new_max_supers = (ammo_collect_fraction
                    * self.inventory.collectible_super_packs as f32)
                    .round() as Capacity
                    * 5;
                self.inventory.max_supers = new_max_supers;
            }
            Item::PowerBomb => {
                self.inventory.collectible_power_bomb_packs += 1;
                let new_max_power_bombs = (ammo_collect_fraction
                    * self.inventory.collectible_power_bomb_packs as f32)
                    .round() as Capacity
                    * 5;
                self.inventory.max_power_bombs = new_max_power_bombs;
            }
            Item::ETank => {
                self.inventory.max_energy += 100;
            }
            Item::ReserveTank => {
                self.inventory.collectible_reserve_tanks += 1;
                self.inventory.max_reserves = self.inventory.collectible_reserve_tanks * 100;
                if !tech[game_data.manage_reserves_tech_idx] {
                    self.inventory.max_reserves =
                        Capacity::min(self.inventory.max_reserves, self.inventory.max_energy);
                }
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

impl ResourceLevel {
    pub fn full(reverse: bool) -> Self {
        if reverse {
            ResourceLevel::Remaining(0)
        } else {
            ResourceLevel::Consumed(0)
        }
    }

    pub fn full_energy(reverse: bool) -> Self {
        if reverse {
            ResourceLevel::Remaining(1)
        } else {
            ResourceLevel::Consumed(0)
        }
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EncodedResourceLevel(pub Capacity);

impl EncodedResourceLevel {
    fn new(r: ResourceLevel) -> Self {
        match r {
            ResourceLevel::Consumed(x) => EncodedResourceLevel(-x - 1),
            ResourceLevel::Remaining(x) => EncodedResourceLevel(x),
        }
    }

    fn decode(&self) -> ResourceLevel {
        if self.0 >= 0 {
            ResourceLevel::Remaining(self.0)
        } else {
            ResourceLevel::Consumed(-self.0 - 1)
        }
    }
}

impl From<ResourceLevel> for EncodedResourceLevel {
    fn from(value: ResourceLevel) -> Self {
        EncodedResourceLevel::new(value)
    }
}

// Using ResourceLevel enums is a bit wasteful here since it costs a byte for each discriminant:
// we could instead use the sign bit of the Capacity to distinguish Remaining vs. Consumed variants.
// That would be more error-prone, so we postpone it as a potential future optimization.
// To support a potential change in representation, we're moving in a direction of making LocalState
// encapsulated, adding methods for interacting with it, so that its member fields eventually
// won't have to be public.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LocalState {
    pub energy: EncodedResourceLevel,
    pub reserves: EncodedResourceLevel,
    pub missiles: EncodedResourceLevel,
    pub supers: EncodedResourceLevel,
    pub power_bombs: EncodedResourceLevel,
    pub shinecharge_frames_remaining: Capacity,
    pub cycle_frames: Capacity,
    pub farm_baseline_energy: EncodedResourceLevel,
    pub farm_baseline_reserves: EncodedResourceLevel,
    pub farm_baseline_missiles: EncodedResourceLevel,
    pub farm_baseline_supers: EncodedResourceLevel,
    pub farm_baseline_power_bombs: EncodedResourceLevel,
    pub flash_suit: bool,
    pub prev_trail_id: StepTrailId,
}

impl LocalState {
    pub fn empty() -> Self {
        LocalState {
            energy: ResourceLevel::Remaining(1).into(),
            reserves: ResourceLevel::Remaining(0).into(),
            missiles: ResourceLevel::Remaining(0).into(),
            supers: ResourceLevel::Remaining(0).into(),
            power_bombs: ResourceLevel::Remaining(0).into(),
            shinecharge_frames_remaining: 0,
            cycle_frames: 0,
            farm_baseline_energy: ResourceLevel::Remaining(1).into(),
            farm_baseline_reserves: ResourceLevel::Remaining(0).into(),
            farm_baseline_missiles: ResourceLevel::Remaining(0).into(),
            farm_baseline_supers: ResourceLevel::Remaining(0).into(),
            farm_baseline_power_bombs: ResourceLevel::Remaining(0).into(),
            flash_suit: false,
            prev_trail_id: -1,
        }
    }

    pub fn full(reverse: bool) -> Self {
        let generic_resource_level = if reverse {
            ResourceLevel::Remaining(0).into()
        } else {
            ResourceLevel::Consumed(0).into()
        };
        LocalState {
            energy: if reverse {
                ResourceLevel::Remaining(1).into()
            } else {
                ResourceLevel::Consumed(0).into()
            },
            reserves: generic_resource_level,
            missiles: generic_resource_level,
            supers: generic_resource_level,
            power_bombs: generic_resource_level,
            shinecharge_frames_remaining: 0,
            cycle_frames: 0,
            farm_baseline_energy: if reverse {
                ResourceLevel::Remaining(1).into()
            } else {
                ResourceLevel::Consumed(0).into()
            },
            farm_baseline_reserves: generic_resource_level,
            farm_baseline_missiles: generic_resource_level,
            farm_baseline_supers: generic_resource_level,
            farm_baseline_power_bombs: generic_resource_level,
            flash_suit: false,
            prev_trail_id: -1,
        }
    }

    pub fn energy(&self) -> ResourceLevel {
        self.energy.decode()
    }

    pub fn reserves(&self) -> ResourceLevel {
        self.reserves.decode()
    }

    pub fn missiles(&self) -> ResourceLevel {
        self.missiles.decode()
    }

    pub fn supers(&self) -> ResourceLevel {
        self.supers.decode()
    }

    pub fn power_bombs(&self) -> ResourceLevel {
        self.power_bombs.decode()
    }

    pub fn farm_baseline_energy(&self) -> ResourceLevel {
        self.farm_baseline_energy.decode()
    }

    pub fn farm_baseline_reserves(&self) -> ResourceLevel {
        self.farm_baseline_reserves.decode()
    }

    pub fn farm_baseline_missiles(&self) -> ResourceLevel {
        self.farm_baseline_missiles.decode()
    }

    pub fn farm_baseline_supers(&self) -> ResourceLevel {
        self.farm_baseline_supers.decode()
    }

    pub fn farm_baseline_power_bombs(&self) -> ResourceLevel {
        self.farm_baseline_power_bombs.decode()
    }

    pub fn energy_remaining(&self, inventory: &Inventory, include_reserves: bool) -> Capacity {
        let energy = match self.energy() {
            ResourceLevel::Consumed(x) => inventory.max_energy - x,
            ResourceLevel::Remaining(x) => x,
        };
        if include_reserves {
            energy + self.reserves_remaining(inventory)
        } else {
            energy
        }
    }

    pub fn energy_missing(&self, inventory: &Inventory, include_reserves: bool) -> Capacity {
        let remaining = self.energy_remaining(inventory, include_reserves);
        if include_reserves {
            inventory.max_energy + inventory.max_reserves - remaining
        } else {
            inventory.max_energy - remaining
        }
    }

    pub fn energy_available(
        &self,
        inventory: &Inventory,
        include_reserves: bool,
        reverse: bool,
    ) -> Capacity {
        if reverse {
            self.energy_missing(inventory, include_reserves)
        } else {
            self.energy_remaining(inventory, include_reserves)
        }
    }

    pub fn resource_missing(level: ResourceLevel, max_resource: Capacity) -> Capacity {
        match level {
            ResourceLevel::Consumed(x) => x,
            ResourceLevel::Remaining(x) => max_resource - x,
        }
    }

    pub fn resource_remaining(level: ResourceLevel, max_resource: Capacity) -> Capacity {
        match level {
            ResourceLevel::Consumed(x) => max_resource - x,
            ResourceLevel::Remaining(x) => x,
        }
    }

    pub fn resource_available(
        level: ResourceLevel,
        max_resource: Capacity,
        reverse: bool,
    ) -> Capacity {
        if reverse {
            Self::resource_missing(level, max_resource)
        } else {
            Self::resource_remaining(level, max_resource)
        }
    }

    pub fn reserves_remaining(&self, inventory: &Inventory) -> Capacity {
        Self::resource_remaining(self.reserves(), inventory.max_reserves)
    }

    pub fn reserves_missing(&self, inventory: &Inventory) -> Capacity {
        Self::resource_missing(self.reserves(), inventory.max_reserves)
    }

    pub fn reserves_available(&self, inventory: &Inventory, reverse: bool) -> Capacity {
        Self::resource_available(self.reserves(), inventory.max_reserves, reverse)
    }

    pub fn missiles_remaining(&self, inventory: &Inventory) -> Capacity {
        Self::resource_remaining(self.missiles(), inventory.max_missiles)
    }

    pub fn missiles_missing(&self, inventory: &Inventory) -> Capacity {
        Self::resource_missing(self.missiles(), inventory.max_missiles)
    }

    pub fn missiles_available(&self, inventory: &Inventory, reverse: bool) -> Capacity {
        Self::resource_available(self.missiles(), inventory.max_missiles, reverse)
    }

    pub fn supers_remaining(&self, inventory: &Inventory) -> Capacity {
        Self::resource_remaining(self.supers(), inventory.max_supers)
    }

    pub fn supers_missing(&self, inventory: &Inventory) -> Capacity {
        Self::resource_missing(self.supers(), inventory.max_supers)
    }

    pub fn supers_available(&self, inventory: &Inventory, reverse: bool) -> Capacity {
        Self::resource_available(self.supers(), inventory.max_supers, reverse)
    }

    pub fn power_bombs_remaining(&self, inventory: &Inventory) -> Capacity {
        Self::resource_remaining(self.power_bombs(), inventory.max_power_bombs)
    }

    pub fn power_bombs_missing(&self, inventory: &Inventory) -> Capacity {
        Self::resource_missing(self.power_bombs(), inventory.max_power_bombs)
    }

    pub fn power_bombs_available(&self, inventory: &Inventory, reverse: bool) -> Capacity {
        Self::resource_available(self.power_bombs(), inventory.max_power_bombs, reverse)
    }

    pub fn shinecharge_frames_available(&self, reverse: bool) -> Capacity {
        Self::resource_available(
            ResourceLevel::Remaining(self.shinecharge_frames_remaining),
            180,
            reverse,
        )
    }

    pub fn flash_suit_available(&self, reverse: bool) -> Capacity {
        let amt = if self.flash_suit { 1 } else { 0 };
        if reverse { 1 - amt } else { amt }
    }

    pub fn farm_baseline_energy_remaining(&self, inventory: &Inventory) -> Capacity {
        Self::resource_remaining(self.farm_baseline_energy(), inventory.max_energy)
    }

    pub fn farm_baseline_energy_available(&self, inventory: &Inventory, reverse: bool) -> Capacity {
        Self::resource_available(self.farm_baseline_energy(), inventory.max_energy, reverse)
    }

    pub fn farm_baseline_reserves_remaining(&self, inventory: &Inventory) -> Capacity {
        Self::resource_remaining(self.farm_baseline_reserves(), inventory.max_reserves)
    }

    pub fn farm_baseline_reserves_available(
        &self,
        inventory: &Inventory,
        reverse: bool,
    ) -> Capacity {
        Self::resource_available(
            self.farm_baseline_reserves(),
            inventory.max_reserves,
            reverse,
        )
    }

    pub fn farm_baseline_missiles_remaining(&self, inventory: &Inventory) -> Capacity {
        Self::resource_remaining(self.farm_baseline_missiles(), inventory.max_missiles)
    }

    pub fn farm_baseline_missiles_available(
        &self,
        inventory: &Inventory,
        reverse: bool,
    ) -> Capacity {
        Self::resource_available(
            self.farm_baseline_missiles(),
            inventory.max_missiles,
            reverse,
        )
    }

    pub fn farm_baseline_supers_remaining(&self, inventory: &Inventory) -> Capacity {
        Self::resource_remaining(self.farm_baseline_supers(), inventory.max_supers)
    }

    pub fn farm_baseline_supers_available(&self, inventory: &Inventory, reverse: bool) -> Capacity {
        Self::resource_available(self.farm_baseline_supers(), inventory.max_supers, reverse)
    }

    pub fn farm_baseline_power_bombs_remaining(&self, inventory: &Inventory) -> Capacity {
        Self::resource_remaining(self.farm_baseline_power_bombs(), inventory.max_power_bombs)
    }

    pub fn farm_baseline_power_bombs_available(
        &self,
        inventory: &Inventory,
        reverse: bool,
    ) -> Capacity {
        Self::resource_available(
            self.farm_baseline_power_bombs(),
            inventory.max_power_bombs,
            reverse,
        )
    }

    #[must_use]
    pub fn auto_reserve_trigger(
        &mut self,
        min_refill: Capacity,
        max_refill: Capacity,
        inventory: &Inventory,
        heated: bool,
        reverse: bool,
    ) -> bool {
        let reserves_remaining = self.reserves_remaining(inventory);
        if reverse {
            let mut reserves_needed = self.energy_remaining(inventory, false);
            if heated {
                reserves_needed = (reserves_needed * 4 + 2) / 3;
            }
            if reserves_needed > inventory.max_reserves
                || reserves_needed > max_refill
                || reserves_remaining > 0
            {
                false
            } else {
                self.energy = ResourceLevel::Remaining(1).into();
                self.reserves =
                    ResourceLevel::Remaining(Capacity::max(min_refill, reserves_needed)).into();
                true
            }
        } else {
            if reserves_remaining <= min_refill {
                return false;
            }
            let mut usable_reserves = Capacity::min(reserves_remaining, inventory.max_energy);
            if heated {
                usable_reserves = usable_reserves * 3 / 4;
            }
            self.energy =
                ResourceLevel::Remaining(Capacity::min(usable_reserves, max_refill)).into();
            self.reserves = ResourceLevel::Remaining(0).into();
            true
        }
    }

    #[must_use]
    pub fn use_energy(
        &mut self,
        amt: Capacity,
        can_transfer_reserves: bool,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        match (reverse, self.energy()) {
            (false, ResourceLevel::Consumed(x)) => {
                if x + amt >= inventory.max_energy {
                    if can_transfer_reserves {
                        self.energy = ResourceLevel::Consumed(inventory.max_energy - 1).into();
                        self.use_reserve_energy(
                            x + amt - (inventory.max_energy - 1),
                            inventory,
                            reverse,
                        )
                    } else {
                        false
                    }
                } else {
                    self.energy = ResourceLevel::Consumed(x + amt).into();
                    true
                }
            }
            (false, ResourceLevel::Remaining(x)) => {
                if x <= amt {
                    if can_transfer_reserves {
                        self.energy = ResourceLevel::Remaining(1).into();
                        self.use_reserve_energy(amt - x + 1, inventory, reverse)
                    } else {
                        false
                    }
                } else {
                    self.energy = ResourceLevel::Remaining(x - amt).into();
                    true
                }
            }
            (true, ResourceLevel::Consumed(x)) => {
                if x <= amt {
                    if can_transfer_reserves {
                        self.energy = ResourceLevel::Consumed(0).into();
                        self.use_reserve_energy(amt - x, inventory, reverse)
                    } else {
                        false
                    }
                } else {
                    self.energy = ResourceLevel::Consumed(x - amt).into();
                    true
                }
            }
            (true, ResourceLevel::Remaining(x)) => {
                if x + amt >= inventory.max_energy {
                    if can_transfer_reserves {
                        self.energy = ResourceLevel::Remaining(inventory.max_energy).into();
                        self.use_reserve_energy(x + amt - inventory.max_energy, inventory, reverse)
                    } else {
                        false
                    }
                } else {
                    self.energy = ResourceLevel::Remaining(x + amt).into();
                    true
                }
            }
        }
    }

    pub fn refill_energy(
        &mut self,
        amt: Capacity,
        can_transfer_reserves: bool,
        inventory: &Inventory,
        reverse: bool,
    ) {
        match (reverse, self.energy()) {
            (false, ResourceLevel::Consumed(x)) => {
                if x < amt {
                    self.energy = ResourceLevel::Consumed(0).into();
                    if can_transfer_reserves {
                        self.refill_reserve_energy(amt - x, inventory, reverse)
                    }
                } else {
                    self.energy = ResourceLevel::Consumed(x - amt).into();
                }
            }
            (false, ResourceLevel::Remaining(x)) => {
                if x + amt > inventory.max_energy {
                    self.energy = ResourceLevel::Remaining(inventory.max_energy).into();
                    if can_transfer_reserves {
                        self.refill_reserve_energy(
                            x + amt - inventory.max_energy,
                            inventory,
                            reverse,
                        )
                    }
                } else {
                    self.energy = ResourceLevel::Remaining(x + amt).into();
                }
            }
            (true, ResourceLevel::Consumed(x)) => {
                if x + amt >= inventory.max_energy {
                    self.energy = ResourceLevel::Consumed(inventory.max_energy - 1).into();
                    if can_transfer_reserves {
                        self.refill_reserve_energy(
                            x + amt - inventory.max_energy + 1,
                            inventory,
                            reverse,
                        );
                    }
                } else {
                    self.energy = ResourceLevel::Consumed(x + amt).into();
                }
            }
            (true, ResourceLevel::Remaining(x)) => {
                if x <= amt {
                    self.energy = ResourceLevel::Remaining(1).into();
                    if can_transfer_reserves {
                        self.refill_reserve_energy(amt - x + 1, inventory, reverse);
                    }
                } else {
                    self.energy = ResourceLevel::Remaining(x - amt).into();
                }
            }
        }
    }

    #[must_use]
    pub fn ensure_energy_available(
        &mut self,
        amt: Capacity,
        can_transfer_reserves: bool,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        if reverse {
            // This is a bit inefficient of a way to implement this, but it's rarely used so it shouldn't matter.
            self.refill_energy(amt, can_transfer_reserves, inventory, reverse);
            self.use_energy(amt, can_transfer_reserves, inventory, reverse)
        } else {
            self.energy_remaining(inventory, can_transfer_reserves) >= amt
        }
    }

    #[must_use]
    pub fn use_resource(
        amt: Capacity,
        max_resource: Capacity,
        level: &mut EncodedResourceLevel,
        reverse: bool,
    ) -> bool {
        *level = match (reverse, level.decode()) {
            (false, ResourceLevel::Consumed(x)) => {
                if x + amt > max_resource {
                    return false;
                } else {
                    ResourceLevel::Consumed(x + amt).into()
                }
            }
            (false, ResourceLevel::Remaining(x)) => {
                if amt > x {
                    return false;
                } else {
                    ResourceLevel::Remaining(x - amt).into()
                }
            }
            (true, ResourceLevel::Consumed(x)) => {
                if amt > x {
                    return false;
                } else {
                    ResourceLevel::Consumed(x - amt).into()
                }
            }
            (true, ResourceLevel::Remaining(x)) => {
                if x + amt > max_resource {
                    return false;
                } else {
                    ResourceLevel::Remaining(x + amt).into()
                }
            }
        };
        true
    }

    #[must_use]
    pub fn use_missiles(&mut self, amt: Capacity, inventory: &Inventory, reverse: bool) -> bool {
        Self::use_resource(amt, inventory.max_missiles, &mut self.missiles, reverse)
    }

    #[must_use]
    pub fn use_supers(&mut self, amt: Capacity, inventory: &Inventory, reverse: bool) -> bool {
        Self::use_resource(amt, inventory.max_supers, &mut self.supers, reverse)
    }

    #[must_use]
    pub fn use_power_bombs(&mut self, amt: Capacity, inventory: &Inventory, reverse: bool) -> bool {
        Self::use_resource(
            amt,
            inventory.max_power_bombs,
            &mut self.power_bombs,
            reverse,
        )
    }

    #[must_use]
    pub fn use_reserve_energy(
        &mut self,
        amt: Capacity,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        Self::use_resource(amt, inventory.max_reserves, &mut self.reserves, reverse)
    }

    #[must_use]
    pub fn ensure_resource_available(
        amt: Capacity,
        max_resource: Capacity,
        level: &mut EncodedResourceLevel,
        reverse: bool,
    ) -> bool {
        if reverse {
            // This is a bit inefficient of a way to implement this, but it's rarely used so it shouldn't matter.
            Self::refill_resource(amt, max_resource, level, reverse);
            Self::use_resource(amt, max_resource, level, reverse)
        } else {
            Self::resource_remaining(level.decode(), max_resource) >= amt
        }
    }

    #[must_use]
    pub fn ensure_reserves_available(
        &mut self,
        amt: Capacity,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        Self::ensure_resource_available(amt, inventory.max_reserves, &mut self.reserves, reverse)
    }

    #[must_use]
    pub fn ensure_missiles_available(
        &mut self,
        amt: Capacity,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        Self::ensure_resource_available(amt, inventory.max_missiles, &mut self.missiles, reverse)
    }

    #[must_use]
    pub fn ensure_supers_available(
        &mut self,
        amt: Capacity,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        Self::ensure_resource_available(amt, inventory.max_supers, &mut self.supers, reverse)
    }

    #[must_use]
    pub fn ensure_power_bombs_available(
        &mut self,
        amt: Capacity,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        Self::ensure_resource_available(
            amt,
            inventory.max_power_bombs,
            &mut self.power_bombs,
            reverse,
        )
    }

    #[must_use]
    pub fn ensure_resource_missing_at_most(
        amt: Capacity,
        max_resource: Capacity,
        level: &mut EncodedResourceLevel,
        reverse: bool,
    ) -> bool {
        let missing = Self::resource_missing(level.decode(), max_resource);
        if reverse {
            *level = ResourceLevel::Consumed(Capacity::min(missing, amt)).into();
            true
        } else {
            missing <= amt
        }
    }

    #[must_use]
    pub fn ensure_reserves_missing_at_most(
        &mut self,
        amt: Capacity,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        Self::ensure_resource_missing_at_most(
            amt,
            inventory.max_reserves,
            &mut self.reserves,
            reverse,
        )
    }

    #[must_use]
    pub fn ensure_missiles_missing_at_most(
        &mut self,
        amt: Capacity,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        Self::ensure_resource_missing_at_most(
            amt,
            inventory.max_missiles,
            &mut self.missiles,
            reverse,
        )
    }

    #[must_use]
    pub fn ensure_supers_missing_at_most(
        &mut self,
        amt: Capacity,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        Self::ensure_resource_missing_at_most(amt, inventory.max_supers, &mut self.supers, reverse)
    }

    #[must_use]
    pub fn ensure_power_bombs_missing_at_most(
        &mut self,
        amt: Capacity,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        Self::ensure_resource_missing_at_most(
            amt,
            inventory.max_power_bombs,
            &mut self.power_bombs,
            reverse,
        )
    }

    #[must_use]
    pub fn ensure_energy_missing_at_most(
        &mut self,
        amt: Capacity,
        include_reserves: bool,
        inventory: &Inventory,
        reverse: bool,
    ) -> bool {
        let missing = self.energy_missing(inventory, include_reserves);
        if reverse {
            if amt < inventory.max_energy {
                self.energy = ResourceLevel::Consumed(Capacity::min(missing, amt)).into();
            } else {
                let missing_reserves = self.reserves_missing(inventory);
                self.reserves = ResourceLevel::Consumed(Capacity::min(
                    missing_reserves,
                    amt - inventory.max_energy + 1,
                ))
                .into();
            }
            true
        } else {
            missing <= amt
        }
    }

    fn refill_resource(
        amt: Capacity,
        max_resource: Capacity,
        level: &mut EncodedResourceLevel,
        reverse: bool,
    ) {
        *level = match (reverse, level.decode()) {
            (false, ResourceLevel::Consumed(x)) => {
                ResourceLevel::Consumed(Capacity::max(0, x - amt)).into()
            }
            (false, ResourceLevel::Remaining(x)) => {
                ResourceLevel::Remaining(Capacity::min(max_resource, x + amt)).into()
            }
            (true, ResourceLevel::Consumed(x)) => {
                ResourceLevel::Consumed(Capacity::min(max_resource, x + amt)).into()
            }
            (true, ResourceLevel::Remaining(x)) => {
                ResourceLevel::Remaining(Capacity::max(0, x - amt)).into()
            }
        }
    }

    pub fn refill_reserve_energy(&mut self, amt: Capacity, inventory: &Inventory, reverse: bool) {
        Self::refill_resource(amt, inventory.max_reserves, &mut self.reserves, reverse);
    }

    pub fn refill_missiles(&mut self, amt: Capacity, inventory: &Inventory, reverse: bool) {
        Self::refill_resource(amt, inventory.max_missiles, &mut self.missiles, reverse);
    }

    pub fn refill_supers(&mut self, amt: Capacity, inventory: &Inventory, reverse: bool) {
        Self::refill_resource(amt, inventory.max_supers, &mut self.supers, reverse);
    }

    pub fn refill_power_bombs(&mut self, amt: Capacity, inventory: &Inventory, reverse: bool) {
        Self::refill_resource(
            amt,
            inventory.max_power_bombs,
            &mut self.power_bombs,
            reverse,
        );
    }
}
