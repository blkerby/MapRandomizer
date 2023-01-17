use hashbrown::HashSet;

use crate::game_data::{Item, Link, Requirement, WeaponMask, self, GameData, ItemId};

#[derive(Clone)]
pub struct GlobalState {
    pub tech: Vec<bool>,
    pub items: Vec<bool>,
    pub flags: Vec<bool>,
    pub max_energy: i32,
    pub max_reserves: i32,
    pub max_missiles: i32,
    pub max_supers: i32,
    pub max_power_bombs: i32,
    pub weapon_mask: WeaponMask,
    pub shine_charge_tiles: i32,
}

#[derive(Copy, Clone)]
pub struct LocalState {
    energy_used: i32,
    reserves_used: i32,
    missiles_used: i32,
    supers_used: i32,
    power_bombs_used: i32,
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

pub fn apply_requirement(
    req: &Requirement,
    global: &GlobalState,
    local: LocalState,
    reverse: bool,
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
        Requirement::HeatFrames(frames) => {
            let varia = global.items[Item::Varia as usize];
            let gravity = global.items[Item::Gravity as usize];
            let mut new_local = local;
            if varia {
                Some(new_local)
            } else if gravity {
                new_local.energy_used += frames / 8;
                validate_energy(new_local, global)
            } else {
                new_local.energy_used += frames / 4;
                validate_energy(new_local, global)
            }
        }
        Requirement::LavaFrames(frames) => {
            let varia = global.items[Item::Varia as usize];
            let gravity = global.items[Item::Gravity as usize];
            let mut new_local = local;
            if gravity {
                Some(new_local)
            } else if varia {
                new_local.energy_used += frames / 4;
                validate_energy(new_local, global)
            } else {
                new_local.energy_used += frames / 2;
                validate_energy(new_local, global)
            }
        }
        Requirement::LavaPhysicsFrames(frames) => {
            let varia = global.items[Item::Varia as usize];
            let mut new_local = local;
            if varia {
                new_local.energy_used += frames / 4;
            } else {
                new_local.energy_used += frames / 2;
            }
            validate_energy(new_local, global)
        }
        Requirement::Damage(base_energy) => {
            let varia = global.items[Item::Varia as usize];
            let gravity = global.items[Item::Gravity as usize];
            let mut new_local = local;
            if gravity && varia {
                new_local.energy_used += base_energy / 4;
            } else if gravity || varia {
                new_local.energy_used += base_energy / 2;
            } else {
                new_local.energy_used += base_energy;
            }
            validate_energy(new_local, global)
        }
        // Requirement::Energy(count) => {
        //     let mut new_local = local;
        //     new_local.energy_used += count;
        //     validate_energy(new_local, global)
        // },
        Requirement::Missiles(count) => {
            let mut new_local = local;
            new_local.missiles_used += count;
            validate_missiles(new_local, global)
        }
        Requirement::Supers(count) => {
            let mut new_local = local;
            new_local.supers_used += count;
            validate_supers(new_local, global)
        }
        Requirement::PowerBombs(count) => {
            let mut new_local = local;
            new_local.power_bombs_used += count;
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
        Requirement::EnemyKill(weapon_mask) => {
            // TODO: Take into account ammo-kill strats
            if global.weapon_mask & *weapon_mask != 0 {
                Some(local)
            } else {
                None
            }
        }
        Requirement::ShineCharge {
            used_tiles,
            shinespark_frames,
        } => {
            if global.items[Item::SpeedBooster as usize] && *used_tiles <= global.shine_charge_tiles
            {
                let mut new_local = local;
                new_local.energy_used += shinespark_frames;
                // TODO: handle this more accurately, to take into account the 29 energy limit:
                validate_energy(new_local, global)
            } else {
                None
            }
        }
        Requirement::And(reqs) => {
            let mut new_local = local;
            for req in reqs {
                new_local = apply_requirement(req, global, new_local, reverse)?;
            }
            Some(new_local)
        }
        Requirement::Or(reqs) => {
            let mut best_local = None;
            let mut best_cost = f32::INFINITY;
            for req in reqs {
                if let Some(new_local) = apply_requirement(req, global, local, reverse) {
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

pub fn is_bireachable(global: &GlobalState, forward_local_state: &Option<LocalState>, reverse_local_state: &Option<LocalState>) -> bool {
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

pub struct TraverseResult {
    pub local_states: Vec<Option<LocalState>>,
    pub cost: Vec<f32>,
    // add route_data for reconstructing paths for spoiler log
}

pub fn traverse(
    links: &[Link],
    global: &GlobalState,
    num_vertices: usize,
    start_vertex_id: usize,
    reverse: bool,
) -> TraverseResult {
    let mut result = TraverseResult {
        local_states: vec![None; num_vertices],
        cost: vec![f32::INFINITY; num_vertices],
    };
    result.local_states[start_vertex_id] = Some(LocalState {
        energy_used: 0,
        reserves_used: 0,
        missiles_used: 0,
        supers_used: 0,
        power_bombs_used: 0,
    });
    result.cost[start_vertex_id] = compute_cost(result.local_states[start_vertex_id].unwrap(), global);

    let mut links_by_src: Vec<Vec<Link>> = vec![Vec::new(); num_vertices];
    for link in links {
        links_by_src[link.from_vertex_id].push(link.clone());
    }

    let mut modified_vertices: HashSet<usize> = HashSet::new();
    while modified_vertices.len() > 0 {
        let mut new_modified_vertices: HashSet<usize> = HashSet::new();
        for &src_id in &modified_vertices {
            let src_local_state = result.local_states[src_id].unwrap();
            for link in &links_by_src[src_id] {
                let dst_id = link.to_vertex_id;
                let dst_old_cost = result.cost[dst_id];
                if let Some(dst_new_local_state) = apply_requirement(&link.requirement, global, src_local_state, reverse) {
                    let dst_new_cost = compute_cost(dst_new_local_state, global);
                    if dst_new_cost < dst_old_cost {
                        result.local_states[dst_id] = Some(dst_new_local_state);
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
            },
            Item::Super => {
                self.max_supers += 5;
            },
            Item::PowerBomb => { 
                self.max_power_bombs += 5;
            },
            Item::ETank => {
                self.max_energy += 100;
            },
            Item::ReserveTank => {
                self.max_reserves += 100;
            },
            _ => {}
        }
        self.weapon_mask = game_data.get_weapon_mask(&self.items);
    }
}