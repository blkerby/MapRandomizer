use anyhow::{Context, Result};
use hashbrown::HashMap;
use log::info;

use maprando_game::GameData;

use super::environment_fx::{self, FxAllocator};
use super::Rom;
use crate::randomize::Randomization;
use crate::water_environment::DEFAULT_WATER_DONOR_ROOM_PTR;

pub struct DryFxPatcher<'a> {
    rom: &'a mut Rom,
    game_data: &'a GameData,
    randomization: &'a Randomization,
    fx_allocator: &'a mut FxAllocator,
}

impl<'a> DryFxPatcher<'a> {
    pub fn new(
        rom: &'a mut Rom,
        game_data: &'a GameData,
        randomization: &'a Randomization,
        fx_allocator: &'a mut FxAllocator,
    ) -> Self {
        Self {
            rom,
            game_data,
            randomization,
            fx_allocator,
        }
    }

    pub fn apply(&mut self) -> Result<()> {
        let dry_count = self.randomization.dry_water_assignments.len()
            + self.randomization.dry_heat_assignments.len();
        if dry_count == 0 {
            return Ok(());
        }

        let mut donor_ptrs = hashbrown::HashSet::new();
        for assignment in self.randomization.dry_water_assignments.values() {
            donor_ptrs.insert(assignment.donor_room_ptr);
        }
        for assignment in self.randomization.dry_heat_assignments.values() {
            donor_ptrs.insert(assignment.donor_room_ptr);
        }
        let donor_templates: HashMap<usize, Vec<u8>> = donor_ptrs
            .into_iter()
            .map(|ptr| {
                let template = read_dry_fx_template(self.rom, ptr)
                    .with_context(|| format!("Unable to read dry FX template from room {ptr:x}"))?;
                Ok((ptr, template))
            })
            .collect::<Result<_>>()?;

        for (&room_id, assignment) in &self.randomization.dry_water_assignments {
            self.apply_dry_room(room_id, assignment.donor_room_ptr, &donor_templates, "water")?;
        }
        for (&room_id, assignment) in &self.randomization.dry_heat_assignments {
            self.apply_dry_room(room_id, assignment.donor_room_ptr, &donor_templates, "heat")?;
        }

        if dry_count <= 10 {
            for (&room_id, _) in &self.randomization.dry_water_assignments {
                info!(
                    "Applied dry FX to water room {} ({room_id})",
                    room_name(self.game_data, room_id)
                );
            }
            for (&room_id, _) in &self.randomization.dry_heat_assignments {
                info!(
                    "Applied dry FX to heated room {} ({room_id})",
                    room_name(self.game_data, room_id)
                );
            }
        } else {
            info!("Applied dry FX to {dry_count} rooms");
        }
        Ok(())
    }

    fn apply_dry_room(
        &mut self,
        room_id: usize,
        donor_ptr: usize,
        donor_templates: &HashMap<usize, Vec<u8>>,
        kind: &str,
    ) -> Result<()> {
        let room_ptr = self.game_data.room_ptr_by_id[&room_id];
        let room_idx = self.game_data.room_idx_by_id[&room_id];
        if !self.randomization.map.room_mask[room_idx] {
            return Ok(());
        }
        let template = &donor_templates[&donor_ptr];
        let fx_bytes = build_dry_fx_bytes(template);
        environment_fx::write_room_fx(self.rom, self.fx_allocator, room_ptr, &fx_bytes)
            .with_context(|| format!("Unable to apply dry {kind} FX to room {room_id}"))?;
        Ok(())
    }
}

fn room_name(game_data: &GameData, room_id: usize) -> String {
    game_data.room_json_map[&room_id]["name"]
        .as_str()
        .unwrap_or("?")
        .to_string()
}

fn read_dry_fx_template(rom: &Rom, room_ptr: usize) -> Result<Vec<u8>> {
    environment_fx::read_default_fx(rom, room_ptr).or_else(|_| {
        environment_fx::read_default_fx(rom, DEFAULT_WATER_DONOR_ROOM_PTR)
    })
}

fn build_dry_fx_bytes(template: &[u8]) -> Vec<u8> {
    let mut fx = template.to_vec();
    if fx.len() < environment_fx::FX_BYTES_PER_ENTRY {
        fx.resize(environment_fx::FX_BYTES_PER_ENTRY, 0);
    }
    fx.truncate(environment_fx::FX_BYTES_PER_ENTRY);
    fx[0] = 0;
    fx[1] = 0;
    fx[2..4].copy_from_slice(&0xFFFFu16.to_le_bytes());
    fx[4..6].copy_from_slice(&0xFFFFu16.to_le_bytes());
    fx[9] = 0;
    fx
}
