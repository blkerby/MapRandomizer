use anyhow::{Context, Result};
use hashbrown::HashMap;
use log::info;

use maprando_game::GameData;

use super::environment_fx::{self, FxAllocator};
use super::Rom;
use crate::randomize::Randomization;

pub struct HeatFxPatcher<'a> {
    rom: &'a mut Rom,
    game_data: &'a GameData,
    randomization: &'a Randomization,
    fx_allocator: &'a mut FxAllocator,
}

impl<'a> HeatFxPatcher<'a> {
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
        if self.randomization.heat_assignments.is_empty() {
            return Ok(());
        }

        let donor_templates: HashMap<usize, Vec<u8>> = self
            .randomization
            .heat_assignments
            .values()
            .map(|a| a.donor_room_ptr)
            .collect::<hashbrown::HashSet<_>>()
            .into_iter()
            .map(|ptr| {
                let template = environment_fx::read_default_fx(self.rom, ptr)
                    .with_context(|| format!("Unable to read heat FX template from room {ptr:x}"))?;
                Ok((ptr, template))
            })
            .collect::<Result<_>>()?;

        for (&room_id, assignment) in &self.randomization.heat_assignments {
            let room_ptr = self.game_data.room_ptr_by_id[&room_id];
            if !self.randomization.map.room_mask[self.game_data.room_idx_by_id[&room_id]] {
                continue;
            }
            let template = &donor_templates[&assignment.donor_room_ptr];
            let fx_bytes = build_heat_fx_bytes(template);
            environment_fx::write_room_fx(
                self.rom,
                self.fx_allocator,
                room_ptr,
                &fx_bytes,
            )?;
            if self.randomization.heat_assignments.len() <= 10 {
                info!(
                    "Applied heat FX to room {} ({room_id}) at {room_ptr:x}",
                    room_name(self.game_data, room_id)
                );
            }
        }
        if self.randomization.heat_assignments.len() > 10 {
            info!(
                "Applied heat FX to {} rooms",
                self.randomization.heat_assignments.len()
            );
        }
        Ok(())
    }
}

fn room_name(game_data: &GameData, room_id: usize) -> String {
    game_data.room_json_map[&room_id]["name"]
        .as_str()
        .unwrap_or("?")
        .to_string()
}

fn build_heat_fx_bytes(template: &[u8]) -> Vec<u8> {
    let mut fx = template.to_vec();
    if fx.len() < environment_fx::FX_BYTES_PER_ENTRY {
        fx.resize(environment_fx::FX_BYTES_PER_ENTRY, 0);
    }
    fx.truncate(environment_fx::FX_BYTES_PER_ENTRY);
    fx[9] = 2; // heat FX type (Volcano Room)
    fx
}
