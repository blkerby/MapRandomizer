use std::path::Path;

use crate::{
    game_data::{GameData, RoomPtr, DoorPtr, RoomStateIdx},
    patch::{snes2pc, pc2snes, Rom, get_room_state_ptrs, apply_ips_patch, bps::BPSPatch},
};
use anyhow::{Result, Context};
use hashbrown::HashMap;

const BPS_PATCH_PATH: &str = "../patches/mosaic";

fn apply_bps_patch(rom: &mut Rom, orig_rom: &Rom, filename: &str) -> Result<()> {
    // let patch_path = format!("{}-{:X}-{}.bps", theme_name, room_ptr, state_idx);
    let path = Path::new(BPS_PATCH_PATH).join(filename);
    let patch_bytes = std::fs::read(path)?;
    let patch = BPSPatch::new(patch_bytes)?;
    patch.apply(&orig_rom.data, &mut rom.data);
    Ok(())
}

pub fn apply_retiling(rom: &mut Rom, orig_rom: &Rom, game_data: &GameData, theme_name: &str) -> Result<()> {
    // "theme" is just a temporary argument, to hard-code a constant theme through the whole game.
    // It will be eliminated once we have all the themes are are ready to assign them based on area.
    let patch_names = vec![
        "Scrolling Sky v1.5",
        "Area FX",
        "Bowling",
    ];
    for name in &patch_names {
        let patch_path_str = format!("../patches/ips/{}.ips", name);
        apply_ips_patch(rom, Path::new(&patch_path_str))?;    
    }

    let mut fx_door_ptr_map: HashMap<(RoomPtr, RoomStateIdx, DoorPtr), DoorPtr> = HashMap::new();
    for &room_ptr in game_data.raw_room_id_by_ptr.keys() {
        let state_ptrs = get_room_state_ptrs(&rom, room_ptr)?;
        for (state_idx, (_event_ptr, state_ptr)) in state_ptrs.iter().enumerate() {
            let orig_fx_ptr = orig_rom.read_u16(state_ptr + 6)? as usize;
            let fx_ptr = rom.read_u16(state_ptr + 6)? as usize;
            assert_eq!(orig_fx_ptr, fx_ptr);
            for i in 0..4 {
                let orig_door_ptr = orig_rom.read_u16(snes2pc(0x830000 + fx_ptr + i * 16))? as DoorPtr;
                let door_ptr = rom.read_u16(snes2pc(0x830000 + fx_ptr + i * 16))? as DoorPtr;
                if orig_door_ptr == 0 || orig_door_ptr == 0xFFFF {
                    break;
                }
                fx_door_ptr_map.insert((room_ptr, state_idx, orig_door_ptr), door_ptr);
            }
        }
    }

    apply_bps_patch(rom, orig_rom, "tilesets.bps")?;
    for &room_ptr in game_data.raw_room_id_by_ptr.keys() {
        let state_ptrs = get_room_state_ptrs(&rom, room_ptr)?;
        for (state_idx, (_event_ptr, state_ptr)) in state_ptrs.iter().enumerate() {
            let patch_filename = format!("{}-{:X}-{}.bps", theme_name, room_ptr, state_idx);
            apply_bps_patch(rom, orig_rom, &patch_filename)?;

            let fx_ptr = rom.read_u16(state_ptr + 6)? as usize;
            for i in 0..4 {
                let door_ptr_addr = snes2pc(0x830000 + fx_ptr + i * 16);
                let door_ptr = rom.read_u16(door_ptr_addr)? as DoorPtr;
                if door_ptr == 0 || door_ptr == 0xffff {
                    break;
                }
                let new_door_ptr = fx_door_ptr_map[&(room_ptr, state_idx, door_ptr)];
                rom.write_u16(door_ptr_addr, new_door_ptr as isize)?;
            }
        }
    }

    Ok(())
}
