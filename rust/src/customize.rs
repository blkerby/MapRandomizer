use crate::{game_data::GameData, patch::{Rom, compress::compress, snes2pc}};
use anyhow::Result;
use std::cmp::max;

pub struct CustomizeSettings {
    pub area_themed_palette: bool,
}

fn encode_palette(pal: &[[u8; 3]; 128]) -> [u8; 256] {
    let mut out = [0u8; 256];
    for i in 0..128 {
        let r = pal[i][0] as u16 / 8;
        let g = pal[i][1] as u16 / 8;
        let b = pal[i][2] as u16 / 8;
        let w = r | (g << 5) | (b << 10);
        out[i * 2] = (w & 0xFF) as u8;
        out[i * 2 + 1] = (w >> 8) as u8;
    }
    out
}

pub fn apply_area_themed_palettes(rom: &mut Rom, game_data: &GameData) -> Result<()> {   
    let area_palette_data = &game_data.palette_data[0];
    let free_space_start_snes = 0xE18000;
    let free_space_end_snes = free_space_start_snes + 0x8000;
    let mut free_space_snes = free_space_start_snes;
    rom.data.resize(max(snes2pc(free_space_end_snes), rom.data.len()), 0xFF);
    for (tileset_idx, pal) in area_palette_data {
        let encoded_pal = encode_palette(pal);
        let compressed_pal = compress(&encoded_pal);
        rom.write_n(snes2pc(free_space_snes), &compressed_pal)?;
        rom.write_u24(snes2pc(0x8FE6A2 + tileset_idx * 9 + 6), free_space_snes as isize)?;
        free_space_snes += compressed_pal.len();
    }
    assert!(free_space_snes <= free_space_end_snes);
    Ok(())
}

pub fn customize_rom(
    rom: &mut Rom,
    seed_patch: &[u8],
    settings: &CustomizeSettings,
    game_data: &GameData,
) -> Result<()> {
    let patch = ips::Patch::parse(seed_patch).unwrap();
    // .with_context(|| format!("Unable to parse patch {}", patch_path.display()))?;
    for hunk in patch.hunks() {
        rom.write_n(hunk.offset(), hunk.payload())?;
    }
    if settings.area_themed_palette {
        apply_area_themed_palettes(rom, game_data)?;
    }
    Ok(())
}
