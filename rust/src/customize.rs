use crate::{game_data::{GameData, TilesetIdx, AreaIdx}, patch::{Rom, compress::compress, snes2pc}};
use anyhow::Result;
use hashbrown::HashMap;
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
    let new_tile_table_snes = 0x8FF900;
    let new_tile_pointers_snes = 0x8FFC00;
    let pal_free_space_start_snes = 0xE18000;
    let pal_free_space_end_snes = pal_free_space_start_snes + 0x8000;
    let mut pal_free_space_snes = pal_free_space_start_snes;
    rom.data.resize(max(snes2pc(pal_free_space_end_snes), rom.data.len()), 0xFF);

    let mut next_tile_idx = 29;
    let mut tile_table: Vec<u8> = rom.read_n(snes2pc(0x8FE6A2), next_tile_idx * 9)?.to_vec();
    let mut tile_map: HashMap<(AreaIdx, TilesetIdx), TilesetIdx> = HashMap::new();
    for (area_idx, area_palette_data) in game_data.palette_data.iter().enumerate() {
        for (&tileset_idx, pal) in area_palette_data {
            let encoded_pal = encode_palette(pal);
            let compressed_pal = compress(&encoded_pal);
            rom.write_n(snes2pc(pal_free_space_snes), &compressed_pal)?;

            let data = tile_table[(tileset_idx * 9)..(tileset_idx * 9 + 6)].to_vec();
            tile_table.extend(&data);
            tile_table.extend(&pal_free_space_snes.to_le_bytes()[0..2]);
            tile_map.insert((area_idx, tileset_idx), next_tile_idx);

            next_tile_idx += 1;
            pal_free_space_snes += compressed_pal.len();
        }
    }
    println!("Tileset table size: {}", tile_table.len());
    assert!(pal_free_space_snes <= pal_free_space_end_snes);
    assert!(tile_table.len() <= new_tile_pointers_snes - new_tile_table_snes);

    rom.write_n(snes2pc(new_tile_table_snes), &tile_table)?;
    for i in 0..tile_table.len() / 3 {
        rom.write_u24(snes2pc(new_tile_pointers_snes + 2 * i), (new_tile_table_snes + 9 * i) as isize)?;
    }

    rom.write_u16(snes2pc(0x82DF03), (new_tile_pointers_snes & 0xFFFF) as isize)?;
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
