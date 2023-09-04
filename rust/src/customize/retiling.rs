use super::Allocator;
use crate::{
    game_data::{GameData, AreaIdx, TilesetIdx},
    patch::{snes2pc, pc2snes, Rom},
};
use anyhow::{bail, Result};
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

pub fn apply_retiling(rom: &mut Rom, game_data: &GameData) -> Result<()> {
    let retiled_theme_data = game_data.retiled_theme_data.as_ref().unwrap();

    let new_tile_table_snes = 0x8FF900;
    let new_tile_pointers_snes = 0x8FFD00;
    let tile_pointers_free_space_end = 0x8FFE00;

    let mut allocator = Allocator::new(vec![
        (snes2pc(0xBAC629), snes2pc(0xC2C2BB)), // Vanilla tile GFX, tilemaps, and palettes, which we overwrite
        (snes2pc(0xE18000), snes2pc(0xE20000)),
        // (snes2pc(0xEA8000), snes2pc(0xF00000)),
        (snes2pc(0xEA8000), snes2pc(0xF80000)),
    ]);

    let mut pal_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut gfx8_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut gfx16_map: HashMap<Vec<u8>, usize> = HashMap::new();

    // let mut tile_table: Vec<u8> = rom.read_n(snes2pc(0x8FE6A2), next_tile_idx * 9)?.to_vec();
    let base_theme = &retiled_theme_data.themes["Base"];
    let num_tilesets = base_theme.sce_tilesets.keys().max().unwrap() + 1;

    for (tileset_idx, tileset) in base_theme.sce_tilesets.iter() {
        let compressed_pal = &tileset.compressed_palette;
        let pal_addr = match pal_map.entry(compressed_pal.clone()) {
            Entry::Occupied(x) => *x.get(),
            Entry::Vacant(view) => {
                let addr = allocator.allocate(compressed_pal.len())?;
                view.insert(addr);
                addr
            }
        };
        rom.write_n(pal_addr, &compressed_pal)?;

        let compressed_gfx8 = &tileset.compressed_gfx8x8;
        let gfx8_addr = match gfx8_map.entry(compressed_gfx8.clone()) {
            Entry::Occupied(x) => *x.get(),
            Entry::Vacant(view) => {
                let addr = allocator.allocate(compressed_gfx8.len())?;
                view.insert(addr);
                addr
            }
        };
        rom.write_n(gfx8_addr, &compressed_gfx8)?;

        let compressed_gfx16 = &tileset.compressed_gfx16x16;
        let gfx16_addr = match gfx16_map.entry(compressed_gfx16.clone()) {
            Entry::Occupied(x) => *x.get(),
            Entry::Vacant(view) => {
                let addr = allocator.allocate(compressed_gfx16.len())?;
                view.insert(addr);
                addr
            }
        };
        rom.write_n(gfx16_addr, &compressed_gfx16)?;

        let tile_table_entry_addr = new_tile_table_snes + 9 * tileset_idx;
        rom.write_u24(snes2pc(tile_table_entry_addr), pc2snes(gfx16_addr) as isize)?;
        rom.write_u24(snes2pc(tile_table_entry_addr + 3), pc2snes(gfx8_addr) as isize)?;
        rom.write_u24(snes2pc(tile_table_entry_addr + 6), pc2snes(pal_addr) as isize)?;
        rom.write_u16(snes2pc(new_tile_pointers_snes + 2 * tileset_idx), 
        (tile_table_entry_addr & 0xFFFF) as isize)?;
    }
    println!("Number of tilesets: {}", num_tilesets);
    println!("Number of unique pal: {}", pal_map.len());
    println!("Number of unique gfx8: {}", gfx8_map.len());
    println!("Number of unique gfx16: {}", gfx16_map.len());
    assert!(num_tilesets * 9  <= new_tile_pointers_snes - new_tile_table_snes);

    rom.write_u16(
        snes2pc(0x82DF03),
        (new_tile_pointers_snes & 0xFFFF) as isize,
    )?;

    let theme = &retiled_theme_data.themes["OuterCrateria"];

    Ok(())
}
