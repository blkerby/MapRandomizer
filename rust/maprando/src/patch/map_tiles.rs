use hashbrown::{HashMap, HashSet};

use crate::{
    randomize::Randomization,
    settings::{
        DoorLocksSize, ItemDotChange, ItemMarkers, MapStationReveal, MapsRevealed, Objective,
    },
};
use maprando_game::{
    AreaIdx, BeamType, Direction, DoorLockType, DoorType, GameData, Item, ItemIdx, Map, MapTile,
    MapTileEdge, MapTileInterior, MapTileSpecialType, RoomGeometryDoor, RoomGeometryItem, RoomId,
};

use super::{snes2pc, xy_to_explored_bit_ptr, xy_to_map_offset, Rom};
use anyhow::{bail, Context, Result};

pub type TilemapOffset = u16;
pub type TilemapWord = u16;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TileSide {
    Right,
    Left,
    Top,
    Bottom,
}

const NUM_AREAS: usize = 6;

pub struct MapPatcher<'a> {
    rom: &'a mut Rom,
    game_data: &'a GameData,
    map: &'a Map,
    randomization: &'a Randomization,
    map_tile_map: HashMap<(AreaIdx, isize, isize), MapTile>,
    gfx_tile_map: HashMap<(AreaIdx, [[u8; 8]; 8]), TilemapWord>,
    free_tiles: Vec<Vec<TilemapWord>>, // set of free tile indexes, by area
    locked_door_state_indices: &'a [usize],
    dynamic_tile_data: Vec<Vec<(ItemIdx, RoomId, MapTile)>>,
    transition_tile_coords: Vec<(AreaIdx, isize, isize)>,
    area_min_x: [isize; NUM_AREAS],
    area_max_x: [isize; NUM_AREAS],
    area_min_y: [isize; NUM_AREAS],
    area_max_y: [isize; NUM_AREAS],
    area_offset_x: [isize; NUM_AREAS],
    area_offset_y: [isize; NUM_AREAS],
}

pub const VANILLA_ELEVATOR_TILE: TilemapWord = 0xCE; // Index of elevator tile in vanilla game
pub const ELEVATOR_TILE: TilemapWord = 0x12; // Index of elevator tile with TR's map patch
pub const TILE_GFX_ADDR_4BPP: usize = 0xE28000; // Where to store area-specific tile graphics (must agree with map_area.asm)
pub const TILE_GFX_ADDR_2BPP: usize = 0xE2C000; // Where to store area-specific tile graphics (must agree with map_area.asm)

const FLIP_X: TilemapWord = 0x4000;
const FLIP_Y: TilemapWord = 0x8000;

fn find_item_xy(addr: usize, room_items: &[RoomGeometryItem]) -> Result<(isize, isize)> {
    for room_item in room_items {
        if room_item.addr == addr {
            return Ok((room_item.x as isize, room_item.y as isize));
        }
    }
    bail!("Could not find item in room: {addr:x}");
}

fn update_tile(tile: &mut [[u8; 8]; 8], value: u8, coords: &[(usize, usize)]) {
    for &(x, y) in coords {
        tile[y][x] = value;
    }
}

pub fn hflip_tile(tile: [[u8; 8]; 8]) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            out[y][x] = tile[y][7 - x];
        }
    }
    out
}

pub fn vflip_tile(tile: [[u8; 8]; 8]) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            out[y][x] = tile[7 - y][x];
        }
    }
    out
}

pub fn diagonal_flip_tile(tile: [[u8; 8]; 8]) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            out[y][x] = tile[x][y];
        }
    }
    out
}

pub fn write_tile_4bpp(rom: &mut Rom, base_addr: usize, data: [[u8; 8]; 8]) -> Result<()> {
    for y in 0..8 {
        let addr = base_addr + y * 2;
        let data_0: u8 = (0..8).map(|x| (data[y][x] & 1) << (7 - x)).sum();
        let data_1: u8 = (0..8).map(|x| ((data[y][x] >> 1) & 1) << (7 - x)).sum();
        let data_2: u8 = (0..8).map(|x| ((data[y][x] >> 2) & 1) << (7 - x)).sum();
        let data_3: u8 = (0..8).map(|x| ((data[y][x] >> 3) & 1) << (7 - x)).sum();
        rom.write_u8(addr, data_0 as isize)?;
        rom.write_u8(addr + 1, data_1 as isize)?;
        rom.write_u8(addr + 16, data_2 as isize)?;
        rom.write_u8(addr + 17, data_3 as isize)?;
    }
    Ok(())
}

impl<'a> MapPatcher<'a> {
    pub fn new(
        rom: &'a mut Rom,
        game_data: &'a GameData,
        map: &'a Map,
        randomization: &'a Randomization,
        locked_door_state_indices: &'a [usize],
    ) -> Self {
        let mut reserved_tiles: HashSet<TilemapWord> = vec![
            // Used on HUD: (skipping "%", which is unused)
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x1C, 0x1D, 0x1E, 0x1F,
            0x20, // reserved for partially revealed door tile, next to 2-sided save/refill rooms
            0x28, // slope tile that triggers tile above Samus to be marked explored
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x38, 0x39, 0x3A, 0x3B,
            // Max ammo display digits: (removed in favor of normal digit graphics)
            // 0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45,
            0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D,
            0xA8, // heated slope tile corresponding to 0x28
            // Message box letters and punctuation (skipping unused ones: "Q", "->", "'", "-", "!")
            0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD,
            0xCE, 0xCF, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDD, 0xDE,
        ]
        .into_iter()
        .collect();

        if randomization
            .settings
            .quality_of_life_settings
            .disableable_etanks
        {
            // Reserve tile $2F for disabled ETank
            reserved_tiles.insert(0x2F);
        }

        let mut free_tiles: Vec<TilemapWord> = Vec::new();
        for word in 0..224 {
            if !reserved_tiles.contains(&word) {
                free_tiles.push(word);
            }
        }
        free_tiles.reverse();

        MapPatcher {
            rom,
            game_data,
            map,
            randomization,
            map_tile_map: HashMap::new(),
            gfx_tile_map: HashMap::new(),
            free_tiles: vec![free_tiles.clone(); 6],
            locked_door_state_indices,
            dynamic_tile_data: vec![vec![]; 6],
            transition_tile_coords: vec![],
            area_min_x: [isize::MAX; NUM_AREAS],
            area_min_y: [isize::MAX; NUM_AREAS],
            area_max_x: [0; NUM_AREAS],
            area_max_y: [0; NUM_AREAS],
            area_offset_x: [0; NUM_AREAS],
            area_offset_y: [0; NUM_AREAS],
        }
    }

    fn index_fixed_tiles(&mut self) -> Result<()> {
        for area_idx in 0..NUM_AREAS {
            let mut tile = MapTile::default();
            tile.special_type = Some(MapTileSpecialType::SlopeUpFloorLow);
            self.index_tile(area_idx, tile.clone(), Some(0x28))?;
            tile.heated = true;
            self.index_tile(area_idx, tile.clone(), Some(0xA8))?;

            let data = [
                [3, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0],
                [4, 0, 0, 0, 0, 0, 0, 0],
                [4, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0],
            ];
            self.write_map_tile_4bpp_area(0x20, data, area_idx)?;
        }
        Ok(())
    }

    fn read_map_tile_2bpp(&self, idx: usize) -> Result<[[u8; 8]; 8]> {
        let base_addr = snes2pc(0x9AB200); // Location of HUD tile GFX in ROM
        let mut out: [[u8; 8]; 8] = [[0; 8]; 8];
        for y in 0..8 {
            let addr = base_addr + idx * 16 + y * 2;
            let data_low = self.rom.read_u8(addr)?;
            let data_high = self.rom.read_u8(addr + 1)?;
            for x in 0..8 {
                let bit_low = (data_low >> (7 - x)) & 1;
                let bit_high = (data_high >> (7 - x)) & 1;
                let c = bit_low | (bit_high << 1);
                out[y][x] = c as u8;
            }
        }
        Ok(out)
    }

    fn read_tile_4bpp(&self, base_addr: usize) -> Result<[[u8; 8]; 8]> {
        let mut out: [[u8; 8]; 8] = [[0; 8]; 8];
        for y in 0..8 {
            let addr = base_addr + y * 2;
            let data_0 = self.rom.read_u8(addr)?;
            let data_1 = self.rom.read_u8(addr + 1)?;
            let data_2 = self.rom.read_u8(addr + 16)?;
            let data_3 = self.rom.read_u8(addr + 17)?;
            for x in 0..8 {
                let bit_0 = (data_0 >> (7 - x)) & 1;
                let bit_1 = (data_1 >> (7 - x)) & 1;
                let bit_2 = (data_2 >> (7 - x)) & 1;
                let bit_3 = (data_3 >> (7 - x)) & 1;
                let c = bit_0 | (bit_1 << 1) | (bit_2 << 2) | (bit_3 << 3);
                out[y][x] = c as u8;
            }
        }
        Ok(out)
    }

    fn read_map_tile_4bpp(&self, idx: usize) -> Result<[[u8; 8]; 8]> {
        let addr = snes2pc(0xB68000) + idx * 32;
        self.read_tile_4bpp(addr)
    }

    fn index_tile(
        &mut self,
        area_idx: usize,
        tile: MapTile,
        fixed_idx: Option<u16>,
    ) -> Result<TilemapWord> {
        let data = self.render_tile(tile)?;
        if self.gfx_tile_map.contains_key(&(area_idx, data)) {
            Ok(self.gfx_tile_map[&(area_idx, data)])
        } else if self
            .gfx_tile_map
            .contains_key(&(area_idx, hflip_tile(data)))
        {
            Ok(self.gfx_tile_map[&(area_idx, hflip_tile(data))] | FLIP_X)
        } else if self
            .gfx_tile_map
            .contains_key(&(area_idx, vflip_tile(data)))
        {
            Ok(self.gfx_tile_map[&(area_idx, vflip_tile(data))] | FLIP_Y)
        } else if self
            .gfx_tile_map
            .contains_key(&(area_idx, hflip_tile(vflip_tile(data))))
        {
            Ok(self.gfx_tile_map[&(area_idx, hflip_tile(vflip_tile(data)))] | FLIP_X | FLIP_Y)
        } else {
            let tile_idx = if let Some(i) = fixed_idx {
                i
            } else {
                self.free_tiles[area_idx]
                    .pop()
                    .context("No more free tiles")?
            };
            let palette = 0x1800;
            let word = tile_idx | palette;
            self.gfx_tile_map.insert((area_idx, data), word);
            Ok(word)
        }
    }

    fn write_map_tiles(&mut self) -> Result<()> {
        // Clear all map tilemap data:
        for area_ptr in &self.game_data.area_map_ptrs {
            for i in 0..(64 * 32) {
                self.rom.write_u16((area_ptr + i * 2) as usize, 0x001F)?;
            }
        }

        // Index map graphics and write map tilemap by room:
        for ((area_idx, x, y), tile) in self.map_tile_map.clone() {
            let word = self.index_tile(area_idx, tile.clone(), None)?;
            let local_x = x - self.area_offset_x[area_idx];
            let local_y = y - self.area_offset_y[area_idx];

            if tile.special_type != Some(MapTileSpecialType::Black) {
                let base_ptr = self.game_data.area_map_ptrs[area_idx];
                let offset = xy_to_map_offset(local_x, local_y);
                let ptr = (base_ptr + offset) as usize;
                self.rom.write_u16(ptr, word as isize)?;
            }

            if let Some(MapTileSpecialType::AreaTransition(_, _)) = tile.special_type {
                self.transition_tile_coords
                    .push((area_idx, local_x, local_y));
            }
        }

        // Index dynamic item/door tile graphics:
        for area_idx in 0..6 {
            for (_, _, tile) in self.dynamic_tile_data[area_idx].clone() {
                let _ = self.index_tile(area_idx, tile, None)?;
            }
        }

        // Write map tile graphics:
        for ((area_idx, data), word) in self.gfx_tile_map.clone() {
            let idx = (word & 0x3FF) as usize;
            self.write_tile_2bpp_area(idx, data, Some(area_idx))?;
            self.write_map_tile_4bpp_area(idx, data, area_idx)?;
        }

        // Write room map offsets:
        for (room_idx, room) in self.game_data.room_geometry.iter().enumerate() {
            let area = self.randomization.map.area[room_idx];
            let room_x = self.map.rooms[room_idx].0 as isize - self.area_offset_x[area];
            let room_y = self.map.rooms[room_idx].1 as isize - self.area_offset_y[area];
            self.rom.write_u8(room.rom_address + 2, room_x)?;
            self.rom.write_u8(room.rom_address + 3, room_y)?;
        }

        Ok(())
    }

    fn write_tile_2bpp_area(
        &mut self,
        idx: usize,
        mut data: [[u8; 8]; 8],
        area_idx: Option<usize>,
    ) -> Result<()> {
        let base_addr = match area_idx {
            Some(area) => snes2pc(TILE_GFX_ADDR_2BPP + area * 0x10000), // New HUD tile GFX in ROM
            None => snes2pc(0x9AB200), // Standard BG3 tiles (used during Kraid)
        };
        for y in 0..8 {
            for x in 0..8 {
                if data[y][x] == 4 || data[y][x] == 12 {
                    data[y][x] = 0;
                } else if data[y][x] > 4 {
                    data[y][x] = 3;
                }
            }
        }

        for y in 0..8 {
            let addr = base_addr + idx * 16 + y * 2;
            let data_low: u8 = (0..8).map(|x| (data[y][x] & 1) << (7 - x)).sum();
            let data_high: u8 = (0..8).map(|x| (data[y][x] >> 1) << (7 - x)).sum();
            self.rom.write_u8(addr, data_low as isize)?;
            self.rom.write_u8(addr + 1, data_high as isize)?;
        }
        Ok(())
    }

    fn write_tile_2bpp(&mut self, idx: usize, data: [[u8; 8]; 8]) -> Result<()> {
        self.write_tile_2bpp_area(idx, data, None)?;
        for area_idx in 0..6 {
            self.write_tile_2bpp_area(idx, data, Some(area_idx))?;
        }
        Ok(())
    }

    fn write_tile_4bpp(&mut self, base_addr: usize, data: [[u8; 8]; 8]) -> Result<()> {
        write_tile_4bpp(&mut self.rom, base_addr, data)
    }

    fn write_map_tile_4bpp_area(
        &mut self,
        idx: usize,
        data: [[u8; 8]; 8],
        area_idx: usize,
    ) -> Result<()> {
        if idx == 0x20 && data[1][1] != 0 {
            println!("write: idx={}, area={}, {:?}", idx, area_idx, data);
            bail!("err");
        }
        let base_addr = snes2pc(TILE_GFX_ADDR_4BPP + area_idx * 0x10000); // Location of pause-menu tile GFX in ROM
        for y in 0..8 {
            let addr = base_addr + idx * 32 + y * 2;
            let data_0: u8 = (0..8).map(|x| (data[y][x] & 1) << (7 - x)).sum();
            let data_1: u8 = (0..8).map(|x| ((data[y][x] >> 1) & 1) << (7 - x)).sum();
            let data_2: u8 = (0..8).map(|x| ((data[y][x] >> 2) & 1) << (7 - x)).sum();
            let data_3: u8 = (0..8).map(|x| ((data[y][x] >> 3) & 1) << (7 - x)).sum();
            self.rom.write_u8(addr, data_0 as isize)?;
            self.rom.write_u8(addr + 1, data_1 as isize)?;
            self.rom.write_u8(addr + 16, data_2 as isize)?;
            self.rom.write_u8(addr + 17, data_3 as isize)?;
        }
        Ok(())
    }

    fn write_map_tile_4bpp(&mut self, idx: usize, data: [[u8; 8]; 8]) -> Result<()> {
        for area_idx in 0..6 {
            self.write_map_tile_4bpp_area(idx, data, area_idx)?;
        }
        Ok(())
    }

    fn draw_edge(&self, tile_side: TileSide, edge: MapTileEdge, tile: &mut [[u8; 8]; 8]) {
        let wall_coords = match tile_side {
            TileSide::Top => [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (0, 6),
                (0, 7),
            ],
            TileSide::Bottom => [
                (7, 7),
                (7, 6),
                (7, 5),
                (7, 4),
                (7, 3),
                (7, 2),
                (7, 1),
                (7, 0),
            ],
            TileSide::Left => [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
            ],
            TileSide::Right => [
                (7, 7),
                (6, 7),
                (5, 7),
                (4, 7),
                (3, 7),
                (2, 7),
                (1, 7),
                (0, 7),
            ],
        };
        let air_coords = match tile_side {
            TileSide::Top => [
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (1, 6),
                (1, 7),
            ],
            TileSide::Bottom => [
                (6, 7),
                (6, 6),
                (6, 5),
                (6, 4),
                (6, 3),
                (6, 2),
                (6, 1),
                (6, 0),
            ],
            TileSide::Left => [
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 1),
                (6, 1),
                (7, 1),
            ],
            TileSide::Right => [
                (7, 6),
                (6, 6),
                (5, 6),
                (4, 6),
                (3, 6),
                (2, 6),
                (1, 6),
                (0, 6),
            ],
        };
        let deep_coords = match tile_side {
            TileSide::Top => [
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (2, 7),
            ],
            TileSide::Bottom => [
                (5, 7),
                (5, 6),
                (5, 5),
                (5, 4),
                (5, 3),
                (5, 2),
                (5, 1),
                (5, 0),
            ],
            TileSide::Left => [
                (0, 2),
                (1, 2),
                (2, 2),
                (3, 2),
                (4, 2),
                (5, 2),
                (6, 2),
                (7, 2),
            ],
            TileSide::Right => [
                (7, 5),
                (6, 5),
                (5, 5),
                (4, 5),
                (3, 5),
                (2, 5),
                (1, 5),
                (0, 5),
            ],
        };

        let set_wall_pixel = |tile: &mut [[u8; 8]; 8], i: usize, color: u8| {
            tile[wall_coords[i].0][wall_coords[i].1] = color;
        };
        let set_air_pixel = |tile: &mut [[u8; 8]; 8], i: usize, color: u8| {
            tile[air_coords[i].0][air_coords[i].1] = color;
        };
        let set_deep_pixel = |tile: &mut [[u8; 8]; 8], i: usize, color: u8| {
            tile[deep_coords[i].0][deep_coords[i].1] = color;
        };
        use DoorLockType::*;
        match edge {
            MapTileEdge::Empty => {}
            MapTileEdge::QolEmpty => {
                if self.randomization.settings.other_settings.ultra_low_qol {
                    set_wall_pixel(tile, 0, 3);
                    set_wall_pixel(tile, 1, 3);
                    set_wall_pixel(tile, 2, 3);
                    set_wall_pixel(tile, 3, 3);
                    set_wall_pixel(tile, 4, 3);
                    set_wall_pixel(tile, 5, 3);
                    set_wall_pixel(tile, 6, 3);
                    set_wall_pixel(tile, 7, 3);
                }
            }
            MapTileEdge::Passage => {
                set_wall_pixel(tile, 0, 3);
                set_wall_pixel(tile, 1, 3);
                if self.randomization.settings.other_settings.ultra_low_qol {
                    set_wall_pixel(tile, 2, 3);
                    set_wall_pixel(tile, 3, 3);
                    set_wall_pixel(tile, 4, 3);
                    set_wall_pixel(tile, 5, 3);
                }
                set_wall_pixel(tile, 6, 3);
                set_wall_pixel(tile, 7, 3);
            }
            MapTileEdge::QolPassage => {
                if !self.randomization.settings.other_settings.ultra_low_qol {
                    set_wall_pixel(tile, 0, 3);
                    set_wall_pixel(tile, 1, 3);
                    set_wall_pixel(tile, 6, 3);
                    set_wall_pixel(tile, 7, 3);
                }
            }
            MapTileEdge::Door | MapTileEdge::QolDoor => {
                set_wall_pixel(tile, 0, 3);
                set_wall_pixel(tile, 1, 3);
                set_wall_pixel(tile, 2, 3);
                if self.randomization.settings.other_settings.ultra_low_qol {
                    set_wall_pixel(tile, 3, 3);
                    set_wall_pixel(tile, 4, 3);
                }
                set_wall_pixel(tile, 5, 3);
                set_wall_pixel(tile, 6, 3);
                set_wall_pixel(tile, 7, 3);
            }
            MapTileEdge::Wall | MapTileEdge::QolWall => {
                set_wall_pixel(tile, 0, 3);
                set_wall_pixel(tile, 1, 3);
                set_wall_pixel(tile, 2, 3);
                set_wall_pixel(tile, 3, 3);
                set_wall_pixel(tile, 4, 3);
                set_wall_pixel(tile, 5, 3);
                set_wall_pixel(tile, 6, 3);
                set_wall_pixel(tile, 7, 3);
            }
            MapTileEdge::Sand | MapTileEdge::QolSand => {
                if self.randomization.settings.other_settings.ultra_low_qol {
                    set_wall_pixel(tile, 0, 3);
                    set_wall_pixel(tile, 1, 3);
                    set_wall_pixel(tile, 2, 3);
                    set_wall_pixel(tile, 3, 3);
                    set_wall_pixel(tile, 4, 3);
                    set_wall_pixel(tile, 5, 3);
                    set_wall_pixel(tile, 6, 3);
                    set_wall_pixel(tile, 7, 3);
                } else {
                    if tile_side == TileSide::Bottom {
                        set_wall_pixel(tile, 0, 3);
                        set_wall_pixel(tile, 1, 3);
                        set_wall_pixel(tile, 6, 3);
                        set_wall_pixel(tile, 7, 3);
                    } else {
                        set_wall_pixel(tile, 0, 3);
                        set_wall_pixel(tile, 1, 3);
                        set_wall_pixel(tile, 2, 3);
                        set_wall_pixel(tile, 5, 3);
                        set_wall_pixel(tile, 6, 3);
                        set_wall_pixel(tile, 7, 3);
                    }
                }
            }
            MapTileEdge::ElevatorEntrance => {
                set_wall_pixel(tile, 0, 3);
                set_wall_pixel(tile, 1, 3);
                set_wall_pixel(tile, 6, 3);
                set_wall_pixel(tile, 7, 3);
                set_air_pixel(tile, 0, 3);
                set_air_pixel(tile, 7, 3);
            }
            MapTileEdge::LockedDoor(lock_type) => {
                if [Gray, Red, Green, Yellow].contains(&lock_type) {
                    let color = match lock_type {
                        DoorLockType::Gray => 15,
                        DoorLockType::Red => 7,
                        DoorLockType::Green => 14,
                        DoorLockType::Yellow => 6,
                        _ => panic!("Internal error"),
                    };
                    match self.randomization.settings.other_settings.door_locks_size {
                        DoorLocksSize::Small => {
                            set_wall_pixel(tile, 0, 3);
                            set_wall_pixel(tile, 1, 3);
                            set_wall_pixel(tile, 2, 12);
                            set_wall_pixel(tile, 3, color);
                            set_wall_pixel(tile, 4, color);
                            set_wall_pixel(tile, 5, 12);
                            set_wall_pixel(tile, 6, 3);
                            set_wall_pixel(tile, 7, 3);
                            set_air_pixel(tile, 3, 4);
                            set_air_pixel(tile, 4, 4);
                        }
                        DoorLocksSize::Large => {
                            set_wall_pixel(tile, 0, 3);
                            set_wall_pixel(tile, 1, 3);
                            set_wall_pixel(tile, 2, 12);
                            set_wall_pixel(tile, 3, color);
                            set_wall_pixel(tile, 4, color);
                            set_wall_pixel(tile, 5, 12);
                            set_wall_pixel(tile, 6, 3);
                            set_wall_pixel(tile, 7, 3);
                            set_air_pixel(tile, 1, 4);
                            set_air_pixel(tile, 2, color);
                            set_air_pixel(tile, 3, color);
                            set_air_pixel(tile, 4, color);
                            set_air_pixel(tile, 5, color);
                            set_air_pixel(tile, 6, 4);
                            set_deep_pixel(tile, 2, 4);
                            set_deep_pixel(tile, 3, 4);
                            set_deep_pixel(tile, 4, 4);
                            set_deep_pixel(tile, 5, 4);
                        }
                    }
                } else if [Charge, Ice, Wave, Spazer, Plasma].contains(&lock_type) {
                    let color = match lock_type {
                        Charge => 15,
                        Ice => 8,
                        Wave => 7,
                        Spazer => 6,
                        Plasma => 14,
                        _ => panic!("Internal error"),
                    };
                    match self.randomization.settings.other_settings.door_locks_size {
                        DoorLocksSize::Small => {
                            set_wall_pixel(tile, 0, 3);
                            set_wall_pixel(tile, 1, 3);
                            set_wall_pixel(tile, 2, 12);
                            set_wall_pixel(tile, 3, color);
                            set_wall_pixel(tile, 4, color);
                            set_wall_pixel(tile, 5, 12);
                            set_wall_pixel(tile, 6, 3);
                            set_wall_pixel(tile, 7, 3);
                            set_air_pixel(tile, 2, 13);
                            set_air_pixel(tile, 3, 4);
                            set_air_pixel(tile, 4, 4);
                            set_air_pixel(tile, 5, 13);
                        }
                        DoorLocksSize::Large => {
                            set_wall_pixel(tile, 0, 3);
                            set_wall_pixel(tile, 1, 3);
                            set_wall_pixel(tile, 2, 3);
                            set_wall_pixel(tile, 3, color);
                            set_wall_pixel(tile, 4, color);
                            set_wall_pixel(tile, 5, 3);
                            set_wall_pixel(tile, 6, 3);
                            set_wall_pixel(tile, 7, 3);
                            set_air_pixel(tile, 2, 13);
                            set_air_pixel(tile, 3, color);
                            set_air_pixel(tile, 4, color);
                            set_air_pixel(tile, 5, 13);
                            set_deep_pixel(tile, 3, 4);
                            set_deep_pixel(tile, 4, 4);
                        }
                    }
                }
            }
        }
    }

    fn render_tile(&mut self, tile: MapTile) -> Result<[[u8; 8]; 8]> {
        let bg_color = if tile.heated && !self.randomization.settings.other_settings.ultra_low_qol {
            2
        } else {
            1
        };
        let mut data: [[u8; 8]; 8] = [[bg_color; 8]; 8];

        let liquid_colors = if tile.water_level.is_some() {
            (0, 1)
        } else {
            (bg_color, bg_color)
        };
        if let Some(water_level) = tile.water_level {
            if !self.randomization.settings.other_settings.ultra_low_qol {
                let level = (water_level * 8.0).floor() as isize;
                for y in level..8 {
                    for x in 0..8 {
                        if (x + y) % 2 == 0 {
                            data[y as usize][x as usize] = 0;
                        } else {
                            data[y as usize][x as usize] = 1;
                        }
                    }
                }
            }
        };

        let item_color = if tile.faded {
            if tile.heated {
                1
            } else {
                2
            }
        } else {
            13
        };
        match tile.interior {
            MapTileInterior::Empty | MapTileInterior::Event => {}
            MapTileInterior::Item => {
                data[3][3] = item_color;
                data[3][4] = item_color;
                data[4][3] = item_color;
                data[4][4] = item_color;
            }
            MapTileInterior::MediumItem => {
                data[2][3] = item_color;
                data[2][4] = item_color;
                data[3][2] = item_color;
                data[3][3] = liquid_colors.0;
                data[3][4] = liquid_colors.0;
                data[3][5] = item_color;
                data[4][2] = item_color;
                data[4][3] = liquid_colors.0;
                data[4][4] = liquid_colors.0;
                data[4][5] = item_color;
                data[5][3] = item_color;
                data[5][4] = item_color;
            }
            MapTileInterior::AmmoItem => {
                data[2][2] = item_color;
                data[2][5] = item_color;
                data[3][3] = item_color;
                data[3][4] = item_color;
                data[4][3] = item_color;
                data[4][4] = item_color;
                data[5][2] = item_color;
                data[5][5] = item_color;
            }
            MapTileInterior::MajorItem => {
                data[2][3] = item_color;
                data[2][4] = item_color;
                data[3][2] = item_color;
                data[3][3] = item_color;
                data[3][4] = item_color;
                data[3][5] = item_color;
                data[4][2] = item_color;
                data[4][3] = item_color;
                data[4][4] = item_color;
                data[4][5] = item_color;
                data[5][3] = item_color;
                data[5][4] = item_color;
            }
            MapTileInterior::DoubleItem => {
                panic!("Unreplaced DoubleItem");
                // data[2][2] = item_color;
                // data[3][3] = item_color;
                // data[4][4] = item_color;
                // data[5][5] = item_color;
            }
            MapTileInterior::HiddenItem => {
                panic!("Unreplaced HiddenItem");
                // data[3][3] = item_color;
                // data[4][4] = item_color;
            }
            MapTileInterior::ElevatorPlatformLow => {
                // Use white instead of red for elevator platform:
                data[5][3] = 3;
                data[5][4] = 3;
            }
            MapTileInterior::ElevatorPlatformHigh => {
                data[2][3] = 3;
                data[2][4] = 3;
            }
            MapTileInterior::SaveStation => {
                update_tile(
                    &mut data,
                    3,
                    &vec![
                        (0, 0),
                        (1, 0),
                        (2, 0),
                        (3, 0),
                        (4, 0),
                        (5, 0),
                        (6, 0),
                        (7, 0),
                        (0, 1),
                        (1, 1),
                        (7, 1),
                        (0, 2),
                        (4, 2),
                        (5, 2),
                        (6, 2),
                        (7, 2),
                        (0, 3),
                        (6, 3),
                        (7, 3),
                        (0, 4),
                        (1, 4),
                        (7, 4),
                        (0, 5),
                        (1, 5),
                        (2, 5),
                        (3, 5),
                        (7, 5),
                        (0, 6),
                        (6, 6),
                        (7, 6),
                        (0, 7),
                        (1, 7),
                        (2, 7),
                        (3, 7),
                        (4, 7),
                        (5, 7),
                        (6, 7),
                        (7, 7),
                    ],
                );
            }
            MapTileInterior::EnergyRefill => {
                if self.randomization.settings.other_settings.ultra_low_qol {
                    data[3][3] = item_color;
                    data[3][4] = item_color;
                    data[4][3] = item_color;
                    data[4][4] = item_color;
                } else {
                    update_tile(
                        &mut data,
                        3,
                        &vec![
                            (0, 0),
                            (1, 0),
                            (2, 0),
                            (3, 0),
                            (4, 0),
                            (5, 0),
                            (6, 0),
                            (7, 0),
                            (0, 1),
                            (1, 1),
                            (2, 1),
                            (5, 1),
                            (6, 1),
                            (7, 1),
                            (0, 2),
                            (1, 2),
                            (2, 2),
                            (5, 2),
                            (6, 2),
                            (7, 2),
                            (0, 3),
                            (7, 3),
                            (0, 4),
                            (7, 4),
                            (0, 5),
                            (1, 5),
                            (2, 5),
                            (5, 5),
                            (6, 5),
                            (7, 5),
                            (0, 6),
                            (1, 6),
                            (2, 6),
                            (5, 6),
                            (6, 6),
                            (7, 6),
                            (0, 7),
                            (1, 7),
                            (2, 7),
                            (3, 7),
                            (4, 7),
                            (5, 7),
                            (6, 7),
                            (7, 7),
                        ],
                    );
                }
            }
            MapTileInterior::AmmoRefill => {
                if self.randomization.settings.other_settings.ultra_low_qol {
                    data[3][3] = item_color;
                    data[3][4] = item_color;
                    data[4][3] = item_color;
                    data[4][4] = item_color;
                } else {
                    update_tile(
                        &mut data,
                        3,
                        &vec![
                            (0, 0),
                            (1, 0),
                            (2, 0),
                            (3, 0),
                            (4, 0),
                            (5, 0),
                            (6, 0),
                            (7, 0),
                            (0, 1),
                            (1, 1),
                            (2, 1),
                            (5, 1),
                            (6, 1),
                            (7, 1),
                            (0, 2),
                            (1, 2),
                            (6, 2),
                            (7, 2),
                            (0, 3),
                            (1, 3),
                            (3, 3),
                            (4, 3),
                            (6, 3),
                            (7, 3),
                            (0, 4),
                            (1, 4),
                            (6, 4),
                            (7, 4),
                            (0, 5),
                            (7, 5),
                            (0, 6),
                            (2, 6),
                            (5, 6),
                            (7, 6),
                            (0, 7),
                            (1, 7),
                            (2, 7),
                            (3, 7),
                            (4, 7),
                            (5, 7),
                            (6, 7),
                            (7, 7),
                        ],
                    );
                }
            }
            MapTileInterior::DoubleRefill | MapTileInterior::Ship => {
                if self.randomization.settings.other_settings.ultra_low_qol {
                    data[3][3] = item_color;
                    data[3][4] = item_color;
                    data[4][3] = item_color;
                    data[4][4] = item_color;
                } else {
                    update_tile(
                        &mut data,
                        3,
                        &vec![
                            (0, 0),
                            (1, 0),
                            (2, 0),
                            (3, 0),
                            (4, 0),
                            (5, 0),
                            (6, 0),
                            (7, 0),
                            (0, 1),
                            (2, 1),
                            (5, 1),
                            (7, 1),
                            (0, 2),
                            (1, 2),
                            (2, 2),
                            (5, 2),
                            (6, 2),
                            (7, 2),
                            (0, 3),
                            (7, 3),
                            (0, 4),
                            (7, 4),
                            (0, 5),
                            (1, 5),
                            (2, 5),
                            (5, 5),
                            (6, 5),
                            (7, 5),
                            (0, 6),
                            (2, 6),
                            (5, 6),
                            (7, 6),
                            (0, 7),
                            (1, 7),
                            (2, 7),
                            (3, 7),
                            (4, 7),
                            (5, 7),
                            (6, 7),
                            (7, 7),
                        ],
                    );
                }
            }
            MapTileInterior::Objective => {
                update_tile(
                    &mut data,
                    3,
                    &vec![
                        (0, 0),
                        (1, 0),
                        (2, 0),
                        (3, 0),
                        (4, 0),
                        (5, 0),
                        (6, 0),
                        (7, 0),
                        (0, 1),
                        (3, 1),
                        (4, 1),
                        (7, 1),
                        (0, 2),
                        (7, 2),
                        (0, 3),
                        (1, 3),
                        (6, 3),
                        (7, 3),
                        (0, 4),
                        (1, 4),
                        (6, 4),
                        (7, 4),
                        (0, 5),
                        (7, 5),
                        (0, 6),
                        (3, 6),
                        (4, 6),
                        (7, 6),
                        (0, 7),
                        (1, 7),
                        (2, 7),
                        (3, 7),
                        (4, 7),
                        (5, 7),
                        (6, 7),
                        (7, 7),
                    ],
                );
            }
            MapTileInterior::MapStation => {
                if self.randomization.settings.other_settings.ultra_low_qol {
                    data[3][3] = item_color;
                    data[3][4] = item_color;
                    data[4][3] = item_color;
                    data[4][4] = item_color;
                } else {
                    update_tile(
                        &mut data,
                        3,
                        &vec![
                            (0, 0),
                            (1, 0),
                            (2, 0),
                            (3, 0),
                            (4, 0),
                            (5, 0),
                            (6, 0),
                            (7, 0),
                            (0, 1),
                            (7, 1),
                            (0, 2),
                            (2, 2),
                            (3, 2),
                            (4, 2),
                            (5, 2),
                            (7, 2),
                            (0, 3),
                            (2, 3),
                            (5, 3),
                            (7, 3),
                            (0, 4),
                            (2, 4),
                            (5, 4),
                            (7, 4),
                            (0, 5),
                            (2, 5),
                            (3, 5),
                            (4, 5),
                            (5, 5),
                            (7, 5),
                            (0, 6),
                            (7, 6),
                            (0, 7),
                            (1, 7),
                            (2, 7),
                            (3, 7),
                            (4, 7),
                            (5, 7),
                            (6, 7),
                            (7, 7),
                        ],
                    );
                }
            }
        }

        let apply_heat = |d: [[u8; 8]; 8]| {
            if tile.heated && !self.randomization.settings.other_settings.ultra_low_qol {
                d.map(|row| row.map(|c| if c == 1 { 2 } else { c }))
            } else {
                d
            }
        };
        match tile.special_type {
            Some(MapTileSpecialType::AreaTransition(area_idx, dir)) => {
                if self
                    .randomization
                    .settings
                    .other_settings
                    .transition_letters
                {
                    match area_idx {
                        0 => {
                            data = [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 3, 3, 3, 3, 0, 0],
                                [0, 3, 3, 0, 0, 3, 3, 0],
                                [0, 3, 3, 0, 0, 0, 0, 0],
                                [0, 3, 3, 0, 0, 0, 0, 0],
                                [0, 3, 3, 0, 0, 3, 3, 0],
                                [0, 0, 3, 3, 3, 3, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ];
                        }
                        1 => {
                            data = [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 3, 3, 3, 3, 3, 0, 0],
                                [0, 3, 3, 0, 0, 3, 3, 0],
                                [0, 3, 3, 3, 3, 3, 0, 0],
                                [0, 3, 3, 0, 0, 3, 3, 0],
                                [0, 3, 3, 0, 0, 3, 3, 0],
                                [0, 3, 3, 3, 3, 3, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ];
                        }
                        2 => {
                            data = [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 3, 3, 0, 0, 0, 3, 0],
                                [0, 3, 3, 3, 0, 0, 3, 0],
                                [0, 3, 3, 3, 3, 0, 3, 0],
                                [0, 3, 3, 0, 3, 3, 3, 0],
                                [0, 3, 3, 0, 0, 3, 3, 0],
                                [0, 3, 3, 0, 0, 0, 3, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ];
                        }
                        3 => {
                            data = [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 3, 3, 0, 0, 0, 3, 0],
                                [0, 3, 3, 0, 0, 0, 3, 0],
                                [0, 3, 3, 0, 3, 0, 3, 0],
                                [0, 3, 3, 3, 3, 3, 3, 0],
                                [0, 3, 3, 3, 0, 3, 3, 0],
                                [0, 3, 3, 0, 0, 0, 3, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ];
                        }
                        4 => {
                            data = [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 3, 3, 0, 0, 0, 3, 0],
                                [0, 3, 3, 3, 0, 3, 3, 0],
                                [0, 3, 3, 3, 3, 3, 3, 0],
                                [0, 3, 3, 0, 3, 0, 3, 0],
                                [0, 3, 3, 0, 0, 0, 3, 0],
                                [0, 3, 3, 0, 0, 0, 3, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ];
                        }
                        5 => {
                            data = [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 3, 3, 3, 3, 3, 3, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ];
                        }
                        _ => panic!("Unexpected area {}", area_idx),
                    }
                } else {
                    match dir {
                        Direction::Right => {
                            data = [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 3, 0, 0],
                                [0, 0, 0, 0, 0, 3, 3, 0],
                                [0, 3, 3, 3, 3, 3, 3, 3],
                                [0, 3, 3, 3, 3, 3, 3, 3],
                                [0, 0, 0, 0, 0, 3, 3, 0],
                                [0, 0, 0, 0, 0, 3, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ];
                        }
                        Direction::Left => {
                            data = [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 3, 0, 0, 0, 0, 0],
                                [0, 3, 3, 0, 0, 0, 0, 0],
                                [3, 3, 3, 3, 3, 3, 3, 0],
                                [3, 3, 3, 3, 3, 3, 3, 0],
                                [0, 3, 3, 0, 0, 0, 0, 0],
                                [0, 0, 3, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ];
                        }
                        Direction::Down => {
                            data = [
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 3, 3, 3, 3, 3, 3, 0],
                                [0, 0, 3, 3, 3, 3, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                            ];
                        }
                        Direction::Up => {
                            data = [
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 3, 3, 3, 3, 0, 0],
                                [0, 3, 3, 3, 3, 3, 3, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 3, 3, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                            ];
                        }
                    }
                }
                // Set up arrows of different colors (one per area):
                let area_arrow_colors: Vec<usize> = vec![
                    11, // Crateria: purple (defined above)
                    14, // Brinstar: green (defined above)
                    10, // Norfair: red (defined above)
                    9,  // Wrecked Ship: yellow (defined above)
                    8,  // Maridia: blue (defined above)
                    6,  // Tourian: orange
                ];

                let color_number = area_arrow_colors[area_idx] as u8;
                data = data.map(|row| row.map(|c| if c == 3 { color_number } else { c }));
            }
            Some(MapTileSpecialType::Black) => {
                data = [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 1, 0],
                ];
            }
            Some(MapTileSpecialType::Elevator | MapTileSpecialType::Tube) => {
                data = apply_heat([
                    [0, 3, 1, 4, 4, 1, 3, 0],
                    [0, 3, 4, 4, 4, 4, 3, 0],
                    [0, 3, 1, 4, 4, 1, 3, 0],
                    [0, 3, 4, 4, 4, 4, 3, 0],
                    [0, 3, 1, 4, 4, 1, 3, 0],
                    [0, 3, 4, 4, 4, 4, 3, 0],
                    [0, 3, 1, 4, 4, 1, 3, 0],
                    [0, 3, 4, 4, 4, 4, 3, 0],
                ]);
            }
            Some(MapTileSpecialType::SlopeUpFloorLow) => {
                data = apply_heat([
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 3, 3],
                    [1, 1, 1, 1, 3, 3, 0, 0],
                    [1, 1, 3, 3, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 0, 0, 0],
                ]);
            }
            Some(MapTileSpecialType::SlopeUpFloorHigh) => {
                data = apply_heat([
                    [1, 1, 1, 1, 1, 1, 3, 3],
                    [1, 1, 1, 1, 3, 3, 0, 0],
                    [1, 1, 3, 3, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]);
            }
            Some(MapTileSpecialType::SlopeUpCeilingLow) => {
                data = apply_heat([
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 3, 3],
                    [0, 0, 0, 0, 3, 3, 1, 1],
                    [0, 0, 3, 3, 1, 1, 1, 1],
                    [3, 3, 1, 1, 1, 1, 1, 1],
                ]);
            }
            Some(MapTileSpecialType::SlopeUpCeilingHigh) => {
                data = apply_heat([
                    [0, 0, 0, 0, 0, 0, 3, 3],
                    [0, 0, 0, 0, 3, 3, 1, 1],
                    [0, 0, 3, 3, 1, 1, 1, 1],
                    [3, 3, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]);
            }
            Some(MapTileSpecialType::SlopeDownCeilingHigh) => {
                data = apply_heat([
                    [3, 3, 0, 0, 0, 0, 0, 0],
                    [1, 1, 3, 3, 0, 0, 0, 0],
                    [1, 1, 1, 1, 3, 3, 0, 0],
                    [1, 1, 1, 1, 1, 1, 3, 3],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]);
            }
            Some(MapTileSpecialType::SlopeDownCeilingLow) => {
                data = apply_heat([
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 0, 0, 0],
                    [1, 1, 3, 3, 0, 0, 0, 0],
                    [1, 1, 1, 1, 3, 3, 0, 0],
                    [1, 1, 1, 1, 1, 1, 3, 3],
                ]);
            }
            Some(MapTileSpecialType::SlopeDownFloorHigh) => {
                data = apply_heat([
                    [3, 3, 1, 1, 1, 1, 1, 1],
                    [0, 0, 3, 3, 1, 1, 1, 1],
                    [0, 0, 0, 0, 3, 3, 1, 1],
                    [0, 0, 0, 0, 0, 0, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]);
            }
            Some(MapTileSpecialType::SlopeDownFloorLow) => {
                data = apply_heat([
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [3, 3, 1, 1, 1, 1, 1, 1],
                    [0, 0, 3, 3, 1, 1, 1, 1],
                    [0, 0, 0, 0, 3, 3, 1, 1],
                    [0, 0, 0, 0, 0, 0, 3, 3],
                ]);
            }
            None => {}
        }

        if tile.special_type.is_some()
            || (!self.randomization.settings.other_settings.ultra_low_qol
                && [
                    MapTileInterior::AmmoRefill,
                    MapTileInterior::EnergyRefill,
                    MapTileInterior::DoubleRefill,
                    MapTileInterior::Ship,
                    MapTileInterior::SaveStation,
                    MapTileInterior::MapStation,
                    MapTileInterior::Objective,
                ]
                .contains(&tile.interior))
        {
            // Skip drawing door & wall edges in special tiles
        } else {
            self.draw_edge(TileSide::Top, tile.top, &mut data);
            self.draw_edge(TileSide::Bottom, tile.bottom, &mut data);
            self.draw_edge(TileSide::Left, tile.left, &mut data);
            self.draw_edge(TileSide::Right, tile.right, &mut data);
        }
        Ok(data)
    }

    fn indicate_obj_tiles(&mut self, objective: &Objective) -> Result<()> {
        use Objective::*;

        match objective {
            Kraid => {
                self.get_room_tile(84, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(84, 1, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(84, 0, 1).interior = MapTileInterior::Objective;
                self.get_room_tile(84, 1, 1).interior = MapTileInterior::Objective;
            }
            Phantoon => {
                self.get_room_tile(158, 0, 0).interior = MapTileInterior::Objective;
            }
            Draygon => {
                self.get_room_tile(193, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(193, 1, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(193, 0, 1).interior = MapTileInterior::Objective;
                self.get_room_tile(193, 1, 1).interior = MapTileInterior::Objective;
            }
            Ridley => {
                self.get_room_tile(142, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(142, 0, 1).interior = MapTileInterior::Objective;
            }
            SporeSpawn => {
                self.get_room_tile(57, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(57, 0, 1).interior = MapTileInterior::Objective;
                self.get_room_tile(57, 0, 2).interior = MapTileInterior::Objective;
            }
            Crocomire => {
                self.get_room_tile(122, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(122, 1, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(122, 2, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(122, 3, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(122, 4, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(122, 5, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(122, 6, 0).interior = MapTileInterior::Objective;
                // We don't mark the last tile, so the item can still be visible.
            }
            Botwoon => {
                self.get_room_tile(185, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(185, 1, 0).interior = MapTileInterior::Objective;
            }
            GoldenTorizo => {
                self.get_room_tile(150, 0, 1).interior = MapTileInterior::Objective;
                self.get_room_tile(150, 1, 1).interior = MapTileInterior::Objective;
                // We don't mark the top row of tiles, so the items can still be visible.
            }
            MetroidRoom1 => {
                self.get_room_tile(226, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(226, 1, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(226, 2, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(226, 3, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(226, 4, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(226, 5, 0).interior = MapTileInterior::Objective;
            }
            MetroidRoom2 => {
                self.get_room_tile(227, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(227, 0, 1).interior = MapTileInterior::Objective;
            }
            MetroidRoom3 => {
                self.get_room_tile(228, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(228, 1, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(228, 2, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(228, 3, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(228, 4, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(228, 5, 0).interior = MapTileInterior::Objective;
            }
            MetroidRoom4 => {
                self.get_room_tile(229, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(229, 0, 1).interior = MapTileInterior::Objective;
            }
            BombTorizo => {
                self.get_room_tile(19, 0, 0).interior = MapTileInterior::Objective;
            }
            BowlingStatue => {
                self.get_room_tile(161, 4, 1).interior = MapTileInterior::Objective;
            }
            AcidChozoStatue => {
                self.get_room_tile(149, 0, 0).interior = MapTileInterior::Objective;
            }
            PitRoom => {
                self.get_room_tile(12, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(12, 1, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(12, 2, 0).interior = MapTileInterior::Objective;
            }
            BabyKraidRoom => {
                self.get_room_tile(82, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(82, 1, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(82, 2, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(82, 3, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(82, 4, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(82, 5, 0).interior = MapTileInterior::Objective;
            }
            PlasmaRoom => {
                self.get_room_tile(219, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(219, 1, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(219, 0, 1).interior = MapTileInterior::Objective;
                self.get_room_tile(219, 1, 1).interior = MapTileInterior::Objective;
                self.get_room_tile(219, 0, 2).interior = MapTileInterior::Objective;
            }
            MetalPiratesRoom => {
                self.get_room_tile(139, 0, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(139, 1, 0).interior = MapTileInterior::Objective;
                self.get_room_tile(139, 2, 0).interior = MapTileInterior::Objective;
            }
        }
        Ok(())
    }

    fn indicate_objective_tiles(&mut self) -> Result<()> {
        for obj in self.randomization.objectives.iter() {
            self.indicate_obj_tiles(obj)?;
        }

        self.get_room_tile(238, 0, 0).interior = MapTileInterior::Objective;
        self.get_room_tile(238, 1, 0).interior = MapTileInterior::Objective;
        self.get_room_tile(238, 2, 0).interior = MapTileInterior::Objective;
        self.get_room_tile(238, 3, 0).interior = MapTileInterior::Objective;

        Ok(())
    }

    fn indicate_gray_doors(&mut self) -> Result<()> {
        // Indicate gray doors by a gray bubble with black border. Some of these may be later overwritten
        // by an X depending on the objective setting.
        let gray_door = MapTileEdge::LockedDoor(DoorLockType::Gray);

        // Pirate rooms:
        // Pit Room:
        self.get_room_tile(12, 0, 0).left = gray_door;
        self.get_room_tile(12, 2, 0).right = gray_door;
        // Baby Kraid Room:
        self.get_room_tile(82, 0, 0).left = gray_door;
        self.get_room_tile(82, 5, 0).right = gray_door;
        // Plasma Room:
        self.get_room_tile(219, 0, 0).left = gray_door;
        // Metal Pirates Room:
        self.get_room_tile(139, 0, 0).left = gray_door;
        self.get_room_tile(139, 2, 0).right = gray_door;

        // Boss rooms:
        // Kraid Room:
        self.get_room_tile(84, 0, 1).left = gray_door;
        self.get_room_tile(84, 1, 1).right = gray_door;
        // Phantoon's Room:
        self.get_room_tile(158, 0, 0).left = gray_door;
        // Draygon's Room:
        self.get_room_tile(193, 0, 1).left = gray_door;
        self.get_room_tile(193, 1, 0).right = gray_door;
        // Ridley's Room:
        self.get_room_tile(142, 0, 0).right = gray_door;
        self.get_room_tile(142, 0, 1).left = gray_door;

        // Miniboss rooms:
        // Bomb Torizo Room:
        self.get_room_tile(19, 0, 0).left = gray_door;
        // Spore Spawn Room:
        self.get_room_tile(57, 0, 2).bottom = gray_door;
        // Crocomire's Room:
        self.get_room_tile(122, 3, 0).top = gray_door;
        // Botwoon's Room:
        self.get_room_tile(185, 0, 0).left = gray_door;
        // Golden Torizo's Room:
        self.get_room_tile(150, 1, 1).right = gray_door;
        Ok(())
    }

    fn indicate_locked_doors(&mut self) -> Result<()> {
        for (i, locked_door) in self
            .randomization
            .locked_door_data
            .locked_doors
            .iter()
            .enumerate()
        {
            let mut ptr_pairs = vec![locked_door.src_ptr_pair];
            if locked_door.bidirectional {
                ptr_pairs.push(locked_door.dst_ptr_pair);
            }
            for ptr_pair in ptr_pairs {
                let (room_idx, door_idx) =
                    self.game_data.room_and_door_idxs_by_door_ptr_pair[&ptr_pair];
                let area = self.randomization.map.area[room_idx];
                let room_geom = &self.game_data.room_geometry[room_idx];
                let door = &room_geom.doors[door_idx];
                let room_id = self.game_data.room_id_by_ptr[&room_geom.rom_address];
                let lock_type = match locked_door.door_type {
                    DoorType::Blue => panic!("unexpected blue door lock"),
                    DoorType::Gray => panic!("unexpected gray door lock"),
                    DoorType::Red => DoorLockType::Red,
                    DoorType::Green => DoorLockType::Green,
                    DoorType::Yellow => DoorLockType::Yellow,
                    DoorType::Beam(beam) => match beam {
                        BeamType::Charge => DoorLockType::Charge,
                        BeamType::Ice => DoorLockType::Ice,
                        BeamType::Wave => DoorLockType::Wave,
                        BeamType::Spazer => DoorLockType::Spazer,
                        BeamType::Plasma => DoorLockType::Plasma,
                    },
                };
                let edge = MapTileEdge::LockedDoor(lock_type);
                let tile = self.get_room_tile(room_id, door.x as isize, door.y as isize);
                let mut new_tile = tile.clone();
                match door.direction.as_str() {
                    "left" => {
                        new_tile.left = edge;
                    }
                    "right" => {
                        new_tile.right = edge;
                    }
                    "up" => {
                        new_tile.top = edge;
                    }
                    "down" => {
                        new_tile.bottom = edge;
                    }
                    _ => panic!("Unexpected door direction: {:?}", door.direction),
                }

                // Here, to make doors disappear once unlocked, we're (slightly awkwardly) reusing the mechanism for
                // making item dots disappear. Door bits are stored at $D8B0, which is 512 bits after $D870 where
                // the item bits start.
                let item_idx = self.locked_door_state_indices[i] + 512;
                self.dynamic_tile_data[area].push((item_idx, room_id, new_tile));
            }
        }
        Ok(())
    }

    fn add_door_arrow(
        &mut self,
        room_idx: usize,
        door: &RoomGeometryDoor,
        other_area: usize,
    ) -> Result<()> {
        let dir = &door.direction;
        let room_ptr = self.game_data.room_geometry[room_idx].rom_address;
        let room_id = self.game_data.room_id_by_ptr[&room_ptr];
        let x = door.x as isize;
        let y = door.y as isize;

        let coords = match dir.as_str() {
            "right" => (x + 1, y),
            "left" => (x - 1, y),
            "down" => (x, y + 1),
            "up" => (x, y - 1),
            _ => bail!("Unrecognized door direction: {dir}"),
        };
        let direction = match dir.as_str() {
            "right" => Direction::Right,
            "left" => Direction::Left,
            "down" => Direction::Down,
            "up" => Direction::Up,
            _ => bail!("Unrecognized door direction: {dir}"),
        };

        let mut tile = MapTile::default();
        tile.special_type = Some(MapTileSpecialType::AreaTransition(other_area, direction));
        self.set_room_tile(room_id, coords.0, coords.1, tile);
        Ok(())
    }

    fn add_cross_area_arrows(&mut self) -> Result<()> {
        // Replace colors to palette used for map tiles in the pause menu, for drawing arrows marking
        // cross-area connections:
        fn rgb(r: u16, g: u16, b: u16) -> u16 {
            (b << 10) | (g << 5) | r
        }

        let extended_map_palette: Vec<(u8, u16)> = vec![
            (14, rgb(7, 31, 7)),   // Brinstar green (and green doors)
            (10, rgb(29, 0, 0)),   // Norfair red
            (8, rgb(4, 13, 31)),   // Maridia blue
            (9, rgb(23, 24, 9)),   // Wrecked Ship yellow
            (11, rgb(20, 3, 31)),  // Crateria purple
            (6, rgb(29, 15, 0)),   // Tourian, (and orange doors)
            (15, rgb(18, 12, 14)), // Gray door
            (7, rgb(27, 2, 27)),   // Red (pink) door
            (12, rgb(0, 0, 0)),    // Black (door lock shadows covering wall)
            (13, rgb(31, 31, 31)), // White (item dots)
        ];
        // Dotted grid lines
        let i = 12;
        let color = rgb(8, 8, 8);
        self.rom
            .write_u16(snes2pc(0xB6F000) + 2 * (0x40 + i), color as isize)?;

        for &(i, color) in &extended_map_palette {
            self.rom
                .write_u16(snes2pc(0xB6F000) + 2 * (0x20 + i as usize), color as isize)?;
            self.rom
                .write_u16(snes2pc(0xB6F000) + 2 * (0x60 + i as usize), color as isize)?;
        }

        // In partially revealed palette, hide room interior, item dots, and door locks setting them all to black:
        for i in [1, 2, 4, 6, 7, 8, 13, 14, 15] {
            self.rom.write_u16(
                snes2pc(0xB6F000) + 2 * (0x30 + i as usize),
                rgb(0, 0, 0) as isize,
            )?;
        }
        // In partially revealed palette, show walls/passages (as gray), and eliminate the door lock shadows covering walls:
        for i in [3, 12] {
            self.rom.write_u16(
                snes2pc(0xB6F000) + 2 * (0x30 + i as usize),
                rgb(16, 16, 16) as isize,
            )?;
        }

        for (src_ptr_pair, dst_ptr_pair, _) in &self.map.doors {
            let (src_room_idx, src_door_idx) =
                self.game_data.room_and_door_idxs_by_door_ptr_pair[src_ptr_pair];
            let (dst_room_idx, dst_door_idx) =
                self.game_data.room_and_door_idxs_by_door_ptr_pair[dst_ptr_pair];
            let src_area = self.map.area[src_room_idx];
            let dst_area = self.map.area[dst_room_idx];
            if src_area != dst_area {
                self.add_door_arrow(
                    src_room_idx,
                    &self.game_data.room_geometry[src_room_idx].doors[src_door_idx],
                    dst_area,
                )?;
                self.add_door_arrow(
                    dst_room_idx,
                    &self.game_data.room_geometry[dst_room_idx].doors[dst_door_idx],
                    src_area,
                )?;
            }
        }
        Ok(())
    }

    fn fix_pause_palettes(&mut self) -> Result<()> {
        // Much of the static content in the pause menu uses palette 2. We change these to use palette 4 instead,
        // since palette 2 is used for explored map tiles. This allows us to use more colors in palette 2
        // (e.g. for area transition arrows) without interfering with existing pause menu content.

        // Copy palette 2 over to palette 4:
        for i in 0..16 {
            let color = self
                .rom
                .read_u16(snes2pc(0xB6F000) + 2 * (0x20 as usize + i as usize))?;
            self.rom.write_u16(
                snes2pc(0xB6F000) + 2 * (0x40 as usize + i as usize),
                color as isize,
            )?;
        }

        // Substitute palette 2 with palette 4 in pause tilemaps:
        let map_range = snes2pc(0xB6E000)..snes2pc(0xB6E800);
        let equipment_range = snes2pc(0x82BF06)..snes2pc(0x82C02C);
        for addr in map_range.chain(equipment_range).step_by(2) {
            let mut word = self.rom.read_u16(addr)?;
            let pal = (word >> 10) & 7;
            if pal == 2 {
                word = (word & !0x1C00) | (4 << 10);
            }
            self.rom.write_u16(addr, word)?;
        }

        Ok(())
    }

    fn set_initial_map(&mut self) -> Result<()> {
        let revealed_addr = snes2pc(0xB5F000);
        let partially_revealed_addr = snes2pc(0xB5F800);
        let area_seen_addr = snes2pc(0xB5F600);
        match self.randomization.settings.other_settings.maps_revealed {
            MapsRevealed::Full => {
                self.rom.write_n(revealed_addr, &vec![0xFF; 0x600])?; // whole map revealed bits: true
                self.rom
                    .write_n(partially_revealed_addr, &vec![0xFF; 0x600])?; // whole map partially revealed bits: true
                self.rom.write_u16(area_seen_addr, 0x003F)?; // area seen bits: true (for pause map area switching)
            }
            MapsRevealed::Partial => {
                self.rom.write_n(revealed_addr, &vec![0; 0x600])?; // whole map revealed bits: false
                self.rom
                    .write_n(partially_revealed_addr, &vec![0xFF; 0x600])?; // whole map partially revealed bits: true
                self.rom.write_u16(area_seen_addr, 0x003F)?; // area seen bits: true (for pause map area switching)

                // Show area-transition markers (arrows or letters) as revealed:
                for &(area, x, y) in &self.transition_tile_coords {
                    let (offset, bitmask) = xy_to_explored_bit_ptr(x, y);
                    let ptr_revealed = revealed_addr + area * 0x100 + offset as usize;
                    self.rom.write_u8(
                        ptr_revealed,
                        self.rom.read_u8(ptr_revealed)? | bitmask as isize,
                    )?;
                }
            }
            MapsRevealed::No => {
                self.rom.write_n(revealed_addr, &vec![0; 0x600])?;
                self.rom.write_n(partially_revealed_addr, &vec![0; 0x600])?;
                self.rom.write_u16(area_seen_addr, 0x0000)?;
            }
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .mark_map_stations
        {
            for (room_idx, room) in self.game_data.room_geometry.iter().enumerate() {
                if !room.name.contains(" Map Room") {
                    continue;
                }
                let area = self.map.area[room_idx];
                let x = self.rom.read_u8(room.rom_address + 2)?;
                let y = self.rom.read_u8(room.rom_address + 3)?;
                let (offset, bitmask) = xy_to_explored_bit_ptr(x, y);

                let ptr_revealed = revealed_addr + area * 0x100 + offset as usize;
                self.rom.write_u8(
                    ptr_revealed,
                    self.rom.read_u8(ptr_revealed)? | bitmask as isize,
                )?;

                let ptr_partial = partially_revealed_addr + area * 0x100 + offset as usize;
                self.rom.write_u8(
                    ptr_partial,
                    self.rom.read_u8(ptr_partial)? | bitmask as isize,
                )?;
            }
        }
        Ok(())
    }

    fn setup_special_door_reveal(&mut self) -> Result<()> {
        // If save/refill rooms with 2 doors, a common problem is that players enter it and leave,
        // and then when looking at the map later, don't remember that there's another room behind it.
        // To avoid this, when entering on of these rooms, we do a "partial reveal" on just the door
        // of the neighboring rooms.
        let room_ids = vec![
            302, // Frog Savestation
            190, // Draygon Save Room
            308, // Nutella Refill
        ];
        let mut table_addr = snes2pc(0x85A180);
        let partial_revealed_bits_base = 0x2700;
        let tilemap_base = 0x4000;
        let palette = 0x0800;
        let left_door_tile_idx = 0x20;

        for room_id in room_ids {
            let room_ptr = self.game_data.room_ptr_by_id[&room_id];
            let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
            let room = &self.game_data.room_geometry[room_idx];
            let area = self.map.area[room_idx] as isize;
            let room_x = self.rom.read_u8(room.rom_address + 2)?;
            let room_y = self.rom.read_u8(room.rom_address + 3)?;

            let (trigger_offset, trigger_bitmask) = xy_to_explored_bit_ptr(room_x, room_y);
            let trigger_addr = partial_revealed_bits_base + 0x100 * area + trigger_offset;
            let (left_offset, left_bitmask) = xy_to_explored_bit_ptr(room_x - 1, room_y);
            let left_revealed_addr = partial_revealed_bits_base + 0x100 * area + left_offset;
            let left_tilemap_offset = xy_to_map_offset(room_x - 1, room_y);
            let (right_offset, right_bitmask) = xy_to_explored_bit_ptr(room_x + 1, room_y);
            let right_revealed_addr = partial_revealed_bits_base + 0x100 * area + right_offset;
            let right_tilemap_offset = xy_to_map_offset(room_x + 1, room_y);

            self.rom.write_u16(table_addr, area)?;
            self.rom.write_u16(table_addr + 2, trigger_addr)?;
            self.rom
                .write_u16(table_addr + 4, trigger_bitmask as isize)?;
            self.rom.write_u16(table_addr + 6, left_revealed_addr)?;
            self.rom.write_u16(table_addr + 8, left_bitmask as isize)?;
            self.rom
                .write_u16(table_addr + 10, tilemap_base + left_tilemap_offset)?;
            self.rom
                .write_u16(table_addr + 12, left_door_tile_idx | palette | 0x4000)?; // 0x4000: horizontal flip
            table_addr += 14;

            self.rom.write_u16(table_addr, area)?;
            self.rom.write_u16(table_addr + 2, trigger_addr)?;
            self.rom
                .write_u16(table_addr + 4, trigger_bitmask as isize)?;
            self.rom.write_u16(table_addr + 6, right_revealed_addr)?;
            self.rom.write_u16(table_addr + 8, right_bitmask as isize)?;
            self.rom
                .write_u16(table_addr + 10, tilemap_base + right_tilemap_offset)?;
            self.rom
                .write_u16(table_addr + 12, left_door_tile_idx | palette)?;
            table_addr += 14;
        }
        self.rom.write_u16(table_addr, 0xFFFF)?;
        Ok(())
    }

    fn set_map_activation_behavior(&mut self) -> Result<()> {
        match self
            .randomization
            .settings
            .other_settings
            .map_station_reveal
        {
            MapStationReveal::Partial => {
                self.rom.write_u16(snes2pc(0x90F700), 0xFFFF)?;
            }
            MapStationReveal::Full => {}
        }
        Ok(())
    }

    fn write_dynamic_tile_data(
        &mut self,
        area_data: &[Vec<(ItemIdx, RoomId, MapTile)>],
    ) -> Result<()> {
        // Write per-area item listings, to be used by the patch `item_dots_disappear.asm`.
        let base_ptr = 0x83B000;
        let mut data_ptr = base_ptr + 24;
        let interior_priority = [
            MapTileInterior::Empty,
            MapTileInterior::Event,
            MapTileInterior::ElevatorPlatformLow,
            MapTileInterior::ElevatorPlatformHigh,
            MapTileInterior::Item,
            MapTileInterior::AmmoItem,
            MapTileInterior::MediumItem,
            MapTileInterior::MajorItem,
        ];
        for (area_idx, data) in area_data.iter().enumerate() {
            self.rom.write_u16(
                snes2pc(base_ptr + area_idx * 2),
                (data_ptr & 0xFFFF) as isize,
            )?;
            self.rom
                .write_u16(snes2pc(base_ptr + 12 + area_idx * 2), data.len() as isize)?;
            let data_start = data_ptr;
            for &(_, room_id, ref tile) in data {
                if !interior_priority.contains(&tile.interior) {
                    panic!(
                        "In room_id={room_id}, unexpected dynamic tile interior: {:?}",
                        tile
                    );
                }
            }
            for &interior in &interior_priority {
                for &(item_idx, room_id, ref tile) in data {
                    if tile.interior != interior {
                        continue;
                    }
                    self.rom
                        .write_u8(snes2pc(data_ptr), (item_idx as isize) >> 3)?; // item byte index
                    self.rom
                        .write_u8(snes2pc(data_ptr + 1), 1 << ((item_idx as isize) & 7))?; // item bitmask
                    let word = self.index_tile(area_idx, tile.clone(), None)?;

                    let (_, x, y) = self.get_room_coords(
                        room_id,
                        tile.coords.0 as isize,
                        tile.coords.1 as isize,
                    );
                    let local_x = x - self.area_offset_x[area_idx];
                    let local_y = y - self.area_offset_y[area_idx];
                    let offset = xy_to_map_offset(local_x, local_y);
                    self.rom.write_u16(snes2pc(data_ptr + 2), offset as isize)?; // tilemap offset
                    self.rom.write_u16(snes2pc(data_ptr + 4), word as isize)?; // tilemap word
                    data_ptr += 6;
                }
            }
            assert_eq!(data_ptr, data_start + 6 * data.len());
        }
        assert!(data_ptr <= 0x83B600);
        Ok(())
    }

    fn indicate_items(&mut self) -> Result<()> {
        let markers = self
            .randomization
            .settings
            .quality_of_life_settings
            .item_markers;
        for (i, &item) in self.randomization.item_placement.iter().enumerate() {
            let (room_id, node_id) = self.game_data.item_locations[i];
            if room_id == 19
                && self
                    .randomization
                    .objectives
                    .contains(&Objective::BombTorizo)
            {
                // If BT is an objective, we don't draw item dot there since an objective X tile will be drawn instead.
                continue;
            }
            let item_ptr = self.game_data.node_ptr_map[&(room_id, node_id)];
            let item_idx = self.rom.read_u8(item_ptr + 4)? as usize;
            let room_ptr = self.game_data.room_ptr_by_id[&room_id];
            let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
            let room = &self.game_data.room_geometry[room_idx];
            let area = self.map.area[room_idx];
            let (x, y) = find_item_xy(item_ptr, &room.items)?;
            let orig_tile = self.get_room_tile(room_id, x, y).clone();
            let mut tile = orig_tile.clone();
            tile.faded = false;
            if [MapTileInterior::HiddenItem, MapTileInterior::DoubleItem].contains(&tile.interior) {
                tile.interior = MapTileInterior::Item;
            }
            let interior = match markers {
                ItemMarkers::Simple => MapTileInterior::Item,
                ItemMarkers::Majors => {
                    if item.is_unique() || item == Item::ETank || item == Item::ReserveTank {
                        MapTileInterior::MajorItem
                    } else {
                        MapTileInterior::Item
                    }
                }
                ItemMarkers::Uniques => {
                    if item.is_unique() {
                        MapTileInterior::MajorItem
                    } else {
                        MapTileInterior::Item
                    }
                }
                ItemMarkers::ThreeTiered => {
                    if item.is_unique() {
                        MapTileInterior::MajorItem
                    } else if item != Item::Missile && item != Item::Nothing {
                        MapTileInterior::MediumItem
                    } else {
                        MapTileInterior::Item
                    }
                }
                ItemMarkers::FourTiered => {
                    if item.is_unique() {
                        MapTileInterior::MajorItem
                    } else if item == Item::ETank || item == Item::ReserveTank {
                        MapTileInterior::MediumItem
                    } else if item == Item::Super || item == Item::PowerBomb {
                        MapTileInterior::AmmoItem
                    } else {
                        assert!(item == Item::Missile || item == Item::Nothing);
                        MapTileInterior::Item
                    }
                }
            };
            tile.interior = interior.clone();
            if self.randomization.settings.other_settings.ultra_low_qol {
                self.set_room_tile(room_id, x, y, tile.clone());
            } else {
                self.dynamic_tile_data[area].push((item_idx, room_id, tile.clone()));
                if self.randomization.settings.other_settings.item_dot_change == ItemDotChange::Fade
                {
                    if interior == MapTileInterior::MajorItem
                        || (interior == MapTileInterior::MediumItem
                            && orig_tile.interior != MapTileInterior::MajorItem)
                        || (interior == MapTileInterior::AmmoItem
                            && orig_tile.interior != MapTileInterior::MediumItem
                            && orig_tile.interior != MapTileInterior::MajorItem)
                        || (interior == MapTileInterior::Item
                            && orig_tile.interior != MapTileInterior::AmmoItem
                            && orig_tile.interior != MapTileInterior::MediumItem
                            && orig_tile.interior != MapTileInterior::MajorItem)
                    {
                        tile.faded = true;
                        self.set_room_tile(room_id, x, y, tile.clone());
                    }
                } else {
                    tile.interior = MapTileInterior::Empty;
                    self.set_room_tile(room_id, x, y, tile.clone());
                }
            }
        }
        Ok(())
    }

    fn fix_message_boxes(&mut self) -> Result<()> {
        // Fix message boxes GFX: use white letters (color 2) instead of dark gray (color 1)
        for idx in 0xC0..0x100 {
            let mut data = self.read_map_tile_2bpp(idx)?;
            for y in 0..8 {
                for x in 0..8 {
                    if data[y][x] == 1 {
                        data[y][x] = 2;
                    }
                }
            }
            self.write_tile_2bpp(idx, data)?;
        }

        // Fix message boxes tilemaps: use palette 3 instead of 2, 6, or 7
        for addr in (snes2pc(0x85877F)..snes2pc(0x859641)).step_by(2) {
            let mut word = self.rom.read_u16(addr)?;
            let pal = (word >> 10) & 7;
            if pal == 2 || pal == 6 || pal == 7 {
                word &= !0x1C00;
                word |= 0x0C00;
                self.rom.write_u16(addr, word)?;
            }
        }
        Ok(())
    }

    fn fix_hud_black(&mut self) -> Result<()> {
        let mut tiles_to_change = vec![];
        tiles_to_change.extend(0..0x0F); // HUD digits, "ENERG"
        tiles_to_change.extend(0x1C..0x20);
        tiles_to_change.push(0x32); // "Y" of ENERGY
        tiles_to_change.push(0x4D); // Save station tile
        tiles_to_change.extend([0x33, 0x46, 0x47, 0x48]); // AUTO

        // Use color 0 instead of color 3 for black in HUD map tiles:
        // Also use color 3 instead of color 2 for white
        for idx in tiles_to_change {
            let mut tile = self.read_map_tile_2bpp(idx)?;
            for y in 0..8 {
                for x in 0..8 {
                    if tile[y][x] == 3 {
                        tile[y][x] = 0;
                    } else if tile[y][x] == 2 {
                        tile[y][x] = 3;
                    }
                }
            }
            self.write_tile_2bpp(idx, tile)?;

            let mut tile = self.read_map_tile_4bpp(idx)?;
            for y in 0..8 {
                for x in 0..8 {
                    if tile[y][x] == 2 {
                        tile[y][x] = 3;
                    }
                }
            }
            self.write_map_tile_4bpp(idx, tile)?;
        }
        Ok(())
    }

    fn darken_hud_grid(&mut self) -> Result<()> {
        // In HUD tiles, replace the white dotted grid lines with dark gray ones.
        self.write_tile_2bpp(
            0x1C,
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ],
        )?;
        self.write_tile_2bpp(
            0x1D,
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0],
            ],
        )?;
        self.write_tile_2bpp(
            0x1E,
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ],
        )?;
        self.write_tile_2bpp(
            0x1F,
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0],
            ],
        )?;
        Ok(())
    }

    fn write_disabled_etank_tile(&mut self) -> Result<()> {
        self.write_tile_2bpp(
            0x2F,
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )?;
        Ok(())
    }

    fn fix_fx_palettes(&mut self) -> Result<()> {
        // use palette 7 for FX (water, lava, etc.) instead of palette 6
        for addr in (snes2pc(0x8A8000)..snes2pc(0x8AB180)).step_by(2) {
            let word = self.rom.read_u16(addr)?;
            if word & 0x1C00 == 0x1800 {
                self.rom.write_u16(addr, word | 0x1C00)?;
            }
        }
        Ok(())
    }

    fn initialize_tiles(&mut self) -> Result<()> {
        // copy original tile GFX into each area-specific copy
        for area_idx in 0..6 {
            // 2bpp tiles (for HUD minimap)
            let src_addr = snes2pc(0x9AB200);
            let dst_addr = snes2pc(TILE_GFX_ADDR_2BPP + area_idx * 0x10000);
            for i in (0..0x1000).step_by(2) {
                let word = self.rom.read_u16(src_addr + i)?;
                self.rom.write_u16(dst_addr + i, word)?;
            }

            // 4bpp tiles (for HUD minimap)
            let src_addr = snes2pc(0xB68000);
            let dst_addr = snes2pc(TILE_GFX_ADDR_4BPP + area_idx * 0x10000);
            for i in (0..0x4000).step_by(2) {
                let word = self.rom.read_u16(src_addr + i)?;
                self.rom.write_u16(dst_addr + i, word)?;
            }
        }
        Ok(())
    }

    fn fix_kraid(&mut self) -> Result<()> {
        // Fix Kraid to copy BG3 tiles from area-specific location:
        let mut kraid_map_area: Option<isize> = None;
        for (i, room) in self.game_data.room_geometry.iter().enumerate() {
            if room.name == "Kraid Room" {
                kraid_map_area = Some(self.randomization.map.area[i] as isize);
            }
        }
        // Kraid alive:
        self.rom.write_u24(
            snes2pc(0x8FB817),
            TILE_GFX_ADDR_2BPP as isize + kraid_map_area.unwrap() * 0x10000,
        )?;
        self.rom.write_u16(snes2pc(0x8FB81C), 0x0C00)?; // avoid overwriting hazard tiles with (unneeded) message box tiles

        // Kraid dead:
        self.rom.write_u24(
            snes2pc(0x8FB842),
            TILE_GFX_ADDR_2BPP as isize + kraid_map_area.unwrap() * 0x10000,
        )?;
        self.rom.write_u16(snes2pc(0x8FB847), 0x0C00)?; // avoid overwriting hazard tiles with (unneeded) message box tiles
        Ok(())
    }

    fn substitute_colors(
        &mut self,
        item_idx: usize,
        tiles: Vec<usize>,
        subst: Vec<(u8, u8)>,
    ) -> Result<()> {
        for i in tiles {
            let addr = snes2pc(0x898000 + item_idx * 0x100 + i * 0x20);
            let mut tile = self.read_tile_4bpp(addr)?;
            for y in 0..8 {
                for x in 0..8 {
                    let mut c = tile[x][y];
                    for &(c_from, c_to) in subst.iter() {
                        if tile[x][y] == c_from {
                            c = c_to;
                        }
                    }
                    tile[x][y] = c;
                }
            }
            self.write_tile_4bpp(addr, tile)?;
        }
        Ok(())
    }

    fn fix_item_colors(&mut self) -> Result<()> {
        // Bombs: keep pink color
        self.substitute_colors(0, vec![0, 1, 2, 3], vec![(9, 13)])?;
        // Speed Booster: dark blue -> dark area color
        self.substitute_colors(10, vec![0, 1, 2, 3, 4, 5, 6, 7], vec![(13, 9)])?;
        // Varia: keep pink color
        self.substitute_colors(3, vec![0, 1, 2, 3, 4, 5, 6, 7], vec![(9, 13)])?;
        // Space Jump: keep pink color
        self.substitute_colors(6, vec![0, 1, 2, 3], vec![(9, 13)])?;
        // Morph: keep pink color, dark blue -> dark area color
        self.substitute_colors(7, vec![0, 1, 2, 3, 4, 5, 6, 7], vec![(9, 13), (13, 9)])?;
        Ok(())
    }

    // Write the tile graphics for the hazard markers on door frames (indicating doors leading to danger in the next room)
    fn write_hazard_tiles(&mut self) -> Result<()> {
        let b = 15; // black
        let w = 12; // white
        let hazard_tile1: [[u8; 8]; 8] = [
            [b, 3, 3, 3, b, b, b, 3],
            [b, b, b, b, b, b, b, b],
            [6, 6, 6, 6, 5, 5, 5, 6],
            [b, b, b, b, b, b, b, b],
            [3, 3, b, b, b, 3, 3, 3],
            [3, 3, 3, b, b, b, 3, 3],
            [b, 3, 3, 3, b, b, b, 3],
            [b, b, 3, 3, 3, b, b, b],
        ];
        let hazard_tile2: [[u8; 8]; 8] = [
            [b, b, b, 3, 3, 3, b, b],
            [3, b, b, b, 3, 3, 3, b],
            [2, 2, b, b, b, 2, 2, 2],
            [2, 2, 2, b, b, b, 2, 2],
            [b, 7, b, b, 7, 7, 7, 7],
            [5, 5, 4, 4, 4, 4, 4, 4],
            [b, 7, b, b, 7, 7, 7, 7],
            [2, b, b, b, 2, 2, 2, b],
        ];
        let hazard_tile3: [[u8; 8]; 8] = [
            [2, 2, b, b, b, 2, 2, 2],
            [2, 2, 2, b, b, b, 2, 2],
            [b, 2, 2, 2, b, b, b, 2],
            [b, b, 2, 2, 2, b, b, b],
            [b, b, b, 1, 1, 1, b, b],
            [1, b, b, b, 1, 1, 1, b],
            [1, 1, b, b, b, 1, 1, 1],
            [1, 1, 1, b, b, b, 1, 1],
        ];
        let hazard_tile4: [[u8; 8]; 8] = [
            [b, 5, b, b, 5, 5, 5, 5],
            [5, 4, 4, 4, w, w, 4, 4],
            [b, 6, b, b, 6, 6, 6, 6],
            [1, b, b, b, 1, 1, 1, b],
            [1, 1, b, b, b, 1, 1, 1],
            [1, 1, 1, b, b, b, 1, 1],
            [b, 1, 1, 1, b, b, b, 1],
            [b, b, 1, 1, 1, b, b, b],
        ];
        let hazard_elev_tile0: [[u8; 8]; 8] = [
            [7, 0, 7, 3, 7, 7, 7, 7],
            [0, 0, 7, 7, 7, 3, 15, 15],
            [0, 0, 7, 15, 3, 3, 2, 15],
            [0, 0, 7, 15, 3, 3, 2, 2],
            [0, 0, 7, 15, 15, 3, 2, 2],
            [0, 0, 7, 7, 7, 15, 2, 2],
            [7, 0, 7, 7, 6, 7, 15, 2],
            [1, 7, 7, 0, 7, 5, 7, 7],
        ];
        let hazard_elev_tile1: [[u8; 8]; 8] = [
            [7, 7, 7, 7, 6, 6, 5, 5],
            [15, 15, 2, 1, 1, 1, 15, 15],
            [15, 15, 15, 1, 1, 1, 1, 15],
            [15, 15, 15, 15, 1, 1, 1, 1],
            [2, 15, 15, 15, 15, 1, 1, 1],
            [2, 2, 15, 15, 15, 15, 1, 1],
            [2, 2, 2, 15, 15, 15, 15, 1],
            [7, 7, 6, 6, 6, 5, 5, 5],
        ];

        // Write 8x8 tiles:
        let base_addr = snes2pc(0xE98000);
        self.write_tile_4bpp(base_addr, hazard_tile1)?;
        self.write_tile_4bpp(base_addr + 0x20, hazard_tile2)?;
        self.write_tile_4bpp(base_addr + 0x40, hazard_tile3)?;
        self.write_tile_4bpp(base_addr + 0x60, hazard_tile4)?;
        self.write_tile_4bpp(base_addr + 0x80, diagonal_flip_tile(hazard_tile1))?;
        self.write_tile_4bpp(base_addr + 0xA0, diagonal_flip_tile(hazard_tile2))?;
        self.write_tile_4bpp(base_addr + 0xC0, diagonal_flip_tile(hazard_tile3))?;
        self.write_tile_4bpp(base_addr + 0xE0, diagonal_flip_tile(hazard_tile4))?;

        // Write 8x8 tiles for rooms with elevator hazard (leading to Main Hall).
        // In this case, the vertical door hazard tiles are replaced with elevator hazard tiles.
        self.write_tile_4bpp(base_addr + 0x180, hazard_tile1)?;
        self.write_tile_4bpp(base_addr + 0x1A0, hazard_tile2)?;
        self.write_tile_4bpp(base_addr + 0x1C0, hazard_tile3)?;
        self.write_tile_4bpp(base_addr + 0x1E0, hazard_tile4)?;
        self.write_tile_4bpp(base_addr + 0x200, hazard_elev_tile0)?;
        self.write_tile_4bpp(base_addr + 0x220, hazard_elev_tile1)?;

        // Write 16x16 tiles (tilemap):
        let hazard_tile1_idx = 0x278;
        let hazard_tile2_idx = 0x279;
        let hazard_tile3_idx = 0x27a;
        let hazard_tile4_idx = 0x27b;
        let flip_hazard_tile1_idx = 0x27c;
        let flip_hazard_tile2_idx = 0x27d;
        let flip_hazard_tile3_idx = 0x27e;
        let flip_hazard_tile4_idx = 0x27f;
        let door_frame1_idx = 0x342;
        let door_frame2_idx = 0x352;
        let door_frame3_idx = 0x362;
        let door_frame4_idx = 0x372;
        let flip_door_frame1_idx = 0x367;
        let flip_door_frame2_idx = 0x366;
        let flip_door_frame3_idx = 0x365;
        let flip_door_frame4_idx = 0x364;
        let elev_hazard_tile1_idx = 0x27c;
        let elev_hazard_tile2_idx = 0x27d;
        let blank_tile_idx = 0x338;
        let elev_top_tile1_idx = 0x2F7;
        let elev_top_tile2_idx = 0x2F6;
        let elev_bottom_tile1_idx = 0x2E8;

        for elev in [false, true] {
            let base_addr = if elev {
                snes2pc(0xE98280)
            } else {
                snes2pc(0xE98100)
            };

            // Top fourth of door going right:
            self.rom.write_u16(base_addr, hazard_tile1_idx | 0x2000)?; // top-left quarter (palette 0)
            self.rom
                .write_u16(base_addr + 2, door_frame1_idx | 0x2400)?; // top-right quarter (palette 1)
            self.rom
                .write_u16(base_addr + 4, hazard_tile2_idx | 0x2000)?; // bottom-left quarter (palette 0)
            self.rom
                .write_u16(base_addr + 6, door_frame2_idx | 0x2400)?; // bottom-right quarter (palette 1)

            // Second-from top fourth of door going right:
            self.rom
                .write_u16(base_addr + 8, hazard_tile3_idx | 0x2000)?; // top-left quarter (palette 0)
            self.rom
                .write_u16(base_addr + 10, door_frame3_idx | 0x2400)?; // top-right quarter (palette 1)
            self.rom
                .write_u16(base_addr + 12, hazard_tile4_idx | 0x2000)?; // bottom-left quarter (palette 0)
            self.rom
                .write_u16(base_addr + 14, door_frame4_idx | 0x2400)?; // bottom-right quarter (palette 1)

            if elev {
                // Top-left of elevator hazard:
                self.rom.write_u16(base_addr + 16, blank_tile_idx)?; // top-left quarter (palette 0)
                self.rom.write_u16(base_addr + 18, blank_tile_idx)?; // top-right quarter (palette 0)
                self.rom
                    .write_u16(base_addr + 20, elev_top_tile1_idx | 0xE000)?; // bottom-left quarter (palette 0, X+Y flip)
                self.rom
                    .write_u16(base_addr + 22, elev_top_tile2_idx | 0xE000)?; // bottom-right quarter (palette 0, X+Y flip)

                // Bottom-left of elevator hazard:
                self.rom
                    .write_u16(base_addr + 24, elev_hazard_tile1_idx | 0x2000)?; // top-left quarter (palette 0)
                self.rom
                    .write_u16(base_addr + 26, elev_hazard_tile2_idx | 0x2000)?; // top-right quarter (palette 0)
                self.rom
                    .write_u16(base_addr + 28, elev_bottom_tile1_idx | 0xE000)?; // bottom-left quarter (palette 0, X+Y flip)
                self.rom.write_u16(base_addr + 30, blank_tile_idx)?; // bottom-right quarter (palette 0)
            } else {
                // Left fourth of door going down:
                self.rom
                    .write_u16(base_addr + 16, flip_hazard_tile1_idx | 0x2000)?; // top-left quarter (palette 0)
                self.rom
                    .write_u16(base_addr + 18, flip_hazard_tile2_idx | 0x2000)?; // top-right quarter (palette 0)
                self.rom
                    .write_u16(base_addr + 20, flip_door_frame1_idx | 0x6400)?; // bottom-left quarter (palette 1, X flip)
                self.rom
                    .write_u16(base_addr + 22, flip_door_frame2_idx | 0x6400)?; // bottom-right quarter (palette 1, X flip)

                // Second-from left fourth of door going down:
                self.rom
                    .write_u16(base_addr + 24, flip_hazard_tile3_idx | 0x2000)?; // top-left quarter (palette 0)
                self.rom
                    .write_u16(base_addr + 26, flip_hazard_tile4_idx | 0x2000)?; // top-right quarter (palette 0)
                self.rom
                    .write_u16(base_addr + 28, flip_door_frame3_idx | 0x6400)?; // bottom-left quarter (palette 1, X flip)
                self.rom
                    .write_u16(base_addr + 30, flip_door_frame4_idx | 0x6400)?; // bottom-right quarter (palette 1, X flip)
            }
        }

        Ok(())
    }

    fn get_room_coords(&self, room_id: usize, x: isize, y: isize) -> (AreaIdx, isize, isize) {
        let room_ptr = self.game_data.room_ptr_by_id[&room_id];
        let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
        let area = self.map.area[room_idx];
        let room_coords = self.map.rooms[room_idx];
        let x = room_coords.0 as isize + x;
        let y = room_coords.1 as isize + y;
        (area, x, y)
    }

    fn get_room_tile(&mut self, room_id: usize, x: isize, y: isize) -> &mut MapTile {
        let (area, x, y) = self.get_room_coords(room_id, x, y);
        self.map_tile_map.get_mut(&(area, x, y)).unwrap()
    }

    fn set_room_tile(&mut self, room_id: usize, x: isize, y: isize, mut tile: MapTile) {
        tile.coords.0 = x as usize;
        tile.coords.1 = y as usize;
        let (area, x, y) = self.get_room_coords(room_id, x, y);
        self.map_tile_map.insert((area, x, y), tile);
    }

    fn apply_room_tiles(&mut self) -> Result<()> {
        // Draw the Toilet first:
        for room in &self.game_data.map_tile_data {
            if room.room_id != 321 {
                continue;
            }
            for tile in &room.map_tiles {
                self.set_room_tile(
                    room.room_id,
                    tile.coords.0 as isize,
                    tile.coords.1 as isize,
                    tile.clone(),
                );
            }
        }
        // Then draw other rooms on top:
        for room in &self.game_data.map_tile_data {
            if room.room_id == 321 {
                continue;
            }
            for tile in &room.map_tiles {
                self.set_room_tile(
                    room.room_id,
                    tile.coords.0 as isize,
                    tile.coords.1 as isize,
                    tile.clone(),
                );
            }
        }
        Ok(())
    }

    pub fn compute_area_bounds(&mut self) -> Result<()> {
        for &(area_idx, x, y) in self.map_tile_map.keys() {
            if x < self.area_min_x[area_idx] {
                self.area_min_x[area_idx] = x;
            }
            if x > self.area_max_x[area_idx] {
                self.area_max_x[area_idx] = x;
            }
            if y < self.area_min_y[area_idx] {
                self.area_min_y[area_idx] = y;
            }
            if y > self.area_max_y[area_idx] {
                self.area_max_y[area_idx] = y;
            }
        }

        for area in 0..NUM_AREAS {
            let margin_x = (64 - (self.area_max_x[area] - self.area_min_x[area])) / 2;
            let margin_y = (32 - (self.area_max_y[area] - self.area_min_y[area])) / 2;
            self.area_offset_x[area] = self.area_min_x[area] - margin_x;
            self.area_offset_y[area] = self.area_min_y[area] - margin_y;
        }

        Ok(())
    }

    pub fn apply_patches(&mut self) -> Result<()> {
        self.initialize_tiles()?;
        self.index_fixed_tiles()?;
        self.fix_pause_palettes()?;
        self.fix_message_boxes()?;
        self.fix_hud_black()?;
        self.darken_hud_grid()?;
        if self
            .randomization
            .settings
            .quality_of_life_settings
            .disableable_etanks
        {
            self.write_disabled_etank_tile()?;
        }
        self.apply_room_tiles()?;
        self.indicate_objective_tiles()?;
        if !self.randomization.settings.other_settings.ultra_low_qol {
            self.indicate_locked_doors()?;
            self.indicate_gray_doors()?;
        }
        self.add_cross_area_arrows()?;
        self.set_map_activation_behavior()?;
        self.indicate_items()?;
        self.compute_area_bounds()?;
        self.write_map_tiles()?;
        self.set_initial_map()?;
        if self
            .randomization
            .settings
            .quality_of_life_settings
            .room_outline_revealed
        {
            self.setup_special_door_reveal()?;
        }
        self.write_dynamic_tile_data(&self.dynamic_tile_data.clone())?;
        self.write_hazard_tiles()?;
        self.fix_fx_palettes()?;
        self.fix_kraid()?;
        self.fix_item_colors()?;

        Ok(())
    }
}
