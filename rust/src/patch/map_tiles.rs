use hashbrown::{HashMap, HashSet};
use log::info;

use crate::{
    game_data::{GameData, Item, ItemIdx, Map, RoomGeometryDoor, RoomGeometryItem},
    randomize::{ItemMarkers, Objectives, Randomization},
};

use super::{snes2pc, xy_to_explored_bit_ptr, xy_to_map_offset, Rom};
use anyhow::{bail, Context, Result};

type TilemapOffset = u16;
type TilemapWord = u16;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
enum Edge {
    Empty,
    Passage,
    Door,
    Wall,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
enum Interior {
    Empty,
    Item,
    MediumItem,
    MajorItem,
    Elevator,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct BasicTile {
    left: Edge,
    right: Edge,
    up: Edge,
    down: Edge,
    interior: Interior,
    faded: bool,
    heated: bool,
}

const NUM_AREAS: usize = 6;

pub struct MapPatcher<'a> {
    rom: &'a mut Rom,
    game_data: &'a GameData,
    map: &'a Map,
    randomization: &'a Randomization,
    next_free_tile_idx: usize,
    basic_tile_map: HashMap<BasicTile, TilemapWord>,
    reverse_map: HashMap<TilemapWord, BasicTile>,
    tile_gfx_map: HashMap<TilemapWord, [[u8; 8]; 8]>,
    edge_pixels_map: HashMap<Edge, Vec<usize>>,
}

const VANILLA_ELEVATOR_TILE: TilemapWord = 0xCE; // Index of elevator tile in vanilla game
const ELEVATOR_TILE: TilemapWord = 0x12; // Index of elevator tile with TR's map patch
pub const TILE_GFX_ADDR_4BPP: usize = 0xE28000; // Where to store area-specific tile graphics (must agree with map_area.asm)
pub const TILE_GFX_ADDR_2BPP: usize = 0xE2C000; // Where to store area-specific tile graphics (must agree with map_area.asm)

const FLIP_X: TilemapWord = 0x4000;
const FLIP_Y: TilemapWord = 0x8000;

const E: Edge = Edge::Empty;
const P: Edge = Edge::Passage;
const D: Edge = Edge::Door;
const W: Edge = Edge::Wall;
const O: Interior = Interior::Empty;
const I: Interior = Interior::Item;
const V: Interior = Interior::Elevator;

fn find_item_xy(addr: usize, room_items: &[RoomGeometryItem]) -> Result<(isize, isize)> {
    for room_item in room_items {
        if room_item.addr == addr {
            return Ok((room_item.x as isize, room_item.y as isize));
        }
    }
    bail!("Could not find item in room: {addr:x}");
}

impl<'a> MapPatcher<'a> {
    pub fn new(
        rom: &'a mut Rom,
        game_data: &'a GameData,
        map: &'a Map,
        randomization: &'a Randomization,
    ) -> Self {
        let mut pixels_map: HashMap<Edge, Vec<usize>> = HashMap::new();
        pixels_map.insert(Edge::Empty, vec![]);
        pixels_map.insert(Edge::Passage, vec![0, 1, 6, 7]);
        pixels_map.insert(Edge::Door, vec![0, 1, 2, 5, 6, 7]);
        pixels_map.insert(Edge::Wall, vec![0, 1, 2, 3, 4, 5, 6, 7]);

        MapPatcher {
            rom,
            game_data,
            map,
            randomization,
            next_free_tile_idx: 256,
            basic_tile_map: HashMap::new(),
            reverse_map: HashMap::new(),
            tile_gfx_map: HashMap::new(),
            edge_pixels_map: pixels_map,
        }
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

    fn write_tiles_area(&mut self, area_idx: usize) -> Result<()> {
        let mut reserved_tiles: HashSet<TilemapWord> = vec![
            // Used on HUD:
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
            0x0E, 0x0F, 0x1C, 0x1D, 0x1E,
            0x28, // slope tile that triggers tile above Samus to be marked explored
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D,
            0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
            0x4B,
            0xA8, // heated slope tile corresponding to 0x28
            // Used by max_ammo_display:
        ]
        .into_iter()
        .collect();

        let mut used_tiles: HashSet<TilemapWord> = HashSet::new();

        let base_ptr = self.game_data.area_map_ptrs[area_idx] as usize;
        for i in 0..0x800 {
            let word = (self.rom.read_u16(base_ptr + i * 2)? & 0x3FF) as TilemapWord;
            if self.tile_gfx_map.contains_key(&word) {
                used_tiles.insert(word);
            } else {
                reserved_tiles.insert(word);
            }
        }

        // Check tiles used by disappearing/fading items
        let ptr = self.rom.read_u16(snes2pc(0x83B000 + area_idx * 2))?;
        let size = self.rom.read_u16(snes2pc(0x83B00C + area_idx * 2))? as usize;
        for i in 0..size {
            let word = (self
                .rom
                .read_u16(snes2pc(0x830000 + (ptr as usize) + i * 6 + 4))?
                & 0x3FF) as TilemapWord;
            used_tiles.insert(word);
        }

        let mut free_tiles: Vec<TilemapWord> = Vec::new();
        for word in 0..192 {
            if !reserved_tiles.contains(&word) {
                free_tiles.push(word);
            }
        }

        info!(
            "Area {} free tiles: {} (of {})",
            area_idx,
            free_tiles.len() as isize - used_tiles.len() as isize,
            free_tiles.len()
        );
        if used_tiles.len() > free_tiles.len() {
            bail!("Not enough free tiles");
        }

        let mut tile_mapping: HashMap<TilemapWord, TilemapWord> = HashMap::new();
        for (&u, &f) in used_tiles.iter().zip(free_tiles.iter()) {
            tile_mapping.insert(u, f);
            let data = self.tile_gfx_map[&u];
            self.write_tile_2bpp_area(f as usize, data, Some(area_idx))?;
            self.write_map_tile_4bpp_area(f as usize, data, area_idx)?;
        }

        let palette = 0x1800;
        let palette_mask = 0x1C00;

        for i in 0..0x800 {
            let old_word = self.rom.read_u16(base_ptr + i * 2)? as TilemapWord;
            let old_idx = old_word & 0x3FF;
            let old_flip = old_word & 0xC000;
            let new_idx = *tile_mapping.get(&old_idx).unwrap_or(&old_idx);
            let new_word = ((new_idx | old_flip) & !palette_mask) | palette;
            self.rom.write_u16(base_ptr + i * 2, new_word as isize)?;
        }

        // Write tiles for disappearing/fading items:
        let ptr = self.rom.read_u16(snes2pc(0x83B000 + area_idx * 2))?;
        let size = self.rom.read_u16(snes2pc(0x83B00C + area_idx * 2))? as usize;
        for i in 0..size {
            let old_word = self
                .rom
                .read_u16(snes2pc(0x830000 + (ptr as usize) + i * 6 + 4))?
                as TilemapWord;
            let old_idx = old_word & 0x3FF;
            let old_flip = old_word & 0xC000;
            let new_idx = *tile_mapping.get(&old_idx).unwrap_or(&old_idx);
            let new_word = ((new_idx | old_flip) & !palette_mask) | palette;
            self.rom.write_u16(
                snes2pc(0x830000 + (ptr as usize) + i * 6 + 4),
                new_word as isize,
            )?;
        }

        Ok(())
    }

    fn write_tiles(&mut self) -> Result<()> {
        for area in 0..6 {
            self.write_tiles_area(area)?;
        }

        Ok(())
    }

    fn write_tile_2bpp_area(
        &mut self,
        idx: usize,
        mut data: [[u8; 8]; 8],
        area_idx: Option<usize,>,
    ) -> Result<()> {
        let base_addr = match area_idx {
            Some(area) => snes2pc(TILE_GFX_ADDR_2BPP + area * 0x10000), // New HUD tile GFX in ROM
            None => snes2pc(0x9AB200), // Standard BG3 tiles (used during Kraid)
        };
        for y in 0..8 {
            for x in 0..8 {
                if data[y][x] >= 4 {
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

    fn write_tile_4bpp(
        &mut self,
        base_addr: usize,
        data: [[u8; 8]; 8],
    ) -> Result<()> {
        for y in 0..8 {
            let addr = base_addr + y * 2;
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

    fn write_map_tile_4bpp_area(
        &mut self,
        idx: usize,
        data: [[u8; 8]; 8],
        area_idx: usize,
    ) -> Result<()> {
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

    fn create_tile(&mut self, data: [[u8; 8]; 8]) -> Result<TilemapWord> {
        let tile_idx = self.next_free_tile_idx;
        self.next_free_tile_idx += 1;
        self.tile_gfx_map.insert(tile_idx as TilemapWord, data);
        Ok(tile_idx as TilemapWord)
    }

    fn index_basic_tile_case(&mut self, tile: BasicTile, word: TilemapWord) {
        self.basic_tile_map.insert(tile, word);
        self.reverse_map.insert(word, tile);
    }

    fn index_basic_tile(&mut self, tile: BasicTile, word: TilemapWord) -> Result<()> {
        let data = self.render_basic_tile(tile)?;
        self.tile_gfx_map.insert(word, data);
        self.index_basic_tile_case(tile, word);
        if tile.interior != Interior::Elevator {
            self.index_basic_tile_case(
                BasicTile {
                    left: tile.right,
                    right: tile.left,
                    up: tile.up,
                    down: tile.down,
                    interior: tile.interior,
                    faded: tile.faded,
                    heated: tile.heated,
                },
                word | FLIP_X,
            );
            self.index_basic_tile_case(
                BasicTile {
                    left: tile.left,
                    right: tile.right,
                    up: tile.down,
                    down: tile.up,
                    interior: tile.interior,
                    faded: tile.faded,
                    heated: tile.heated,
                },
                word | FLIP_Y,
            );
            self.index_basic_tile_case(
                BasicTile {
                    left: tile.right,
                    right: tile.left,
                    up: tile.down,
                    down: tile.up,
                    interior: tile.interior,
                    faded: tile.faded,
                    heated: tile.heated,
                },
                word | FLIP_X | FLIP_Y,
            );
        }
        Ok(())
    }

    fn index_basic(
        &mut self,
        word: TilemapWord,
        left: Edge,
        right: Edge,
        up: Edge,
        down: Edge,
        interior: Interior,
    ) -> Result<()> {
        self.index_basic_tile(
            BasicTile {
                left,
                right,
                up,
                down,
                interior,
                faded: false,
                heated: false,
            },
            word,
        )?;
        Ok(())
    }

    fn index_vanilla_tiles(&mut self) -> Result<()> {
        self.index_basic(0x10, W, W, E, P, V)?; // Elevator: walls on left & right; passage on bottom
                                                // let tile = self.render_basic_tile(BasicTile { left: W, right: W, up: E, down: P, interior: V })?;
                                                // self.write_tile_4bpp(0x10, tile)?;

        self.index_basic(0x4F, W, W, W, P, V)?; // Elevator: walls on left, right, & top; passage on bottom
                                                // let tile = self.render_basic_tile(BasicTile { left: W, right: W, up: W, down: P, interior: V })?;
                                                // self.write_tile_4bpp(0x4F, tile)?;

        self.index_basic(0x5F, P, P, P, W, V)?; // Elevator: passages on left, right, & top; wall on bottom
                                                // let tile = self.render_basic_tile(BasicTile { left: P, right: P, up: P, down: W, interior: V })?;
                                                // self.write_tile_4bpp(0x5F, tile)?;

        self.index_basic(0x1B, E, E, E, E, O)?; // Empty tile with no walls
        self.index_basic(0x20, W, W, W, W, O)?; // Empty tile with wall on all four sides
        self.index_basic(0x21, W, E, W, W, O)?; // Empty tile with wall on top, left, and bottom
        self.index_basic(0x22, E, E, W, W, O)?; // Empty tile with wall on top and bottom
        self.index_basic(0x23, W, W, E, E, O)?; // Empty tile with wall on left and right
        self.index_basic(0x24, W, W, W, E, O)?; // Empty tile with wall on top, left, and right
        self.index_basic(0x25, W, E, W, E, O)?; // Empty tile with wall on top and left
        self.index_basic(0x26, E, E, W, E, O)?; // Empty tile with wall on top
        self.index_basic(0x27, E, W, E, E, O)?; // Empty tile with wall on right

        self.index_basic(0x76, E, E, W, E, I)?; // Item (dot) tile with a wall on top
        self.index_basic(0x77, W, E, E, E, I)?; // Item (dot) tile with a wall on left
        self.index_basic(0x5E, E, E, W, W, I)?; // Item (dot) tile with a wall on top and bottom
        self.index_basic(0x6E, W, W, W, E, I)?; // Item (dot) tile with a wall on top, left, and right
        self.index_basic(0x6F, W, W, W, W, I)?; // Item (dot) tile with a wall on all four sides
        self.index_basic(0x8E, W, E, W, E, I)?; // Item (dot) tile with a wall on top and left
        self.index_basic(0x8F, W, E, W, W, I)?; // Item (dot) tile with a wall on top, left, and bottom
                                                // Note: there's no item tile with walls on left and right.
        Ok(())
    }

    fn render_basic_tile(&mut self, tile: BasicTile) -> Result<[[u8; 8]; 8]> {
        let bg_color = if tile.heated { 2 } else { 1 };
        let mut data: [[u8; 8]; 8] = [[bg_color; 8]; 8];
        for &i in &self.edge_pixels_map[&tile.left] {
            data[i][0] = 3;
        }
        for &i in &self.edge_pixels_map[&tile.right] {
            data[i][7] = 3;
        }
        for &i in &self.edge_pixels_map[&tile.up] {
            data[0][i] = 3;
        }
        for &i in &self.edge_pixels_map[&tile.down] {
            data[7][i] = 3;
        }

        let item_color = if tile.faded {
            if tile.heated {
                1
            } else {
                2
            }
        } else {
            3
        };
        match tile.interior {
            Interior::Empty => {}
            Interior::Item => {
                data[3][3] = item_color;
                data[3][4] = item_color;
                data[4][3] = item_color;
                data[4][4] = item_color;
            }
            Interior::MediumItem => {
                data[2][3] = item_color;
                data[2][4] = item_color;
                data[5][3] = item_color;
                data[5][4] = item_color;
                data[3][2] = item_color;
                data[4][2] = item_color;
                data[3][5] = item_color;
                data[4][5] = item_color;
            }
            Interior::MajorItem => {
                for i in 2..6 {
                    for j in 2..6 {
                        data[i][j] = item_color;
                    }
                }
                data[2][2] = bg_color;
                data[5][2] = bg_color;
                data[2][5] = bg_color;
                data[5][5] = bg_color;
            }
            Interior::Elevator => {
                // Use white instead of red for elevator platform:
                data[5][3] = 3;
                data[5][4] = 3;
            }
        }
        Ok(data)
    }

    fn add_basic_tile(&mut self, tile: BasicTile) -> Result<()> {
        let data = self.render_basic_tile(tile)?;
        let tile_idx = self.create_tile(data)?;
        self.index_basic_tile(tile, tile_idx)?;

        Ok(())
    }

    fn get_basic_tile(&mut self, tile: BasicTile) -> Result<TilemapWord> {
        if !self.basic_tile_map.contains_key(&tile) {
            self.add_basic_tile(tile)?;
        }
        Ok(self.basic_tile_map[&tile])
    }

    fn fix_elevators(&mut self) -> Result<()> {
        // Replace vanilla elevator tiles with the new tile compatible with TR's map patch:
        let base_ptr = 0x1A8000; // Location of map tilemaps
        for i in 0..0x3000 {
            let word = self.rom.read_u16(base_ptr + i * 2)? as TilemapWord;
            if (word & 0x3FF) == VANILLA_ELEVATOR_TILE {
                self.rom
                    .write_u16(base_ptr + i * 2, (ELEVATOR_TILE | 0x0C00) as isize)?;
            }
        }

        let elevator_tile_pause: [[u8; 8]; 8] = [
            [0, 3, 1, 4, 4, 1, 3, 0],
            [0, 3, 4, 4, 4, 4, 3, 0],
            [0, 3, 1, 4, 4, 1, 3, 0],
            [0, 3, 4, 4, 4, 4, 3, 0],
            [0, 3, 1, 4, 4, 1, 3, 0],
            [0, 3, 4, 4, 4, 4, 3, 0],
            [0, 3, 1, 4, 4, 1, 3, 0],
            [0, 3, 4, 4, 4, 4, 3, 0],
        ];
        let elevator_tile_hud: [[u8; 8]; 8] = [
            [0, 3, 1, 0, 0, 1, 3, 0],
            [0, 3, 0, 0, 0, 0, 3, 0],
            [0, 3, 1, 0, 0, 1, 3, 0],
            [0, 3, 0, 0, 0, 0, 3, 0],
            [0, 3, 1, 0, 0, 1, 3, 0],
            [0, 3, 0, 0, 0, 0, 3, 0],
            [0, 3, 1, 0, 0, 1, 3, 0],
            [0, 3, 0, 0, 0, 0, 3, 0],
        ];
        self.write_tile_2bpp(ELEVATOR_TILE as usize, elevator_tile_hud)?;
        self.write_map_tile_4bpp(ELEVATOR_TILE as usize, elevator_tile_pause)?;

        // In top elevator rooms, replace down arrow tiles with elevator tiles:
        self.patch_room("Green Brinstar Elevator Room", vec![(0, 3, ELEVATOR_TILE)])?;
        self.patch_room("Red Brinstar Elevator Room", vec![(0, 3, ELEVATOR_TILE)])?;
        self.patch_room("Blue Brinstar Elevator Room", vec![(0, 3, ELEVATOR_TILE)])?;
        self.patch_room("Forgotten Highway Elevator", vec![(0, 3, ELEVATOR_TILE)])?;
        self.patch_room("Statues Room", vec![(0, 4, ELEVATOR_TILE)])?;
        // We skip "Warehouse Entrance" since the room geometry (arbitrarily) did not include the arrow tile.
        // self.patch_room("Warehouse Entrance", vec![(0, 3, ELEVATOR_TILE)])?;

        // Likewise, in bottom elevator rooms, replace up arrow tiles with elevator tiles:
        // Oddly, in Main Shaft there wasn't an arrow here in the vanilla game. But we left a spot in the room geometry as if there were.
        self.patch_room("Green Brinstar Main Shaft", vec![(0, 0, ELEVATOR_TILE)])?;
        self.patch_room("Maridia Elevator Room", vec![(0, 0, ELEVATOR_TILE)])?;
        self.patch_room("Business Center", vec![(0, 0, ELEVATOR_TILE)])?;
        // Skipping Morph Ball Room, Tourian First Room, and Caterpillar room, since we didn't include the arrow tile in these
        // rooms in the room geometry (an inconsistency which doesn't really matter because its only observable effect is in the
        // final length of the elevator on the map, which already has variations across rooms). We skip Lower Norfair Elevator
        // and Main Hall because these have no arrows on the vanilla map (since these don't cross regions in vanilla).

        // Patch map tile in Aqueduct to replace Botwoon Hallway with tube/elevator tile
        self.patch_room("Aqueduct", vec![(2, 3, ELEVATOR_TILE)])?;

        Ok(())
    }

    fn patch_room(
        &mut self,
        room_name: &str,
        tiles: Vec<(isize, isize, TilemapWord)>,
    ) -> Result<()> {
        let room_idx = self.game_data.room_idx_by_name[room_name];
        let room = &self.game_data.room_geometry[room_idx];
        let area = self.map.area[room_idx];
        let x0 = self.rom.read_u8(room.rom_address + 2)? as isize;
        let y0 = self.rom.read_u8(room.rom_address + 3)? as isize;
        for (x, y, word) in &tiles {
            let base_ptr = self.game_data.area_map_ptrs[area] as usize;
            let offset = super::xy_to_map_offset(x0 + x, y0 + y) as usize;
            self.rom
                .write_u16(base_ptr + offset, (word | 0x0C00) as isize)?;
        }
        Ok(())
    }

    fn patch_room_basic(
        &mut self,
        room_name: &str,
        tiles: Vec<(isize, isize, Edge, Edge, Edge, Edge, Interior)>,
    ) -> Result<()> {
        let mut word_tiles: Vec<(isize, isize, TilemapWord)> = Vec::new();
        for &(x, y, left, right, up, down, interior) in &tiles {
            let basic_tile = BasicTile {
                left,
                right,
                up,
                down,
                interior,
                faded: false,
                heated: false,
            };
            word_tiles.push((x, y, self.get_basic_tile(basic_tile)?));
        }
        self.patch_room(room_name, word_tiles)?;
        Ok(())
    }

    fn indicate_boss_tiles(&mut self, boss_tile: u16, heated_boss_tile: u16) -> Result<()> {
        self.patch_room(
            "Kraid Room",
            vec![
                (0, 0, boss_tile),
                (0, 1, boss_tile),
                (1, 0, boss_tile),
                (1, 1, boss_tile),
            ],
        )?;
        self.patch_room("Phantoon's Room", vec![(0, 0, boss_tile)])?;
        self.patch_room(
            "Draygon's Room",
            vec![
                (0, 0, boss_tile),
                (0, 1, boss_tile),
                (1, 0, boss_tile),
                (1, 1, boss_tile),
            ],
        )?;
        self.patch_room(
            "Ridley's Room",
            vec![(0, 0, heated_boss_tile), (0, 1, heated_boss_tile)],
        )?;

        Ok(())
    }

    fn indicate_miniboss_tiles(&mut self, boss_tile: u16, heated_boss_tile: u16) -> Result<()> {
        self.patch_room(
            "Spore Spawn Room",
            vec![(0, 0, boss_tile), (0, 1, boss_tile), (0, 2, boss_tile)],
        )?;
        self.patch_room(
            "Crocomire's Room",
            vec![
                (0, 0, boss_tile),
                (1, 0, boss_tile),
                (2, 0, boss_tile),
                (3, 0, boss_tile),
                (4, 0, boss_tile),
                (5, 0, boss_tile),
                (6, 0, boss_tile),
                // We don't mark the last tile, so the item can still be visible.
            ],
        )?;
        self.patch_room("Botwoon's Room", vec![(0, 0, boss_tile), (1, 0, boss_tile)])?;
        self.patch_room(
            "Golden Torizo's Room",
            vec![
                (0, 1, heated_boss_tile),
                (1, 1, heated_boss_tile),
                // We don't mark the top row of tiles, so the items can still be visible.
            ],
        )?;

        Ok(())
    }

    fn indicate_metroid_tiles(&mut self, boss_tile: u16) -> Result<()> {
        self.patch_room(
            "Metroid Room 1",
            vec![
                (0, 0, boss_tile),
                (1, 0, boss_tile),
                (2, 0, boss_tile),
                (3, 0, boss_tile),
                (4, 0, boss_tile),
                (5, 0, boss_tile),
            ],
        )?;

        self.patch_room("Metroid Room 2", vec![(0, 0, boss_tile), (0, 1, boss_tile)])?;

        self.patch_room(
            "Metroid Room 3",
            vec![
                (0, 0, boss_tile),
                (1, 0, boss_tile),
                (2, 0, boss_tile),
                (3, 0, boss_tile),
                (4, 0, boss_tile),
                (5, 0, boss_tile),
            ],
        )?;

        self.patch_room("Metroid Room 4", vec![(0, 0, boss_tile), (0, 1, boss_tile)])?;

        Ok(())
    }

    fn indicate_chozo_tiles(&mut self, boss_tile: u16, heated_boss_tile: u16) -> Result<()> {
        self.patch_room("Bomb Torizo Room", vec![(0, 0, boss_tile)])?;

        self.patch_room("Bowling Alley", vec![(4, 1, boss_tile)])?;

        self.patch_room("Acid Statue Room", vec![(0, 0, heated_boss_tile)])?;

        self.patch_room(
            "Golden Torizo's Room",
            vec![
                (0, 1, heated_boss_tile),
                (1, 1, heated_boss_tile),
                // We don't mark the top row of tiles, so the items can still be visible.
            ],
        )?;

        Ok(())
    }

    fn indicate_pirates_tiles(&mut self, boss_tile: u16, heated_boss_tile: u16) -> Result<()> {
        self.patch_room(
            "Pit Room",
            vec![(0, 0, boss_tile), (1, 0, boss_tile), (2, 0, boss_tile)],
        )?;

        self.patch_room(
            "Baby Kraid Room",
            vec![
                (0, 0, boss_tile),
                (1, 0, boss_tile),
                (2, 0, boss_tile),
                (3, 0, boss_tile),
                (4, 0, boss_tile),
                (5, 0, boss_tile),
            ],
        )?;

        self.patch_room(
            "Plasma Room",
            vec![
                (0, 0, boss_tile),
                (1, 0, boss_tile),
                (0, 1, boss_tile),
                (1, 1, boss_tile),
                (0, 2, boss_tile),
            ],
        )?;

        self.patch_room(
            "Metal Pirates Room",
            vec![
                (0, 0, heated_boss_tile),
                (1, 0, heated_boss_tile),
                (2, 0, heated_boss_tile),
            ],
        )?;

        Ok(())
    }

    fn indicate_special_tiles(&mut self) -> Result<()> {
        let refill_tile = self.create_tile([
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 1, 1, 3, 3, 3],
            [3, 3, 3, 1, 1, 3, 3, 3],
            [3, 1, 1, 1, 1, 1, 1, 3],
            [3, 1, 1, 1, 1, 1, 1, 3],
            [3, 3, 3, 1, 1, 3, 3, 3],
            [3, 3, 3, 1, 1, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ])?;
        let map_tile = self.create_tile([
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 1, 1, 1, 1, 1, 1, 3],
            [3, 1, 3, 3, 3, 3, 1, 3],
            [3, 1, 3, 1, 1, 3, 1, 3],
            [3, 1, 3, 1, 1, 3, 1, 3],
            [3, 1, 3, 3, 3, 3, 1, 3],
            [3, 1, 1, 1, 1, 1, 1, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ])?;
        let boss_tile = self.create_tile([
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 1, 1, 3, 3, 1, 1, 3],
            [3, 1, 1, 1, 1, 1, 1, 3],
            [3, 3, 1, 1, 1, 1, 3, 3],
            [3, 3, 1, 1, 1, 1, 3, 3],
            [3, 1, 1, 1, 1, 1, 1, 3],
            [3, 1, 1, 3, 3, 1, 1, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ])?;
        let heated_boss_tile = self.create_tile([
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 2, 2, 3, 3, 2, 2, 3],
            [3, 2, 2, 2, 2, 2, 2, 3],
            [3, 3, 2, 2, 2, 2, 3, 3],
            [3, 3, 2, 2, 2, 2, 3, 3],
            [3, 2, 2, 2, 2, 2, 2, 3],
            [3, 2, 2, 3, 3, 2, 2, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ])?;

        self.patch_room("Landing Site", vec![(4, 4, refill_tile)])?;
        for room in &self.game_data.room_geometry {
            if room.name.contains("Refill") || room.name.contains("Recharge") {
                self.patch_room(&room.name, vec![(0, 0, refill_tile)])?;
            }
        }

        for room in &self.game_data.room_geometry {
            if room.name.contains(" Map Room") {
                self.patch_room(&room.name, vec![(0, 0, map_tile)])?;
            }
        }

        match self.randomization.difficulty.objectives {
            Objectives::Bosses => {
                self.indicate_boss_tiles(boss_tile, heated_boss_tile)?;
            }
            Objectives::Minibosses => {
                self.indicate_miniboss_tiles(boss_tile, heated_boss_tile)?;
            }
            Objectives::Metroids => {
                self.indicate_metroid_tiles(boss_tile)?;
            }
            Objectives::Chozos => {
                self.indicate_chozo_tiles(boss_tile, heated_boss_tile)?;
            }
            Objectives::Pirates => {
                self.indicate_pirates_tiles(boss_tile, heated_boss_tile)?;
            }
        }

        if self.randomization.difficulty.save_animals {
            self.patch_room("Bomb Torizo Room", vec![(0, 0, boss_tile)])?;
        }

        self.patch_room(
            "Mother Brain Room",
            vec![
                (0, 0, boss_tile),
                (1, 0, boss_tile),
                (2, 0, boss_tile),
                (3, 0, boss_tile),
            ],
        )?;

        Ok(())
    }

    fn fix_item_dots(&mut self) -> Result<()> {
        // Add map dots for items that are hidden in the vanilla game:
        self.patch_room_basic("West Ocean", vec![(1, 0, E, E, W, E, I)])?;
        self.patch_room_basic(
            "Blue Brinstar Energy Tank Room",
            vec![(1, 2, E, E, W, W, I)],
        )?;
        self.patch_room_basic("Warehouse Kihunter Room", vec![(2, 0, E, W, W, W, I)])?;
        self.patch_room_basic("Cathedral", vec![(2, 1, E, W, E, W, I)])?;
        self.patch_room_basic("Speed Booster Hall", vec![(11, 1, E, W, W, W, I)])?;
        self.patch_room_basic("Crumble Shaft", vec![(0, 0, W, W, W, E, I)])?;
        self.patch_room_basic("Ridley Tank Room", vec![(0, 0, W, W, W, W, I)])?;
        self.patch_room_basic("Bowling Alley", vec![(3, 2, E, W, W, W, I)])?;
        self.patch_room_basic("Mama Turtle Room", vec![(2, 1, E, W, E, E, I)])?;
        self.patch_room_basic("The Precious Room", vec![(1, 0, E, W, W, W, I)])?;

        // Remove map dots for locations that are not items.
        self.patch_room_basic("Statues Room", vec![(0, 0, W, W, W, E, O)])?;
        self.patch_room_basic("Spore Spawn Room", vec![(0, 2, W, W, E, W, O)])?;
        self.patch_room_basic("Crocomire's Room", vec![(5, 0, E, E, W, W, O)])?;
        self.patch_room_basic("Acid Statue Room", vec![(0, 0, W, E, W, E, O)])?;
        self.patch_room_basic("Bowling Alley", vec![(4, 1, E, W, W, W, O)])?;
        self.patch_room_basic("Botwoon's Room", vec![(0, 0, W, E, W, W, O)])?;
        self.patch_room_basic("Phantoon's Room", vec![(0, 0, W, W, W, W, O)])?;
        Ok(())
    }

    fn fix_walls(&mut self) -> Result<()> {
        // Add missing external walls to make sure they give double-pixel walls when touching an adjacent room:
        // (Much of this will be overridden by changes below, e.g. ones adding doors, sand transitions.
        // TODO: clean it up.)
        self.patch_room_basic(
            "West Ocean",
            vec![
                (2, 2, E, W, E, E, O),
                (3, 1, E, E, E, W, O),
                (4, 1, E, E, E, W, O),
                (3, 3, E, E, W, E, O),
                (4, 3, E, E, W, E, O),
                (5, 2, W, W, E, E, O),
            ],
        )?;
        self.patch_room_basic("Warehouse Entrance", vec![(1, 1, W, W, E, W, O)])?;
        self.patch_room_basic("Main Street", vec![(2, 2, E, W, W, W, O)])?;
        self.patch_room_basic(
            "West Aqueduct Quicksand Room",
            vec![(0, 0, W, W, W, E, O), (0, 1, W, W, E, W, O)],
        )?;
        self.patch_room_basic(
            "East Aqueduct Quicksand Room",
            vec![(0, 0, W, W, W, E, O), (0, 1, W, W, E, W, O)],
        )?;
        self.patch_room_basic(
            "Botwoon Quicksand Room",
            vec![(0, 0, W, E, W, W, O), (1, 0, E, W, W, W, O)],
        )?;
        self.patch_room_basic("Plasma Beach Quicksand Room", vec![(0, 0, W, W, W, W, O)])?;
        self.patch_room_basic("Lower Norfair Fireflea Room", vec![(1, 3, W, W, E, W, O)])?;

        Ok(())
    }

    fn indicate_doors(&mut self) -> Result<()> {
        // Make doors appear as two-pixel-wide opening instead of as a solid wall:
        for (room_idx, room) in self.game_data.room_geometry.iter().enumerate() {
            let area = self.map.area[room_idx];
            let x0 = self.rom.read_u8(room.rom_address + 2)?;
            let y0 = self.rom.read_u8(room.rom_address + 3)?;
            let mut doors_by_xy: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
            for (i, door) in room.doors.iter().enumerate() {
                doors_by_xy.entry((door.x, door.y)).or_default().push(i);
            }
            for (&(x, y), idxs) in doors_by_xy.iter() {
                let base_ptr = self.game_data.area_map_ptrs[area] as usize;
                let offset = xy_to_map_offset(x0 + x as isize, y0 + y as isize) as usize;
                let word = (self.rom.read_u16(base_ptr + offset)? as TilemapWord) & 0xC3FF;
                let basic_tile_opt = self.reverse_map.get(&word);
                if let Some(basic_tile) = basic_tile_opt {
                    let mut new_tile = basic_tile.clone();
                    for &i in idxs {
                        let dir = &room.doors[i].direction;
                        if dir == "left" {
                            new_tile.left = Edge::Door;
                        } else if dir == "right" {
                            new_tile.right = Edge::Door;
                        } else if dir == "up" {
                            new_tile.up = Edge::Door;
                        } else if dir == "down" {
                            new_tile.down = Edge::Door;
                        } else {
                            bail!("Unexpected door direction: {dir}")
                        }
                    }
                    let modified_word = self.get_basic_tile(new_tile)?;
                    self.rom
                        .write_u16(base_ptr + offset, (modified_word | 0x0C00) as isize)?;
                }
            }
        }
        Ok(())
    }

    fn indicate_heat(&mut self) -> Result<()> {
        // Make heated rooms appear lighter:
        for (room_idx, room) in self.game_data.room_geometry.iter().enumerate() {
            if !room.heated {
                continue;
            }
            let area = self.map.area[room_idx];
            let x0 = self.rom.read_u8(room.rom_address + 2)?;
            let y0 = self.rom.read_u8(room.rom_address + 3)?;
            for y in 0..room.map.len() {
                for x in 0..room.map[0].len() {
                    let base_ptr = self.game_data.area_map_ptrs[area] as usize;
                    let offset = xy_to_map_offset(x0 + x as isize, y0 + y as isize) as usize;
                    if room.map[y][x] == 0 {
                        continue;
                    }
                    let word = (self.rom.read_u16(base_ptr + offset)? as TilemapWord) & 0xC3FF;
                    let basic_tile_opt = self.reverse_map.get(&word);
                    if let Some(basic_tile) = basic_tile_opt {
                        let mut new_tile = basic_tile.clone();
                        new_tile.heated = true;
                        let modified_word = self.get_basic_tile(new_tile)?;
                        self.rom
                            .write_u16(base_ptr + offset, (modified_word | 0x0C00) as isize)?;
                    }
                }
            }
        }

        // Create heated slope tiles for Crocomire Speedway:
        fn make_heated(mut tile: [[u8; 8]; 8]) -> [[u8; 8]; 8] {
            for y in 0..8 {
                for x in 0..8 {
                    if tile[y][x] == 1 {
                        tile[y][x] = 2;
                    } else if tile[y][x] == 2 {
                        tile[y][x] = 3;
                    }
                }
            }
            tile
        }
        let slope1 = 0xA8;  // We have to put this tile in a specific place because it (like 0x28) has special behavior to explore the tile above it.
        let slope1_tile = make_heated(self.read_map_tile_4bpp(0x28)?);
        self.write_tile_2bpp(0xA8, slope1_tile)?;
        self.write_map_tile_4bpp(0xA8, slope1_tile)?;
        // The remaining 3 slope tiles can go anywhere, so we assign them dynamically:
        let slope2 = self.create_tile(make_heated(self.read_map_tile_4bpp(0x29)?))?;
        let slope3 = self.create_tile(make_heated(self.read_map_tile_4bpp(0x2A)?))?;
        let slope4 = self.create_tile(make_heated(self.read_map_tile_4bpp(0x2B)?))?;
        self.patch_room(
            "Crocomire Speedway",
            vec![
                (4, 1, slope1 | FLIP_X),
                (4, 0, slope2 | FLIP_X),
                (3, 1, slope3 | FLIP_X),
                (3, 0, slope4 | FLIP_X),
                (6, 2, slope1 | FLIP_X),
                (6, 1, slope2 | FLIP_X),
                (5, 2, slope3 | FLIP_X),
                (5, 1, slope4 | FLIP_X),
            ],
        )?;

        Ok(())
    }

    fn indicate_passages(&mut self) -> Result<()> {
        // Indicate hidden passages as 4-pixel-wide openings (and single pixel thick) instead of as solid walls:

        // Crateria:
        self.patch_room_basic(
            "Climb",
            vec![
                (1, 0, W, P, D, E, O),
                (1, 7, W, P, E, E, O),
                (1, 8, P, D, E, W, O),
            ],
        )?;
        self.patch_room_basic("Crateria Super Room", vec![(3, 0, E, W, W, P, I)])?;
        self.patch_room_basic("Landing Site", vec![(2, 2, P, E, E, E, O)])?;
        self.patch_room_basic("Parlor and Alcatraz", vec![(1, 0, P, E, W, E, O)])?;
        self.patch_room_basic(
            "Pit Room",
            vec![(0, 0, D, E, W, P, O), (0, 1, W, W, E, W, I)],
        )?;
        self.patch_room_basic("Final Missile Bombway", vec![(1, 0, P, D, W, W, O)])?;
        self.patch_room_basic("Crateria Kihunter Room", vec![(1, 0, E, E, W, P, O)])?;
        self.patch_room_basic(
            "West Ocean",
            vec![(0, 5, W, P, E, W, I), (1, 5, E, E, E, W, O)],
        )?;
        self.patch_room_basic(
            "Gauntlet Entrance",
            vec![
                (0, 0, D, P, W, W, O),
                (1, 0, E, P, W, W, O),
                (2, 0, E, P, W, W, O),
                (3, 0, E, P, W, W, O),
            ],
        )?;
        self.patch_room_basic(
            "Gauntlet Energy Tank Room",
            vec![
                (0, 0, D, P, W, W, O),
                (2, 0, P, E, W, W, O),
                (3, 0, P, E, W, W, O),
                (4, 0, P, E, W, W, O),
                (5, 0, P, D, W, W, I),
            ],
        )?;
        self.patch_room_basic(
            "Green Pirates Shaft",
            vec![
                (0, 0, W, D, W, P, O),
                (0, 1, W, W, E, P, I),
                (0, 4, W, D, P, E, O),
            ],
        )?;
        self.patch_room_basic(
            "Statues Room",
            vec![(0, 0, D, W, W, P, O), (0, 1, W, W, E, P, V)],
        )?;
        self.patch_room_basic("Green Brinstar Elevator Room", vec![(0, 0, W, D, W, P, V)])?;
        self.patch_room_basic("Blue Brinstar Elevator Room", vec![(0, 0, D, W, W, P, V)])?;
        self.patch_room_basic("Red Brinstar Elevator Room", vec![(0, 0, W, W, D, P, V)])?;
        self.patch_room_basic("Forgotten Highway Elevator", vec![(0, 0, W, W, D, P, V)])?;

        // Brinstar:
        self.patch_room_basic("Brinstar Pre-Map Room", vec![(0, 0, D, P, W, W, O)])?;
        self.patch_room_basic(
            "Early Supers Room",
            vec![(0, 1, D, E, P, W, O), (2, 1, E, D, P, W, O)],
        )?;
        self.patch_room_basic("Brinstar Reserve Tank Room", vec![(0, 0, D, P, W, W, I)])?;
        self.patch_room_basic(
            "Etecoon Energy Tank Room",
            vec![(0, 0, D, E, W, P, I), (1, 1, E, E, E, W, O)],
        )?;
        self.patch_room_basic(
            "Green Brinstar Main Shaft",
            vec![
                (0, 6, D, D, E, P, O),
                (0, 7, W, D, E, E, O),
                (1, 7, D, E, W, W, O),
                (2, 7, E, P, W, E, O),
                (0, 10, D, P, E, W, O),
                (2, 10, P, W, E, E, O),
            ],
        )?;
        self.patch_room_basic(
            "Big Pink",
            vec![
                (1, 7, E, E, W, W, O),
                (2, 0, P, E, W, E, O),
                (2, 6, W, E, E, P, I),
                (2, 7, P, W, E, W, I),
                (3, 5, E, P, E, E, O),
            ],
        )?;
        self.patch_room_basic(
            "Pink Brinstar Power Bomb Room",
            vec![(0, 0, W, E, W, P, O), (1, 1, E, D, E, W, O)],
        )?;
        self.patch_room_basic("Waterway Energy Tank Room", vec![(1, 0, P, E, W, W, O)])?;
        self.patch_room_basic("Dachora Room", vec![(4, 0, E, E, W, P, O)])?;
        self.patch_room_basic(
            "Morph Ball Room",
            vec![
                (1, 2, E, P, W, W, O),
                (3, 2, P, E, W, W, O),
                (4, 2, P, E, W, W, I),
                (5, 2, E, E, P, W, V),
            ],
        )?;
        self.patch_room_basic(
            "Blue Brinstar Energy Tank Room",
            vec![(2, 2, E, W, P, W, I)],
        )?;
        self.patch_room_basic(
            "Alpha Power Bomb Room",
            vec![(0, 0, W, E, W, W, I), (1, 0, P, E, W, W, I)],
        )?;
        self.patch_room_basic(
            "Below Spazer",
            vec![(0, 1, D, E, P, W, O), (1, 1, E, D, P, W, O)],
        )?;
        self.patch_room_basic("Beta Power Bomb Room", vec![(0, 0, W, E, W, P, O)])?;
        self.patch_room_basic(
            "Caterpillar Room",
            vec![(0, 3, D, P, E, E, O), (0, 5, D, W, E, P, O)],
        )?;
        self.patch_room_basic("Red Tower", vec![(0, 6, D, W, E, P, O)])?;
        self.patch_room_basic("Kraid Eye Door Room", vec![(0, 1, D, E, P, W, O)])?;
        self.patch_room_basic("Warehouse Entrance", vec![(0, 0, D, P, W, E, V)])?;
        self.patch_room_basic("Warehouse Zeela Room", vec![(0, 1, D, P, E, W, O)])?;
        self.patch_room_basic(
            "Warehouse Kihunter Room",
            vec![(1, 0, E, E, W, P, O), (2, 0, E, P, W, W, I)],
        )?;
        self.patch_room_basic("Green Hill Zone", vec![(2, 1, E, P, W, E, O)])?;

        // Wrecked Ship:
        self.patch_room_basic("Basement", vec![(3, 0, E, P, W, W, O)])?;
        self.patch_room_basic("Electric Death Room", vec![(0, 1, W, D, P, E, O)])?;
        self.patch_room_basic("Wrecked Ship East Super Room", vec![(3, 0, P, W, W, W, I)])?;
        self.patch_room_basic(
            "Wrecked Ship Main Shaft",
            vec![
                (4, 2, W, W, P, E, O),
                (4, 5, P, W, E, E, O),
                (4, 6, D, P, E, P, O),
            ],
        )?;
        self.patch_room_basic(
            "Bowling Alley",
            vec![
                (1, 1, E, E, W, P, O),
                (1, 2, D, P, E, W, O),
                (3, 2, E, P, W, W, I),
                (5, 0, E, W, W, P, I),
            ],
        )?;

        // Maridia:
        self.patch_room_basic("Oasis", vec![(0, 1, D, D, P, W, O)])?;
        self.patch_room_basic(
            "Pants Room",
            vec![(0, 3, D, D, P, W, O), (1, 3, D, W, P, W, O)],
        )?;
        self.patch_room_basic("Shaktool Room", vec![(0, 0, D, P, W, W, O)])?;
        self.patch_room_basic("Shaktool Room", vec![(3, 0, P, D, W, W, O)])?;
        self.patch_room_basic(
            "Botwoon's Room",
            vec![(0, 0, D, P, W, W, O), (1, 0, E, D, W, W, O)],
        )?;
        self.patch_room_basic(
            "Crab Shaft",
            vec![(0, 2, W, W, E, E, O), (0, 3, W, E, P, W, O)],
        )?;
        self.patch_room_basic("Halfie Climb Room", vec![(0, 2, D, P, E, W, O)])?;
        self.patch_room_basic("The Precious Room", vec![(0, 0, D, E, W, P, O)])?;
        self.patch_room_basic("Northwest Maridia Bug Room", vec![(2, 1, P, E, W, W, O)])?;
        self.patch_room_basic("Pseudo Plasma Spark Room", vec![(1, 2, E, P, E, W, O)])?;
        self.patch_room_basic("Watering Hole", vec![(0, 1, W, W, E, P, O)])?;
        self.patch_room_basic(
            "East Tunnel",
            vec![(0, 0, W, E, W, P, O), (0, 1, D, D, E, W, O)],
        )?;
        self.patch_room_basic("Crab Tunnel", vec![(0, 0, D, P, W, W, O)])?;
        self.patch_room_basic("Crab Hole", vec![(0, 0, D, D, W, P, O)])?;
        self.patch_room_basic(
            "Fish Tank",
            vec![
                (1, 2, E, P, E, W, O),
                (2, 2, E, E, E, W, O),
                (3, 2, E, D, E, W, O),
            ],
        )?;
        self.patch_room_basic("Glass Tunnel", vec![(0, 1, D, D, P, P, O)])?;
        self.patch_room_basic("Main Street", vec![(1, 2, E, P, E, E, I)])?;
        self.patch_room_basic("Red Fish Room", vec![(2, 0, P, W, W, E, O)])?;

        // Norfair:
        self.patch_room_basic(
            "Post Crocomire Jump Room",
            vec![(3, 0, E, P, W, E, O), (4, 1, E, W, P, E, O)],
        )?;
        self.patch_room_basic("Crocomire Speedway", vec![(12, 2, P, D, E, D, O)])?;
        self.patch_room_basic("Hi Jump Energy Tank Room", vec![(1, 0, P, D, W, W, I)])?;
        self.patch_room_basic("Ice Beam Gate Room", vec![(3, 2, D, E, E, P, O)])?;
        self.patch_room_basic("Ice Beam Tutorial Room", vec![(0, 0, D, P, W, W, O)])?;
        self.patch_room_basic("Ice Beam Snake Room", vec![(0, 1, W, P, E, E, O)])?;
        self.patch_room_basic(
            "Bubble Mountain",
            vec![(0, 2, D, E, E, P, O), (1, 2, E, W, E, P, O)],
        )?;
        self.patch_room_basic("Green Bubbles Missile Room", vec![(1, 0, P, D, W, W, I)])?;
        self.patch_room_basic("Kronic Boost Room", vec![(1, 1, P, W, E, E, O)])?;
        self.patch_room_basic("Single Chamber", vec![(0, 0, D, P, W, E, O)])?;
        self.patch_room_basic("Volcano Room", vec![(1, 2, E, P, W, W, O)])?;
        self.patch_room_basic("Spiky Platforms Tunnel", vec![(2, 0, P, E, W, W, O)])?;
        self.patch_room_basic("Fast Pillars Setup Room", vec![(0, 1, D, W, E, P, O)])?;
        self.patch_room_basic(
            "Pillar Room",
            vec![
                (0, 0, D, P, W, W, O),
                (1, 0, E, P, W, W, O),
                (3, 0, P, D, W, W, O),
            ],
        )?;
        self.patch_room_basic(
            "Lower Norfair Fireflea Room",
            vec![(1, 0, P, D, W, E, O), (1, 3, D, P, E, W, O)],
        )?;
        self.patch_room_basic(
            "Lower Norfair Spring Ball Maze Room",
            vec![(2, 0, E, P, W, W, I)],
        )?;
        self.patch_room_basic("Mickey Mouse Room", vec![(3, 1, P, W, E, E, O)])?;
        self.patch_room_basic(
            "Red Kihunter Shaft",
            vec![(0, 0, D, D, W, P, O), (0, 4, W, P, E, W, O)],
        )?;
        self.patch_room_basic("Three Musketeers' Room", vec![(1, 2, P, E, E, W, O)])?;
        self.patch_room_basic(
            "Wasteland",
            vec![(1, 0, P, E, W, P, O), (5, 0, P, W, D, W, O)],
        )?;
        self.patch_room_basic(
            "Acid Statue Room",
            vec![
                (0, 1, W, E, E, P, O),
                (0, 2, W, E, E, W, O),
                (1, 2, E, E, E, W, O),
            ],
        )?;
        self.patch_room_basic("Screw Attack Room", vec![(0, 1, W, D, E, P, O)])?;
        self.patch_room_basic("Fast Ripper Room", vec![(3, 0, P, D, W, W, O)])?;
        self.patch_room_basic("Golden Torizo's Room", vec![(0, 0, D, W, W, P, I)])?;
        self.patch_room_basic("Lower Norfair Elevator", vec![(0, 0, D, D, W, P, V)])?;

        self.patch_room_basic("Big Boy Room", vec![(2, 0, P, E, W, W, O)])?;

        Ok(())
    }

    fn indicate_sand(&mut self) -> Result<()> {
        // Indicate sand transitions with a passage (4-pixel) on top and door (2-pixel) on bottom,
        // so it appears a bit like a funnel:

        self.patch_room_basic(
            "Aqueduct",
            vec![(1, 6, E, E, E, P, O), (3, 6, E, E, E, P, O)],
        )?;

        self.patch_room_basic("West Aqueduct Quicksand Room", vec![(0, 1, W, W, E, P, O)])?;
        self.patch_room_basic("East Aqueduct Quicksand Room", vec![(0, 1, W, W, E, P, O)])?;
        self.patch_room_basic("West Sand Hole", vec![(0, 1, W, E, E, P, O)])?;
        self.patch_room_basic("East Sand Hole", vec![(1, 1, E, W, E, P, O)])?;
        self.patch_room_basic(
            "Botwoon Energy Tank Room",
            vec![(2, 0, E, E, W, P, O), (3, 0, E, E, W, P, I)],
        )?;
        self.patch_room_basic(
            "Botwoon Quicksand Room",
            vec![(0, 0, W, E, D, P, O), (1, 0, E, W, D, P, O)],
        )?;
        self.patch_room_basic("Bug Sand Hole", vec![(0, 0, D, D, W, P, O)])?;
        self.patch_room_basic("Plasma Beach Quicksand Room", vec![(0, 0, W, W, D, P, O)])?;

        Ok(())
    }

    fn add_door_arrow(
        &mut self,
        room_idx: usize,
        door: &RoomGeometryDoor,
        right_arrow_tile: TilemapWord,
        down_arrow_tile: TilemapWord,
    ) -> Result<()> {
        let dir = &door.direction;
        let x = door.x as isize;
        let y = door.y as isize;
        if dir == "right" {
            self.patch_room(
                &self.game_data.room_geometry[room_idx].name,
                vec![(x + 1, y, right_arrow_tile)],
            )?;
        } else if dir == "left" {
            self.patch_room(
                &self.game_data.room_geometry[room_idx].name,
                vec![(x - 1, y, right_arrow_tile | FLIP_X)],
            )?;
        } else if dir == "down" {
            self.patch_room(
                &self.game_data.room_geometry[room_idx].name,
                vec![(x, y + 1, down_arrow_tile)],
            )?;
        } else if dir == "up" {
            self.patch_room(
                &self.game_data.room_geometry[room_idx].name,
                vec![(x, y - 1, down_arrow_tile | FLIP_Y)],
            )?;
        } else {
            bail!("Unrecognized door direction: {dir}");
        }
        Ok(())
    }

    fn add_door_letter(
        &mut self,
        room_idx: usize,
        door: &RoomGeometryDoor,
        letter_tile: TilemapWord,
    ) -> Result<()> {
        let dir = &door.direction;
        let x = door.x as isize;
        let y = door.y as isize;
        if dir == "right" {
            self.patch_room(
                &self.game_data.room_geometry[room_idx].name,
                vec![(x + 1, y, letter_tile)],
            )?;
        } else if dir == "left" {
            self.patch_room(
                &self.game_data.room_geometry[room_idx].name,
                vec![(x - 1, y, letter_tile)],
            )?;
        } else if dir == "down" {
            self.patch_room(
                &self.game_data.room_geometry[room_idx].name,
                vec![(x, y + 1, letter_tile)],
            )?;
        } else if dir == "up" {
            self.patch_room(
                &self.game_data.room_geometry[room_idx].name,
                vec![(x, y - 1, letter_tile)],
            )?;
        } else {
            bail!("Unrecognized door direction: {dir}");
        }
        Ok(())
    }

    fn add_cross_area_arrows(&mut self) -> Result<()> {
        // Replace colors to palette used for map tiles in the pause menu, for drawing arrows marking
        // cross-area connections:
        // TODO: Fix color of reserve energy in equipment menu, which gets messed up by this.
        fn rgb(r: u16, g: u16, b: u16) -> u16 {
            (b << 10) | (g << 5) | r
        }

        let extended_map_palette: Vec<(u8, u16)> = vec![
            (14, rgb(0, 24, 0)),  // Brinstar green
            (10, rgb(29, 0, 0)),  // Norfair red
            (8, rgb(4, 13, 31)),  // Maridia blue
            (9, rgb(24, 22, 6)),  // Wrecked Ship yellow
            (11, rgb(18, 3, 31)), // Crateria purple
            (6, rgb(27, 14, 0)),  // Tourian
        ];
        // Dotted grid lines
        let i = 12;
        let color = rgb(6, 6, 6);
        self.rom
            .write_u16(snes2pc(0xB6F000) + 2 * (0x40 + i), color as isize)?;

        for &(i, color) in &extended_map_palette {
            self.rom
                .write_u16(snes2pc(0xB6F000) + 2 * (0x20 + i as usize), color as isize)?;
            self.rom
                .write_u16(snes2pc(0xB6F000) + 2 * (0x60 + i as usize), color as isize)?;
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

        let letter_tiles: Vec<[[u8; 8]; 8]> = vec![
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 3, 3, 3, 0, 0],
                [0, 3, 3, 0, 0, 3, 3, 0],
                [0, 3, 3, 0, 0, 0, 0, 0],
                [0, 3, 3, 0, 0, 0, 0, 0],
                [0, 3, 3, 0, 0, 3, 3, 0],
                [0, 0, 3, 3, 3, 3, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 3, 3, 3, 3, 0, 0],
                [0, 3, 3, 0, 0, 3, 3, 0],
                [0, 3, 3, 3, 3, 3, 0, 0],
                [0, 3, 3, 0, 0, 3, 3, 0],
                [0, 3, 3, 0, 0, 3, 3, 0],
                [0, 3, 3, 3, 3, 3, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 3, 0, 0, 0, 3, 0],
                [0, 3, 3, 3, 0, 0, 3, 0],
                [0, 3, 3, 3, 3, 0, 3, 0],
                [0, 3, 3, 0, 3, 3, 3, 0],
                [0, 3, 3, 0, 0, 3, 3, 0],
                [0, 3, 3, 0, 0, 0, 3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 3, 0, 0, 0, 3, 0],
                [0, 3, 3, 0, 0, 0, 3, 0],
                [0, 3, 3, 0, 3, 0, 3, 0],
                [0, 3, 3, 3, 3, 3, 3, 0],
                [0, 3, 3, 3, 0, 3, 3, 0],
                [0, 3, 3, 0, 0, 0, 3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 3, 0, 0, 0, 3, 0],
                [0, 3, 3, 3, 0, 3, 3, 0],
                [0, 3, 3, 3, 3, 3, 3, 0],
                [0, 3, 3, 0, 3, 0, 3, 0],
                [0, 3, 3, 0, 0, 0, 3, 0],
                [0, 3, 3, 0, 0, 0, 3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 3, 3, 3, 3, 3, 0],
                [0, 0, 0, 3, 3, 0, 0, 0],
                [0, 0, 0, 3, 3, 0, 0, 0],
                [0, 0, 0, 3, 3, 0, 0, 0],
                [0, 0, 0, 3, 3, 0, 0, 0],
                [0, 0, 0, 3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ];

        let right_arrow_tile: [[u8; 8]; 8] = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 3, 3],
            [0, 3, 3, 3, 3, 3, 3, 3],
            [0, 0, 0, 0, 0, 3, 3, 0],
            [0, 0, 0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ];
        let down_arrow_tile: [[u8; 8]; 8] = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 3, 0, 0, 0],
            [0, 0, 0, 3, 3, 0, 0, 0],
            [0, 0, 0, 3, 3, 0, 0, 0],
            [0, 0, 0, 3, 3, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 3, 0],
            [0, 0, 3, 3, 3, 3, 0, 0],
            [0, 0, 0, 3, 3, 0, 0, 0],
        ];

        let mut right_arrow_tile_idxs: Vec<TilemapWord> = Vec::new();
        let mut down_arrow_tile_idxs: Vec<TilemapWord> = Vec::new();
        let mut letter_tile_idxs: Vec<TilemapWord> = Vec::new();
        if self.randomization.difficulty.transition_letters {
            for area in 0..NUM_AREAS {
                let color_number = area_arrow_colors[area] as u8;

                let letter_tile_data = letter_tiles[area]
                    .map(|row| row.map(|c| if c == 3 { color_number } else { c }));
                let letter_tile_idx = self.create_tile(letter_tile_data)?;
                letter_tile_idxs.push(letter_tile_idx);
            }
        } else {
            for area in 0..NUM_AREAS {
                let color_number = area_arrow_colors[area] as u8;

                let right_tile_data =
                    right_arrow_tile.map(|row| row.map(|c| if c == 3 { color_number } else { c }));
                let right_tile_idx = self.create_tile(right_tile_data)?;
                right_arrow_tile_idxs.push(right_tile_idx);

                let down_tile_data =
                    down_arrow_tile.map(|row| row.map(|c| if c == 3 { color_number } else { c }));
                let down_tile_idx = self.create_tile(down_tile_data)?;
                down_arrow_tile_idxs.push(down_tile_idx);
            }
        }

        for (src_ptr_pair, dst_ptr_pair, _) in &self.map.doors {
            let (src_room_idx, src_door_idx) =
                self.game_data.room_and_door_idxs_by_door_ptr_pair[src_ptr_pair];
            let (dst_room_idx, dst_door_idx) =
                self.game_data.room_and_door_idxs_by_door_ptr_pair[dst_ptr_pair];
            let src_area = self.map.area[src_room_idx];
            let dst_area = self.map.area[dst_room_idx];
            if src_area != dst_area {
                if self.randomization.difficulty.transition_letters {
                    self.add_door_letter(
                        src_room_idx,
                        &self.game_data.room_geometry[src_room_idx].doors[src_door_idx],
                        letter_tile_idxs[dst_area],
                    )?;
                    self.add_door_letter(
                        dst_room_idx,
                        &self.game_data.room_geometry[dst_room_idx].doors[dst_door_idx],
                        letter_tile_idxs[src_area],
                    )?;
                } else {
                    self.add_door_arrow(
                        src_room_idx,
                        &self.game_data.room_geometry[src_room_idx].doors[src_door_idx],
                        right_arrow_tile_idxs[dst_area],
                        down_arrow_tile_idxs[dst_area],
                    )?;
                    self.add_door_arrow(
                        dst_room_idx,
                        &self.game_data.room_geometry[dst_room_idx].doors[dst_door_idx],
                        right_arrow_tile_idxs[src_area],
                        down_arrow_tile_idxs[src_area],
                    )?;
                }
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

    fn set_map_stations_explored(&mut self) -> Result<()> {
        if self.randomization.difficulty.maps_revealed {
            self.rom.write_n(snes2pc(0xB5F000), &vec![0xFF; 0x600])?;
            self.rom.write_u16(snes2pc(0xB5F600), 0x003F)?;
            return Ok(());
        }
        self.rom.write_n(snes2pc(0xB5F000), &vec![0; 0x600])?;
        self.rom.write_u16(snes2pc(0xB5F600), 0x0000)?;
        if !self.randomization.difficulty.mark_map_stations {
            return Ok(());
        }
        for (room_idx, room) in self.game_data.room_geometry.iter().enumerate() {
            if !room.name.contains(" Map Room") {
                continue;
            }
            let area = self.map.area[room_idx];
            let x = self.rom.read_u8(room.rom_address + 2)?;
            let y = self.rom.read_u8(room.rom_address + 3)?;
            let (offset, bitmask) = xy_to_explored_bit_ptr(x, y);
            let base_ptr = 0xB5F000 + area * 0x100;
            self.rom
                .write_u8(snes2pc(base_ptr + offset as usize), bitmask as isize)?;
        }
        Ok(())
    }

    fn add_items_disappear_data(
        &mut self,
        area_data: &[Vec<(ItemIdx, TilemapOffset, TilemapWord, Interior)>],
    ) -> Result<()> {
        // Write per-area item listings, to be used by the patch `item_dots_disappear.asm`.
        let base_ptr = 0x83B000;
        let mut data_ptr = base_ptr + 24;
        for (area_idx, data) in area_data.iter().enumerate() {
            self.rom.write_u16(
                snes2pc(base_ptr + area_idx * 2),
                (data_ptr & 0xFFFF) as isize,
            )?;
            self.rom
                .write_u16(snes2pc(base_ptr + 12 + area_idx * 2), data.len() as isize)?;
            for interior in [
                Interior::Empty,
                Interior::Elevator,
                Interior::Item,
                Interior::MediumItem,
                Interior::MajorItem,
            ] {
                for &(item_idx, offset, word, interior1) in data {
                    if interior1 != interior {
                        continue;
                    }
                    assert!(interior != Interior::Empty && interior != Interior::Elevator);
                    self.rom
                        .write_u8(snes2pc(data_ptr), (item_idx as isize) >> 3)?; // item byte index
                    self.rom
                        .write_u8(snes2pc(data_ptr + 1), 1 << ((item_idx as isize) & 7))?; // item bitmask
                    self.rom.write_u16(snes2pc(data_ptr + 2), offset as isize)?; // tilemap offset
                    self.rom.write_u16(snes2pc(data_ptr + 4), word as isize)?; // tilemap word
                    data_ptr += 6;
                }
            }
        }
        assert!(data_ptr <= 0x83B300);
        Ok(())
    }

    fn indicate_major_items(&mut self) -> Result<()> {
        let markers = self.randomization.difficulty.item_markers;
        let mut area_data: Vec<Vec<(ItemIdx, TilemapOffset, TilemapWord, Interior)>> =
            vec![vec![]; 6];
        for (i, &item) in self.randomization.item_placement.iter().enumerate() {
            let (room_id, node_id) = self.game_data.item_locations[i];
            if room_id == 19
                && (self.randomization.difficulty.objectives == Objectives::Chozos
                    || self.randomization.difficulty.save_animals)
            {
                // With Chozos objective or "Save the animals" option, we don't draw item dot in Bomb Torizo Room since a boss X tile will be drawn instead.
                continue;
            }
            let item_ptr = self.game_data.node_ptr_map[&(room_id, node_id)];
            let item_idx = self.rom.read_u8(item_ptr + 4)? as usize;
            let room_ptr = self.game_data.room_ptr_by_id[&room_id];
            let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
            let room = &self.game_data.room_geometry[room_idx];
            let area = self.map.area[room_idx];
            let x0 = self.rom.read_u8(room.rom_address + 2)? as isize;
            let y0 = self.rom.read_u8(room.rom_address + 3)? as isize;
            let (x, y) = find_item_xy(item_ptr, &room.items)?;
            let base_ptr = self.game_data.area_map_ptrs[area] as usize;
            let offset = super::xy_to_map_offset(x0 + x, y0 + y) as usize;
            let tile0 = (self.rom.read_u16(base_ptr + offset)? & 0xC3FF) as TilemapWord;
            let orig_basic_tile = self
                .reverse_map
                .get(&tile0)
                .with_context(|| {
                    format!(
                        "Tile not found: {tile0} at ({x}, {y}) in {}",
                        self.game_data.room_geometry[room_idx].name
                    )
                })?
                .clone();
            let mut basic_tile = orig_basic_tile;
            basic_tile.faded = false;
            if basic_tile.interior == Interior::Empty {
                basic_tile.interior = Interior::Item;
            }
            let interior = match markers {
                ItemMarkers::Simple => Interior::Item,
                ItemMarkers::Majors => {
                    if item.is_unique() || item == Item::ETank || item == Item::ReserveTank {
                        Interior::MajorItem
                    } else {
                        Interior::Item
                    }
                }
                ItemMarkers::Uniques => {
                    if item.is_unique() {
                        Interior::MajorItem
                    } else {
                        Interior::Item
                    }
                }
                ItemMarkers::ThreeTiered => {
                    if item.is_unique() {
                        Interior::MajorItem
                    } else if item != Item::Missile {
                        Interior::MediumItem
                    } else {
                        Interior::Item
                    }
                }
            };
            basic_tile.interior = interior;
            let tile1 = self.get_basic_tile(basic_tile)?;
            area_data[area].push((item_idx, offset as TilemapOffset, tile1 | 0x0C00, interior));
            if interior == Interior::MajorItem
                || (interior == Interior::MediumItem
                    && orig_basic_tile.interior != Interior::MajorItem)
                || (interior == Interior::Item
                    && (orig_basic_tile.interior == Interior::Empty
                        || orig_basic_tile.interior == Interior::Item))
            {
                basic_tile.faded = true;
                let tile_faded = self.get_basic_tile(basic_tile)?;
                self.rom
                    .write_u16(base_ptr + offset, (tile_faded | 0x0C00) as isize)?;
            }
        }
        self.add_items_disappear_data(&area_data)?;
        Ok(())
    }

    fn fix_etank_color(&mut self) -> Result<()> {
        // let etank_tile: [[u8; 8]; 8] = [
        //     [0, 0, 0, 0, 0, 0, 0, 0],
        //     [0, 2, 2, 2, 2, 2, 2, 0],
        //     [0, 2, 3, 3, 3, 3, 3, 0],
        //     [0, 2, 3, 3, 3, 3, 3, 0],
        //     [0, 2, 3, 3, 3, 3, 3, 0],
        //     [0, 2, 3, 3, 3, 3, 3, 0],
        //     [0, 0, 0, 0, 0, 0, 0, 0],
        //     [0, 0, 0, 0, 0, 0, 0, 0],
        // ];
        // self.write_tile_2bpp(0x31, etank_tile, false)?;
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
        tiles_to_change.extend(0x10..0x30); // minimap dotted grid lines (skipping 0x0F = blank tile)
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

        // Patch slope tiles:
        let coords = vec![
            (0x28, vec![(2, 7), (4, 7), (6, 7)]),
            (0x29, vec![(0, 1), (0, 3), (0, 5)]),
            (0x2A, vec![(0, 4), (0, 5), (0, 7), (2, 7), (4, 7), (6, 7)]),
            (0x2B, vec![(0, 1)]),
        ];
        for (idx, v) in coords {
            let mut tile = self.read_map_tile_2bpp(idx)?;
            for (x, y) in v {
                tile[y][x] = 0;
            }
            self.write_tile_2bpp(idx, tile)?;
        }
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
        self.rom.write_u24(snes2pc(0x8FB817), TILE_GFX_ADDR_2BPP as isize + kraid_map_area.unwrap() * 0x10000)?;
        // Kraid dead:
        self.rom.write_u24(snes2pc(0x8FB842), TILE_GFX_ADDR_2BPP as isize + kraid_map_area.unwrap() * 0x10000)?;
        Ok(())
    }

    fn substitute_colors(&mut self, item_idx: usize, tiles: Vec<usize>, subst: Vec<(u8, u8)>) -> Result<()> {
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

    fn fix_acid(&mut self) -> Result<()> {
        // In the vanilla game, unlike other FX, acid uses palette 0 (same as Power Bomb doors). We made palette 0
        // not fade through transitions, since we use it for the non-empty Reserve Tank color
        // (and Samus location indicator on the mini-map). This wouldn't look right when entering
        // or exiting a room with acid, as the acid would appear/disappear abruptly. So we change
        // acid to use the FX palette (which normally would be palette 6, but we moved it to palette 7
        // so that unexplored map tiles could use palette 6).

        for addr in (snes2pc(0x8A8840)..snes2pc(0x8A9080)).step_by(2) {
            let word = self.rom.read_u16(addr)?;
            if word & 0x1C00 == 0x0000 {
                self.rom.write_u16(addr, word | 0x1C00)?;
            }
        }
        Ok(())
    }

    pub fn apply_patches(&mut self) -> Result<()> {
        self.initialize_tiles()?;
        self.fix_pause_palettes()?;
        self.fix_message_boxes()?;
        self.fix_hud_black()?;
        self.darken_hud_grid()?;
        self.fix_etank_color()?;
        self.index_vanilla_tiles()?;
        self.fix_elevators()?;
        self.fix_item_dots()?;
        self.fix_walls()?;
        self.indicate_passages()?;
        self.indicate_doors()?;
        self.indicate_heat()?;
        self.indicate_sand()?;
        self.indicate_special_tiles()?;
        self.add_cross_area_arrows()?;
        self.set_map_stations_explored()?;
        self.indicate_major_items()?;
        self.write_tiles()?;
        self.fix_fx_palettes()?;
        self.fix_kraid()?;
        self.fix_item_colors()?;
        self.fix_acid()?;
        // info!("Free tiles: {} (out of {})", self.free_tiles.len() - self.next_free_tile_idx, self.free_tiles.len());
        Ok(())
    }
}
