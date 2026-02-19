use hashbrown::{HashMap, HashSet};

use crate::{
    customize::{CustomizeSettings, ItemDotChange},
    randomize::{LockedDoor, Randomization},
    settings::{
        DisableETankSetting, DoorLocksSize, InitialMapRevealSettings, ItemMarkers, MapRevealLevel,
        MapStationReveal, Objective, RandomizerSettings,
    },
};
use maprando_game::{
    AreaIdx, BeamType, Direction, DoorLockType, DoorType, GameData, Item, ItemIdx, Map,
    MapLiquidType, MapTile, MapTileEdge, MapTileInterior, MapTileSpecialType, RoomGeometryDoor,
    RoomGeometryItem, RoomId, RoomPtr, util::sorted_hashmap_iter,
};

use super::{Rom, snes2pc, xy_to_explored_bit_ptr, xy_to_map_offset};
use anyhow::{Context, Result, bail};

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
    pub rom: &'a mut Rom,
    pub game_data: &'a GameData,
    pub map: &'a Map,
    pub settings: &'a RandomizerSettings,
    pub customize_settings: &'a CustomizeSettings,
    pub randomization: &'a Randomization,
    pub map_tile_map: HashMap<(AreaIdx, isize, isize), MapTile>,
    pub gfx_tile_map: HashMap<[[u8; 8]; 8], TilemapWord>,
    pub gfx_tile_reverse_map: HashMap<TilemapWord, [[u8; 8]; 8]>,
    pub free_tiles: Vec<TilemapWord>, // set of free tile indexes
    pub locked_door_state_indices: &'a [usize],
    pub dynamic_tile_data: Vec<Vec<(ItemIdx, RoomId, MapTile)>>,
    pub transition_tile_coords: Vec<(AreaIdx, isize, isize)>,
    pub area_min_x: [isize; NUM_AREAS],
    pub area_max_x: [isize; NUM_AREAS],
    pub area_min_y: [isize; NUM_AREAS],
    pub area_max_y: [isize; NUM_AREAS],
    pub area_offset_x: [isize; NUM_AREAS],
    pub area_offset_y: [isize; NUM_AREAS],
    pub room_map_gfx: HashMap<RoomPtr, Vec<TilemapWord>>,
    pub room_map_tilemap: HashMap<RoomPtr, Vec<TilemapWord>>,
    pub room_map_dynamic_tiles: HashMap<RoomPtr, Vec<(ItemIdx, TilemapOffset, TilemapWord)>>,
}

pub const VANILLA_ELEVATOR_TILE: TilemapWord = 0xCE; // Index of elevator tile in vanilla game
pub const ELEVATOR_TILE: TilemapWord = 0x12; // Index of elevator tile with TR's map patch
pub const TILE_GFX_ADDR_4BPP: usize = 0xE28000; // Where to store area-specific tile graphics (must agree with map_area.asm)
pub const TILE_GFX_ADDR_2BPP: usize = 0xE38000; // Where to store area-specific tile graphics (must agree with map_area.asm)

const FLIP_X: TilemapWord = 0x4000;
const FLIP_Y: TilemapWord = 0x8000;

pub fn find_item_xy(addr: usize, room_items: &[RoomGeometryItem]) -> Result<(isize, isize)> {
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

#[allow(clippy::needless_range_loop)]
pub fn diagonal_flip_tile(tile: [[u8; 8]; 8]) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            out[y][x] = tile[x][y];
        }
    }
    out
}

pub fn read_tile_4bpp(rom: &Rom, base_addr: usize, idx: usize) -> Result<[[u8; 8]; 8]> {
    let mut out: [[u8; 8]; 8] = [[0; 8]; 8];
    for y in 0..8 {
        let addr = base_addr + idx * 32 + y * 2;
        let data_0 = rom.read_u8(addr)?;
        let data_1 = rom.read_u8(addr + 1)?;
        let data_2 = rom.read_u8(addr + 16)?;
        let data_3 = rom.read_u8(addr + 17)?;
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

pub fn write_tile_4bpp(rom: &mut Rom, base_addr: usize, data: [[u8; 8]; 8]) -> Result<()> {
    for (y, row) in data.iter().enumerate() {
        let addr = base_addr + y * 2;
        let data_0: u8 = (0..8).map(|x| (row[x] & 1) << (7 - x)).sum();
        let data_1: u8 = (0..8).map(|x| ((row[x] >> 1) & 1) << (7 - x)).sum();
        let data_2: u8 = (0..8).map(|x| ((row[x] >> 2) & 1) << (7 - x)).sum();
        let data_3: u8 = (0..8).map(|x| ((row[x] >> 3) & 1) << (7 - x)).sum();
        rom.write_u8(addr, data_0 as isize)?;
        rom.write_u8(addr + 1, data_1 as isize)?;
        rom.write_u8(addr + 16, data_2 as isize)?;
        rom.write_u8(addr + 17, data_3 as isize)?;
    }
    Ok(())
}

fn draw_edge(
    tile_side: TileSide,
    edge: MapTileEdge,
    tile: &mut [[u8; 8]; 8],
    settings: &RandomizerSettings,
) {
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
            if settings.other_settings.ultra_low_qol {
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
            if settings.other_settings.ultra_low_qol {
                set_wall_pixel(tile, 2, 3);
                set_wall_pixel(tile, 3, 3);
                set_wall_pixel(tile, 4, 3);
                set_wall_pixel(tile, 5, 3);
            }
            set_wall_pixel(tile, 6, 3);
            set_wall_pixel(tile, 7, 3);
        }
        MapTileEdge::QolPassage => {
            if !settings.other_settings.ultra_low_qol {
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
            if settings.other_settings.ultra_low_qol {
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
            if settings.other_settings.ultra_low_qol {
                set_wall_pixel(tile, 0, 3);
                set_wall_pixel(tile, 1, 3);
                set_wall_pixel(tile, 2, 3);
                set_wall_pixel(tile, 3, 3);
                set_wall_pixel(tile, 4, 3);
                set_wall_pixel(tile, 5, 3);
                set_wall_pixel(tile, 6, 3);
                set_wall_pixel(tile, 7, 3);
            } else if tile_side == TileSide::Bottom {
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
        MapTileEdge::ElevatorEntrance => {
            set_wall_pixel(tile, 0, 3);
            set_wall_pixel(tile, 1, 3);
            set_wall_pixel(tile, 6, 3);
            set_wall_pixel(tile, 7, 3);
            set_air_pixel(tile, 0, 3);
            set_air_pixel(tile, 7, 3);
        }
        MapTileEdge::LockedDoor(lock_type) => match lock_type {
            Gray | Red | Green | Yellow => {
                let color = match lock_type {
                    DoorLockType::Gray => 15,
                    DoorLockType::Red => 7,
                    DoorLockType::Green => 14,
                    DoorLockType::Yellow => 6,
                    _ => panic!("Internal error"),
                };
                match settings.other_settings.door_locks_size {
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
            }
            Charge | Ice | Wave | Spazer | Plasma => {
                let color = match lock_type {
                    Charge => 15,
                    Ice => 8,
                    Wave => 7,
                    Spazer => 6,
                    Plasma => 14,
                    _ => panic!("Internal error"),
                };
                match settings.other_settings.door_locks_size {
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
            Wall => {
                set_wall_pixel(tile, 0, 3);
                set_wall_pixel(tile, 1, 3);
                set_wall_pixel(tile, 2, 3);
                set_wall_pixel(tile, 3, 13);
                set_wall_pixel(tile, 4, 13);
                set_wall_pixel(tile, 5, 3);
                set_wall_pixel(tile, 6, 3);
                set_wall_pixel(tile, 7, 3);
            }
        },
    }
}

pub fn render_tile(
    tile: MapTile,
    settings: &RandomizerSettings,
    customize_settings: &CustomizeSettings,
) -> Result<[[u8; 8]; 8]> {
    let bg_color = if tile.heated && !settings.other_settings.ultra_low_qol {
        2
    } else {
        1
    };
    let mut data: [[u8; 8]; 8] = [[bg_color; 8]; 8];

    let liquid_colors = match (tile.liquid_type, tile.heated) {
        (MapLiquidType::None, _) => (bg_color, bg_color),
        (MapLiquidType::Water, false) => (4, 1),
        (MapLiquidType::Lava, true) => (2, 1),
        (MapLiquidType::Acid, false) => (1, 2),
        (MapLiquidType::Acid, true) => (2, 1),
        _ => panic!("unexpected liquid type"),
    };
    if let Some(liquid_level) = tile.liquid_level
        && !settings.other_settings.ultra_low_qol
    {
        let level = (liquid_level * 8.0).floor() as isize;
        for y in level..8 {
            for x in 0..8 {
                match tile.liquid_type {
                    MapLiquidType::Water => {
                        if (x + y) % 2 == 0 {
                            data[y as usize][x as usize] = liquid_colors.0;
                        } else {
                            data[y as usize][x as usize] = liquid_colors.1;
                        }
                    }
                    MapLiquidType::Lava => {
                        data[y as usize][x as usize] = liquid_colors.1;
                    }
                    MapLiquidType::Acid => {
                        if (x + y) % 2 == 0 {
                            data[y as usize][x as usize] = liquid_colors.1;
                        }
                    }
                    MapLiquidType::None => bail!("unexpected liquid type None"),
                }
            }
        }

        if tile.faded
            && tile.interior.is_item()
            && (tile.liquid_type == MapLiquidType::Lava
                || tile.liquid_type == MapLiquidType::Acid
                || (tile.liquid_type == MapLiquidType::Water && tile.liquid_level.unwrap() > 0.5))
        {
            // Improve contrast around faded items:
            match tile.interior {
                MapTileInterior::Item => {
                    for y in 2..6 {
                        for x in 2..6 {
                            data[y][x] = bg_color;
                        }
                    }
                }
                _ => {
                    for y in 1..7 {
                        for x in 1..7 {
                            data[y][x] = bg_color;
                        }
                    }
                }
            }
        }
    };

    let item_color = if tile.faded {
        if tile.heated { 1 } else { 2 }
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
            data[3][5] = item_color;
            data[4][2] = item_color;
            data[4][5] = item_color;
            data[5][3] = item_color;
            data[5][4] = item_color;
            if let Some(liquid_level) = tile.liquid_level
                && liquid_level < 0.5
            {
                data[3][3] = liquid_colors.0;
                data[3][4] = liquid_colors.0;
                data[4][3] = liquid_colors.0;
                data[4][4] = liquid_colors.0;
            }
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
            if settings.other_settings.ultra_low_qol {
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
            if settings.other_settings.ultra_low_qol {
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
            if settings.other_settings.ultra_low_qol {
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
            if settings.other_settings.ultra_low_qol {
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
        if tile.heated && !settings.other_settings.ultra_low_qol {
            d.map(|row| row.map(|c| if c == 1 { 2 } else { c }))
        } else {
            d
        }
    };
    match tile.special_type {
        Some(MapTileSpecialType::AreaTransition(area_idx, dir)) => {
            if customize_settings.transition_letters {
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
                    _ => panic!("Unexpected area {area_idx}"),
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
        || (!settings.other_settings.ultra_low_qol
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
        draw_edge(TileSide::Top, tile.top, &mut data, settings);
        draw_edge(TileSide::Bottom, tile.bottom, &mut data, settings);
        draw_edge(TileSide::Left, tile.left, &mut data, settings);
        draw_edge(TileSide::Right, tile.right, &mut data, settings);
    }
    Ok(data)
}

pub fn get_item_interior(item: Item, settings: &RandomizerSettings) -> MapTileInterior {
    match settings.quality_of_life_settings.item_markers {
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
    }
}

pub fn apply_item_interior(
    tile: MapTile,
    item: Item,
    settings: &RandomizerSettings,
) -> MapTileInterior {
    let item_interior = get_item_interior(item, settings);
    use MapTileInterior::{AmmoItem, Item, MajorItem, MediumItem};
    match (tile.interior, item_interior) {
        (MajorItem, _) | (_, MajorItem) => MajorItem,
        (MediumItem, _) | (_, MediumItem) => MediumItem,
        (AmmoItem, _) | (_, AmmoItem) => AmmoItem,
        (Item, _) | (_, Item) => Item,
        _ => panic!("unexpected item interior"),
    }
}

pub fn apply_door_lock(
    tile: &MapTile,
    locked_door: &LockedDoor,
    door: &RoomGeometryDoor,
) -> MapTile {
    let lock_type = match locked_door.door_type {
        DoorType::Blue => panic!("unexpected blue door lock"),
        DoorType::Gray => panic!("unexpected gray door lock"),
        DoorType::Wall => DoorLockType::Wall,
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
    new_tile
}

pub fn get_gray_doors() -> Vec<(RoomId, isize, isize, Direction)> {
    use Direction::{Down, Left, Right, Up};
    vec![
        // Pirate rooms:
        (12, 0, 0, Left),
        (12, 2, 0, Right), // Pit Room
        (82, 0, 0, Left),
        (82, 5, 0, Right), // Baby Kraid Room
        (219, 0, 0, Left), // Plasma Room
        (139, 0, 0, Left),
        (139, 2, 0, Right), // Metal Pirates Room
        // Boss rooms:
        (84, 0, 1, Left),
        (84, 1, 1, Right), // Kraid Room
        (158, 0, 0, Left), // Phantoon's Room
        (193, 0, 1, Left),
        (193, 1, 0, Right), // Draygon's Room
        (142, 0, 0, Right),
        (142, 0, 1, Left), // Ridley's Room
        // Miniboss rooms:
        (19, 0, 0, Left),   // Bomb Torizo Room
        (57, 0, 2, Down),   // Spore Spawn Room
        (122, 3, 0, Up),    // Crocomire's Room
        (185, 0, 0, Left),  // Botwoon's Room
        (150, 1, 1, Right), // Golden Torizo's Room
    ]
}

pub fn get_objective_tiles(objectives: &[Objective]) -> Vec<(RoomId, usize, usize)> {
    use Objective::*;
    let mut out: Vec<(RoomId, usize, usize)> = vec![];
    for objective in objectives {
        match objective {
            Kraid => {
                out.push((84, 0, 0));
                out.push((84, 1, 0));
                out.push((84, 0, 1));
                out.push((84, 1, 1));
            }
            Phantoon => {
                out.push((158, 0, 0));
            }
            Draygon => {
                out.push((193, 0, 0));
                out.push((193, 1, 0));
                out.push((193, 0, 1));
                out.push((193, 1, 1));
            }
            Ridley => {
                out.push((142, 0, 0));
                out.push((142, 0, 1));
            }
            SporeSpawn => {
                out.push((57, 0, 0));
                out.push((57, 0, 1));
                out.push((57, 0, 2));
            }
            Crocomire => {
                out.push((122, 0, 0));
                out.push((122, 1, 0));
                out.push((122, 2, 0));
                out.push((122, 3, 0));
                out.push((122, 4, 0));
                out.push((122, 5, 0));
                out.push((122, 6, 0));
                // We don't mark the last tile, so the item can still be visible.
            }
            Botwoon => {
                out.push((185, 0, 0));
                out.push((185, 1, 0));
            }
            GoldenTorizo => {
                out.push((150, 0, 1));
                out.push((150, 1, 1));
                // We don't mark the top row of tiles, so the items can still be visible.
            }
            MetroidRoom1 => {
                out.push((226, 0, 0));
                out.push((226, 1, 0));
                out.push((226, 2, 0));
                out.push((226, 3, 0));
                out.push((226, 4, 0));
                out.push((226, 5, 0));
            }
            MetroidRoom2 => {
                out.push((227, 0, 0));
                out.push((227, 0, 1));
            }
            MetroidRoom3 => {
                out.push((228, 0, 0));
                out.push((228, 1, 0));
                out.push((228, 2, 0));
                out.push((228, 3, 0));
                out.push((228, 4, 0));
                out.push((228, 5, 0));
            }
            MetroidRoom4 => {
                out.push((229, 0, 0));
                out.push((229, 0, 1));
            }
            BombTorizo => {
                out.push((19, 0, 0));
            }
            BowlingStatue => {
                out.push((161, 4, 1));
            }
            AcidChozoStatue => {
                out.push((149, 0, 0));
            }
            PitRoom => {
                out.push((12, 0, 0));
                out.push((12, 1, 0));
                out.push((12, 2, 0));
            }
            BabyKraidRoom => {
                out.push((82, 0, 0));
                out.push((82, 1, 0));
                out.push((82, 2, 0));
                out.push((82, 3, 0));
                out.push((82, 4, 0));
                out.push((82, 5, 0));
            }
            PlasmaRoom => {
                out.push((219, 0, 0));
                out.push((219, 1, 0));
                out.push((219, 0, 1));
                out.push((219, 1, 1));
                out.push((219, 0, 2));
            }
            MetalPiratesRoom => {
                out.push((139, 0, 0));
                out.push((139, 1, 0));
                out.push((139, 2, 0));
            }
        }
    }

    // Mother Brain Room:
    out.push((238, 0, 0));
    out.push((238, 1, 0));
    out.push((238, 2, 0));
    out.push((238, 3, 0));

    out
}

impl<'a> MapPatcher<'a> {
    pub fn new(
        rom: &'a mut Rom,
        game_data: &'a GameData,
        map: &'a Map,
        settings: &'a RandomizerSettings,
        customize_settings: &'a CustomizeSettings,
        randomization: &'a Randomization,
        locked_door_state_indices: &'a [usize],
    ) -> Self {
        let mut reserved_tiles: HashSet<TilemapWord> = vec![
            // Used on HUD: (skipping "%", which is unused)
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x10, // slope tile that triggers tile above Samus to be marked explored
            0x11, // heated slope tile that triggers tile above Samus to be marked explored
            0x12, // reserved for partially revealed door tile, next to 2-sided save/refill rooms
            0x1C, 0x1D, 0x1E, 0x1F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x38, 0x39, 0x3A,
            0x3B,
            // Max ammo display digits: (removed in favor of normal digit graphics)
            // 0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45,
            0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D,
            0xA8, // heated slope tile corresponding to 0x28
            // Message box letters and punctuation (skipping unused ones: "Q", "->", "'", "-")
            0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD,
            0xCE, 0xCF, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDD, 0xDE,
            0xDF, // Vanilla buttons at the bottom left and right of the pause screen
            0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED,
            0xEE, 0xEF, 0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB,
            0xFC, 0xFD, 0xFE, 0xFF,  // Colon (used on objective screen)
            0x106, // Green checkmark (used on objective screen)
            0x10B, // OBJ button at bottom-left of pause screen
            0x10C, 0x10D, 0x10E, 0x11C, 0x11D, 0x11E,
            // Vanilla buttons at bottom-center of pause screen
            0x290, 0x291, 0x292, 0x2A0, 0x2A1, 0x2A2, 0x2A3, 0x2B0, 0x2B1, 0x2B2, 0x2B3, 0x2B8,
            0x2C0, 0x2C1, 0x2C2, 0x2C3, // Sprite tiles:
            0x228, 0x229, 0x22A, 0x22E, 0x23C, 0x23D, 0x243, 0x251, 0x29D, 0x29E, 0x2AF, 0x2B4,
            0x2B5, 0x2B6, 0x2C4, 0x2C5, 0x2C6, 0x2C7,
        ]
        .into_iter()
        .collect();

        if settings.quality_of_life_settings.disableable_etanks != DisableETankSetting::Off {
            // Reserve tile $2F for disabled ETank
            reserved_tiles.insert(0x2F);
        }

        let mut free_tiles: Vec<TilemapWord> = Vec::new();
        for word in 0..768 {
            if !reserved_tiles.contains(&word) {
                free_tiles.push(word);
            }
        }
        free_tiles.reverse();
        println!("total available tiles: {}", free_tiles.len());

        MapPatcher {
            rom,
            game_data,
            map,
            settings,
            customize_settings,
            randomization,
            map_tile_map: HashMap::new(),
            gfx_tile_map: HashMap::new(),
            gfx_tile_reverse_map: HashMap::new(),
            free_tiles,
            locked_door_state_indices,
            dynamic_tile_data: vec![vec![]; 6],
            transition_tile_coords: vec![],
            area_min_x: [isize::MAX; NUM_AREAS],
            area_min_y: [isize::MAX; NUM_AREAS],
            area_max_x: [0; NUM_AREAS],
            area_max_y: [0; NUM_AREAS],
            area_offset_x: [0; NUM_AREAS],
            area_offset_y: [0; NUM_AREAS],
            room_map_gfx: HashMap::new(),
            room_map_tilemap: HashMap::new(),
            room_map_dynamic_tiles: HashMap::new(),
        }
    }

    fn index_fixed_tiles(&mut self) -> Result<()> {
        let mut tile = MapTile {
            special_type: Some(MapTileSpecialType::SlopeUpFloorLow),
            ..MapTile::default()
        };
        self.index_tile(tile.clone(), Some(0x10))?;
        self.write_hud_tile_2bpp(0x10, self.render_tile(tile.clone())?)?;
        tile.heated = true;
        self.index_tile(tile.clone(), Some(0x11))?;
        self.write_hud_tile_2bpp(0x11, self.render_tile(tile.clone())?)?;

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
        self.write_map_tile_4bpp(0x12, data)?;
        self.write_hud_tile_2bpp(0x12, data)?;
        Ok(())
    }

    fn read_hud_tile_2bpp(&self, idx: usize) -> Result<[[u8; 8]; 8]> {
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

    fn _read_map_tile_4bpp(&self, idx: usize) -> Result<[[u8; 8]; 8]> {
        let addr = snes2pc(0xB68000) + idx * 32;
        self.read_tile_4bpp(addr)
    }

    fn index_tile(&mut self, tile: MapTile, fixed_idx: Option<u16>) -> Result<TilemapWord> {
        let data = self.render_tile(tile)?;
        if self.gfx_tile_map.contains_key(&data) {
            Ok(self.gfx_tile_map[&data])
        } else if self.gfx_tile_map.contains_key(&hflip_tile(data)) {
            Ok(self.gfx_tile_map[&hflip_tile(data)] | FLIP_X)
        } else if self.gfx_tile_map.contains_key(&vflip_tile(data)) {
            Ok(self.gfx_tile_map[&vflip_tile(data)] | FLIP_Y)
        } else if self
            .gfx_tile_map
            .contains_key(&hflip_tile(vflip_tile(data)))
        {
            Ok(self.gfx_tile_map[&hflip_tile(vflip_tile(data))] | FLIP_X | FLIP_Y)
        } else {
            let tile_idx = if let Some(i) = fixed_idx {
                i
            } else {
                self.free_tiles.pop().context("No more free tiles")?
            };
            let palette = 0x1C00;
            let word = tile_idx | palette;
            self.gfx_tile_map.insert(data, word);
            self.gfx_tile_reverse_map.insert(word & 0x3FF, data);
            Ok(word)
        }
    }

    fn render_tile(&self, tile: MapTile) -> Result<[[u8; 8]; 8]> {
        render_tile(tile, self.settings, self.customize_settings)
    }

    fn write_map_tiles(&mut self) -> Result<()> {
        // Clear all map tilemap data:
        for area_ptr in &self.game_data.area_map_ptrs {
            for i in 0..(64 * 32) {
                self.rom.write_u16((area_ptr + i * 2) as usize, 0x001F)?;
            }
        }

        // Index map graphics and write map tilemap by room:
        for (&(area_idx, x, y), tile) in sorted_hashmap_iter(&self.map_tile_map.clone()) {
            let word = self.index_tile(tile.clone(), None)?;
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
                let _ = self.index_tile(tile, None)?;
            }
        }

        // Write map tile graphics:
        for (data, word) in self.gfx_tile_map.clone() {
            let idx = (word & 0x3FF) as usize;
            self.write_map_tile_2bpp(idx, data)?;
            self.write_map_tile_4bpp(idx, data)?;
        }

        for &idx in &self.free_tiles.clone() {
            // Placeholder for unused tiles:
            let data = [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 3, 0, 0],
                [0, 0, 0, 3, 3, 0, 0, 0],
                [0, 0, 0, 3, 3, 0, 0, 0],
                [0, 0, 3, 0, 0, 3, 0, 0],
                [0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ];
            self.write_map_tile_2bpp(idx as usize, data)?;
            self.write_map_tile_4bpp(idx as usize, data)?;
        }

        // Write room map offsets:
        for (room_idx, room) in self.game_data.room_geometry.iter().enumerate() {
            if !self.randomization.map.room_mask[room_idx] {
                continue;
            }
            let area = self.randomization.map.area[room_idx];
            let room_x = self.map.rooms[room_idx].0 as isize - self.area_offset_x[area];
            let room_y = self.map.rooms[room_idx].1 as isize - self.area_offset_y[area];
            self.rom.write_u8(room.rom_address + 2, room_x)?;
            self.rom.write_u8(room.rom_address + 3, room_y)?;

            if let Some(twin_address) = room.twin_rom_address {
                match twin_address {
                    0x7D69A => {
                        // East Pants Room:
                        self.rom.write_u8(twin_address + 2, room_x + 1)?;
                        self.rom.write_u8(twin_address + 3, room_y + 1)?;
                    }
                    0x7968F => {
                        // Homing Geemer Room:
                        self.rom.write_u8(twin_address + 2, room_x + 5)?;
                        self.rom.write_u8(twin_address + 3, room_y + 2)?;
                    }
                    _ => bail!("Unrecognized twin_address: {}", twin_address),
                }
            }
        }
        println!("free tiles={}", self.free_tiles.len());
        Ok(())
    }

    fn find_tile(
        data: [[u8; 8]; 8],
        gfx_tile_map: &HashMap<[[u8; 8]; 8], TilemapWord>,
    ) -> Option<TilemapWord> {
        if gfx_tile_map.contains_key(&data) {
            Some(gfx_tile_map[&data])
        } else if gfx_tile_map.contains_key(&hflip_tile(data)) {
            Some(gfx_tile_map[&hflip_tile(data)] | FLIP_X)
        } else if gfx_tile_map.contains_key(&vflip_tile(data)) {
            Some(gfx_tile_map[&vflip_tile(data)] | FLIP_Y)
        } else if gfx_tile_map.contains_key(&hflip_tile(vflip_tile(data))) {
            Some(gfx_tile_map[&hflip_tile(vflip_tile(data))] | FLIP_X | FLIP_Y)
        } else {
            None
        }
    }

    fn create_room_map_tilemaps(&mut self) -> Result<()> {
        let mut dynamic_tiles_by_coords: HashMap<(AreaIdx, isize, isize), Vec<(ItemIdx, MapTile)>> =
            HashMap::new();
        for data in &self.dynamic_tile_data {
            for (item_idx, room_id, map_tile) in data.iter().cloned() {
                let room_ptr = self.game_data.room_ptr_by_id[&room_id];
                let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
                let area = self.map.area[room_idx];
                let coords = (area, map_tile.coords.0 as isize, map_tile.coords.1 as isize);
                if !dynamic_tiles_by_coords.contains_key(&coords) {
                    dynamic_tiles_by_coords.insert(coords, vec![]);
                }
                dynamic_tiles_by_coords
                    .get_mut(&coords)
                    .unwrap()
                    .push((item_idx, map_tile));
            }
        }

        for &room_ptr in &self.game_data.room_ptrs {
            let room_id = self.game_data.raw_room_id_by_ptr[&room_ptr];
            let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
            if !self.map.room_mask[room_idx] {
                continue;
            }
            let room_width = self.rom.read_u8(room_ptr + 4)?;
            let room_height = self.rom.read_u8(room_ptr + 5)?;

            let mut gfx_tiles: Vec<TilemapWord> = vec![];
            let mut gfx_tile_map: HashMap<[[u8; 8]; 8], TilemapWord> = HashMap::new();
            let mut next_tile = 0x50; // Starting tile number where map tiles are written in BG3
            let mut tilemap: Vec<TilemapWord> = vec![];
            let empty_tile = [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0],
            ];
            gfx_tile_map.insert(empty_tile, 0x001F);
            let mut slope = MapTile {
                special_type: Some(MapTileSpecialType::SlopeUpFloorLow),
                ..MapTile::default()
            };
            gfx_tile_map.insert(self.render_tile(slope.clone())?, 0x10);
            slope.heated = true;
            gfx_tile_map.insert(self.render_tile(slope)?, 0x11);

            let mut add_tile = |data| -> TilemapWord {
                if Self::find_tile(data, &gfx_tile_map).is_none() {
                    if let Some(t) = Self::find_tile(data, &self.gfx_tile_map) {
                        let idx = t & 0x3FF;
                        gfx_tiles.push(idx);
                        gfx_tile_map.insert(self.gfx_tile_reverse_map[&idx], next_tile);
                        next_tile += 1;
                    } else {
                        panic!("Tile not found in global map tileset: {data:?}");
                    }
                }
                Self::find_tile(data, &gfx_tile_map).unwrap()
            };
            let mut dynamic_tile_data: Vec<(ItemIdx, TilemapOffset, TilemapWord)> = vec![];

            for y in -1..room_height + 1 {
                for x in -2..room_width + 2 {
                    let (area, x1, y1) = self.get_room_coords(room_id, x, y).unwrap();
                    let tile = self.map_tile_map.get(&(area, x1, y1));
                    let data: [[u8; 8]; 8] = if let Some(x) = tile {
                        self.render_tile(x.clone())?
                    } else {
                        empty_tile
                    };
                    let word = add_tile(data);
                    tilemap.push(word);

                    let empty_vec = vec![];
                    for (item_idx, tile) in dynamic_tiles_by_coords
                        .get(&(area, x1, y1))
                        .unwrap_or(&empty_vec)
                    {
                        let data = self.render_tile(tile.clone())?;
                        let word = add_tile(data);
                        let local_x = x1 - self.area_offset_x[area];
                        let local_y = y1 - self.area_offset_y[area];
                        let offset = xy_to_map_offset(local_x, local_y) as TilemapOffset;
                        dynamic_tile_data.push((*item_idx, offset, word));
                    }
                }
            }

            self.room_map_gfx.insert(room_ptr, gfx_tiles);
            self.room_map_tilemap.insert(room_ptr, tilemap);
            self.room_map_dynamic_tiles
                .insert(room_ptr, dynamic_tile_data);
        }

        Ok(())
    }

    fn write_tile_2bpp(
        &mut self,
        base_addr: usize,
        idx: usize,
        mut data: [[u8; 8]; 8],
    ) -> Result<()> {
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

    fn write_map_tile_2bpp(&mut self, idx: usize, data: [[u8; 8]; 8]) -> Result<()> {
        let base_addr = snes2pc(TILE_GFX_ADDR_2BPP);
        self.write_tile_2bpp(base_addr, idx, data)
    }

    fn write_hud_tile_2bpp(&mut self, idx: usize, data: [[u8; 8]; 8]) -> Result<()> {
        let base_addr = snes2pc(0x9AB200); // Standard BG3 tiles
        self.write_tile_2bpp(base_addr, idx, data)
    }

    fn write_tile_4bpp(&mut self, base_addr: usize, data: [[u8; 8]; 8]) -> Result<()> {
        write_tile_4bpp(self.rom, base_addr, data)
    }

    fn write_map_tile_4bpp(&mut self, idx: usize, data: [[u8; 8]; 8]) -> Result<()> {
        let base_addr = snes2pc(TILE_GFX_ADDR_4BPP); // Location of pause-menu tile GFX in ROM
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

    fn indicate_objective_tiles(&mut self) -> Result<()> {
        for (room_id, x, y) in get_objective_tiles(&self.randomization.objectives) {
            let Some(tile) = self.get_room_tile(room_id, x as isize, y as isize) else {
                continue;
            };
            tile.interior = MapTileInterior::Objective;
        }
        Ok(())
    }

    fn indicate_gray_doors(&mut self) -> Result<()> {
        // Indicate gray doors by a gray bubble with black border. Some of these may be later overwritten
        // by an X depending on the objective setting.
        let gray_door = MapTileEdge::LockedDoor(DoorLockType::Gray);
        for (room_id, x, y, dir) in get_gray_doors() {
            let Some(tile) = self.get_room_tile(room_id, x, y) else {
                continue;
            };
            match dir {
                Direction::Left => {
                    tile.left = gray_door;
                }
                Direction::Right => {
                    tile.right = gray_door;
                }
                Direction::Up => {
                    tile.top = gray_door;
                }
                Direction::Down => {
                    tile.bottom = gray_door;
                }
            }
        }
        Ok(())
    }

    fn indicate_locked_doors(&mut self) -> Result<()> {
        for pass in [0, 1] {
            for (i, locked_door) in self.randomization.locked_doors.iter().enumerate() {
                // Process wall doors on a first pass, all other door types on second pass.
                let valid = match (pass, locked_door.door_type) {
                    (0, DoorType::Wall) => true,
                    (0, _) => false,
                    (1, DoorType::Wall) => false,
                    (1, _) => true,
                    _ => panic!("internal error"),
                };
                if !valid {
                    continue;
                }
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
                    let Some(tile) = self.get_room_tile(room_id, door.x as isize, door.y as isize)
                    else {
                        continue;
                    };
                    let new_tile = apply_door_lock(tile, locked_door, door);

                    if locked_door.door_type == DoorType::Wall {
                        // Walls are permanent, so we apply the change to the tile directly.
                        // This is necessary in order to support multiple walls on the same tile.
                        *tile = new_tile;
                    } else {
                        // Here, to make doors disappear once unlocked, we're (slightly awkwardly) reusing the mechanism for
                        // making item dots disappear. Door bits are stored at $D8B0, which is 512 bits after $D870 where
                        // the item bits start.
                        let item_idx = self.locked_door_state_indices[i] + 512;
                        self.dynamic_tile_data[area].push((item_idx, room_id, new_tile));
                    }
                }
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

        let tile = MapTile {
            special_type: Some(MapTileSpecialType::AreaTransition(other_area, direction)),
            ..MapTile::default()
        };
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
                .write_u16(snes2pc(0xB6F000) + 2 * (0x70 + i as usize), color as isize)?;
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

    fn fix_equipment_graphics(&mut self) -> Result<()> {
        // For equipment screen graphics using 4bpp palette 1, move white color 14 to color 15.
        let equip_tile_idxs = vec![
            0x119, 0x11A, 0x11B, 0x173, 0x174, 0x178, 0x184, 0x185, 0x188, 0x195, 0x196, 0x197,
            0x19C, 0x19D, 0x1E3, 0x1E4, 0x1E5, 0x1E6, 0x1E7, 0x17C, 0x17D, 0x17E, 0x17F, 0x1A0,
            0x1A1, 0x1ED, 0x1EE, 0x1FE,
        ];
        for idx in equip_tile_idxs {
            let mut data = read_tile_4bpp(self.rom, snes2pc(0xB68000), idx)?;
            for y in 0..8 {
                for x in 0..8 {
                    if data[y][x] == 14 {
                        data[y][x] = 15;
                    }
                }
            }
            write_tile_4bpp(self.rom, snes2pc(0xB68000) + idx * 32, data)?;
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
                .read_u16(snes2pc(0xB6F000) + 2 * (0x20 + i as usize))?;
            self.rom
                .write_u16(snes2pc(0xB6F000) + 2 * (0x40 + i as usize), color as isize)?;
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

    fn get_initial_tile_reveal(
        tile: &MapTile,
        imr_settings: &InitialMapRevealSettings,
    ) -> MapRevealLevel {
        if let Some(MapTileSpecialType::AreaTransition(_, _)) = tile.special_type {
            return match imr_settings.area_transitions {
                MapRevealLevel::No => MapRevealLevel::No,
                // Area transition tiles (arrows/letters) wouldn't render correctly
                // with partial reveal, so we override it to full reveal:
                MapRevealLevel::Partial | MapRevealLevel::Full => MapRevealLevel::Full,
            };
        }
        match tile.interior {
            MapTileInterior::MapStation => imr_settings.map_stations,
            MapTileInterior::SaveStation => imr_settings.save_stations,
            MapTileInterior::EnergyRefill => imr_settings.refill_stations,
            MapTileInterior::AmmoRefill => imr_settings.refill_stations,
            MapTileInterior::DoubleRefill => imr_settings.refill_stations,
            MapTileInterior::Ship => imr_settings.ship,
            MapTileInterior::Item => imr_settings.items1,
            MapTileInterior::AmmoItem => imr_settings.items2,
            MapTileInterior::MediumItem => imr_settings.items3,
            MapTileInterior::MajorItem => imr_settings.items4,
            MapTileInterior::Objective => imr_settings.objectives,
            _ => imr_settings.other,
        }
    }

    fn set_initial_map(&mut self) -> Result<()> {
        let revealed_addr = snes2pc(0xB5F000);
        let partially_revealed_addr = snes2pc(0xB5F800);
        let area_seen_addr = snes2pc(0xB5F600);
        let imr_settings = &self
            .settings
            .quality_of_life_settings
            .initial_map_reveal_settings;

        if imr_settings.all_areas {
            // allow pause map area switching to all areas from start of game:
            self.rom.write_u16(area_seen_addr, 0x003F)?;
        } else {
            self.rom.write_u16(area_seen_addr, 0x0000)?;
        }

        // Initialize all tiles to not-revealed by default
        self.rom.write_n(revealed_addr, &vec![0; 0x600])?;
        self.rom.write_n(partially_revealed_addr, &vec![0; 0x600])?;

        for (&(area, x, y), tile) in sorted_hashmap_iter(&self.map_tile_map) {
            let local_x = x - self.area_offset_x[area];
            let local_y = y - self.area_offset_y[area];
            let (offset, bitmask) = xy_to_explored_bit_ptr(local_x, local_y);
            let ptr_revealed = revealed_addr + area * 0x100 + offset as usize;
            let ptr_partial = partially_revealed_addr + area * 0x100 + offset as usize;
            match Self::get_initial_tile_reveal(tile, imr_settings) {
                MapRevealLevel::No => {}
                MapRevealLevel::Partial => {
                    self.rom.write_u8(
                        ptr_partial,
                        self.rom.read_u8(ptr_partial)? | bitmask as isize,
                    )?;
                }
                MapRevealLevel::Full => {
                    self.rom.write_u8(
                        ptr_partial,
                        self.rom.read_u8(ptr_partial)? | bitmask as isize,
                    )?;
                    self.rom.write_u8(
                        ptr_revealed,
                        self.rom.read_u8(ptr_revealed)? | bitmask as isize,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn setup_special_door_reveal(&mut self) -> Result<()> {
        // If save/refill rooms with 2 doors, a common problem is that players enter it and leave,
        // and then when looking at the map later, don't remember that there's another room behind it.
        // To avoid this, when entering on of these rooms, we do a "partial reveal" on just the door
        // of the neighboring rooms.
        // TODO: consider extending this behavior to objective tiles as well.
        let imr_settings = &self
            .settings
            .quality_of_life_settings
            .initial_map_reveal_settings;
        let save_partial = imr_settings.save_stations == MapRevealLevel::Partial;
        let refill_partial = imr_settings.refill_stations == MapRevealLevel::Partial;
        let mut room_ids = vec![
            (302, save_partial),   // Frog Savestation
            (190, save_partial),   // Draygon Save Room
            (308, refill_partial), // Nutella Refill
        ];

        // Don't show the special reveal if the room is unplaced or blocked by a wall.
        room_ids.retain(|&(room_id, _)| {
            let room_idx = self.game_data.room_idx_by_id[&room_id];
            if !self.randomization.map.room_mask[room_idx] {
                return false;
            }
            for door in &self.randomization.locked_doors {
                if door.door_type == DoorType::Wall {
                    let (src_room_idx, _) =
                        self.game_data.room_and_door_idxs_by_door_ptr_pair[&door.src_ptr_pair];
                    let src_room_id = self.game_data.room_geometry[src_room_idx].room_id;
                    if src_room_id == room_id {
                        return false;
                    }
                }
            }
            true
        });

        let mut table_addr = snes2pc(0x85A180);
        let partial_revealed_bits_base = 0x2700;
        let tilemap_base = 0x4000;
        let left_door_tile_idx = 0x12;

        for (room_id, partial) in room_ids {
            // If the save/refill tile is initially partially revealed, then we also use
            // the partial reveal palette for the neighboring markings. This is not completely
            // ideal as the markings will still show as partial revealed even after the
            // save/refill tile is explored, but it's the best we can do without a significant
            // overhaul of the ASM; and it's unclear if this option will find much use anyway.
            let palette = if partial { 0x0C00 } else { 0x0800 };
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
        table_addr += 2;
        assert!(table_addr <= snes2pc(0x85A1D6));
        Ok(())
    }

    fn set_map_activation_behavior(&mut self) -> Result<()> {
        match self.settings.other_settings.map_station_reveal {
            MapStationReveal::Partial => {
                self.rom.write_u16(snes2pc(0x90F700), 0xFFFF)?;
            }
            MapStationReveal::Full => {}
        }
        Ok(())
    }

    fn sort_dynamic_tile_data(&mut self) -> Result<()> {
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
        for data in self.dynamic_tile_data.iter_mut() {
            for (_, room_id, tile) in &*data {
                if !interior_priority.contains(&tile.interior) {
                    panic!("In room_id={room_id}, unexpected dynamic tile interior: {tile:?}");
                }
            }

            data.sort_by_key(|(_, _, tile)| {
                interior_priority.iter().position(|&y| y == tile.interior)
            });
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
        for (area_idx, data) in area_data.iter().enumerate() {
            self.rom.write_u16(
                snes2pc(base_ptr + area_idx * 2),
                (data_ptr & 0xFFFF) as isize,
            )?;
            self.rom
                .write_u16(snes2pc(base_ptr + 12 + area_idx * 2), data.len() as isize)?;
            let data_start = data_ptr;
            for &(item_idx, _room_id, ref tile) in data {
                self.rom
                    .write_u8(snes2pc(data_ptr), (item_idx as isize) >> 3)?; // item byte index
                self.rom
                    .write_u8(snes2pc(data_ptr + 1), 1 << ((item_idx as isize) & 7))?; // item bitmask
                let word = self.index_tile(tile.clone(), None)?;

                let local_x = tile.coords.0 as isize - self.area_offset_x[area_idx];
                let local_y = tile.coords.1 as isize - self.area_offset_y[area_idx];
                let offset = xy_to_map_offset(local_x, local_y);
                self.rom.write_u16(snes2pc(data_ptr + 2), offset)?; // tilemap offset
                self.rom.write_u16(snes2pc(data_ptr + 4), word as isize)?; // tilemap word
                data_ptr += 6;
            }
            assert_eq!(data_ptr, data_start + 6 * data.len());
        }
        assert!(data_ptr <= 0x83B600);
        Ok(())
    }

    fn indicate_items(&mut self) -> Result<()> {
        for (i, &item) in self.randomization.item_placement.iter().enumerate() {
            let (room_id, node_id) = self.game_data.item_locations[i];
            let room_ptr = self.game_data.room_ptr_by_id[&room_id];
            let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
            if !self.map.room_mask[room_idx] {
                continue;
            }
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
            let orig_tile = self.get_room_tile(room_id, x, y).unwrap().clone();
            let mut tile = orig_tile.clone();
            tile.faded = false;
            if [MapTileInterior::HiddenItem, MapTileInterior::DoubleItem].contains(&tile.interior) {
                tile.interior = MapTileInterior::Item;
            }
            if self.settings.other_settings.ultra_low_qol {
                tile.interior = MapTileInterior::Item;
                self.set_room_tile(room_id, x, y, tile.clone());
            } else {
                tile.interior = get_item_interior(item, self.settings);
                self.dynamic_tile_data[area].push((item_idx, room_id, tile.clone()));
                if self.customize_settings.item_dot_change == ItemDotChange::Fade {
                    tile.interior = apply_item_interior(orig_tile, item, self.settings);
                    tile.faded = true;
                    self.set_room_tile(room_id, x, y, tile.clone());
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
            let mut data = self.read_hud_tile_2bpp(idx)?;
            for y in 0..8 {
                for x in 0..8 {
                    if data[y][x] == 1 {
                        data[y][x] = 2;
                    }
                }
            }
            self.write_hud_tile_2bpp(idx, data)?;
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
        tiles_to_change.extend([0x33, 0x46, 0x47, 0x48]); // AUTO

        // Use color 0 instead of color 3 for black in HUD map tiles:
        // Also use color 3 instead of color 2 for white
        for idx in tiles_to_change {
            let mut tile = self.read_hud_tile_2bpp(idx)?;
            for y in 0..8 {
                for x in 0..8 {
                    if tile[y][x] == 3 {
                        tile[y][x] = 0;
                    } else if tile[y][x] == 2 {
                        tile[y][x] = 3;
                    }
                }
            }
            self.write_hud_tile_2bpp(idx, tile)?;
        }
        Ok(())
    }

    fn darken_hud_grid(&mut self) -> Result<()> {
        // In HUD tiles, replace the white dotted grid lines with dark gray ones.
        self.write_hud_tile_2bpp(
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
        self.write_hud_tile_2bpp(
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
        self.write_hud_tile_2bpp(
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
        self.write_hud_tile_2bpp(
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
        self.write_hud_tile_2bpp(
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

    fn initialize_tiles(&mut self) -> Result<()> {
        // copy original pause menu tile GFX into map tile GFX
        let src_addr = snes2pc(0xB68000);
        let dst_addr = snes2pc(TILE_GFX_ADDR_4BPP);
        for i in (0..0x6000).step_by(2) {
            let word = self.rom.read_u16(src_addr + i)?;
            self.rom.write_u16(dst_addr + i, word)?;
        }
        Ok(())
    }

    fn fix_kraid(&mut self) -> Result<()> {
        // Kraid alive:
        self.rom.write_u16(snes2pc(0x8FB81C), 0x0C00)?; // avoid overwriting hazard tiles with (unneeded) message box tiles

        // Kraid dead:
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
        // Plasma: black color 15 -> 8
        self.substitute_colors(14, vec![1, 5], vec![(15, 8)])?;
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

    fn get_room_coords(
        &self,
        room_id: usize,
        x: isize,
        y: isize,
    ) -> Option<(AreaIdx, isize, isize)> {
        let room_ptr = self.game_data.room_ptr_by_id[&room_id];
        let room_idx = self.game_data.room_idx_by_ptr[&room_ptr];
        if !self.map.room_mask[room_idx] {
            return None;
        }
        let area = self.map.area[room_idx];
        let mut room_coords = self.map.rooms[room_idx];
        if room_id == 313 {
            // Homing Geemer Room
            room_coords = (room_coords.0 + 5, room_coords.1 + 2);
        } else if room_id == 322 {
            // East Pants Room
            room_coords = (room_coords.0 + 1, room_coords.1 + 1);
        }
        let x = room_coords.0 as isize + x;
        let y = room_coords.1 as isize + y;
        Some((area, x, y))
    }

    fn get_room_tile(&mut self, room_id: usize, x: isize, y: isize) -> Option<&mut MapTile> {
        let (area, x1, y1) = self.get_room_coords(room_id, x, y)?;
        Some(self.map_tile_map.get_mut(&(area, x1, y1)).unwrap())
    }

    fn set_room_tile(&mut self, room_id: usize, x: isize, y: isize, mut tile: MapTile) {
        let Some((area, x1, y1)) = self.get_room_coords(room_id, x, y) else {
            return;
        };
        tile.coords.0 = x1 as usize;
        tile.coords.1 = y1 as usize;
        self.map_tile_map.insert((area, x1, y1), tile);
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
            if self.area_min_x[area] == isize::MAX {
                continue;
            }
            let margin_x = (64 - (self.area_max_x[area] - self.area_min_x[area])) / 2;
            let margin_y = (32 - (self.area_max_y[area] - self.area_min_y[area])) / 2;
            self.area_offset_x[area] = self.area_min_x[area] - margin_x;
            self.area_offset_y[area] = self.area_min_y[area] - margin_y;
        }

        Ok(())
    }

    fn update_acid_palette(&mut self) -> Result<()> {
        // use palette 6 for acid FX instead of palette 0, so that it can properly fade
        // through transitions (since in the randomizer palette 0 does not fade, as it's
        // used for the reserve HUD).
        for addr in (snes2pc(0x8A8840)..snes2pc(0x8A9080)).step_by(2) {
            let word = self.rom.read_u16(addr)?;
            if word & 0x1C00 == 0x0000 {
                self.rom.write_u16(addr, word | 0x1800)?;
            }
        }
        Ok(())
    }

    pub fn apply_patches(&mut self) -> Result<()> {
        self.update_acid_palette()?;
        self.fix_equipment_graphics()?;
        self.initialize_tiles()?;
        self.index_fixed_tiles()?;
        self.fix_pause_palettes()?;
        self.fix_message_boxes()?;
        self.fix_hud_black()?;
        self.darken_hud_grid()?;
        if self.settings.quality_of_life_settings.disableable_etanks != DisableETankSetting::Off {
            self.write_disabled_etank_tile()?;
        }
        self.apply_room_tiles()?;
        self.indicate_objective_tiles()?;
        if !self.settings.other_settings.ultra_low_qol {
            self.indicate_gray_doors()?;
            self.indicate_locked_doors()?;
        }
        self.add_cross_area_arrows()?;
        self.set_map_activation_behavior()?;
        self.indicate_items()?;
        self.compute_area_bounds()?;
        self.write_map_tiles()?;
        self.set_initial_map()?;
        if self.settings.quality_of_life_settings.room_outline_revealed {
            self.setup_special_door_reveal()?;
        }
        self.sort_dynamic_tile_data()?;
        self.write_dynamic_tile_data(&self.dynamic_tile_data.clone())?;
        self.create_room_map_tilemaps()?;
        self.write_hazard_tiles()?;
        self.fix_kraid()?;
        self.fix_item_colors()?;

        Ok(())
    }
}
