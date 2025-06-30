use anyhow::Result;
use hashbrown::HashMap;
use image::{Rgba, RgbaImage};
use std::io::Cursor;

pub use image;

use crate::{
    patch::map_tiles::{
        apply_door_lock, apply_item_interior, get_gray_doors, get_objective_tiles, render_tile,
    },
    randomize::Randomization,
    settings::RandomizerSettings,
};
use maprando_game::{
    Direction, DoorLockType, GameData, ItemPtr, MapTile, MapTileEdge, MapTileInterior,
    MapTileSpecialType,
};

fn get_rgb(r: isize, g: isize, b: isize) -> Rgba<u8> {
    Rgba([
        (r * 255 / 31) as u8,
        (g * 255 / 31) as u8,
        (b * 255 / 31) as u8,
        255,
    ])
}

fn get_explored_color(value: u8, area: usize) -> Rgba<u8> {
    let cool_area_color = match area {
        0 => get_rgb(18, 0, 27),  // Crateria
        1 => get_rgb(0, 18, 0),   // Brinstar
        2 => get_rgb(23, 0, 0),   // Norfair
        3 => get_rgb(16, 17, 0),  // Wrecked Ship
        4 => get_rgb(3, 12, 29),  // Maridia
        5 => get_rgb(21, 12, 0),  // Tourian
        6 => get_rgb(10, 10, 10), // unexplored
        _ => panic!("Unexpected area {area}"),
    };
    let hot_area_color = match area {
        0 => get_rgb(27, 15, 31), // Crateria
        1 => get_rgb(12, 25, 12), // Brinstar
        2 => get_rgb(31, 12, 12), // Norfair
        3 => get_rgb(23, 23, 11), // Wrecked Ship
        4 => get_rgb(12, 20, 31), // Maridia
        5 => get_rgb(29, 17, 12), // Tourian
        6 => get_rgb(17, 17, 17), // unexplored
        _ => panic!("Unexpected area {area}"),
    };
    match value {
        0 => get_rgb(0, 0, 0),
        1 => cool_area_color,
        2 => hot_area_color,
        3 => get_rgb(31, 31, 31),  // Wall/passage (white)
        4 => get_rgb(0, 0, 0), // Opaque black (used in elevators, covers up dotted grid background)
        6 => get_rgb(29, 15, 0), // Yellow (orange) door (Power Bomb, Spazer)
        7 => get_rgb(27, 2, 27), // Red (pink) door (Missile, Wave)
        8 => get_rgb(4, 13, 31), // Blue door (Ice)
        12 => get_rgb(0, 0, 0), // Door lock shadow covering wall (black)
        13 => get_rgb(31, 31, 31), // Item dots (white)
        14 => get_rgb(7, 31, 7), // Green door (Super, Plasma)
        15 => get_rgb(18, 12, 14), // Gray door (including Charge)
        _ => panic!("Unexpected color value {value}"),
    }
}

fn get_outline_color(value: u8) -> Rgba<u8> {
    match value {
        0 => get_rgb(0, 0, 0),
        1 => get_rgb(0, 0, 0),
        2 => get_rgb(0, 0, 0),
        3 => get_rgb(16, 16, 16), // Wall/passage (grey)
        4 => get_rgb(0, 0, 0),
        6 => get_rgb(0, 0, 0),
        7 => get_rgb(0, 0, 0),
        8 => get_rgb(0, 0, 0),
        12 => get_rgb(16, 16, 16), // Door lock shadow, should appear same as wall/passage (grey)
        13 => get_rgb(0, 0, 0),
        14 => get_rgb(0, 0, 0),
        15 => get_rgb(0, 0, 0),
        _ => panic!("Unexpected color value {value}"),
    }
}

pub struct SpoilerMaps {
    pub explored: Vec<u8>,
    pub outline: Vec<u8>,
}

fn add_vanilla_elevators(tiles: &mut [Vec<MapTile>]) {
    let coords = [(7, 16, 23), (24, 25, 31), (35, 15, 27), (60, 18, 23)];
    for (x, y0, y1) in coords {
        for y in y0..y1 {
            tiles[y][x].special_type = Some(MapTileSpecialType::Elevator);
            tiles[y][x].area = Some(6); // Special area to draw as gray color
        }
    }
}

pub fn get_spoiler_images(
    randomization: &Randomization,
    game_data: &GameData,
    settings: &RandomizerSettings,
    show_grid: bool,
) -> Result<(RgbaImage, RgbaImage)> {
    let map = &randomization.map;
    let max_tiles = 72;
    let width = max_tiles;
    let height = max_tiles;
    let mut tiles: Vec<Vec<MapTile>> = vec![vec![MapTile::default(); width]; height];

    if settings.map_layout == "Vanilla" {
        add_vanilla_elevators(&mut tiles);
    }

    // Create the base form of the room tiles
    for room in &game_data.map_tile_data {
        let room_id = room.room_id;
        let room_ptr = game_data.room_ptr_by_id[&room_id];
        let room_idx = game_data.room_idx_by_ptr[&room_ptr];
        let room_x = map.rooms[room_idx].0;
        let room_y = map.rooms[room_idx].1;
        let area = map.area[room_idx];
        for tile in &room.map_tiles {
            let x = room_x + tile.coords.0;
            let y = room_y + tile.coords.1;
            if tile.special_type == Some(MapTileSpecialType::Black) {
                continue;
            }
            if tiles[y][x].area.is_none()
                || tiles[y][x].special_type == Some(MapTileSpecialType::Tube)
                || tiles[y][x].special_type == Some(MapTileSpecialType::Elevator)
            {
                // Allow other tiles to take priority (draw on top of) tube and elevator tiles,
                // because of the Toilet, and also Tourian elevator on vanilla map.
                tiles[y][x] = tile.clone();
                tiles[y][x].area = Some(area);
            }
        }
    }

    // Add item dots:
    let mut item_coords: HashMap<ItemPtr, (usize, usize)> = HashMap::new();
    for room in &game_data.room_geometry {
        for item in &room.items {
            item_coords.insert(item.addr, (item.x, item.y));
        }
    }
    for (i, &item) in randomization.item_placement.iter().enumerate() {
        let (room_id, node_id) = game_data.item_locations[i];
        let item_ptr = game_data.node_ptr_map[&(room_id, node_id)];
        let (item_x, item_y) = item_coords[&item_ptr];
        let room_ptr = game_data.room_ptr_by_id[&room_id];
        let room_idx = game_data.room_idx_by_ptr[&room_ptr];
        let room_x = map.rooms[room_idx].0;
        let room_y = map.rooms[room_idx].1;
        let x = room_x + item_x;
        let y = room_y + item_y;
        tiles[y][x].interior = apply_item_interior(tiles[y][x].clone(), item, settings);
    }

    // Add door locks:
    for locked_door in randomization.locked_doors.iter() {
        let mut ptr_pairs = vec![locked_door.src_ptr_pair];
        if locked_door.bidirectional {
            ptr_pairs.push(locked_door.dst_ptr_pair);
        }
        for ptr_pair in ptr_pairs {
            let (room_idx, door_idx) = game_data.room_and_door_idxs_by_door_ptr_pair[&ptr_pair];
            let room_geom = &game_data.room_geometry[room_idx];
            let door = &room_geom.doors[door_idx];
            let room_x = map.rooms[room_idx].0;
            let room_y = map.rooms[room_idx].1;
            let x = room_x + door.x;
            let y = room_y + door.y;
            tiles[y][x] = apply_door_lock(&tiles[y][x], locked_door, door);
        }
    }

    // Add gray doors:
    let gray_door = MapTileEdge::LockedDoor(DoorLockType::Gray);
    for (room_id, door_x, door_y, dir) in get_gray_doors() {
        let room_ptr = game_data.room_ptr_by_id[&room_id];
        let room_idx = game_data.room_idx_by_ptr[&room_ptr];
        let room_x = map.rooms[room_idx].0;
        let room_y = map.rooms[room_idx].1;
        let x = room_x + door_x as usize;
        let y = room_y + door_y as usize;
        let mut tile = tiles[y][x].clone();
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
        tiles[y][x] = tile;
    }

    // Add objectives:
    for (room_id, tile_x, tile_y) in get_objective_tiles(&randomization.objectives) {
        let room_ptr = game_data.room_ptr_by_id[&room_id];
        let room_idx = game_data.room_idx_by_ptr[&room_ptr];
        let room_x = map.rooms[room_idx].0;
        let room_y = map.rooms[room_idx].1;
        let x = room_x + tile_x;
        let y = room_y + tile_y;
        tiles[y][x].interior = MapTileInterior::Objective;
    }

    // Render the map tiles into image (one in explored form, and one in partially revealed/outline form):
    let mut img_explored = RgbaImage::new((width + 2) as u32 * 8, (height + 2) as u32 * 8);
    let mut img_outline = RgbaImage::new((width + 2) as u32 * 8, (height + 2) as u32 * 8);

    if show_grid {
        let grid_color = get_rgb(6, 6, 6);
        for y in 0..height + 2 {
            for x in 0..width + 2 {
                for py in (1..8).step_by(2) {
                    img_explored.put_pixel(x as u32 * 8, y as u32 * 8 + py, grid_color);
                }
                for px in (0..8).step_by(2) {
                    img_explored.put_pixel(x as u32 * 8 + px, y as u32 * 8 + 7, grid_color);
                }
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            let tile = &tiles[y][x];
            if tile.area.is_none() {
                continue;
            }
            let data = render_tile(tile.clone(), settings)?;
            for py in 0..8 {
                for px in 0..8 {
                    if data[py][px] == 0 {
                        continue;
                    }
                    let x1 = (x + 1) * 8 + px;
                    let y1 = (y + 1) * 8 + py;
                    img_explored.put_pixel(
                        x1 as u32,
                        y1 as u32,
                        get_explored_color(data[py][px], tile.area.unwrap()),
                    );
                    img_outline.put_pixel(x1 as u32, y1 as u32, get_outline_color(data[py][px]));
                }
            }
        }
    }

    Ok((img_explored, img_outline))
}

pub fn get_spoiler_map(
    randomization: &Randomization,
    game_data: &GameData,
    settings: &RandomizerSettings,
    show_grid: bool,
) -> Result<SpoilerMaps> {
    let (img_explored, img_outline) =
        get_spoiler_images(randomization, game_data, settings, show_grid)?;

    let mut vec_explored: Vec<u8> = Vec::new();
    img_explored.write_to(
        &mut Cursor::new(&mut vec_explored),
        image::ImageOutputFormat::Png,
    )?;

    let mut vec_outline: Vec<u8> = Vec::new();
    img_outline.write_to(
        &mut Cursor::new(&mut vec_outline),
        image::ImageOutputFormat::Png,
    )?;

    Ok(SpoilerMaps {
        explored: vec_explored,
        outline: vec_outline,
    })
}
