use anyhow::{Context, Result, ensure};
use hashbrown::{HashMap, HashSet};
use maprando_game::{GameData, Map};
use serde::Deserialize;

use crate::patch::{Rom, get_room_state_ptrs, pc2snes, snes2pc};

use super::Allocator;

// Must agree with sprite_tube.asm. Each 20-byte record contains:
// room header (word), right-column X (word), OAM attributes (word),
// graphics pointer (24 bits + padding), five target-palette offsets (words).
// A room header of $FFFF terminates the table. Each 256-byte DMA payload
// contains four top tiles followed by four bottom tiles, all in bank EA.
const TABLE_START: usize = 0xEAB000;
const GRAPHICS_START: usize = 0xEAB800;
const GRAPHICS_END: usize = 0xEB0000;
const RECORD_SIZE: usize = 20;
const GRAPHICS_SIZE: usize = 256;

#[derive(Clone, Deserialize)]
struct PaletteMapping {
    name: String,
    room_header: String,
    palette: u16,
    colors: [u8; 5],
    #[serde(default, rename = "xOffset")]
    x_offset: isize,
    #[serde(default)]
    related_rooms: Vec<PaletteMapping>,
    #[serde(default)]
    enemy_palette_overrides: Vec<EnemyPaletteOverride>,
}

#[derive(Clone, Deserialize)]
struct EnemyPaletteOverride {
    enemy_id: String,
    from_palette: u16,
    to_palette: u16,
}

#[derive(Deserialize)]
struct PaletteData {
    tube_source_colors: [u8; 5],
    rooms: Vec<PaletteMapping>,
}

#[derive(Deserialize)]
struct TestPlacement {
    name: String,
    x: Vec<isize>,
}

fn parse_address(value: &str) -> Result<usize> {
    parse_int::parse(value).with_context(|| format!("Invalid tube address {value}"))
}

fn read_cre_data(rom: &Rom, bank_operand: usize, pointer_operand: usize) -> Result<Vec<u8>> {
    // These operands are updated by build-mosaic.rs when installing tilesets.bps.
    let address = (rom.read_u8(snes2pc(bank_operand))? as usize) << 16
        | rom.read_u16(snes2pc(pointer_operand))? as usize;
    let compressed = rom
        .data
        .get(snes2pc(address)..)
        .context("CRE data pointer outside ROM")?;
    lznint::decompress(compressed).context("Decompressing CRE data for the sprite tube")
}

fn extract_graphics(rom: &Rom) -> Result<Vec<u8>> {
    let tiles = read_cre_data(rom, 0x82E415, 0x82E419)?;
    let blocks = read_cre_data(rom, 0x82E83D, 0x82E841)?;
    // Standard BG2 tube blocks used by build-mosaic.rs: body $F0, joint $EE.
    // Block entries are TL, TR, BL, BR; OBJ bottom tiles sit 16 indices after
    // the top tiles. Pack the rows separately for two 128-byte DMA uploads.
    let parts = [
        (0xF0, 0),
        (0xF0, 1),
        (0xEE, 0),
        (0xEE, 1),
        (0xF0, 2),
        (0xF0, 3),
        (0xEE, 2),
        (0xEE, 3),
    ];
    let mut output = Vec::with_capacity(GRAPHICS_SIZE);
    for (block, quadrant) in parts {
        let offset = block * 8 + quadrant * 2;
        let entry = blocks
            .get(offset..offset + 2)
            .context("Missing CRE tube block")?;
        let entry = u16::from_le_bytes([entry[0], entry[1]]);
        // CRE 8x8 graphics begin at BG tile $280.
        let tile = (entry as usize & 0x3FF)
            .checked_sub(0x280)
            .context("Tube block references graphics outside CRE")?;
        let source = tiles
            .get(tile * 32..tile * 32 + 32)
            .context("Missing CRE tube tile")?;
        let mut graphics = [0; 32];
        for plane in [0, 1, 16, 17] {
            for y in 0..8 {
                let source_y = if entry & 0x8000 != 0 { 7 - y } else { y };
                let bits = source[plane + source_y * 2];
                graphics[plane + y * 2] = if entry & 0x4000 != 0 {
                    bits.reverse_bits()
                } else {
                    bits
                };
            }
        }
        output.extend(graphics);
    }
    Ok(output)
}

fn remap_graphics(graphics: &[u8], source_colors: &[u8; 5], colors: &[u8; 5]) -> Result<Vec<u8>> {
    let mut mapping = [0u8; 16];
    let mut seen = HashSet::new();
    for (&src, &dst) in source_colors.iter().zip(colors) {
        ensure!((1..16).contains(&src) && (1..16).contains(&dst));
        ensure!(seen.insert(dst), "Repeated tube destination color {dst}");
        mapping[src as usize] = dst;
    }
    let mut output = vec![0; graphics.len()];
    for (src, dst) in graphics.chunks_exact(32).zip(output.chunks_exact_mut(32)) {
        for y in 0..8 {
            for x in 0..8 {
                let mut color = 0;
                for (plane, offset) in [0, 1, 16, 17].into_iter().enumerate() {
                    color |= ((src[offset + y * 2] >> x) & 1) << plane;
                }
                ensure!(
                    color == 0 || mapping[color as usize] != 0,
                    "Unexpected tube source color {color}"
                );
                let new_color = mapping[color as usize];
                for (plane, offset) in [0, 1, 16, 17].into_iter().enumerate() {
                    dst[offset + y * 2] |= ((new_color >> plane) & 1) << x;
                }
            }
        }
    }
    Ok(output)
}

fn apply_enemy_overrides(rom: &mut Rom, mapping: &PaletteMapping) -> Result<()> {
    if mapping.enemy_palette_overrides.is_empty() {
        return Ok(());
    }
    let room_ptr = snes2pc(parse_address(&mapping.room_header)?);
    let mut sets = HashSet::new();
    for (_, state_ptr) in get_room_state_ptrs(rom, room_ptr)? {
        sets.insert(snes2pc(0xB40000 | rom.read_u16(state_ptr + 10)? as usize));
    }
    for set in sets {
        for replacement in &mapping.enemy_palette_overrides {
            let enemy_id = parse_address(&replacement.enemy_id)?;
            let mut ptr = set;
            let mut found = false;
            while rom.read_u16(ptr)? != 0xFFFF {
                if rom.read_u16(ptr)? as usize == enemy_id {
                    let palette = rom.read_u16(ptr + 2)? as u16;
                    ensure!(
                        palette == replacement.from_palette || palette == replacement.to_palette,
                        "Unexpected palette for {} in {}",
                        replacement.enemy_id,
                        mapping.name
                    );
                    rom.write_u16(ptr + 2, replacement.to_palette as isize)?;
                    found = true;
                }
                ptr += 4;
                ensure!(
                    ptr < set + 0x100,
                    "Unterminated enemy set in {}",
                    mapping.name
                );
            }
            ensure!(
                found,
                "Missing enemy {} in {}",
                replacement.enemy_id,
                mapping.name
            );
        }
    }
    Ok(())
}

fn add_room(
    rom: &Rom,
    rooms: &mut Vec<(PaletteMapping, isize)>,
    mapping: &PaletteMapping,
    x: isize,
) -> Result<()> {
    let ptr = snes2pc(parse_address(&mapping.room_header)?);
    ensure!(
        x >= 0 && x < rom.read_u8(ptr + 4)? as isize,
        "Tube X {x} outside {}",
        mapping.name
    );
    rooms.push((mapping.clone(), x));
    for related in &mapping.related_rooms {
        let local_x = x - related.x_offset;
        let related_ptr = snes2pc(parse_address(&related.room_header)?);
        if local_x >= 0 && local_x < rom.read_u8(related_ptr + 4)? as isize {
            rooms.push((related.clone(), local_x));
        }
    }
    Ok(())
}

pub fn apply_sprite_tubes(
    rom: &mut Rom,
    map: &Map,
    game_data: &GameData,
    test_all_rooms: bool,
) -> Result<()> {
    let data: PaletteData = serde_json::from_str(include_str!("../../../data/tube_palettes.json"))?;
    ensure!(
        data.tube_source_colors == [4, 5, 6, 7, 15],
        "Unsupported tube source palette"
    );
    let mut rooms = Vec::new();
    if test_all_rooms {
        let placements: Vec<TestPlacement> =
            serde_json::from_str(include_str!("../../../../transit-tube-data/Base.json"))?;
        let placements: HashMap<_, _> = placements.into_iter().map(|p| (p.name, p.x)).collect();
        for mapping in &data.rooms {
            let x = match mapping.name.as_str() {
                "Pants Room" => 1,
                "West Ocean" => 5,
                _ => *placements
                    .get(&mapping.name)
                    .and_then(|xs| xs.first())
                    .with_context(|| format!("Missing tube test X for {}", mapping.name))?,
            };
            add_room(rom, &mut rooms, mapping, x)?;
        }
    } else if map.room_mask[game_data.toilet_room_idx] {
        let room_header = rom.read_u16(snes2pc(0xB5FE70))? as usize;
        if room_header == 0xFFFF {
            // Vanilla-map special case: these are independent randomizer rooms.
            let aqueduct = data
                .rooms
                .iter()
                .find(|p| p.name == "Aqueduct")
                .context("Missing Aqueduct tube palette")?;
            add_room(rom, &mut rooms, aqueduct, 2)?;
            let mut hallway = aqueduct.clone();
            hallway.name = "Botwoon Hallway".to_string();
            hallway.room_header = "0x8FD617".to_string();
            hallway.related_rooms.clear();
            hallway.enemy_palette_overrides.clear();
            add_room(rom, &mut rooms, &hallway, 2)?;
        } else {
            let mapping = data
                .rooms
                .iter()
                .find(|p| parse_address(&p.room_header).ok() == Some(0x8F0000 | room_header))
                .context("Missing intersecting room tube palette")?;
            let x = rom.read_u8(snes2pc(0xB5FE72))? as i8 as isize;
            add_room(rom, &mut rooms, mapping, x)?;
        }
    }

    ensure!(
        rooms.len() * RECORD_SIZE + 2 <= snes2pc(GRAPHICS_START) - snes2pc(TABLE_START),
        "Sprite tube table is full"
    );
    let mut graphics_allocator =
        Allocator::new(vec![(snes2pc(GRAPHICS_START), snes2pc(GRAPHICS_END))]);
    let mut graphics_pointers = HashMap::new();
    let mut headers = HashSet::new();
    let mut ptr = snes2pc(TABLE_START);
    if rooms.is_empty() {
        rom.write_u16(ptr, 0xFFFF)?;
        return Ok(());
    }
    let source_graphics = extract_graphics(rom)?;
    for (mapping, x) in rooms {
        let header = parse_address(&mapping.room_header)?;
        ensure!(
            header >> 16 == 0x8F && headers.insert(header),
            "Invalid or duplicate tube room"
        );
        ensure!(
            [1, 2, 3, 7].contains(&mapping.palette),
            "Invalid tube palette"
        );
        let graphics = remap_graphics(&source_graphics, &data.tube_source_colors, &mapping.colors)?;
        let graphics_ptr = match graphics_pointers.get(&graphics) {
            Some(&address) => address,
            None => {
                let address = graphics_allocator.allocate(graphics.len())?;
                rom.write_n(address, &graphics)?;
                graphics_pointers.insert(graphics, address);
                address
            }
        };
        apply_enemy_overrides(rom, &mapping)?;
        rom.write_u16(ptr, (header & 0xFFFF) as isize)?;
        rom.write_u16(ptr + 2, x * 256 + 0x80)?;
        rom.write_u16(ptr + 4, (0x20D0 | mapping.palette << 9) as isize)?;
        rom.write_u24(ptr + 6, pc2snes(graphics_ptr) as isize)?;
        rom.write_u8(ptr + 9, 0)?;
        for (i, color) in mapping.colors.into_iter().enumerate() {
            let offset = 0x100 + mapping.palette * 32 + color as u16 * 2;
            rom.write_u16(ptr + 10 + i * 2, offset as isize)?;
        }
        ptr += RECORD_SIZE;
    }
    rom.write_u16(ptr, 0xFFFF)?;
    Ok(())
}
