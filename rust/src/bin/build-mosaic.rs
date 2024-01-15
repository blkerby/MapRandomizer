use anyhow::{Context, Result};
use clap::Parser;
use crypto_hash;
use flips;
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;
use json::JsonValue;
use log::info;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use smart_xml::{Screen, Layer2Type};
use maprando::customize::Allocator;
use maprando::game_data::smart_xml;
use maprando::patch::{pc2snes, snes2pc, Rom, get_room_state_ptrs};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    compressor: PathBuf,
    #[arg(long)]
    input_rom: PathBuf,
}

struct MosaicBuilder {
    rom: Rom,
    compressed_data_cache_dir: PathBuf,
    compressor_path: PathBuf,
    tmp_dir: PathBuf,
    mosaic_dir: PathBuf,
    output_patches_dir: PathBuf,
    main_allocator: Allocator,
    fx_allocator: Allocator,
}


fn extract_screen_words(screen: &Screen, out: &mut [u8], width: usize, _height: usize) {
    let base_pos = (screen.y * width * 256 + screen.x * 16) * 2;
    assert!(screen.data.len() == 256);
    for y in 0..16 {
        for x in 0..16 {
            let c = screen.data[y * 16 + x];
            let pos = base_pos + (y * width * 16 + x) * 2;
            out[pos] = (c & 0xFF) as u8;
            out[pos + 1] = (c >> 8) as u8;
        }
    }
}

fn extract_screen_bytes(screen: &Screen, out: &mut [u8], width: usize, _height: usize) {
    let base_pos = screen.y * width * 256 + screen.x * 16;
    assert!(screen.data.len() == 256);
    for y in 0..16 {
        for x in 0..16 {
            let c = screen.data[y * 16 + x];
            let pos = base_pos + y * width * 16 + x;
            out[pos] = c as u8;
        }
    }
}

fn extract_all_screen_words(screens: &[Screen], out: &mut [u8], width: usize, height: usize) {
    for screen in screens {
        extract_screen_words(screen, out, width, height);
    }
}

fn extract_all_screen_bytes(screens: &[Screen], out: &mut [u8], width: usize, height: usize) {
    for screen in screens {
        extract_screen_bytes(screen, out, width, height);
    }
}

pub fn extract_uncompressed_level_data(state_xml: &smart_xml::RoomState) -> Vec<u8> {
    let height = state_xml.level_data.height;
    let width = state_xml.level_data.width;
    let num_tiles = height * width * 256;
    let level_data_size = if state_xml.layer2_type == Layer2Type::Layer2 {
        2 + num_tiles * 5
    } else {
        2 + num_tiles * 3
    };
    let mut level_data = vec![0u8; level_data_size];
    level_data[0] = ((num_tiles * 2) & 0xFF) as u8;
    level_data[1] = ((num_tiles * 2) >> 8) as u8;
    extract_all_screen_words(
        &state_xml.level_data.layer_1.screen,
        &mut level_data[2..],
        width,
        height,
    );
    extract_all_screen_bytes(
        &state_xml.level_data.bts.screen,
        &mut level_data[2 + num_tiles * 2..],
        width,
        height,
    );
    if state_xml.layer2_type == Layer2Type::Layer2 {
        extract_all_screen_words(
            &state_xml.level_data.layer_2.screen,
            &mut level_data[2 + num_tiles * 3..],
            width,
            height,
        );
    }
    level_data
}

// fn make_fx1(fx1: &smart_xml::FX1) -> FX1 {
//     FX1 {
//         fx1_reference: None,
//         fx1_door: if fx1.default {
//             None
//         } else {
//             Some(FX1Door {
//                 room_area: fx1.roomarea,
//                 room_index: fx1.roomindex,
//                 door_index: fx1.fromdoor,
//             })
//         },
//         fx1_data: fx1.clone(),
//     }
// }

impl MosaicBuilder {
    fn get_compressed_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let digest = crypto_hash::hex_digest(crypto_hash::Algorithm::SHA256, &data);
        let output_path = self.compressed_data_cache_dir.join(digest);
        if !output_path.exists() {
            let tmp_path = self.tmp_dir.join("tmpfile");
            std::fs::write(&tmp_path, data)?;
            Command::new(&self.compressor_path)
                .arg("-c")
                .arg(format!("-f={}", tmp_path.to_str().unwrap()))
                .arg(format!("-o={}", output_path.to_str().unwrap()))
                .status()
                .context("error running compressor")?;
        }
        return Ok(std::fs::read(output_path)?);
    }

    fn make_tileset_patch(&mut self) -> Result<()> {
        info!("Processing tilesets");
        let mut new_rom = self.rom.clone();
        self.apply_cre_tileset(&mut new_rom)?;
        self.apply_sce_tilesets(&mut new_rom)?;
        let patch = flips::BpsDeltaBuilder::new()
            .source(&self.rom.data)
            .target(&new_rom.data)
            .build()?;
        let output_path = self.output_patches_dir.join("tilesets.bps");
        std::fs::write(&output_path, &patch)?;
        Ok(())
    }

    fn apply_cre_tileset(&mut self, new_rom: &mut Rom) -> Result<()> {
        let tileset_path = self.mosaic_dir.join("Projects/Base/Export/Tileset/CRE/00/");

        // Write CRE 8x8 tile graphics and update pointers to it:
        let gfx8x8_path = tileset_path.join("8x8tiles.gfx");
        let gfx8x8_bytes = std::fs::read(&gfx8x8_path)
            .with_context(|| format!("Unable to load CRE 8x8 gfx at {}", gfx8x8_path.display()))?;
        let compressed_gfx8x8 = self.get_compressed_data(&gfx8x8_bytes)?;
        let gfx8x8_addr = self.main_allocator.allocate(compressed_gfx8x8.len())?;
        new_rom.write_n(gfx8x8_addr, &compressed_gfx8x8)?;
        new_rom.write_u8(snes2pc(0x82E415), (pc2snes(gfx8x8_addr) >> 16) as isize)?;
        new_rom.write_u16(snes2pc(0x82E419), (pc2snes(gfx8x8_addr) & 0xFFFF) as isize)?;
        new_rom.write_u8(snes2pc(0x82E797), (pc2snes(gfx8x8_addr) >> 16) as isize)?;
        new_rom.write_u16(snes2pc(0x82E79B), (pc2snes(gfx8x8_addr) & 0xFFFF) as isize)?;

        // Write CRE 16x16 tile graphics and update pointers to it:
        let gfx16x16_path = tileset_path.join("16x16tiles.ttb");
        let gfx16x16_bytes = std::fs::read(&gfx16x16_path).with_context(|| {
            format!(
                "Unable to load CRE 16x16 gfx at {}",
                gfx16x16_path.display()
            )
        })?;
        let gfx16x16_addr = self.main_allocator.allocate(gfx16x16_bytes.len())?;
        new_rom.write_n(gfx16x16_addr, &gfx16x16_bytes)?;
        new_rom.write_u8(snes2pc(0x82E83D), (pc2snes(gfx16x16_addr) >> 16) as isize)?;
        new_rom.write_u16(
            snes2pc(0x82E841),
            (pc2snes(gfx16x16_addr) & 0xFFFF) as isize,
        )?;
        new_rom.write_u8(snes2pc(0x82EAED), (pc2snes(gfx16x16_addr) >> 16) as isize)?;
        new_rom.write_u16(
            snes2pc(0x82EAF1),
            (pc2snes(gfx16x16_addr) & 0xFFFF) as isize,
        )?;
        Ok(())
    }

    fn apply_sce_tilesets(&mut self, new_rom: &mut Rom) -> Result<()> {
        let tilesets_path = self.mosaic_dir.join("Projects/Base/Export/Tileset/SCE");
        let mut pal_map: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut gfx8_map: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut gfx16_map: HashMap<Vec<u8>, usize> = HashMap::new();
        let new_tile_table_snes = 0x8FF900;
        let new_tile_pointers_snes = 0x8FFD00;
        let tile_pointers_free_space_end = 0x8FFE00;

        let tileset_it = std::fs::read_dir(&tilesets_path)
            .with_context(|| format!("Unable to list tilesets at {}", tilesets_path.display()))?;
        for tileset_dir in tileset_it {
            let tileset_dir = tileset_dir?;
            let tileset_idx = usize::from_str_radix(tileset_dir.file_name().to_str().unwrap(), 16)?;
            let tileset_path = tileset_dir.path();

            let palette_path = tileset_path.join("palette.snes");
            let palette_bytes = std::fs::read(&palette_path)
                .with_context(|| format!("Unable to read palette at {}", palette_path.display()))?;
            let compressed_pal = self.get_compressed_data(&palette_bytes)?;

            let gfx8x8_path = tileset_path.join("8x8tiles.gfx");
            let gfx8x8_bytes = std::fs::read(&gfx8x8_path)
                .with_context(|| format!("Unable to read 8x8 gfx at {}", gfx8x8_path.display()))?;
            let compressed_gfx8x8 = self.get_compressed_data(&gfx8x8_bytes)?;

            let gfx16x16_path = tileset_path.join("16x16tiles.ttb");
            let gfx16x16_bytes = std::fs::read(&gfx16x16_path).with_context(|| {
                format!("Unable to read 16x16 gfx at {}", gfx16x16_path.display())
            })?;
            let compressed_gfx16x16 = self.get_compressed_data(&gfx16x16_bytes)?;

            if tileset_idx >= 0x0F && tileset_idx <= 0x14 {
                // Skip Ceres tilesets
                continue;
            }

            // Write SCE palette:
            let pal_addr = match pal_map.entry(compressed_pal.clone()) {
                Entry::Occupied(x) => *x.get(),
                Entry::Vacant(view) => {
                    let addr = self.main_allocator.allocate(compressed_pal.len())?;
                    view.insert(addr);
                    addr
                }
            };
            new_rom.write_n(pal_addr, &compressed_pal)?;

            // Write SCE 8x8 graphics:
            let gfx8_addr = match gfx8_map.entry(compressed_gfx8x8.clone()) {
                Entry::Occupied(x) => *x.get(),
                Entry::Vacant(view) => {
                    let addr = self.main_allocator.allocate(compressed_gfx8x8.len())?;
                    view.insert(addr);
                    addr
                }
            };
            new_rom.write_n(gfx8_addr, &compressed_gfx8x8)?;

            // Write SCE 16x16 graphics:
            let gfx16_addr = match gfx16_map.entry(compressed_gfx16x16.clone()) {
                Entry::Occupied(x) => *x.get(),
                Entry::Vacant(view) => {
                    let addr = self.main_allocator.allocate(compressed_gfx16x16.len())?;
                    view.insert(addr);
                    addr
                }
            };
            new_rom.write_n(gfx16_addr, &compressed_gfx16x16)?;

            // Update tileset pointers:
            let tile_table_entry_addr = new_tile_table_snes + 9 * tileset_idx;
            new_rom.write_u24(snes2pc(tile_table_entry_addr), pc2snes(gfx16_addr) as isize)?;
            new_rom.write_u24(
                snes2pc(tile_table_entry_addr + 3),
                pc2snes(gfx8_addr) as isize,
            )?;
            new_rom.write_u24(
                snes2pc(tile_table_entry_addr + 6),
                pc2snes(pal_addr) as isize,
            )?;
            new_rom.write_u16(
                snes2pc(new_tile_pointers_snes + 2 * tileset_idx),
                (tile_table_entry_addr & 0xFFFF) as isize,
            )?;
            assert!(new_tile_pointers_snes + 2 * tileset_idx < tile_pointers_free_space_end);
        }
        Ok(())
    }

    fn make_all_room_patches(&mut self) -> Result<()> {
        let projects_dir = self.mosaic_dir.join("Projects");
        let mut project_names = vec![];
        for project_dir in std::fs::read_dir(projects_dir)? {
            let project_dir = project_dir?;
            let project_name = project_dir.file_name().to_str().unwrap().to_owned();
            if project_name == "Base" {
                continue;
            }
            project_names.push(project_name);
        }

        let base_rooms_dir = self.mosaic_dir.join("Projects/Base/Export/Rooms/");
        for room_path in std::fs::read_dir(base_rooms_dir)? {
            let room_filename = room_path?.file_name().to_str().unwrap().to_owned();
            info!("Processing {}", room_filename);
            self.make_room_patch(&room_filename, &project_names)?;
        }
        Ok(())
    }

    fn make_room_patch(&mut self, room_filename: &str, project_names: &[String]) -> Result<()> {
        let base_room_path = self.mosaic_dir.join("Projects/Base/Export/Rooms").join(room_filename);
        let base_room_str = std::fs::read_to_string(&base_room_path)
            .with_context(|| format!("Unable to load room at {}", base_room_path.display()))?;
        let base_room: smart_xml::Room = serde_xml_rs::from_str(base_room_str.as_str())
            .with_context(|| format!("Unable to parse XML in {}", base_room_path.display()))?;

        
        // let state_ptrs = get_room_state_ptrs(&self.rom, room_ptr)?;

        // for state_xml in &base_room.states.state {
        //     let level_data_vec = vec![];
        //     for project in project_names {
        //         let project_path = self.mosaic_dir.join("Projects").join(project);
        //         let room_path = project_path.join("Export/Rooms").join(room_filename);
        //         let room_str = std::fs::read_to_string(&room_path)
        //             .with_context(|| format!("Unable to load room at {}", room_path.display()))?;
        //         let room: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str())
        //             .with_context(|| format!("Unable to parse XML in {}", room_path.display()))?;

        //         let level_data = extract_uncompressed_level_data(state_xml);
        //         let compressed_level_data = self.get_compressed_data(&level_data)?;
        //         level_data_vec.push(compressed_level_data);
        //     }
        //     let max_level_data_size = level_data_vec.iter().map(|x| x.len()).max().unwrap_or(0);
        //     let level_data_addr = self.main_allocator.allocate(max_level_data_size);

        //     for (, project) in project_names.iter().enumerate() {
        //         let mut new_rom = self.rom.clone();

        //         new_rom.write_n(level_data_addr, &level_data)?;
        //         new_rom.write_u24(state_ptr, pc2snes(level_data_addr) as isize)?;

        //         let patch = flips::BpsDeltaBuilder::new()
        //             .source(&self.rom.data)
        //             .target(&new_rom.data)
        //             .build()?;
        //         let output_path = self.output_patches_dir.join(format!("{}.bps"));
        //         std::fs::write(&output_path, &patch)?;
        //     }
        // }
        Ok(())
    }        
}

fn read_json(path: &Path) -> Result<JsonValue> {
    let file = File::open(path).with_context(|| format!("unable to open {}", path.display()))?;
    let json_str = std::io::read_to_string(file)
        .with_context(|| format!("unable to read {}", path.display()))?;
    let json_data =
        json::parse(&json_str).with_context(|| format!("unable to parse {}", path.display()))?;
    Ok(json_data)
}

// Returns a mapping from (area_id, room_id) to room pointer.
fn load_room_ptrs(sm_json_data_path: &Path) -> Result<Vec<usize>> {
    let room_pattern = sm_json_data_path.to_str().unwrap().to_string() + "/region/**/*.json";
    let mut out: Vec<usize> = vec![];
    for entry in glob::glob(&room_pattern).unwrap() {
        if let Ok(path) = entry {
            let path_str = path.to_str().with_context(|| {
                format!("Unable to convert path to string: {}", path.display())
            })?;
            if path_str.contains("ceres") || path_str.contains("roomDiagrams") {
                continue;
            }
            let room_json = read_json(&path)?;
            let room_ptr =
                parse_int::parse::<usize>(room_json["roomAddress"].as_str().unwrap()).unwrap();
            out.push(room_ptr);
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    let args = Args::parse();

    let sm_json_data_path = Path::new("../sm-json-data");
    let room_ptrs = load_room_ptrs(sm_json_data_path)?;

    let main_allocator = Allocator::new(vec![
        // (snes2pc(0xBAC629), snes2pc(0xC2C2BB)), // Vanilla tile GFX, tilemaps, and palettes, which we overwrite
        (snes2pc(0xBAC629), snes2pc(0xCF8000)), // Vanilla tile GFX, tilemaps, palettes, and level data, which we overwrite
        (snes2pc(0xE18000), snes2pc(0xE20000)),
        (snes2pc(0xE2B000), snes2pc(0xE30000)),
        (snes2pc(0xE3B000), snes2pc(0xE40000)),
        (snes2pc(0xE4B000), snes2pc(0xE50000)),
        (snes2pc(0xE5B000), snes2pc(0xE60000)),
        (snes2pc(0xE6B000), snes2pc(0xE70000)),
        (snes2pc(0xE7B000), snes2pc(0xE80000)),
        (snes2pc(0xE99000), snes2pc(0xEA0000)),
        (snes2pc(0xEA8000), snes2pc(0xF80000)),
    ]);

    let fx_allocator = Allocator::new(vec![
        (snes2pc(0x838000), snes2pc(0x8388FC)),
        (snes2pc(0x839AC2), snes2pc(0x83A0A4)),
        (snes2pc(0x83A0D4), snes2pc(0x83A18A)),
        (snes2pc(0x83F000), snes2pc(0x840000)),
    ]);
    let mut mosaic_builder = MosaicBuilder {
        rom: Rom::load(&args.input_rom)?,
        compressed_data_cache_dir: Path::new("../compressed_data").to_owned(),
        compressor_path: args.compressor.clone(),
        tmp_dir: Path::new("../tmp").to_owned(),
        mosaic_dir: Path::new("../Mosaic").to_owned(),
        output_patches_dir: Path::new("../patches/bps").to_owned(),
        main_allocator,
        fx_allocator,
    };
    std::fs::create_dir_all(&mosaic_builder.tmp_dir)?;
    std::fs::create_dir_all(&mosaic_builder.compressed_data_cache_dir)?;
    std::fs::create_dir_all(&mosaic_builder.output_patches_dir)?;

    mosaic_builder.make_tileset_patch()?;
    mosaic_builder.make_all_room_patches()?;
    // let projects_path = mosaic_dir.join("Projects");
    // for project_dir in std::fs::read_dir(projects_path)? {
    //     let project_dir = project_dir?;
    //     println!("Project: {}", project_dir.file_name().to_str().unwrap());

    // // Process CRE tileset:
    // let cre_path = project_dir.path().join("Export/Tileset/CRE/00");
    // let gfx8x8_path = cre_path.join("8x8tiles.gfx");
    // write_compressed_file(output_dir, compressor_path, tmp_dir, &gfx8x8_path)?;
    // let gfx16x16_path = cre_path.join("16x16tiles.ttb");
    // write_compressed_file(output_dir, compressor_path, tmp_dir, &gfx16x16_path)?;

    // // Process SRE tilesets:
    // let sce_path = project_dir.path().join("Export/Tileset/SCE");
    // for tileset_path in std::fs::read_dir(sce_path)? {
    //     let tileset_path = tileset_path?.path();

    //     let palette_path = tileset_path.join("palette.snes");
    //     write_compressed_file(output_dir, compressor_path, tmp_dir, &palette_path)?;

    //     let gfx8x8_path = tileset_path.join("8x8tiles.gfx");
    //     write_compressed_file(output_dir, compressor_path, tmp_dir, &gfx8x8_path)?;

    //     let gfx16x16_path = tileset_path.join("16x16tiles.ttb");
    //     write_compressed_file(output_dir, compressor_path, tmp_dir, &gfx16x16_path)?;
    // }

    // // Process rooms:
    // let room_dir = project_dir.path().join("Export/Rooms");
    // for room_path in std::fs::read_dir(room_dir)? {
    //     let room_path = room_path?.path();
    //     let room_str = std::fs::read_to_string(&room_path)?;
    //     let room: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str()).unwrap();
    //     for state in &room.states.state {
    //         let level_data = extract_uncompressed_level_data(state);
    //         write_compressed_data(output_dir, compressor_path, tmp_dir, &level_data, room_path.to_str().unwrap())?;
    //     }
    // }
    // }
    Ok(())
}
