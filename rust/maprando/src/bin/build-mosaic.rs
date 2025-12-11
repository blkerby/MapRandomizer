// TODO: consider removing this later. It's not a bad lint but I don't want to deal with it now.
#![allow(clippy::too_many_arguments)]

use anyhow::{Context, Result};
use clap::Parser;
use hashbrown::HashMap;
use hashbrown::hash_map::Entry;
use json::JsonValue;
use log::{error, info};
use maprando::customize::Allocator;
use maprando::patch::bps::BPSEncoder;
use maprando::patch::suffix_tree::SuffixTree;
use maprando::patch::{Rom, get_room_state_ptrs, pc2snes, snes2pc};
use maprando_game::{
    DoorPtr, RoomGeometry,
    smart_xml::{self, Layer2Type, RoomState, Screen},
};
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    compressor: Option<PathBuf>,
    #[arg(long)]
    input_rom: PathBuf,
}

#[derive(Deserialize, Default, Clone, Debug)]
struct TransitData {
    /// Intersecting room name (matching room geometry)
    name: String,
    x: Vec<usize>,
    /// Transit tube theme above room (matching room name in TransitTube SMART project)
    top: String,
    /// Transit tube theme below room (matching room name in TransitTube SMART project)
    bottom: String,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct FXDoor {
    pub room_area: usize,
    pub room_index: usize,
    pub door_index: usize,
}

struct MosaicPatchBuilder {
    rom: Rom,
    source_suffix_tree: SuffixTree,
    room_ptr_map: HashMap<(usize, usize), usize>,
    compressed_data_cache_dir: PathBuf,
    compressor_path: Option<PathBuf>,
    tmp_dir: PathBuf,
    mosaic_dir: PathBuf,
    output_patches_dir: PathBuf,
    bgdata_map: HashMap<smart_xml::BGData, isize>, // Mapping from BGData to pointer
    fx_door_map: HashMap<FXDoor, DoorPtr>,
    main_allocator: Allocator,
    fx_allocator: Allocator,
    dust_torizo_bgdata_ptr: usize,
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
    let mut level_data: Vec<u8> = vec![];
    level_data.push(((num_tiles * 2) & 0xFF) as u8);
    level_data.push(((num_tiles * 2) >> 8) as u8);
    let layer1_fill_word = 0x8000;
    let layer2_fill_word = 0x0000;
    let bts_fill_byte = 0x00;

    for _ in 0..0x3200 {
        level_data.push((layer1_fill_word & 0xFF) as u8);
        level_data.push((layer1_fill_word >> 8) as u8);
    }
    extract_all_screen_words(
        &state_xml.level_data.layer_1.screen,
        &mut level_data[2..],
        width,
        height,
    );

    level_data.extend(vec![bts_fill_byte; 0x3200]);
    extract_all_screen_bytes(
        &state_xml.level_data.bts.screen,
        &mut level_data[0x6402..],
        width,
        height,
    );

    if state_xml.layer2_type == Layer2Type::Layer2 {
        for _ in 0..num_tiles {
            level_data.push((layer2_fill_word & 0xFF) as u8);
            level_data.push((layer2_fill_word >> 8) as u8);
        }
        extract_all_screen_words(
            &state_xml.level_data.layer_2.screen,
            &mut level_data[0x9602..],
            width,
            height,
        );
    }
    level_data
}

impl MosaicPatchBuilder {
    fn get_compressed_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let digest = crypto_hash::hex_digest(crypto_hash::Algorithm::SHA256, data);
        let output_path = self.compressed_data_cache_dir.join(digest);
        let output: Vec<u8>;
        if !output_path.exists() {
            if let Some(compressor_path) = &self.compressor_path {
                let tmp_path = self.tmp_dir.join("tmpfile");
                std::fs::write(&tmp_path, data)?;
                Command::new(compressor_path)
                    .arg("-c")
                    .arg(format!("-f={}", tmp_path.to_str().unwrap()))
                    .arg(format!("-o={}", output_path.to_str().unwrap()))
                    .status()
                    .context("error running compressor")?;
                output = std::fs::read(output_path)?;
            } else {
                output = lznint::compress(data);
                std::fs::write(&output_path, output.clone())?;
            }
        } else {
            output = std::fs::read(output_path)?;
        }
        let decompressed_data = lznint::decompress(&output)?;
        assert_eq!(data, decompressed_data);
        Ok(output)
    }

    fn get_fx_data(&self, state_xml: &smart_xml::RoomState, default_only: bool) -> Vec<u8> {
        let mut out: Vec<u8> = vec![];

        for fx in &state_xml.fx1s.fx1 {
            if default_only && !fx.default {
                continue;
            }

            // Door pointer (for non-default FX) is map-specific and will be substituted during customization.
            if fx.default {
                out.extend(0u16.to_le_bytes());
            } else {
                let fx_door = FXDoor {
                    room_area: fx.roomarea,
                    room_index: fx.roomindex,
                    door_index: fx.fromdoor,
                };
                let door_ptr = self.fx_door_map[&fx_door] as u16;
                out.extend(door_ptr.to_le_bytes());
            }

            out.extend((fx.surfacestart as u16).to_le_bytes());
            out.extend((fx.surfacenew as u16).to_le_bytes());
            out.extend((fx.surfacespeed as u16).to_le_bytes());
            out.extend([fx.surfacedelay as u8]);
            out.extend([fx.type_ as u8]);
            out.extend([fx.transparency1_a as u8]);
            out.extend([fx.transparency2_b as u8]);
            out.extend([fx.liquidflags_c as u8]);
            out.extend([fx.paletteflags as u8]);
            out.extend([fx.animationflags as u8]);
            out.extend([fx.paletteblend as u8]);
        }
        if out.is_empty() {
            out.extend(vec![0xFF, 0xFF]);
        }

        out
    }

    fn build_bgdata_map(&mut self) -> Result<()> {
        info!("Processing BGData");
        let base_rooms_dir = self.mosaic_dir.join("Projects/Base/Export/Rooms/");
        for room_path in std::fs::read_dir(base_rooms_dir)? {
            let room_path = room_path?.path();
            let room_str = std::fs::read_to_string(&room_path)
                .with_context(|| format!("Unable to load room at {}", room_path.display()))?;
            let room_xml: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str())
                .with_context(|| format!("Unable to parse XML in {}", room_path.display()))?;
            let room_ptr = self
                .room_ptr_map
                .get(&(room_xml.area, room_xml.index))
                .copied()
                .unwrap_or(0);
            if room_ptr == 0 {
                continue;
            }
            let state_ptrs = get_room_state_ptrs(&self.rom, room_ptr)?;
            for (state_idx, state_xml) in room_xml.states.state.into_iter().enumerate() {
                let (_event_ptr, state_ptr) = state_ptrs[state_idx];
                let bg_ptr = self.rom.read_u16(state_ptr + 22)?;
                self.bgdata_map.insert(state_xml.bg_data, bg_ptr);
            }
        }
        Ok(())
    }

    fn build_fx_door_map(&mut self) -> Result<()> {
        info!("Processing FX doors");
        let base_rooms_dir = self.mosaic_dir.join("Projects/Base/Export/Rooms/");
        for room_path in std::fs::read_dir(base_rooms_dir)? {
            let room_path = room_path?.path();
            let room_str = std::fs::read_to_string(&room_path)
                .with_context(|| format!("Unable to load room at {}", room_path.display()))?;
            let room_xml: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str())
                .with_context(|| format!("Unable to parse XML in {}", room_path.display()))?;
            let room_ptr = self
                .room_ptr_map
                .get(&(room_xml.area, room_xml.index))
                .copied()
                .unwrap_or(0);
            if room_ptr == 0 {
                continue;
            }
            let state_ptrs = get_room_state_ptrs(&self.rom, room_ptr)?;
            for (state_idx, state_xml) in room_xml.states.state.into_iter().enumerate() {
                let (_event_ptr, state_ptr) = state_ptrs[state_idx];
                let fx_ptr = self.rom.read_u16(state_ptr + 6)? as usize;
                for (i, fx) in state_xml.fx1s.fx1.iter().enumerate() {
                    let fx_door = FXDoor {
                        room_area: fx.roomarea,
                        room_index: fx.roomindex,
                        door_index: fx.fromdoor,
                    };
                    let door_ptr =
                        self.rom.read_u16(snes2pc(0x830000 + fx_ptr + i * 16))? as DoorPtr;
                    self.fx_door_map.insert(fx_door, door_ptr);
                }
            }
        }
        Ok(())
    }

    fn make_tileset_patch(&mut self) -> Result<()> {
        info!("Processing tilesets");
        let mut new_rom = self.rom.clone();
        new_rom.enable_tracking();
        self.apply_cre_tileset(&mut new_rom)?;
        self.apply_sce_tilesets(&mut new_rom)?;

        let modified_ranges = new_rom.get_modified_ranges();
        let mut encoder =
            BPSEncoder::new(&self.source_suffix_tree, &new_rom.data, &modified_ranges);
        encoder.encode();
        let output_path = self.output_patches_dir.join("tilesets.bps");
        std::fs::write(&output_path, &encoder.patch_bytes)?;

        Ok(())
    }

    fn apply_cre_tileset(&mut self, new_rom: &mut Rom) -> Result<()> {
        let tileset_path = self.mosaic_dir.join("Projects/Base/Export/Tileset/CRE/00/");

        // Write CRE 8x8 tile graphics and update pointers to it:
        let gfx8x8_path = tileset_path.join("8x8tiles.gfx");
        let gfx8x8_bytes = std::fs::read(&gfx8x8_path)
            .with_context(|| format!("Unable to load CRE 8x8 gfx at {}", gfx8x8_path.display()))?;
        // Truncate CRE tileset to exclude bottom 2 rows (used for item PLMs):
        let gfx8x8_bytes = gfx8x8_bytes[..11264].to_vec();
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
        let compressed_gfx16x16 = self.get_compressed_data(&gfx16x16_bytes)?;
        let gfx16x16_addr = self.main_allocator.allocate(gfx16x16_bytes.len())?;
        new_rom.write_n(gfx16x16_addr, &compressed_gfx16x16)?;
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

            if (0x0F..=0x14).contains(&tileset_idx) {
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

        new_rom.write_u16(
            snes2pc(0x82DF03),
            (new_tile_pointers_snes & 0xFFFF) as isize,
        )?;
        Ok(())
    }

    fn make_all_room_patches(&mut self, project_names: &[String]) -> Result<()> {
        let base_rooms_dir = self.mosaic_dir.join("Projects/Base/Export/Rooms/");
        for room_path in std::fs::read_dir(base_rooms_dir)? {
            let room_filename = room_path?.file_name().to_str().unwrap().to_owned();
            self.make_room_patch(&room_filename, project_names)?;
        }
        Ok(())
    }

    fn make_room_patch(&mut self, room_filename: &str, project_names: &[String]) -> Result<()> {
        let room_name = room_filename
            .strip_suffix(".xml")
            .context("Expecting room filename to end in .xml")?;
        let base_room_path = self
            .mosaic_dir
            .join("Projects/Base/Export/Rooms")
            .join(room_filename);
        let base_room_str = std::fs::read_to_string(&base_room_path)
            .with_context(|| format!("Unable to load room at {}", base_room_path.display()))?;
        let base_room: smart_xml::Room = serde_xml_rs::from_str(base_room_str.as_str())
            .with_context(|| format!("Unable to parse XML in {}", base_room_path.display()))?;
        let room_ptr = self
            .room_ptr_map
            .get(&(base_room.area, base_room.index))
            .copied()
            .unwrap_or(0);
        if room_ptr == 0 {
            info!("Skipping {room_filename}");
            return Ok(());
        }

        info!(
            "Processing {}: main alloc {:?}, FX alloc {:?}",
            room_name,
            self.main_allocator.get_stats(),
            self.fx_allocator.get_stats()
        );
        let state_ptrs = get_room_state_ptrs(&self.rom, room_ptr)?;
        for (state_idx, &(_event_ptr, state_ptr)) in state_ptrs.iter().enumerate() {
            // println!("{}: {:x}", room_name, state_ptr);
            let mut compressed_level_data_vec = vec![];
            let mut fx_data_vec = vec![];
            let mut state_xml_vec = vec![];
            for project in project_names {
                let project_path = self.mosaic_dir.join("Projects").join(project);
                let room_path = project_path.join("Export/Rooms").join(room_filename);
                let room_str = std::fs::read_to_string(&room_path)
                    .with_context(|| format!("Unable to load room at {}", room_path.display()))?;
                let room: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str())
                    .with_context(|| format!("Unable to parse XML in {}", room_path.display()))?;
                let state_xml = &room.states.state[state_idx];
                let level_data = extract_uncompressed_level_data(state_xml);
                let compressed_level_data = self.get_compressed_data(&level_data)?;
                compressed_level_data_vec.push(compressed_level_data);
                let fx_data = self.get_fx_data(state_xml, false);
                fx_data_vec.push(fx_data);
                state_xml_vec.push(state_xml.clone());
            }

            // For a given room state, allocate enough space for the level data to fit whichever theme has the largest.
            // This approach can duplicate level data across room states, but we're not going to worry about that now.
            let max_level_data_size = compressed_level_data_vec
                .iter()
                .map(|x| x.len())
                .max()
                .unwrap_or(0);
            let level_data_addr = self.main_allocator.allocate(max_level_data_size)?;

            // Similarly, allocate enough space for FX data to fit whichever theme has the largest:
            let max_fx_data_size = fx_data_vec.iter().map(|x| x.len()).max().unwrap_or(0);
            let fx_data_addr = self.fx_allocator.allocate(max_fx_data_size)?;

            for (i, project) in project_names.iter().enumerate() {
                let mut new_rom = self.rom.clone();
                new_rom.enable_tracking();
                let state_xml = &state_xml_vec[i];

                if state_xml.gfx_set == 1 {
                    println!("Gfx set = 1: {room_name} {project}");
                }

                // Write the tileset index
                new_rom.write_u8(state_ptr + 3, state_xml.gfx_set as isize)?;

                // Write (or clear) the BGData pointer:
                if state_xml.layer2_type == Layer2Type::BGData {
                    if room_ptr == 0x7DC65 {
                        // Dust Torizo Room: allow a custom BGData (assumed to be the same across both room states)
                        new_rom.write_u16(state_ptr + 22, 0xBCD1)?;
                        for data in &state_xml.bg_data.data {
                            if data.type_ == "DECOMP" {
                                let mut data_u8: Vec<u8> = vec![];
                                for &w in &data.source {
                                    data_u8.push((w & 0xFF) as u8);
                                    data_u8.push((w >> 8) as u8);
                                }
                                let compressed_bgdata = self.get_compressed_data(&data_u8)?;
                                new_rom.write_n(self.dust_torizo_bgdata_ptr, &compressed_bgdata)?;
                                // BG pointer instruction to load tilemap:
                                new_rom.write_u16(snes2pc(0x8FBCD1), 0x0004)?;
                                new_rom.write_u24(
                                    snes2pc(0x8FBCD1 + 2),
                                    pc2snes(self.dust_torizo_bgdata_ptr) as isize,
                                )?;
                                new_rom.write_u16(snes2pc(0x8FBCD1 + 5), 0x4000)?;

                                info!("{}: {}", project, data_u8.len());
                                if data_u8.len() == 2048 {
                                    // Instruction to copy single screen of BGData twice:
                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 7), 0x0002)?;
                                    new_rom.write_u24(snes2pc(0x8FBCD1 + 9), 0x7E4000)?;
                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 12), 0x4800)?;
                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 14), 0x0800)?;

                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 16), 0x0002)?;
                                    new_rom.write_u24(snes2pc(0x8FBCD1 + 18), 0x7E4000)?;
                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 21), 0x4C00)?;
                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 23), 0x0800)?;

                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 25), 0x0000)?;
                                } else if data_u8.len() == 4096 {
                                    // Instructions to copy two screens of BGData
                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 7), 0x0002)?;
                                    new_rom.write_u24(snes2pc(0x8FBCD1 + 9), 0x7E4000)?;
                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 12), 0x4800)?;
                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 14), 0x1000)?;

                                    new_rom.write_u16(snes2pc(0x8FBCD1 + 16), 0x0000)?;
                                } else {
                                    panic!("Unexpected BGData size {}", data_u8.len());
                                }
                                break;
                            }
                        }
                    } else {
                        // All other rooms: BGData is required to match a vanilla one, so we look up which one:
                        let bg_ptr = self
                            .bgdata_map
                            .get(&state_xml.bg_data)
                            .copied()
                            .unwrap_or(0);
                        if bg_ptr == 0 {
                            error!("Unrecognized BGData in {project}");
                        }
                        new_rom.write_u16(state_ptr + 22, bg_ptr)?;
                    }
                } else {
                    new_rom.write_u16(state_ptr + 22, 0)?;
                }

                // Write BG scroll speeds:
                let mut speed_x = state_xml.layer2_xscroll;
                let mut speed_y = state_xml.layer2_yscroll;
                if state_xml.layer2_type == Layer2Type::BGData {
                    speed_x |= 0x01;
                    speed_y |= 0x01;
                }
                new_rom.write_u8(state_ptr + 12, speed_x as isize)?;
                new_rom.write_u8(state_ptr + 13, speed_y as isize)?;

                // Write the level data and the pointer to it:
                let level_data = &compressed_level_data_vec[i];
                new_rom.write_n(level_data_addr, level_data)?;
                new_rom.write_u24(state_ptr, pc2snes(level_data_addr) as isize)?;

                // Write FX:
                if pc2snes(room_ptr) & 0xFFFF == 0xDD58 {
                    // Mother Brain Room has special a FX not in the FX list, for lowering the acid after MB1 fight.
                    // Instead of replacing the whole FX header, patch the existing ones to use the correct
                    // palette blend for the acid:
                    new_rom.write_u8(snes2pc(0x83A0B3), 8)?;
                    new_rom.write_u8(snes2pc(0x83A0C3), 8)?;
                    new_rom.write_u8(snes2pc(0x83A0D3), 8)?;
                } else {
                    new_rom.write_n(fx_data_addr, &fx_data_vec[i])?;
                    new_rom.write_u16(state_ptr + 6, (pc2snes(fx_data_addr) & 0xFFFF) as isize)?;
                }

                // Write setup & main ASM pointers:
                if pc2snes(state_ptr) & 0xFFFF == 0xDDA2 {
                    // Don't overwrite ASM for special Mother Brain Room state used by randomizer for escape sequence.
                } else {
                    // Main ASM:
                    new_rom.write_u16(state_ptr + 18, state_xml.fx2 as isize)?;

                    // Setup ASM:
                    new_rom.write_u16(state_ptr + 24, state_xml.layer1_2 as isize)?;
                }

                // Encode the BPS patch:
                let modified_ranges = new_rom.get_modified_ranges();
                let mut encoder =
                    BPSEncoder::new(&self.source_suffix_tree, &new_rom.data, &modified_ranges);
                encoder.encode();

                // Save the BPS patch to a file:
                let output_filename = format!("{project}-{room_ptr:X}-{state_idx}.bps");
                let output_path = self.output_patches_dir.join(output_filename);
                std::fs::write(&output_path, &encoder.patch_bytes)?;
            }
        }
        Ok(())
    }

    fn load_room_state(project_path: &Path, room_name: &str) -> Result<RoomState> {
        let room_path = project_path
            .join("Export/Rooms")
            .join(format!("{room_name}.xml"));
        let room_str = std::fs::read_to_string(&room_path)
            .with_context(|| format!("Unable to load room at {}", room_path.display()))?;
        let room: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str())
            .with_context(|| format!("Unable to parse XML in {}", room_path.display()))?;
        let state_xml = room.states.state.last().unwrap().clone();
        Ok(state_xml)
    }

    fn copy_screen(
        dst_level_data: &mut [u8],
        dst_screen_x: usize,
        dst_screen_y: usize,
        dst_width: usize,
        src_level_data: &[u8],
        src_screen_x: usize,
        src_screen_y: usize,
        src_width: usize,
        src_layer_2: &[u8],
    ) {
        for y in 0..16 {
            for x in 0..16 {
                let src_x = src_screen_x * 16 + x;
                let src_y = src_screen_y * 16 + y;
                let src_i = src_y * src_width * 16 + src_x;
                let dst_x = dst_screen_x * 16 + x;
                let dst_y = dst_screen_y * 16 + y;
                let dst_i = dst_y * dst_width * 16 + dst_x;
                // Layer1:
                dst_level_data[2 + dst_i * 2] = src_level_data[2 + src_i * 2];
                dst_level_data[2 + dst_i * 2 + 1] = (dst_level_data[2 + dst_i * 2 + 1] & 0xF0)
                    | (src_level_data[2 + src_i * 2 + 1] & 0x0F);
                // We skip touching BTS, leaving it unchanged.
                // Layer2:
                dst_level_data[0x9602 + dst_i * 2] = src_layer_2[src_i * 2];
                dst_level_data[0x9602 + dst_i * 2 + 1] = src_layer_2[src_i * 2 + 1];
            }
        }
    }

    fn get_canonical_tileset(tileset_idx: usize) -> usize {
        if tileset_idx == 5 {
            4 // Wrecked Ship Powered Off -> Wrecked Ship Powered On
        } else if tileset_idx == 9 {
            10 // Norfair Hot -> Norfair Cool
        } else if tileset_idx == 12 {
            // This one is a little risky because these tilesets overlap but are not fully compatible.
            11 // East Maridia -> West Maridia
        } else {
            tileset_idx
        }
    }

    fn is_compatible_tileset(tileset_idx_1: usize, tileset_idx_2: usize) -> bool {
        Self::get_canonical_tileset(tileset_idx_1) == Self::get_canonical_tileset(tileset_idx_2)
    }

    fn get_layer_2_data(
        state_xml: &RoomState,
        level_data: &[u8],
        room_width: usize,
        room_height: usize,
    ) -> Vec<u8> {
        let size = level_data[0] as usize + ((level_data[1] as usize) << 8);
        if state_xml.layer2_type == Layer2Type::Layer2 {
            let mut out = level_data[0x9602..].to_vec();
            if state_xml.layer1_2 == 0x91C9 {
                // Scrolling sky BG: replicate first column of screens
                for sy in 0..room_height {
                    for sx in 1..room_width {
                        for y in 0..16 {
                            for x in 0..16 {
                                let src_i = (y + sy * 16) * room_width * 16 + x;
                                let dst_i = (y + sy * 16) * room_width * 16 + x + sx * 16;
                                out[dst_i * 2] = out[src_i * 2];
                                out[dst_i * 2 + 1] = out[src_i * 2 + 1];
                            }
                        }
                    }
                }
            }
            out
        } else {
            error!("Room has BGData instead of Layer2 background");
            vec![0; size]
        }
    }

    fn draw_tube(
        layer2: &mut [u8],
        room_width: usize,
        room_height: usize,
        screen_x: usize,
        priority: bool,
        tileset_idx: usize,
        theme_name: &str,
    ) {
        let maridia_style_tube = tileset_idx == 12 && theme_name != "YellowMaridia";
        for screen_y in 0..room_height {
            for y in 0..16 {
                let i = (screen_y * 16 + y) * room_width * 16 + screen_x * 16 + 7;
                let tile: u16 = match (maridia_style_tube, priority, y == 0 || y == 15) {
                    // Tube tiles specific to East Maridia (as in vanilla):
                    (true, true, true) => 0x3B4,
                    (true, true, false) => 0x3AB ^ 0x400,
                    (true, false, true) => 0x33D,
                    (true, false, false) => 0x33A ^ 0x400,
                    // Tube tiles in other tilesets:
                    (false, true, true) => 0xEE,
                    (false, true, false) => 0xF0,
                    (false, false, true) => 0xF1,
                    (false, false, false) => 0xF3,
                };
                let vflip = if y == 15 { 0x08 } else { 0x00 };
                (layer2[i * 2], layer2[i * 2 + 1]) =
                    ((tile & 0xFF) as u8, ((tile >> 8) as u8) ^ 0x04 ^ vflip);
                (layer2[i * 2 + 2], layer2[i * 2 + 3]) =
                    ((tile & 0xFF) as u8, ((tile >> 8) as u8) ^ vflip);
            }
        }
    }

    fn write_toilet_intersection_room(
        &mut self,
        new_rom: &mut Rom,
        state_xml: &RoomState,
        room_ptr: usize,
        level_data_addr: usize,
        compressed_level_data: &[u8],
    ) -> Result<()> {
        new_rom.write_n(level_data_addr, compressed_level_data)?;
        for (_event_ptr, state_ptr) in get_room_state_ptrs(&self.rom, room_ptr)? {
            new_rom.write_u24(state_ptr, pc2snes(level_data_addr) as isize)?;

            // Set BG X scroll rate to 100%
            new_rom.write_u8(state_ptr + 12, 0x00)?;

            if state_xml.layer1_2 == 0x91C9 {
                // Disable scrolling sky, in order to be able to draw the tube in Layer2.
                new_rom.write_u16(state_ptr + 18, 0x0000)?;
                new_rom.write_u16(state_ptr + 24, 0x0000)?;
            }
        }
        Ok(())
    }

    fn make_toilet_patches(
        &mut self,
        dry_run: bool,
        max_transit_level_data: &mut usize,
        max_intersection_level_data: &mut usize,
        theme_names: &[String],
    ) -> Result<()> {
        let transit_level_data_addr = if !dry_run {
            self.main_allocator.allocate(*max_transit_level_data)?
        } else {
            0
        };

        let intersection_level_data_addr = if !dry_run {
            self.main_allocator.allocate(*max_intersection_level_data)?
        } else {
            0
        };

        let fx_data_addr = if !dry_run {
            self.fx_allocator.allocate(16)?
        } else {
            0
        };

        if !dry_run {
            info!(
                "Transit: main alloc {:?}, FX alloc {:?}",
                self.main_allocator.get_stats(),
                self.fx_allocator.get_stats()
            );
        }

        // Index SMART project rooms:
        let mut room_name_by_pair: HashMap<(usize, usize), String> = HashMap::new();
        for room_path in std::fs::read_dir(self.mosaic_dir.join("Projects/Base/Export/Rooms"))? {
            let room_path = room_path?.path();
            let room_filename = room_path.file_name().unwrap().to_str().unwrap().to_owned();
            let room_name = room_filename
                .strip_suffix(".xml")
                .context("Expecting room filename to end in .xml")?;
            let room_str = std::fs::read_to_string(&room_path)
                .with_context(|| format!("Unable to load room at {}", room_path.display()))?;
            let room: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str())
                .with_context(|| format!("Unable to parse XML in {}", room_path.display()))?;
            room_name_by_pair.insert((room.area, room.index), room_name.to_string());
        }

        let room_geometry_path = Path::new("../room_geometry.json");
        let room_geometry_str = std::fs::read_to_string(room_geometry_path).with_context(|| {
            format!(
                "Unable to load room geometry at {}",
                room_geometry_path.display()
            )
        })?;
        let room_geometry: Vec<RoomGeometry> = serde_json::from_str(&room_geometry_str)?;
        let mut room_idx_by_name: HashMap<String, usize> = HashMap::new();
        for (i, room) in room_geometry.iter().enumerate() {
            room_idx_by_name.insert(room.name.clone(), i);
        }

        let transit_tube_data_path = Path::new("../transit-tube-data");
        for theme_name in theme_names {
            let theme_transit_data_path = transit_tube_data_path.join(format!("{theme_name}.json"));
            let theme_transit_data_str = std::fs::read_to_string(&theme_transit_data_path)
                .with_context(|| {
                    format!(
                        "Unable to load transit tube data at {}",
                        theme_transit_data_path.display()
                    )
                })?;
            let theme_transit_data_vec: Vec<TransitData> =
                serde_json::from_str(&theme_transit_data_str)?;

            let transit_project_path = self.mosaic_dir.join("Projects/TransitTube");
            let theme_project_path = self.mosaic_dir.join("Projects").join(theme_name);

            for transit_data in &theme_transit_data_vec {
                info!("{} transit room: {}", theme_name, transit_data.name);
                let room_idx = room_idx_by_name[&transit_data.name];
                let room_geometry = &room_geometry[room_idx];
                let room_ptr = room_geometry.rom_address;
                let room_area = self.rom.read_u8(room_ptr + 1)? as usize;
                let room_index = self.rom.read_u8(room_ptr)? as usize;
                let room_width = self.rom.read_u8(room_ptr + 4)? as usize;
                let room_height = self.rom.read_u8(room_ptr + 5)? as usize;
                let smart_room_name = &room_name_by_pair[&(room_area, room_index)];

                let tube_theme_top = transit_data.top.to_ascii_uppercase();
                let tube_theme_bottom = transit_data.bottom.to_ascii_uppercase();

                let top_state_xml = Self::load_room_state(&transit_project_path, &tube_theme_top)?;
                let bottom_state_xml =
                    Self::load_room_state(&transit_project_path, &tube_theme_bottom)?;
                let middle_state_xml = Self::load_room_state(&theme_project_path, smart_room_name)?;
                let twin_state_xml = if room_geometry.name == "Pants Room" {
                    let east_pants_room_name = room_name_by_pair[&(4, 37)].as_str();
                    Some(Self::load_room_state(
                        &theme_project_path,
                        east_pants_room_name,
                    )?)
                } else if room_geometry.name == "West Ocean" {
                    let homing_geemer_room_name = room_name_by_pair[&(0, 17)].as_str();
                    Some(Self::load_room_state(
                        &theme_project_path,
                        homing_geemer_room_name,
                    )?)
                } else if room_geometry.name == "Aqueduct" {
                    let botwoon_hallway_name = room_name_by_pair[&(4, 35)].as_str();
                    Some(Self::load_room_state(
                        &theme_project_path,
                        botwoon_hallway_name,
                    )?)
                } else {
                    None
                };

                let tileset_idx = middle_state_xml.gfx_set;
                assert!(Self::is_compatible_tileset(
                    top_state_xml.gfx_set,
                    tileset_idx
                ));
                assert!(Self::is_compatible_tileset(
                    bottom_state_xml.gfx_set,
                    tileset_idx
                ));

                let top_level_data = extract_uncompressed_level_data(&top_state_xml);
                let bottom_level_data = extract_uncompressed_level_data(&bottom_state_xml);
                let middle_level_data = extract_uncompressed_level_data(&middle_state_xml);
                let compressed_twin_level_data = twin_state_xml.as_ref().map(|tx| {
                    let orig_level_data = extract_uncompressed_level_data(tx);
                    let mut level_data = orig_level_data.clone();
                    let (width, height, x) = if room_geometry.name == "Pants Room" {
                        (1, 3, 0)
                    } else if room_geometry.name == "West Ocean" {
                        (1, 1, 0)
                    } else if room_geometry.name == "Aqueduct" {
                        (4, 1, 2)
                    } else {
                        panic!("Unrecognized room with twin data: {}", room_geometry.name)
                    };
                    let mut layer_2 = Self::get_layer_2_data(
                        twin_state_xml.as_ref().unwrap(),
                        &level_data,
                        width,
                        height,
                    );
                    let twin_tileset_idx = twin_state_xml.as_ref().unwrap().gfx_set;
                    Self::draw_tube(
                        &mut layer_2,
                        width,
                        height,
                        x,
                        false,
                        twin_tileset_idx,
                        theme_name,
                    );

                    for sy in 0..height {
                        Self::copy_screen(
                            &mut level_data,
                            x,
                            sy,
                            width,
                            &orig_level_data,
                            x,
                            sy,
                            width,
                            &layer_2,
                        );
                    }

                    self.get_compressed_data(&level_data).unwrap()
                });

                let top_layer_2 = Self::get_layer_2_data(&top_state_xml, &top_level_data, 1, 10);
                let bottom_layer_2 =
                    Self::get_layer_2_data(&bottom_state_xml, &bottom_level_data, 1, 10);
                let orig_middle_layer_2 = Self::get_layer_2_data(
                    &middle_state_xml,
                    &middle_level_data,
                    room_width,
                    room_height,
                );

                for &x in &transit_data.x {
                    let mut y_min = isize::MAX;
                    let mut y_max = 0;

                    let mut middle_layer_2 = orig_middle_layer_2.clone();
                    Self::draw_tube(
                        &mut middle_layer_2,
                        room_width,
                        room_height,
                        x,
                        true,
                        tileset_idx,
                        theme_name,
                    );

                    for y in 0..(room_geometry.map.len() as isize) {
                        if room_geometry.map[y as usize][x] == 1 {
                            if y < y_min {
                                y_min = y;
                            }
                            if y > y_max {
                                y_max = y;
                            }
                        }
                    }

                    // Now construct level data for the intersecting room, modified to show the tube passing through.
                    // This is independent of the vertical alignment of the tube (i.e. how many screens above it starts).
                    let mut new_middle_level_data = middle_level_data.clone();
                    let mut middle_layer_2_behind = orig_middle_layer_2.clone();
                    Self::draw_tube(
                        &mut middle_layer_2_behind,
                        room_width,
                        room_height,
                        x,
                        false,
                        tileset_idx,
                        theme_name,
                    );
                    for sy in 0..room_height {
                        for sx in 0..room_width {
                            Self::copy_screen(
                                &mut new_middle_level_data,
                                sx,
                                sy,
                                room_width,
                                &middle_level_data,
                                sx,
                                sy,
                                room_width,
                                &middle_layer_2_behind,
                            );
                        }
                    }

                    let compressed_middle_level_data =
                        self.get_compressed_data(&new_middle_level_data)?;
                    let twin_level_data_len =
                        compressed_twin_level_data.as_ref().map_or(0, |x| x.len());
                    if dry_run
                        && compressed_middle_level_data.len() + twin_level_data_len
                            > *max_intersection_level_data
                    {
                        *max_intersection_level_data =
                            compressed_middle_level_data.len() + twin_level_data_len;
                    }

                    // Construct level data for the Toilet room, one version for each possible vertical position:
                    assert!(2 - y_min < 8 - y_max);
                    for y in (2 - y_min)..(8 - y_max) {
                        let mut transit_level_data = bottom_level_data.clone();

                        // Top part of the tube:
                        for sy in 0..(y + y_min - 1) {
                            Self::copy_screen(
                                &mut transit_level_data,
                                0,
                                sy as usize,
                                1,
                                &top_level_data,
                                0,
                                sy as usize,
                                1,
                                &top_layer_2,
                            );
                        }

                        // Tube screen immediately above the intersecting room:
                        Self::copy_screen(
                            &mut transit_level_data,
                            0,
                            (y + y_min - 1) as usize,
                            1,
                            &top_level_data,
                            0,
                            4,
                            1,
                            &top_layer_2,
                        );

                        if y + y_min - 1 > 4 {
                            assert!(y + y_min - 1 == 5);
                            // One more tube screen above: make it a connecting screen instead of a terminator
                            Self::copy_screen(
                                &mut transit_level_data,
                                0,
                                4,
                                1,
                                &top_level_data,
                                0,
                                2,
                                1,
                                &top_layer_2,
                            );
                        }

                        // Intersecting room
                        for sy in y_min..=y_max {
                            Self::copy_screen(
                                &mut transit_level_data,
                                0,
                                (y + sy) as usize,
                                1,
                                &middle_level_data,
                                x,
                                sy as usize,
                                room_width,
                                &middle_layer_2,
                            );
                        }

                        // Tube screen immediately below the intersecting room:
                        Self::copy_screen(
                            &mut transit_level_data,
                            0,
                            (y + y_max + 1) as usize,
                            1,
                            &bottom_level_data,
                            0,
                            5,
                            1,
                            &bottom_layer_2,
                        );

                        if y + y_max + 1 < 5 {
                            assert!(y + y_max + 1 == 4);
                            // One more tube screen below: make it a connecting screen instead of a terminator
                            Self::copy_screen(
                                &mut transit_level_data,
                                0,
                                5,
                                1,
                                &bottom_level_data,
                                0,
                                7,
                                1,
                                &bottom_layer_2,
                            );
                        }

                        let compressed_transit_level_data =
                            self.get_compressed_data(&transit_level_data)?;

                        if dry_run {
                            if compressed_transit_level_data.len() > *max_transit_level_data {
                                *max_transit_level_data = compressed_transit_level_data.len();
                            }
                        } else {
                            let mut new_rom = self.rom.clone();
                            new_rom.enable_tracking();
                            let toilet_state_ptr = 0x7D415;

                            // Write the tileset index
                            new_rom.write_u8(toilet_state_ptr + 3, tileset_idx as isize)?;

                            // Set enemy list to empty:
                            new_rom.write_u16(toilet_state_ptr + 8, 0x85a9)?;
                            new_rom.write_u16(toilet_state_ptr + 10, 0x80eb)?;

                            // Write the transit level data and the pointer to it:
                            new_rom
                                .write_n(transit_level_data_addr, &compressed_transit_level_data)?;
                            new_rom.write_u24(
                                toilet_state_ptr,
                                pc2snes(transit_level_data_addr) as isize,
                            )?;

                            // Set BG scroll rate to 100%
                            new_rom.write_u8(toilet_state_ptr + 13, 0x00)?;

                            // Setup ASM:
                            // Copy from middle room if it's the type that loads special tiles (for Crab Hole, etc.)
                            if middle_state_xml.layer1_2 == 0xC11B {
                                new_rom.write_u16(
                                    toilet_state_ptr + 24,
                                    middle_state_xml.layer1_2 as isize,
                                )?;
                            }

                            // Write FX:
                            new_rom.write_u16(
                                toilet_state_ptr + 6,
                                (pc2snes(fx_data_addr) & 0xFFFF) as isize,
                            )?;
                            let mut fx_data = self.get_fx_data(&middle_state_xml, true);
                            assert!(fx_data.len() <= 16);
                            if fx_data.len() == 16 {
                                if fx_data[3] != 0xFF {
                                    // Adjust liquid level start:
                                    fx_data[3] = ((fx_data[3] as i8) + (y as i8)) as u8;
                                }
                                if fx_data[5] != 0xFF {
                                    // Adjust liquid level new:
                                    fx_data[5] = ((fx_data[5] as i8) + (y as i8)) as u8;
                                }
                                fx_data[13] &= 0x7F; // Disable heat FX
                            }
                            new_rom.write_n(fx_data_addr, &fx_data)?;

                            // Write level data and other modifications for the intersecting room:
                            self.write_toilet_intersection_room(
                                &mut new_rom,
                                &middle_state_xml,
                                room_ptr,
                                intersection_level_data_addr,
                                &compressed_middle_level_data,
                            )?;

                            if let Some(twin_state_xml) = twin_state_xml.as_ref()
                                && room_geometry.name != "Aqueduct"
                                && (room_idx != 138 || x == 0)
                            {
                                let twin_room_ptr = if room_idx == 138 { 0x7D69A } else { 0x7968F };
                                self.write_toilet_intersection_room(
                                    &mut new_rom,
                                    twin_state_xml,
                                    twin_room_ptr,
                                    intersection_level_data_addr
                                        + compressed_middle_level_data.len(),
                                    compressed_twin_level_data.as_ref().unwrap(),
                                )?;
                            }

                            // Encode the BPS patch:
                            let modified_ranges = new_rom.get_modified_ranges();
                            let mut encoder = BPSEncoder::new(
                                &self.source_suffix_tree,
                                &new_rom.data,
                                &modified_ranges,
                            );
                            encoder.encode();

                            // Save the BPS patch to a file:
                            let output_filename =
                                format!("{}-{:X}-Transit-{}-{}.bps", theme_name, room_ptr, x, -y);
                            let output_path = self.output_patches_dir.join(output_filename);
                            std::fs::write(&output_path, &encoder.patch_bytes)?;

                            if room_geometry.name == "Aqueduct" && x == 2 && y == 4 {
                                // Make a special patch for vanilla map tube position, where it passes through
                                // both Botwoon Hallway and Aqueduct, but where we leave the Toilet room itself untouched.

                                let mut new_rom = self.rom.clone();
                                new_rom.enable_tracking();

                                // Write tube in background (Layer2) of Aqueduct
                                self.write_toilet_intersection_room(
                                    &mut new_rom,
                                    &middle_state_xml,
                                    room_ptr,
                                    intersection_level_data_addr,
                                    &compressed_middle_level_data,
                                )?;

                                // Write tube in background (Layer2) of Botwoon Hallway
                                self.write_toilet_intersection_room(
                                    &mut new_rom,
                                    twin_state_xml.as_ref().unwrap(),
                                    0x7D617,
                                    intersection_level_data_addr
                                        + compressed_middle_level_data.len(),
                                    compressed_twin_level_data.as_ref().unwrap(),
                                )?;

                                // Encode the BPS patch:
                                let modified_ranges = new_rom.get_modified_ranges();
                                let mut encoder = BPSEncoder::new(
                                    &self.source_suffix_tree,
                                    &new_rom.data,
                                    &modified_ranges,
                                );
                                encoder.encode();

                                // Save the BPS patch to a file:
                                let output_filename = format!("{theme_name}-VanillaMapTransit.bps");
                                info!("Writing {output_filename}");
                                let output_path = self.output_patches_dir.join(output_filename);
                                std::fs::write(&output_path, &encoder.patch_bytes)?;
                            }
                        }
                    }
                }
            }
        }
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

// Returns a list of room pointers used by the randomizer.
fn load_room_ptrs(sm_json_data_path: &Path) -> Result<Vec<usize>> {
    let room_pattern = sm_json_data_path.to_str().unwrap().to_string() + "/region/**/*.json";
    let mut out: Vec<usize> = vec![];
    for path in glob::glob(&room_pattern).unwrap().flatten() {
        let path_str = path
            .to_str()
            .with_context(|| format!("Unable to convert path to string: {}", path.display()))?;
        if path_str.contains("ceres") || path_str.contains("roomDiagrams") {
            continue;
        }
        let room_json = read_json(&path)?;
        let room_ptr =
            parse_int::parse::<usize>(room_json["roomAddress"].as_str().unwrap()).unwrap();
        out.push(room_ptr);
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

    let mut main_allocator = Allocator::new(vec![
        // Vanilla tile GFX, which we overwrite:
        (snes2pc(0xBAC629), snes2pc(0xBB8000)),
        // Skipping bank BB, used by Mosaic "Area Palette Glows"
        (snes2pc(0xBC8000), snes2pc(0xC08000)),
        // Skipping banks C0 and C1, used by Mosaic "Area Palettes"
        // Vanilla palettes and level data, which we overwrite:
        (snes2pc(0xC2C2BB), snes2pc(0xCEB240)),
        // Skipping rest of bank CE: used by credits data
        (snes2pc(0xE08000), snes2pc(0xE10000)),
        (snes2pc(0xE18000), snes2pc(0xE20000)),
        // Skipping bank E2, E3, and E4: used for map tile graphics
        (snes2pc(0xE58000), snes2pc(0xE80000)),
        // Skipping bank E9, used for hazard markers and title screen graphics
        // Skipping bank EA, used for beam door graphics
        (snes2pc(0xEB8000), snes2pc(0xF80000)),
        // Skipping banks F8 through FF: used by SpriteSomething for Samus sprite customization
    ]);

    let fx_allocator = Allocator::new(vec![
        (snes2pc(0x838000), snes2pc(0x8388FC)),
        (snes2pc(0x839AC2), snes2pc(0x83A0A4)),
        (snes2pc(0x83A0D4), snes2pc(0x83A18A)),
        (snes2pc(0x83F000), snes2pc(0x840000)),
    ]);

    info!(
        "Initial state: main alloc {:?}, FX alloc {:?}",
        main_allocator.get_stats(),
        fx_allocator.get_stats()
    );

    let project_names: Vec<String> = vec![
        "Base",
        "OuterCrateria",
        "InnerCrateria",
        "BlueBrinstar",
        "GreenBrinstar",
        "PinkBrinstar",
        "RedBrinstar",
        "WarehouseBrinstar",
        "UpperNorfair",
        "LowerNorfair",
        "WreckedShip",
        "WestMaridia",
        "YellowMaridia",
        "MechaTourian",
        "MetroidHabitat",
        "Outline",
        "Invisible",
    ]
    .into_iter()
    .map(|x| x.to_string())
    .collect();

    let mut rom = Rom::load(&args.input_rom)?;
    rom.data.resize(0x400000, 0xFF);
    let mut room_ptr_map: HashMap<(usize, usize), usize> = HashMap::new();
    for room_ptr in room_ptrs {
        let area = rom.read_u8(room_ptr + 1)?;
        let idx = rom.read_u8(room_ptr)?;
        room_ptr_map.insert((area as usize, idx as usize), room_ptr);
    }
    info!("Building vanilla ROM suffix tree");
    let source_suffix_tree = SuffixTree::new(&rom.data);
    info!("Done building vanilla ROM suffix tree");

    let dust_torizo_bgdata_ptr = main_allocator.allocate(4096)?;
    let mut mosaic_builder = MosaicPatchBuilder {
        rom,
        source_suffix_tree,
        room_ptr_map,
        bgdata_map: HashMap::new(),
        fx_door_map: HashMap::new(),
        compressed_data_cache_dir: Path::new("../compressed_data").to_owned(),
        compressor_path: args.compressor.clone(),
        tmp_dir: Path::new("../tmp").to_owned(),
        mosaic_dir: Path::new("../Mosaic").to_owned(),
        output_patches_dir: Path::new("../patches/mosaic").to_owned(),
        main_allocator,
        fx_allocator,
        dust_torizo_bgdata_ptr,
    };
    std::fs::create_dir_all(&mosaic_builder.tmp_dir)?;
    std::fs::create_dir_all(&mosaic_builder.compressed_data_cache_dir)?;
    std::fs::create_dir_all(&mosaic_builder.output_patches_dir)?;

    mosaic_builder.make_tileset_patch()?;
    mosaic_builder.build_bgdata_map()?;
    mosaic_builder.build_fx_door_map()?;
    mosaic_builder.make_all_room_patches(&project_names)?;

    // For Toilet, do a dry run first to determine size to allocate for level data
    // (based on max possible size across all possible themes and intersecting rooms):
    let mut max_transit_level_data = 0;
    let mut max_intersection_level_data = 0;
    mosaic_builder.make_toilet_patches(
        true,
        &mut max_transit_level_data,
        &mut max_intersection_level_data,
        &project_names,
    )?;
    mosaic_builder.make_toilet_patches(
        false,
        &mut max_transit_level_data,
        &mut max_intersection_level_data,
        &project_names,
    )?;
    Ok(())
}
