use anyhow::{Context, Result};
use clap::Parser;
use crypto_hash;
use flips::{self, BpsPatch};
use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use json::JsonValue;
use log::{info, error};
use maprando::patch::bps::BPSEncoder;
use maprando::patch::suffix_tree::SuffixTree;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use smart_xml::{Screen, Layer2Type};
use maprando::customize::Allocator;
use maprando::game_data::{smart_xml, DoorPtr};
use maprando::patch::{pc2snes, snes2pc, Rom, get_room_state_ptrs, self};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    compressor: PathBuf,
    #[arg(long)]
    input_rom: PathBuf,
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
    compressor_path: PathBuf,
    tmp_dir: PathBuf,
    mosaic_dir: PathBuf,
    output_patches_dir: PathBuf,
    bgdata_map: HashMap<smart_xml::BGData, isize>, // Mapping from BGData to pointer
    fx_door_map: HashMap<FXDoor, DoorPtr>,
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

impl MosaicPatchBuilder {
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

    fn get_fx_data(&self, state_xml: &smart_xml::RoomState) -> Vec<u8> {
        let mut out: Vec<u8> = vec![];

        for fx in &state_xml.fx1s.fx1 {
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
        if out.len() == 0 {
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
            let room_ptr = self.room_ptr_map.get(&(room_xml.area, room_xml.index)).map(|x| *x).unwrap_or(0);
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
            let room_ptr = self.room_ptr_map.get(&(room_xml.area, room_xml.index)).map(|x| *x).unwrap_or(0);
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
                    let door_ptr = self.rom.read_u16(snes2pc(0x830000 + fx_ptr + i * 16))? as DoorPtr;
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
        self.apply_palettes(&mut new_rom)?;

        // let patch = flips::BpsDeltaBuilder::new()
        //     .source(&self.rom.data)
        //     .target(&new_rom.data)
        //     .build()?;
        // let output_path = self.output_patches_dir.join("tilesets.bps");
        // std::fs::write(&output_path, &patch)?;
        let modified_ranges = new_rom.get_modified_ranges();
        let mut encoder = BPSEncoder::new(&self.source_suffix_tree, &new_rom.data, &modified_ranges);
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

        new_rom.write_u16(
            snes2pc(0x82DF03),
            (new_tile_pointers_snes & 0xFFFF) as isize,
        )?;
        Ok(())
    }

    fn apply_palettes(&mut self, new_rom: &mut Rom) -> Result<()> {
        let mut pal_map: HashMap<Vec<u8>, usize> = HashMap::new();
        let project_names: Vec<String> = vec![
            "CrateriaPalette",
            "BrinstarPalette",
            "NorfairPalette",
            "WreckedShipPalette",
            "MaridiaPalette",
            "TourianPalette",
        ].into_iter().map(|x| x.to_string()).collect();
        
        let main_palette_table_addr = snes2pc(0x80DD00); // Must match address used in area_palettes.asm
        let palette_table_end = snes2pc(0x80E100);
        let max_tilesets = 55;
        let mut area_palette_table_addrs = vec![];
        for i in 0..6 {
            let area_table_addr = main_palette_table_addr + 12 + i * max_tilesets * 3;
            area_palette_table_addrs.push(area_table_addr);
            new_rom.write_u16(main_palette_table_addr + 2 * i, (pc2snes(area_table_addr) & 0xFFFF) as isize)?;
        }

        for (area_idx, project) in project_names.iter().enumerate() {
            let tilesets_path = self.mosaic_dir.join("Projects").join(project).join("Export/Tileset/SCE");
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

                assert!(tileset_idx < max_tilesets);
                let palette_table_addr = area_palette_table_addrs[area_idx] + tileset_idx * 3;
                assert!(palette_table_addr + 2 <= palette_table_end);
                new_rom.write_u24(palette_table_addr, pc2snes(pal_addr) as isize)?;
                println!("{:x}, {:x}: {:x}", area_idx, tileset_idx, pc2snes(pal_addr));
            }
        }
        Ok(())
    }

    fn make_all_room_patches(&mut self) -> Result<()> {
        let project_names: Vec<String> = vec![
            "Base",
            "OuterCrateria",
            "InnerCrateria",
            "GreenBrinstar",
            "UpperNorfair",
            "WreckedShip",
        ].into_iter().map(|x| x.to_string()).collect();
        let base_rooms_dir = self.mosaic_dir.join("Projects/Base/Export/Rooms/");
        for room_path in std::fs::read_dir(base_rooms_dir)? {
            let room_filename = room_path?.file_name().to_str().unwrap().to_owned();
            self.make_room_patch(&room_filename, &project_names)?;
        }
        Ok(())
    }

    fn make_room_patch(&mut self, room_filename: &str, project_names: &[String]) -> Result<()> {
        let room_name = room_filename.strip_suffix(".xml").context("Expecting room filename to end in .xml")?;
        let base_room_path = self.mosaic_dir.join("Projects/Base/Export/Rooms").join(room_filename);
        let base_room_str = std::fs::read_to_string(&base_room_path)
            .with_context(|| format!("Unable to load room at {}", base_room_path.display()))?;
        let base_room: smart_xml::Room = serde_xml_rs::from_str(base_room_str.as_str())
            .with_context(|| format!("Unable to parse XML in {}", base_room_path.display()))?;
        let room_ptr = self.room_ptr_map.get(&(base_room.area, base_room.index)).map(|x| *x).unwrap_or(0);
        if room_ptr == 0 {
            info!("Skipping {}", room_filename);
            return Ok(());
        }
        
        info!("Processing {}: main alloc {:?}, FX alloc {:?}", room_name, self.main_allocator.get_stats(), self.fx_allocator.get_stats());
        let state_ptrs = get_room_state_ptrs(&self.rom, room_ptr)?;
        for (state_idx, &(_event_ptr, state_ptr)) in state_ptrs.iter().enumerate() {
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
                let fx_data = self.get_fx_data(&state_xml);
                fx_data_vec.push(fx_data);
                state_xml_vec.push(state_xml.clone());
            }

            // For a given room state, allocate enough space for the level data to fit whichever theme has the largest.
            // This approach can duplicate level data across room states, but we're not going to worry about that now.
            let max_level_data_size = compressed_level_data_vec.iter().map(|x| x.len()).max().unwrap_or(0);
            let level_data_addr = self.main_allocator.allocate(max_level_data_size)?;

            // Similarly, allocate enough space for FX data to fit whichever theme has the largest:
            let max_fx_data_size = fx_data_vec.iter().map(|x| x.len()).max().unwrap_or(0);
            let fx_data_addr = self.fx_allocator.allocate(max_fx_data_size)?;

            for (i, project) in project_names.iter().enumerate() {
                let mut new_rom = self.rom.clone();
                new_rom.enable_tracking();
                let state_xml = &state_xml_vec[i];

                // Write the tileset index
                new_rom.write_u8(state_ptr + 3, state_xml.gfx_set as isize)?;

                // Write (or clear) the BGData pointer:
                if state_xml.layer2_type == Layer2Type::BGData {
                    let bg_ptr = self.bgdata_map.get(&state_xml.bg_data).map(|x| *x).unwrap_or(0);
                    if bg_ptr == 0 {
                        error!("Unrecognized BGData in {}", project);
                    }
                    new_rom.write_u16(state_ptr + 22, bg_ptr)?;
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
                    // Skip for Mother Brain Room, which has special FX not in the FX list.
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
                let mut encoder = BPSEncoder::new(&self.source_suffix_tree, &new_rom.data, &modified_ranges);
                encoder.encode();
        
                // Save the BPS patch to a file:
                let output_filename = format!("{}-{:X}-{}.bps", project, room_ptr, state_idx);
                let output_path = self.output_patches_dir.join(output_filename);
                std::fs::write(&output_path, &encoder.patch_bytes)?;
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

    let rom = Rom::load(&args.input_rom)?;
    let mut room_ptr_map: HashMap<(usize, usize), usize> = HashMap::new();
    for room_ptr in room_ptrs {
        let area = rom.read_u8(room_ptr + 1)?;
        let idx = rom.read_u8(room_ptr)?;
        room_ptr_map.insert((area as usize, idx as usize), room_ptr);
    }
    info!("Building vanilla ROM suffix tree");
    let source_suffix_tree = SuffixTree::new(&rom.data);
    info!("Done building vanilla ROM suffix tree");

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
    };
    std::fs::create_dir_all(&mosaic_builder.tmp_dir)?;
    std::fs::create_dir_all(&mosaic_builder.compressed_data_cache_dir)?;
    std::fs::create_dir_all(&mosaic_builder.output_patches_dir)?;

    mosaic_builder.make_tileset_patch()?;
    mosaic_builder.build_bgdata_map()?;
    mosaic_builder.build_fx_door_map()?;
    mosaic_builder.make_all_room_patches()?;
    Ok(())
}
