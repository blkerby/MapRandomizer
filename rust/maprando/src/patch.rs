mod beam_doors_tiles;
pub mod bps;
pub mod compress;
pub mod decompress;
pub mod glowpatch_writer;
pub mod ips_write;
pub mod map_tiles;
pub mod suffix_tree;
pub mod title;

use std::path::Path;

use crate::{
    customize::vanilla_music::override_music,
    patch::map_tiles::diagonal_flip_tile,
    randomize::{LockedDoor, Randomization},
    settings::{
        AreaAssignment, ETankRefill, MotherBrainFight, Objective, ObjectiveScreen,
        RandomizerSettings, SaveAnimals, StartLocationMode, WallJump,
    },
};
use anyhow::{ensure, Context, Result};
use hashbrown::{HashMap, HashSet};
use ips;
use log::info;
use maprando_game::{
    DoorPtr, DoorPtrPair, DoorType, GameData, Item, Map, NodePtr, RoomGeometryDoor, RoomPtr,
};
use ndarray::Array3;
use rand::{Rng, SeedableRng};
use std::iter;
use strum::VariantNames;

use self::map_tiles::write_tile_4bpp;

pub const NUM_AREAS: usize = 6;

type PcAddr = usize; // PC pointer to ROM data
type AsmPtr = usize; // 16-bit SNES pointer to ASM code in bank 0x8F

pub fn snes2pc(addr: usize) -> usize {
    addr >> 1 & 0x3F8000 | addr & 0x7FFF
}

pub fn pc2snes(addr: usize) -> usize {
    addr << 1 & 0xFF0000 | addr & 0xFFFF | 0x808000
}

#[derive(Clone)]
pub struct Rom {
    pub data: Vec<u8>,
    track_touched: bool,
    touched: HashSet<usize>,
}

impl Rom {
    pub fn new(data: Vec<u8>) -> Self {
        Rom {
            data,
            track_touched: false,
            touched: HashSet::new(),
        }
    }

    pub fn enable_tracking(&mut self) {
        self.track_touched = true;
        self.touched.clear();
    }

    pub fn resize(&mut self, new_size: usize) {
        self.data.resize(new_size, 0xFF);
    }

    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .with_context(|| format!("Unable to load ROM at path {}", path.display()))?;
        Ok(Rom::new(data))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, &self.data)
            .with_context(|| format!("Unable to save ROM at path {}", path.display()))?;
        Ok(())
    }

    pub fn read_u8(&self, addr: usize) -> Result<isize> {
        ensure!(addr + 1 <= self.data.len(), "read_u8 address out of bounds");
        Ok(self.data[addr] as isize)
    }

    pub fn read_u16(&self, addr: usize) -> Result<isize> {
        ensure!(
            addr + 2 <= self.data.len(),
            "read_u16 address out of bounds"
        );
        let b0 = self.data[addr] as isize;
        let b1 = self.data[addr + 1] as isize;
        Ok(b0 | b1 << 8)
    }

    pub fn read_u24(&self, addr: usize) -> Result<isize> {
        ensure!(
            addr + 3 <= self.data.len(),
            "read_u24 address out of bounds"
        );
        let b0 = self.data[addr] as isize;
        let b1 = self.data[addr + 1] as isize;
        let b2 = self.data[addr + 2] as isize;
        Ok(b0 | b1 << 8 | b2 << 16)
    }

    pub fn read_n(&self, addr: usize, n: usize) -> Result<&[u8]> {
        ensure!(addr + n <= self.data.len(), "read_n address out of bounds");
        Ok(&self.data[addr..(addr + n)])
    }

    pub fn write_u8(&mut self, addr: usize, x: isize) -> Result<()> {
        ensure!(
            addr + 1 <= self.data.len(),
            "write_u8 address out of bounds"
        );
        ensure!(x >= 0 && x <= 0xFF, "write_u8 data does not fit");
        self.data[addr] = x as u8;
        if self.track_touched {
            self.touched.insert(addr);
        }
        Ok(())
    }

    pub fn write_u16(&mut self, addr: usize, x: isize) -> Result<()> {
        ensure!(
            addr + 2 <= self.data.len(),
            "write_u16 address out of bounds"
        );
        ensure!(x >= 0 && x <= 0xFFFF, "write_u16 data does not fit");
        self.write_u8(addr, x & 0xFF)?;
        self.write_u8(addr + 1, x >> 8)?;
        if self.track_touched {
            self.touched.insert(addr);
            self.touched.insert(addr + 1);
        }
        Ok(())
    }

    pub fn write_u24(&mut self, addr: usize, x: isize) -> Result<()> {
        ensure!(
            addr + 3 <= self.data.len(),
            "write_u24 address out of bounds"
        );
        ensure!(x >= 0 && x <= 0xFFFFFF, "write_u24 data does not fit");
        self.write_u8(addr, x & 0xFF)?;
        self.write_u8(addr + 1, (x >> 8) & 0xFF)?;
        self.write_u8(addr + 2, x >> 16)?;
        if self.track_touched {
            self.touched.insert(addr);
            self.touched.insert(addr + 1);
            self.touched.insert(addr + 2);
        }
        Ok(())
    }

    pub fn write_n(&mut self, addr: usize, x: &[u8]) -> Result<()> {
        ensure!(
            addr + x.len() <= self.data.len(),
            "write_n address out of bounds"
        );
        for i in 0..x.len() {
            self.write_u8(addr + i, x[i] as isize)?;
        }
        if self.track_touched {
            for i in 0..x.len() {
                self.touched.insert(addr + i);
            }
        }
        Ok(())
    }

    // Returns a list of [start, end) ranges.
    pub fn get_modified_ranges(&self) -> Vec<(usize, usize)> {
        let mut addresses: Vec<usize> = self.touched.iter().copied().collect();
        addresses.sort();
        let mut ranges: Vec<(usize, usize)> = vec![];

        let mut i = 0;
        'r: while i < addresses.len() {
            for j in i..addresses.len() - 1 {
                if addresses[j + 1] != addresses[j] + 1 {
                    ranges.push((addresses[i], addresses[j] + 1));
                    i = j + 1;
                    continue 'r;
                }
            }
            ranges.push((addresses[i], addresses[addresses.len() - 1] + 1));
            break;
        }
        assert!(ranges.iter().map(|x| x.1 - x.0).sum::<usize>() == addresses.len());
        ranges
    }
}

pub struct Patcher<'a> {
    pub orig_rom: &'a mut Rom,
    pub rom: &'a mut Rom,
    pub randomization: &'a Randomization,
    pub game_data: &'a GameData,
    pub map: &'a Map,
    pub other_door_ptr_pair_map: HashMap<DoorPtrPair, DoorPtrPair>,
    pub extra_setup_asm: HashMap<RoomPtr, Vec<u8>>,
    pub locked_door_state_indices: Vec<usize>,
    pub nothing_item_bitmask: [u8; 0x40],
    // per-area vec of (addr, bitmask) of cross-area tiles to reveal when map is activated:
    pub map_reveal_bitmasks: Vec<Vec<(u16, u16)>>,
}

pub fn xy_to_map_offset(x: isize, y: isize) -> isize {
    let y1 = y + 1;
    if x < 32 {
        (y1 * 32 + x) * 2
    } else {
        ((y1 + 32) * 32 + x - 32) * 2
    }
}

fn xy_to_explored_bit_ptr(x: isize, y: isize) -> (isize, u8) {
    let y1 = y + 1;
    let offset_in_bits = if x < 32 {
        y1 * 32 + x
    } else {
        (y1 + 32) * 32 + x - 32
    };
    let offset_byte_part = offset_in_bits / 8;
    let offset_bit_part = 7 - offset_in_bits % 8;
    let offset_bitmask = 1 << offset_bit_part;
    (offset_byte_part, offset_bitmask)
}

fn item_to_plm_type(item: Item, orig_plm_type: isize) -> isize {
    let item_id = item as isize;

    // Item container: 0 = none, 1 = chozo orb, 2 = shot block (scenery)
    let item_container = (orig_plm_type - 0xEED7) / 84;

    let plm_table: [[isize; 23]; 3] = [
        [
            0xEED7, // Energy tank
            0xEEDB, // Missile tank
            0xEEDF, // Super missile tank
            0xEEE3, // Power bomb tank
            0xEEE7, // Bombs
            0xEEEB, // Charge beam
            0xEEEF, // Ice beam
            0xEEF3, // Hi-jump
            0xEEF7, // Speed booster
            0xEEFB, // Wave beam
            0xEEFF, // Spazer beam
            0xEF03, // Spring ball
            0xEF07, // Varia suit
            0xEF0B, // Gravity suit
            0xEF0F, // X-ray scope
            0xEF13, // Plasma beam
            0xEF17, // Grapple beam
            0xEF1B, // Space jump
            0xEF1F, // Screw attack
            0xEF23, // Morph ball
            0xEF27, // Reserve tank
            0xF000, // Wall-jump boots
            0xEEDB, // Missile tank (nothing)
        ],
        [
            0xEF2B, // Energy tank, chozo orb
            0xEF2F, // Missile tank, chozo orb
            0xEF33, // Super missile tank, chozo orb
            0xEF37, // Power bomb tank, chozo orb
            0xEF3B, // Bombs, chozo orb
            0xEF3F, // Charge beam, chozo orb
            0xEF43, // Ice beam, chozo orb
            0xEF47, // Hi-jump, chozo orb
            0xEF4B, // Speed booster, chozo orb
            0xEF4F, // Wave beam, chozo orb
            0xEF53, // Spazer beam, chozo orb
            0xEF57, // Spring ball, chozo orb
            0xEF5B, // Varia suit, chozo orb
            0xEF5F, // Gravity suit, chozo orb
            0xEF63, // X-ray scope, chozo orb
            0xEF67, // Plasma beam, chozo orb
            0xEF6B, // Grapple beam, chozo orb
            0xEF6F, // Space jump, chozo orb
            0xEF73, // Screw attack, chozo orb
            0xEF77, // Morph ball, chozo orb
            0xEF7B, // Reserve tank, chozo orb
            0xF004, // Wall-jump boots, chozo orb
            0xEF2F, // Missile tank (nothing)
        ],
        [
            0xEF7F, // Energy tank, shot block
            0xEF83, // Missile tank, shot block
            0xEF87, // Super missile tank, shot block
            0xEF8B, // Power bomb tank, shot block
            0xEF8F, // Bombs, shot block
            0xEF93, // Charge beam, shot block
            0xEF97, // Ice beam, shot block
            0xEF9B, // Hi-jump, shot block
            0xEF9F, // Speed booster, shot block
            0xEFA3, // Wave beam, shot block
            0xEFA7, // Spazer beam, shot block
            0xEFAB, // Spring ball, shot block
            0xEFAF, // Varia suit, shot block
            0xEFB3, // Gravity suit, shot block
            0xEFB7, // X-ray scope, shot block
            0xEFBB, // Plasma beam, shot block
            0xEFBF, // Grapple beam, shot block
            0xEFC3, // Space jump, shot block
            0xEFC7, // Screw attack, shot block
            0xEFCB, // Morph ball, shot block
            0xEFCF, // Reserve tank, shot block
            0xF008, // Wall-jump boots, shot block
            0xEF83, // Missile tank (nothing)
        ],
    ];

    plm_table[item_container as usize][item_id as usize]
}

fn write_credits_big_letter(rom: &mut Rom, letter: char, addr: usize) -> Result<()> {
    if letter <= 'P' {
        rom.write_u16(addr, letter as isize - 'A' as isize + 0x0020)?;
        rom.write_u16(addr + 0x40, letter as isize - 'A' as isize + 0x0030)?;
    } else {
        rom.write_u16(addr, letter as isize - 'Q' as isize + 0x0040)?;
        rom.write_u16(addr + 0x40, letter as isize - 'Q' as isize + 0x0050)?;
    }
    Ok(())
}

fn write_credits_big_digit(rom: &mut Rom, digit: usize, addr: usize) -> Result<()> {
    rom.write_u16(addr, digit as isize + 0x0060)?;
    rom.write_u16(addr + 0x40, digit as isize + 0x0070)?;
    Ok(())
}

pub fn write_credits_big_char(rom: &mut Rom, c: char, addr: usize) -> Result<()> {
    if c >= '0' && c <= '9' {
        write_credits_big_digit(rom, c as usize - '0' as usize, addr)?;
    } else if c >= 'A' && c <= 'Z' {
        write_credits_big_letter(rom, c, addr)?;
    }
    Ok(())
}

// Returns list of (event_ptr, state_ptr):
pub fn get_room_state_ptrs(rom: &Rom, room_ptr: usize) -> Result<Vec<(usize, usize)>> {
    let mut pos = 11;
    let mut ptr_pairs: Vec<(usize, usize)> = Vec::new();
    loop {
        let ptr = rom.read_u16(room_ptr + pos)? as usize;
        if ptr == 0xE5E6 {
            // This is the standard state, which is the last one.
            ptr_pairs.push((ptr, room_ptr + pos + 2));
            return Ok(ptr_pairs);
        } else if ptr == 0xE612 || ptr == 0xE629 {
            // This is an event state.
            let state_ptr = 0x70000 + rom.read_u16(room_ptr + pos + 3)?;
            ptr_pairs.push((ptr, state_ptr as usize));
            pos += 5;
        } else {
            // This is another kind of state.
            let state_ptr = 0x70000 + rom.read_u16(room_ptr + pos + 2)?;
            ptr_pairs.push((ptr, state_ptr as usize));
            pos += 4;
        }
    }
}

pub fn apply_ips_patch(rom: &mut Rom, patch_path: &Path) -> Result<()> {
    let patch_data = std::fs::read(&patch_path)
        .with_context(|| format!("Unable to read patch {}", patch_path.display()))?;
    let patch = ips::Patch::parse(&patch_data)
        .with_context(|| format!("Unable to parse patch {}", patch_path.display()))?;
    for hunk in patch.hunks() {
        rom.write_n(hunk.offset(), hunk.payload())?;
    }
    Ok(())
}

fn apply_orig_ips_patches(rom: &mut Rom, settings: &RandomizerSettings) -> Result<()> {
    let patches_dir = Path::new("../patches/ips/");
    let mut patches: Vec<&'static str> = vec![
        "mb_barrier",
        "mb_barrier_clear",
        "mb_left_entrance",
        "gray_doors",
    ];
    patches.push("hud_expansion_opaque");

    match settings.objective_settings.objective_screen {
        ObjectiveScreen::Disabled => {}
        ObjectiveScreen::Enabled => {
            // For the pause menu tileset changes (for green checkmark, etc.)
            patches.push("pause_menu_objectives");
        }
    }

    for patch_name in patches {
        let patch_path = patches_dir.join(patch_name.to_string() + ".ips");
        apply_ips_patch(rom, &patch_path)?;
    }

    Ok(())
}

impl<'a> Patcher<'a> {
    fn apply_ips_patches(&mut self) -> Result<()> {
        self.rom.data.resize(0x400000, 0);
        let patches_dir = Path::new("../patches/ips/");
        let mut patches = vec![
            "complementary_suits",
            "disable_map_icons",
            "escape",
            "tourian_map",
            "no_explosions_before_escape",
            "sound_effect_disables",
            "title_map_animation",
            "shaktool",
            "fix_water_fx_bug",
            "seed_hash_display",
            "max_ammo_display_fast",
            "stats",
            "credits",
            "sram_check_disable",
            "map_area",
            "map_progress_maintain",
            "item_dots_disappear",
            "fast_reload",
            "saveload",
            "hazard_markers",
            "rng_fix",
            "intro_song",
            "msu1",
            "escape_timer",
            "nothing_item",
            "beam_doors",
            "horizontal_door_fix",
            "samus_tiles_optim_animated_tiles_fix",
        ];

        if self.randomization.settings.other_settings.ultra_low_qol {
            patches.extend(["ultra_low_qol_vanilla_bugfixes"]);
        } else {
            patches.extend([
                "vanilla_bugfixes",
                "itemsounds",
                "missile_refill_all",
                "decompression",
                "aim_anything",
                "fast_saves",
                "fast_mother_brain_cutscene",
                "fast_big_boy_cutscene",
                "fix_kraid_vomit",
                "fix_kraid_hud",
                "escape_autosave",
                "boss_exit",
                "oob_death",
                "vertical_door_fix",
                "spin_lock",
                "fix_transition_bad_tiles",
            ]);
        }

        patches.push("new_game");

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .all_items_spawn
        {
            patches.push("all_items_spawn");
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .escape_movement_items
            || self
                .randomization
                .settings
                .item_progression_settings
                .stop_item_placement_early
        {
            patches.push("escape_items");
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .fast_elevators
        {
            patches.push("elevators_speed");
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .fast_doors
        {
            patches.push("fast_doors");
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .fast_pause_menu
        {
            patches.push("fast_pause_menu");
            self.rom.write_u16(snes2pc(0x82fffc), 0x8000)?;
            self.rom.write_u16(snes2pc(0xdfff07), 0)?; // no artificial lag in unpause black screen
        } else {
            // With fast pause menu QoL disabled, use 6 artificial lag frames during unpause black screen,
            // to make it approximately align with vanilla, compensating for the optimized decompression
            // which makes it faster.
            self.rom.write_u16(snes2pc(0xdfff07), 6)?;
        }

        match self.randomization.settings.other_settings.wall_jump {
            WallJump::Vanilla => {}
            WallJump::Collectible => {
                patches.push("walljump_item");
            }
        }

        match self
            .randomization
            .settings
            .quality_of_life_settings
            .etank_refill
        {
            ETankRefill::Disabled => {
                patches.push("etank_refill_disabled");
            }
            ETankRefill::Vanilla => {}
            ETankRefill::Full => {
                patches.push("etank_refill_full");
            }
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .energy_station_reserves
        {
            patches.push("energy_station_reserves");
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .reserve_backward_transfer
        {
            patches.push("reserve_backward_fill");
        }

        if self
            .randomization
            .settings
            .other_settings
            .energy_free_shinesparks
        {
            patches.push("energy_free_shinesparks");
        }

        if self.randomization.settings.quality_of_life_settings.respin {
            patches.push("respin");
            // patches.push("spinjumprestart");
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .momentum_conservation
        {
            patches.push("momentum_conservation");
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .buffed_drops
        {
            patches.push("buffed_drops");
        }

        if self.randomization.settings.map_layout != "Vanilla" {
            patches.push("zebes_asleep_music");
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .mother_brain_fight
            == MotherBrainFight::Skip
        {
            patches.push("fix_hyper_slowlock");
        }

        match self
            .randomization
            .settings
            .objective_settings
            .objective_screen
        {
            ObjectiveScreen::Disabled => {}
            ObjectiveScreen::Enabled => {
                patches.push("pause_menu_objectives");
            }
        }

        for patch_name in patches {
            let patch_path = patches_dir.join(patch_name.to_string() + ".ips");
            apply_ips_patch(&mut self.rom, &patch_path)?;
        }

        // Write settings flags, e.g. for use by auto-tracking tools:
        // For now this is just to indicate if walljump-boots exists as an item.
        let mut settings_flag = 0x0000;
        if self.randomization.settings.other_settings.wall_jump == WallJump::Collectible {
            settings_flag |= 0x0001;
        }
        self.rom.write_u16(snes2pc(0xdfff05), settings_flag)?;

        Ok(())
    }

    fn place_items(&mut self) -> Result<()> {
        for (&item, &loc) in iter::zip(
            &self.randomization.item_placement,
            &self.game_data.item_locations,
        ) {
            let item_plm_ptr = self.game_data.node_ptr_map[&loc];
            let orig_plm_type = self.orig_rom.read_u16(item_plm_ptr)?;
            let new_plm_type = item_to_plm_type(item, orig_plm_type);
            self.rom.write_u16(item_plm_ptr, new_plm_type)?;
            if item == Item::Nothing {
                let idx = self.rom.read_u16(item_plm_ptr + 4).unwrap() as usize;
                self.nothing_item_bitmask[idx >> 3] |= 1 << (idx & 7);

                if loc == (19, 2) {
                    // Placing nothing item in the Bomb Torizo Room could be a problem because then there would be
                    // no way to activate the fight (which could be required on Chozos objective mode).
                    // So we put an invisible fake item there, using a special new PLM type.
                    self.rom.write_u16(item_plm_ptr, 0xF700)?;
                }
            }
        }
        Ok(())
    }

    fn write_one_door_data(
        &mut self,
        src_exit_ptr: usize,
        dst_entrance_ptr: usize,
        cross_area: bool,
        extra_door_asm_map: &HashMap<NodePtr, (AsmPtr, AsmPtr)>,
    ) -> Result<()> {
        let mut door_data = self.orig_rom.read_n(dst_entrance_ptr, 12)?.to_vec();
        // Trigger the map to reload if the door crosses areas:
        if cross_area {
            door_data[2] |= 0x40;
        } else {
            door_data[2] &= !0x40;
        }
        if let Some(&(new_asm, end_asm)) = extra_door_asm_map.get(&src_exit_ptr) {
            // Set extra custom ASM applicable to exiting from the given door exit:
            door_data[10..12].copy_from_slice(&((new_asm as u16).to_le_bytes()));

            // Patch the custom ASM to jump to original ASM for the given door entrance:
            let orig_asm = self.orig_rom.read_u16(dst_entrance_ptr + 10)?;
            if orig_asm == 0 {
                // There is no original ASM for this entrance, so just return:
                self.rom.write_u8(snes2pc(0x8F8000 | end_asm), 0x60)?; // RTS
            } else {
                let jmp_asm = vec![0x4C, (orig_asm & 0xFF) as u8, (orig_asm >> 8) as u8]; // JMP orig_asm
                self.rom.write_n(snes2pc(0x8F8000 | end_asm), &jmp_asm)?;
            }
        }
        self.rom.write_n(src_exit_ptr, &door_data)?;
        if src_exit_ptr == 0x1A798 {
            // Pants Room right door
            // Also write the same data to the East Pants Room right door
            self.rom.write_n(0x1A7BC, &door_data)?;
        }
        Ok(())
    }

    // Appends asm to mark the map tile (x, y) as explored and revealed. This is used for elevators and arrows.
    // (See `auto_explore_elevators` and `auto_explore_arrows` below for details.)
    fn add_explore_tile_asm(
        &mut self,
        current_area: usize,
        tile_area: usize,
        x: isize,
        y: isize,
        asm: &mut Vec<u8>,
        explore: bool,
    ) -> Result<()> {
        let (offset, bitmask) = xy_to_explored_bit_ptr(x, y);

        // Mark as revealed (which will persist after deaths/reloads):
        let addr = 0x2000 + (tile_area as isize) * 0x100 + offset;
        asm.extend([0xAF, (addr & 0xFF) as u8, (addr >> 8) as u8, 0x70]); // LDA $70:{addr}
        asm.extend([0x09, bitmask, 0x00]); // ORA #{bitmask}
        asm.extend([0x8F, (addr & 0xFF) as u8, (addr >> 8) as u8, 0x70]); // STA $70:{addr}

        // Mark as explored (for elevators. Not needed for area transition arrows/letters except in ultra-low QoL mode):
        if explore {
            if current_area == tile_area {
                // We want to write an explored bit to the current area's map, so we have to write it to
                // the temporary copy at 0x07F7 (otherwise it wouldn't take effect and would just be overwritten
                // on the next map reload).
                let addr = 0x07F7 + offset;
                asm.extend([0xAD, (addr & 0xFF) as u8, (addr >> 8) as u8]); // LDA {addr}
                asm.extend([0x09, bitmask, 0x00]); // ORA #{bitmask}
                asm.extend([0x8D, (addr & 0xFF) as u8, (addr >> 8) as u8]); // STA {addr}
            } else {
                // We want to write an explored bit to a different area's map, so we have to write it to
                // the main explored bits at 0x7ECD52 (which will get copied over to 0x07F7 on the map reload
                // when entering the different area).
                let addr = 0xCD52 + tile_area as isize * 0x100 + offset;
                asm.extend([0xAF, (addr & 0xFF) as u8, (addr >> 8) as u8, 0x7E]); // LDA $7E:{addr}
                asm.extend([0x09, bitmask, 0x00]); // ORA #{bitmask}
                asm.extend([0x8F, (addr & 0xFF) as u8, (addr >> 8) as u8, 0x7E]);
                // STA $7E:{addr}
            }
        }
        Ok(())
    }

    // Adds ASM to mark the tile (x, y) explored when entering or exiting through `door_ptr_pair`.
    // Here (x, y) is in room-local coordinates. This is used for elevators.
    fn add_double_explore_tile_asm(
        &mut self,
        door_ptr_pair: &DoorPtrPair,
        x: isize,
        y: isize,
        extra_door_asm: &mut HashMap<DoorPtr, Vec<u8>>,
        explore: bool,
    ) -> Result<()> {
        let (room_idx, _door_idx) =
            self.game_data.room_and_door_idxs_by_door_ptr_pair[door_ptr_pair];
        let room = &self.game_data.room_geometry[room_idx];
        let room_x = self.rom.read_u8(room.rom_address + 2)?;
        let room_y = self.rom.read_u8(room.rom_address + 3)?;
        let area = self.map.area[room_idx];
        let other_door_ptr_pair = self.other_door_ptr_pair_map[door_ptr_pair];
        let (other_room_idx, _other_door_idx) =
            self.game_data.room_and_door_idxs_by_door_ptr_pair[&other_door_ptr_pair];
        let other_area = self.map.area[other_room_idx];
        if let Some(ptr) = door_ptr_pair.0 {
            // ASM for when exiting through the given door:
            let asm = extra_door_asm.entry(ptr).or_default();
            self.add_explore_tile_asm(other_area, area, room_x + x, room_y + y, asm, explore)?;
        }
        if let Some(ptr) = other_door_ptr_pair.0 {
            // ASM for when entering through the given door:
            let asm = extra_door_asm.entry(ptr).or_default();
            self.add_explore_tile_asm(area, area, room_x + x, room_y + y, asm, explore)?;
        }
        Ok(())
    }

    fn add_map_reveal_tile(
        &mut self,
        door_ptr_pair: &DoorPtrPair,
        local_x: isize,
        local_y: isize,
    ) -> Result<()> {
        let (room_idx, _door_idx) =
            self.game_data.room_and_door_idxs_by_door_ptr_pair[door_ptr_pair];
        let room = &self.game_data.room_geometry[room_idx];
        let room_x = self.rom.read_u8(room.rom_address + 2)?;
        let room_y = self.rom.read_u8(room.rom_address + 3)?;
        let area = self.map.area[room_idx];
        let other_door_ptr_pair = self.other_door_ptr_pair_map[door_ptr_pair];
        let (other_room_idx, _other_door_idx) =
            self.game_data.room_and_door_idxs_by_door_ptr_pair[&other_door_ptr_pair];
        let other_area = self.map.area[other_room_idx];
        let x = room_x + local_x;
        let y = room_y + local_y;
        let (offset, bitmask) = xy_to_explored_bit_ptr(x as isize, y as isize);
        self.map_reveal_bitmasks[area]
            .push(((offset + area as isize * 0x100) as u16, bitmask as u16));
        if self
            .randomization
            .settings
            .quality_of_life_settings
            .opposite_area_revealed
        {
            self.map_reveal_bitmasks[other_area]
                .push(((offset + area as isize * 0x100) as u16, bitmask as u16));
        }
        Ok(())
    }

    // There are map tiles at the bottom of elevator rooms (at the top of elevators) that
    // Samus does not pass through and hence would never be marked explored. This is an issue
    // that exists in the vanilla game but would be more noticeable in Map Rando because
    // elevators are frequently connected within the same area, so it would leave a hole
    // in the map (if the area map is not yet acquired). We fix this by having the game
    // automatically mark these tiles as explored when using the elevator in either direction.
    fn auto_explore_elevators(
        &mut self,
        extra_door_asm: &mut HashMap<DoorPtr, Vec<u8>>,
    ) -> Result<()> {
        let mut add_explore =
            |exit_ptr: usize, entrance_ptr: usize, coords: Vec<(isize, isize)>| -> Result<()> {
                for &(x, y) in &coords {
                    let door_ptr_pair = (Some(exit_ptr), Some(entrance_ptr));
                    self.add_double_explore_tile_asm(&door_ptr_pair, x, y, extra_door_asm, true)?;
                }
                Ok(())
            };

        // Green Brinstar Elevator Room
        add_explore(0x18C0A, 0x18CA6, vec![(0, 1), (0, 2), (0, 3)])?;
        // Statues Room
        add_explore(0x19222, 0x1A990, vec![(0, 2), (0, 3), (0, 4)])?;
        // Forgotten Highway Elevator
        add_explore(0x18A5A, 0x1A594, vec![(0, 1), (0, 2), (0, 3)])?;
        // Red Brinstar Elevator Room
        add_explore(0x18B02, 0x190BA, vec![(0, 1), (0, 2), (0, 3)])?;
        // Blue Brinstar Elevator Room
        add_explore(0x18B9E, 0x18EB6, vec![(0, 1), (0, 2), (0, 3)])?;
        // Warehouse Entrance
        add_explore(0x19246, 0x192EE, vec![(0, 1)])?;
        // We skip Lower Norfair Elevator Room because there are no extra tiles to explore there.

        Ok(())
    }

    fn get_arrow_xy(&self, door: &RoomGeometryDoor) -> (isize, isize) {
        if door.direction == "left" {
            (door.x as isize - 1, door.y as isize)
        } else if door.direction == "right" {
            (door.x as isize + 1, door.y as isize)
        } else if door.direction == "up" {
            (door.x as isize, door.y as isize - 1)
        } else if door.direction == "down" {
            (door.x as isize, door.y as isize + 1)
        } else {
            panic!("Unrecognized door direction: {}", door.direction);
        }
    }

    // Similarly, for the arrow tiles that we add to mark area transitions, Samus doesn't
    // pass through these tiles, so normally they wouldn't show up until the map station is
    // obtained. We change this by having these tiles be automatically marked revealed
    // when passing through the area transition (it isn't necessary to mark them explored):
    fn auto_reveal_arrows(&mut self, extra_door_asm: &mut HashMap<DoorPtr, Vec<u8>>) -> Result<()> {
        for (src_pair, dst_pair, _bidirectional) in &self.map.doors {
            let (src_room_idx, src_door_idx) =
                self.game_data.room_and_door_idxs_by_door_ptr_pair[src_pair];
            let (dst_room_idx, dst_door_idx) =
                self.game_data.room_and_door_idxs_by_door_ptr_pair[dst_pair];
            let src_area = self.map.area[src_room_idx];
            let dst_area = self.map.area[dst_room_idx];
            if src_area == dst_area {
                continue;
            }
            let (src_x, src_y) =
                self.get_arrow_xy(&self.game_data.room_geometry[src_room_idx].doors[src_door_idx]);
            let (dst_x, dst_y) =
                self.get_arrow_xy(&self.game_data.room_geometry[dst_room_idx].doors[dst_door_idx]);
            self.add_double_explore_tile_asm(src_pair, src_x, src_y, extra_door_asm, false)?;
            self.add_double_explore_tile_asm(dst_pair, dst_x, dst_y, extra_door_asm, false)?;
            self.add_map_reveal_tile(src_pair, src_x, src_y)?;
            self.add_map_reveal_tile(dst_pair, dst_x, dst_y)?;
        }
        Ok(())
    }

    fn block_escape_return(
        &mut self,
        extra_door_asm: &mut HashMap<DoorPtr, Vec<u8>>,
    ) -> Result<()> {
        // For testing, using Landing Site bottom left door
        // let mother_brain_left_door_pair = (Some(0x18916), Some(0x1896A));
        let mother_brain_left_door_pair = (Some(0x1AA8C), Some(0x1AAE0));

        // Finding the matching door on the map:
        let mut other_door_pair = (None, None);
        for door in &self.randomization.map.doors {
            if door.1 == mother_brain_left_door_pair {
                other_door_pair = door.0;
                break;
            }
        }

        // Get x & y position of door (which we will turn gray during escape).
        let screen_x = self.orig_rom.read_u8(other_door_pair.1.unwrap() + 6)? as u8;
        let screen_y = self.orig_rom.read_u8(other_door_pair.1.unwrap() + 7)? as u8;
        let entrance_x = screen_x * 16 + 14;
        let entrance_y = screen_y * 16 + 6;

        let asm: Vec<u8> = vec![
            0xA9, 0x0E, 0x00, // LDA #$000E   (Escape flag)
            0x22, 0x33, 0x82, 0x80, // JSL $808233  (Check if flag is set)
            0x90, 0x0A, // BCC $0A  (Skip spawning gray door if not in escape)
            0x22, 0x80, 0xF3, 0x84, // JSL $84F380  (Spawn hard-coded PLM with room argument)
            entrance_x, entrance_y, 0x42, 0xC8, 0x00,
            0x10, // PLM type 0xC8CA (gray door), argument 0x1000 (always closed)
        ];

        extra_door_asm
            .entry(mother_brain_left_door_pair.0.unwrap())
            .or_default()
            .extend(asm);

        Ok(())
    }

    fn clamp_samus_position(
        &mut self,
        extra_door_asm: &mut HashMap<DoorPtr, Vec<u8>>,
    ) -> Result<()> {
        let sand_entrances = vec![
            // (door_pair, min_position_x, max_position_x)
            ((Some(0x1A6C0), Some(0x1A6FC)), 0x65, 0x9B), // East Sand Hole
            ((Some(0x1A6A8), Some(0x1A6E4)), 0x165, 0x19B), // West Sand Hole
            ((Some(0x1A654), Some(0x1A6B4)), 0x265, 0x29B), // West Sand Hall
            ((Some(0x1A69C), Some(0x1A6CC)), 0x165, 0x19B), // East Sand Hall
            ((None, Some(0x1A624)), 0x45, 0xBB),          // Plasma Beach Quicksand Room
            ((None, Some(0x1A8A0)), 0x65, 0x9B),          // Butterfly Room
            ((None, Some(0x1A864)), 0x85, 0xDB),          // Botwoon Quicksand Room (left)
            ((None, Some(0x1A858)), 0x125, 0x19B),        // Botwoon Quicksand Room (right)
            ((None, Some(0x1A8AC)), 0x265, 0x2BB),        // Below Botwoon Energy Tank (left)
            ((None, Some(0x1A8B8)), 0x345, 0x3BB),        // Below Botwoon Energy Tank (right)
        ];
        for (door_pair, min_position, max_position) in sand_entrances {
            let other_door_pair = self.other_door_ptr_pair_map[&door_pair];

            // Note: we don't bother with adjusting subpixels.
            let asm = vec![
                // Check if Samus X position is less than min_position, and if so set it to min_position:
                0xA9,
                (min_position & 0xFF) as u8,
                (min_position >> 8) as u8, // LDA #min_position
                0xCD,
                0xF6,
                0x0A, // CMP $0AF6
                0x90,
                0x06, // BCC .no_clamp_min
                0x8D,
                0xF6,
                0x0A, // STA $0AF6
                0x8D,
                0x10,
                0x0B, // STA $0B10  ; also set samus previous X position (to prevent camera glitching)
                // .no_clamp_min:

                // Check if Samus X position is greater than max_position, and if so set it to max_position:
                0xA9,
                (max_position & 0xFF) as u8,
                (max_position >> 8) as u8, // LDA #max_position
                0xCD,
                0xF6,
                0x0A, // CMP $0AF6
                0xB0,
                0x06, // BCS .no_clamp_max
                0x8D,
                0xF6,
                0x0A, // STA $0AF6
                0x8D,
                0x10,
                0x0B, // STA $0B10  ; also set samus previous X position (to prevent camera glitching)
                      // .no_clamp_max:
            ];

            extra_door_asm
                .entry(other_door_pair.0.unwrap())
                .or_default()
                .extend(asm);
        }
        Ok(())
    }

    // Returns map from door data PC address to 1) new custom door ASM pointer, 2) end of custom door ASM
    // where an RTS or JMP instruction must be added (based on the connecting door).
    fn prepare_extra_door_asm(&mut self) -> Result<HashMap<DoorPtr, (AsmPtr, AsmPtr)>> {
        let toilet_exit_asm: Vec<u8> = vec![0x20, 0x01, 0xE3]; // JSR 0xE301
        let boss_exit_asm: Vec<u8> = vec![0x20, 0xF0, 0xF7]; // JSR 0xF7F0
        let mut extra_door_asm: HashMap<DoorPtr, Vec<u8>> = HashMap::new();
        extra_door_asm.insert(0x1A600, toilet_exit_asm.clone()); // Aqueduct toilet door down
        extra_door_asm.insert(0x1A60C, toilet_exit_asm.clone()); // Aqueduct toilet door up
        if !self.randomization.settings.other_settings.ultra_low_qol {
            extra_door_asm.insert(0x191CE, boss_exit_asm.clone()); // Kraid left exit
            extra_door_asm.insert(0x191DA, boss_exit_asm.clone()); // Kraid right exit
            extra_door_asm.insert(0x1A96C, boss_exit_asm.clone()); // Draygon left exit
            extra_door_asm.insert(0x1A978, boss_exit_asm.clone()); // Draygon right exit
            extra_door_asm.insert(0x193DE, boss_exit_asm.clone()); // Crocomire left exit
            extra_door_asm.insert(0x193EA, boss_exit_asm.clone()); // Crocomire top exit
            extra_door_asm.insert(0x1A2C4, boss_exit_asm.clone()); // Phantoon exit
        }
        self.auto_explore_elevators(&mut extra_door_asm)?;
        self.auto_reveal_arrows(&mut extra_door_asm)?;
        self.block_escape_return(&mut extra_door_asm)?;
        self.clamp_samus_position(&mut extra_door_asm)?;

        let mut door_asm_free_space = 0xEE10; // in bank 0x8F
        let mut extra_door_asm_map: HashMap<DoorPtr, (AsmPtr, AsmPtr)> = HashMap::new();
        for (&door_ptr, asm) in &extra_door_asm {
            extra_door_asm_map.insert(
                door_ptr,
                (door_asm_free_space, door_asm_free_space + asm.len()),
            );
            self.rom
                .write_n(snes2pc(0x8F8000 | door_asm_free_space), asm)?;
            // Reserve 3 bytes for the JMP instruction to the original ASM (if applicable, or RTS otherwise):
            door_asm_free_space += asm.len() + 3;
        }
        assert!(door_asm_free_space <= 0xF600);
        Ok(extra_door_asm_map)
    }

    fn write_door_data(&mut self) -> Result<()> {
        let extra_door_asm_map = self.prepare_extra_door_asm()?;
        for &((src_exit_ptr, src_entrance_ptr), (dst_exit_ptr, dst_entrance_ptr), _bidirectional) in
            &self.randomization.map.doors
        {
            let (src_room_idx, _) = self.game_data.room_and_door_idxs_by_door_ptr_pair
                [&(src_exit_ptr, src_entrance_ptr)];
            let (dst_room_idx, _) = self.game_data.room_and_door_idxs_by_door_ptr_pair
                [&(dst_exit_ptr, dst_entrance_ptr)];
            let src_area = self.map.area[src_room_idx];
            let dst_area = self.map.area[dst_room_idx];
            let cross_area = src_area != dst_area;

            if src_exit_ptr.is_some() && dst_entrance_ptr.is_some() {
                self.write_one_door_data(
                    src_exit_ptr.unwrap(),
                    dst_entrance_ptr.unwrap(),
                    cross_area,
                    &extra_door_asm_map,
                )?;
            }
            if dst_exit_ptr.is_some() && src_entrance_ptr.is_some() {
                self.write_one_door_data(
                    dst_exit_ptr.unwrap(),
                    src_entrance_ptr.unwrap(),
                    cross_area,
                    &extra_door_asm_map,
                )?;
            }
        }
        Ok(())
    }

    fn write_map_reveal_tiles(&mut self) -> Result<()> {
        // Write data to indicate cross-area tiles that should be revealed when a map station is activated.
        // This is used to mark area transition arrows/letters in the other area's map.
        let table_ptr = 0x90FA00;
        let mut ptr = 0x90FA10;
        for area in 0..NUM_AREAS {
            self.rom
                .write_u16(snes2pc(table_ptr + area * 2), ptr & 0xFFFF)?;
            for &(offset, bitmask) in &self.map_reveal_bitmasks[area] {
                self.rom.write_u16(snes2pc(ptr as usize), offset as isize)?;
                self.rom
                    .write_u16(snes2pc(ptr as usize + 2), bitmask as isize)?;
                ptr += 4;
            }
            // Write terminator:
            self.rom.write_u16(snes2pc(ptr as usize), 0)?;
            self.rom.write_u16(snes2pc(ptr as usize + 2), 0)?;
            ptr += 2
        }
        assert!(ptr <= 0x90FC00);
        Ok(())
    }

    fn fix_save_stations(&mut self) -> Result<()> {
        let save_station_ptrs = vec![
            0x44C5, 0x44D3, 0x44E1, 0x45CF, 0x45DD, 0x45EB, 0x45F9, 0x4607, 0x46D9, 0x46E7, 0x46F5,
            0x4703, 0x4711, 0x471F, 0x481B, 0x4917, 0x4925, 0x4933, 0x4941, 0x4A2F, 0x4A3D,
        ];

        let mut orig_door_map: HashMap<NodePtr, NodePtr> = HashMap::new();
        let mut new_door_map: HashMap<NodePtr, NodePtr> = HashMap::new();
        for &((src_exit_ptr, src_entrance_ptr), (dst_exit_ptr, dst_entrance_ptr), _bidirectional) in
            &self.randomization.map.doors
        {
            if src_exit_ptr.is_some() && src_entrance_ptr.is_some() {
                orig_door_map.insert(src_exit_ptr.unwrap(), src_entrance_ptr.unwrap());
            }
            if dst_exit_ptr.is_some() && dst_entrance_ptr.is_some() {
                orig_door_map.insert(dst_exit_ptr.unwrap(), dst_entrance_ptr.unwrap());
            }
            if src_exit_ptr.is_some() && dst_exit_ptr.is_some() {
                new_door_map.insert(src_exit_ptr.unwrap(), dst_exit_ptr.unwrap());
                new_door_map.insert(dst_exit_ptr.unwrap(), src_exit_ptr.unwrap());
            }
        }

        for ptr in save_station_ptrs {
            let orig_entrance_door_ptr = (self.rom.read_u16(ptr + 2)? + 0x10000) as NodePtr;
            let exit_door_ptr = orig_door_map[&orig_entrance_door_ptr];
            let entrance_door_ptr = new_door_map[&exit_door_ptr];
            self.rom
                .write_u16(ptr + 2, (entrance_door_ptr & 0xFFFF) as isize)?;
        }
        Ok(())
    }

    fn make_map_revealed(&mut self) -> Result<()> {
        // Zero out map revealed bits. These will be filled in later to identify area-transition markers,
        // which are tiles that need to be only fully revealed (not partial) since otherwise they would
        // have wrong colors.
        for i in snes2pc(0x829727)..snes2pc(0x829D27) {
            self.rom.write_u8(i, 0x00)?;
        }
        Ok(())
    }

    fn fix_twin_rooms(&mut self) -> Result<()> {
        // East Pants Room:
        let pants_room_x = self.rom.read_u8(0x7D646 + 2)?;
        let pants_room_y = self.rom.read_u8(0x7D646 + 3)?;
        self.rom.write_u8(0x7D69A + 2, pants_room_x + 1)?;
        self.rom.write_u8(0x7D69A + 3, pants_room_y + 1)?;
        // Homing Geemer Room:
        let west_ocean_x = self.rom.read_u8(0x793FE + 2)?;
        let west_ocean_y = self.rom.read_u8(0x793FE + 3)?;
        self.rom.write_u8(0x7968F + 2, west_ocean_x + 5)?;
        self.rom.write_u8(0x7968F + 3, west_ocean_y + 2)?;
        Ok(())
    }

    fn write_map_areas(&mut self) -> Result<()> {
        let mut room_index_area_hashmaps: Vec<HashMap<usize, usize>> =
            vec![HashMap::new(); NUM_AREAS];
        for (i, room) in self.game_data.room_geometry.iter().enumerate() {
            let room_index = self.orig_rom.read_u8(room.rom_address)? as usize;
            let orig_room_area = self.orig_rom.read_u8(room.rom_address + 1)? as usize;
            assert!(!room_index_area_hashmaps[orig_room_area].contains_key(&room_index));
            let new_area = self.map.area[i];
            room_index_area_hashmaps[orig_room_area].insert(room_index, new_area);
        }

        // Handle twin rooms:
        let pants_room_idx = self.game_data.room_idx_by_name["Pants Room"];
        room_index_area_hashmaps[4].insert(0x25, self.map.area[pants_room_idx]); // Set East Pants Room to same area as Pants Room
        let west_ocean_room_idx = self.game_data.room_idx_by_name["West Ocean"];
        room_index_area_hashmaps[0].insert(0x11, self.map.area[west_ocean_room_idx]); // Set Homing Geemer Room to same area as West Ocean

        // Write the information about each room's map area to some free space in bank 0x8F
        // which will be read by the `map_area` patch.
        let area_data_base_ptr = snes2pc(0x8FE99B);
        let mut area_data_ptr_pc = area_data_base_ptr + 2 * NUM_AREAS;
        for area in 0..NUM_AREAS {
            // Write pointer to the start of the table for the given area:
            let area_data_ptr_snes = (area_data_ptr_pc & 0x7FFF) | 0x8000;
            self.rom
                .write_u16(area_data_base_ptr + 2 * area, area_data_ptr_snes as isize)?;

            // Write the table contents:
            for (&room_index, &new_area) in &room_index_area_hashmaps[area] {
                self.rom
                    .write_u8(area_data_ptr_pc + room_index, new_area as isize)?;
            }

            // Advance the pointer keeping track of the next available free space:
            area_data_ptr_pc += room_index_area_hashmaps[area].keys().max().unwrap() + 1;
        }
        assert!(area_data_ptr_pc <= snes2pc(0x8FEB00));
        Ok(())
    }

    fn apply_map_tile_patches(&mut self) -> Result<()> {
        map_tiles::MapPatcher::new(
            &mut self.rom,
            self.game_data,
            self.map,
            self.randomization,
            &self.locked_door_state_indices,
        )
        .apply_patches()?;
        Ok(())
    }

    fn remove_non_blue_doors(&mut self) -> Result<()> {
        let plm_types_to_remove = vec![
            0xC88A, 0xC85A, 0xC872, // right pink/yellow/green door
            0xC890, 0xC860, 0xC878, // left pink/yellow/green door
            0xC896, 0xC866, 0xC87E, // down pink/yellow/green door
            0xC89C, 0xC86C, 0xC884, // up pink/yellow/green door
            0xDB48, 0xDB4C, 0xDB52, 0xDB56, 0xDB5A, 0xDB60, // eye doors
            0xC8CA, // wall in Escape Room 1
        ];
        let gray_door_plm_types: HashMap<isize, isize> = vec![
            (0xC848, 0xBAF4), // left gray door
            (0xC842, 0xFA00), // right gray door
            (0xC854, 0xFA06), // up gray door
            (0xC84E, 0xFA0C), // down gray door
        ]
        .into_iter()
        .collect();
        let keep_gray_door_room_names: Vec<String> = vec![
            "Bomb Torizo Room",
            "Kraid Room",
            "Phantoon's Room",
            "Draygon's Room",
            "Ridley's Room",
            "Golden Torizo's Room",
            "Botwoon's Room",
            "Spore Spawn Room",
            "Crocomire's Room",
            "Baby Kraid Room",
            "Plasma Room",
            "Metal Pirates Room",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        for room in &self.game_data.room_geometry {
            let event_state_ptrs = get_room_state_ptrs(&self.rom, room.rom_address)?;
            for &(event_ptr, state_ptr) in &event_state_ptrs {
                let plm_set_ptr = self.rom.read_u16(state_ptr + 20)? as usize;
                let mut ptr = plm_set_ptr + 0x70000;
                loop {
                    let plm_type = self.rom.read_u16(ptr)?;
                    if plm_type == 0 {
                        break;
                    }
                    let room_keep_gray_door = keep_gray_door_room_names.contains(&room.name)
                        || (room.name == "Pit Room" && event_ptr == 0xE652);
                    let is_removable_grey_door =
                        gray_door_plm_types.contains_key(&plm_type) && !room_keep_gray_door;
                    if plm_types_to_remove.contains(&plm_type) || is_removable_grey_door {
                        self.rom.write_u16(ptr, 0xB63F)?; // left continuation arrow (should have no effect, giving a blue door)
                        self.rom.write_u16(ptr + 2, 0)?; // position = (0, 0)
                    } else if gray_door_plm_types.contains_key(&plm_type) {
                        let new_type = gray_door_plm_types[&plm_type];

                        // Don't replace the gray doors in BT Room. In particular we want to leave its escape-state gray door
                        // vanilla (so that it closes immediately and doesn't unlock until animals are saved).
                        if room.name != "Bomb Torizo Room" {
                            self.rom.write_u16(ptr, new_type)?;
                        }
                    }
                    ptr += 6;
                }
            }
        }
        Ok(())
    }

    fn use_area_based_music(&mut self) -> Result<()> {
        // Start by applying vanilla music (since even for tracks that we don't change below, we need
        // to make sure they apply concrete tracks instead of "no change" like the vanilla game has):
        override_music(&mut self.rom)?;

        let area_music: [[u16; 2]; NUM_AREAS] = [
            [
                0x050C, // Return to Crateria (ASM can replace with intro track or storm track)
                0x0509, // Crateria Space Pirates (ASM can replace with zebes asleep track, with or without storm)
            ],
            [
                0x050F, // Green Brinstar
                0x0512, // Red Brinstar
            ],
            [
                0x0515, // Upper Norfair
                0x0518, // Lower Norfair
            ],
            [
                0x0530, // Wrecked Ship (power off)
                0x0530, // Wrecked Ship (power off)  (power on version, 0x0630, to be used by ASM)
            ],
            [
                0x061B, // Outer Maridia
                0x051B, // Inner Maridia
            ],
            [
                0x0609, // Tourian Entrance (Statues Room)
                0x051E, // Tourian Main
            ],
        ];

        let songs_to_keep: Vec<u16> = vec![
            // Elevator (item room music):
            0x0300, // Elevator
            0x0309, // Space Pirate Elevator
            0x0312, // Lower Brinstar Elevator
            0x0324, // Golden Torizo incoming fight
            0x0330, // Bowling Alley
            // Bosses:
            0x052A, // Miniboss Fight (Spore Spawn, Botwoon)
            // 0x0627,  // Boss Fight (Phantoon, Kraid), also Baby Kraid room.
            0x0527, // Boss Fight (?)
            0x0424, // Boss Fight (Ridley)
            0x0524, // Boss Fight (Draygon)
        ];

        let rooms_to_leave_unchanged = [
            "Mother Brain Room",
            "Big Boy Room",
            "Kraid Room",
            "Phantoon's Room",
        ]
        .map(|x| x.to_string());
        for (room_idx, room) in self.game_data.room_geometry.iter().enumerate() {
            if rooms_to_leave_unchanged.contains(&room.name) {
                continue;
            }
            let area = self.map.area[room_idx];
            let subarea = self.map.subarea[room_idx];
            let event_state_ptrs = get_room_state_ptrs(&self.rom, room.rom_address)?;
            for &(_event_ptr, state_ptr) in &event_state_ptrs {
                let song = self.rom.read_u16(state_ptr + 4)? as u16;
                if songs_to_keep.contains(&song) && room.name != "Golden Torizo Energy Recharge" {
                    // In vanilla, Golden Torizo Energy Recharge plays the item/elevator music,
                    // but that only seems to be because of it being next to Screw Attack Room.
                    // We want it to behave like the other Refill rooms and use area-themed music.
                    continue;
                }
                let new_song = area_music[area][subarea];
                self.rom.write_u16(state_ptr + 4, new_song as isize)?;
                if room.name == "Pants Room" {
                    // Set music for East Pants Room:
                    self.rom
                        .write_u16(snes2pc(0x8FD6A7 + 4), new_song as isize)?;
                } else if room.name == "West Ocean" {
                    // Set music for Homing Geemer Room:
                    self.rom
                        .write_u16(snes2pc(0x8F969C + 4), new_song as isize)?;
                }
            }
        }
        Ok(())
    }

    fn setup_door_specific_fx(&mut self) -> Result<()> {
        // Set up door-specific FX (which otherwise would get broken by the changes in door connections):
        let mut door_fx: HashMap<DoorPtrPair, usize> = HashMap::new();
        door_fx.insert((Some(0x19732), Some(0x1929A)), 0x8386D0); // Rising Tide left door: lava rising
        door_fx.insert((Some(0x1965A), Some(0x19672)), 0x838650); // Volcano Room left door: lava rising
        door_fx.insert((Some(0x195B2), Some(0x195BE)), 0x8385E0); // Speed Booster Hall right door: lava rising when Speed Booster collected
        door_fx.insert((Some(0x1983A), Some(0x19876)), 0x83876A); // Acid Statue Room bottom-right door: acid lowered
        door_fx.insert((Some(0x199A2), Some(0x199F6)), 0x83883C); // Amphitheatre right door: acid raised
        door_fx.insert((Some(0x18B6E), Some(0x1AB34)), 0x838060); // Climb bottom-left door: lava rising

        for (door1, door2, _) in &self.map.doors {
            if door_fx.contains_key(door1) {
                self.rom.write_u16(
                    snes2pc(door_fx[door1]),
                    (door2.0.unwrap() & 0xffff) as isize,
                )?;
            }
            if door_fx.contains_key(door2) {
                self.rom.write_u16(
                    snes2pc(door_fx[door2]),
                    (door1.0.unwrap() & 0xffff) as isize,
                )?;
            }
        }

        Ok(())
    }

    fn apply_mother_brain_fight_patches(&mut self) -> Result<()> {
        if self
            .randomization
            .settings
            .quality_of_life_settings
            .supers_double
        {
            // Make Supers do double damage to Mother Brain:
            self.rom.write_u8(snes2pc(0xB4F1D5), 0x84)?;
        }

        match self
            .randomization
            .settings
            .quality_of_life_settings
            .mother_brain_fight
        {
            MotherBrainFight::Vanilla => {
                // See fast_mother_brain_fight.asm patch for baseline changes to speed up cutscenes.
            }
            MotherBrainFight::Short => {
                // Make Mother Brain 1 finish faster:
                for addr in &[0x897D, 0x89AF, 0x89E1, 0x8A09, 0x8A31, 0x8A63, 0x8A95] {
                    self.rom.write_u16(snes2pc(0xA90000 + addr), 0x10)?; // cut delay in half for tubes to fall
                }

                // Skip the slow movement to the right when MB2 is preparing to finish Samus off
                self.rom.write_n(snes2pc(0xA9BB24), &[0xEA; 3])?;

                // After finishing Samus off (down to 100 energy or lower), skip to Mother Brain exploding:
                self.rom.write_u16(snes2pc(0xA9BD9E), 0xAEE1)?;

                self.rom.write_n(
                    snes2pc(0xA9AEFD),
                    &[
                        // (skip part where mother brain stumbles backwards before death; instead get hyper beam)
                        0xA9, 0x03, 0x00, // LDA #$0003
                        0x22, 0xAD, 0xE4, 0x91, // JSL $91E4AD
                        0xA9, 0x21, 0xAF, // LDA #$AF21             ;\
                        0x8D, 0xA8,
                        0x0F, // STA $0FA8  [$7E:0FA8]  ;} Mother Brain's body function = $AF21
                        0x60, // RTS
                    ],
                )?;

                // Silence the music and make Samus stand up when Mother Brain starts to fade to corpse
                self.rom.write_n(
                    snes2pc(0xA9B1BE),
                    &[0x20, 0x00, 0xFD], // JSR 0xFD00  (must match address in fast_mother_brain_cutscene.asm)
                )?;

                if self
                    .randomization
                    .settings
                    .quality_of_life_settings
                    .escape_movement_items
                    || self
                        .randomization
                        .settings
                        .item_progression_settings
                        .stop_item_placement_early
                {
                    // 0xA9FB70: new hyper beam collect routine in escape_items.asm.
                    self.rom.write_u24(snes2pc(0xA9AF01), 0xA9FB70)?;
                }
            }
            MotherBrainFight::Skip => {
                // Make Mother Brain 1 finish faster:
                for addr in &[0x897D, 0x89AF, 0x89E1, 0x8A09, 0x8A31, 0x8A63, 0x8A95] {
                    self.rom.write_u16(snes2pc(0xA90000 + addr), 0x10)?; // cut delay in half for tubes to fall
                }

                // Skip MB2 and MB3:
                self.rom.write_u16(snes2pc(0xA98D80), 0xAEE1)?;
                self.rom.write_n(
                    snes2pc(0xA9AEFD),
                    &[
                        // (skip part where mother brain stumbles backwards before death; instead get hyper beam)
                        0xA9, 0x03, 0x00, // LDA #$0003
                        0x22, 0xAD, 0xE4, 0x91, // JSL $91E4AD
                        0xEA, 0xEA, // nop : nop
                    ],
                )?;
                self.rom.write_u16(snes2pc(0xA9AF07), 0xB115)?; // skip MB moving forward, drooling, exploding
                self.rom.write_u16(snes2pc(0xA9B19F), 1)?; // accelerate fade to gray (which wouldn't have an effect here except for a delay)

                if self
                    .randomization
                    .settings
                    .quality_of_life_settings
                    .escape_movement_items
                {
                    // 0xA9FB70: new hyper beam collect routine in escape_items.asm.
                    self.rom.write_u24(snes2pc(0xA9AF01), 0xA9FB70)?;
                }
            }
        }

        Ok(())
    }

    fn apply_miscellaneous_patches(&mut self) -> Result<()> {
        // Copy the item at Morph Ball to the Zebes-awake state (so it doesn't become unobtainable after Zebes is awake).
        // For this we overwrite the PLM slot for the gray door at the left of the room (which we would get rid of anyway).
        let plm_data = self.rom.read_n(0x786DE, 6)?.to_vec();
        self.rom.write_n(0x78746, &plm_data)?;

        // Disable demo (by overwriting the branch on the timer reaching zero):
        self.rom.write_n(snes2pc(0x8B9F2C), &vec![0x80, 0x0A])?; // BRA $0A

        // In Kraid's room, no longer restrict Samus X position to left screen:
        self.rom.write_u8(snes2pc(0xA7C9EE), 0x60)?; // RTS

        // In Shaktool room, skip setting screens to red scroll (so that it won't glitch out when entering from the right):
        self.rom.write_u8(snes2pc(0x84B8DC), 0x60)?; // RTS

        // Restore acid in Tourian Escape Room 4:
        self.rom.write_u16(snes2pc(0x8FDF03), 0xC953)?; // Vanilla setup ASM pointer (to undo effect of `no_explosions_before_escape` patch)
        self.rom.write_u8(snes2pc(0x8FC95B), 0x60)?; // RTS (return early from setup ASM to skip setting up shaking)

        // Remove fake gray door that gets drawn in Phantoon's Room:
        self.rom.write_n(snes2pc(0xA7D4E5), &vec![0xEA; 8])?;

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .all_items_spawn
        {
            // Copy the item in Pit Room to the Zebes-asleep state.
            // For this we overwrite the PLM slot for the gray door at the left of the room (which we would get rid of anyway).
            let plm_data = self.rom.read_n(0x783EE, 6)?.to_vec();
            self.rom.write_n(0x783C8, &plm_data)?;
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .acid_chozo
        {
            // Remove Space Jump check
            self.rom.write_n(snes2pc(0x84D195), &[0xEA, 0xEA])?; // NOP : NOP
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .remove_climb_lava
        {
            // Replace the Tourian Escape Room 4 door with the value 0xFFFF which does not match any door,
            // effectively disabling the door-specific FX for this room.
            self.rom.write_u16(snes2pc(0x838060), 0xFFFF)?;
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .infinite_space_jump
        {
            // self.rom.write_n(0x82493, &[0x80, 0x0D])?;  // BRA $0D  (Infinite Space Jump)
            // self.rom.write_n(snes2pc(0x90A493), &[0xEA, 0xEA])?; // NOP : NOP  (old Lenient Space Jump)

            // Lenient Space Jump: Remove check on maximum Y speed for Space Jump to trigger:
            self.rom.write_n(snes2pc(0x90A4A0), &[0xEA, 0xEA])?; // NOP : NOP
            self.rom.write_n(snes2pc(0x90A4AF), &[0xEA, 0xEA])?; // NOP : NOP
        }

        if !self.randomization.settings.other_settings.ultra_low_qol {
            // In Crocomire's initialization, skip setting the leftmost screens to red scroll. Even in the vanilla game there
            // is no purpose to this, as they are already red. But it important to skip here in the rando, because when entering
            // from the left door with Crocomire still alive, these scrolls are set to blue by the door ASM, and if they
            // were overridden with red it would break the graphics.
            self.rom.write_n(snes2pc(0xA48A92), &vec![0xEA; 4])?; // NOP:NOP:NOP:NOP

            // Release Spore Spawn camera so it won't be glitched when entering from the right:
            self.rom.write_n(snes2pc(0xA5EADA), &vec![0xEA; 3])?; // NOP:NOP:NOP

            // Likewise release Kraid camera so it won't be as glitched when entering from the right:
            self.rom.write_n(snes2pc(0xA7A9F4), &vec![0xEA; 4])?; // NOP:NOP:NOP:NOP

            // Adjust the door cap location for the Green Brinstar Main Shaft door to itself left-to-right:
            // In vanilla it spawns a screen to the left of where it "should". We keep it wrong, to retain
            // the behavior of the door appearing immediately closed, but move the spawn location to be in an
            // out-of-the-way off-camera location, in the top-right of the room.
            self.rom.write_u8(snes2pc(0x838CF2), 0x21)?;
            self.rom.write_u8(snes2pc(0x838CF3), 0x06)?;
        }

        match self.randomization.save_animals {
            SaveAnimals::No => {
                // Escape typewriter text is left as vanilla: "ESCAPE IMMEDIATELY!"
            }
            SaveAnimals::Yes => {
                // Change end-game behavior to require saving the animals. Address here must match escape.asm:
                self.rom.write_u16(snes2pc(0xA1F000), 0xFFFF)?;

                // Replace the escape typewriter text: "SAVE THE ANIMALS!"
                self.rom.write_n(
                    snes2pc(0xA6C4B6),
                    &[
                        0x53, 0x41, 0x56, 0x45, 0x20, 0x54, 0x48, 0x45, 0x20, 0x41, 0x4E, 0x49,
                        0x4D, 0x41, 0x4C, 0x53, 0x21, 0x00, 0x00,
                    ],
                )?;
            }
            SaveAnimals::Optional => {
                // Replace the escape typewriter text: "THINK OF THE ANIMALS!"
                let bank_a6_free_space_start = 0xa6febc;
                let bank_a6_free_space_end = 0xa6ff00;
                self.rom.write_u16(
                    snes2pc(0xa6c250),
                    (bank_a6_free_space_start & 0xffff) as isize,
                )?;
                let typewriter_text = vec![
                    0x01, 0x00, 0x02, 0x00, // Timer reset value = 2
                    0x0D, 0x00, 0x05, 0x49, // VRAM tilemap address = $4905 (BG2 tile (5, 8))
                    // "TIME BOMB SET!":
                    0x54, 0x49, 0x4D, 0x45, 0x20, 0x42, 0x4F, 0x4D, 0x42, 0x20, 0x53, 0x45, 0x54,
                    0x21, //  VRAM tilemap address = $4945 (BG2 tile (5, Ah)):
                    0x0D, 0x00, 0x45, 0x49, // "THINK OF THE ANIMALS!":
                    0x54, 0x4E, 0x49, 0x4E, 0x4B, 0x20, 0x4F, 0x46, 0x20, 0x54, 0x48, 0x45, 0x20,
                    0x41, 0x4E, 0x49, 0x4D, 0x41, 0x4C, 0x53, 0x21, 0x00, 0x00,
                ];
                self.rom
                    .write_n(snes2pc(bank_a6_free_space_start), &typewriter_text)?;
                assert!(bank_a6_free_space_start + typewriter_text.len() <= bank_a6_free_space_end)
            }
            SaveAnimals::Random => {
                panic!("Unexpected SaveAnimals::Random");
            }
        }

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .escape_enemies_cleared
        {
            // Change escape behavior to clear enemies. Address here must match escape.asm:
            self.rom.write_u16(snes2pc(0xA1f004), 0xFFFF)?;
        }

        if !self
            .randomization
            .settings
            .quality_of_life_settings
            .escape_refill
        {
            // Disable the energy refill at the start of the escape. Address here must match escape.asm:
            self.rom.write_u16(snes2pc(0xA1F002), 0x0001)?;
        }

        Ok(())
    }

    fn apply_title_screen_patches(&mut self) -> Result<()> {
        let mut rng_seed = [0u8; 32];
        rng_seed[..8].copy_from_slice(&self.randomization.seed.to_le_bytes());
        let mut rng = rand::rngs::StdRng::from_seed(rng_seed);

        let mut img = Array3::<u8>::zeros((224, 256, 3));
        loop {
            let top_left_idx = rng.gen_range(0..self.game_data.title_screen_data.top_left.len());
            let top_right_idx = rng.gen_range(0..self.game_data.title_screen_data.top_right.len());
            let bottom_left_idx =
                rng.gen_range(0..self.game_data.title_screen_data.bottom_left.len());
            let bottom_right_idx =
                rng.gen_range(0..self.game_data.title_screen_data.bottom_right.len());

            let top_left_slice = self.game_data.title_screen_data.top_left[top_left_idx]
                .slice(ndarray::s![32..144, 0..128, ..]);
            let top_right_slice = self.game_data.title_screen_data.top_right[top_right_idx]
                .slice(ndarray::s![32..144, 128..256, ..]);
            let bottom_left_slice = self.game_data.title_screen_data.bottom_left[bottom_left_idx]
                .slice(ndarray::s![112..224, 0..128, ..]);
            let bottom_right_slice = self.game_data.title_screen_data.bottom_right
                [bottom_right_idx]
                .slice(ndarray::s![112..224, 128..256, ..]);

            img.slice_mut(ndarray::s![0..112, 0..128, ..])
                .assign(&top_left_slice);
            img.slice_mut(ndarray::s![0..112, 128..256, ..])
                .assign(&top_right_slice);
            img.slice_mut(ndarray::s![112..224, 0..128, ..])
                .assign(&bottom_left_slice);
            img.slice_mut(ndarray::s![112..224, 128..256, ..])
                .assign(&bottom_right_slice);

            let map = &self.game_data.title_screen_data.map_station;
            for y in 0..224 {
                for x in 0..256 {
                    if map[(y, x, 0)] == 0 && map[(y, x, 1)] == 0 && map[(y, x, 2)] == 0 {
                        continue;
                    }
                    img[(y, x, 0)] = map[(y, x, 0)];
                    img[(y, x, 1)] = map[(y, x, 1)];
                    img[(y, x, 2)] = map[(y, x, 2)];
                }
            }

            let mut title_patcher = title::TitlePatcher::new(&mut self.rom);
            let bg_result = title_patcher.patch_title_background(&img);
            if bg_result.is_err() {
                info!(
                    "Failed title screen randomization: {}",
                    bg_result.unwrap_err()
                );
                continue;
            }
            title_patcher.patch_title_foreground()?;
            title_patcher.patch_title_gradient()?;
            title_patcher.patch_title_blue_light()?;
            println!(
                "Title screen data end: {:x}",
                pc2snes(title_patcher.next_free_space_pc)
            );
            return Ok(());
        }
    }

    fn setup_reload_cre(&mut self) -> Result<()> {
        // Find the rooms connected to Kraid and Crocomire and set them to reload CRE, to prevent graphical glitches.
        // Not sure if this is necessary for Crocomire. The vanilla game does it but we skip it since it doesn't seem to be a problem.
        let reload_cre_door_pairs: HashSet<DoorPtrPair> = [
            (Some(0x191DA), Some(0x19252)), // Kraid right door
            (Some(0x191CE), Some(0x191B6)), // Kraid left door
                                            // (Some(0x193DE), Some(0x19432)), // Crocomire left door
                                            // (Some(0x193EA), Some(0x193D2)), // Crocomire top door
        ]
        .into();
        for (src_pair, dst_pair, _bidirectional) in &self.map.doors {
            if reload_cre_door_pairs.contains(src_pair) {
                let (room_idx, _door_idx) =
                    self.game_data.room_and_door_idxs_by_door_ptr_pair[dst_pair];
                self.rom
                    .write_u8(self.game_data.room_geometry[room_idx].rom_address + 8, 2)?;
            }
            if reload_cre_door_pairs.contains(dst_pair) {
                let (room_idx, _door_idx) =
                    self.game_data.room_and_door_idxs_by_door_ptr_pair[src_pair];
                if self.game_data.room_geometry[room_idx].name == "Pants Room" {
                    // Apply reload CRE to East Pants Room rather than Pants Room:
                    self.rom.write_u8(0x7D69A + 8, 2)?;
                } else {
                    self.rom
                        .write_u8(self.game_data.room_geometry[room_idx].rom_address + 8, 2)?;
                }
            }
        }
        Ok(())
    }

    fn customize_escape_timer(&mut self) -> Result<()> {
        let escape_time = self.randomization.spoiler_log.escape.final_time_seconds as isize;
        let minutes = escape_time / 60;
        let seconds = escape_time % 60;
        self.rom
            .write_u8(snes2pc(0x809E21), (seconds % 10) + 16 * (seconds / 10))?;
        self.rom
            .write_u8(snes2pc(0x809E22), (minutes % 10) + 16 * (minutes / 10))?;
        Ok(())
    }

    fn fix_crateria_scrolling_sky(&mut self) -> Result<()> {
        // This function probably isn't needed anymore since we're using Mosaic's
        // reimplementation of scrolling sky.
        let data = vec![
            (0x8FB76C, (0x1892E, 0x18946)), // Landing Site
            (0x8FB777, (0x18916, 0x1896A)), // Landing Site
            (0x8FB782, (0x1893A, 0x189B2)), // Landing Site
            (0x8FB78D, (0x18922, 0x18AC6)), // Landing Site
            (0x8FB7B0, (0x189E2, 0x18A12)), // West Ocean
            (0x8FB7BB, (0x189CA, 0x18AEA)), // West Ocean (Bottom-left door, to Moat)
            (0x8FB7C6, (0x189FA, 0x1A18C)), // West Ocean
            (0x8FB7D1, (0x189D6, 0x1A1B0)), // West Ocean
            (0x8FB7DC, (0x189EE, 0x1A1E0)), // West Ocean
            (0x8FB7E7, (0x18A06, 0x1A300)), // West Ocean
            (0x8FB7F4, (0x18A72, 0x18A7E)), // East Ocean
            (0x8FB7FF, (0x18A66, 0x1A264)), // East Ocean
        ];
        for (addr, (exit_ptr, entrance_ptr)) in data {
            let door_pair = (Some(exit_ptr), Some(entrance_ptr));
            let other_door_pair = self.other_door_ptr_pair_map[&door_pair];
            self.rom.write_u16(
                snes2pc(addr),
                (other_door_pair.0.unwrap() & 0xFFFF) as isize,
            )?;
        }

        Ok(())
    }

    fn apply_seed_identifiers(&mut self) -> Result<()> {
        let cartridge_name = "SUPERMETROID MAPRANDO";
        self.rom.write_n(0x7FC0, cartridge_name.as_bytes())?;

        // Write seed name as a null-terminated URL-safe ASCII string.
        // This can be used to look up seeds on the website as https://maprando.com/seed/{seed name}/
        assert!(self.randomization.seed_name.as_bytes().len() < 16);
        self.rom.write_n(snes2pc(0xdffef0), &[0; 16])?;
        self.rom
            .write_n(snes2pc(0xdffef0), self.randomization.seed_name.as_bytes())?;

        // Write the display_seed, used by "seed_hash_display.asm" to show enemy names in the main menu.
        let seed_bytes = (self.randomization.display_seed as u32).to_le_bytes();
        self.rom.write_n(snes2pc(0xdfff00), &seed_bytes)?;
        Ok(())
    }

    fn write_item_credits(
        &mut self,
        idx: usize,
        step: Option<usize>,
        item: &str,
        item_idx: Option<usize>,
        area: &str,
    ) -> Result<()> {
        let base_addr = snes2pc(0xceb240 + (164 - 128 + idx * 2) * 0x40);

        if let Some(step) = step {
            // Write step number
            if step >= 10 {
                write_credits_big_digit(self.rom, step / 10, base_addr + 2)?;
            }
            write_credits_big_digit(self.rom, step % 10, base_addr + 4)?;

            // Write colon after step number:
            self.rom.write_u16(base_addr + 6, 0x5A)?;
            self.rom.write_u16(base_addr + 6 + 0x40, 0x5A)?;
        }

        // Write item text
        for (i, c) in item.chars().enumerate() {
            let c = c.to_ascii_uppercase();
            if c >= 'A' && c <= 'Z' {
                let word = 0x0400 | (c as isize - 'A' as isize);
                self.rom.write_u16(base_addr + (i + 5) * 2, word)?;
            }
        }

        // Write area text
        for (i, c) in area.chars().enumerate() {
            let c = c.to_ascii_uppercase();
            if c >= 'A' && c <= 'Z' {
                let word = 0x0C00 | (c as isize - 'A' as isize);
                self.rom.write_u16(base_addr + (i + 5) * 2 + 0x40, word)?;
            }
        }

        if let Some(item_idx) = item_idx {
            // Write stats address for collection time
            let stats_table_addr = snes2pc(0xdfe000);
            let item_time_addr = 0xfe06;
            self.rom.write_u16(
                stats_table_addr + idx * 8,
                (item_time_addr + 4 * item_idx) as isize,
            )?;
            // Write stats type (2 = Time):
            self.rom.write_u16(stats_table_addr + idx * 8 + 6, 2)?;
        }
        Ok(())
    }

    fn write_preset(&mut self, row: usize, preset: Option<String>) -> Result<()> {
        let preset = preset.unwrap_or("Custom".to_string());
        let base_addr = snes2pc(0xceb240 + (row - 128) * 0x40);
        for (i, c) in preset.chars().enumerate() {
            let c = c.to_ascii_uppercase();
            if c >= 'A' && c <= 'Z' {
                write_credits_big_letter(self.rom, c, base_addr + 0x3E - preset.len() * 2 + i * 2)?;
            }
        }
        Ok(())
    }

    fn apply_credits(&mut self) -> Result<()> {
        // Write randomizer settings to credits tilemap
        self.write_preset(
            224,
            self.randomization
                .settings
                .skill_assumption_settings
                .preset
                .clone(),
        )?;
        self.write_preset(
            226,
            self.randomization
                .settings
                .item_progression_settings
                .preset
                .clone(),
        )?;
        self.write_preset(
            228,
            self.randomization
                .settings
                .quality_of_life_settings
                .preset
                .clone(),
        )?;

        // Write item locations in credits tilemap
        let item_name_pairs: Vec<(String, String)> = [
            ("ETank", "Energy Tank"),
            ("Missile", "Missile"),
            ("Super", "Super Missile"),
            ("PowerBomb", "Power Bomb"),
            ("Bombs", "Bombs"),
            ("Charge", "Charge Beam"),
            ("Ice", "Ice Beam"),
            ("HiJump", "HiJump Boots"),
            ("SpeedBooster", "Speed Booster"),
            ("Wave", "Wave Beam"),
            ("Spazer", "Spazer"),
            ("SpringBall", "Spring Ball"),
            ("Varia", "Varia Suit"),
            ("Gravity", "Gravity Suit"),
            ("XRayScope", "XRay Scope"),
            ("Plasma", "Plasma Beam"),
            ("Grapple", "Grappling Beam"),
            ("SpaceJump", "Space Jump"),
            ("ScrewAttack", "Screw Attack"),
            ("Morph", "Morph Ball"),
            ("ReserveTank", "Reserve Tank"),
            ("WallJump", "WallJump Boots"),
        ]
        .into_iter()
        .map(|(x, y)| (x.to_string(), y.to_string()))
        .collect();
        let item_display_name_map: HashMap<String, String> =
            item_name_pairs.iter().cloned().collect();
        let item_name_index: HashMap<String, usize> = item_name_pairs
            .iter()
            .enumerate()
            .map(|(i, x)| (x.0.clone(), i))
            .collect();
        let mut items_set: HashSet<String> = HashSet::new();

        // Show starting items at the top:
        for x in &self
            .randomization
            .settings
            .item_progression_settings
            .starting_items
        {
            if x.count > 0 {
                let raw_name = Item::VARIANTS[x.item as usize].to_string();
                let item_name = item_display_name_map[&raw_name].clone();
                self.write_item_credits(items_set.len(), None, &item_name, None, "starting item")?;
                items_set.insert(raw_name.clone());
            }
        }

        // Show collectible items in the middle:
        for (step, step_summary) in self.randomization.spoiler_log.summary.iter().enumerate() {
            for item_info in step_summary.items.iter() {
                if !items_set.contains(&item_info.item) {
                    let item_name = item_display_name_map[&item_info.item].clone();
                    let item_idx = item_name_index[&item_info.item];
                    self.write_item_credits(
                        items_set.len(),
                        Some(step + 1),
                        &item_name,
                        Some(item_idx),
                        &item_info.location.area,
                    )?;
                    items_set.insert(item_info.item.clone());
                }
            }
        }

        // Show logically uncollectible items:
        for loc in &self.randomization.spoiler_log.all_items {
            if loc.item == "Nothing" {
                continue;
            }
            if !items_set.contains(&loc.item) {
                let item_name = item_display_name_map[&loc.item].clone();
                let item_idx = item_name_index[&loc.item];
                self.write_item_credits(
                    items_set.len(),
                    None,
                    &item_name,
                    Some(item_idx),
                    &loc.location.area,
                )?;
                items_set.insert(loc.item.clone());
            }
        }

        // Show unplaced items at the bottom:
        for (name, display_name) in &item_name_pairs {
            if self.randomization.settings.other_settings.wall_jump != WallJump::Collectible
                && name == "WallJump"
            {
                // Don't show "WallJump" item unless using Collectible mode.
                continue;
            }
            if !items_set.contains(name) {
                self.write_item_credits(items_set.len(), None, &display_name, None, "not placed")?;
                items_set.insert(name.clone());
            }
        }
        Ok(())
    }

    fn set_starting_items(&mut self) -> Result<()> {
        // Addresses used in new_game.asm:
        let initial_items_collected = snes2pc(0xB5FE04);
        let initial_items_equipped = snes2pc(0xB5FE06);
        let initial_beams_collected = snes2pc(0xB5FE08);
        let initial_beams_equipped = snes2pc(0xB5FE0A);
        let initial_item_bits = snes2pc(0xB5FE12);
        let initial_energy = snes2pc(0xB5FE52);
        let initial_max_energy = snes2pc(0xB5FE54);
        let initial_reserve_energy = snes2pc(0xB5FE56);
        let initial_max_reserve_energy = snes2pc(0xB5FE58);
        let initial_reserve_mode = snes2pc(0xB5FE5A);
        let initial_missiles = snes2pc(0xB5FE5C);
        let initial_max_missiles = snes2pc(0xB5FE5E);
        let initial_supers = snes2pc(0xB5FE60);
        let initial_max_supers = snes2pc(0xB5FE62);
        let initial_power_bombs = snes2pc(0xB5FE64);
        let initial_max_power_bombs = snes2pc(0xB5FE66);

        let mut item_mask = 0;
        let mut beam_mask = 0;
        let mut starting_missiles = 0;
        let mut starting_energy = 99;
        let mut starting_reserves = 0;
        let mut starting_supers = 0;
        let mut starting_powerbombs = 0;
        let item_bitmask_map: HashMap<Item, u16> = vec![
            (Item::Varia, 0x0001),
            (Item::SpringBall, 0x0002),
            (Item::Morph, 0x0004),
            (Item::ScrewAttack, 0x0008),
            (Item::Gravity, 0x0020),
            (Item::HiJump, 0x0100),
            (Item::SpaceJump, 0x0200),
            (Item::Bombs, 0x1000),
            (Item::SpeedBooster, 0x2000),
            (Item::Grapple, 0x4000),
            (Item::XRayScope, 0x8000),
        ]
        .into_iter()
        .collect();
        let beam_bitmask_map: HashMap<Item, u16> = vec![
            (Item::Wave, 0x0001),
            (Item::Ice, 0x0002),
            (Item::Spazer, 0x0004),
            (Item::Plasma, 0x0008),
            (Item::Charge, 0x1000),
        ]
        .into_iter()
        .collect();
        for x in &self
            .randomization
            .settings
            .item_progression_settings
            .starting_items
        {
            if x.count == 0 {
                continue;
            }
            if item_bitmask_map.contains_key(&x.item) {
                item_mask |= item_bitmask_map[&x.item];
            } else if beam_bitmask_map.contains_key(&x.item) {
                beam_mask |= beam_bitmask_map[&x.item];
            } else if x.item == Item::Missile {
                starting_missiles += (x.count as isize) * 5;
            } else if x.item == Item::ETank {
                starting_energy += (x.count as isize) * 100;
            } else if x.item == Item::ReserveTank {
                starting_reserves += (x.count as isize) * 100;
            } else if x.item == Item::Super {
                starting_supers += (x.count as isize) * 5;
            } else if x.item == Item::PowerBomb {
                starting_powerbombs += (x.count as isize) * 5;
            }
        }
        let beam_equipped_mask = if beam_mask & 0x000C == 0x000C {
            // Don't equip Spazer if Plasma equipped
            beam_mask & !0x0004
        } else {
            beam_mask
        };

        // Set items collected/equipped:
        self.rom
            .write_u16(initial_items_collected, item_mask as isize)?;
        self.rom
            .write_u16(initial_items_equipped, item_mask as isize)?;
        self.rom
            .write_u16(initial_beams_collected, beam_mask as isize)?;
        self.rom
            .write_u16(initial_beams_equipped, beam_equipped_mask as isize)?;
        self.rom.write_u16(initial_energy, starting_energy)?;
        self.rom.write_u16(initial_max_energy, starting_energy)?;
        self.rom
            .write_u16(initial_reserve_energy, starting_reserves)?;
        self.rom
            .write_u16(initial_max_reserve_energy, starting_reserves)?;
        self.rom.write_u16(
            initial_reserve_mode,
            if starting_reserves > 0 { 1 } else { 0 },
        )?; // 0 = Not obtained, 1 = Auto
        self.rom
            .write_u16(initial_max_missiles, starting_missiles)?;
        self.rom.write_u16(initial_max_supers, starting_supers)?;
        self.rom
            .write_u16(initial_max_power_bombs, starting_powerbombs)?;
        if self.randomization.settings.start_location_mode == StartLocationMode::Escape
            && self
                .randomization
                .settings
                .quality_of_life_settings
                .mother_brain_fight
                != MotherBrainFight::Skip
        {
            self.rom.write_u16(initial_missiles, 0)?;
            self.rom.write_u16(initial_supers, 0)?;
            self.rom.write_u16(initial_power_bombs, 0)?;
        } else {
            self.rom.write_u16(initial_missiles, starting_missiles)?;
            self.rom.write_u16(initial_supers, starting_supers)?;
            self.rom
                .write_u16(initial_power_bombs, starting_powerbombs)?;
        }
        self.rom
            .write_n(initial_item_bits, &self.nothing_item_bitmask)?;

        Ok(())
    }

    fn set_start_location(&mut self) -> Result<()> {
        let initial_area_addr = snes2pc(0xB5FE00);
        let initial_load_station_addr = snes2pc(0xB5FE02);
        let initial_boss_bits = snes2pc(0xB5FE0C);

        if self.randomization.settings.start_location_mode == StartLocationMode::Escape {
            // Use Tourian load station 2, set up in escape_autosave.asm
            self.rom.write_u16(initial_area_addr, 5)?;
            self.rom.write_u16(initial_load_station_addr, 2)?;

            // Set all bosses defeated:
            self.rom.write_n(initial_boss_bits, &[7, 7, 7, 7, 7, 7])?;

            return Ok(());
        }

        // Use Crateria load station 2, to support random start (also used for Ship start)
        self.rom.write_u16(initial_area_addr, 0)?;
        self.rom.write_u16(initial_load_station_addr, 2)?;

        // Set no bosses defeated:
        self.rom.write_n(initial_boss_bits, &[0; 6])?;

        // Set starting room and Samus and camera starting location:
        let loc = self.randomization.start_location.clone();
        let room_addr = self.game_data.room_ptr_by_id[&loc.room_id];
        let door_node_id = loc.door_load_node_id.unwrap_or(loc.node_id);
        let (_, entrance_ptr) =
            self.game_data.reverse_door_ptr_pair_map[&(loc.room_id, door_node_id)];
        let x_pixels = (loc.x * 16.0) as isize;
        let y_pixels = (loc.y * 16.0) as isize - 24;
        let mut screen_x = x_pixels & 0xFF00;
        let mut screen_y = y_pixels & 0xFF00;
        screen_x += (loc.camera_offset_x.unwrap_or(0.0) * 16.0) as isize;
        screen_y += (loc.camera_offset_y.unwrap_or(0.0) * 16.0) as isize;
        let samus_x = x_pixels - (screen_x + 0x80);
        let samus_y = y_pixels - screen_y;
        let station_addr = snes2pc(0x80C4E1);
        self.rom
            .write_u16(station_addr, (room_addr & 0xFFFF) as isize)?;
        self.rom
            .write_u16(station_addr + 2, (entrance_ptr.unwrap() & 0xFFFF) as isize)?;
        self.rom.write_u16(station_addr + 6, screen_x)?;
        self.rom.write_u16(station_addr + 8, screen_y)?;
        self.rom
            .write_u16(station_addr + 10, ((samus_y as i16) as u16) as isize)?;
        self.rom
            .write_u16(station_addr + 12, ((samus_x as i16) as u16) as isize)?;
        Ok(())
    }

    fn apply_door_hazard_marker(&mut self, door_ptr_pair: DoorPtrPair) -> Result<()> {
        let mut other_door_ptr_pair = self.other_door_ptr_pair_map[&door_ptr_pair];

        if other_door_ptr_pair == (Some(0x1AA8C), Some(0x1AAE0)) {
            // Don't draw hazard marker on left side of Mother Brain Room:
            return Ok(());
        }

        if other_door_ptr_pair == (Some(0x1A600), Some(0x1A678)) {
            // For the Toilet, pass through to room above:
            other_door_ptr_pair = self.other_door_ptr_pair_map[&(Some(0x1A60C), Some(0x1A5AC))];
        }
        let (room_idx, door_idx) =
            self.game_data.room_and_door_idxs_by_door_ptr_pair[&other_door_ptr_pair];
        let room = &self.game_data.room_geometry[room_idx];
        let door = &room.doors[door_idx];

        let plm_id: u16;
        let tile_x: usize;
        let tile_y: usize;
        if door.direction == "right" {
            plm_id = 0xF800; // must match address in hazard_markers.asm
            tile_x = door.x * 16 + 15;
            tile_y = door.y * 16 + 6;
        } else if door.direction == "left" {
            plm_id = 0xF80C; // must match address in hazard_markers.asm
            tile_x = door.x * 16;
            tile_y = door.y * 16 + 6;
        } else if door.direction == "down" {
            if door.offset == Some(0) {
                plm_id = 0xF808; // hazard marking overlaid on transition tiles
            } else {
                plm_id = 0xF804;
            }
            tile_x = door.x * 16 + 6;
            tile_y = door.y * 16 + 15 - door.offset.unwrap_or(0);
        } else {
            panic!(
                "Unsupported door direction for hazard marker: {}",
                door.direction
            );
        }

        let mut write_asm = |room_ptr: usize, x: usize, y: usize| {
            self.extra_setup_asm
                .entry(room_ptr)
                .or_insert(vec![])
                .extend(vec![
                    0x22,
                    0xD7,
                    0x83,
                    0x84, // jsl $8483D7
                    x as u8,
                    y as u8, // X and Y coordinates in 16x16 tiles
                    (plm_id & 0x00FF) as u8,
                    (plm_id >> 8) as u8,
                ]);
        };

        write_asm(room.rom_address, tile_x, tile_y);
        if room.rom_address == 0x793FE && door.x == 5 && door.y == 2 {
            // Homing Geemer Room
            write_asm(room.twin_rom_address.unwrap(), tile_x % 16, tile_y % 16);
        }
        if room.rom_address == 0x7D646 && door.x == 1 && door.y == 2 {
            // East Pants Room
            write_asm(
                room.twin_rom_address.unwrap(),
                tile_x % 16,
                tile_y % 16 + 16,
            );
        }
        Ok(())
    }

    fn apply_hazard_markers(&mut self) -> Result<()> {
        let mut door_ptr_pairs = vec![
            (Some(0x1A42C), Some(0x1A474)), // Mt. Everest (top)
            (Some(0x1A678), Some(0x1A600)), // Oasis (top)
            (Some(0x1A3F0), Some(0x1A444)), // Fish Tank (top left)
            (Some(0x1A3FC), Some(0x1A450)), // Fish Tank (top right)
            (Some(0x19996), Some(0x1997E)), // Amphitheatre (left)
            (Some(0x1AA14), Some(0x1AA20)), // Tourian Blue Hoppers (left)
            (Some(0x18DDE), Some(0x18E6E)), // Big Pink crumble blocks (left),
            (Some(0x19312), Some(0x1934E)), // Ice Beam Gate Room crumbles (top left)
            (Some(0x1A630), Some(0x1A5C4)), // Bug Sand Hole (left)
            (Some(0x1A618), Some(0x1A564)), // Bug Sand Hole (right)
            (Some(0x198EE), Some(0x1992A)), // Fast Pillars Setup Room (top right)
        ];
        if self.randomization.settings.other_settings.wall_jump != WallJump::Vanilla {
            door_ptr_pairs.extend(vec![
                (Some(0x18A06), Some(0x1A300)), // West Ocean Gravity Suit door
                (Some(0x198BE), Some(0x198CA)), // Ridley's Room top door
                (Some(0x193EA), Some(0x193D2)), // Crocomire's Room top door
            ]);
        }
        for pair in door_ptr_pairs {
            self.apply_door_hazard_marker(pair)?;
        }
        Ok(())
    }

    fn apply_single_locked_door(&mut self, locked_door: LockedDoor, state_index: u8) -> Result<()> {
        let (room_idx, door_idx) =
            self.game_data.room_and_door_idxs_by_door_ptr_pair[&locked_door.src_ptr_pair];
        let room = &self.game_data.room_geometry[room_idx];
        let door = &room.doors[door_idx];
        let (x, y) = match door.direction.as_str() {
            "right" => (door.x * 16 + 14 - door.offset.unwrap_or(0), door.y * 16 + 6),
            "left" => (door.x * 16 + 1 + door.offset.unwrap_or(0), door.y * 16 + 6),
            "up" => (door.x * 16 + 6, door.y * 16 + 1 + door.offset.unwrap_or(0)),
            "down" => (door.x * 16 + 6, door.y * 16 + 14 - door.offset.unwrap_or(0)),
            _ => panic!("Unexpected door direction: {}", door.direction),
        };
        let plm_id = match (locked_door.door_type, door.direction.as_str()) {
            (DoorType::Yellow, "right") => 0xC85A,
            (DoorType::Yellow, "left") => 0xC860,
            (DoorType::Yellow, "down") => 0xC866,
            (DoorType::Yellow, "up") => 0xC86C,
            (DoorType::Green, "right") => 0xC872,
            (DoorType::Green, "left") => 0xC878,
            (DoorType::Green, "down") => 0xC87E,
            (DoorType::Green, "up") => 0xC884,
            (DoorType::Red, "right") => 0xC88A,
            (DoorType::Red, "left") => 0xC890,
            (DoorType::Red, "down") => 0xC896,
            (DoorType::Red, "up") => 0xC89C,
            (DoorType::Beam(_), "right") => 0xFCC0,
            (DoorType::Beam(_), "left") => 0xFCC6,
            (DoorType::Beam(_), "down") => 0xFCCC,
            (DoorType::Beam(_), "up") => 0xFCD2,
            (a, b) => panic!("Unexpected door type: {:?} {}", a, b),
        };
        // TODO: Instead of using extra setup ASM to spawn the doors, it might be better to just rewrite
        // the room PLM list, to add the new door PLMs.
        let mut write_asm = |room_ptr: usize, x: usize, y: usize| {
            self.extra_setup_asm
                .entry(room_ptr)
                .or_insert(vec![])
                .extend(vec![
                    0x22,
                    0x80,
                    0xF3,
                    0x84, // JSL $84F380  (Spawn hard-coded PLM with room argument)
                    x as u8,
                    y as u8, // X and Y coordinates in 16x16 tiles
                    (plm_id & 0x00FF) as u8,
                    (plm_id >> 8) as u8,
                    state_index,
                    0x00, // PLM argument (index for door unlock state)
                ]);
            if let DoorType::Beam(beam) = locked_door.door_type {
                let gfx_base_addr = if door.direction == "right" || door.direction == "left" {
                    (beam as usize) * 0xC40
                } else {
                    (beam as usize) * 0xC40 + 0x620
                } + 0xEA8000;
                self.extra_setup_asm
                    .get_mut(&room_ptr)
                    .unwrap()
                    .extend(vec![
                        // LDA #beam_type
                        0xA9,
                        beam as u8,
                        0x00,
                        // LDX #gfx_base_addr
                        0xA2,
                        (gfx_base_addr & 0xFF) as u8,
                        (gfx_base_addr >> 8) as u8,
                        // JSL $84FCD8 (run `load_beam_tiles` in beam_doors.asm)
                        0x22,
                        0xD8,
                        0xFC,
                        0x84,
                    ]);
            }
        };
        write_asm(room.rom_address, x, y);
        if room.rom_address == 0x793FE && door.x == 5 && door.y == 2 {
            // Homing Geemer Room
            write_asm(room.twin_rom_address.unwrap(), x % 16, y % 16);
        }
        if room.rom_address == 0x7D646 && door.x == 1 && door.y == 2 {
            // East Pants Room
            write_asm(room.twin_rom_address.unwrap(), x % 16, y % 16 + 16);
        }
        if self.randomization.toilet_intersections.contains(&room_idx) {
            let toilet_pos = self.randomization.map.rooms[self.game_data.toilet_room_idx];
            let room_pos = self.randomization.map.rooms[room_idx];
            let rel_x = room_pos.0 as isize - toilet_pos.0 as isize;
            let rel_y = room_pos.1 as isize - toilet_pos.1 as isize;
            let new_x = x as isize + rel_x * 16;
            let new_y = y as isize + rel_y * 16;
            if new_x >= 0 && new_x < 16 && new_y >= 0 && new_y < 160 {
                write_asm(0x7D408, new_x as usize, new_y as usize);
            }
        }

        Ok(())
    }

    fn assign_locked_door_states(&mut self) {
        // PLM arguments used for gray door states (we reserve all of them even though not all are used)
        let reserved_state_indexes: HashSet<usize> = [
            0x2, 0x3, 0x4, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0x11, 0x12, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1a, 0x1b, 0x1c, 0x24, 0x25, 0x2c, 0x2d, 0x2e, 0x2f, 0x31, 0x36, 0x37, 0x3e,
            0x40, 0x41, 0x42, 0x43, 0x46, 0x47, 0x48, 0x4f, 0x50, 0x59, 0x5a, 0x5b, 0x5d, 0x60,
            0x80, 0x81, 0x82, 0x83, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x91, 0x93, 0x97, 0x9c, 0x9d,
            0x9e, 0x9f, 0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xaa, 0xab, 0xac,
        ]
        .into_iter()
        .collect();
        let mut next_state_index: usize = 0;
        let mut state_idxs: Vec<usize> = vec![];

        for _door in &self.randomization.locked_door_data.locked_doors {
            while reserved_state_indexes.contains(&next_state_index) {
                next_state_index += 1;
            }
            state_idxs.push(next_state_index);
            next_state_index += 1;
        }
        self.locked_door_state_indices = state_idxs;
    }

    fn apply_locked_doors(&mut self) -> Result<()> {
        self.assign_locked_door_states();
        for (i, door) in self
            .randomization
            .locked_door_data
            .locked_doors
            .iter()
            .enumerate()
        {
            let mut door = *door;
            if door.door_type == DoorType::Gray {
                // Skipping gray doors since they are already handled elsewhere.
                // TODO: simplify things by just handling it here?
                continue;
            }
            self.apply_single_locked_door(door, self.locked_door_state_indices[i] as u8)?;
            if door.bidirectional {
                std::mem::swap(&mut door.src_ptr_pair, &mut door.dst_ptr_pair);
                self.apply_single_locked_door(door, self.locked_door_state_indices[i] as u8)?;
            }
        }
        Ok(())
    }

    fn apply_extra_setup_asm(&mut self) -> Result<()> {
        // remove unused pointer from Bomb Torizo room (Zebes ablaze state), to avoid misinterpreting it as an
        // extra setup ASM pointer.
        self.rom.write_u16(snes2pc(0x8f985f), 0x0000)?;

        let mut next_addr = snes2pc(0xB88200);

        for (&room_ptr, asm) in &self.extra_setup_asm {
            for (_, state_ptr) in get_room_state_ptrs(&self.rom, room_ptr)? {
                let mut asm = asm.clone();
                asm.push(0x60); // RTS
                self.rom.write_n(next_addr, &asm)?;
                self.rom
                    .write_u16(state_ptr + 16, (pc2snes(next_addr) & 0xFFFF) as isize)?;
                next_addr += asm.len();
            }
        }
        println!("extra setup ASM end: {:x}", next_addr);
        assert!(next_addr <= snes2pc(0xB8FFFF));

        Ok(())
    }

    fn write_walljump_item_graphics(&mut self) -> Result<()> {
        let f = 0xF;
        let frame_1: [[u8; 16]; 16] = [
            [0, 0, 0, f, f, f, f, f, f, f, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, f, 4, 4, 5, 6, 6, f, f, f, f, 0, 0, 0],
            [0, 0, 0, f, 4, 5, 6, 6, f, f, 6, 4, f, 0, 0, 0],
            [0, 0, 0, f, 5, 6, 6, f, f, 6, 4, 5, f, 0, 0, 0],
            [0, 0, 0, f, 5, 6, f, f, 6, 4, 5, f, f, 0, 0, 0],
            [0, 0, 0, f, 6, 6, f, 6, 4, 5, 6, f, 0, 0, 0, 0],
            [0, 0, f, f, f, f, 6, f, 6, 5, f, 0, 0, 0, 0, 0],
            [0, 0, f, 5, 6, 6, f, 6, f, 6, f, 0, 0, 0, 0, 0],
            [0, 0, f, 4, 5, 6, f, 5, 6, f, f, 0, 0, 0, 0, 0],
            [0, 0, f, f, 6, 5, f, 6, f, f, 0, 0, 0, 0, 0, 0],
            [0, f, 6, f, f, 5, f, f, f, 6, f, 0, 0, 0, 0, 0],
            [f, f, 6, 6, f, f, f, 6, 4, 5, f, f, 0, 0, 0, 0],
            [f, 6, 6, 4, 6, f, 6, 6, 6, 6, 6, 6, f, f, 0, 0],
            [f, 6, 4, 4, 6, f, 5, 5, 5, 4, 4, 4, 5, 5, f, f],
            [f, 6, 5, 5, 6, f, f, f, 6, 6, 6, 6, 6, 6, 6, f],
            [f, f, f, f, f, 0, 0, 0, f, f, f, f, f, f, f, f],
        ];
        let frame_2 = frame_1.map(|row| {
            row.map(|x| match x {
                4 => 0xb,
                5 => 4,
                6 => 5,
                7 => 6,
                0xf => 7,
                y => y,
            })
        });
        let frames: [[[u8; 16]; 16]; 2] = [frame_1, frame_2];
        let mut addr = snes2pc(0x899100);
        for f in 0..2 {
            for tile_y in 0..2 {
                for tile_x in 0..2 {
                    let mut tile: [[u8; 8]; 8] = [[0; 8]; 8];
                    for y in 0..8 {
                        for x in 0..8 {
                            tile[y][x] = frames[f][tile_y * 8 + y][tile_x * 8 + x];
                        }
                    }
                    write_tile_4bpp(self.rom, addr, tile)?;
                    addr += 32;
                }
            }
        }
        Ok(())
    }

    fn apply_room_outline(&mut self, room_idx: usize, room_ptr: usize) -> Result<()> {
        let room = &self.game_data.room_geometry[room_idx];
        let room_x = self.rom.read_u8(room.rom_address + 2)?;
        let room_y = self.rom.read_u8(room.rom_address + 3)?;
        let area = self.map.area[room_idx];
        let mut asm: Vec<u8> = vec![];
        for y in 0..room.map.len() {
            for x in 0..room.map[0].len() {
                if room.map[y][x] == 0 && room_idx != self.game_data.toilet_room_idx {
                    continue;
                }
                let (offset, bitmask) =
                    xy_to_explored_bit_ptr(room_x + x as isize, room_y + y as isize);

                // Mark as partially revealed (which will persist after deaths/reloads):
                let addr = 0x2700 + (area as isize) * 0x100 + offset;
                asm.extend([0xAF, (addr & 0xFF) as u8, (addr >> 8) as u8, 0x70]); // LDA $70:{addr}
                asm.extend([0x09, bitmask, 0x00]); // ORA #{bitmask}
                asm.extend([0x8F, (addr & 0xFF) as u8, (addr >> 8) as u8, 0x70]);
                // STA $70:{addr}
            }
        }
        self.extra_setup_asm
            .entry(room_ptr)
            .or_insert(vec![])
            .extend(asm.clone());
        Ok(())
    }

    fn apply_all_room_outlines(&mut self) -> Result<()> {
        // Disable routine that marks tiles explored (used in vanilla game when entering boss rooms)
        // It's obsoleted by this more general "room outline" option.
        self.rom.write_u8(snes2pc(0x90A8A6), 0x60)?; // RTS

        for (room_idx, room) in self.game_data.room_geometry.iter().enumerate() {
            let room_ptr = room.rom_address;
            self.apply_room_outline(room_idx, room_ptr)?;
            if let Some(twin_rom_address) = room.twin_rom_address {
                self.apply_room_outline(room_idx, twin_rom_address)?;
            }
        }

        // For the Toilet, apply room outlines for the intersecting room(s) and vice versa:
        for &room_idx in &self.randomization.toilet_intersections {
            self.apply_room_outline(room_idx, 0x7D408)?;
            self.apply_room_outline(
                self.game_data.toilet_room_idx,
                self.game_data.room_geometry[room_idx].rom_address,
            )?;
        }
        Ok(())
    }

    fn apply_toilet_data(&mut self) -> Result<()> {
        let toilet_intersecting_room_ptr_addr = snes2pc(0xB5FE70);
        let toilet_rel_x_addr = snes2pc(0xB5FE72);
        let toilet_rel_y_addr = snes2pc(0xB5FE73);

        info!("Applying toilet data");
        if self.randomization.toilet_intersections.len() == 1 {
            let room_idx = self.randomization.toilet_intersections[0];
            let room_ptr = self.game_data.room_geometry[room_idx].rom_address;
            let room_pos = self.map.rooms[room_idx];
            let toilet_pos = self.map.rooms[self.game_data.toilet_room_idx];
            let rel_x = toilet_pos.0 as isize - room_pos.0 as isize;
            let rel_y = toilet_pos.1 as isize - room_pos.1 as isize;
            self.rom.write_u16(
                toilet_intersecting_room_ptr_addr,
                ((room_ptr & 0x7FFF) | 0x8000) as isize,
            )?;
            self.rom.write_u8(toilet_rel_x_addr, rel_x as u8 as isize)?;
            self.rom.write_u8(toilet_rel_y_addr, rel_y as u8 as isize)?;

            let intersection_state_ptr = get_room_state_ptrs(self.rom, room_ptr)?[0].1 | 0x70000;
            let mut intersection_plm_ptr =
                self.rom.read_u16(intersection_state_ptr + 20)? as usize | 0x70000;
            let mut toilet_plm_data: Vec<u8> = vec![];
            loop {
                let plm_type = self.rom.read_u16(intersection_plm_ptr)?;
                if plm_type == 0x0000 {
                    break;
                }
                if plm_type >= 0xEED7 && plm_type <= 0xF100 {
                    // item PLM
                    let mut plm_x = self.rom.read_u8(intersection_plm_ptr + 2)?;
                    let mut plm_y = self.rom.read_u8(intersection_plm_ptr + 3)?;
                    let plm_var = self.rom.read_u16(intersection_plm_ptr + 4)?;

                    plm_x -= rel_x * 16;
                    plm_y -= rel_y * 16;
                    if plm_x >= 0 && plm_x < 16 && plm_y >= 0 && plm_y < 160 {
                        toilet_plm_data.extend([
                            (plm_type & 0xFF) as u8,
                            (plm_type >> 8) as u8,
                            plm_x as u8,
                            plm_y as u8,
                            (plm_var & 0xFF) as u8,
                            (plm_var >> 8) as u8,
                        ]);
                    }
                }
                intersection_plm_ptr += 6;
            }
            toilet_plm_data.extend([0x00, 0x00]);
            assert!(toilet_plm_data.len() <= 0x40);

            let toilet_state_ptr = 0x7D415;
            let toilet_plm_ptr = snes2pc(0x8FFE00);
            self.rom.write_n(toilet_plm_ptr, &toilet_plm_data)?;
            self.rom.write_u16(
                toilet_state_ptr + 20,
                (toilet_plm_ptr as isize & 0x7FFF) | 0x8000,
            )?;
        }
        Ok(())
    }

    fn write_beam_door_tiles(&mut self) -> Result<()> {
        // Beam doors are limited to at most one per room, and their graphics (8x8 and 16x16) are loaded dynamically
        // when entering the room. The 16x16 tilemaps stay constant while the underlying 8x8 tiles get updated for the
        // idle animation and door opening.
        use beam_doors_tiles::*;
        let beam_door_gfx_idx = 0x260; // 0x260 through 0x267
        let beam_palettes = vec![
            1, // Charge
            3, // Ice
            2, // Wave
            0, // Spazer
            1, // Plasma
        ];
        let free_space_addr = snes2pc(0xEA8000);
        let gfx_size = 0x620; // Size of graphics + tilemaps per combination of beam type and orientation

        // Tilemap (16x16 tiles) indexed by orientation (0=horizontal, 1=vertical), then beam (0..5):
        // each address points to data for four 16x16 tiles, half of which is standard door frame and half of which is beam stuff
        // These are ROM addresses in bank EA, giving the source for data which will get copied to 0x7EA720 thru 0x7EA740
        for beam_idx in 0..5 {
            let door_pal = 1 << 10;
            let flip_x = 0x4000;
            let flip_y = 0x8000;
            let beam_pal = beam_palettes[beam_idx] << 10;

            // horizontal (left-side) door:
            let tilemap_ptr = free_space_addr + beam_idx * 2 * gfx_size;

            // 16x16 tile 0 (top)
            self.rom
                .write_u16(tilemap_ptr + 0x00, 0x2340 | door_pal | flip_x)?; // door frame tile 0
            self.rom
                .write_u16(tilemap_ptr + 0x02, 0x2000 | beam_door_gfx_idx | door_pal)?; // beam door tile 0
            self.rom
                .write_u16(tilemap_ptr + 0x04, 0x2350 | door_pal | flip_x)?; // door frame tile 1
            self.rom.write_u16(
                tilemap_ptr + 0x06,
                0x2000 | (beam_door_gfx_idx + 1) | door_pal,
            )?; // beam door tile 1

            // 16x16 tile 1 (top middle):
            self.rom
                .write_u16(tilemap_ptr + 0x08, 0x2360 | door_pal | flip_x)?; // door frame tile 2
            self.rom.write_u16(
                tilemap_ptr + 0x0A,
                0x2000 | (beam_door_gfx_idx + 2) | beam_pal,
            )?; // beam door tile 2
            self.rom
                .write_u16(tilemap_ptr + 0x0C, 0x2370 | door_pal | flip_x)?; // door frame tile 3
            self.rom.write_u16(
                tilemap_ptr + 0x0E,
                0x2000 | (beam_door_gfx_idx + 3) | beam_pal,
            )?; // beam door tile 3

            // 16x16 tile 2 (bottom middle):
            self.rom
                .write_u16(tilemap_ptr + 0x10, 0x2370 | door_pal | flip_x | flip_y)?; // door frame tile 3 (vertical flip)
            self.rom.write_u16(
                tilemap_ptr + 0x12,
                0x2000 | (beam_door_gfx_idx + 4) | beam_pal,
            )?; // beam door tile 4
            self.rom
                .write_u16(tilemap_ptr + 0x14, 0x2360 | door_pal | flip_x | flip_y)?; // door frame tile 2 (vertical flip)
            self.rom.write_u16(
                tilemap_ptr + 0x16,
                0x2000 | (beam_door_gfx_idx + 5) | beam_pal,
            )?; // beam door tile 5

            // 16x16 tile 3 (bottom):
            self.rom
                .write_u16(tilemap_ptr + 0x18, 0x2350 | door_pal | flip_x | flip_y)?; // door frame tile 1 (vertical flip)
            self.rom.write_u16(
                tilemap_ptr + 0x1A,
                0x2000 | (beam_door_gfx_idx + 6) | door_pal,
            )?; // beam door tile 6
            self.rom
                .write_u16(tilemap_ptr + 0x1C, 0x2340 | door_pal | flip_x | flip_y)?; // door frame tile 0 (vertical flip)
            self.rom.write_u16(
                tilemap_ptr + 0x1E,
                0x2000 | (beam_door_gfx_idx + 7) | door_pal,
            )?; // beam door tile 7

            // vertical (top-side) door:
            let tilemap_ptr = free_space_addr + (beam_idx * 2 + 1) * gfx_size;

            // 16x16 tile 0 (left)
            self.rom
                .write_u16(tilemap_ptr + 0x00, 0x2347 | door_pal | flip_x | flip_y)?; // door frame tile 0 (horizontal flip)
            self.rom
                .write_u16(tilemap_ptr + 0x02, 0x2346 | door_pal | flip_x | flip_y)?; // door frame tile 1 (horizontal flip)
            self.rom
                .write_u16(tilemap_ptr + 0x04, 0x2000 | beam_door_gfx_idx | door_pal)?; // beam door tile 0
            self.rom.write_u16(
                tilemap_ptr + 0x06,
                0x2000 | (beam_door_gfx_idx + 1) | door_pal,
            )?; // beam door tile 1

            // 16x16 tile 1 (left middle):
            self.rom
                .write_u16(tilemap_ptr + 0x08, 0x2345 | door_pal | flip_x | flip_y)?; // door frame tile 2 (horizontal flip)
            self.rom
                .write_u16(tilemap_ptr + 0x0A, 0x2344 | door_pal | flip_x | flip_y)?; // door frame tile 3 (horizontal flip)
            self.rom.write_u16(
                tilemap_ptr + 0x0C,
                0x2000 | (beam_door_gfx_idx + 2) | beam_pal,
            )?; // beam door tile 2
            self.rom.write_u16(
                tilemap_ptr + 0x0E,
                0x2000 | (beam_door_gfx_idx + 3) | beam_pal,
            )?; // beam door tile 3

            // 16x16 tile 2 (right middle):
            self.rom
                .write_u16(tilemap_ptr + 0x10, 0x2344 | door_pal | flip_y)?; // door frame tile 3
            self.rom
                .write_u16(tilemap_ptr + 0x12, 0x2345 | door_pal | flip_y)?; // door frame tile 2
            self.rom.write_u16(
                tilemap_ptr + 0x14,
                0x2000 | (beam_door_gfx_idx + 4) | beam_pal,
            )?; // beam door tile 4
            self.rom.write_u16(
                tilemap_ptr + 0x16,
                0x2000 | (beam_door_gfx_idx + 5) | beam_pal,
            )?; // beam door tile 5

            // 16x16 tile 3 (right):
            self.rom
                .write_u16(tilemap_ptr + 0x18, 0x2346 | door_pal | flip_y)?; // door frame tile 1
            self.rom
                .write_u16(tilemap_ptr + 0x1A, 0x2347 | door_pal | flip_y)?; // door frame tile 0
            self.rom.write_u16(
                tilemap_ptr + 0x1C,
                0x2000 | (beam_door_gfx_idx + 6) | door_pal,
            )?; // beam door tile 6
            self.rom.write_u16(
                tilemap_ptr + 0x1E,
                0x2000 | (beam_door_gfx_idx + 7) | door_pal,
            )?; // beam door tile 7
        }

        // Idle beam door animation (8x8 tiles): 4 tiles per frame, 4 frame loop:
        // These get copied into VRAM at $2620-$2660:
        for beam_idx in 0..5 {
            // horizontal orientation:
            let gfx_ptr = free_space_addr + beam_idx * 2 * gfx_size + 0x420;
            for frame in 0..4 {
                for tile_idx in 0..4 {
                    let addr = gfx_ptr + frame * 0x80 + tile_idx * 0x20;
                    let tile = idle_beam_tiles[beam_idx][tile_idx][frame];
                    write_tile_4bpp(self.rom, addr, tile)?;
                }
            }

            // vertical orientation:
            let gfx_ptr = free_space_addr + (beam_idx * 2 + 1) * gfx_size + 0x420;
            for frame in 0..4 {
                for tile_idx in 0..4 {
                    let addr = gfx_ptr + frame * 0x80 + tile_idx * 0x20;
                    let tile = diagonal_flip_tile(idle_beam_tiles[beam_idx][tile_idx][frame]);
                    write_tile_4bpp(self.rom, addr, tile)?;
                }
            }
        }

        // Opening beam door animation (8x8 tiles): 8 tiles per frame, 3 frame animation:
        // These get copied into VRAM at $2600-$2680:
        for beam_idx in 0..5 {
            // horizontal orientation:
            let gfx_ptr = free_space_addr + beam_idx * 2 * gfx_size + 0x20;
            for frame in 0..4 {
                for tile_idx in 0..8 {
                    let addr = gfx_ptr + frame * 0x100 + tile_idx * 0x20;
                    let tile = opening_beam_tiles[beam_idx][tile_idx][frame];
                    write_tile_4bpp(self.rom, addr, tile)?;
                }
            }

            // vertical orientation:
            let gfx_ptr = free_space_addr + (beam_idx * 2 + 1) * gfx_size + 0x20;
            for frame in 0..4 {
                for tile_idx in 0..8 {
                    // horizontal orientation:
                    let addr = gfx_ptr + frame * 0x100 + tile_idx * 0x20;
                    let tile = diagonal_flip_tile(opening_beam_tiles[beam_idx][tile_idx][frame]);
                    write_tile_4bpp(self.rom, addr, tile)?;
                }
            }
        }

        Ok(())
    }

    fn write_objective_data(&mut self) -> Result<()> {
        use Objective::*;
        let obj_set: HashSet<Objective> = self.randomization.objectives.iter().copied().collect();
        let bosses: Vec<Objective> = vec![Kraid, Phantoon, Draygon, Ridley]
            .into_iter()
            .filter(|x| obj_set.contains(x))
            .collect();
        let mut minibosses: Vec<Objective> = vec![SporeSpawn, Crocomire, Botwoon, GoldenTorizo]
            .into_iter()
            .filter(|x| obj_set.contains(x))
            .collect();
        let mut chozos: Vec<Objective> =
            vec![BombTorizo, BowlingStatue, AcidChozoStatue, GoldenTorizo]
                .into_iter()
                .filter(|x| obj_set.contains(x))
                .collect();
        let pirates: Vec<Objective> = vec![PitRoom, BabyKraidRoom, PlasmaRoom, MetalPiratesRoom]
            .into_iter()
            .filter(|x| obj_set.contains(x))
            .collect();
        let metroids: Vec<Objective> = vec![MetroidRoom1, MetroidRoom2, MetroidRoom3, MetroidRoom4]
            .into_iter()
            .filter(|x| obj_set.contains(x))
            .collect();

        // Show Golden Torizo under Miniboss if there are other Miniboss objectives; otherwise put it under Chozos:
        if obj_set.contains(&Objective::GoldenTorizo) {
            if minibosses.len() == 1 {
                minibosses = vec![];
            } else {
                chozos = chozos
                    .into_iter()
                    .filter(|x| x != &Objective::GoldenTorizo)
                    .collect();
            }
        }

        let char_mapping: HashMap<char, i16> = vec![
            ('0', 0x2800),
            ('1', 0x2801),
            ('2', 0x2802),
            ('3', 0x2803),
            ('4', 0x2804),
            ('5', 0x2805),
            ('6', 0x2806),
            ('7', 0x2807),
            ('8', 0x2808),
            ('9', 0x2809),
            (' ', 0x280E),
            ('A', 0x28C0),
            ('B', 0x28C1),
            ('C', 0x28C2),
            ('D', 0x28C3),
            ('E', 0x28C4),
            ('F', 0x28C5),
            ('G', 0x28C6),
            ('H', 0x28C7),
            ('I', 0x28C8),
            ('J', 0x28C9),
            ('K', 0x28CA),
            ('L', 0x28CB),
            ('M', 0x28CC),
            ('N', 0x28CD),
            ('O', 0x28CE),
            ('P', 0x28CF),
            ('Q', 0x28D0),
            ('R', 0x28D1),
            ('S', 0x28D2),
            ('T', 0x28D3),
            ('U', 0x28D4),
            ('V', 0x28D5),
            ('W', 0x28D6),
            ('X', 0x28D7),
            ('Y', 0x28D8),
            ('Z', 0x28D9),
            ('.', 0x28DA),
            ('/', 0x28DC),
            ('-', 0x28DD),
            ('?', 0x28DE),
            ('!', 0x28DF),
            (':', 0x2906),
        ]
        .into_iter()
        .collect();

        let mut tile_data = [[char_mapping[&' ']; 30]; 18];
        let mut col = 1;
        let mut row = 1;
        let mut row_max = 1;

        let draw_row = |s: &str,
                        tile_data: &mut [[i16; 30]; 18],
                        row: &mut usize,
                        row_max: &mut usize,
                        col: usize| {
            for (i, c) in s.chars().enumerate() {
                tile_data[*row][col + i] = *char_mapping
                    .get(&c)
                    .context(format!("Unexpected character '{}'", c))
                    .unwrap();
            }
            *row += 1;
            if *row > *row_max {
                *row_max = *row;
            }
        };

        let advance_pos = |row: &mut usize, row_max: &mut usize, col: &mut usize| {
            *row += 1;
            if *row > *row_max {
                *row_max = *row;
            }
            if *row >= 12 {
                *row = 1;
                *col += 15;
            }
        };

        // (row, col) coordinates of the objective checkmark,
        // relative to the top-left of the 30x18 space:
        let mut obj_coords: HashMap<(usize, usize), Objective> = HashMap::new();

        if bosses.len() > 0 {
            draw_row("BOSSES:", &mut tile_data, &mut row, &mut row_max, col);
            for obj in bosses {
                obj_coords.insert((row, col), obj);
                match obj {
                    Kraid => draw_row("- KRAID", &mut tile_data, &mut row, &mut row_max, col),
                    Phantoon => draw_row("- PHANTOON", &mut tile_data, &mut row, &mut row_max, col),
                    Draygon => draw_row("- DRAYGON", &mut tile_data, &mut row, &mut row_max, col),
                    Ridley => draw_row("- RIDLEY", &mut tile_data, &mut row, &mut row_max, col),
                    _ => panic!("unexpected objective: {:?}", obj),
                }
            }
            advance_pos(&mut row, &mut row_max, &mut col);
        }

        if minibosses.len() > 0 {
            draw_row("MINIBOSSES:", &mut tile_data, &mut row, &mut row_max, col);
            for obj in minibosses {
                obj_coords.insert((row, col), obj);
                match obj {
                    SporeSpawn => {
                        draw_row("- SPORE SPAWN", &mut tile_data, &mut row, &mut row_max, col)
                    }
                    Crocomire => {
                        draw_row("- CROCOMIRE", &mut tile_data, &mut row, &mut row_max, col)
                    }
                    Botwoon => draw_row("- BOTWOON", &mut tile_data, &mut row, &mut row_max, col),
                    GoldenTorizo => {
                        draw_row("- GOLD TORIZO", &mut tile_data, &mut row, &mut row_max, col)
                    }
                    _ => panic!("unexpected objective: {:?}", obj),
                }
            }
            advance_pos(&mut row, &mut row_max, &mut col);
        }

        if chozos.len() > 0 {
            draw_row("CHOZOS:", &mut tile_data, &mut row, &mut row_max, col);
            for obj in chozos {
                obj_coords.insert((row, col), obj);
                match obj {
                    BombTorizo => {
                        draw_row("- BOMB TORIZO", &mut tile_data, &mut row, &mut row_max, col)
                    }
                    BowlingStatue => {
                        draw_row("- BOWLING", &mut tile_data, &mut row, &mut row_max, col)
                    }
                    AcidChozoStatue => {
                        draw_row("- ACID STATUE", &mut tile_data, &mut row, &mut row_max, col)
                    }
                    GoldenTorizo => {
                        draw_row("- GOLD TORIZO", &mut tile_data, &mut row, &mut row_max, col)
                    }
                    _ => panic!("unexpected objective: {:?}", obj),
                }
            }
            advance_pos(&mut row, &mut row_max, &mut col);
        }

        if pirates.len() > 0 {
            draw_row("PIRATES:", &mut tile_data, &mut row, &mut row_max, col);
            for obj in pirates {
                obj_coords.insert((row, col), obj);
                match obj {
                    PitRoom => draw_row("- PIT ROOM", &mut tile_data, &mut row, &mut row_max, col),
                    BabyKraidRoom => {
                        draw_row("- BABY KRAID", &mut tile_data, &mut row, &mut row_max, col)
                    }
                    PlasmaRoom => draw_row("- PLASMA", &mut tile_data, &mut row, &mut row_max, col),
                    MetalPiratesRoom => {
                        draw_row("- METAL", &mut tile_data, &mut row, &mut row_max, col)
                    }
                    _ => panic!("unexpected objective: {:?}", obj),
                }
            }
            advance_pos(&mut row, &mut row_max, &mut col);
        }

        if metroids.len() > 0 {
            row = row_max;
            col = 1;
            draw_row("METROIDS:", &mut tile_data, &mut row, &mut row_max, col);
            let mut i = 0;
            for obj in metroids {
                obj_coords.insert((row, col + i * 3), obj);
                tile_data[row][col + i * 3] = char_mapping[&'-'];
                let num = match obj {
                    MetroidRoom1 => '1',
                    MetroidRoom2 => '2',
                    MetroidRoom3 => '3',
                    MetroidRoom4 => '4',
                    _ => panic!("unexpected objective: {:?}", obj),
                };
                tile_data[row][col + i * 3 + 1] = char_mapping[&num];
                i += 1;
            }
            row += 1;
            advance_pos(&mut row, &mut row_max, &mut col);
        }

        if self.randomization.settings.save_animals == SaveAnimals::Yes {
            row = row_max;
            col = 1;
            draw_row(
                "SAVE THE ANIMALS!",
                &mut tile_data,
                &mut row,
                &mut row_max,
                col,
            );
        } else if self.randomization.settings.save_animals == SaveAnimals::Random {
            row = row_max;
            col = 1;
            draw_row(
                "SAVE THE ANIMALS?",
                &mut tile_data,
                &mut row,
                &mut row_max,
                col,
            );
        }

        let mut addr = snes2pc(0xB6F200);
        for row in 0..18 {
            for &c in &tile_data[row] {
                self.rom.write_u16(addr, c as isize)?;
                addr += 2;
            }
            self.rom.write_u16(addr, 0x8000)?; // line terminator
            addr += 2;
        }

        assert!(addr < snes2pc(0xB6F660));

        if self
            .randomization
            .settings
            .quality_of_life_settings
            .fast_pause_menu
        {
            self.rom.write_u16(
                snes2pc(0x82fffc),
                0x8000 | self.randomization.objectives.len() as isize,
            )?;
        } else {
            self.rom.write_u16(
                snes2pc(0x82fffc),
                self.randomization.objectives.len() as isize,
            )?;
        }

        // Sort the coordinates of objective checkboxes in row-major order, to correspond
        // with the order that the mb_barrier_clear.asm expects.
        let mut obj_coords_vec: Vec<(usize, usize)> = obj_coords.keys().copied().collect();
        assert!(obj_coords_vec.len() == self.randomization.objectives.len());
        obj_coords_vec.sort();
        let mut obj_i = 0;
        for coords in &obj_coords_vec {
            let obj = obj_coords[coords];
            let (addr, mask) = match obj {
                Kraid => (0xD829, 1),
                Ridley => (0xD82A, 1),
                Phantoon => (0xD82B, 1),
                Draygon => (0xD82C, 1),
                SporeSpawn => (0xD829, 2),
                Crocomire => (0xD82A, 2),
                Botwoon => (0xD82C, 2),
                GoldenTorizo => (0xD82A, 4),
                MetroidRoom1 => (0xD822, 1),
                MetroidRoom2 => (0xD822, 2),
                MetroidRoom3 => (0xD822, 4),
                MetroidRoom4 => (0xD822, 8),
                BombTorizo => (0xD828, 4),
                BowlingStatue => (0xD823, 1),
                AcidChozoStatue => (0xD821, 0x10),
                PitRoom => (0xD823, 2),
                BabyKraidRoom => (0xD823, 4),
                PlasmaRoom => (0xD823, 8),
                MetalPiratesRoom => (0xD823, 0x10),
            };
            self.rom.write_u16(snes2pc(0x8FEBC0) + obj_i * 2, addr)?;
            self.rom.write_u16(snes2pc(0x8FEBE8) + obj_i * 2, mask)?;
            obj_i += 1;
        }

        Ok(())
    }
}

fn get_other_door_ptr_pair_map(map: &Map) -> HashMap<DoorPtrPair, DoorPtrPair> {
    let mut other_door_ptr_pair_map: HashMap<DoorPtrPair, DoorPtrPair> = HashMap::new();
    for (src_door_ptr_pair, dst_door_ptr_pair, _bidirectional) in &map.doors {
        other_door_ptr_pair_map.insert(src_door_ptr_pair.clone(), dst_door_ptr_pair.clone());
        other_door_ptr_pair_map.insert(dst_door_ptr_pair.clone(), src_door_ptr_pair.clone());
    }
    other_door_ptr_pair_map
}

pub fn make_rom(
    base_rom: &Rom,
    randomization: &Randomization,
    game_data: &GameData,
) -> Result<Rom> {
    let mut orig_rom = base_rom.clone();
    apply_orig_ips_patches(&mut orig_rom, &randomization.settings)?;

    // Remove solid wall that spawns in Tourian Escape Room 1 while coming through right door.
    // Note that this wall spawns in two ways: 1) as a normal PLM which spawns when entering through either door
    // (and which we remove in `remove_non_blue_doors`), and 2) as a door cap closing when coming in from the right
    // (removed here). Both of these have to be removed in order to successfully get rid of this wall.
    // (The change has to be applied to the original ROM before doors are reconnected based on the randomized map.)
    orig_rom.write_u8(snes2pc(0x83AA8F), 0x01)?; // Door direction = 0x01

    // Even though there is no door cap closing animation, we need to move the door cap X out of the way to the left,
    // otherwise corrupts the hazard marker PLM somehow:
    orig_rom.write_u8(snes2pc(0x83AA90), 0x1E)?; // Door cap X = 0x1E

    let mut rom = orig_rom.clone();
    let mut patcher = Patcher {
        orig_rom: &mut orig_rom,
        rom: &mut rom,
        randomization,
        game_data,
        map: &randomization.map,
        other_door_ptr_pair_map: get_other_door_ptr_pair_map(&randomization.map),
        extra_setup_asm: HashMap::new(),
        locked_door_state_indices: vec![],
        nothing_item_bitmask: [0; 0x40],
        map_reveal_bitmasks: vec![vec![]; NUM_AREAS],
    };
    patcher.apply_ips_patches()?;
    patcher.place_items()?;
    patcher.set_start_location()?;
    patcher.set_starting_items()?;
    patcher.fix_save_stations()?;
    patcher.write_map_areas()?;
    patcher.make_map_revealed()?;
    patcher.write_beam_door_tiles()?;
    patcher.apply_locked_doors()?;
    patcher.apply_map_tile_patches()?;
    patcher.write_door_data()?;
    patcher.write_map_reveal_tiles()?;
    patcher.remove_non_blue_doors()?;
    if randomization.settings.map_layout != "Vanilla"
        || randomization.settings.other_settings.area_assignment == AreaAssignment::Random
    {
        patcher.use_area_based_music()?;
    }
    patcher.setup_door_specific_fx()?;
    if !randomization.settings.other_settings.ultra_low_qol {
        patcher.setup_reload_cre()?;
    }
    patcher.fix_twin_rooms()?;
    patcher.fix_crateria_scrolling_sky()?;
    patcher.apply_title_screen_patches()?;
    patcher.customize_escape_timer()?;
    patcher.apply_miscellaneous_patches()?;
    patcher.apply_mother_brain_fight_patches()?;
    patcher.write_walljump_item_graphics()?;
    patcher.write_objective_data()?;
    patcher.apply_seed_identifiers()?;
    patcher.apply_credits()?;
    if !randomization.settings.other_settings.ultra_low_qol {
        patcher.apply_hazard_markers()?;
    }
    if randomization
        .settings
        .quality_of_life_settings
        .room_outline_revealed
    {
        patcher.apply_all_room_outlines()?;
    }
    patcher.apply_toilet_data()?;
    patcher.apply_extra_setup_asm()?;
    Ok(rom)
}
