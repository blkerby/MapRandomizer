pub mod compress;
pub mod decompress;
pub mod ips_write;
mod map_tiles;
mod title;

use std::path::Path;

use crate::{
    game_data::{DoorPtr, DoorPtrPair, GameData, Item, Map, NodePtr, RoomGeometryDoor},
    randomize::{MotherBrainFight, Objectives, Randomization, SpoilerItemSummary},
};
use anyhow::{ensure, Context, Result};
use hashbrown::{HashMap, HashSet};
use ips;
use std::iter;

const NUM_AREAS: usize = 6;

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
    touched: Vec<bool>,
}

impl Rom {
    pub fn new(data: Vec<u8>) -> Self {
        let len = data.len();
        Rom {
            data,
            touched: vec![false; len],
        }
    }

    pub fn resize(&mut self, new_size: usize) {
        self.data.resize(new_size, 0xFF);
        self.touched.resize(new_size, false);
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
        // if self.touched[addr] {
        //     println!("Rewritten: {:x}", pc2snes(addr));
        // }
        self.touched[addr] = true;
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
        Ok(())
    }
}

pub struct Patcher<'a> {
    pub orig_rom: &'a mut Rom,
    pub rom: &'a mut Rom,
    pub randomization: &'a Randomization,
    pub game_data: &'a GameData,
    pub map: &'a Map,
    pub other_door_ptr_pair_map: HashMap<DoorPtrPair, DoorPtrPair>,
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
    let old_item_id = ((orig_plm_type - 0xEED7) / 4) % 21;
    orig_plm_type + (item_id - old_item_id) * 4
}

fn apply_ips_patch(rom: &mut Rom, patch_path: &Path) -> Result<()> {
    let patch_data = std::fs::read(&patch_path)
        .with_context(|| format!("Unable to read patch {}", patch_path.display()))?;
    let patch = ips::Patch::parse(&patch_data)
        .with_context(|| format!("Unable to parse patch {}", patch_path.display()))?;
    for hunk in patch.hunks() {
        rom.write_n(hunk.offset(), hunk.payload())?;
    }
    Ok(())
}

fn apply_orig_ips_patches(rom: &mut Rom, randomization: &Randomization) -> Result<()> {
    let patches_dir = Path::new("../patches/ips/");
    let mut patches: Vec<&'static str> = vec!["mb_barrier", "mb_barrier_clear", "gray_doors"];
    // if randomization.difficulty.ultra_low_qol {
    //     patches.push("ultra_low_qol_hud_expansion_opaque");
    // } else {
    patches.push("hud_expansion_opaque");
    // }
    for patch_name in patches {
        let patch_path = patches_dir.join(patch_name.to_string() + ".ips");
        apply_ips_patch(rom, &patch_path)?;
    }

    // Overwrite door ASM for entering Mother Brain room from right, used for clearing objective barriers:
    match randomization.difficulty.objectives {
        Objectives::Bosses => {}
        Objectives::Minibosses => {
            rom.write_u16(snes2pc(0x83AAD2), 0xEB60)?;
        }
        Objectives::Metroids => {
            rom.write_u16(snes2pc(0x83AAD2), 0xEBC0)?;
        }
        Objectives::Chozos => {
            rom.write_u16(snes2pc(0x83AAD2), 0xEC20)?;
        }
        Objectives::Pirates => {
            rom.write_u16(snes2pc(0x83AAD2), 0xEC80)?;
        }
    }
    Ok(())
}

impl<'a> Patcher<'a> {
    fn apply_ips_patches(&mut self) -> Result<()> {
        self.rom.data.resize(0x400000, 0);
        let patches_dir = Path::new("../patches/ips/");
        let mut patches = vec![
            "everest_tube",
            "sandfalls3",
            "complementary_suits",
            "disable_map_icons",
            "escape",
            "tourian_map",
            "tourian_eye_door",
            "no_explosions_before_escape",
            "escape_room_1",
            "sound_effect_disables",
            "title_map_animation",
            "shaktool",
            "fix_water_fx_bug",
            "crateria_tube_black",
            "seed_hash_display",
            "max_ammo_display",
            "stats",
            "credits",
            "sram_check_disable",
        ];

        if !self.randomization.difficulty.vanilla_map {
            patches.push("music");
        }

        if self.randomization.difficulty.ultra_low_qol {
            patches.extend([
                "ultra_low_qol_vanilla_bugfixes",
                "ultra_low_qol_new_game",
                "ultra_low_qol_saveload",
                "ultra_low_qol_map_area",
            ]);
        } else {
            patches.extend([
                "vanilla_bugfixes",
                "saveload",
                "itemsounds",
                "missile_refill_all",
                "decompression",
                "aim_anything",
                "fast_reload",
                "fast_saves",
                "fast_mother_brain_cutscene",
                "fast_big_boy_cutscene",
                "unexplore",
                "fix_kraid_vomit",
                "escape_autosave",
                "map_area",
                "map_progress_maintain",
                "tourian_blue_hopper",
                "boss_exit",
                "oob_death",
            ]);

            let mut new_game = "new_game";
            if let Some(options) = &self.randomization.difficulty.debug_options {
                if options.new_game_extra {
                    new_game = "new_game_extra";
                }
            }
            patches.push(new_game);
        }

        if self.randomization.difficulty.all_items_spawn {
            patches.push("all_items_spawn");
        }

        if self.randomization.difficulty.escape_movement_items {
            patches.push("escape_items");
            // patches.push("mother_brain_no_drain");
        }

        if self.randomization.difficulty.fast_elevators {
            patches.push("elevators_speed");
        }

        if self.randomization.difficulty.fast_doors {
            patches.push("fast_doors");
        }

        if self.randomization.difficulty.fast_pause_menu {
            patches.push("fast_pause_menu");
        }

        if self.randomization.difficulty.item_dots_disappear {
            patches.push("item_dots_disappear");
        }

        if self.randomization.difficulty.disable_walljump {
            patches.push("disable_walljump");
        }

        if self.randomization.difficulty.respin {
            patches.push("spinjumprestart");
        }

        for patch_name in patches {
            let patch_path = patches_dir.join(patch_name.to_string() + ".ips");
            apply_ips_patch(&mut self.rom, &patch_path)?;
        }
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

        if !self.randomization.difficulty.ultra_low_qol {
            // Mark as revealed (which will persist after deaths/reloads):
            let addr = 0x2000 + (tile_area as isize) * 0x100 + offset;
            asm.extend([0xAF, (addr & 0xFF) as u8, (addr >> 8) as u8, 0x70]); // LDA $70:{addr}
            asm.extend([0x09, bitmask, 0x00]); // ORA #{bitmask}
            asm.extend([0x8F, (addr & 0xFF) as u8, (addr >> 8) as u8, 0x70]); // STA $70:{addr}
        }
        // Mark as explored (for elevators. Not needed for area transition arrows/letters except in ultra-low QoL mode):
        if explore || self.randomization.difficulty.ultra_low_qol {
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
        // let entrance_x = self.orig_rom.read_u8(other_door_pair.1.unwrap() + 4)? as u8;
        // let entrance_y = self.orig_rom.read_u8(other_door_pair.1.unwrap() + 5)? as u8;
        // println!("entrance: {} {}", entrance_x, entrance_y);
        let screen_x = self.orig_rom.read_u8(other_door_pair.1.unwrap() + 6)? as u8;
        let screen_y = self.orig_rom.read_u8(other_door_pair.1.unwrap() + 7)? as u8;
        let entrance_x = screen_x * 16 + 14;
        let entrance_y = screen_y * 16 + 6;

        let asm: Vec<u8> = vec![
            0xA9, 0x0E, 0x00, // LDA #$000E   (Escape flag)
            0x22, 0x33, 0x82, 0x80, // JSL $808233  (Check if flag is set)
            0x90, 0x09, // BCC $09  (Skip spawning gray door if not in escape)
            0x22, 0x80, 0xF3, 0x84, // JSL $84F380  (Spawn hard-coded PLM with room argument)
            entrance_x, entrance_y, 0x42, 0xC8,
            0x10, // PLM type 0xC8CA (gray door), argument 0x10 (always closed)
        ];

        extra_door_asm
            .entry(mother_brain_left_door_pair.0.unwrap())
            .or_default()
            .extend(asm);

        Ok(())
    }

    // This doesn't work (changing enemy set in door ASM):
    // fn fix_tourian_blue_hopper(&mut self, extra_door_asm: &mut HashMap<DoorPtr, Vec<u8>>) -> Result<()> {
    //     let door_pair = (Some(0x1AA14), Some(0x1AA20));
    //     let other_door_pair = self.other_door_ptr_pair_map[&door_pair];
    //     // When entering from the left, switch to alternative enemy set that moves the
    //     // leftmost Hopper to the right. See `tourian_blue_hopper.asm` which creates the new enemy set.
    //     let asm: Vec<u8> = vec![
    //         // 0xa9, 0x00, 0xf2,  // lda #$F200  ;\
    //         0xa9, 0xa9, 0x85,
    //        0x8d, 0xcf, 0x07,  // sta $07CF   ;} Enemy set pointer = $A1F200
    //     ];
    //     extra_door_asm.entry(other_door_pair.0.unwrap()).or_default().extend(asm);
    //     Ok(())
    // }

    // Returns map from door data PC address to 1) new custom door ASM pointer, 2) end of custom door ASM
    // where an RTS or JMP instruction must be added (based on the connecting door).
    fn prepare_extra_door_asm(&mut self) -> Result<HashMap<DoorPtr, (AsmPtr, AsmPtr)>> {
        let toilet_exit_asm: Vec<u8> = vec![0x20, 0x01, 0xE3]; // JSR 0xE301
        let boss_exit_asm: Vec<u8> = vec![0x20, 0xF0, 0xF7]; // JSR 0xF7F0
        let mut extra_door_asm: HashMap<DoorPtr, Vec<u8>> = HashMap::new();
        extra_door_asm.insert(0x1A600, toilet_exit_asm.clone()); // Aqueduct toilet door down
        extra_door_asm.insert(0x1A60C, toilet_exit_asm.clone()); // Aqueduct toilet door up
        if !self.randomization.difficulty.ultra_low_qol {
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
        // self.fix_tourian_blue_hopper(&mut extra_door_asm)?;

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
        assert!(door_asm_free_space <= 0xF500);
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
        // Make the whole map revealed (after the player uses the map station) -- no more hidden tiles.
        for i in 0x11727..0x11D27 {
            self.rom.write_u8(i, 0xFF)?;
        }
        Ok(())
    }

    fn write_map_tilemaps(&mut self) -> Result<()> {
        // Patch Aqueduct map Y position to include the toilet, for the purposes of determining
        // the map. We change it back later in `fix_twin_rooms`.
        self.orig_rom
            .write_u8(0x7D5A7 + 3, self.orig_rom.read_u8(0x7D5A7 + 3)? - 4)?;

        // Determine upper-left corner of each area:
        let mut area_map_min_x = [isize::MAX; NUM_AREAS];
        let mut area_map_max_x = [0; NUM_AREAS];
        let mut area_map_min_y = [isize::MAX; NUM_AREAS];
        let mut area_map_max_y = [0; NUM_AREAS];
        for i in 0..self.map.area.len() {
            let area = self.map.area[i];
            let x0 = self.map.rooms[i].0 as isize;
            let y0 = self.map.rooms[i].1 as isize;
            let x1 = self.map.rooms[i].0 as isize
                + self.game_data.room_geometry[i].map[0].len() as isize;
            let y1 =
                self.map.rooms[i].1 as isize + self.game_data.room_geometry[i].map.len() as isize;
            if x0 < area_map_min_x[area] {
                area_map_min_x[area] = x0;
            }
            if x1 > area_map_max_x[area] {
                area_map_max_x[area] = x1;
            }
            if y0 < area_map_min_y[area] {
                area_map_min_y[area] = y0;
            }
            if y1 > area_map_max_y[area] {
                area_map_max_y[area] = y1;
            }
        }

        // Clear all map tilemap data:
        for area_ptr in &self.game_data.area_map_ptrs {
            for i in 0..(64 * 32) {
                self.rom.write_u16((area_ptr + i * 2) as usize, 0x001F)?;
            }
        }

        // Write new map tilemap data (and room X & Y map position) by room:
        for (i, room) in self.game_data.room_geometry.iter().enumerate() {
            let orig_area = self.orig_rom.read_u8(room.rom_address + 1)? as usize;
            let orig_base_x = self.orig_rom.read_u8(room.rom_address + 2)?;
            let orig_base_y = self.orig_rom.read_u8(room.rom_address + 3)?;
            let orig_base_ptr = self.game_data.area_map_ptrs[orig_area];
            let new_area = self.map.area[i];
            let new_base_ptr = self.game_data.area_map_ptrs[new_area];
            let new_margin_x = (64 - (area_map_max_x[new_area] - area_map_min_x[new_area])) / 2;
            let new_margin_y = (32 - (area_map_max_y[new_area] - area_map_min_y[new_area])) / 2 - 1;
            let new_base_x = self.map.rooms[i].0 as isize - area_map_min_x[new_area] + new_margin_x;
            let new_base_y = self.map.rooms[i].1 as isize - area_map_min_y[new_area] + new_margin_y;
            assert!(new_base_x >= 2);
            assert!(new_base_y >= 0);
            self.rom.write_u8(room.rom_address + 2, new_base_x)?;
            self.rom.write_u8(room.rom_address + 3, new_base_y)?;
            for y in 0..room.map.len() {
                for x in 0..room.map[0].len() {
                    if room.map[y][x] == 0 {
                        continue;
                    }
                    let orig_x = orig_base_x + x as isize;
                    let orig_y = orig_base_y + y as isize;
                    let orig_offset = xy_to_map_offset(orig_x as isize, orig_y as isize);
                    let orig_ptr = (orig_base_ptr + orig_offset) as usize;
                    let new_x = new_base_x + x as isize;
                    let new_y = new_base_y + y as isize;
                    let new_offset = xy_to_map_offset(new_x, new_y);
                    let new_ptr = (new_base_ptr + new_offset) as usize;
                    let data = self.orig_rom.read_u16(orig_ptr)?;
                    self.rom.write_u16(new_ptr, data)?;
                }
            }
        }
        Ok(())
    }

    fn fix_twin_rooms(&mut self) -> Result<()> {
        // Fix map X & Y of Aqueduct and twin rooms:
        let old_aqueduct_x = self.rom.read_u8(0x7D5A7 + 2)?;
        let old_aqueduct_y = self.rom.read_u8(0x7D5A7 + 3)?;
        self.rom.write_u8(0x7D5A7 + 3, old_aqueduct_y + 4)?;
        // Toilet:
        self.rom.write_u8(0x7D408 + 2, old_aqueduct_x + 2)?;
        self.rom.write_u8(0x7D408 + 3, old_aqueduct_y)?;
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
        let aqueduct_room_idx = self.game_data.room_idx_by_name["Aqueduct"];
        room_index_area_hashmaps[4].insert(0x18, self.map.area[aqueduct_room_idx]); // Set Toilet to same map area as Aqueduct
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

    // Returns list of (event_ptr, state_ptr):
    fn get_room_state_ptrs(&self, room_ptr: usize) -> Result<Vec<(usize, usize)>> {
        let mut pos = 11;
        let mut ptr_pairs: Vec<(usize, usize)> = Vec::new();
        loop {
            let ptr = self.rom.read_u16(room_ptr + pos)? as usize;
            if ptr == 0xE5E6 {
                // This is the standard state, which is the last one.
                ptr_pairs.push((ptr, room_ptr + pos + 2));
                return Ok(ptr_pairs);
            } else if ptr == 0xE612 || ptr == 0xE629 {
                // This is an event state.
                let state_ptr = 0x70000 + self.rom.read_u16(room_ptr + pos + 3)?;
                ptr_pairs.push((ptr, state_ptr as usize));
                pos += 5;
            } else {
                // This is another kind of state.
                let state_ptr = 0x70000 + self.rom.read_u16(room_ptr + pos + 2)?;
                ptr_pairs.push((ptr, state_ptr as usize));
                pos += 4;
            }
        }
    }

    fn apply_map_tile_patches(&mut self) -> Result<()> {
        map_tiles::MapPatcher::new(&mut self.rom, self.game_data, self.map, self.randomization)
            .apply_patches()?;
        Ok(())
    }

    fn remove_non_blue_doors(&mut self) -> Result<()> {
        let plm_types_to_remove = vec![
            0xC88A, 0xC85A, 0xC872, // right pink/yellow/green door
            0xC890, 0xC860, 0xC878, // left pink/yellow/green door
            0xC896, 0xC866, 0xC87E, // down pink/yellow/green door
            0xC89C, 0xC86C, 0xC884, // up pink/yellow/green door
            0xDB48, 0xDB4C, 0xDB52, 0xDB56, 0xDB5A,
            0xDB60, // eye doors
                    // 0xC8CA, // wall in Escape Room 1 (TODO: Check if this is needed)
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
            let event_state_ptrs = self.get_room_state_ptrs(room.rom_address)?;
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
                        self.rom.write_u16(ptr, 0xB63B)?; // right continuation arrow (should have no effect, giving a blue door)
                        self.rom.write_u16(ptr + 2, 0)?; // position = (0, 0)
                    } else if gray_door_plm_types.contains_key(&plm_type) {
                        let new_type = gray_door_plm_types[&plm_type];
                        self.rom.write_u16(ptr, new_type)?;
                    }
                    ptr += 6;
                }
            }
        }
        Ok(())
    }

    fn use_area_based_music(&mut self) -> Result<()> {
        let area_music: [[u16; 2]; NUM_AREAS] = [
            [
                // (0x06, 0x05),   // Empty Crateria
                0x0509, // Crateria Space Pirates
                0x050C, // Return to Crateria
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
                0x0530, // Wrecked Ship (off)
                0x0630, // Wrecked Ship (on)
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
            "Blue Hopper Room",
        ]
        .map(|x| x.to_string());
        for (room_idx, room) in self.game_data.room_geometry.iter().enumerate() {
            if rooms_to_leave_unchanged.contains(&room.name) {
                continue;
            }
            let area = self.map.area[room_idx];
            let subarea = self.map.subarea[room_idx];
            let event_state_ptrs = self.get_room_state_ptrs(room.rom_address)?;
            for &(_event_ptr, state_ptr) in &event_state_ptrs {
                let song = self.rom.read_u16(state_ptr + 4)? as u16;
                if songs_to_keep.contains(&song) && room.name != "Golden Torizo Energy Recharge" {
                    // In vanilla, Golden Torizo Energy Recharge plays the item/elevator music,
                    // but that only seems to be because of it being next to Screw Attack Room.
                    // We want it to behave like the other Refill rooms and use area-themed music.
                    continue;
                }
                let mut new_song = area_music[area][subarea];
                if room.name == "Landing Site" {
                    // Set all Landing Site states to use the same track, the one that plays in vanilla before
                    // Power Bombs but after Zebes is awake:
                    new_song = 0x0606;
                }
                self.rom.write_u16(state_ptr + 4, new_song as isize)?;
                if room.name == "Pants Room" {
                    // Set music for East Pants Room:
                    self.rom
                        .write_u16(snes2pc(0x8FD6A7 + 4), new_song as isize)?;
                } else if room.name == "West Ocean" {
                    // Set music for Homing Geemer Room:
                    self.rom
                        .write_u16(snes2pc(0x8F969C + 4), new_song as isize)?;
                } else if room.name == "Aqueduct" {
                    // Set music for Toilet:
                    self.rom
                        .write_u16(snes2pc(0x8FD415 + 4), new_song as isize)?;
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

        // We skip applying this to Climb, because otherwise the lava would rise every time when entering
        // the bottom-left door (not only during the escape):
        // door_fx.insert((0x18B6E, 0x1AB34), 0x838060);  // Climb bottom-left door: lava rising

        // Even still, as in the vanilla game, lava would rise in Climb if entered through Tourian Escape Room 4
        // (even not in the escape). We prevent this possibility by replacing the Tourian Escape Room 4 door
        // with the value 0xFFFF which does not match any door, effectively disabling the door-specific FX
        // for this room. Note that lava will still rise in Climb during the escape because of the different
        // room state (unrelated to door-specific FX).
        self.rom.write_u16(snes2pc(0x838060), 0xffff)?;

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
        if self.randomization.difficulty.supers_double {
            // Make Supers do double damage to Mother Brain:
            self.rom.write_u8(snes2pc(0xB4F1D5), 0x84)?;
        }

        match self.randomization.difficulty.mother_brain_fight {
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

                if self.randomization.difficulty.escape_movement_items {
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

                if self.randomization.difficulty.escape_movement_items {
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

        // Stop wall from spawning in Tourian Escape Room 1:
        self.rom.write_u16(snes2pc(0x84BB34), 0x86BC)?;
        self.rom.write_u16(snes2pc(0x84BB44), 0x86BC)?;
        // Alternative way to stop the wall from spawning, but would need to be applied earlier to the
        // original ROM before the door data was rewritten:
        // self.rom.write_u8(snes2pc(0x83AA8F), 0x04)?;  // Door direction = 0x04

        // Restore acid in Tourian Escape Room 4:
        self.rom.write_u16(snes2pc(0x8FDF03), 0xC953)?; // Vanilla setup ASM pointer (to undo effect of `no_explosions_before_escape` patch)
        self.rom.write_u8(snes2pc(0x8FC95B), 0x60)?; // RTS (return early from setup ASM to skip setting up shaking)

        // Remove fake gray door that gets drawn in Phantoon's Room:
        self.rom.write_n(snes2pc(0xA7D4E5), &vec![0xEA; 8])?;

        if self.randomization.difficulty.all_items_spawn {
            // Copy the item in Pit Room to the Zebes-asleep state.
            // For this we overwrite the PLM slot for the gray door at the left of the room (which we would get rid of anyway).
            let plm_data = self.rom.read_n(0x783EE, 6)?.to_vec();
            self.rom.write_n(0x783C8, &plm_data)?;
        }

        if self.randomization.difficulty.acid_chozo {
            // Remove Space Jump check
            self.rom.write_n(snes2pc(0x84D195), &[0xEA, 0xEA])?; // NOP : NOP
        }

        if self.randomization.difficulty.infinite_space_jump {
            // self.rom.write_n(0x82493, &[0x80, 0x0D])?;  // BRA $0D  (Infinite Space Jump)
            self.rom.write_n(snes2pc(0x90A493), &[0xEA, 0xEA])?; // NOP : NOP  (Lenient Space Jump)
        }

        if !self.randomization.difficulty.ultra_low_qol {
            // In Crocomire's initialization, skip setting the leftmost screens to red scroll. Even in the vanilla game there
            // is no purpose to this, as they are already red. But it important to skip here in the rando, because when entering
            // from the left door with Crocomire still alive, these scrolls are set to blue by the door ASM, and if they
            // were overridden with red it would break the graphics.
            self.rom.write_n(snes2pc(0xA48A92), &vec![0xEA; 4])?; // NOP:NOP:NOP:NOP

            // Release Spore Spawn camera so it won't be glitched when entering from the right:
            self.rom.write_n(snes2pc(0xA5EADA), &vec![0xEA; 3])?; // NOP:NOP:NOP

            // Likewise release Kraid camera so it won't be as glitched when entering from the right:
            self.rom.write_n(snes2pc(0xA7A9F4), &vec![0xEA; 4])?; // NOP:NOP:NOP:NOP

            // Fix the door cap X location for the Green Brinstar Main Shaft door to itself left-to-right:
            self.rom.write_u8(snes2pc(0x838CF2), 0x11)?;
        }

        if self.randomization.difficulty.save_animals {
            // Change end-game behavior to require saving the animals. Address here must match escape.asm:
            self.rom.write_u16(snes2pc(0xA1F000), 0xFFFF)?;
        }

        if self.randomization.difficulty.escape_enemies_cleared {
            // Change escape behavior to clear enemies. Address here must match escape.asm:
            self.rom.write_u16(snes2pc(0xA1f004), 0xFFFF)?;
        }

        if !self.randomization.difficulty.escape_refill {
            // Disable the energy refill at the start of the escape. Address here must match escape.asm:
            self.rom.write_u16(snes2pc(0xA1F002), 0x0001)?;
        }

        Ok(())
    }

    fn apply_title_screen_patches(&mut self) -> Result<()> {
        let mut title_patcher = title::TitlePatcher::new(&mut self.rom);
        title_patcher.patch_title_background()?;
        title_patcher.patch_title_foreground()?;
        Ok(())
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

    fn apply_seed_hash(&mut self) -> Result<()> {
        let seed_bytes = (self.randomization.display_seed as u32).to_le_bytes();
        self.rom.write_n(snes2pc(0xdfff00), &seed_bytes)?;
        Ok(())
    }

    fn write_letter(&mut self, letter: char, addr: usize) -> Result<()> {
        if letter <= 'P' {
            self.rom.write_u16(addr, letter as isize - 'A' as isize + 0x0020)?;
            self.rom.write_u16(addr + 0x40, letter as isize - 'A' as isize + 0x0030)?;
        } else {
            self.rom.write_u16(addr, letter as isize - 'Q' as isize + 0x0040)?;
            self.rom.write_u16(addr + 0x40, letter as isize - 'Q' as isize + 0x0050)?;
        }
        Ok(())
    }

    fn write_digit(&mut self, digit: usize, addr: usize) -> Result<()> {
        self.rom.write_u16(addr, digit as isize + 0x0060)?;
        self.rom.write_u16(addr + 0x40, digit as isize + 0x0070)?;
        Ok(())
    }

    fn write_item_credits(
        &mut self,
        idx: usize,
        step: usize,
        item: &str,
        item_idx: usize,
        area: &str,
    ) -> Result<()> {
        let base_addr = snes2pc(0xceb240 + (164 - 128 + idx * 2) * 0x40);
        println!("{}: {} {:x}", idx, item, base_addr);

        // Write step number
        if step >= 10 {
            self.write_digit(step / 10, base_addr + 2)?;
        }
        self.write_digit(step % 10, base_addr + 4)?;

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

        // Write stats address for collection time
        let stats_table_addr = snes2pc(0xdfdf80);
        let item_time_addr = 0xfe06;
        self.rom.write_u16(
            stats_table_addr + idx * 8,
            (item_time_addr + 4 * item_idx) as isize,
        )?;
        Ok(())
    }

    fn write_preset(&mut self, row: usize, preset: Option<String>) -> Result<()> {
        let preset = preset.unwrap_or("Custom".to_string());
        let base_addr = snes2pc(0xceb240 + (row - 128) * 0x40);
        println!("{}: {}", row, preset);
        for (i, c) in preset.chars().enumerate() {
            let c = c.to_ascii_uppercase();
            if c >= 'A' && c <= 'Z' {
                self.write_letter(c, base_addr + 0x3E - preset.len() * 2 + i * 2)?;
            }
        }
        Ok(())
    }

    fn apply_credits(&mut self) -> Result<()> {
        // Write randomizer settings to credits tilemap
        self.write_preset(222, self.randomization.difficulty.skill_assumptions_preset.clone())?;
        self.write_preset(224, self.randomization.difficulty.item_progression_preset.clone())?;
        self.write_preset(226, self.randomization.difficulty.quality_of_life_preset.clone())?;

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

        // Write item locations in credits tilemap
        let mut items_set: HashSet<String> = HashSet::new();
        for (step, step_summary) in self.randomization.spoiler_log.summary.iter().enumerate() {
            for item_info in step_summary.items.iter() {
                if !items_set.contains(&item_info.item) {
                    let item_name = item_display_name_map[&item_info.item].clone();
                    let item_idx = item_name_index[&item_info.item];
                    self.write_item_credits(
                        items_set.len(),
                        step + 1,
                        &item_name,
                        item_idx,
                        &item_info.location.area,
                    )?;
                    items_set.insert(item_info.item.clone());
                }
            }
        }
        Ok(())
    }

    fn set_start_location(&mut self) -> Result<()> {
        // let start_locations: Vec<StartLocation> =
        //     serde_json::from_str(&std::fs::read_to_string(&"data/start_locations.json").unwrap()).unwrap();
        // let loc = start_locations.last().unwrap();
        let loc = self.randomization.start_location.clone();
        let room_addr = self.game_data.room_ptr_by_id[&loc.room_id];
        let door_node_id = loc.door_load_node_id.unwrap_or(loc.node_id);
        let (_, entrance_ptr) =
            self.game_data.reverse_door_ptr_pair_map[&(loc.room_id, door_node_id)];
        // let room_width_screens = self.rom.read_u8(room_addr + 4)?;
        // let room_height_screens = self.rom.read_u8(room_addr + 5)?;
        // let room_width_pixels = room_width_screens * 256;
        // let room_height_pixels = room_height_screens * 256;
        let x_pixels = (loc.x * 16.0) as isize;
        let y_pixels = (loc.y * 16.0) as isize - 24;
        let mut screen_x = x_pixels & 0xFF00;
        let mut screen_y = y_pixels & 0xFF00;
        screen_x += (loc.camera_offset_x.unwrap_or(0.0) * 16.0) as isize;
        screen_y += (loc.camera_offset_y.unwrap_or(0.0) * 16.0) as isize;
        // screen_x = max(screen_x, 0);
        // screen_x = min(screen_x, room_width_pixels - 0x100);
        // screen_y = max(screen_y, 0);
        // screen_y = min(screen_y, room_height_pixels - 0x100);
        let samus_x = x_pixels - (screen_x + 0x80);
        let samus_y = y_pixels - screen_y;
        println!(
            "screen: {:x} {:x}, samus: {:x} {:x}",
            screen_x, screen_y, samus_x, samus_y
        );
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
    apply_orig_ips_patches(&mut orig_rom, randomization)?;

    let mut rom = orig_rom.clone();
    let mut patcher = Patcher {
        orig_rom: &mut orig_rom,
        rom: &mut rom,
        randomization,
        game_data,
        map: &randomization.map,
        other_door_ptr_pair_map: get_other_door_ptr_pair_map(&randomization.map),
        // door_room_map: get_door_room_map(&self.game_data.)
    };
    patcher.apply_ips_patches()?;
    patcher.place_items()?;
    patcher.set_start_location()?;
    patcher.fix_save_stations()?;
    patcher.write_map_tilemaps()?;
    patcher.write_map_areas()?;
    patcher.make_map_revealed()?;
    patcher.apply_map_tile_patches()?;
    patcher.write_door_data()?;
    patcher.remove_non_blue_doors()?;
    if !randomization.difficulty.vanilla_map {
        patcher.use_area_based_music()?;
    }
    patcher.setup_door_specific_fx()?;
    if !randomization.difficulty.ultra_low_qol {
        patcher.setup_reload_cre()?;
    }
    patcher.fix_twin_rooms()?;
    patcher.fix_crateria_scrolling_sky()?;
    patcher.apply_title_screen_patches()?;
    patcher.customize_escape_timer()?;
    patcher.apply_miscellaneous_patches()?;
    patcher.apply_mother_brain_fight_patches()?;
    patcher.apply_seed_hash()?;
    patcher.apply_credits()?;
    Ok(rom)
}
